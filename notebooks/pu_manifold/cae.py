"""
Pure torch functions and ``nn.Module``s for Phase 02.2's Chart Auto-Encoder
(arXiv:1912.10094). Tensors in, tensors and dicts out -- no file I/O, no cache handling;
the runners under ``notebooks/diagnostics/`` own paths. Constants live in
``02.2-PREREGISTRATION.md``.

Unlike its sibling modules, this one imports torch at module level: Phase 02.2's model
genuinely needs it. For the same reason ``curvature.py`` and ``mknn.py`` are excluded from
``pu_manifold/__init__.py``'s eager imports (so Phase 1-only callers do not need torch
installed to import the package), this module is deliberately NOT re-exported there either.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn


# --- activations ----------------------------------------------------------------------


def activation_module(name: str) -> nn.Module:
    """C2-smooth activation module for the pre-registered ``ACTIVATION``. Supports
    "silu", "tanh", "softplus", plus "relu" reachable only for the CAE-06 ReLU control
    fit -- ReLU's second derivative is identically zero almost everywhere, incompatible
    with Phase 3's Jacobian/Hessian curvature computation, so it must never be a model
    constructor's default."""
    key = name.lower()
    if key == "silu":
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "softplus":
        return nn.Softplus()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name!r}")


def mlp_stack(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: str = "silu",
    out_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """``nn.Sequential`` of ``nn.Linear`` layers separated by ``activation``, ending in
    ``Linear(*, out_dim)`` optionally followed by ``out_activation``."""
    layers: List[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        layers.append(activation_module(activation))
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


# --- reconstruction statistics (eq. 19 + CAE-03 per-dim distribution) -------------------


def reconstruction_stats(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """eq. 19 plus the CAE-03 per-output-dimension distribution. Returns a dict of native
    floats: ``mse_per_dim`` (mean squared error divided by the ambient dimension),
    ``mse_total``, ``mean_norm`` (mean unsquared L2 reconstruction norm), the
    per-dimension MSE summary (``dim_mse_mean``, ``dim_mse_median``, ``dim_mse_p95``,
    ``dim_mse_max``), plus the full per-dimension array under ``dim_mse``."""
    diff = (x.detach() - y.detach()).cpu().numpy().astype(np.float64)
    out_dim = diff.shape[1]
    sq = diff**2
    row_sq_sum = sq.sum(axis=1)  # (n,) -- ||x - y||^2 per row
    dim_mse = sq.mean(axis=0)  # (m,) -- per-dimension MSE across rows

    return {
        "mse_per_dim": float(row_sq_sum.mean() / out_dim),
        "mse_total": float(row_sq_sum.mean()),
        "mean_norm": float(np.sqrt(row_sq_sum).mean()),
        "dim_mse_mean": float(dim_mse.mean()),
        "dim_mse_median": float(np.median(dim_mse)),
        "dim_mse_p95": float(np.percentile(dim_mse, 95)),
        "dim_mse_max": float(dim_mse.max()),
        "dim_mse": [float(v) for v in dim_mse],
    }


# --- CAE-03 matched-capacity baseline controls --------------------------------------------


class PlainAutoEncoder(nn.Module):
    """The paper's own eq. 22 deterministic autoencoder reference architecture -- the
    CAE-03 plain single-chart control. Three hidden layers of ``HIDDEN_WIDTH`` each side
    of a single bottleneck, no chart predictor, no cross-entropy term: precisely the
    question CAE-03 asks is whether the atlas bought anything over one chart at matched
    capacity, and building this as the paper's own stated architecture (not an invented
    control) is what keeps that comparison defensible."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden: Sequence[int] = (250, 250, 250),
        activation: str = "silu",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = mlp_stack(in_dim, hidden, latent_dim, activation, out_activation=None)
        self.decoder = mlp_stack(latent_dim, hidden, in_dim, activation, out_activation=None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        y = self.decode(z)
        return {"z": z, "y": y}


def _train_decoder_protocol(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: Any,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Shared training-loop body for :func:`train_plain_ae` and :func:`train_mlp_decoder`:
    the identical protocol as :func:`train_cae`'s main loop -- same optimizer, learning
    rate, weight decay, batch size, three-way stopping rule, seeding discipline, and
    returned fit-artifact shape -- but with no pre-training stage, no chart predictor, no
    cross-entropy term, and no Lipschitz penalty, because a single-chart / fixed-coordinate
    model has no chart-encoder family for the penalty to range over."""
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = inputs.shape[0]
    batch_size = cfg["batch"]
    max_epochs = cfg["max_epochs"]

    effective_cfg: Dict[str, Any] = dict(cfg)
    effective_cfg.setdefault("seed", seed)
    effective_cfg.setdefault("early_stop_min_delta", 0.0)
    effective_cfg.setdefault("early_stop_patience", max_epochs + 1)
    effective_cfg.setdefault("wallclock_ceiling_s", float("inf"))

    early_stop_min_delta = effective_cfg["early_stop_min_delta"]
    early_stop_patience = effective_cfg["early_stop_patience"]
    wallclock_ceiling_s = effective_cfg["wallclock_ceiling_s"]

    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history: List[Dict[str, Any]] = []
    start = time.monotonic()
    wallclock_truncated = False
    early_stopped = False
    epochs_run = 0
    best_loss = float("inf")
    plateau_count = 0

    for epoch in range(max_epochs):
        # rng.permutation is numpy, unaffected by device -- only the resulting index
        # tensor is moved, so the minibatch shuffle sequence itself is untouched.
        perm = torch.from_numpy(rng.permutation(n)).to(inputs.device)
        epoch_recon = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = inputs[idx]
            yb = targets[idx]
            optimizer.zero_grad()
            y_hat = forward_fn(xb)
            recon = ((yb - y_hat) ** 2).sum(dim=-1).mean()
            recon.backward()
            optimizer.step()
            epoch_recon += recon.item()
            n_batches += 1

        epoch_mean = epoch_recon / n_batches
        history.append({"epoch": epoch, "stage": "main", "recon": epoch_mean, "xent": 0.0, "total": epoch_mean})
        epochs_run = epoch + 1

        if time.monotonic() - start > wallclock_ceiling_s:
            wallclock_truncated = True
            break

        if epoch_mean < best_loss * (1 - early_stop_min_delta):
            best_loss = epoch_mean
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= early_stop_patience:
                early_stopped = True
                break

    wallclock_s = time.monotonic() - start

    return {
        "history": history,
        "epochs_run": epochs_run,
        "wallclock_s": wallclock_s,
        "wallclock_truncated": wallclock_truncated,
        "early_stopped": early_stopped,
        "cfg": effective_cfg,
    }


def train_plain_ae(model: "PlainAutoEncoder", x_train: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Trains the CAE-03 plain single-chart control under the identical protocol as
    :func:`train_cae`'s main loop (see :func:`_train_decoder_protocol`), returning a fit
    artifact of the identical shape. The differences from ``train_cae`` -- no
    pre-training, no chart predictor, no cross-entropy term, no Lipschitz penalty -- are
    recorded explicitly in the returned ``cfg["protocol_difference"]`` rather than left
    implicit."""

    def forward_fn(xb: torch.Tensor) -> torch.Tensor:
        return model(xb)["y"]

    fit = _train_decoder_protocol(forward_fn, model.parameters(), x_train, x_train, cfg)
    fit["cfg"]["protocol_difference"] = (
        "no pre-training stage, no chart predictor, no cross-entropy term, no Lipschitz "
        "penalty -- a single-chart model has no chart-encoder family for the penalty to "
        "range over"
    )
    return fit


# --- verdict rule (Section 5) ------------------------------------------------------------

VERDICT_RULE = (
    "PASS requires all three gates to hold. Every comparison is strict less-than -- a "
    "value exactly at a threshold does not clear it. There is no MARGINAL tier: every "
    "non-PASS outcome routes to the same halt-for-user-decision consequence, so a middle "
    "tier would carry no distinct consequence."
)
