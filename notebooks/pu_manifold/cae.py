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

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from . import cache

# --- fit-artifact contract (plans 02.2-04 and 02.2-05 build against this, not a guess) --

FIT_ARTIFACT_CONTRACT = """
npz stem `cae_fit_{fit_key}_seed{seed}` carries:
  - every state_dict_to_arrays key (the trained model's full parameter set)
  - z_all            -- the initial encoder's output for all 10,000 rows
  - p_all            -- chart probabilities for all rows
  - chart_argmax_all -- per-row argmax chart index
  - train_idx        -- training-split row indices
  - holdout_idx      -- holdout-split row indices
  - y_holdout        -- argmax-chart ambient reconstruction of the holdout rows

Per-chart weight-mass quantities are deliberately NOT stored: they are derivable from the
persisted state_dict, and storing a derived value invites it drifting from the weights it
summarises.

Companion json stem `cae_fit_meta_{fit_key}_seed{seed}` carries the train_cae return dict
plus seed, activation, torch_version, numpy_version, timestamp.
"""

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


# --- architecture modules ---------------------------------------------------------------


class InitialEncoder(nn.Module):
    """E: ambient R^m -> embedding space R^l, linear output."""

    def __init__(self, in_dim: int, embed_dim: int, hidden: Sequence[int], activation: str = "silu"):
        super().__init__()
        self.net = mlp_stack(in_dim, hidden, embed_dim, activation, out_activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChartEncoder(nn.Module):
    """E_alpha: embedding space R^l -> chart space, sigmoid output so coordinates land
    strictly inside the open unit cube (0,1)^d."""

    def __init__(self, embed_dim: int, chart_dim: int, hidden: Sequence[int], activation: str = "silu"):
        super().__init__()
        self.net = mlp_stack(embed_dim, hidden, chart_dim, activation, out_activation=nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ChartDecoder(nn.Module):
    """D_alpha: chart space -> embedding space R^l, linear output. Output dimension is
    ``embed_dim``, never ``in_dim`` -- the two-hop decode (RESEARCH.md Pitfall 1)."""

    def __init__(self, chart_dim: int, embed_dim: int, hidden: Sequence[int], activation: str = "silu"):
        super().__init__()
        self.net = mlp_stack(chart_dim, hidden, embed_dim, activation, out_activation=None)

    def forward(self, z_alpha: torch.Tensor) -> torch.Tensor:
        return self.net(z_alpha)


class EmbeddingDecoder(nn.Module):
    """D: embedding space R^l -> ambient R^m, linear output. Exactly one instance is
    shared across every chart's forward pass."""

    def __init__(self, embed_dim: int, out_dim: int, hidden: Sequence[int], activation: str = "silu"):
        super().__init__()
        self.net = mlp_stack(embed_dim, hidden, out_dim, activation, out_activation=None)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.net(w)


class ChartPredictor(nn.Module):
    """P: embedding space R^l -> per-chart probabilities via a softmax output
    (partition of unity)."""

    def __init__(self, embed_dim: int, n_charts: int, hidden: Sequence[int], activation: str = "silu"):
        super().__init__()
        self.net = mlp_stack(embed_dim, hidden, n_charts, activation, out_activation=nn.Softmax(dim=-1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ChartAutoEncoder(nn.Module):
    """One ``InitialEncoder``, an ``nn.ModuleList`` of ``ChartEncoder``, an
    ``nn.ModuleList`` of ``ChartDecoder``, exactly ONE shared ``EmbeddingDecoder``
    instance, and one ``ChartPredictor``. The shared decoder is a single attribute
    called by every chart's forward pass -- instantiating one per chart is the failure
    mode RESEARCH.md Pitfall 1 describes."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        chart_dim: int,
        n_charts: int,
        hidden: Sequence[int],
        activation: str = "silu",
    ):
        super().__init__()
        self.n_charts = n_charts
        self.chart_dim = chart_dim
        self.embed_dim = embed_dim
        self.out_dim = in_dim
        self.activation = activation

        self.initial_encoder = InitialEncoder(in_dim, embed_dim, hidden, activation)
        self.chart_encoders = nn.ModuleList(
            [ChartEncoder(embed_dim, chart_dim, hidden, activation) for _ in range(n_charts)]
        )
        self.chart_decoders = nn.ModuleList(
            [ChartDecoder(chart_dim, embed_dim, hidden, activation) for _ in range(n_charts)]
        )
        self.embedding_decoder = EmbeddingDecoder(embed_dim, in_dim, hidden, activation)
        self.chart_predictor = ChartPredictor(embed_dim, n_charts, hidden, activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.initial_encoder(x)

    def chart_coords(self, z: torch.Tensor) -> torch.Tensor:
        """(batch, l) -> (batch, n_charts, chart_dim)."""
        return torch.stack([enc(z) for enc in self.chart_encoders], dim=1)

    def _decode_from_chart_coords(self, z_charts: torch.Tensor) -> torch.Tensor:
        """(batch, n_charts, chart_dim) -> (batch, n_charts, out_dim), each chart routed
        through its own decoder then the single shared embedding decoder."""
        y_list = []
        for i in range(self.n_charts):
            w = self.chart_decoders[i](z_charts[:, i, :])
            y_list.append(self.embedding_decoder(w))
        return torch.stack(y_list, dim=1)

    def decode_all(self, z: torch.Tensor) -> torch.Tensor:
        """(batch, l) -> (batch, n_charts, out_dim) of per-chart ambient reconstructions."""
        return self._decode_from_chart_coords(self.chart_coords(z))

    def chart_probs(self, z: torch.Tensor) -> torch.Tensor:
        return self.chart_predictor(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        z_charts = self.chart_coords(z)
        y_charts = self._decode_from_chart_coords(z_charts)
        p = self.chart_probs(z)
        e = ((x.unsqueeze(1) - y_charts) ** 2).sum(dim=-1)
        return {"z": z, "z_charts": z_charts, "y_charts": y_charts, "p": p, "e": e}

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Argmax-chart reconstruction."""
        out = self.forward(x)
        idx = out["p"].argmax(dim=1)
        y_charts = out["y_charts"]
        return y_charts[torch.arange(y_charts.shape[0]), idx]


# --- loss (eq. 3) -----------------------------------------------------------------------


def chart_loss(x: torch.Tensor, y_charts: torch.Tensor, p: torch.Tensor) -> Dict[str, torch.Tensor]:
    """eq. 3: ``L(x,W) = min_alpha e_alpha - sum_beta ell_beta log(p_beta)``, with
    ``ell = softmax(-e)`` detached and treated as a fixed target distribution against the
    predictor's already-normalized probabilities. Returns a dict with ``recon``, ``xent``,
    ``total`` (the batch mean of the per-row sum)."""
    e = ((x.unsqueeze(1) - y_charts) ** 2).sum(dim=-1)  # (batch, n_charts)
    recon = e.min(dim=1).values  # (batch,) -- min_alpha e_alpha
    ell = torch.softmax(-e, dim=1).detach()
    xent = -(ell * torch.log(p.clamp_min(1e-12))).sum(dim=1)  # (batch,)
    total = (recon + xent).mean()
    return {"recon": recon.mean(), "xent": xent.mean(), "total": total}


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


# --- training loop -----------------------------------------------------------------------


def train_cae(model: nn.Module, x_train: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A real training loop, minimal but not a stub: AdamW with ``cfg["lr"]`` and
    ``cfg["weight_decay"]``, shuffled minibatches of ``cfg["batch"]``,
    ``cfg["max_epochs"]`` epochs, accumulating a per-epoch mean-loss history.

    Returns the fit-artifact dict every later plan builds against: ``history`` (list of
    per-epoch dicts with ``epoch``, ``recon``, ``xent``, ``total``), ``epochs_run``,
    ``wallclock_s``, ``wallclock_truncated``, ``early_stopped``, ``cfg``. Early stopping,
    the wall-clock ceiling, the Lipschitz term, and the FPS pre-training stage are
    expansion points filled by plan 02.2-03 -- the keys are present and honestly valued
    (``wallclock_truncated: False``, ``early_stopped: False``) here rather than absent, so
    the contract does not change shape later."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = x_train.shape[0]
    batch_size = cfg["batch"]
    max_epochs = cfg["max_epochs"]

    history: List[Dict[str, Any]] = []
    start = time.monotonic()
    for epoch in range(max_epochs):
        perm = torch.randperm(n)
        epoch_recon = 0.0
        epoch_xent = 0.0
        epoch_total = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss_dict = chart_loss(xb, out["y_charts"], out["p"])
            loss_dict["total"].backward()
            optimizer.step()
            epoch_recon += loss_dict["recon"].item()
            epoch_xent += loss_dict["xent"].item()
            epoch_total += loss_dict["total"].item()
            n_batches += 1
        history.append(
            {
                "epoch": epoch,
                "recon": epoch_recon / n_batches,
                "xent": epoch_xent / n_batches,
                "total": epoch_total / n_batches,
            }
        )
    wallclock_s = time.monotonic() - start

    return {
        "history": history,
        "epochs_run": max_epochs,
        "wallclock_s": wallclock_s,
        "wallclock_truncated": False,
        "early_stopped": False,
        "cfg": cfg,
    }


# --- native casting ------------------------------------------------------------------------


def to_native(obj: Any) -> Any:
    """Recursively cast numpy/torch scalars to Python float/int/bool -- ``json_cache``'s
    ``json.dumps(..., sort_keys=True)`` cannot serialize numpy scalars. Replicates
    ``notebooks/diagnostics/geometry_probes_run.py``'s ``_to_native``, extended to also
    unwrap torch tensors so no torch scalar reaches the JSON encoder either."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return to_native(obj.item())
        return to_native(obj.detach().cpu().tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# --- verdict rule (Section 5) ------------------------------------------------------------

GATING_METRICS: Tuple[str, ...] = ("distortion", "rcycle_ratio", "recon_margin")

VERDICT_RULE = (
    "PASS requires all three gates to hold. Every comparison is strict less-than -- a "
    "value exactly at a threshold does not clear it. There is no MARGINAL tier: every "
    "non-PASS outcome routes to the same halt-for-user-decision consequence, so a middle "
    "tier would carry no distinct consequence."
)


def verdict_from_metrics(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Section 5's rule: reads the three gating metrics named in ``GATING_METRICS``,
    applies strict less-than against each threshold, and returns ``(verdict,
    gate_detail)`` where ``verdict`` is ``"PASS"`` or ``"FAIL"`` and ``gate_detail`` maps
    each gate name to its measured value, its threshold, and its boolean outcome.

    Before evaluating any gate, asserts that each of the three gating metric names is
    present in ``metrics`` and that its value is finite, raising ``ValueError`` naming the
    offending metric otherwise -- a missing or non-finite measurement must never be able
    to become a PASS. This is the terminal artifact of a hard gate, so halting is the
    honest behaviour; emitting a FAIL would be indistinguishable from a measured FAIL."""
    for gate in GATING_METRICS:
        if gate not in metrics:
            raise ValueError(f"verdict_from_metrics: gating metric {gate!r} is absent from metrics")
        value = float(metrics[gate])
        if not math.isfinite(value):
            raise ValueError(
                f"verdict_from_metrics: gating metric {gate!r} is non-finite ({value!r}) "
                "-- a missing or non-finite measurement can never become a PASS"
            )

    gate_detail: Dict[str, Dict[str, Any]] = {}
    all_pass = True
    for gate in GATING_METRICS:
        value = float(metrics[gate])
        threshold = thresholds[gate]
        passed = bool(value < threshold)
        gate_detail[gate] = {"value": value, "threshold": threshold, "passed": passed}
        if not passed:
            all_pass = False
    verdict = "PASS" if all_pass else "FAIL"
    return verdict, gate_detail


def write_cae_verdict(
    fit_key: str,
    metrics: Dict[str, Any],
    thresholds: Dict[str, Any],
    verdict: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Delegates persistence entirely to ``cache.json_cache`` under the stem
    ``f"cae_verdict_{fit_key}"`` with cfg ``{"fit_key": ..., "phase": "02.2"}``. The
    schema precedent ``gate_verdict_43cf438bc944c509.json`` used a plain ``verdict`` key,
    but CAE-07 names this field ``CAE_VERDICT``, so that name is used here. Every value
    passes through :func:`to_native` before it reaches ``json_cache``, so the encoder
    never sees a numpy or torch scalar."""
    cfg = {"fit_key": fit_key, "phase": "02.2"}

    def _compute() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "fit_key": fit_key,
            "phase": "02.2",
            "CAE_VERDICT": verdict,
            "metrics": metrics,
            "thresholds": thresholds,
            "verdict_rule": VERDICT_RULE,
        }
        if extra:
            payload.update(extra)
        return to_native(payload)

    return cache.json_cache(f"cae_verdict_{fit_key}", cfg, _compute)


# --- fit-artifact serialization (no pickle path) ------------------------------------------


def state_dict_to_arrays(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Flatten a torch ``state_dict`` to a flat ``Dict[str, np.ndarray]`` of float64
    arrays suitable for ``cache.npz_cache``. Persisting fits this way keeps the phase free
    of pickle: the package's separate pickle-backed cache helper exists, but pickle
    deserialization is a path the codebase's own threat model deliberately keeps narrow,
    so every fit artifact takes this array route instead."""
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in state_dict.items()}


def arrays_to_state_dict(
    arrays: Dict[str, np.ndarray], reference_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Invert :func:`state_dict_to_arrays`, casting each array to the dtype and shape of
    the corresponding tensor in ``reference_state_dict``."""
    out: Dict[str, torch.Tensor] = {}
    for key, ref in reference_state_dict.items():
        arr = arrays[key]
        out[key] = torch.tensor(arr, dtype=ref.dtype).reshape(ref.shape)
    return out


# --- D-03 PASS handoff writer --------------------------------------------------------------


def write_cae_handoff(fit_key: str, verdict: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """D-03: on a ``"PASS"`` verdict, writes a machine-readable artifact naming exactly
    what Phase 3 consumes -- the global embedding, chart coordinates, chart assignments
    and probabilities, and the trained chart and embedding decoders. Refuses to write
    (raises ``ValueError``) unless ``verdict == "PASS"``."""
    if verdict != "PASS":
        raise ValueError(
            f"write_cae_handoff refuses to write a handoff for a non-PASS verdict: {verdict!r}"
        )

    cfg = {"fit_key": fit_key, "phase": "02.2"}

    def _compute() -> Dict[str, Any]:
        result = {
            "fit_key": fit_key,
            "phase": "02.2",
            "verdict": verdict,
            "consumes": payload["consumes"],
            "global_embedding_key": payload["global_embedding_key"],
            "chart_coords_key": payload["chart_coords_key"],
            "chart_probs_key": payload["chart_probs_key"],
            "chart_assignments_key": payload["chart_assignments_key"],
            "decoder_state_keys": payload["decoder_state_keys"],
            "surviving_charts": payload["surviving_charts"],
            "activation": payload["activation"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return to_native(result)

    return cache.json_cache(f"cae_handoff_{fit_key}", cfg, _compute)
