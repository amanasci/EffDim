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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from . import cache
from . import geometry_probes

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


# --- Lipschitz regularizer (eq. 4) -------------------------------------------------------


def _robust_spectral_norm(weight: torch.Tensor) -> torch.Tensor:
    """``torch.linalg.matrix_norm(weight, ord=2)`` -- the exact largest singular value --
    with a numerical-precision fallback: LAPACK's SVD occasionally raises
    ``torch.linalg.LinAlgError`` ("failed to converge") on a real trained weight matrix
    that is ill-conditioned or has near-repeated singular values, a failure mode the small
    random matrices in this module's unit tests never happened to hit. Retrying the
    identical decomposition in float64 resolves this in practice (double precision uses a
    different, more numerically stable LAPACK code path) without changing the quantity
    being computed -- still the exact spectral norm, not an approximation -- and the
    float64 result is cast back to the input's dtype so gradients keep flowing through the
    original dtype's computation graph."""
    try:
        return torch.linalg.matrix_norm(weight, ord=2)
    except torch.linalg.LinAlgError:
        return torch.linalg.matrix_norm(weight.double(), ord=2).to(weight.dtype)


def _chart_encoder_spectral_product(encoder: nn.Module) -> torch.Tensor:
    """Product, over an encoder's ``nn.Linear`` layers in forward order, of that layer's
    exact spectral norm (largest singular value) via :func:`_robust_spectral_norm`. Exact
    SVD, not the power-iteration reparametrization route -- see
    :func:`lipschitz_penalty`'s docstring."""
    prod = torch.ones((), dtype=torch.float64)
    for layer in encoder.modules():
        if isinstance(layer, nn.Linear):
            prod = prod * _robust_spectral_norm(layer.weight)
    return prod


def lipschitz_penalty(chart_encoders: Sequence[nn.Module], weight: float) -> torch.Tensor:
    """eq. 4: ``R_Lip = max_alpha(prod_k ||W_alpha^k||_2) + mean_alpha(prod_k
    ||W_alpha^k||_2)``, computed by **exact SVD** (``torch.linalg.matrix_norm(W,
    ord=2)``) per chart-encoder Linear layer -- cheap at these layer widths (at most
    250x250 per chart, 768x250 for the initial encoder) and removes the power-iteration
    convergence-tolerance question entirely. The first term stops one chart's encoder
    from dominating; the second promotes overall smoothness across every chart.

    This is a **soft additive penalty term on the loss**, not a hard constraint: it must
    never be replaced by the power-iteration spectral-norm reparametrization helper
    bundled with ``torch.nn.utils``, which reparametrizes the weight matrix itself to
    force its spectral norm to approximately one at all times. A hard constraint resists
    exactly the weight-decay pressure CAE-05's a-posteriori chart pruning depends on -- the
    warning sign, if this is ever confused, is a chart-encoder weight set that never
    shrinks under weight decay and a penalty value pinned to a constant across training
    rather than moving. Returns ``weight`` times the quantity above as a differentiable
    float64 scalar tensor the optimizer can trade off against reconstruction."""
    products = torch.stack([_chart_encoder_spectral_product(enc) for enc in chart_encoders])
    return weight * (products.max() + products.mean())


# --- farthest-point sampling and eq. 5 FPS-seeded per-chart pre-training -----------------


def farthest_point_sample(x: torch.Tensor, n_seeds: int, seed: int) -> torch.Tensor:
    """Greedy farthest-point sampling over ``x`` ``(n, features)``. Returns ``n_seeds``
    row indices as a ``torch.long`` tensor, order ``n_seeds * n`` via a running per-row
    minimum-distance array -- no O(n^2) all-pairs matrix is ever materialised. The first
    pick is drawn from a ``torch.Generator`` seeded with ``seed``, so two calls under the
    same seed return identical index arrays. No new package is installed for this: at
    ``N_CHARTS_INIT=16`` seeds over ten thousand rows the hand-rolled greedy routine is
    trivially fast."""
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    chosen = torch.empty(n_seeds, dtype=torch.long)
    chosen[0] = torch.randint(0, n, (1,), generator=g)
    min_dist = torch.full((n,), float("inf"))
    for i in range(n_seeds):
        d = torch.cdist(x, x[chosen[i]].unsqueeze(0)).squeeze(1)
        min_dist = torch.minimum(min_dist, d)
        if i + 1 < n_seeds:
            chosen[i + 1] = torch.argmax(min_dist)
    return chosen


def fps_pretrain_loss(
    model: "ChartAutoEncoder", x_seeds: torch.Tensor, seed_chart_index: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """eq. 5: for each FPS-selected seed point paired with the chart index it seeds, the
    sum of (a) the squared ambient reconstruction error of that point through its own
    chart, (b) the squared distance between that point's chart coordinate and the centre
    ``[.5]^d`` of the chart space, and (c) the categorical chart-prediction term pushing
    the predictor toward that chart for that point. Prevents the dead-chart failure mode
    the paper warns about, where a chart decoder is never activated, receives no gradient
    signal, and stays at its initialisation forever. Returns a dict with the three named
    components (``recon``, ``center``, ``xent``) plus their ``total``."""
    n_seeds = x_seeds.shape[0]
    idx = torch.arange(n_seeds)
    z = model.encode(x_seeds)
    z_charts_all = model.chart_coords(z)  # (n_seeds, n_charts, chart_dim)
    z_alpha = z_charts_all[idx, seed_chart_index]  # (n_seeds, chart_dim) -- own chart's coord
    y_charts_all = model._decode_from_chart_coords(z_charts_all)  # (n_seeds, n_charts, out_dim)
    y_alpha = y_charts_all[idx, seed_chart_index]  # (n_seeds, out_dim) -- own chart's decode

    recon = ((x_seeds - y_alpha) ** 2).sum(dim=-1)
    center = torch.full_like(z_alpha, 0.5)
    center_term = ((z_alpha - center) ** 2).sum(dim=-1)
    p = model.chart_probs(z)
    p_target = p.gather(1, seed_chart_index.unsqueeze(1)).squeeze(1)
    xent = -torch.log(p_target.clamp_min(1e-12))

    total = (recon + center_term + xent).mean()
    return {"recon": recon.mean(), "center": center_term.mean(), "xent": xent.mean(), "total": total}


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
    """The full pre-registered training protocol: FPS-seeded per-chart pre-training (eq.
    5, stage one) followed by the main AdamW loop (stage two) minimizing eq. 3's
    ``chart_loss`` plus the eq. 4 Lipschitz penalty, stopped by whichever of three
    pre-registered limits fires first -- the epoch cap ``cfg["max_epochs"]``, a
    relative-plateau early-stopping rule, or a wall-clock ceiling checked at every epoch
    boundary, measured from the start of stage one.

    Every random source -- the minibatch shuffle and the FPS seed draw -- derives from
    the single ``cfg["seed"]`` (default 0), so a fit is reproducible from one recorded
    integer (CAE-02). Every protocol value this function reads is read from ``cfg``, not
    hardcoded, and is echoed back into the returned ``cfg`` (T-02.2-08): a caller that
    omits an optional key sees the default this function actually used, not silence about
    it.

    Returns the plan 02.2-02 fit-artifact contract, unchanged in shape: ``history`` (list
    of per-epoch/per-pretrain-step dicts, each carrying a ``stage`` field of
    ``"pretrain"`` or ``"main"``), ``epochs_run``, ``wallclock_s``, ``wallclock_truncated``,
    ``early_stopped``, ``cfg``."""
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = x_train.shape[0]
    batch_size = cfg["batch"]
    max_epochs = cfg["max_epochs"]

    effective_cfg: Dict[str, Any] = dict(cfg)
    effective_cfg.setdefault("seed", seed)
    effective_cfg.setdefault("n_charts", model.n_charts)
    effective_cfg.setdefault("fps_pretrain_epochs", 0)
    effective_cfg.setdefault("lip_weight", 0.0)
    effective_cfg.setdefault("lip_every_n_steps", 1)
    effective_cfg.setdefault("early_stop_min_delta", 0.0)
    effective_cfg.setdefault("early_stop_patience", max_epochs + 1)
    effective_cfg.setdefault("wallclock_ceiling_s", float("inf"))

    n_charts_cfg = effective_cfg["n_charts"]
    fps_pretrain_epochs = effective_cfg["fps_pretrain_epochs"]
    lip_weight = effective_cfg["lip_weight"]
    lip_every_n_steps = effective_cfg["lip_every_n_steps"]
    early_stop_min_delta = effective_cfg["early_stop_min_delta"]
    early_stop_patience = effective_cfg["early_stop_patience"]
    wallclock_ceiling_s = effective_cfg["wallclock_ceiling_s"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history: List[Dict[str, Any]] = []
    start = time.monotonic()
    wallclock_truncated = False
    early_stopped = False
    epochs_run = 0
    step_counter = 0

    # --- stage one: FPS-seeded per-chart pre-training (eq. 5) ---
    if fps_pretrain_epochs > 0:
        with torch.no_grad():
            z_train = model.encode(x_train)
        seed_idx = farthest_point_sample(z_train, n_charts_cfg, seed=seed)
        x_seeds = x_train[seed_idx]
        seed_chart_index = torch.arange(n_charts_cfg)
        for epoch in range(fps_pretrain_epochs):
            optimizer.zero_grad()
            loss_dict = fps_pretrain_loss(model, x_seeds, seed_chart_index)
            loss_dict["total"].backward()
            optimizer.step()
            history.append(
                {
                    "epoch": epoch,
                    "stage": "pretrain",
                    "recon": loss_dict["recon"].item(),
                    "center": loss_dict["center"].item(),
                    "xent": loss_dict["xent"].item(),
                    "total": loss_dict["total"].item(),
                }
            )
            if time.monotonic() - start > wallclock_ceiling_s:
                wallclock_truncated = True
                break

    # --- stage two: the main loop, three-way stopping rule ---
    best_loss = float("inf")
    plateau_count = 0
    if not wallclock_truncated:
        for epoch in range(max_epochs):
            perm = torch.from_numpy(rng.permutation(n))
            epoch_recon = 0.0
            epoch_xent = 0.0
            epoch_lip = 0.0
            epoch_total = 0.0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb = x_train[idx]
                optimizer.zero_grad()
                out = model(xb)
                loss_dict = chart_loss(xb, out["y_charts"], out["p"])
                total = loss_dict["total"]
                lip_value = 0.0
                if lip_weight and step_counter % lip_every_n_steps == 0:
                    lip_term = lipschitz_penalty(model.chart_encoders, lip_weight)
                    total = total + lip_term
                    lip_value = lip_term.item()
                total.backward()
                optimizer.step()
                epoch_recon += loss_dict["recon"].item()
                epoch_xent += loss_dict["xent"].item()
                epoch_lip += lip_value
                epoch_total += total.item()
                n_batches += 1
                step_counter += 1

            epoch_mean_total = epoch_total / n_batches
            history.append(
                {
                    "epoch": epoch,
                    "stage": "main",
                    "recon": epoch_recon / n_batches,
                    "xent": epoch_xent / n_batches,
                    "lip": epoch_lip / n_batches,
                    "total": epoch_mean_total,
                }
            )
            epochs_run = epoch + 1

            if time.monotonic() - start > wallclock_ceiling_s:
                wallclock_truncated = True
                break

            if epoch_mean_total < best_loss * (1 - early_stop_min_delta):
                best_loss = epoch_mean_total
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


def timing_probe(
    model_factory: Callable[[], nn.Module], x_train: torch.Tensor, cfg: Dict[str, Any], n_steps: int
) -> Dict[str, Any]:
    """D-07's pre-registered compute contingency: run ``n_steps`` real training steps
    (a fresh model from ``model_factory``, the same optimizer/loss/Lipschitz-penalty
    machinery as :func:`train_cae`'s main loop) and project the wall-clock time
    ``cfg["max_epochs"]`` would take at this training-set size. The runner in plan
    02.2-05 calls this before any real fit and applies the pre-registered contingency: if
    the projection exceeds ``cfg["wallclock_ceiling_s"]``, ``lip_every_n_steps`` moves
    from one to four for every fit in the protocol, including the controls -- applied
    un-rescaled, not tuned per-fit."""
    model = model_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = x_train.shape[0]
    batch_size = cfg["batch"]
    lip_weight = cfg.get("lip_weight", 0.0)

    steps_done = 0
    start = time.monotonic()
    while steps_done < n_steps:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if steps_done >= n_steps:
                break
            idx = perm[i : i + batch_size]
            xb = x_train[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss_dict = chart_loss(xb, out["y_charts"], out["p"])
            total = loss_dict["total"]
            if lip_weight and hasattr(model, "chart_encoders"):
                total = total + lipschitz_penalty(model.chart_encoders, lip_weight)
            total.backward()
            optimizer.step()
            steps_done += 1
    elapsed = time.monotonic() - start

    seconds_per_step = elapsed / n_steps
    n_batches_per_epoch = math.ceil(n / batch_size)
    projected_wallclock_s = seconds_per_step * n_batches_per_epoch * cfg["max_epochs"]
    exceeds_ceiling = projected_wallclock_s > cfg["wallclock_ceiling_s"]

    return {
        "seconds_per_step": seconds_per_step,
        "projected_wallclock_s": projected_wallclock_s,
        "exceeds_ceiling": bool(exceeds_ceiling),
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


class MlpDecoder(nn.Module):
    """A decoder from a fixed coordinate array back to ambient space, matched in hidden
    width/depth to ``PlainAutoEncoder``'s decoder half -- the CAE-03 classical-MDS
    baseline decoder. Classical MDS has no native inverse map to ambient space, so one
    must be fitted explicitly rather than substituted by the ``sklearn.manifold.Isomap``
    estimator's own residual-variance reconstruction-quality method, which is a different
    quantity (RESEARCH.md Pitfall 4)."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (250, 250, 250),
        activation: str = "silu",
    ):
        super().__init__()
        self.net = mlp_stack(latent_dim, hidden, out_dim, activation, out_activation=None)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"y": self.net(z)}


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
        perm = torch.from_numpy(rng.permutation(n))
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


def train_mlp_decoder(
    model: "MlpDecoder", z_train: torch.Tensor, x_train: torch.Tensor, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Fits the CAE-03 classical-MDS baseline decoder from fixed coordinates ``z_train``
    to ambient targets ``x_train``, under the identical protocol as :func:`train_cae`'s
    main loop (see :func:`_train_decoder_protocol`), returning a fit artifact of the
    identical shape."""

    def forward_fn(zb: torch.Tensor) -> torch.Tensor:
        return model(zb)["y"]

    fit = _train_decoder_protocol(forward_fn, model.parameters(), z_train, x_train, cfg)
    fit["cfg"]["protocol_difference"] = (
        "fits a decoder only, from fixed input coordinates z_train, not a jointly-trained "
        "encoder-decoder pair"
    )
    return fit


def fit_linear_decoder(z_train: torch.Tensor, x_train: torch.Tensor) -> Dict[str, np.ndarray]:
    """Least-squares linear map ``x_hat = z @ weight.T + bias`` from coordinates back to
    ambient space, reported as a floor beneath the fitted ``MlpDecoder`` so a reader can
    see how much of the baseline's performance is linear."""
    z_np = z_train.detach().cpu().numpy().astype(np.float64)
    x_np = x_train.detach().cpu().numpy().astype(np.float64)
    n = z_np.shape[0]
    z_aug = np.concatenate([z_np, np.ones((n, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(z_aug, x_np, rcond=None)
    weight = coef[:-1].T
    bias = coef[-1]
    return {"weight": weight, "bias": bias}


# --- CAE-05 a-posteriori chart pruning ------------------------------------------------------


def _chart_decoder_logmass(decoder: nn.Module) -> float:
    """Sum, in float64, of the natural logarithms of a chart decoder's per-``nn.Linear``
    layer exact spectral norm (the same per-layer quantity :func:`lipschitz_penalty`
    uses). Working in log space rather than a float32 product is the contract, not an
    optimisation: a chart decoder with several small-spectral-norm layers underflows a
    float32 product to exactly zero, which would silently report a live chart as dead and
    change the surviving count CAE-05 reports as an outcome."""
    total = 0.0
    for layer in decoder.modules():
        if isinstance(layer, nn.Linear):
            sv = float(_robust_spectral_norm(layer.weight).item())
            total += -math.inf if sv <= 0.0 else math.log(sv)
    return total


def chart_survival(model: "ChartAutoEncoder", prune_tol: float) -> Dict[str, Any]:
    """CAE-05: the chart count is obtained a posteriori, never chosen as an input. For
    each chart decoder, computes :func:`_chart_decoder_logmass`, then compares every
    chart's mass against the largest chart's mass: a chart is pruned when
    ``exp(log_mass_alpha - log_mass_max)`` is **strictly less than** ``prune_tol`` -- a
    chart whose ratio equals ``prune_tol`` exactly survives, mirroring the strict-less-
    than discipline :func:`verdict_from_metrics` uses throughout, so a value sitting
    exactly on a stated tolerance never clears it by accident in one direction and gets
    caught by it in the other.

    ``model`` only needs to expose ``n_charts`` and an iterable ``chart_decoders`` of
    ``nn.Module`` decoders -- it need not be a full ``ChartAutoEncoder`` instance.
    Returns a dict with ``n_charts_initial``, ``n_charts_surviving``,
    ``surviving_indices``, ``pruned_indices`` (native Python ints/lists), plus
    ``chart_logmass`` and ``chart_mass_ratio`` as float64 numpy arrays, and
    ``prune_tol`` (native float)."""
    n_charts = model.n_charts
    logmass = np.array(
        [_chart_decoder_logmass(dec) for dec in model.chart_decoders], dtype=np.float64
    )
    log_mass_max = float(logmass.max())
    mass_ratio = np.exp(logmass - log_mass_max)

    surviving_indices = [int(i) for i in range(n_charts) if mass_ratio[i] >= prune_tol]
    pruned_indices = [int(i) for i in range(n_charts) if mass_ratio[i] < prune_tol]

    return {
        "n_charts_initial": int(n_charts),
        "n_charts_surviving": int(len(surviving_indices)),
        "surviving_indices": surviving_indices,
        "pruned_indices": pruned_indices,
        "chart_logmass": logmass,
        "chart_mass_ratio": mass_ratio,
        "prune_tol": float(prune_tol),
    }


# --- CAE-04 chart-transition cycle residual (eq. 8) and the overlap-pair selection rule -----


def _decode_through_chart(model: "ChartAutoEncoder", z: torch.Tensor, chart_idx: int) -> torch.Tensor:
    """(batch, l) -> (batch, m): the two-hop decode through one named chart -- that
    chart's coordinate, its own chart decoder, then the single shared embedding decoder."""
    z_charts = model.chart_coords(z)
    w = model.chart_decoders[chart_idx](z_charts[:, chart_idx, :])
    return model.embedding_decoder(w)


def r_cycle(model: "ChartAutoEncoder", x: torch.Tensor, alpha: int, beta: int) -> torch.Tensor:
    """eq. 8 verbatim, both directions: encode ``x``, decode through chart ``alpha``,
    re-encode that ambient reconstruction and decode through chart ``beta``, compare to
    ``x`` with an unsquared L2 norm; plus the symmetric ``beta``-then-``alpha`` direction.
    Returns the sum of the two per-row norms as a ``(batch,)`` tensor. Both directions are
    required -- the residual is a statement about the atlas's consistency where two charts
    overlap, and a one-directional version would measure something weaker.

    ``model`` only needs ``encode``, ``chart_coords``, ``chart_decoders``, and
    ``embedding_decoder`` -- it need not be a full ``ChartAutoEncoder`` instance."""
    z = model.encode(x)

    y1_ab = _decode_through_chart(model, z, alpha)
    z2_ab = model.encode(y1_ab)
    y2_ab = _decode_through_chart(model, z2_ab, beta)
    dir_ab = torch.linalg.vector_norm(x - y2_ab, dim=-1)

    y1_ba = _decode_through_chart(model, z, beta)
    z2_ba = model.encode(y1_ba)
    y2_ba = _decode_through_chart(model, z2_ba, alpha)
    dir_ba = torch.linalg.vector_norm(x - y2_ba, dim=-1)

    return dir_ab + dir_ba


def select_overlap_pairs(
    p: np.ndarray,
    p_min: float,
    min_points: int,
    surviving_indices: Sequence[int],
) -> Dict[str, np.ndarray]:
    """The pre-registered T2 overlap-pair selection rule. For each row, restricts to the
    surviving (post-pruning) charts, takes the two highest probabilities, and admits the
    row only when **both** reach ``p_min``. Raises ``ValueError`` naming the shortfall
    when fewer than ``min_points`` rows qualify: the paper frames R_cycle as applying only
    where a point is in the intersection of two chart neighbourhoods but supplies no
    numeric cutoff, so without a pre-registered one the evaluation set is ill-defined, and
    an under-populated set would let a near-zero residual be reported from a handful of
    points. Halting is the honest behaviour, not a fallback to a looser cutoff.

    Returns a dict with ``rows`` (admitted row indices), ``alpha``, ``beta`` (each
    admitted row's two highest-probability surviving chart indices), as equal-length
    int64 numpy arrays."""
    p = np.asarray(p, dtype=np.float64)
    surviving = np.asarray(sorted(surviving_indices), dtype=np.int64)
    p_surv = p[:, surviving]

    order = np.argsort(-p_surv, axis=1)
    top2_local = order[:, :2]
    p_top1 = np.take_along_axis(p_surv, top2_local[:, 0:1], axis=1)[:, 0]
    p_top2 = np.take_along_axis(p_surv, top2_local[:, 1:2], axis=1)[:, 0]

    qualifies = (p_top1 >= p_min) & (p_top2 >= p_min)
    n_qualify = int(qualifies.sum())
    if n_qualify < min_points:
        raise ValueError(
            f"select_overlap_pairs: only {n_qualify} rows qualify for the T2 overlap "
            f"evaluation set (both top chart probabilities >= {p_min}), below the "
            f"pre-registered minimum of {min_points} -- T2 is undefined and the run "
            "must halt rather than report a value."
        )

    admitted_rows = np.flatnonzero(qualifies).astype(np.int64)
    alpha = surviving[top2_local[admitted_rows, 0]].astype(np.int64)
    beta = surviving[top2_local[admitted_rows, 1]].astype(np.int64)
    return {"rows": admitted_rows, "alpha": alpha, "beta": beta}


# --- CAE-05 non-gating topology-preservation measures (eqs. 20/21) --------------------------


def unfaithfulness_coverage(
    model: "ChartAutoEncoder",
    x_train: torch.Tensor,
    surviving_indices: Sequence[int],
    n_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """eqs. 20 and 21, CAE-05's reported, non-gating topology-preservation measures.
    Samples ``n_samples`` points uniformly in the chart space of the surviving
    (post-pruning) charts -- split round-robin across ``surviving_indices`` so the
    sample-to-chart assignment is deterministic given ``seed`` and needs no separate
    random draw of its own -- decodes each through its chart decoder and the shared
    embedding decoder, and for each decoded sample finds its nearest training point.

    Unfaithfulness (eq. 20) is the mean squared distance from each decoded sample to that
    nearest training point; coverage (eq. 21) is the number of *distinct* training points
    that are some sample's nearest neighbour, divided by ``n_samples``. The paper's own
    runs used a hundred samples; the pre-registered value here (``UNFAITHFUL_SAMPLES``)
    is larger for a less noisy coverage estimate. Returns ``unfaithfulness``,
    ``coverage``, ``n_samples``, ``n_distinct`` as native Python floats/ints."""
    surviving = list(surviving_indices)
    n_surv = len(surviving)
    chart_idx = torch.tensor([surviving[i % n_surv] for i in range(n_samples)], dtype=torch.long)

    g = torch.Generator().manual_seed(seed)
    coords = torch.rand((n_samples, model.chart_dim), generator=g)

    decoded = torch.empty((n_samples, model.out_dim), dtype=x_train.dtype)
    with torch.no_grad():
        for alpha in surviving:
            mask = chart_idx == alpha
            if bool(mask.any()):
                w = model.chart_decoders[alpha](coords[mask])
                decoded[mask] = model.embedding_decoder(w)

        d2 = torch.cdist(decoded, x_train) ** 2
        nearest_dist2, nearest_idx = d2.min(dim=1)

    unfaithfulness = float(nearest_dist2.mean().item())
    n_distinct = int(torch.unique(nearest_idx).numel())
    coverage = n_distinct / n_samples

    return {
        "unfaithfulness": unfaithfulness,
        "coverage": coverage,
        "n_samples": int(n_samples),
        "n_distinct": n_distinct,
    }


def fit_global_scale(d2_rep: np.ndarray, d2_geo: np.ndarray, mask: np.ndarray) -> float:
    """The one global isotropic scale factor T1's pre-registration grants: the median
    ratio of target (geodesic) to representation squared distances, fit on the supplied
    training-split-only subset and then applied by the caller to every pair. A neural
    encoder carries no constraint tying its output scale to the geodesic scale, unlike the
    classical and Krein coordinates 02.1 measured, whose scale comes from the
    eigendecomposition by construction; without this one disclosed scalar the statistic
    would be dominated by an arbitrary global factor and would measure nothing about
    geometry."""
    d2_rep = np.asarray(d2_rep, dtype=np.float64)
    d2_geo = np.asarray(d2_geo, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return float(np.median(d2_geo[mask] / d2_rep[mask]))


def embedding_distortion(
    z: np.ndarray,
    geo_pairs: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    train_mask: np.ndarray,
    chart_dim: int,
    embed_dim: int,
) -> Dict[str, Any]:
    """T1: the geodesic-distortion statistic, computed on the initial encoder's **global**
    embedding ``z`` in ``R^embed_dim`` -- never on chart-local coordinates. Two points
    assigned to different charts have coordinates in unrelated parameterisations, so a
    pairwise distance between them carries no geometric meaning; the initial encoder's
    output is the one global, near-isometric coordinate system a Chart Auto-Encoder
    produces, and it is the correct analogue of 02.1's classical and Krein coordinates.

    Squares ``geo_pairs`` before use, because the cached array it is drawn from holds
    **unsquared** geodesic distances despite the ``r2`` in its name (that ``r2`` refers to
    the pair-draw random seed, not to squaring). Fits exactly one global isotropic scale
    factor on the pairs whose both endpoints are in the training split
    (:func:`fit_global_scale`), applies it to every supplied pair, and delegates to
    ``geometry_probes.distortion_stats`` -- the same statistic 02.1 already validated,
    imported rather than reimplemented, so the number stays bit-comparable with 02.1's own
    measurement. Returns that function's dict plus ``global_scale_factor`` and
    ``n_pairs``.

    Raises ``ValueError`` when the supplied array's feature dimension equals
    ``chart_dim`` while differing from ``embed_dim`` -- exactly the chart-local-instead-
    of-global misuse T-02.2-11 mitigates against."""
    z = np.asarray(z, dtype=np.float64)
    if z.shape[1] == chart_dim and chart_dim != embed_dim:
        raise ValueError(
            f"embedding_distortion: supplied array has feature dimension {z.shape[1]}, "
            f"equal to the model's chart dimension ({chart_dim}) and differing from its "
            f"embedding dimension ({embed_dim}) -- this looks like a chart-local "
            "coordinate array, not the initial encoder's global embedding. Distortion "
            "must be computed on the global embedding only (RESEARCH.md Pitfall 3)."
        )

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    geo_pairs = np.asarray(geo_pairs, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)

    d2_rep = ((z[rows] - z[cols]) ** 2).sum(axis=1)
    d2_geo = geo_pairs**2  # cached array holds unsquared geodesic distances -- square first

    scale = fit_global_scale(d2_rep, d2_geo, train_mask)
    stats = geometry_probes.distortion_stats(scale * d2_rep, d2_geo)

    result: Dict[str, Any] = dict(stats)
    result["global_scale_factor"] = scale
    result["n_pairs"] = int(len(rows))
    return result


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
