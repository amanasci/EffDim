"""
Pure numpy/torch functions for Phase 02.4's Topological Auto-Encoder (Moor, Horn, Rieck,
Borgwardt, "Topological Autoencoders," ICML 2020, arXiv:1906.00722). Tensors and arrays
in, tensors and dicts out -- no file I/O, no cache handling; the runners under
``notebooks/diagnostics/`` own paths. Constants live in ``02.4-PREREGISTRATION.md``.

Unlike its sibling modules, this one imports torch at module level: the TopoAE model
genuinely needs it. For the same reason ``curvature.py`` and ``mknn.py`` are excluded
from ``pu_manifold/__init__.py``'s eager imports (so Phase 1-only callers do not need
torch installed to import the package), this module is deliberately NOT re-exported
there either.

``notebooks/pu_manifold/cae.py`` is Phase 02.2's sealed artifact and is never edited by
this phase -- every generic piece this module needs (the matched-capacity baseline, the
gate engine, artifact writers) is imported from it, never copied.
"""

import math
import time
from typing import Any, Dict

import numpy as np
import torch

from . import cache
from . import cae as cae_mod


# --- distances ---------------------------------------------------------------------------


def pairwise_distances_f64(a: torch.Tensor) -> torch.Tensor:
    """``torch.cdist`` on ``a.double()``, returning a float64 ``(n, n)`` distance matrix.
    Every distance matrix this module computes goes through this one function so the
    float64 discipline required for deterministic tie detection (RESEARCH.md Pitfall 10)
    cannot be forgotten at a call site."""
    a64 = a.double()
    return torch.cdist(a64, a64)


# --- 0-dimensional persistence -------------------------------------------------------------


def persistence_pairs(D: np.ndarray) -> np.ndarray:
    """0-dimensional persistence pairs -- the minimum spanning tree edge set of the
    complete graph on ``D``'s pairwise distances. ``D`` must be a square, symmetric
    array with ``n >= 2`` points; whatever dtype it arrives in (including float32), it
    is cast with ``np.asarray(D, dtype=np.float64)`` before any comparison -- explicitly,
    not left implicit -- so tie detection is exact equality after float64 computation
    and never depends on a float32 rounding accident.

    Ties are broken lexicographically on ``(distance, row_index, col_index)``: because
    ``np.triu_indices`` already emits row-major upper-triangular index order, a
    *stable* sort (``np.argsort(..., kind="stable")``) on weight alone realizes the
    full lexicographic rule with no extra sort key. Do NOT "optimize" this to an
    unstable sort -- that would silently break the determinism this function exists to
    guarantee.

    An early break once ``n - 1`` pairs are found keeps the dominant cost the
    ``O(n^2 log n)`` sort rather than a full Python loop over every edge: every edge
    considered after the ``(n-1)``th merge necessarily closes a cycle in a complete
    graph and would be skipped anyway (RESEARCH.md Pitfall 5) -- this matters at the T1
    gate's ``n ~ 2000`` scale (~2,000,000 upper-triangular edges).

    Raises ``ValueError`` naming the received batch size when ``n < 2``: a zero-loss
    fallback is deliberately rejected here because it would silently disable the
    topological term rather than surfacing the batch as unusable. Also raises
    ``ValueError`` if ``D`` is not a square 2-d array or is not symmetric.

    Do NOT reach for scipy's sparse-graph MST routine here -- its internal tie-break
    order is not a documented, version-stable API contract, and this loss's value
    depends on which edges were selected (RESEARCH.md Pitfall 1)."""
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"persistence_pairs: D must be a square 2-d array, got shape {D.shape!r}")
    n = D.shape[0]
    if n < 2:
        raise ValueError(
            f"persistence_pairs: batch size {n} < 2 -- undefined below 2 points. A "
            "zero-loss fallback is deliberately rejected: it would silently disable "
            "the topological term rather than surfacing the batch as unusable."
        )
    if not np.allclose(D, D.T):
        raise ValueError("persistence_pairs: D must be symmetric")

    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(D[iu, ju], kind="stable")

    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for k in order:
        i, j = int(iu[k]), int(ju[k])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        pairs.append((i, j))
        if len(pairs) == n - 1:
            break
    return np.array(pairs, dtype=np.int64)


def topological_loss(d_x: torch.Tensor, d_z: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The paper's ``match_edges="symmetric"`` topological loss: pairs selected by
    ``d_x``'s own MST compare their length in ``d_x`` vs. ``d_z`` (catches destroyed
    structure); pairs selected by ``d_z``'s own MST compare their length in ``d_z`` vs.
    ``d_x`` (catches invented structure). MST edge *selection* runs on a
    ``.detach().cpu().numpy()`` copy of each distance matrix -- a non-differentiable
    numpy step -- but the returned loss terms index the **original, gradient-carrying**
    ``d_x``/``d_z`` tensors at those positions, so gradients flow through edge
    *lengths*, never through edge *selection*. Callers must not ``.detach()`` ``d_x``
    or ``d_z`` themselves -- only this function's internal numpy step touches a
    detached copy.

    Returns ``{"loss_x_to_z": ..., "loss_z_to_x": ..., "total": loss_x_to_z +
    loss_z_to_x}`` as torch scalars. The **training** loss uses ``total``; the T1
    **gate** evaluates the two directional terms separately and gates on the worse one
    (CONTEXT.md D-04) -- callers computing a gate value must not collapse them back to
    ``total``.

    Raises ``ValueError`` if ``d_x`` and ``d_z`` do not share a shape, or if either is
    not float64 -- distance matrices must go through :func:`pairwise_distances_f64` (or
    an equivalent float64 cast) before reaching this function, matching SPEC's
    determinism constraint."""
    if d_x.shape != d_z.shape:
        raise ValueError(
            f"topological_loss: d_x and d_z must have the same shape, got "
            f"{tuple(d_x.shape)} vs {tuple(d_z.shape)}"
        )
    if d_x.dtype != torch.float64 or d_z.dtype != torch.float64:
        raise ValueError(
            f"topological_loss: d_x and d_z must both be float64, got d_x.dtype="
            f"{d_x.dtype} and d_z.dtype={d_z.dtype}"
        )
    pairs_x = persistence_pairs(d_x.detach().cpu().numpy())
    pairs_z = persistence_pairs(d_z.detach().cpu().numpy())
    ix, jx = pairs_x[:, 0], pairs_x[:, 1]
    iz, jz = pairs_z[:, 0], pairs_z[:, 1]

    loss_x_to_z = ((d_x[ix, jx] - d_z[ix, jx]) ** 2).sum()
    loss_z_to_x = ((d_z[iz, jz] - d_x[iz, jz]) ** 2).sum()
    return {
        "loss_x_to_z": loss_x_to_z,
        "loss_z_to_x": loss_z_to_x,
        "total": loss_x_to_z + loss_z_to_x,
    }


# --- latent normalization ------------------------------------------------------------------


def latent_unit_scale(z: torch.Tensor) -> float:
    """D-05's latent normalization, resolved as the **global isotropic** reading: the
    single scalar ``1 / sqrt(mean(var(z, axis=0)))``, so ``z * latent_unit_scale(z)``
    has mean per-dimension variance 1. Per-dimension standardization (dividing each
    axis by its own standard deviation) is deliberately rejected -- that would be a
    non-isometric rescaling that changes which points are nearest neighbours along
    different axes, corrupting the very topology T1 measures (RESEARCH.md Pitfall 3 /
    Assumption A1)."""
    z64 = z.detach().double()
    mean_var = z64.var(dim=0, unbiased=False).mean()
    return float(1.0 / torch.sqrt(mean_var))


# --- lambda warm-up schedule ----------------------------------------------------------------


def lambda_schedule(epoch: int, cfg: Dict[str, Any]) -> float:
    """D-15's warm-up-then-constant schedule: reconstruction-only (``lambda_t = 0.0``)
    while ``epoch < floor(warmup_frac * max_epochs)``; then a **linear** ramp from 0 to
    ``cfg["lambda_topo"]`` across the next ``floor(ramp_frac * max_epochs)`` epochs;
    then constant ``cfg["lambda_topo"]`` for every epoch after. Linear is the only ramp
    shape this function implements."""
    lambda_topo = float(cfg["lambda_topo"])
    max_epochs = cfg["max_epochs"]
    warmup_frac = cfg["warmup_frac"]
    ramp_frac = cfg["ramp_frac"]

    warmup_epochs = math.floor(warmup_frac * max_epochs)
    ramp_epochs = math.floor(ramp_frac * max_epochs)

    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0 or epoch >= warmup_epochs + ramp_epochs:
        return lambda_topo
    progress = (epoch - warmup_epochs + 1) / ramp_epochs
    return lambda_topo * min(progress, 1.0)


# --- training loop --------------------------------------------------------------------------


def train_topoae(model: "cae_mod.PlainAutoEncoder", x_train: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mirrors ``cae._train_decoder_protocol``'s single-stage seeding/batching/
    three-way-stopping discipline -- same ``torch.manual_seed(cfg["seed"])`` +
    ``np.random.default_rng(seed)`` seeding, same ``torch.optim.AdamW`` optimizer, same
    per-epoch ``rng.permutation(n)`` batching, the same three-way stopping rule
    (``max_epochs`` cap, relative-plateau early stop, ``wallclock_ceiling_s`` checked
    every epoch), and the same returned dict keys (``history``, ``epochs_run``,
    ``wallclock_s``, ``wallclock_truncated``, ``early_stopped``, ``cfg``). Does NOT
    import the private ``_train_decoder_protocol`` helper and does NOT reuse
    ``cae.train_cae``'s two-stage FPS-pre-training structure -- a TopoAE has no charts,
    so there is nothing to pre-train per chart (RESEARCH.md Pitfall 8).

    Differences from the mirrored protocol are recorded explicitly in the returned
    ``cfg["protocol_difference"]`` string, the way ``cae.train_plain_ae`` does: batches
    use ``drop_last=True`` -- a final batch holding fewer than 2 points is dropped and
    never reaches :func:`persistence_pairs`, since persistence is undefined below 2
    points -- and the per-batch loss is ``recon + lambda_schedule(epoch, cfg) * topo``
    where ``d_x = pairwise_distances_f64(xb)``, ``z = model.encode(xb.float())``,
    ``d_z = pairwise_distances_f64(z)``, ``recon = ((xb.double() -
    model.decode(z).double()) ** 2).sum(-1).mean()``, and ``topo =
    topological_loss(d_x, d_z)["total"] / xb.shape[0]`` (batch-size normalized, the
    paper's own convention). Each ``history`` entry records ``epoch``, ``stage``,
    ``recon``, ``topo``, ``lambda_t``, ``total``.

    Raises ``ValueError`` naming the epoch and batch index if a batch's total loss is
    non-finite -- a rung whose loss diverges halts the run rather than being silently
    dropped from the ladder."""
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = x_train.shape[0]
    batch_size = cfg["batch"]
    max_epochs = cfg["max_epochs"]

    effective_cfg: Dict[str, Any] = dict(cfg)
    effective_cfg.setdefault("seed", seed)
    effective_cfg.setdefault("early_stop_min_delta", 0.0)
    effective_cfg.setdefault("early_stop_patience", max_epochs + 1)
    effective_cfg.setdefault("wallclock_ceiling_s", float("inf"))
    effective_cfg["protocol_difference"] = (
        "mirrors cae._train_decoder_protocol's single-stage seeding/batching/"
        "three-way-stopping discipline, not cae.train_cae's two-stage FPS-pretraining "
        "structure (a TopoAE has no charts, nothing to pre-train per chart); batches "
        "use drop_last=True so a final batch holding fewer than 2 points is dropped, "
        "never reaching persistence_pairs; per-batch loss adds "
        "lambda_schedule(epoch, cfg) * topological_loss(d_x, d_z)['total'] / "
        "batch_size to the reconstruction term"
    )

    early_stop_min_delta = effective_cfg["early_stop_min_delta"]
    early_stop_patience = effective_cfg["early_stop_patience"]
    wallclock_ceiling_s = effective_cfg["wallclock_ceiling_s"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history: list = []
    start = time.monotonic()
    wallclock_truncated = False
    early_stopped = False
    epochs_run = 0
    best_loss = float("inf")
    plateau_count = 0

    for epoch in range(max_epochs):
        perm = torch.from_numpy(rng.permutation(n))
        lam_t = lambda_schedule(epoch, effective_cfg)
        epoch_recon = 0.0
        epoch_topo = 0.0
        epoch_total = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if idx.shape[0] < 2:
                # drop_last=True: a final batch below the persistence floor of 2 points
                # is dropped, never reaching persistence_pairs.
                continue

            xb = x_train[idx].double()
            z = model.encode(xb.float()).double()
            d_x = pairwise_distances_f64(xb)
            d_z = pairwise_distances_f64(z)
            recon = ((xb - model.decode(z.float()).double()) ** 2).sum(-1).mean()
            topo = topological_loss(d_x, d_z)["total"] / xb.shape[0]
            total = recon + lam_t * topo

            if not math.isfinite(total.item()):
                raise ValueError(
                    f"train_topoae: non-finite total loss at epoch {epoch}, batch "
                    f"{n_batches} (recon={recon.item()!r}, topo={topo.item()!r}) -- "
                    "halting rather than silently dropping this rung from the ladder"
                )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_topo += topo.item()
            epoch_total += total.item()
            n_batches += 1

        if n_batches == 0:
            raise ValueError(
                f"train_topoae: epoch {epoch} produced zero usable batches "
                f"(n={n}, batch={batch_size}) -- every batch was below the "
                "persistence floor of 2 points"
            )

        epoch_mean_total = epoch_total / n_batches
        history.append(
            {
                "epoch": epoch,
                "stage": "main",
                "recon": epoch_recon / n_batches,
                "topo": epoch_topo / n_batches,
                "lambda_t": lam_t,
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
