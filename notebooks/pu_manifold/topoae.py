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
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.manifold import trustworthiness

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


# --- T1: topological fidelity -------------------------------------------------------------


def topological_fidelity(x: torch.Tensor, z: torch.Tensor) -> Dict[str, float]:
    """T1's raw statistic (CONTEXT.md D-01/D-03/D-04/D-05): the **entire** held-out
    array in one pass (n ~ 2,000; a 2,000x2,000 float64 distance matrix is trivial), not
    a batch-wise average -- a latent can preserve every 64-point neighbourhood while
    scrambling how regions connect, and D-03 gates on global topology.

    ``d_x = pairwise_distances_f64(x)`` uses **no ambient normalization** -- this
    resolves RESEARCH.md Open Question 1: the PU ambient data is L2-normalized onto the
    unit sphere so its Euclidean distances are already bounded in [0, 2], and the Swiss
    roll fixture is divided by one global standard deviation, so both are O(1) without
    any per-batch-maximum rescaling. The paper's own per-batch-max convention is
    deliberately not adopted: a per-batch maximum differs between the batch-wise
    *training* loss and this whole-set *gate*, which would make the two numbers
    incomparable.

    ``d_z = pairwise_distances_f64(z * latent_unit_scale(z))`` applies D-05's global
    isotropic latent rescaling to the latent side only, before distances are computed.

    Returns ``{"loss_x_to_z", "loss_z_to_x", "worse"}`` as native floats, where
    ``worse = max(loss_x_to_z, loss_z_to_x)``. input->latent (``loss_x_to_z``) catches
    destroyed structure; latent->input (``loss_z_to_x``) catches invented structure --
    these are different failure modes (D-04). Summing them, which is what the
    *training* loss's ``total`` does, would collapse the two failure modes D-04 keeps
    apart -- this function never returns ``total``."""
    d_x = pairwise_distances_f64(x)
    z_scaled = z * latent_unit_scale(z)
    d_z = pairwise_distances_f64(z_scaled)
    loss = topological_loss(d_x, d_z)
    loss_x_to_z = float(loss["loss_x_to_z"].item())
    loss_z_to_x = float(loss["loss_z_to_x"].item())
    return {
        "loss_x_to_z": loss_x_to_z,
        "loss_z_to_x": loss_z_to_x,
        "worse": max(loss_x_to_z, loss_z_to_x),
    }


# --- T3: rank structure ---------------------------------------------------------------------


def rank_structure(x_ambient: torch.Tensor, z_latent: torch.Tensor, k: int) -> Dict[str, float]:
    """T3's raw statistic (D-06/D-07/D-10): two calls to
    ``sklearn.manifold.trustworthiness`` -- ``trustworthiness(x_ambient, z_latent,
    n_neighbors=k)`` and ``trustworthiness(z_latent, x_ambient, n_neighbors=k)``, the
    second being continuity by definition (the same identity the paper's own reference
    ``measures.py`` uses -- continuity is never hand-rolled here).

    Returns ``{"trustworthiness", "continuity", "gate_value"}`` where ``gate_value =
    1.0 - min(trustworthiness, continuity)`` -- the inversion D-10 requires so that
    higher-is-better (both quantities lie in [0, 1]) becomes lower-is-better and the
    strict-less-than gate comparison runs the same direction as T1 and T2.

    ``sklearn`` raises when ``n_neighbors >= n_samples / 2``; at the planned holdout
    size of ~2,000 and the k sweep {5, 15, 30} this is far from binding, so a future
    holdout-fraction change would surface as an explicit error rather than a silent
    ``nan``."""
    x_raw = x_ambient.detach().cpu().numpy() if isinstance(x_ambient, torch.Tensor) else x_ambient
    z_raw = z_latent.detach().cpu().numpy() if isinstance(z_latent, torch.Tensor) else z_latent
    x_np = np.asarray(x_raw, dtype=np.float64)
    z_np = np.asarray(z_raw, dtype=np.float64)
    trust = float(trustworthiness(x_np, z_np, n_neighbors=k))
    cont = float(trustworthiness(z_np, x_np, n_neighbors=k))  # continuity = swapped args
    return {
        "trustworthiness": trust,
        "continuity": cont,
        "gate_value": 1.0 - min(trust, cont),
    }


# --- baseline-relative gate values -----------------------------------------------------------


def _baseline_relative_ratio(name: str, model_value: float, baseline_value: float) -> float:
    """Shared division-by-baseline guard for the three gate-value functions below:
    raises ``ValueError`` naming ``name`` when ``baseline_value`` is zero or
    non-finite, because a silent division producing ``inf``/``nan`` must never reach
    the verdict."""
    baseline_value = float(baseline_value)
    if baseline_value == 0.0 or not math.isfinite(baseline_value):
        raise ValueError(
            f"{name}: baseline quantity is zero or non-finite ({baseline_value!r}) -- "
            "refusing to divide, which would silently produce inf or nan"
        )
    return float(model_value) / baseline_value


def t1_gate_value(fid_model: Dict[str, float], fid_baseline: Dict[str, float]) -> float:
    """T1's baseline-relative margin (D-02/D-08): ``model_quantity / baseline_quantity``
    using each side's ``worse`` -- the max over the two *directional* terms, taken
    before the ratio, exactly as ``cae_evaluate_run.py`` takes a max over two controls.
    Rejected alternative: forming one ratio from the summed ``total`` would collapse
    D-04's two failure modes back together. Raises ``ValueError`` naming the metric
    when the baseline quantity is zero or non-finite."""
    return _baseline_relative_ratio("t1_topo_fidelity", fid_model["worse"], fid_baseline["worse"])


def t2_gate_value(stats_model: Dict[str, float], stats_baseline: Dict[str, float]) -> float:
    """T2's baseline-relative margin (CAE-03's inherited convention): ``model
    mse_per_dim / baseline mse_per_dim``. Raises ``ValueError`` naming the metric when
    the baseline quantity is zero or non-finite."""
    return _baseline_relative_ratio(
        "t2_recon_margin", stats_model["mse_per_dim"], stats_baseline["mse_per_dim"]
    )


def t3_gate_value(rank_model: Dict[str, float], rank_baseline: Dict[str, float]) -> float:
    """T3's baseline-relative margin: ``model gate_value / baseline gate_value``.
    Raises ``ValueError`` naming the metric when the baseline quantity is zero or
    non-finite."""
    return _baseline_relative_ratio(
        "t3_rank_structure", rank_model["gate_value"], rank_baseline["gate_value"]
    )


# --- non-gating context metric: unfaithfulness / coverage shim -------------------------------


def unfaithfulness_adapter(model: "cae_mod.PlainAutoEncoder", z_reference: torch.Tensor) -> SimpleNamespace:
    """The non-gating context metric's shim: returns a lightweight duck-typed object
    exposing exactly the attributes ``cae.unfaithfulness_coverage`` reads --
    ``chart_dim``, ``out_dim``, ``chart_decoders``, ``embedding_decoder`` -- so
    ``cae.unfaithfulness_coverage(adapter, x_holdout, [0], n_samples, seed)`` runs
    unmodified against a TopoAE, which has one global latent and no charts at all.

    ``chart_dim = model.latent_dim``, ``out_dim`` is read off the last ``nn.Linear`` in
    ``model.decoder`` (the ambient dimension), ``chart_decoders = [f]`` where ``f`` is
    the affine map carrying the unit box ``[0, 1]^chart_dim`` onto the componentwise
    min/max bounding box of ``z_reference``, and ``embedding_decoder = model.decode``.

    The affine map is what keeps the numbers comparable with 02.2's: CAE chart
    coordinates live in ``(0, 1)^d`` by construction while a TopoAE latent is unbounded,
    so without the map the uniform ``[0, 1]^d`` sample ``cae.unfaithfulness_coverage``
    draws would fall almost entirely outside the latent's support. This rescaling is a
    stated difference from 02.2's measurement, the same way D-05 records T1's
    non-comparability."""
    z_ref = z_reference.detach()
    lo = z_ref.min(dim=0).values
    hi = z_ref.max(dim=0).values
    span = hi - lo

    def _affine_box_to_latent(coords: torch.Tensor) -> torch.Tensor:
        return coords.to(span.dtype) * span + lo

    return SimpleNamespace(
        chart_dim=int(model.latent_dim),
        out_dim=int(model.decoder[-1].out_features),
        chart_decoders=[_affine_box_to_latent],
        embedding_decoder=model.decode,
    )


# --- verdict rule (threat T-02.4-11) --------------------------------------------------------

# Exactly the three T1/T2/T3 names, in the fixed order the SPEC's "ordering / R5" edge
# requires. These are the ONLY names that may appear in any verdict artifact this phase
# writes -- cae.py's own GATING_METRICS tuple names a different statistic (T1's borrowed
# slot is literally the geodesic-distortion name) and must never leak into output here.
GATING_METRICS: Tuple[str, ...] = ("t1_topo_fidelity", "t2_recon_margin", "t3_rank_structure")

# `cae.verdict_from_metrics` reads `cae.py`'s own module-level three-name tuple, so it
# cannot be called directly against a differently-named metric dict. Reusing those three
# borrowed names as the gate names in the artifact is rejected (the first of them is the
# geodesic-distortion name, and an artifact whose gating-metric list printed that name
# would read as gating on the exact statistic SPEC R4 forbids gating on).
# Monkeypatching `cae.GATING_METRICS` is also rejected -- it mutates state another
# module owns. The adopted route is a positional slot remap built by zipping the two
# tuples (never by hard-coding the borrowed strings), so the tested comparison logic in
# `cae.verdict_from_metrics` is executed, never re-implemented, and the borrowed names
# never appear in any artifact this phase writes.
assert len(GATING_METRICS) == 3 and len(cae_mod.GATING_METRICS) == 3, (
    "GATING_METRICS and cae.GATING_METRICS must both have exactly three entries for "
    "the positional slot remap in CAE_SLOT_ALIASES to be well-defined"
)
CAE_SLOT_ALIASES: Dict[str, str] = dict(zip(GATING_METRICS, cae_mod.GATING_METRICS))

VERDICT_RULE = (
    "PASS requires all three gates (t1_topo_fidelity, t2_recon_margin, "
    "t3_rank_structure) to hold. Every comparison is strict less-than -- a value "
    "exactly at a threshold does not clear it. There is no MARGINAL tier: every "
    "non-PASS outcome routes to the same halt-for-user-decision consequence, so a "
    "middle tier would carry no distinct consequence."
)


def verdict_from_topoae_metrics(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Delegating verdict wrapper (threat T-02.4-11). For each name in
    :data:`GATING_METRICS`, raises ``ValueError`` naming **that** topoae name if it is
    absent from ``metrics`` or if ``float(metrics[name])`` is not finite, before
    anything else runs -- so a raised error message never leaks a borrowed ``cae.py``
    slot name, and this check always fires before ``cae.verdict_from_metrics``'s own
    (slot-named) absent/non-finite guard could.

    Then builds slot-keyed ``metrics``/``thresholds`` dicts through
    :data:`CAE_SLOT_ALIASES` and calls ``cae.verdict_from_metrics`` on them -- the
    strict-less-than comparison itself is performed only by that already-tested
    function, never re-implemented here -- and maps the returned ``gate_detail`` keys
    back to the topoae names before returning ``(verdict, gate_detail)``.

    The shared ``cae`` module global (``cae.GATING_METRICS``) is never monkeypatched;
    this wrapper reads it once at import time (see :data:`CAE_SLOT_ALIASES`) and
    otherwise leaves ``cae.py`` untouched."""
    for name in GATING_METRICS:
        if name not in metrics:
            raise ValueError(f"verdict_from_topoae_metrics: gating metric {name!r} is absent from metrics")
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(
                f"verdict_from_topoae_metrics: gating metric {name!r} is non-finite "
                f"({value!r}) -- a missing or non-finite measurement can never become a PASS"
            )

    slot_metrics = {CAE_SLOT_ALIASES[name]: float(metrics[name]) for name in GATING_METRICS}
    slot_thresholds = {CAE_SLOT_ALIASES[name]: thresholds[name] for name in GATING_METRICS}

    verdict, slot_gate_detail = cae_mod.verdict_from_metrics(slot_metrics, slot_thresholds)

    gate_detail = {name: slot_gate_detail[CAE_SLOT_ALIASES[name]] for name in GATING_METRICS}
    return verdict, gate_detail


def write_topoae_verdict(
    fit_key: str,
    metrics: Dict[str, Any],
    thresholds: Dict[str, Any],
    verdict: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mirrors ``cae.write_cae_verdict``'s shape but does not call it -- that function
    hard-codes the ``cae_verdict_`` stem and this phase's stem is
    ``topoae_verdict_{fit_key}``. Delegates persistence to ``cache.json_cache`` with
    ``cfg = {"fit_key": fit_key, "phase": "02.4", "thresholds": thresholds}`` --
    thresholds go **inside** the cfg dict so that editing one afterwards raises
    ``cache._manifest_matches``'s mismatch ``ValueError`` instead of silently
    re-verdicting, the guard that already caught the superseded Krein handoff in
    Phase 02.1.

    Recomputes ``gate_detail`` from ``metrics``/``thresholds`` via
    :func:`verdict_from_topoae_metrics` (rather than trusting a separately supplied
    value) and raises ``ValueError`` if the recomputed verdict disagrees with the
    supplied ``verdict`` -- a verdict artifact whose stored outcome disagrees with its
    own gates must never be written.

    The payload carries ``fit_key``, ``phase`` (``"02.4"``), ``TOPOAE_VERDICT``,
    ``gating_metrics`` (:data:`GATING_METRICS` as a list, so the artifact states its own
    gate list), ``metrics``, ``thresholds``, ``gate_detail``, ``verdict_rule``, and,
    under the single non-gating key ``context_metrics``, everything reported but never
    gated (geodesic distortion, unfaithfulness/coverage, the batch-wise T1 average --
    supplied by the caller through ``extra``). Every value passes through
    ``cae.to_native`` before it reaches ``json_cache``, unrounded, so full float
    precision survives.

    Called on PASS and FAIL alike -- there is no branch that skips the write."""
    computed_verdict, gate_detail = verdict_from_topoae_metrics(metrics, thresholds)
    if computed_verdict != verdict:
        raise ValueError(
            f"write_topoae_verdict: supplied verdict {verdict!r} does not match the "
            f"verdict computed from metrics/thresholds ({computed_verdict!r}) -- "
            "refusing to write a verdict artifact whose stored outcome disagrees with "
            "its own gates"
        )

    cfg = {"fit_key": fit_key, "phase": "02.4", "thresholds": thresholds}

    def _compute() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "fit_key": fit_key,
            "phase": "02.4",
            "TOPOAE_VERDICT": verdict,
            "gating_metrics": list(GATING_METRICS),
            "metrics": metrics,
            "thresholds": thresholds,
            "gate_detail": gate_detail,
            "verdict_rule": VERDICT_RULE,
        }
        if extra:
            payload["context_metrics"] = extra
        return cae_mod.to_native(payload)

    return cache.json_cache(f"topoae_verdict_{fit_key}", cfg, _compute)


# --- PASS-only handoff and R6's active deletion of a stale handoff on FAIL -----------------


def write_topoae_handoff(fit_key: str, verdict: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mirrors ``cae.write_cae_handoff``'s shape and its refusal: raises ``ValueError``
    unless ``verdict == "PASS"``, so a non-PASS caller cannot produce a handoff even by
    mistake. Persists through ``cache.json_cache`` under stem
    ``topoae_handoff_{fit_key}`` with ``cfg = {"fit_key": fit_key, "phase": "02.4"}``.

    Fields, modelled on ``gate_verdict_{fit_key}.json`` and Phase 02.1's
    ``geometry_handoff_{fit_key}.json`` and naming exactly what Phase 3 consumes:
    ``representation_id`` (the string ``"topoae"``, the identity field this milestone's
    handoffs now key on rather than the method's prose name), ``primary_d``,
    ``fit_artifact_keys`` (the npz stems of the primary-rung fits), ``encoder_state_key``,
    ``gate_values`` and ``thresholds`` for all three gates, ``evidence_criteria`` (the
    criteria Phase 3 should judge the representation by, in prose),
    ``reinstated_requirements`` (the DEC and CURV requirement IDs a coordinate-producing
    win reinstates), ``activation``, ``torch_version``, ``numpy_version``, and a UTC
    ``timestamp``. Every value through ``cae.to_native``."""
    if verdict != "PASS":
        raise ValueError(
            f"write_topoae_handoff refuses to write a handoff for a non-PASS verdict: {verdict!r}"
        )

    cfg = {"fit_key": fit_key, "phase": "02.4"}

    def _compute() -> Dict[str, Any]:
        result = {
            "fit_key": fit_key,
            "phase": "02.4",
            "representation_id": "topoae",
            "primary_d": payload["primary_d"],
            "fit_artifact_keys": payload["fit_artifact_keys"],
            "encoder_state_key": payload["encoder_state_key"],
            "gate_values": payload["gate_values"],
            "thresholds": payload["thresholds"],
            "evidence_criteria": payload["evidence_criteria"],
            "reinstated_requirements": payload["reinstated_requirements"],
            "activation": payload["activation"],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return cae_mod.to_native(result)

    return cache.json_cache(f"topoae_handoff_{fit_key}", cfg, _compute)


def clear_stale_handoff(fit_key: str) -> bool:
    """R6's active deletion. Unlinks both
    ``cache.cache_path(f"topoae_handoff_{fit_key}", "json")`` and its
    ``cache.cache_path(f"topoae_handoff_{fit_key}", "meta.json")`` sidecar when present,
    returning ``True`` if anything was removed and ``False`` otherwise, and never
    raising when nothing is there.

    Deliberate deviation from precedent: Phase 02.1 archived its superseded Krein
    handoff as a ``.superseded-krein.json`` variant rather than removing it. R6 chooses
    deletion instead, because the failure this guards against is a stale PASS handoff
    from an earlier run surviving a later FAIL and being read by Phase 3 as live. The
    sidecar must go too -- leaving it behind would let a later ``json_cache`` call see a
    manifest with no artifact."""
    json_path = cache.cache_path(f"topoae_handoff_{fit_key}", "json")
    meta_path = cache.cache_path(f"topoae_handoff_{fit_key}", "meta.json")

    removed = False
    if json_path.exists():
        json_path.unlink()
        removed = True
    if meta_path.exists():
        meta_path.unlink()
        removed = True
    return removed
