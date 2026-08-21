"""
Phase 03.1 plan 01 (D-13): a decoder-prior weight-ladder runner on the `d=20` saddle fixture --
train one cell under `decoder_priors.decoder_prior_active`, measure curvature on the reduced
evaluation set, score the four fidelity axes, and record the D-15/CURV-04 absolute-scale fields
alongside the existing `cond(g)`.

**Why this file exists.** Phase 3 measured `cond(g)` ranking two collapsed seeds (1.01e+03,
1.76e+03) *ahead of* the only healthy one (3.30e+07) -- `cond(g)` alone is scale-invariant and
cannot see a uniformly collapsed metric. `chart_curvature.py` now also returns `lambda_min`,
`lambda_max`, `det_g` and `log10_det_g` (D-15), and this runner is the first consumer of them
outside a test.

**Copied verbatim, never re-typed** (see `03.1-PATTERNS.md`): the saddle fixture, the matched-
architecture builder and the blocked trainer come from `synthetic_control_run.py`
(`build_fixture`, `build_matched_cae`, `train_blocked`, `_fidelity_axes`); the matched PU
protocol constants come from `curvature_field_pu_run.py`; the resumable-JSONL and `cell_key`
idioms come from `swiss_roll_isometry_prior_sweep_run.py`.

**What this plan does NOT do.** It does not run the 24-cell ladder -- that is a later plan,
gated behind this file's `--dry-run` sizing report (Task 3) and a separate probe. This file's
`--smoke` path proves the full stack (train under a prior -> curvature on the reduced
evaluation set -> four fidelity axes -> D-15's four absolute-scale fields -> one JSONL record ->
a printed read-out) at a deliberately tiny scale, so nothing about the real ladder's wiring is
discovered only after eight hours of CPU are spent on it.

Invoke:
    .venv/bin/python notebooks/diagnostics/decoder_prior_ladder_run.py --dry-run
    .venv/bin/python notebooks/diagnostics/decoder_prior_ladder_run.py --smoke
"""

import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Matched protocol and matched fixture/trainer, imported and never re-typed.
import curvature_field_pu_run as pu  # noqa: E402
import synthetic_control_run as sc  # noqa: E402
from pu_manifold import cae, cache, chart_curvature, decoder_curvature, decoder_priors  # noqa: E402

RECORD_STEM = "03.1_decoder_prior_ladder"

# =============================================================================================
# Section A -- architecture and fixture, inherited from synthetic_control_run.py, not restated.
# See this plan's <pre_declared_constants_and_reading_rules> section A for the full table and
# the reasoning (fixture seed FIXED, only model seed varies; early stopping disabled for every
# cell by inheriting pu._converge_cfg via sc.train_blocked).
# =============================================================================================

LADDER_FIXTURE = "saddle"
"""D-01: the saddle is the only fixture with genuine trace-cancellation points."""

LADDER_N = sc.CONTROL_N
LADDER_CHART_DIM = pu.PU_CHART_DIM
LADDER_AMBIENT = pu.AMBIENT_DIM
LADDER_EMBED_DIM = pu.PU_EMBED_DIM
LADDER_N_CHARTS = pu.CONVERGE_N_CHARTS
LADDER_DOMAIN_RADIUS = sc.SADDLE_DOMAIN_RADIUS
LADDER_FIXTURE_SEED = sc.CONTROL_FIXTURE_SEED
"""Held FIXED for every cell; only the model/training seed varies (LADDER_SEEDS below). The
mandatory anchor cell is the sole exception -- it reproduces the sealed row bit for bit."""
LADDER_EPOCH_BLOCK = sc.CONTROL_EPOCH_BLOCK

# =============================================================================================
# Section B -- ladder sizing, pre-declared, probe-selected.
# =============================================================================================

LADDER_SEEDS: Tuple[int, ...] = (20260813, 20260814, 20260815)
"""D-03: Phase 3's own three-seed reporting unit."""

LADDER_MODES: Tuple[str, ...] = ("scale", "christoffel")
"""D-02: `conformal` and `isometry` are deferred."""

LADDER_FRACTIONS: Tuple[float, ...] = (0.01, 0.1, 1.0)
"""D-04: declared fractions of base reconstruction loss at initialization; 3 rungs per mode."""

LADDER_EVAL_N = 2000
"""D-16: the pu._split holdout rows, exactly curvature_field_pu_run.py's own selection-grid
evaluation set. See this plan's Section B for the full evaluation-sample-size projection."""

LADDER_EPOCH_CANDIDATES: Tuple[int, ...] = (100, 75, 50, 40, 30, 25, 20)
"""The probe (a later plan) picks the largest that fits LADDER_BUDGET_S."""

LADDER_BUDGET_S = 28800
"""8.0 h -- D-06's envelope, less the anchor cell."""

ANCHOR_CONFIG_ID = "control_saddle_nc4_d20_ep300"
"""D-13: the sealed row this runner's faithfulness check (a later plan) must reproduce."""

ANCHOR_RHO_SPEARMAN = -0.015106571347065712
"""D-13: matched with `==`, never a tolerance."""

CHRISTOFFEL_MAX_ROWS_PER_CHART = decoder_priors.CHRISTOFFEL_MAX_ROWS_PER_CHART
"""D-07's named valve (F3), `christoffel` arm only -- recorded per cell for provenance. Not
threaded into `decoder_priors.christoffel_penalty` by this plan; only Section C's relief
ladder (a later plan) actually varies it."""

# =============================================================================================
# Section B (continued) -- 03.1-03: the anchor's sealed reference wall clocks, and the probe's
# own sizing. Declared here, before any probe number exists.
# =============================================================================================

ANCHOR_TRAIN_WALLCLOCK_S = 6253.836718825012
"""D-13: `control_saddle_nc4_d20_ep300`'s sealed `train_wallclock_s` -- the anchor's own
machine-speed correction denominator. Same sealed measurement as `SEALED_TRAIN_WALLCLOCK_S`
below, named separately because this constant is `--anchor-check`'s own record-keeping value,
not the dry-run projection's."""

ANCHOR_CURV_WALLCLOCK_S = 2993.912286339997
"""D-13: `control_saddle_nc4_d20_ep300`'s sealed `curv_wallclock_s` -- ditto."""

PROBE_EPOCHS = 3
"""Half A's per-mode training-cost probe trains for this many epochs, at the real ladder
architecture, and reports `train_wallclock_s / epochs_run`. Deliberately tiny -- the probe
exists to be far cheaper than the anchor, never to train a real cell."""

PROBE_EVAL_SIZES: Tuple[int, ...] = (500, 1000, 2000)
"""Half B's curvature-evaluation-cost probe times `chart_curvature.chart_curvature_field` at
each of these three row counts, on ONE model, so `n_charts_used` is held constant across the
three measurements and the only thing varying is row count -- the direct test of research
assumption A2 (evaluation cost is linear in row count).

All three sizes must fit inside the `pu._split` holdout (`n=10000, PU_HOLDOUT_FRACTION=0.2` ->
2000 rows) -- `probe()` raises rather than silently truncating if a requested size exceeds it.
A first probe run declared `(1000, 2000, 4000)`; `4000` silently truncated to the 2000-row
holdout, producing an identical-cost measurement at what looked like two different sizes and a
false `EVAL COST IS NOT LINEAR IN ROW COUNT` verdict. Corrected to three sizes that all fit."""

F4_OVERRIDE_LABEL = "F4 (D-07 OVERRIDE: epoch cut)"
"""The literal string `apply_relief_ladder` appends to `relief_applied` when F4 fires under
`--confirm-f4-override` -- the single field that carries the override into `03.1-04`'s per-cell
records and `03.1-05`'s write-up, per `03.1-01-PLAN.md` §C item 4."""

RELIEF_LADDER: Tuple[Dict[str, str], ...] = (
    {
        "id": "F1",
        "description": "rungs 3 -> 2, keeping the extremes (0.01, 1.0) and dropping the middle",
        "consequence": "each mode's curve has 3 points (weight=0 plus 2 rungs) instead of 4; "
        "monotone movement is still readable, resolution is coarser.",
    },
    {
        "id": "F2",
        "description": "LADDER_EVAL_N 2000 -> 1000",
        "consequence": "more sampling noise per cell; eval_sample_size recorded per cell so a "
        "re-evaluation is well defined.",
    },
    {
        "id": "F3",
        "description": "CHRISTOFFEL_MAX_ROWS_PER_CHART 8 -> 4, christoffel arm ONLY",
        "consequence": "wider error bars on the christoffel arm specifically; does nothing for "
        "the scale arm because isometry_penalty has no row subsampling.",
    },
)
"""D-07's own fence stops here. F1-F3 are applied automatically, in this order, by
`apply_relief_ladder`. F4 is declared separately below `anchor_check` -- it is an override of
D-07, not a fourth entry in this tuple, and `apply_relief_ladder` never walks into it without
`confirm_f4_override=True`."""

COMBINATION_POST_HOC_LABEL = (
    "POST-HOC (weights selected after seeing the ladders; not an independent measurement)"
)
"""D-02: the combination cell's weights are chosen by `select_best_rung` AFTER the ladder's own
Tier-1 numbers exist, so it is a confirmation, not an independent third arm. This exact string
is printed on every table carrying a combination row -- `03.1-05-PLAN.md` calls this "not a
footnote" -- and `print_summary`'s post-render audit greps for it verbatim."""

# Smoke sizes -- deliberately tiny, to prove the wiring, never a real measurement.
SMOKE_N = 400
SMOKE_D = 4
SMOKE_AMBIENT = 24
SMOKE_N_CHARTS = 2
SMOKE_EPOCHS = 2
SMOKE_EVAL_N = 80


def _banner(msg: str) -> None:
    print("\n" + "=" * 92)
    print(msg)
    print("=" * 92)


# =============================================================================================
# D-04 -- per-mode weight calibration: absolute weights derived from declared fractions of the
# base reconstruction loss, measured once at initialization. No training run.
# =============================================================================================


def calibrate_weights(
    model: Any,
    x_batch: torch.Tensor,
    fractions: Sequence[float],
    christoffel_max_rows: int = CHRISTOFFEL_MAX_ROWS_PER_CHART,
) -> Dict[Tuple[str, float], Dict[str, float]]:
    """D-04: derive an absolute weight for every ``(mode, fraction)`` rung from
    ``fraction * base_recon / base_penalty[mode]``, where ``base_recon`` and each mode's
    ``base_penalty`` are measured by ONE forward pass on a freshly-initialized model --
    never by training.

    ``base_recon = cae.chart_loss(x_batch, out["y_charts"], out["p"])["recon"]``, the same
    per-row-min reconstruction term ``chart_loss`` already computes. Each mode's raw
    (``weight=1.0``, unweighted) penalty is measured by calling
    ``decoder_priors.isometry_penalty(model, x_batch, weight=1.0, mode="scale")`` and
    ``decoder_priors.christoffel_penalty(model, x_batch, weight=1.0,
    max_rows_per_chart=christoffel_max_rows)`` -- both penalty functions already multiply by
    ``weight`` internally, so ``weight=1.0`` returns the raw unweighted batch mean.

    **Two hard rules, both load-bearing:**

    1. **The calibration is computed ONCE**, on a freshly-constructed untrained model seeded
       with ``torch.manual_seed(LADDER_SEEDS[0])`` at the full ladder architecture, using the
       first ``pu.BATCH`` (64) rows of ``x_train32``, and the resulting absolute weights are
       reused for every seed. If each seed calibrated its own weights, a "rung" would be a
       different absolute weight per seed and would not be a rung at all -- this function does
       not enforce that on its own (it takes ``model``/``x_batch`` as arguments), so the
       caller is responsible for constructing them exactly this way and calling this function
       exactly once per ladder run.
    2. **Refuse a zero or non-finite base penalty**, by raising ``ValueError`` naming the mode
       and the measured value, rather than dividing. D-04's ladder is meaningless if the
       penalty is identically zero at init, and a silently computed ``inf`` weight would
       present downstream as a diverged cell rather than as this calibration failure.

    Returns a dict keyed by ``(mode, fraction)``, each value carrying ``weight``,
    ``base_recon`` and ``base_penalty`` -- recorded verbatim on every cell as
    ``weight_fraction``, ``base_recon_at_init`` and ``base_penalty_at_init`` so the
    derivation is reconstructable from the JSONL alone.
    """
    with torch.no_grad():
        out = model(x_batch)
    base_recon = float(cae.chart_loss(x_batch, out["y_charts"], out["p"])["recon"])

    # Deliberately NOT wrapped in torch.no_grad(): christoffel_penalty's Hessian goes through
    # jacfwd(jacfwd(...)), and at least one PyTorch build's forward-over-forward decomposition
    # for silu's second derivative itself needs an active (backward-mode) autograd graph to
    # decompose against -- wrapping this call in no_grad reproducibly raises
    # "Trying to use forward AD with aten::silu_backward that does not support it." No training
    # happens here regardless: the values are read once via float(...detach()...) and the graph
    # is discarded immediately, exactly as no_grad would have discarded it, without disabling
    # the machinery the decomposition needs.
    base_penalty: Dict[str, float] = {
        "scale": float(
            decoder_priors.isometry_penalty(model, x_batch, weight=1.0, mode="scale").detach()
        ),
        "christoffel": float(
            decoder_priors.christoffel_penalty(
                model, x_batch, weight=1.0, max_rows_per_chart=christoffel_max_rows
            ).detach()
        ),
    }
    for mode, penalty in base_penalty.items():
        if penalty == 0.0 or not np.isfinite(penalty):
            raise ValueError(
                f"calibrate_weights: base_penalty[{mode!r}] = {penalty!r} at initialization. "
                f"D-04's fraction-of-reconstruction weight derivation divides by this value "
                f"and is meaningless at zero; a silently computed inf/nan weight would present "
                f"downstream as a diverged cell rather than as this calibration failure."
            )

    calibrated: Dict[Tuple[str, float], Dict[str, float]] = {}
    for mode in LADDER_MODES:
        for fraction in fractions:
            weight = float(fraction) * base_recon / base_penalty[mode]
            calibrated[(mode, float(fraction))] = {
                "weight": weight,
                "base_recon": base_recon,
                "base_penalty": base_penalty[mode],
            }
    return calibrated


def _default_record_path() -> Path:
    return cache.cache_path(RECORD_STEM, "jsonl")


# =============================================================================================
# Resumable JSONL record -- same idiom as synthetic_control_run.py / the sweep runner.
# =============================================================================================


def cell_key(mode: str, weight: float, seed: int) -> Tuple[str, float, int]:
    """`(mode, weight, seed)` -- except at `weight == 0.0`, where the mode string is replaced
    by the sentinel `"any"`. `decoder_prior_active` installs nothing at all at weight 0.0
    regardless of which mode string is passed, so a weight=0.0 cell recorded under one mode is
    valid evidence for every other mode too -- no arm re-runs its own baseline."""
    weight = float(weight)
    mode_key = "any" if weight == 0.0 else mode
    return (mode_key, weight, int(seed))


def load_completed(record_path: Path) -> Dict[Tuple[str, float, int], Dict[str, Any]]:
    """Every `kind in ('ladder_cell', 'smoke_ladder_cell')` record on disk, keyed by
    :func:`cell_key`. Missing file -> empty dict (a fresh run, not an error).

    The shared JSONL record file also carries `kind == 'anchor_cell'` (03.1-03 Task 1) and
    `kind == 'probe'` (03.1-03 Task 2) records, neither of which has a `mode`/`weight`/
    `model_seed` triple to key on -- both are appended to the SAME file this runner's `--run-
    ladder` resumes from. Filtering to the two ladder-cell kinds here, rather than assuming
    every line is a cell, is what makes `--resume` safe against a record file the anchor/probe
    have already written into (as every real 03.1-03 run leaves behind)."""
    completed: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    if not record_path.exists():
        return completed
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") not in ("ladder_cell", "smoke_ladder_cell"):
                continue
            key = cell_key(rec["mode"], rec["weight"], rec["model_seed"])
            completed[key] = rec
    return completed


def append_record(record_path: Path, record: Dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as f:
        f.write(json.dumps(pu._to_jsonable(record)) + "\n")


# =============================================================================================
# One cell: train (with a prior optionally active) -> curvature on the reduced evaluation set
# -> four fidelity axes -> the D-15 absolute-scale fields -> one record.
# =============================================================================================


def run_cell(
    mode: str,
    weight: float,
    weight_fraction: Optional[float],
    model_seed: int,
    max_epochs: int,
    n: int,
    chart_dim: int,
    ambient: int,
    n_charts: int,
    embed_dim: int,
    eval_n: int,
    epoch_block: int,
    device: torch.device,
    christoffel_max_rows: int = CHRISTOFFEL_MAX_ROWS_PER_CHART,
    base_recon_at_init: Optional[float] = None,
    base_penalty_at_init: Optional[float] = None,
    relief_applied: Optional[List[str]] = None,
    second_mode: Optional[str] = None,
    second_weight: Optional[float] = None,
    second_weight_fraction: Optional[float] = None,
    selection_rule: Optional[str] = None,
) -> Dict[str, Any]:
    """One `(mode, weight, model_seed)` measurement, at the given `max_epochs`/`n`/`chart_dim`/
    `ambient`/`n_charts`/`embed_dim`/`eval_n` -- callers pass the real §A/§B constants for a
    ladder cell, or the SMOKE_* constants for `run_smoke`.

    D-12's combination arm (`03.1-05`): when `second_mode`/`second_weight` are given, training
    enters TWO nested `decoder_priors.decoder_prior_active` blocks around the same
    `sc.train_blocked` call -- the primary `(mode, weight)` pair OUTER, the `(second_mode,
    second_weight)` pair INNER, matching D-12 and the entry order
    `test_decoder_prior_active_nesting_composes_and_restores_exact` asserts (`scale` outer,
    `christoffel` inner). Zero new machinery: no `decoder_prior_active` signature change, no
    joined `PRIOR_MODES` entry -- just two `with` statements around one training call.
    `selection_rule`, when given, is recorded verbatim on the resulting cell so the weight
    choice is reconstructable from the JSONL alone. `mode`/`second_mode` are expected to be the
    two distinct values of `LADDER_MODES` (`"scale"` and `"christoffel"`, in either order) --
    the combination-specific `scale_weight`/`christoffel_weight` fields below are derived from
    whichever of the two matches each name."""
    fx = sc.build_fixture(LADDER_FIXTURE, n=n, d=chart_dim, D=ambient, seed=LADDER_FIXTURE_SEED)
    X = np.asarray(fx["X"], dtype=np.float64)
    H_true = np.asarray(fx["H_vec"], dtype=np.float64)

    x32 = torch.tensor(X, dtype=torch.float32).to(device)
    x64 = torch.tensor(X, dtype=torch.float64).to(device)

    train_idx, holdout_idx = pu._split(X.shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long, device=device)]
    x_holdout64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long, device=device)]

    # Reduced evaluation set (D-16): the same index array selects both the ambient rows and
    # the analytic truth, so eval rows and truth rows are the same rows in the same order.
    eval_idx = holdout_idx[:eval_n]
    x_eval64 = x64[torch.as_tensor(eval_idx, dtype=torch.long, device=device)]
    H_true_eval = H_true[eval_idx]

    torch.manual_seed(model_seed)
    model = sc.build_matched_cae(n_charts, chart_dim, embed_dim, ambient, device)

    t0 = time.monotonic()
    if second_mode is not None:
        with decoder_priors.decoder_prior_active(model, weight, mode):
            with decoder_priors.decoder_prior_active(model, second_weight, second_mode):
                fit = sc.train_blocked(
                    model, x_train32, model_seed, n_charts, max_epochs, epoch_block
                )
    else:
        with decoder_priors.decoder_prior_active(model, weight, mode):
            fit = sc.train_blocked(model, x_train32, model_seed, n_charts, max_epochs, epoch_block)
    train_wallclock_s = time.monotonic() - t0

    model.eval().double()
    c2_activation = decoder_curvature.assert_c2_decoder(model)

    # Non-finite guard (research assumption A3): never read a diverged cell as evidence.
    nonfinite_where: List[str] = [
        name for name, p in model.named_parameters() if not torch.isfinite(p).all()
    ]
    nonfinite_guard = {"tripped": len(nonfinite_where) > 0, "where": nonfinite_where}

    with torch.no_grad():
        recon_holdout = model.reconstruct(x_holdout64)
    recon_stats = cae.reconstruction_stats(x_holdout64, recon_holdout)

    t1 = time.monotonic()
    field = chart_curvature.chart_curvature_field(model, x_eval64, mode="reverse")
    curv_wallclock_s = time.monotonic() - t1

    H_est = field["H_vec"].detach().cpu().numpy().astype(np.float64)
    h_est_norm = field["H_norm"].detach().cpu().numpy().astype(np.float64)
    cond = field["metric_condition_number"].detach().cpu().numpy().astype(np.float64)
    lambda_min = field["lambda_min"].detach().cpu().numpy().astype(np.float64)
    lambda_max = field["lambda_max"].detach().cpu().numpy().astype(np.float64)
    det_g = field["det_g"].detach().cpu().numpy().astype(np.float64)
    log10_det_g = field["log10_det_g"].detach().cpu().numpy().astype(np.float64)
    assignment = field["chart_assignment"].detach().cpu().numpy()
    distinct, counts = np.unique(assignment, return_counts=True)

    axes = sc._fidelity_axes(H_est, H_true_eval)

    # Same flagging policy as the PU field: the within-config percentile and nothing else.
    flag_threshold = float(np.percentile(cond, pu.COND_FLAG_PERCENTILE))
    flagged_mask = cond > flag_threshold

    combination = second_mode is not None
    if combination:
        weights_by_mode = {mode: float(weight), second_mode: float(second_weight)}
        fractions_by_mode = {mode: weight_fraction, second_mode: second_weight_fraction}
        config_id = (
            f"ladder_combination_scale{weights_by_mode['scale']:.6e}_"
            f"christoffel{weights_by_mode['christoffel']:.6e}_s{model_seed}_e{max_epochs}_n{eval_n}"
        )
    else:
        mode_segment = "any" if float(weight) == 0.0 else mode
        config_id = f"ladder_{mode_segment}_w{weight:.6e}_s{model_seed}_e{max_epochs}_n{eval_n}"

    rec: Dict[str, Any] = {
        "kind": "ladder_cell",
        "config_id": config_id,
        "fixture": LADDER_FIXTURE,
        "n": n,
        "chart_dim": chart_dim,
        "ambient_dim": ambient,
        "embed_dim": embed_dim,
        "n_charts": n_charts,
        "max_epochs": max_epochs,
        "fixture_seed": LADDER_FIXTURE_SEED,
        "epochs_run": fit["epochs_run"],
        "early_stopped": fit["early_stopped"],
        "epoch_block": fit["epoch_block"],
        "n_blocks": fit["n_blocks"],
        "c2_activation": c2_activation,
        "fidelity": axes,
        "h_norm_est": pu._dist_summary(h_est_norm, pu.FIELD_HIST_BINS),
        "h_norm_true": pu._dist_summary(np.linalg.norm(H_true_eval, axis=-1), pu.FIELD_HIST_BINS),
        "cond": pu._dist_summary(cond, pu.FIELD_HIST_BINS),
        "flagged": {
            "percentile": pu.COND_FLAG_PERCENTILE,
            "threshold": flag_threshold,
            "count": int(flagged_mask.sum()),
            "fraction": float(flagged_mask.mean()),
        },
        "n_charts_used": int(field["n_charts_used"]),
        "per_chart_counts": {int(c): int(v) for c, v in zip(distinct.tolist(), counts.tolist())},
        "reconstruction": recon_stats,
        "train_wallclock_s": train_wallclock_s,
        "curv_wallclock_s": curv_wallclock_s,
        "device": str(device),
        "torch_version": torch.__version__,
        # --- this phase's new fields ---
        # Combination cells record mode='combination' rather than the outer mode string --
        # cell_key(rec['mode'], rec['weight'], rec['model_seed']) would otherwise collide with
        # the ordinary single-arm cell that happens to share the outer weight (best_scale is
        # always one of the ladder's own rungs), corrupting load_completed's bookkeeping. The
        # outer/inner modes and weights are fully recoverable from scale_weight/
        # christoffel_weight/second_mode below regardless.
        "mode": "combination" if combination else mode,
        "weight": float(weight),
        "weight_fraction": weight_fraction,
        "base_recon_at_init": base_recon_at_init,
        "base_penalty_at_init": base_penalty_at_init,
        "model_seed": int(model_seed),
        "eval_sample_size": int(eval_n),
        "eval_row_source": "pu._split holdout, first eval_n rows",
        "christoffel_max_rows_per_chart": int(christoffel_max_rows),
        "lambda_min": pu._dist_summary(lambda_min, pu.FIELD_HIST_BINS),
        "lambda_max": pu._dist_summary(lambda_max, pu.FIELD_HIST_BINS),
        "det_g": pu._dist_summary(det_g, pu.FIELD_HIST_BINS),
        "log10_det_g": pu._dist_summary(log10_det_g, pu.FIELD_HIST_BINS),
        "nonfinite_guard": nonfinite_guard,
        "relief_applied": list(relief_applied) if relief_applied else [],
        "arm_label": (
            COMBINATION_POST_HOC_LABEL
            if combination
            else ("baseline" if float(weight) == 0.0 else mode)
        ),
        # --- D-12 combination arm (03.1-05) ---
        "combination": combination,
        "combination_post_hoc": combination,
        "scale_weight": weights_by_mode.get("scale") if combination else None,
        "christoffel_weight": weights_by_mode.get("christoffel") if combination else None,
        "scale_weight_fraction": fractions_by_mode.get("scale") if combination else None,
        "christoffel_weight_fraction": fractions_by_mode.get("christoffel") if combination else None,
        "selection_rule": selection_rule if combination else None,
        "second_mode": second_mode,
        "second_weight": float(second_weight) if second_weight is not None else None,
        "second_weight_fraction": second_weight_fraction,
    }
    return rec


def print_cell_row(rec: Dict[str, Any]) -> None:
    if rec.get("nonfinite_guard", {}).get("tripped"):
        print("  DIVERGED -- not read as evidence")
    f = rec["fidelity"]
    print(
        f"\n--- {rec['config_id']}  (mode={rec['mode']} weight={rec['weight']:.6e} "
        f"seed={rec['model_seed']} eval_n={rec['eval_sample_size']})"
    )
    print(
        f"  epochs_run={rec['epochs_run']}/{rec['max_epochs']}  "
        f"early_stopped={rec['early_stopped']}  c2={rec['c2_activation']}"
    )
    if not f.get("applicable", True):
        print("  four axes: NOT APPLICABLE")
        print(f"    {f['reason']}")
    else:
        print("  four axes, never combined:")
        print(f"    direction   median cosine = {f['direction_median_cosine']:.6f}")
        print(f"    magnitude   median ratio  = {f['magnitude_median_ratio']:.6f}")
        if f.get("rank_calibration_applicable", True):
            print(
                f"    calibration slope={f['calibration_slope']:.6f} "
                f"R2={f['calibration_r2']:.6f}"
            )
            print(f"    rank        spearman rho = {f['rank_spearman_rho']:.6f}")
        else:
            print("    calibration/rank UNDEFINED (constant analytic field)")
            print(f"      {f['rank_calibration_note']}")
    print(
        f"  Tier 1 (D-15): lambda_min={rec['lambda_min']['median']:.6e}  "
        f"lambda_max={rec['lambda_max']['median']:.6e}  "
        f"log10_det_g={rec['log10_det_g']['median']:.6f}  "
        f"cond(g)={rec['cond']['median']:.6e}"
    )
    print(
        f"  held-out mse_per_dim={rec['reconstruction']['mse_per_dim']:.6e}  "
        f"charts_used={rec['n_charts_used']}"
    )
    print(
        f"  wall clock: train={rec['train_wallclock_s']:.1f}s  "
        f"curvature={rec['curv_wallclock_s']:.1f}s"
    )


# =============================================================================================
# --smoke: two cells at a deliberately tiny scale, proving the wiring end to end.
# =============================================================================================


def run_smoke(record_path: Path, device: torch.device) -> None:
    _banner(
        f"SMOKE -- decoder_prior_ladder_run.py "
        f"(n={SMOKE_N}, d={SMOKE_D}, D={SMOKE_AMBIENT}, {SMOKE_EPOCHS} epochs)"
    )
    smoke_cells = (
        ("scale", 0.0, 0.0),
        ("christoffel", 1e-3, 1.0),
    )
    for mode, weight, weight_fraction in smoke_cells:
        rec = run_cell(
            mode=mode,
            weight=weight,
            weight_fraction=weight_fraction,
            model_seed=LADDER_SEEDS[0],
            max_epochs=SMOKE_EPOCHS,
            n=SMOKE_N,
            chart_dim=SMOKE_D,
            ambient=SMOKE_AMBIENT,
            n_charts=SMOKE_N_CHARTS,
            embed_dim=2 * SMOKE_D,
            eval_n=SMOKE_EVAL_N,
            epoch_block=LADDER_EPOCH_BLOCK,
            device=device,
            relief_applied=[],
        )
        rec["kind"] = "smoke_ladder_cell"
        append_record(record_path, rec)
        print_cell_row(rec)
    print("SMOKE: exit 0.")


# =============================================================================================
# --anchor-check (03.1-03 Task 1): the faithfulness gate. == , never a numeric near-miss (D-13).
# =============================================================================================


def anchor_check(device: torch.device, record_path: Path) -> None:
    """Calls `synthetic_control_run.run_one_control` UNCHANGED -- not this runner's own
    `run_cell` -- because the point is to prove that this runner's IMPORTS of
    `synthetic_control_run`'s fixture, matched-architecture builder, blocked trainer and metric
    call sequence resolve to exactly the same computation the sealed row
    `control_saddle_nc4_d20_ep300` was produced by. `run_cell` deliberately is NOT the sealed
    code path -- it splits the fixture seed from the model seed and evaluates on a reduced
    sample, both deviations D-16 sanctions for the ladder but not for this proof.

    Not wrapped in `decoder_priors.decoder_prior_active` -- `run_one_control` builds its own
    model internally, so there is no model to wrap the training call of from here. The
    `weight == 0.0` no-op guarantee documented in `decoder_priors`'s module docstring is why the
    ladder's own `weight=0` cells are structurally the untouched `cae.train_cae` path (D-05),
    even though this anchor never exercises the wrapper itself.

    `run_one_control`'s own record does not carry the CURV-04 absolute-scale fields --
    `synthetic_control_run.py` is sealed and was written before D-15 existed, so it only reads
    `H_vec`/`H_norm`/`metric_condition_number` off the field dict it gets back. This is the one
    full-cloud (n=10000) evaluation this phase runs, and the sealed fit's absolute metric scale
    has to be on record for the first time (D-15), so this function temporarily wraps
    `chart_curvature.chart_curvature_field` -- the SAME module-level function object
    `synthetic_control_run.py` calls, since both modules import the same `pu_manifold`
    submodule -- to OBSERVE its return dict without changing what it computes or returns. The
    wrapper forwards every call through to the original function unchanged and is restored in a
    `finally` block, the same restore-on-exit shape `decoder_priors.decoder_prior_active` uses
    for `cae.chart_loss`. This never re-runs the curvature field a second time and never touches
    `synthetic_control_run.py`'s source.

    Costs roughly 2.6h (`03.1-03-PLAN.md`'s own compute note) -- the one full-cloud evaluation
    this phase runs.
    """
    print(
        "--anchor-check: sc.run_one_control('saddle', n=10000, chart_dim=20, ambient=768, "
        "n_charts=4, embed_dim=40, max_epochs=300, seed=CONTROL_FIXTURE_SEED, epoch_block=25) "
        "-- the sealed protocol, called through this runner's own imports, unchanged."
    )

    captured_field: Dict[str, Any] = {}
    original_chart_curvature_field = chart_curvature.chart_curvature_field

    def _capturing_chart_curvature_field(*args: Any, **kwargs: Any) -> Any:
        field = original_chart_curvature_field(*args, **kwargs)
        captured_field.update(field)
        return field

    chart_curvature.chart_curvature_field = _capturing_chart_curvature_field
    t0 = time.monotonic()
    try:
        rec = sc.run_one_control(
            "saddle",
            n=sc.CONTROL_N,
            chart_dim=pu.PU_CHART_DIM,
            ambient=pu.AMBIENT_DIM,
            n_charts=pu.CONVERGE_N_CHARTS,
            embed_dim=pu.PU_EMBED_DIM,
            max_epochs=300,
            seed=sc.CONTROL_FIXTURE_SEED,
            device=device,
            epoch_block=sc.CONTROL_EPOCH_BLOCK,
        )
    finally:
        chart_curvature.chart_curvature_field = original_chart_curvature_field
    wall_s = time.monotonic() - t0
    print(f"anchor_check total wall clock: {wall_s:.1f}s")

    measured_rho = rec["fidelity"]["rank_spearman_rho"]
    print(f"measured rank_spearman_rho = {measured_rho!r}")
    if measured_rho != ANCHOR_RHO_SPEARMAN:
        raise ValueError(
            f"--anchor-check: measured rank_spearman_rho={measured_rho!r} != expected "
            f"{ANCHOR_RHO_SPEARMAN!r}. This is a FAITHFULNESS FAILURE -- this runner's "
            "fixture, protocol or metric call sequence has drifted from "
            "synthetic_control_run.py, not a question of numeric rounding."
        )
    print(f"ANCHOR OK: rank_spearman_rho == {ANCHOR_RHO_SPEARMAN!r} exactly.")

    if rec["config_id"] != ANCHOR_CONFIG_ID:
        raise ValueError(
            f"--anchor-check: measured config_id={rec['config_id']!r} != expected "
            f"{ANCHOR_CONFIG_ID!r} -- rho matched but the row identity did not, so this is not "
            "the sealed row."
        )
    if rec["epochs_run"] != 300:
        raise ValueError(
            f"--anchor-check: measured epochs_run={rec['epochs_run']!r} != expected 300 -- rho "
            "matched but the training length did not, so this is not the sealed row."
        )
    if rec["n_charts_used"] != 4:
        raise ValueError(
            f"--anchor-check: measured n_charts_used={rec['n_charts_used']!r} != expected 4 -- "
            "rho matched but the chart count did not, so this is not the sealed row."
        )
    if rec["n"] != 10000:
        raise ValueError(
            f"--anchor-check: measured n={rec['n']!r} != expected 10000 -- rho matched but the "
            "row count did not, so this is not the sealed row."
        )
    print(
        f"config_id={rec['config_id']!r}  epochs_run={rec['epochs_run']}  "
        f"n_charts_used={rec['n_charts_used']}  n={rec['n']}  -- all match the sealed row."
    )

    train_ratio = rec["train_wallclock_s"] / ANCHOR_TRAIN_WALLCLOCK_S
    curv_ratio = rec["curv_wallclock_s"] / ANCHOR_CURV_WALLCLOCK_S
    print(
        f"machine-speed correction (train): measured train_wallclock_s="
        f"{rec['train_wallclock_s']!r} / sealed {ANCHOR_TRAIN_WALLCLOCK_S!r} = "
        f"{train_ratio:.6f}"
    )
    print(
        f"machine-speed correction (curvature): measured curv_wallclock_s="
        f"{rec['curv_wallclock_s']!r} / sealed {ANCHOR_CURV_WALLCLOCK_S!r} = {curv_ratio:.6f}"
    )

    if not captured_field:
        raise ValueError(
            "--anchor-check: the chart_curvature_field capture wrapper never observed a call -- "
            "run_one_control's internal call sequence has changed shape."
        )
    lambda_min_arr = captured_field["lambda_min"].detach().cpu().numpy().astype(np.float64)
    lambda_max_arr = captured_field["lambda_max"].detach().cpu().numpy().astype(np.float64)
    det_g_arr = captured_field["det_g"].detach().cpu().numpy().astype(np.float64)
    log10_det_g_arr = captured_field["log10_det_g"].detach().cpu().numpy().astype(np.float64)
    lambda_min_dist = pu._dist_summary(lambda_min_arr, pu.FIELD_HIST_BINS)
    lambda_max_dist = pu._dist_summary(lambda_max_arr, pu.FIELD_HIST_BINS)
    det_g_dist = pu._dist_summary(det_g_arr, pu.FIELD_HIST_BINS)
    log10_det_g_dist = pu._dist_summary(log10_det_g_arr, pu.FIELD_HIST_BINS)
    print(
        f"CURV-04 absolute scale, the sealed d=20 saddle fit, on record for the first time: "
        f"lambda_min median={lambda_min_dist['median']:.6e}  "
        f"lambda_max median={lambda_max_dist['median']:.6e}  "
        f"det_g median={det_g_dist['median']:.6e}  "
        f"log10_det_g median={log10_det_g_dist['median']:.6f}"
    )

    anchor_rec = dict(rec)
    anchor_rec["kind"] = "anchor_cell"
    anchor_rec["lambda_min"] = lambda_min_dist
    anchor_rec["lambda_max"] = lambda_max_dist
    anchor_rec["det_g"] = det_g_dist
    anchor_rec["log10_det_g"] = log10_det_g_dist
    anchor_rec["anchor_train_wallclock_s_sealed"] = ANCHOR_TRAIN_WALLCLOCK_S
    anchor_rec["anchor_curv_wallclock_s_sealed"] = ANCHOR_CURV_WALLCLOCK_S
    anchor_rec["anchor_train_wallclock_ratio"] = train_ratio
    anchor_rec["anchor_curv_wallclock_ratio"] = curv_ratio
    append_record(record_path, anchor_rec)
    print(f"anchor record appended to {record_path} with kind='anchor_cell'.")


# =============================================================================================
# --probe (03.1-03 Task 2): the two-sided cost probe -- training per mode, curvature evaluation
# per row -- and the pre-declared relief ladder (F1-F3 automatic, F4 an override of D-07 that
# fires only under --confirm-f4-override).
# =============================================================================================


def _read_anchor_curv_wallclock_s(record_path: Path) -> Optional[float]:
    """The anchor cell's own fresh full-cloud `curv_wallclock_s`, read back from the JSONL
    `kind == 'anchor_cell'` record (Task 1). `None` if no anchor record is on file -- the probe
    still runs; only the cross-check against the fresh anchor number is skipped."""
    if not record_path.exists():
        return None
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "anchor_cell":
                return float(rec["curv_wallclock_s"])
    return None


def _read_anchor_n_charts_used(record_path: Path) -> Optional[int]:
    """The anchor cell's own `n_charts_used`, read back from the JSONL `kind == 'anchor_cell'`
    record (Task 1) -- the full 300-epoch protocol's chart count, for the Half B chart-count
    caveat below. `None` if no anchor record is on file."""
    if not record_path.exists():
        return None
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "anchor_cell":
                return int(rec["n_charts_used"])
    return None


def _read_probe_train_s_per_epoch(record_path: Path, probe_mode: str) -> Optional[float]:
    """The most recent `kind == 'probe', probe_type == 'train'` record's `s_per_epoch` for the
    given `probe_mode` label (`'scale'` or `'christoffel'`), read back from the JSONL --
    `--probe-skip-train`'s reuse path, so the costly Half A training cells are not re-run just to
    fix a Half B measurement defect."""
    if not record_path.exists():
        return None
    found: Optional[float] = None
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if (
                rec.get("kind") == "probe"
                and rec.get("probe_type") == "train"
                and rec.get("probe_mode") == probe_mode
            ):
                found = float(rec["s_per_epoch"])
    return found


def project_total_s(
    max_epochs: int,
    fractions: Sequence[float],
    eval_n: int,
    s_per_epoch_by_mode: Dict[str, float],
    curv_s_per_row: float,
    christoffel_max_rows: int,
    scale_epochs: Optional[int] = None,
) -> Tuple[float, int]:
    """Projected total wall clock (s) for the full cell set, built from measured numbers only.

    3 baseline cells at the `weight=0` rate, `len(fractions) x 3` cells per mode at that mode's
    rate, 3 combination cells at `s_scale + s_christoffel - s_baseline` (the two penalties are
    computed additively inside one patched `chart_loss`, so their costs add and the shared base
    is counted once), plus `n_cells * eval_n * curv_s_per_row` for evaluation.

    `christoffel_max_rows` rescales the MEASURED christoffel rate (taken at
    `CHRISTOFFEL_MAX_ROWS_PER_CHART == 8`, the probe's own measurement) by holding the shared
    baseline-forward-pass cost fixed and scaling only the second-order surcharge linearly with
    `christoffel_max_rows / CHRISTOFFEL_MAX_ROWS_PER_CHART` -- `christoffel_penalty`'s cost is
    bounded by `n_charts_used x max_rows_per_chart` (D-07's own accounting), so a row-count
    change moves the surcharge proportionally. F3 (christoffel arm only) is expressed through
    this argument; F1/F2 are expressed through `fractions`/`eval_n`.

    `scale_epochs`, when given, decouples the `scale` arm's epoch count from every other arm's
    `max_epochs` -- the one knob F4 turns. `None` (the default) means every arm trains at
    `max_epochs`, which is every call site except F4's own projection.

    Returns `(projected_total_s, n_cells)`.
    """
    if scale_epochs is None:
        scale_epochs = max_epochs

    t0 = s_per_epoch_by_mode["weight0"]
    t_scale = s_per_epoch_by_mode["scale"]
    t_christoffel_measured = s_per_epoch_by_mode["christoffel"]
    row_ratio = float(christoffel_max_rows) / float(CHRISTOFFEL_MAX_ROWS_PER_CHART)
    t_christoffel = t0 + (t_christoffel_measured - t0) * row_ratio

    n_baseline = 3
    n_per_mode = len(fractions) * 3
    n_combo = 3
    n_cells = n_baseline + 2 * n_per_mode + n_combo

    train_s = (
        n_baseline * max_epochs * t0
        + n_per_mode * scale_epochs * t_scale
        + n_per_mode * max_epochs * t_christoffel
        + n_combo * max_epochs * (t_scale + t_christoffel - t0)
    )
    eval_s = n_cells * eval_n * curv_s_per_row
    return train_s + eval_s, n_cells


def print_projection_table(rows: Sequence[Tuple[int, float, int]]) -> None:
    """`rows`: `(max_epochs, projected_total_s, n_cells)` triples, one per candidate."""
    for max_e, total_s, n_cells in rows:
        print(
            f"  max_epochs={max_e:<4d}  n_cells={n_cells:<3d}  "
            f"projected_total_s={total_s:.1f}  ({total_s / 3600.0:.2f}h)"
        )


def apply_relief_ladder(
    max_epochs_candidates: Sequence[int],
    budget_s: float,
    fractions: Sequence[float],
    eval_n: int,
    s_per_epoch_by_mode: Dict[str, float],
    curv_s_per_row: float,
    christoffel_max_rows: int,
    confirm_f4_override: bool = False,
) -> Tuple[Optional[int], List[str], Dict[str, Any], Optional[float]]:
    """Walks `RELIEF_LADDER` (F1 -> F2 -> F3) in order, re-projecting across
    `max_epochs_candidates` after every step, and stops at the first configuration whose
    projection is at or under `budget_s`. Never calls `sys.exit` -- the caller decides what to do
    with a `None` `chosen_max_epochs` (F1-F3, and F4 if confirmed, all failed to fit).

    F4 is NOT a fourth entry walked automatically. If F1-F3 are exhausted without fitting, this
    function computes the F4 configuration -- the largest `scale`-arm epoch count that fits the
    budget at the smallest declared `max_epochs_candidates` value, holding every other arm at
    that same smallest candidate -- and prints it under `F4 AVAILABLE BUT NOT APPLIED -- OVERRIDES
    D-07` together with both halves of its cost. It only APPLIES that configuration (and appends
    `F4_OVERRIDE_LABEL` to `relief_applied`) when `confirm_f4_override=True`.

    Returns `(chosen_max_epochs, relief_applied, final_state, chosen_total_s)`.
    """
    state: Dict[str, Any] = {
        "fractions": tuple(fractions),
        "eval_n": eval_n,
        "christoffel_max_rows": christoffel_max_rows,
    }
    relief_applied: List[str] = []

    def _project_all(s: Dict[str, Any]) -> List[Tuple[int, float, int]]:
        rows = []
        for max_e in max_epochs_candidates:
            total_s, n_cells = project_total_s(
                max_e,
                s["fractions"],
                s["eval_n"],
                s_per_epoch_by_mode,
                curv_s_per_row,
                s["christoffel_max_rows"],
            )
            rows.append((max_e, total_s, n_cells))
        return rows

    def _pick(rows: List[Tuple[int, float, int]]) -> Tuple[Optional[int], Optional[float]]:
        chosen, chosen_total = None, None
        for max_e, total_s, n_cells in rows:
            if total_s <= budget_s and (chosen is None or max_e > chosen):
                chosen, chosen_total = max_e, total_s
        return chosen, chosen_total

    print("\nProjection at declared sizing (no relief valve applied):")
    rows = _project_all(state)
    print_projection_table(rows)
    chosen, chosen_total = _pick(rows)
    if chosen is not None:
        return chosen, relief_applied, state, chosen_total

    for valve in RELIEF_LADDER:
        if valve["id"] == "F1":
            state["fractions"] = (fractions[0], fractions[-1])
        elif valve["id"] == "F2":
            state["eval_n"] = 1000
        elif valve["id"] == "F3":
            state["christoffel_max_rows"] = 4
        relief_applied.append(valve["id"])
        print(f"\n{valve['id']} applied: {valve['description']}")
        print(f"  consequence: {valve['consequence']}")
        rows = _project_all(state)
        print_projection_table(rows)
        chosen, chosen_total = _pick(rows)
        if chosen is not None:
            return chosen, relief_applied, state, chosen_total

    # F1-F3 exhausted without fitting. Compute the F4 configuration anyway, print it, and apply
    # it ONLY under the explicit override -- never automatically (D-07, D-11).
    f4_base_epochs = min(max_epochs_candidates)
    floor_total, _ = project_total_s(
        f4_base_epochs,
        state["fractions"],
        state["eval_n"],
        s_per_epoch_by_mode,
        curv_s_per_row,
        state["christoffel_max_rows"],
        scale_epochs=0,
    )
    t_scale = s_per_epoch_by_mode["scale"]
    n_per_mode = len(state["fractions"]) * 3
    remaining_budget = budget_s - floor_total
    if remaining_budget <= 0 or t_scale <= 0 or n_per_mode == 0:
        f4_scale_epochs = 0
    else:
        f4_scale_epochs = int(remaining_budget // (n_per_mode * t_scale))
    f4_scale_epochs = max(0, min(f4_scale_epochs, f4_base_epochs))
    f4_total, _ = project_total_s(
        f4_base_epochs,
        state["fractions"],
        state["eval_n"],
        s_per_epoch_by_mode,
        curv_s_per_row,
        state["christoffel_max_rows"],
        scale_epochs=f4_scale_epochs,
    )

    print("\nF4 AVAILABLE BUT NOT APPLIED -- OVERRIDES D-07")
    print(
        f"  F4 would cut the scale arm's epoch count to {f4_scale_epochs} (from "
        f"{f4_base_epochs}), holding the christoffel arm, baseline and combination cells at "
        f"{f4_base_epochs} epochs. projected_total_s={f4_total:.1f} "
        f"({f4_total / 3600.0:.2f}h) against LADDER_BUDGET_S={budget_s} "
        f"({budget_s / 3600.0:.2f}h)."
    )
    print(
        "  Cost, both halves: (1) the D-07 under-training confound -- 03-08 found training to "
        "the full budget instead of stopping early cut held-out mse_per_dim 62.2% while leaving "
        "cond(g) unmoved, so this cut costs reconstruction (the axis D-09's bias guard reads) "
        "and buys nothing on the metric this phase exists to move; (2) arm-to-arm epoch "
        "incomparability -- the scale and christoffel arms would no longer share a training "
        "protocol."
    )

    if not confirm_f4_override:
        return None, relief_applied, state, None

    relief_applied.append(F4_OVERRIDE_LABEL)
    state["scale_epochs_f4"] = f4_scale_epochs
    if f4_total <= budget_s:
        return f4_base_epochs, relief_applied, state, f4_total
    return None, relief_applied, state, None


def probe(
    device: torch.device,
    record_path: Path,
    confirm_f4_override: bool = False,
    probe_epochs: int = PROBE_EPOCHS,
    probe_eval_sizes: Sequence[int] = PROBE_EVAL_SIZES,
    probe_skip_train: bool = False,
) -> None:
    """The two-sided cost probe (03.1-03 Task 2). Both halves are measured at the real ladder
    architecture (`n=10000, chart_dim=20, ambient=768, n_charts=4, embed_dim=40`) -- never on the
    Swiss roll, and never on the halted spike's own `chart_dim=2` per-epoch figures.

    `probe_skip_train`, when true, does not retrain Half A's costly `scale` and `christoffel`
    cells; it reuses their `s_per_epoch` from the most recent prior `--probe` run's JSONL record
    (raising if none exists). The `weight=0` cell is always trained fresh regardless -- its model
    is what Half B evaluates on. This is the defect-fix re-run path: Half B's original
    measurement silently truncated `eval_size=4000` to the 2000-row `pu._split` holdout, and
    fixing that requires only Half B to re-run, not the far costlier Half A."""
    _banner(
        f"--probe: two-sided cost probe at the real architecture "
        f"(n={LADDER_N}, chart_dim={LADDER_CHART_DIM}, ambient={LADDER_AMBIENT}, "
        f"n_charts={LADDER_N_CHARTS}, embed_dim={LADDER_EMBED_DIM})"
    )

    fx = sc.build_fixture(
        LADDER_FIXTURE, n=LADDER_N, d=LADDER_CHART_DIM, D=LADDER_AMBIENT, seed=LADDER_FIXTURE_SEED
    )
    X = np.asarray(fx["X"], dtype=np.float64)
    x32 = torch.tensor(X, dtype=torch.float32).to(device)
    x64 = torch.tensor(X, dtype=torch.float64).to(device)
    train_idx, holdout_idx = pu._split(X.shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long, device=device)]

    probe_seed = LADDER_SEEDS[0]

    # --- Half A: training cost per mode, at the largest declared fraction (the worst case) -----
    torch.manual_seed(probe_seed)
    calib_model = sc.build_matched_cae(
        LADDER_N_CHARTS, LADDER_CHART_DIM, LADDER_EMBED_DIM, LADDER_AMBIENT, device
    )
    calibrated = calibrate_weights(calib_model, x_train32[: pu.BATCH], fractions=[1.0])

    print(
        f"\nHalf A -- training cost per mode, PROBE_EPOCHS={probe_epochs}, model_seed={probe_seed}"
        f"{'  (probe_skip_train=True -- scale/christoffel REUSED, not retrained)' if probe_skip_train else ''}:"
    )
    probe_cells = (
        ("weight=0", "weight0", 0.0, "scale"),
        ("scale", "scale", calibrated[("scale", 1.0)]["weight"], "scale"),
        ("christoffel", "christoffel", calibrated[("christoffel", 1.0)]["weight"], "christoffel"),
    )
    s_per_epoch_by_mode: Dict[str, float] = {}
    weight0_model = None
    for print_label, dict_key, weight, mode in probe_cells:
        if probe_skip_train and print_label != "weight=0":
            reused_s_per_epoch = _read_probe_train_s_per_epoch(record_path, print_label)
            if reused_s_per_epoch is None:
                raise RuntimeError(
                    f"--probe-skip-train requested reuse of Half A's {print_label!r} s_per_epoch "
                    f"from {record_path}, but no prior kind=='probe', probe_type=='train', "
                    f"probe_mode=={print_label!r} record exists on disk. Run --probe once "
                    "without --probe-skip-train first, or omit --probe-skip-train."
                )
            s_per_epoch_by_mode[dict_key] = reused_s_per_epoch
            print(
                f"  {print_label:<12s} weight={weight:.6e}  REUSED s_per_epoch="
                f"{reused_s_per_epoch:.4f}  (not retrained -- see probe_skip_train docstring)"
            )
            append_record(
                record_path,
                {
                    "kind": "probe",
                    "probe_type": "train",
                    "probe_mode": print_label,
                    "probe_max_epochs": probe_epochs,
                    "weight": float(weight),
                    "model_seed": int(probe_seed),
                    "s_per_epoch": reused_s_per_epoch,
                    "reused": True,
                    "reused_from": "prior --probe run's JSONL record",
                },
            )
            continue

        torch.manual_seed(probe_seed)
        model = sc.build_matched_cae(
            LADDER_N_CHARTS, LADDER_CHART_DIM, LADDER_EMBED_DIM, LADDER_AMBIENT, device
        )
        t0 = time.monotonic()
        with decoder_priors.decoder_prior_active(model, weight, mode):
            fit = sc.train_blocked(
                model, x_train32, probe_seed, LADDER_N_CHARTS, probe_epochs, LADDER_EPOCH_BLOCK
            )
        train_wallclock_s = time.monotonic() - t0
        s_per_epoch = train_wallclock_s / fit["epochs_run"]
        s_per_epoch_by_mode[dict_key] = s_per_epoch
        print(
            f"  {print_label:<12s} weight={weight:.6e}  epochs_run={fit['epochs_run']}  "
            f"train_wallclock_s={train_wallclock_s:.3f}  s_per_epoch={s_per_epoch:.4f}"
        )
        append_record(
            record_path,
            {
                "kind": "probe",
                "probe_type": "train",
                "probe_mode": print_label,
                "probe_max_epochs": probe_epochs,
                "weight": float(weight),
                "model_seed": int(probe_seed),
                "epochs_run": fit["epochs_run"],
                "train_wallclock_s": train_wallclock_s,
                "s_per_epoch": s_per_epoch,
            },
        )
        if dict_key == "weight0":
            weight0_model = model

    mult_scale = s_per_epoch_by_mode["scale"] / s_per_epoch_by_mode["weight0"]
    mult_christoffel = s_per_epoch_by_mode["christoffel"] / s_per_epoch_by_mode["weight0"]
    print(
        f"  per-mode multiplier vs weight=0: scale={mult_scale:.4f}x  "
        f"christoffel={mult_christoffel:.4f}x"
    )

    # --- Half B: curvature evaluation cost per row, on the weight=0 probe model only -----------
    print("\nHalf B -- curvature evaluation cost per row, on the weight=0 probe model:")
    assert weight0_model is not None
    weight0_model.eval().double()
    x_eval_base64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long, device=device)]
    holdout_n = int(x_eval_base64.shape[0])
    print(
        f"  pu._split holdout size = {holdout_n} rows "
        f"(PU_HOLDOUT_FRACTION={pu.PU_HOLDOUT_FRACTION!r}, n={LADDER_N})"
    )

    curv_s_per_row_list: List[float] = []
    n_charts_used_probe: Optional[int] = None
    for eval_size in probe_eval_sizes:
        if eval_size > holdout_n:
            raise ValueError(
                f"probe_eval_sizes requested eval_size={eval_size} rows but the pu._split "
                f"holdout has only {holdout_n} rows (PU_HOLDOUT_FRACTION="
                f"{pu.PU_HOLDOUT_FRACTION!r} on n={LADDER_N}) -- refusing to silently truncate "
                f"x_eval_base64[:{eval_size}] into a shorter, duplicate measurement. Choose "
                f"PROBE_EVAL_SIZES values <= {holdout_n}."
            )
        x_eval64 = x_eval_base64[:eval_size]
        t1 = time.monotonic()
        field = chart_curvature.chart_curvature_field(weight0_model, x_eval64, mode="reverse")
        curv_s_at_eval_size = time.monotonic() - t1
        curv_s_per_row = curv_s_at_eval_size / eval_size
        curv_s_per_row_list.append(curv_s_per_row)
        n_charts_used_probe = int(field["n_charts_used"])
        print(
            f"  eval_size={eval_size:<5d}  curv_s_at_eval_size={curv_s_at_eval_size:.3f}  "
            f"curv_s_per_row={curv_s_per_row:.6f}  n_charts_used={n_charts_used_probe}"
        )
        append_record(
            record_path,
            {
                "kind": "probe",
                "probe_type": "eval",
                "eval_size": int(eval_size),
                "curv_s_at_eval_size": curv_s_at_eval_size,
                "curv_s_per_row": curv_s_per_row,
                "n_charts_used": n_charts_used_probe,
            },
        )

    # research assumption A2 -- is evaluation cost linear in row count? Report the true spread,
    # then ALWAYS project from the largest measured per-row cost (conservative direction),
    # regardless of the verdict.
    curv_s_per_row_max = max(curv_s_per_row_list)
    curv_s_per_row_min = min(curv_s_per_row_list)
    ratio_span = curv_s_per_row_max / curv_s_per_row_min
    curv_s_per_row_for_projection = curv_s_per_row_max
    if ratio_span > 2.0:
        print(
            f"\nEVAL COST IS NOT LINEAR IN ROW COUNT -- curv_s_per_row varies {ratio_span:.2f}x "
            f"across {list(probe_eval_sizes)}: {curv_s_per_row_list}"
        )
    else:
        print(
            f"\nEVAL COST IS APPROXIMATELY LINEAR IN ROW COUNT (A2 CONFIRMED) -- curv_s_per_row "
            f"varies only {ratio_span:.2f}x across {list(probe_eval_sizes)}: {curv_s_per_row_list}"
        )
    print(
        f"  projecting from the largest measured per-row cost, the conservative direction: "
        f"{curv_s_per_row_for_projection:.6f}"
    )

    extrapolated_n10000 = curv_s_per_row_for_projection * LADDER_N
    anchor_curv_s = _read_anchor_curv_wallclock_s(record_path)
    print(
        f"\n  extrapolated n={LADDER_N} (from the largest measured curv_s_per_row): "
        f"{extrapolated_n10000:.1f}s"
    )
    print(f"  sealed full-cloud curv_wallclock_s = {SEALED_CURV_WALLCLOCK_S!r}")
    rel_vs_sealed = abs(extrapolated_n10000 - SEALED_CURV_WALLCLOCK_S) / SEALED_CURV_WALLCLOCK_S
    print(f"  relative disagreement vs sealed = {rel_vs_sealed:.4f}")
    if anchor_curv_s is not None:
        print(f"  anchor cell's fresh full-cloud curv_wallclock_s = {anchor_curv_s!r}")
        rel_vs_anchor = abs(extrapolated_n10000 - anchor_curv_s) / anchor_curv_s
        print(f"  relative disagreement vs anchor = {rel_vs_anchor:.4f}")
    else:
        rel_vs_anchor = None
        print(
            "  no anchor_cell record found on disk -- run --anchor-check first for the full "
            "cross-check."
        )

    # n_charts_used caveat: every probe eval trains a 3-epoch model, which populates fewer charts
    # than the anchor's full 300-epoch fit. Stated explicitly, never folded silently into the
    # projection.
    anchor_n_charts = _read_anchor_n_charts_used(record_path)
    chart_count_ratio: Optional[float] = None
    if anchor_n_charts is not None and n_charts_used_probe:
        chart_count_ratio = anchor_n_charts / n_charts_used_probe
        print(
            f"\n  CAVEAT -- n_charts_used: every probe eval populated n_charts_used="
            f"{n_charts_used_probe} chart(s) (the {probe_epochs}-epoch probe model), vs the "
            f"anchor cell's full 300-epoch n_charts_used={anchor_n_charts}. If curvature cost "
            f"tracks chart count, real ladder cells cost roughly {chart_count_ratio:.4f}x more "
            f"per row than this probe measured. NOT folded into the projection below -- stated "
            f"as an explicit caveat, carried into the JSONL summary record and the SUMMARY."
        )
    else:
        print(
            "\n  CAVEAT -- n_charts_used: no anchor_cell record found on disk to compare "
            "against; the chart-count ratio cannot be computed."
        )

    # --- The projection and the relief ladder ---------------------------------------------------
    print("\nProjection and relief ladder:")
    chosen, relief_applied, final_state, chosen_total = apply_relief_ladder(
        LADDER_EPOCH_CANDIDATES,
        LADDER_BUDGET_S,
        LADDER_FRACTIONS,
        LADDER_EVAL_N,
        s_per_epoch_by_mode,
        curv_s_per_row_for_projection,
        CHRISTOFFEL_MAX_ROWS_PER_CHART,
        confirm_f4_override=confirm_f4_override,
    )

    caveat_total_s: Optional[float] = None
    if chosen is not None and chart_count_ratio is not None:
        caveat_curv_s_per_row = curv_s_per_row_for_projection * chart_count_ratio
        caveat_total_s, _ = project_total_s(
            chosen,
            final_state["fractions"],
            final_state["eval_n"],
            s_per_epoch_by_mode,
            caveat_curv_s_per_row,
            final_state["christoffel_max_rows"],
        )
        print(
            f"\n  CAVEAT-ADJUSTED projected_total_s at chosen_max_epochs={chosen}, with the "
            f"n_charts_used {chart_count_ratio:.4f}x factor applied to curvature-evaluation cost "
            f"only: {caveat_total_s:.1f}s ({caveat_total_s / 3600.0:.2f}h) against "
            f"LADDER_BUDGET_S={LADDER_BUDGET_S} ({LADDER_BUDGET_S / 3600.0:.2f}h)."
        )

    summary_record: Dict[str, Any] = {
        "kind": "probe",
        "probe_type": "summary",
        "probe_skip_train": probe_skip_train,
        "s_per_epoch_by_mode": dict(s_per_epoch_by_mode),
        "curv_s_per_row": curv_s_per_row_for_projection,
        "curv_s_per_row_by_eval_size": dict(zip(probe_eval_sizes, curv_s_per_row_list)),
        "curv_s_per_row_ratio_span": ratio_span,
        "eval_cost_linear": ratio_span <= 2.0,
        "rel_vs_sealed": rel_vs_sealed,
        "rel_vs_anchor": rel_vs_anchor,
        "n_charts_used": n_charts_used_probe,
        "n_charts_used_probe": n_charts_used_probe,
        "n_charts_used_anchor": anchor_n_charts,
        "chart_count_ratio": chart_count_ratio,
        "caveat_adjusted_projected_total_s": caveat_total_s,
        "relief_applied": list(relief_applied),
        "final_fractions": list(final_state["fractions"]),
        "final_eval_n": final_state["eval_n"],
        "final_christoffel_max_rows": final_state["christoffel_max_rows"],
        "projected_total_s": chosen_total,
        "chosen_max_epochs": chosen,
    }
    append_record(record_path, summary_record)

    if chosen is None:
        print("\nBUDGET NOT MET.")
        sys.exit(1)

    print(
        f"\nchosen_max_epochs = {chosen}  relief_applied={relief_applied}  "
        f"projected_total_s={chosen_total:.1f}"
    )


# =============================================================================================
# D-10 / D-16: the ladder sizing, the F1-F4 relief ladder and the D-08 Tier-1/Tier-2 reading
# rules are declared in source, before any number exists, and printed by --dry-run -- the same
# idiom as swiss_roll_isometry_prior_sweep_run.py's "the ladder is declared in source" block.
# =============================================================================================

SEALED_CURV_WALLCLOCK_S = 2993.912286339997
"""control_saddle_nc4_d20_ep300: ONE full-cloud (n=10000) chart_curvature_field call at
chart_dim=20 / ambient=768 / n_charts_used=4 -- mode- and weight-independent, because
curvature evaluation runs after training."""

SEALED_TRAIN_WALLCLOCK_S = 6253.836718825012
"""control_saddle_nc4_d20_ep300 at max_epochs=300 -> ~20.85 s/epoch, corroborated by the
Phase 3 nine-cell PU grid (nc4_seed20260813: epochs_run=30, train_wallclock_s=646.6931414349965
-> ~21.56 s/epoch, same architecture)."""

READING_RULES_TEXT = """\
D-08 TIER 1 -- MECHANISM. The exact wording of "moves toward healthy".

Read per mode over the rungs ordered by ascending weight, on the 3-seed MEDIAN of each
per-cell distribution statistic, with the full 3-seed SPREAD always printed beside it (D-03).

  lambda_min moves toward healthy = median lambda_min is non-decreasing across ascending
    weight rungs AND is strictly greater at the top rung than at weight=0.
  lambda_max moves toward healthy = median lambda_max does not fall toward the collapse
    band. A lambda_max that DROPS while cond(g) improves is the collapse signature and is
    read as the prior making the metric worse, never better.
  det(g) moves toward healthy = median log10_det_g is non-decreasing across ascending
    rungs and strictly greater at the top rung than at weight=0 -- i.e. the geometric-mean
    metric eigenvalue moves toward 1, exactly the quantity scale's penalty (log det g / d)^2
    minimizes.
  cond(g) moves toward healthy = median cond(g) is non-increasing across ascending rungs
    and strictly lower at the top rung than at weight=0. cond(g) ALONE IS NEVER SUFFICIENT.

Per-mode verdict, printed as one of exactly three strings:
  MECHANISM DEMONSTRATED -- the mode's own target field moved as above. scale's target
    fields are log10_det_g and lambda_min; christoffel's target field is cond(g), with
    lambda_max not falling.
  COLLAPSE -- median cond(g) improved while median lambda_max fell by at least one decade
    from the weight=0 cell. Scored as the prior making the metric worse.
  MECHANISM NOT DEMONSTRATED -- neither of the above.

Refusal rule: if a mode's Tier 1 is MECHANISM NOT DEMONSTRATED or COLLAPSE, its Tier 2 table
is still printed in full (D-10 forbids hiding numbers) but is labelled "not read as evidence
about the estimand" -- a prior that did not move its own target quantity was never actually
tested.

D-08 TIER 2 -- ESTIMAND. The exact wording of "moves toward 1".

Four axes, scored separately, never collapsed into one score, and no arithmetic anywhere
combines them. Per axis, over the rungs ordered by ascending weight, on the 3-seed median
with the full spread printed. The weight=0 reference values on record for this fixture at
the sealed 300-epoch protocol are cosine -0.000478, magnitude ratio 9955, R2 0.000002, rho
-0.0151 -- quoted as context only; the actual comparison is against this phase's own
re-fitted weight=0 cells at the same epoch budget (D-05).

  direction  -- direction_median_cosine moves toward +1: |cosine - 1| strictly smaller at
    the mode's best rung than at weight=0.
  magnitude  -- magnitude_median_ratio moves toward 1: |log10(ratio)| strictly smaller at
    the best rung than at weight=0. Log scale because the baseline sits four decades off; a
    linear distance would be dominated by that one number.
  calibration -- calibration_r2 moves toward 1: strictly larger than at weight=0, AND
    calibration_slope moves toward 1. Both reported; neither alone.
  rank       -- rank_spearman_rho moves toward +1: strictly larger than at weight=0.

No bar, no verdict artifact (D-10). These are reading rules for a diagnostics table, not a
gate. A null -- every axis unmoved within the 3-seed spread -- is a complete answer and is
reported as plainly as a success (D-11).

Seed consistency (D-03): three draws cannot separate a small real effect from seed noise.
Any Tier-1 or Tier-2 movement whose direction is not consistent across all three seeds is
labelled NOT SEED-CONSISTENT and is not claimed as an effect.

Bias guard (D-09): held-out reconstruction.mse_per_dim is printed beside every cell. Stated
rule: an improvement bought with materially worse reconstruction is not an improvement. No
numeric threshold -- the last reconstruction threshold proposed in this milestone was
deferred as circular and that objection is unchanged. The comparison is against this
phase's own weight=0 cells at the same three seeds, reported as a ratio, and the reader
makes the call. The bias guard is a reading rule, never a selector: no rule here selects on
reconstruction.
"""


# =============================================================================================
# 03.1-04 Task 1: the D-08 two-tier read-out -- Tier 1 (mechanism) then Tier 2 (estimand),
# proven against a synthetic record before any real ladder cell runs.
# =============================================================================================

HEALTHY_D4_COND = 4.098
"""03-FINDINGS.md §5: the healthy `d=4` saddle diagnostic's median `cond(g)` -- the reference
band the baseline-pathology precondition (`03.1-01-PLAN.md` §G) compares against."""

HEALTHY_LAMBDA_MAX = 3.352
"""03-FINDINGS.md §5: seed 20260813's median `lambda_max` -- the one PU fit whose metric had a
real, non-collapsed absolute scale."""

SEALED_HEALTHY_SEED_COND = 3.30e07
"""03-FINDINGS.md §5: seed 20260813's median `cond(g)`, recorded beside `HEALTHY_LAMBDA_MAX` so
the baseline-pathology check's reference numbers live in one place."""

COLLAPSE_LAMBDA_MAX = 5.4e-07
"""03-FINDINGS.md §5: seeds 20260814/20260815's median `lambda_max` -- the collapsed-seed
reference band."""

TIER1_VERDICTS: Tuple[str, str, str] = (
    "MECHANISM DEMONSTRATED",
    "COLLAPSE",
    "MECHANISM NOT DEMONSTRATED",
)
"""D-08 Tier 1's exactly-one-of-three verdict strings, spelled verbatim as `READING_RULES_TEXT`
declares them."""


def median_and_spread(values: Sequence[float]) -> Tuple[float, List[float]]:
    """D-03: every print shows both the 3-seed median AND the full sorted spread, never the
    median alone."""
    vals = sorted(float(v) for v in values)
    return float(statistics.median(vals)), vals


def seed_consistent(per_seed_deltas: Sequence[float]) -> bool:
    """D-03: True iff every non-zero delta shares the same sign. A delta of exactly zero
    contradicts no direction, so it never breaks consistency on its own; an empty input is
    vacuously consistent (nothing to disagree)."""
    signs = {(1 if d > 0 else -1 if d < 0 else 0) for d in per_seed_deltas}
    return len(signs - {0}) <= 1


def rows_for_mode(
    records: Sequence[Dict[str, Any]], mode: str
) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """This mode's own cells plus the shared `weight=0.0` baseline cells (matched by
    `cell_key`'s `"any"` sentinel -- the weight check below, not the `mode` field, is what
    selects them, since a baseline cell's stored `mode` field is whatever literal mode string
    ran it first), grouped by weight and sorted ascending. Cells are expected pre-filtered to
    exclude `DIVERGED` cells by the caller."""
    mode_cells = [r for r in records if float(r["weight"]) == 0.0 or r["mode"] == mode]
    by_weight: Dict[float, List[Dict[str, Any]]] = {}
    for r in mode_cells:
        by_weight.setdefault(float(r["weight"]), []).append(r)
    return [
        (w, sorted(by_weight[w], key=lambda r: int(r["model_seed"]))) for w in sorted(by_weight)
    ]


def baseline_pathology_check(baseline_rows: Sequence[Dict[str, Any]]) -> bool:
    """`03.1-01-PLAN.md` §G, implemented exactly. Prints `BASELINE PATHOLOGY CHECK` with the
    four medians per seed against the three sealed reference bands, and returns `True` iff the
    ladder should proceed (pathology IS present in at least one baseline seed). When it returns
    `False`, the caller prints `NO PATHOLOGY AT THIS EPOCH BUDGET` and exits 2 before training
    any prior-active cell -- the pathology being absent at this epoch budget is itself the
    finding, and escalation is an explicit user decision (D-11)."""
    print("\nBASELINE PATHOLOGY CHECK")
    print(
        f"  reference bands (03-FINDINGS.md §5): healthy-metric seed 20260813 "
        f"lambda_max={HEALTHY_LAMBDA_MAX:.3f} cond={SEALED_HEALTHY_SEED_COND:.2e}; "
        f"collapsed seeds lambda_max~={COLLAPSE_LAMBDA_MAX:.1e}; "
        f"healthy d=4 saddle control cond={HEALTHY_D4_COND:.3f}"
    )
    if not baseline_rows:
        print("  no weight=0.0 baseline cell on record -- cannot evaluate; treating as pathological (proceed).")
        return True

    all_healthy = True
    for rec in sorted(baseline_rows, key=lambda r: int(r["model_seed"])):
        cond_med = float(rec["cond"]["median"])
        lmin_med = float(rec["lambda_min"]["median"])
        lmax_med = float(rec["lambda_max"]["median"])
        logdet_med = float(rec["log10_det_g"]["median"])
        cond_within_decade = abs(math.log10(max(cond_med, 1e-300)) - math.log10(HEALTHY_D4_COND)) <= 1.0
        lmax_within_decade = abs(math.log10(max(lmax_med, 1e-300))) <= 1.0  # within a decade of 1
        seed_healthy = cond_within_decade and lmax_within_decade
        all_healthy = all_healthy and seed_healthy
        print(
            f"  seed={rec['model_seed']}  cond_median={cond_med:.6e}  lambda_min_median={lmin_med:.6e}  "
            f"lambda_max_median={lmax_med:.6e}  log10_det_g_median={logdet_med:.6f}  "
            f"{'d=4-HEALTHY BAND' if seed_healthy else 'pathological (or ambiguous)'}"
        )

    if all_healthy:
        print("NO PATHOLOGY AT THIS EPOCH BUDGET")
        return False
    print("  pathology confirmed in at least one baseline seed at this epoch budget -- proceeding.")
    return True


def _cell_field_median(cell: Dict[str, Any], field: str) -> float:
    return float(cell[field]["median"])


def _per_seed_map(cells: Sequence[Dict[str, Any]], field: str) -> Dict[int, float]:
    return {int(c["model_seed"]): _cell_field_median(c, field) for c in cells}


def tier1_readout(rows: Sequence[Tuple[float, List[Dict[str, Any]]]], mode: str) -> str:
    """D-08 Tier 1 -- mechanism. `rows`: `rows_for_mode`'s output, weight ascending, `DIVERGED`
    cells already excluded by the caller. Prints the per-rung medians and full 3-seed spreads,
    the per-field toward-healthy judgement with `NOT SEED-CONSISTENT` labelling, and returns
    exactly one of `TIER1_VERDICTS`."""
    print(f"\n--- TIER 1 (mechanism) -- mode={mode} ---")
    fields = ("lambda_min", "lambda_max", "log10_det_g", "cond")
    weights = [w for w, _ in rows]
    per_field_medians: Dict[str, List[float]] = {f: [] for f in fields}
    per_field_seed_maps: Dict[str, List[Dict[int, float]]] = {f: [] for f in fields}
    for w, cells in rows:
        print(f"  weight={w:.6e}  n_cells={len(cells)}")
        for f in fields:
            vals = [_cell_field_median(c, f) for c in cells]
            med, spread = median_and_spread(vals)
            per_field_medians[f].append(med)
            per_field_seed_maps[f].append(_per_seed_map(cells, f))
            spread_str = ", ".join(f"{v:.6e}" for v in spread)
            print(f"    {f:<12s} median={med:.6e}  spread=[{spread_str}]")

    if len(weights) < 2:
        print("  fewer than two rungs on record -- verdict withheld until weight=0 and at "
              "least one prior-active rung both exist.")
        return "MECHANISM NOT DEMONSTRATED"

    w0_idx, top_idx = 0, len(weights) - 1

    def _monotone(vals: List[float], non_decreasing: bool) -> bool:
        if non_decreasing:
            return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def _seed_consistent_for(field: str) -> bool:
        seed_map_w0 = per_field_seed_maps[field][w0_idx]
        seed_map_top = per_field_seed_maps[field][top_idx]
        common = sorted(set(seed_map_w0) & set(seed_map_top))
        if not common:
            return True
        deltas = [seed_map_top[s] - seed_map_w0[s] for s in common]
        return seed_consistent(deltas)

    lambda_min_toward = (
        _monotone(per_field_medians["lambda_min"], non_decreasing=True)
        and per_field_medians["lambda_min"][top_idx] > per_field_medians["lambda_min"][w0_idx]
    )
    logdet_toward = (
        _monotone(per_field_medians["log10_det_g"], non_decreasing=True)
        and per_field_medians["log10_det_g"][top_idx] > per_field_medians["log10_det_g"][w0_idx]
    )
    cond_toward = (
        _monotone(per_field_medians["cond"], non_decreasing=False)
        and per_field_medians["cond"][top_idx] < per_field_medians["cond"][w0_idx]
    )
    lambda_max_w0 = per_field_medians["lambda_max"][w0_idx]
    lambda_max_top = per_field_medians["lambda_max"][top_idx]
    lambda_max_fell_a_decade = (
        lambda_max_w0 > 0 and lambda_max_top > 0 and (lambda_max_top / lambda_max_w0) <= 0.1
    )
    lambda_max_not_falling = not lambda_max_fell_a_decade
    cond_improved = per_field_medians["cond"][top_idx] < per_field_medians["cond"][w0_idx]

    for field, toward in (
        ("lambda_min", lambda_min_toward),
        ("log10_det_g", logdet_toward),
        ("cond(g)", cond_toward),
        ("lambda_max", lambda_max_not_falling),
    ):
        consistent = _seed_consistent_for(field if field != "cond(g)" else "cond")
        tag = "" if consistent else "  NOT SEED-CONSISTENT"
        print(f"    {field} moves toward healthy = {toward}{tag}")

    collapse = cond_improved and lambda_max_fell_a_decade
    if collapse:
        verdict = "COLLAPSE"
    elif mode == "scale" and lambda_min_toward and logdet_toward:
        verdict = "MECHANISM DEMONSTRATED"
    elif mode == "christoffel" and cond_toward and lambda_max_not_falling:
        verdict = "MECHANISM DEMONSTRATED"
    else:
        verdict = "MECHANISM NOT DEMONSTRATED"
    assert verdict in TIER1_VERDICTS

    print(f"  TIER 1 VERDICT ({mode}): {verdict}")
    if verdict != "MECHANISM DEMONSTRATED":
        print(
            "    the penalty did not demonstrably move its own target quantity -- no Tier-2 "
            "column below is read as evidence about the estimand for this mode."
        )
    return verdict


def _fidelity_extractor(key: str):
    def _extract(cell: Dict[str, Any]) -> Optional[float]:
        fid = cell.get("fidelity", {})
        if not fid.get("applicable", True):
            return None
        if key in ("calibration_r2", "calibration_slope", "rank_spearman_rho") and not fid.get(
            "rank_calibration_applicable", True
        ):
            return None
        val = fid.get(key)
        return None if val is None else float(val)

    return _extract


def _rung_field_values(
    cells: Sequence[Dict[str, Any]], extractor
) -> Tuple[List[float], Dict[int, float]]:
    values: List[float] = []
    seed_map: Dict[int, float] = {}
    for c in cells:
        v = extractor(c)
        if v is not None:
            values.append(v)
            seed_map[int(c["model_seed"])] = v
    return values, seed_map


def tier2_axis_readout(
    rows: Sequence[Tuple[float, List[Dict[str, Any]]]],
    axis_name: str,
    extractors: Dict[str, Any],
    target_label: str,
    distance_fn: Dict[str, Any],
    and_semantics: bool = False,
) -> str:
    """One Tier-2 axis block -- `direction`/`magnitude`/`rank` pass one `extractors` entry;
    `calibration` passes two (`slope`, `r2`), both required (`and_semantics=True`) since D-08
    reports both and neither alone. Prints median + full 3-seed spread per rung for every
    sub-metric, the D-03 seed-consistency label on the weight=0 -> best-rung delta, and one
    toward-target verdict for the whole axis. `distance_fn[sub](value) -> float`, smaller is
    closer to the axis's declared target -- never a combination across axes (D-08's own no-
    arithmetic-combines-them rule stays inside one axis's own sub-metrics only)."""
    print(f"\n  {axis_name}:")
    weights = [w for w, _ in rows]
    w0_idx = 0
    per_sub_series: Dict[str, List[Tuple[float, Optional[float], List[float], Dict[int, float]]]] = {}
    for sub, extractor in extractors.items():
        series: List[Tuple[float, Optional[float], List[float], Dict[int, float]]] = []
        for w, cells in rows:
            values, seed_map = _rung_field_values(cells, extractor)
            if values:
                med, spread = median_and_spread(values)
            else:
                med, spread = None, []
            series.append((w, med, spread, seed_map))
        per_sub_series[sub] = series
        print(f"    {sub}:")
        for w, med, spread, _ in series:
            if med is None:
                print(f"      weight={w:.6e}  UNDEFINED")
            else:
                spread_str = ", ".join(f"{v:.6f}" for v in spread)
                print(f"      weight={w:.6e}  median={med:.6f}  spread=[{spread_str}]")

    w0_vals = {sub: per_sub_series[sub][w0_idx][1] for sub in extractors}
    if any(v is None for v in w0_vals.values()):
        print(f"    {axis_name} verdict: UNDEFINED (weight=0 baseline undefined for at least one sub-metric)")
        return "UNDEFINED"

    best_idx: Optional[int] = None
    best_score: Optional[float] = None
    for idx in range(1, len(weights)):
        vals = {sub: per_sub_series[sub][idx][1] for sub in extractors}
        if any(v is None for v in vals.values()):
            continue
        score = max(distance_fn[sub](vals[sub]) for sub in extractors)
        if best_score is None or score < best_score:
            best_score, best_idx = score, idx

    if best_idx is None:
        print(f"    {axis_name} verdict: UNDEFINED (no rung has every required sub-metric defined)")
        return "UNDEFINED"

    moved_flags = []
    consistency_flags = []
    for sub in extractors:
        w0_d = distance_fn[sub](w0_vals[sub])
        best_val = per_sub_series[sub][best_idx][1]
        assert best_val is not None
        best_d = distance_fn[sub](best_val)
        moved_flags.append(best_d < w0_d)
        seed_map_w0 = per_sub_series[sub][w0_idx][3]
        seed_map_best = per_sub_series[sub][best_idx][3]
        common = sorted(set(seed_map_w0) & set(seed_map_best))
        if common:
            deltas = [
                distance_fn[sub](seed_map_w0[s]) - distance_fn[sub](seed_map_best[s]) for s in common
            ]
            consistency_flags.append(seed_consistent(deltas))
        else:
            consistency_flags.append(True)

    moved = all(moved_flags) if and_semantics else any(moved_flags)
    consistent = all(consistency_flags)
    tag = "" if consistent else "  NOT SEED-CONSISTENT"
    verdict_word = f"MOVES TOWARD {target_label}" if moved else f"DOES NOT MOVE toward {target_label}"
    verdict = f"{verdict_word} at weight={weights[best_idx]:.6e}{tag}"
    print(f"    {axis_name} verdict: {verdict}")
    return verdict


def tier2_readout(
    rows: Sequence[Tuple[float, List[Dict[str, Any]]]], mode: str, tier1_verdict: str
) -> None:
    """D-08 Tier 2 -- estimand. Four axes, printed as four separately labelled blocks, never
    collapsed into one score. `direction`/`magnitude`/`rank` each carry one sub-metric;
    `calibration` carries two (`slope` and `r2`), both required. Ends with the D-09 bias guard
    printed beside every rung -- a reading rule, never a selector."""
    print(f"\n--- TIER 2 (estimand) -- mode={mode} ---")
    if tier1_verdict != "MECHANISM DEMONSTRATED":
        print(
            "  not read as evidence about the estimand -- Tier 1 did not demonstrate the "
            "mechanism for this mode."
        )

    tier2_axis_readout(
        rows,
        "direction",
        {"cosine": _fidelity_extractor("direction_median_cosine")},
        "+1",
        {"cosine": lambda v: abs(v - 1.0)},
    )
    tier2_axis_readout(
        rows,
        "magnitude",
        {"ratio": _fidelity_extractor("magnitude_median_ratio")},
        "1",
        {"ratio": lambda v: abs(math.log10(max(v, 1e-300)))},
    )
    tier2_axis_readout(
        rows,
        "calibration",
        {
            "r2": _fidelity_extractor("calibration_r2"),
            "slope": _fidelity_extractor("calibration_slope"),
        },
        "1",
        {"r2": lambda v: 1.0 - v, "slope": lambda v: abs(v - 1.0)},
        and_semantics=True,
    )
    tier2_axis_readout(
        rows,
        "rank",
        {"rho": _fidelity_extractor("rank_spearman_rho")},
        "+1",
        {"rho": lambda v: 1.0 - v},
    )

    print(
        "\n  bias guard (D-09): held-out mse_per_dim beside every rung; an improvement bought "
        "with materially worse reconstruction is not an improvement. Reading rule only -- "
        "never a selector."
    )
    mse_w0: Optional[float] = None
    for w, cells in rows:
        vals = [float(c["reconstruction"]["mse_per_dim"]) for c in cells]
        med, spread = median_and_spread(vals)
        if mse_w0 is None:
            mse_w0 = med
        ratio_str = f"{med / mse_w0:.4f}" if mse_w0 else "n/a"
        spread_str = ", ".join(f"{v:.6e}" for v in spread)
        print(
            f"    weight={w:.6e}  mse_per_dim median={med:.6e}  spread=[{spread_str}]  "
            f"ratio_vs_weight0={ratio_str}"
        )


def print_summary(record_path: Path) -> None:
    """`--summary`: read the record back and print `READING_RULES_TEXT`, the epochs-uniformity
    guard, any `DIVERGED` cells, the D-07 relief banner, `BASELINE PATHOLOGY CHECK`, then
    Tier 1 and Tier 2 per mode, then the full per-cell table. Runs no new cells."""
    _banner("--summary: 03.1 decoder-prior ladder read-out")
    print(READING_RULES_TEXT)

    if not record_path.exists():
        print("No record file found at", record_path, "-- nothing to summarize.")
        return

    records: List[Dict[str, Any]] = []
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    cells = [r for r in records if r.get("kind") in ("ladder_cell", "smoke_ladder_cell")]
    if not cells:
        print("No ladder cells on record yet -- nothing to summarize.")
        return

    non_diverged = [c for c in cells if not c.get("nonfinite_guard", {}).get("tripped")]
    diverged = [c for c in cells if c.get("nonfinite_guard", {}).get("tripped")]

    bad = [
        c
        for c in non_diverged
        if int(c.get("epochs_run", -1)) != int(c.get("max_epochs", -2))
        or bool(c.get("early_stopped", True))
    ]
    if bad:
        print(
            "\nTRAINING-LENGTH CONFOUND -- the following non-diverged cells did not run for the "
            "full declared max_epochs, or early-stopped:"
        )
        for c in bad:
            print(
                f"  {c.get('config_id')}  epochs_run={c.get('epochs_run')}  "
                f"max_epochs={c.get('max_epochs')}  early_stopped={c.get('early_stopped')}"
            )
        print("refusing to print the read-outs below.")
        return
    print(
        "\nepochs_run uniformity: OK -- every non-diverged cell has epochs_run == max_epochs "
        "and early_stopped == False."
    )

    print(f"\nDIVERGED cells (excluded from every median and spread): {len(diverged)}")
    for c in diverged:
        print(
            f"  DIVERGED -- not read as evidence: {c.get('config_id')}  "
            f"where={c.get('nonfinite_guard', {}).get('where')}"
        )

    f4_cells = [c for c in non_diverged if F4_OVERRIDE_LABEL in c.get("relief_applied", [])]
    if f4_cells:
        print("\nD-07 OVERRIDE FIRED")
        print(f"  affected config_ids: {[c['config_id'] for c in f4_cells]}")
        print(
            "  These cells were trained on a cut epoch budget as a CONFIRMED OVERRIDE of D-07's "
            "fence against cutting epochs. Under-training is the confound D-07 fenced against "
            "(03-08: training to the full budget instead of stopping early cut held-out "
            "mse_per_dim 62.2% and left cond(g) unmoved). No reading of these cells may omit "
            "this override."
        )
    else:
        print("\nD-07's fence held: no cell in this record carries the F4 override.")

    baseline_rows = [c for c in non_diverged if float(c["weight"]) == 0.0]
    proceed = baseline_pathology_check(baseline_rows)
    if not proceed:
        print(
            "\n(NO PATHOLOGY AT THIS EPOCH BUDGET printed for the record; a fresh run_ladder "
            "invocation would have exited 2 before training any prior-active cell. --summary "
            "still prints whatever Tier-1/Tier-2 tables the record on disk supports.)"
        )

    # Combination cells (D-12) are excluded from the single-arm rows below -- a combination
    # cell shares its scale-arm weight with that arm's own top rung, and folding it in would
    # silently inflate that rung's seed count and corrupt its median. Combination cells get
    # their own labelled read-out further down.
    arm_cells = [c for c in non_diverged if not c.get("combination")]
    combination_cells = [c for c in non_diverged if c.get("combination")]

    modes_present = sorted({c["mode"] for c in arm_cells if float(c["weight"]) != 0.0})
    for mode in modes_present:
        rows = rows_for_mode(arm_cells, mode)
        tier1_verdict = tier1_readout(rows, mode)
        tier2_readout(rows, mode, tier1_verdict)

    combination_config_ids = [c["config_id"] for c in combination_cells]
    rendered_tables: List[str] = []
    if combination_cells:
        rendered_tables.append(_render_and_capture(print_combination_tier1, combination_cells))
        rendered_tables.append(_render_and_capture(print_combination_tier2, combination_cells))

    rendered_tables.append(_render_and_capture(print_full_cell_table, cells))

    if combination_cells:
        n = sum(1 for t in rendered_tables if any(cid in t for cid in combination_config_ids))
        m = sum(
            1
            for t in rendered_tables
            if any(cid in t for cid in combination_config_ids) and COMBINATION_POST_HOC_LABEL in t
        )
        if n > 0:
            if m == n:
                print(
                    f"\nPOST-HOC LABEL RENDERED ON EVERY TABLE CONTAINING THE COMBINATION CELL "
                    f"({m}/{n})"
                )
            else:
                print(
                    f"\nPOST-HOC LABEL MISSING FROM {n - m} OF {n} TABLES CONTAINING THE "
                    f"COMBINATION CELL"
                )


def _render_and_capture(fn, *args: Any, **kwargs: Any) -> str:
    """Runs `fn(*args, **kwargs)` with stdout captured, writes the captured text through to the
    REAL stdout unchanged (so `--summary | tee log` still sees it), and returns the captured
    text so `print_summary`'s post-render audit can grep it independently of everything else
    printed before or after. Never used to suppress output -- only to let two audiences (the
    terminal/log and the audit) read the identical bytes."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    text = buf.getvalue()
    sys.stdout.write(text)
    return text


def print_combination_tier1(cells: Sequence[Dict[str, Any]]) -> None:
    """The combination cell's own Tier-1-shaped block -- median + full spread of `lambda_min`,
    `lambda_max`, `log10_det_g` and `cond`, across whatever seeds are on record. Always carries
    `COMBINATION_POST_HOC_LABEL` (D-02) -- there is no un-labelled combination table."""
    print(f"\n--- TIER 1 (combination) -- {COMBINATION_POST_HOC_LABEL} ---")
    if not cells:
        print("  no combination cells on record.")
        return
    config_ids = [c["config_id"] for c in cells]
    print(f"  cells: {config_ids}")
    print(
        f"  scale_weight={float(cells[0]['scale_weight']):.6e}  "
        f"christoffel_weight={float(cells[0]['christoffel_weight']):.6e}"
    )
    print(f"  selection_rule: {cells[0]['selection_rule']}")
    for f in ("lambda_min", "lambda_max", "log10_det_g", "cond"):
        vals = [_cell_field_median(c, f) for c in cells]
        med, spread = median_and_spread(vals)
        spread_str = ", ".join(f"{v:.6e}" for v in spread)
        print(f"    {f:<12s} median={med:.6e}  spread=[{spread_str}]")
    print(f"  {COMBINATION_POST_HOC_LABEL}")


def print_combination_tier2(cells: Sequence[Dict[str, Any]]) -> None:
    """The combination cell's own Tier-2-shaped block -- four axes, never collapsed, plus the
    D-09 bias guard. Always carries `COMBINATION_POST_HOC_LABEL` (D-02)."""
    print(f"\n--- TIER 2 (combination) -- {COMBINATION_POST_HOC_LABEL} ---")
    if not cells:
        print("  no combination cells on record.")
        return
    config_ids = [c["config_id"] for c in cells]
    print(f"  cells: {config_ids}")
    for axis_name, key in (
        ("direction", "direction_median_cosine"),
        ("magnitude", "magnitude_median_ratio"),
        ("rank", "rank_spearman_rho"),
    ):
        vals = [
            float(c["fidelity"][key])
            for c in cells
            if c.get("fidelity", {}).get("applicable", True) and c["fidelity"].get(key) is not None
        ]
        if vals:
            med, spread = median_and_spread(vals)
            spread_str = ", ".join(f"{v:.6f}" for v in spread)
            print(f"    {axis_name:<12s} median={med:.6f}  spread=[{spread_str}]")
        else:
            print(f"    {axis_name:<12s} UNDEFINED")
    cal_pairs = [
        (float(c["fidelity"]["calibration_r2"]), float(c["fidelity"]["calibration_slope"]))
        for c in cells
        if c.get("fidelity", {}).get("applicable", True)
        and c["fidelity"].get("rank_calibration_applicable", True)
    ]
    if cal_pairs:
        r2_med, r2_spread = median_and_spread([v[0] for v in cal_pairs])
        slope_med, slope_spread = median_and_spread([v[1] for v in cal_pairs])
        r2_spread_str = ", ".join(f"{v:.6f}" for v in r2_spread)
        slope_spread_str = ", ".join(f"{v:.6f}" for v in slope_spread)
        print(
            f"    calibration  r2 median={r2_med:.6f}  spread=[{r2_spread_str}]  "
            f"slope median={slope_med:.6f}  spread=[{slope_spread_str}]"
        )
    else:
        print("    calibration  UNDEFINED")
    mse_vals = [float(c["reconstruction"]["mse_per_dim"]) for c in cells]
    mse_med, mse_spread = median_and_spread(mse_vals)
    mse_spread_str = ", ".join(f"{v:.6e}" for v in mse_spread)
    print(
        f"\n  bias guard (D-09): held-out mse_per_dim median={mse_med:.6e}  "
        f"spread=[{mse_spread_str}]"
    )
    print(f"  {COMBINATION_POST_HOC_LABEL}")


def print_full_cell_table(cells: Sequence[Dict[str, Any]]) -> None:
    """The full per-cell table -- every cell on record, one line each, combination cells tagged
    with `COMBINATION_POST_HOC_LABEL` inline so the label survives even a reader who only scans
    this one table."""
    print("\n--- full per-cell table ---")
    for c in sorted(
        cells, key=lambda r: (r.get("mode", ""), float(r.get("weight", 0.0)), int(r.get("model_seed", 0)))
    ):
        diverged_tag = " DIVERGED" if c.get("nonfinite_guard", {}).get("tripped") else ""
        combo_tag = f"  {COMBINATION_POST_HOC_LABEL}" if c.get("combination") else ""
        print(
            f"  {c.get('config_id')}{diverged_tag}  mode={c.get('mode')}  "
            f"weight={float(c.get('weight', 0.0)):.6e}  seed={c.get('model_seed')}  "
            f"epochs_run={c.get('epochs_run')}/{c.get('max_epochs')}  "
            f"relief_applied={c.get('relief_applied')}{combo_tag}"
        )


# =============================================================================================
# 03.1-04 Task 2: run_ladder -- baseline, scale arm, christoffel arm, at the sizing ratified by
# the most recent kind=='probe', probe_type=='summary' record on disk (03.1-03 Task 3).
# =============================================================================================


def _read_latest_probe_summary(record_path: Path) -> Optional[Dict[str, Any]]:
    """The most recent ratified `kind == 'probe', probe_type == 'summary'` record carrying a
    non-null `chosen_max_epochs` -- the sizing `run_ladder` stamps onto every cell. `None` if no
    such record exists on disk."""
    if not record_path.exists():
        return None
    found: Optional[Dict[str, Any]] = None
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if (
                rec.get("kind") == "probe"
                and rec.get("probe_type") == "summary"
                and rec.get("chosen_max_epochs") is not None
            ):
                found = rec
    return found


def run_ladder(
    record_path: Path,
    resume: bool,
    arm: Optional[str],
    max_epochs: Optional[int],
    fractions: Optional[Sequence[float]],
    eval_n: Optional[int],
    christoffel_max_rows: Optional[int],
    seeds: Sequence[int],
    device: torch.device,
) -> None:
    """The overnight run: calibrate once, fit the three `weight=0` baseline cells (D-05), run
    the baseline-pathology precondition (§G), then both single-mode arms, at the sizing ratified
    at `03.1-03` Task 3 -- read from the most recent probe summary record on disk, never
    synthesized locally. `--resume` skips any `cell_key` already present."""
    probe_summary = _read_latest_probe_summary(record_path)
    if probe_summary is None:
        raise RuntimeError(
            "run_ladder: no kind=='probe', probe_type=='summary' record with a non-null "
            "chosen_max_epochs found on disk. Run --probe (and, if F1-F3 do not fit, "
            "--confirm-f4-override once ratified) before --run-ladder -- the ratified sizing "
            "must exist on the record before the overnight run can stamp it onto every cell."
        )
    relief_applied: List[str] = list(probe_summary["relief_applied"])
    resolved_max_epochs = int(max_epochs if max_epochs is not None else probe_summary["chosen_max_epochs"])
    resolved_fractions: Tuple[float, ...] = tuple(
        fractions if fractions is not None else probe_summary["final_fractions"]
    )
    resolved_eval_n = int(eval_n if eval_n is not None else probe_summary["final_eval_n"])
    resolved_christoffel_max_rows = int(
        christoffel_max_rows
        if christoffel_max_rows is not None
        else probe_summary["final_christoffel_max_rows"]
    )

    _banner(
        f"--run-ladder: max_epochs={resolved_max_epochs}  fractions={resolved_fractions}  "
        f"eval_n={resolved_eval_n}  christoffel_max_rows={resolved_christoffel_max_rows}  "
        f"relief_applied={relief_applied}  seeds={list(seeds)}  resume={resume}  arm={arm!r}"
    )
    if F4_OVERRIDE_LABEL in relief_applied:
        print(
            "D-07 OVERRIDE FIRED -- this run's ratified relief_applied carries the F4 epoch-cut "
            "override; every cell below is stamped with it."
        )

    completed = load_completed(record_path)
    t_start = time.monotonic()
    projected_total_s = probe_summary.get("caveat_adjusted_projected_total_s") or probe_summary.get(
        "projected_total_s"
    )

    def _elapsed_report() -> None:
        elapsed = time.monotonic() - t_start
        if projected_total_s:
            pct = 100.0 * elapsed / projected_total_s
            print(f"    elapsed={elapsed:.1f}s  ratified_projection={projected_total_s:.1f}s  ({pct:.1f}%)")
        else:
            print(f"    elapsed={elapsed:.1f}s")

    # --- 1. Calibrate once -----------------------------------------------------------------
    fx = sc.build_fixture(
        LADDER_FIXTURE, n=LADDER_N, d=LADDER_CHART_DIM, D=LADDER_AMBIENT, seed=LADDER_FIXTURE_SEED
    )
    X = np.asarray(fx["X"], dtype=np.float64)
    x32 = torch.tensor(X, dtype=torch.float32).to(device)
    train_idx, _holdout_idx = pu._split(X.shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long, device=device)]

    torch.manual_seed(LADDER_SEEDS[0])
    calib_model = sc.build_matched_cae(
        LADDER_N_CHARTS, LADDER_CHART_DIM, LADDER_EMBED_DIM, LADDER_AMBIENT, device
    )
    calibrated = calibrate_weights(
        calib_model,
        x_train32[: pu.BATCH],
        fractions=resolved_fractions,
        christoffel_max_rows=resolved_christoffel_max_rows,
    )
    print("\nCalibrated weights (computed once, reused for every seed):")
    for (mode, frac), info in calibrated.items():
        print(
            f"  mode={mode}  fraction={frac}  weight={info['weight']:.6e}  "
            f"base_recon={info['base_recon']:.6e}  base_penalty={info['base_penalty']:.6e}"
        )

    def _run_and_record(mode: str, weight: float, weight_fraction: Optional[float], seed: int) -> None:
        key = cell_key(mode, weight, seed)
        if resume and key in completed:
            print(f"  --resume: skipping already-completed cell mode={mode} weight={weight:.6e} seed={seed}")
            return
        try:
            rec = run_cell(
                mode=mode,
                weight=weight,
                weight_fraction=weight_fraction,
                model_seed=seed,
                max_epochs=resolved_max_epochs,
                n=LADDER_N,
                chart_dim=LADDER_CHART_DIM,
                ambient=LADDER_AMBIENT,
                n_charts=LADDER_N_CHARTS,
                embed_dim=LADDER_EMBED_DIM,
                eval_n=resolved_eval_n,
                epoch_block=LADDER_EPOCH_BLOCK,
                device=device,
                christoffel_max_rows=resolved_christoffel_max_rows,
                base_recon_at_init=calibrated.get((mode, weight_fraction), {}).get("base_recon"),
                base_penalty_at_init=calibrated.get((mode, weight_fraction), {}).get("base_penalty"),
                relief_applied=relief_applied,
            )
        except Exception as exc:  # noqa: BLE001 -- an unattended overnight run must not die on one cell
            print(
                f"  DIVERGED -- not read as evidence (exception during cell): mode={mode} "
                f"weight={weight:.6e} seed={seed}: {exc!r}"
            )
            mode_segment = "any" if float(weight) == 0.0 else mode
            rec = {
                "kind": "ladder_cell",
                "config_id": f"ladder_{mode_segment}_w{weight:.6e}_s{seed}_e{resolved_max_epochs}_n{resolved_eval_n}",
                "fixture": LADDER_FIXTURE,
                "fixture_seed": LADDER_FIXTURE_SEED,
                "mode": mode,
                "weight": float(weight),
                "weight_fraction": weight_fraction,
                "model_seed": int(seed),
                "max_epochs": resolved_max_epochs,
                "eval_sample_size": int(resolved_eval_n),
                "christoffel_max_rows_per_chart": int(resolved_christoffel_max_rows),
                "nonfinite_guard": {"tripped": True, "where": [f"exception: {exc!r}"]},
                "relief_applied": list(relief_applied),
                "arm_label": "baseline" if float(weight) == 0.0 else mode,
            }
            completed[key] = rec
            append_record(record_path, rec)
            _elapsed_report()
            return
        completed[key] = rec
        append_record(record_path, rec)
        print_cell_row(rec)
        _elapsed_report()

    # --- 2. The three weight=0 baseline cells (D-05) ----------------------------------------
    print("\n--- baseline cells (weight=0, shared across arms via cell_key's 'any' sentinel) ---")
    for seed in seeds:
        _run_and_record(LADDER_MODES[0], 0.0, 0.0, seed)

    baseline_rows = [
        completed[cell_key("any", 0.0, seed)] for seed in seeds if cell_key("any", 0.0, seed) in completed
    ]

    # --- 3. Baseline-pathology precondition (03.1-01-PLAN.md §G) ----------------------------
    proceed = baseline_pathology_check(baseline_rows)
    if not proceed:
        print("\nNO PATHOLOGY AT THIS EPOCH BUDGET -- exiting before any prior-active cell.")
        sys.exit(2)

    arms = [m for m in LADDER_MODES if arm is None or m == arm]

    # --- 4/5. scale and christoffel arms ------------------------------------------------------
    for mode in arms:
        print(f"\n--- {mode} arm ---")
        for fraction in resolved_fractions:
            weight = calibrated[(mode, float(fraction))]["weight"]
            for seed in seeds:
                _run_and_record(mode, weight, float(fraction), seed)

    print("\n--- run_ladder complete; printing --summary against the real record ---")
    print_summary(record_path)


# =============================================================================================
# 03.1-05 Task 1: the post-hoc combination cell -- D-02's best-of-each selection rule, D-12's
# two-nested-contextmanagers mechanism, zero new machinery.
# =============================================================================================


def select_best_rung(
    rows: Sequence[Tuple[float, List[Dict[str, Any]]]],
    mode: str,
    tier1_verdict: str,
) -> Tuple[float, Optional[float], str]:
    """D-02's best-of-each selection rule, declared in `03.1-01-PLAN.md` §F before the ladder
    ran -- not one word of it changes now that the ladder's numbers exist. `rows`:
    `rows_for_mode`'s output for this mode, weight ascending, `DIVERGED` cells already excluded
    by the caller. `tier1_verdict`: this mode's own `tier1_readout` verdict -- used only to
    attach the "no mechanism in {mode} arm" label; it never substitutes for the metric-based
    rule below, and does not change which weight is returned when the rule below has an answer.

    **Nothing in this function reads `reconstruction.mse_per_dim`** -- D-09's bias guard is a
    reading rule for the human, never a selector, and a selector on reconstruction would
    silently reintroduce the circular threshold this milestone already deferred.

    Returns `(weight, weight_fraction, reason)`.
    """
    nonzero_rows = [(w, cells) for w, cells in rows if w > 0.0]
    if not nonzero_rows:
        raise ValueError(f"select_best_rung: no prior-active rung on record for mode={mode!r}.")
    weight_fraction_by_weight = {
        float(w): cells[0].get("weight_fraction") for w, cells in nonzero_rows
    }

    if mode == "scale":

        def _abs_log10_det_g(item: Tuple[float, List[Dict[str, Any]]]) -> float:
            _, cells = item
            med, _ = median_and_spread([_cell_field_median(c, "log10_det_g") for c in cells])
            return abs(med)

        best_w, _ = min(nonzero_rows, key=_abs_log10_det_g)
        reason = (
            "best_scale = the rung whose median log10_det_g is closest to 0.0 -- the quantity "
            "scale's own penalty (log det g / d)^2 minimizes."
        )
    elif mode == "christoffel":
        w0_cells = next((cells for w, cells in rows if w == 0.0), None)
        if not w0_cells:
            raise ValueError("select_best_rung: no weight=0.0 baseline cell on record.")
        lambda_max_w0, _ = median_and_spread(
            [_cell_field_median(c, "lambda_max") for c in w0_cells]
        )

        eligible: List[Tuple[float, List[Dict[str, Any]]]] = []
        for w, cells in nonzero_rows:
            lambda_max_med, _ = median_and_spread(
                [_cell_field_median(c, "lambda_max") for c in cells]
            )
            fell_a_decade = (
                lambda_max_w0 > 0
                and lambda_max_med > 0
                and (lambda_max_med / lambda_max_w0) <= 0.1
            )
            if not fell_a_decade:
                eligible.append((w, cells))

        if not eligible:
            best_w, _ = max(nonzero_rows, key=lambda item: item[0])
            reason = (
                "best_christoffel: every prior-active rung's median lambda_max fell by at "
                "least one decade from weight=0 (the same COLLAPSE guard as Tier 1) -- no rung "
                "passes the guard, so the largest weight rung is used per the declared fallback."
            )
        else:

            def _cond_med(item: Tuple[float, List[Dict[str, Any]]]) -> float:
                _, cells = item
                med, _ = median_and_spread([_cell_field_median(c, "cond") for c in cells])
                return med

            best_w, _ = min(eligible, key=_cond_med)
            reason = (
                "best_christoffel = the rung with the lowest median cond(g) among rungs whose "
                "median lambda_max did not fall by at least one decade from the weight=0 cell "
                "-- the same COLLAPSE guard as Tier 1."
            )
    else:
        raise ValueError(f"select_best_rung: mode must be 'scale' or 'christoffel'; got {mode!r}.")

    if tier1_verdict == "MECHANISM NOT DEMONSTRATED":
        reason += (
            f"  Tier 1 for {mode} is MECHANISM NOT DEMONSTRATED -- D-02's rule above still "
            f"applies (D-08 ruled this mode's Tier 2 'not read as evidence about the "
            f"estimand', so the pick can only come from Tier 1), and this weight contributes "
            f"to the combination cell on Tier-1 evidence alone: no mechanism in the {mode} arm."
        )

    return float(best_w), weight_fraction_by_weight.get(float(best_w)), reason


def run_combination(
    record_path: Path,
    device: torch.device,
    seeds: Sequence[int] = LADDER_SEEDS,
) -> None:
    """D-02's post-hoc combination arm: two nested `decoder_prior_active` blocks -- `scale`
    OUTER, `christoffel` INNER (D-12) -- at the weights `select_best_rung` derives from the
    ladder's own Tier-1 read-out. The sizing (max_epochs, eval_n, christoffel_max_rows,
    relief_applied) is REUSED from the most recent ratified `--probe` summary record on disk,
    never re-derived locally -- the combination cell shares the ladder's own protocol.

    Weights were chosen after seeing the ladder's numbers, so this cell is a confirmation, not
    an independent measurement -- `COMBINATION_POST_HOC_LABEL` is stamped on every combination
    record by `run_cell` itself."""
    _banner(
        "--combination: the post-hoc combination cell (D-02) -- scale outer, christoffel "
        "inner (D-12)"
    )
    if not record_path.exists():
        raise RuntimeError("--combination: no record file on disk -- run --run-ladder first.")

    records: List[Dict[str, Any]] = []
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    ladder_cells = [r for r in records if r.get("kind") == "ladder_cell"]
    non_diverged = [c for c in ladder_cells if not c.get("nonfinite_guard", {}).get("tripped")]
    arm_cells = [c for c in non_diverged if not c.get("combination")]

    probe_summary = _read_latest_probe_summary(record_path)
    if probe_summary is None:
        raise RuntimeError(
            "--combination: no ratified kind=='probe', probe_type=='summary' record with a "
            "non-null chosen_max_epochs on disk -- run --probe and --run-ladder first. The "
            "combination cell reuses the ladder's own ratified sizing, never a locally "
            "re-derived one."
        )
    resolved_max_epochs = int(probe_summary["chosen_max_epochs"])
    resolved_eval_n = int(probe_summary["final_eval_n"])
    resolved_christoffel_max_rows = int(probe_summary["final_christoffel_max_rows"])
    relief_applied: List[str] = list(probe_summary["relief_applied"])

    print(
        f"combination sizing (reused from the ratified ladder configuration): "
        f"max_epochs={resolved_max_epochs}  eval_n={resolved_eval_n}  "
        f"christoffel_max_rows={resolved_christoffel_max_rows}  relief_applied={relief_applied}"
    )

    selections: Dict[str, Tuple[float, Optional[float], str]] = {}
    for mode in LADDER_MODES:
        rows = rows_for_mode(arm_cells, mode)
        tier1_verdict = tier1_readout(rows, mode)
        weight, weight_fraction, reason = select_best_rung(rows, mode, tier1_verdict)
        selections[mode] = (weight, weight_fraction, reason)
        print(
            f"\nselect_best_rung({mode!r}) -> weight={weight:.6e}  "
            f"weight_fraction={weight_fraction}  tier1_verdict={tier1_verdict!r}"
        )
        print(f"  reason: {reason}")

    scale_weight, scale_fraction, scale_reason = selections["scale"]
    christoffel_weight, christoffel_fraction, christoffel_reason = selections["christoffel"]
    selection_rule = f"scale: {scale_reason} | christoffel: {christoffel_reason}"

    print(
        f"\ncombination weights: scale={scale_weight:.6e} (fraction={scale_fraction})  "
        f"christoffel={christoffel_weight:.6e} (fraction={christoffel_fraction})"
    )
    print(COMBINATION_POST_HOC_LABEL)

    completed = load_completed(record_path)
    t_start = time.monotonic()
    for seed in seeds:
        key = cell_key("combination", scale_weight, seed)
        if key in completed:
            print(f"  skipping already-completed combination cell seed={seed}")
            continue
        rec = run_cell(
            mode="scale",
            weight=scale_weight,
            weight_fraction=scale_fraction,
            model_seed=seed,
            max_epochs=resolved_max_epochs,
            n=LADDER_N,
            chart_dim=LADDER_CHART_DIM,
            ambient=LADDER_AMBIENT,
            n_charts=LADDER_N_CHARTS,
            embed_dim=LADDER_EMBED_DIM,
            eval_n=resolved_eval_n,
            epoch_block=LADDER_EPOCH_BLOCK,
            device=device,
            christoffel_max_rows=resolved_christoffel_max_rows,
            relief_applied=relief_applied,
            second_mode="christoffel",
            second_weight=christoffel_weight,
            second_weight_fraction=christoffel_fraction,
            selection_rule=selection_rule,
        )
        # rec['mode'] == 'combination' (run_cell's own bookkeeping fix, see its docstring), so
        # this matches exactly what a future load_completed() will derive from the appended
        # record -- a re-invocation with the same seeds is idempotent.
        completed[key] = rec
        append_record(record_path, rec)
        print_cell_row(rec)
        elapsed = time.monotonic() - t_start
        print(f"    elapsed={elapsed:.1f}s")

    print("\n--combination complete; printing --summary against the full record")
    print_summary(record_path)


def print_dry_run_report() -> None:
    _banner("decoder_prior_ladder_run.py -- DRY RUN")

    # --- Section A: architecture and fixture, inherited from synthetic_control_run.py -------
    print("Section A -- architecture and fixture, inherited, not restated:")
    print(f"  fixture              = {LADDER_FIXTURE!r}   (D-01)")
    print(f"  n                    = {LADDER_N}   (synthetic_control_run.CONTROL_N)")
    print(f"  chart_dim            = {LADDER_CHART_DIM}   (pu.PU_CHART_DIM)")
    print(f"  ambient D            = {LADDER_AMBIENT}   (pu.AMBIENT_DIM)")
    print(f"  embed_dim            = {LADDER_EMBED_DIM}   (pu.PU_EMBED_DIM)")
    print(f"  n_charts             = {LADDER_N_CHARTS}   (pu.CONVERGE_N_CHARTS)")
    print(f"  domain_radius        = {LADDER_DOMAIN_RADIUS}   (sc.SADDLE_DOMAIN_RADIUS)")
    print(f"  LADDER_FIXTURE_SEED  = {LADDER_FIXTURE_SEED}")
    print(f"  LADDER_EPOCH_BLOCK   = {LADDER_EPOCH_BLOCK}")
    print(
        "  The fixture seed is held FIXED at 20260816 for every cell and only the model/"
        "training seed varies. The mandatory anchor cell is the one exception -- it calls "
        "run_one_control unchanged with seed=CONTROL_FIXTURE_SEED, because it must reproduce "
        "the sealed row bit for bit."
    )
    print(
        "  Early stopping is DISABLED for every cell, inherited via pu._converge_cfg, which "
        "sets early_stop_patience = max_epochs + 1 and wallclock_ceiling_s = inf. 03-08 "
        "measured that removing total-loss early stopping cut held-out mse_per_dim 62.2% and "
        "left cond(g) unmoved, so the converged protocol is the better one, and the anchor "
        "cell requires it -- epochs_run == max_epochs for every cell, directly checkable."
    )

    # --- Section B: ladder sizing, pre-declared, probe-selected -----------------------------
    print("\nSection B -- ladder sizing, pre-declared, probe-selected:")
    print(f"  LADDER_SEEDS            = {LADDER_SEEDS}   (D-03)")
    print(f"  LADDER_MODES            = {LADDER_MODES}   (D-02; conformal/isometry deferred)")
    print(f"  LADDER_FRACTIONS        = {LADDER_FRACTIONS}   (D-04; 3 rungs per mode)")
    print(f"  LADDER_EVAL_N           = {LADDER_EVAL_N}   (D-16)")
    print(f"  LADDER_EPOCH_CANDIDATES = {LADDER_EPOCH_CANDIDATES}")
    print(f"  LADDER_BUDGET_S         = {LADDER_BUDGET_S}   (8.0 h, D-06's envelope less the anchor cell)")
    print(f"  ANCHOR_CONFIG_ID        = {ANCHOR_CONFIG_ID!r}   (D-13)")
    print(f"  ANCHOR_RHO_SPEARMAN     = {ANCHOR_RHO_SPEARMAN!r}   (D-13; matched with ==, never a tolerance)")
    print(
        "  Cell count: 3 baseline (weight=0, shared across modes by the cell_key 'any' "
        "sentinel) + 3 fractions x 3 seeds x 2 modes (18) + 3 combination = 24 cells."
    )
    print(
        f"\n  Sealed measurements this projection rests on (control_saddle_nc4_d20_ep300):"
        f"\n    curv_wallclock_s  = {SEALED_CURV_WALLCLOCK_S!r}"
        f"\n    train_wallclock_s = {SEALED_TRAIN_WALLCLOCK_S!r}"
    )
    eval_projection_s = SEALED_CURV_WALLCLOCK_S * LADDER_EVAL_N / LADDER_N
    print(
        f"  Evaluation projection at LADDER_EVAL_N={LADDER_EVAL_N}, ASSUMING evaluation cost is "
        f"linear in row count: {SEALED_CURV_WALLCLOCK_S:.6f} * {LADDER_EVAL_N}/{LADDER_N} ~= "
        f"{eval_projection_s:.1f} s/cell, 24 x {eval_projection_s:.1f} ~= "
        f"{24 * eval_projection_s:.1f} s ~= {24 * eval_projection_s / 3600.0:.2f} h."
    )
    print(
        "  That linearity is research assumption A2 and is UNVERIFIED -- plan 03.1-03's probe "
        "measures evaluation cost at both n=10000 and n=2000 directly and the projection is "
        "recomputed from the measurement, never from this number."
    )
    print(
        "  D-16 consequence: ladder cells evaluate on the 2,000-row pu._split holdout rather "
        "than the full 10,000-row cloud, so per-cell four-axis scores carry more sampling "
        "noise than the sealed full-cloud rows, and ladder cells are not cell-for-cell "
        "comparable to Phase 3's sealed d=20 row on evaluation-set size."
    )
    print(
        "  Training projection is NOT CLOSED in this plan: the per-mode multiplier m at "
        "chart_dim=20 is not known (research assumption A1) and is measured by the probe, "
        "never extrapolated."
    )

    # --- Section C: pre-declared relief ladder -----------------------------------------------
    print("\nSection C -- pre-declared relief ladder, F1-F3 in order, F4 only under override:")
    print(
        "  F1 -- rungs 3 -> 2, keeping the extremes (0.01, 1.0) and dropping the middle. "
        "24 -> 18 cells. Cost: each mode's curve has 3 points instead of 4; monotone "
        "movement is still readable, resolution is coarser."
    )
    print(
        "  F2 -- LADDER_EVAL_N 2000 -> 1000. D-16's own permitted lever. Cost: more sampling "
        "noise per cell; eval_sample_size recorded per cell so a re-evaluation is well "
        "defined."
    )
    print(
        "  F3 -- CHRISTOFFEL_MAX_ROWS_PER_CHART 8 -> 4. D-07's named valve. christoffel arm "
        "ONLY -- isometry_penalty (which scale reuses) has no row subsampling, so this valve "
        "does nothing for the scale arm. Cost: wider error bars on the christoffel arm "
        "specifically; recorded per cell as christoffel_max_rows_per_chart."
    )
    print("\n  F4 — OVERRIDES D-07 (NOT AN ORDINARY FALLBACK)")
    print(
        "    D-07's own fence: \"Explicitly not the valve: cutting the fixture's sample "
        "count ... or cutting epochs (under-training is a confound Phase 3 spent a whole "
        "quick task on).\""
    )
    print(
        "    D-07's own stated reason, measured not rhetorical: 03-08 found that training "
        "to the full budget instead of stopping early cut held-out mse_per_dim 62.2% while "
        "leaving cond(g) unmoved. Under-training costs reconstruction directly and buys "
        "nothing on the metric this phase exists to move."
    )
    print(
        "    Scope of the override, stated precisely: the INITIAL epoch-count choice (the "
        "probe picking the largest LADDER_EPOCH_CANDIDATES that fits) is ORDINARY "
        "DISCRETION and is NOT an override. F4 is a different act: it cuts epochs AS A "
        "RESPONSE TO A BUDGET OVERRUN, which is precisely the use D-07 fenced off."
    )
    print(
        "    F4 is scale-arm-only epoch reduction to the next candidate down. F4 is NEVER "
        "APPLIED AUTOMATICALLY -- apply_relief_ladder walks F1-F3 only. If F1-F3 are "
        "exhausted without fitting, the probe prints the F4-inclusive projection under this "
        "same heading, prints BUDGET NOT MET, and exits 1 into 03.1-03 Task 3's blocking "
        "checkpoint. F4 fires only when the runner is re-invoked with the explicit "
        "--confirm-f4-override flag."
    )
    print(
        "    Cost, both halves: the D-07 confound (the scale arm is under-trained relative "
        "to the very fence D-07 raised) and a protocol asymmetry between arms. Recorded per "
        "cell as max_epochs and as the literal string 'F4 (D-07 OVERRIDE: epoch cut)' inside "
        "that cell's relief_applied list."
    )
    print(
        "\n  If F1-F3 are exhausted and the projection still exceeds LADDER_BUDGET_S, the "
        "runner prints the full projection table for every step, plus the F4-inclusive "
        "projection under 'F4 AVAILABLE BUT NOT APPLIED -- OVERRIDES D-07', prints "
        "BUDGET NOT MET, and exits 1 -- an explicit user decision (D-11), never a silent "
        "scope cut."
    )

    # --- Sections D/E: the D-08 reading rules, declared verbatim -----------------------------
    print("\nSections D/E -- D-08 Tier-1/Tier-2 reading rules, declared before any number exists:")
    print(READING_RULES_TEXT)

    # --- Section F: combination-arm weight selection ------------------------------------------
    print("Section F -- combination-arm weight selection (D-02), declared before the ladder runs:")
    print(
        "  best_scale = the rung whose median log10_det_g is closest to 0.0 -- the quantity "
        "scale's own penalty minimizes."
    )
    print(
        "  best_christoffel = the rung with the lowest median cond(g) among rungs whose "
        "median lambda_max did not fall by at least one decade from the weight=0 cell (the "
        "same COLLAPSE guard as Tier 1)."
    )
    print(
        "  If a mode's Tier 1 is MECHANISM NOT DEMONSTRATED at every rung, its best is the "
        "largest weight rung and the combination cell carries the extra label "
        "'no mechanism in the {mode} arm'."
    )
    print(
        "  Every table carrying the combination cell labels it POST-HOC (weights selected "
        "after seeing the ladders; not an independent measurement) -- D-02 requires this and "
        "it is not a footnote."
    )

    # --- Section G: baseline-pathology precondition --------------------------------------------
    print("\nSection G -- baseline-pathology precondition, declared before the ladder runs:")
    print(
        "  After the three weight=0 cells complete at the chosen epoch budget, the runner "
        "prints BASELINE PATHOLOGY CHECK with median cond(g), lambda_min, lambda_max and "
        "log10_det_g per seed, against the sealed reference bands in 03-FINDINGS.md Section 5:"
    )
    print("    healthy-metric seed 20260813:  lambda_max = 3.352,     cond = 3.30e+07")
    print("    collapsed seeds:               lambda_max ~= 5.4e-07,  det(g) ~= 1e-162")
    print("    healthy d=4 saddle control:     cond = 4.098")
    print(
        "  If all three baseline seeds land in the d=4-healthy band -- median cond(g) within "
        "one decade of 4.098 and median lambda_max within one decade of 1 -- the runner "
        "prints NO PATHOLOGY AT THIS EPOCH BUDGET and exits 2 without training a single "
        "prior-active cell. That is itself a finding and escalation is an explicit user "
        "decision per D-11."
    )

    print("\n--dry-run: nothing executed, nothing written.")


# =============================================================================================
# CLI
# =============================================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the declared §A/§B constants and exit writing nothing.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run two cells at a deliberately tiny size (n={SMOKE_N}, d={SMOKE_D}, "
        f"D={SMOKE_AMBIENT}, {SMOKE_EPOCHS} epochs) to prove the wiring end to end.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help=f"Override the append-only record path (default: notebooks/.cache/"
        f"{RECORD_STEM}.jsonl).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' (default), 'cuda', or 'cuda:N'.",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=None,
        help=f"Override curvature evaluation sample size for --run-ladder (default: None, "
        f"meaning the most recent probe summary's final_eval_n is used -- D-16/F2).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs for a cell (default: resolved from LADDER_EPOCH_CANDIDATES "
        "by the probe in a later plan).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=list(LADDER_MODES),
        help=f"Restrict to one prior mode (default: all of LADDER_MODES={LADDER_MODES}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(LADDER_SEEDS),
        help=f"Model seeds to run (default: LADDER_SEEDS={LADDER_SEEDS}).",
    )
    parser.add_argument(
        "--anchor-check",
        action="store_true",
        help="Call synthetic_control_run.run_one_control unchanged and prove its "
        "rank_spearman_rho reproduces the sealed control_saddle_nc4_d20_ep300 row exactly "
        "(D-13). Costs roughly 2.6h -- the one full-cloud evaluation this phase runs.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Two-sided cost probe: training s/epoch per mode at the real architecture, "
        "curvature-evaluation s/row at three sample sizes, the 24-cell projection from measured "
        "numbers only, and the pre-declared F1-F3 relief ladder. Exits 1 with BUDGET NOT MET if "
        "no configuration fits LADDER_BUDGET_S.",
    )
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=PROBE_EPOCHS,
        help=f"Override PROBE_EPOCHS (default: {PROBE_EPOCHS}) for Half A's training-cost probe.",
    )
    parser.add_argument(
        "--probe-eval-sizes",
        type=str,
        default=None,
        help=f"Comma-separated override of PROBE_EVAL_SIZES (default: {PROBE_EVAL_SIZES}) for "
        "Half B's curvature-evaluation-cost probe.",
    )
    parser.add_argument(
        "--probe-skip-train",
        action="store_true",
        help="Skip re-training Half A's costly 'scale' and 'christoffel' cells; reuse their "
        "s_per_epoch from a prior --probe run's JSONL record (raises if none exists). The "
        "'weight=0' cell is always trained fresh -- its model is what Half B evaluates on. The "
        "defect-fix re-run path: fixing Half B's truncation bug does not require re-running "
        "Half A.",
    )
    parser.add_argument(
        "--confirm-f4-override",
        action="store_true",
        help="Apply F4 -- the scale-arm-only epoch cut that overrides D-07's fence against "
        "cutting epochs -- if F1-F3 do not fit LADDER_BUDGET_S. Never inferred from any other "
        "flag; defaults to False. Fires only when this flag is passed explicitly, ratified at "
        "03.1-03 Task 3's blocking checkpoint.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Read the record back and print READING_RULES_TEXT, the epochs-uniformity guard, "
        "any DIVERGED cells, the D-07 relief banner, BASELINE PATHOLOGY CHECK, then Tier 1 and "
        "Tier 2 per mode, then the full per-cell table. Runs no new cells.",
    )
    parser.add_argument(
        "--run-ladder",
        action="store_true",
        help="Run the ladder: calibrate once, fit the three weight=0 baseline cells (D-05), run "
        "the baseline-pathology precondition (03.1-01-PLAN.md §G), then both single-mode arms, "
        "at the sizing ratified by the most recent --probe summary record on disk. Overnight "
        "run, up to the ratified projection. --resume to continue an interrupted run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --run-ladder: skip any (mode, weight, seed) cell already present in the "
        "record file. weight=0.0 cells are mode-independent (cell_key's 'any' sentinel).",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        choices=list(LADDER_MODES),
        help="With --run-ladder: restrict to one arm (default: both LADDER_MODES). The three "
        "baseline cells always run regardless.",
    )
    parser.add_argument(
        "--combination",
        action="store_true",
        help="Run the post-hoc combination cell (D-02): two nested decoder_prior_active "
        "blocks, scale outer / christoffel inner (D-12), at the best-of-each weights "
        "select_best_rung derives from the ladder's own Tier-1 read-out on disk. Reuses the "
        "ratified --probe sizing. One cell per --seeds seed (default: LADDER_SEEDS). Idempotent "
        "-- a repeat invocation skips any (mode='combination', weight, seed) already recorded.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    record_path = Path(args.record_path) if args.record_path else _default_record_path()
    device = pu.resolve_device(args.device)

    _banner("Phase 03.1 plan 01: decoder-prior weight-ladder runner")
    print(f"record_path={record_path}")
    print(f"device={device}  torch_version={torch.__version__}")

    if args.dry_run:
        print_dry_run_report()
        return

    if args.smoke:
        run_smoke(record_path, device)
        return

    if args.anchor_check:
        anchor_check(device, record_path)
        return

    if args.probe:
        probe_eval_sizes = (
            tuple(int(v) for v in args.probe_eval_sizes.split(","))
            if args.probe_eval_sizes
            else PROBE_EVAL_SIZES
        )
        probe(
            device,
            record_path,
            confirm_f4_override=args.confirm_f4_override,
            probe_epochs=args.probe_epochs,
            probe_eval_sizes=probe_eval_sizes,
            probe_skip_train=args.probe_skip_train,
        )
        return

    if args.summary:
        print_summary(record_path)
        return

    if args.run_ladder:
        run_ladder(
            record_path=record_path,
            resume=args.resume,
            arm=args.arm,
            max_epochs=args.max_epochs,
            fractions=None,
            eval_n=args.eval_n,
            christoffel_max_rows=None,
            seeds=args.seeds,
            device=device,
        )
        return

    if args.combination:
        run_combination(record_path=record_path, device=device, seeds=args.seeds)
        return

    print(
        "Nothing to do: pass --dry-run, --smoke, --anchor-check, --probe, --run-ladder, "
        "--combination or --summary. --run-ladder trains the real ladder cells, gated behind "
        "this file's dry-run sizing report and this plan's ratified probe configuration on "
        "disk. --combination runs the post-hoc combination cell after the ladder completes."
    )


if __name__ == "__main__":
    main()
