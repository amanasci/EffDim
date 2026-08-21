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
import json
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
    """Every JSON-lines record on disk, keyed by :func:`cell_key`. Missing file -> empty dict
    (a fresh run, not an error)."""
    completed: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    if not record_path.exists():
        return completed
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
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
) -> Dict[str, Any]:
    """One `(mode, weight, model_seed)` measurement, at the given `max_epochs`/`n`/`chart_dim`/
    `ambient`/`n_charts`/`embed_dim`/`eval_n` -- callers pass the real §A/§B constants for a
    ladder cell, or the SMOKE_* constants for `run_smoke`."""
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
        "mode": mode,
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
        "arm_label": "baseline" if float(weight) == 0.0 else mode,
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
        default=LADDER_EVAL_N,
        help=f"Curvature evaluation sample size (default: LADDER_EVAL_N={LADDER_EVAL_N}, D-16).",
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

    print(
        "Nothing to do: pass --dry-run, --smoke or --anchor-check. The real 24-cell ladder "
        "loop and its probe are a later plan, gated behind this file's dry-run sizing report."
    )


if __name__ == "__main__":
    main()
