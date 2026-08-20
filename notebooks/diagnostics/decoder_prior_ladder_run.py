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
from typing import Any, Dict, List, Optional, Tuple

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
# --dry-run: prints the §A/§B constants. Stub only -- the full sizing/relief/reading-rules
# report is Task 3's job (READING_RULES_TEXT, print_dry_run_report's real body).
# =============================================================================================


def print_dry_run_report() -> None:
    _banner("decoder_prior_ladder_run.py -- DRY RUN")
    print(f"  fixture              = {LADDER_FIXTURE!r}")
    print(f"  n                    = {LADDER_N}")
    print(f"  chart_dim            = {LADDER_CHART_DIM}")
    print(f"  ambient D            = {LADDER_AMBIENT}")
    print(f"  embed_dim            = {LADDER_EMBED_DIM}")
    print(f"  n_charts             = {LADDER_N_CHARTS}")
    print(f"  fixture seed (fixed) = {LADDER_FIXTURE_SEED}")
    print(f"  epoch_block          = {LADDER_EPOCH_BLOCK}")
    print(f"  LADDER_SEEDS         = {LADDER_SEEDS}")
    print(f"  LADDER_MODES         = {LADDER_MODES}")
    print(f"  LADDER_FRACTIONS     = {LADDER_FRACTIONS}")
    print(f"  LADDER_EVAL_N        = {LADDER_EVAL_N}")
    print(f"  LADDER_EPOCH_CANDIDATES = {LADDER_EPOCH_CANDIDATES}")
    print(f"  LADDER_BUDGET_S      = {LADDER_BUDGET_S}")
    print(f"  ANCHOR_CONFIG_ID     = {ANCHOR_CONFIG_ID!r}")
    print(f"  ANCHOR_RHO_SPEARMAN  = {ANCHOR_RHO_SPEARMAN!r}")
    print(
        "\nThis is a STUB dry-run: the full sizing arithmetic, the F1-F4 relief ladder and the "
        "D-08 Tier-1/Tier-2 reading rules are declared and printed by a later task in this same "
        "plan (03.1-01 Task 3), not by this one."
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

    print(
        "Nothing to do: pass --dry-run or --smoke. The real 24-cell ladder loop is a later "
        "plan, gated behind this file's probe and dry-run sizing report."
    )


if __name__ == "__main__":
    main()
