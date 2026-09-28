"""Phase 3 Plan 01: Swiss roll n_charts x seed chart-decoder curvature sweep runner.

**D-15: this file and this plan are the whole of the gate machinery.** No
``PREREGISTRATION.md``, no ratification commit, no git-ancestry proof script, no verdict
JSON artifact, no threshold table. The bar below is declared in source, in this module,
before any sweep number exists -- that is the entire mechanism.

**D-02, ratified at Task 1's blocking checkpoint, restated here rather than only in the
plan SUMMARY.** The gate is an absolute floor: ``median rho_chart > ROLL_FLOOR`` (0.65)
over >=5 torch seeds, full spread reported. The raw-point centroid baseline
(``RAW_BASELINE_CONTEXT = 0.6712``) is reported as CONTEXT ONLY and gates nothing --
``02.5-09-SUMMARY.md`` section 3 warns it should not be read as a validated reference
point (it missed its own notebook's ``>0.90`` sanity bar). Swiss roll only; PU has no
analytic ``H`` so no equivalent gate exists there.

**D-05, ratified at Task 2's blocking checkpoint, restated here.** ``n_charts`` is an
in-scope Phase 3 hyperparameter -- an explicit user scope ruling overriding Phase 02.3's
on-hold status for this one knob, and nothing else in the phase-2 stage is reopened. The
roll sweep spans the measured monotone range plus one untested lower value:
``N_CHARTS_SWEEP = (2, 3, 5, 8)``. Nothing measured on the roll ever selects a PU
hyperparameter (D-06) -- the roll's job is solely to show the pipeline recovers a known
answer at *some* n_charts.

**This task (03-01 Task 3) proves the measurement chain on exactly ONE cell**
(``--n-charts 8 --seeds 0 --max-combos 1``), reproducing ``02.5-09``'s single-seed
``rho_chart = -0.0604`` at ``n_charts=8, seed=0`` through the complete
fit -> chart decoder -> ``torch.func`` curvature -> Spearman chain. The full 20-cell grid
(4 ``n_charts`` values x 5 seeds) and its median/floor/read-out summary are plan 03-02's
job, not this one's -- this file only measures and records, one cell at a time.

Every model hyperparameter below is ``02.5_swiss_roll_chart_curvature_check.ipynb``'s
values verbatim: reproducing that notebook's seed-0 measurement is the whole point of
Task 3, so nothing here is retuned.

Invoke:
    .venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --dry-run
    .venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --smoke
    .venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --n-charts 8 --seeds 0 --max-combos 1
    .venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --resume
    .venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --device cuda --resume

**03-07-SUPPLEMENT-01: opt-in GPU support, three caveats stated here and in ``--device``'s
help text.** (1) GPU runs do NOT reproduce CPU runs bit-for-bit -- CUDA RNG differs from CPU
RNG, so ``torch.manual_seed(seed)`` draws a different model on a different device; a GPU cell
is a different draw, not a reproduction, which is why every record now carries a ``device``
field. (2) float64 throughput is hardware-dependent -- curvature is float64-only
(``chart_curvature._assert_float64`` is never relaxed), and float64 on a consumer GPU can run
at 1/32-1/64 of float32 throughput (vs ~1/2 on a data-center GPU), so the curvature term may
be SLOWER on a consumer GPU than on CPU even though training (float32) should be faster.
(3) Do not mix devices within one sweep -- every cell in a comparable table must share one
device, or the seed spread mixes two different draws.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import cae, chart_curvature, curvature_probe  # noqa: E402
from pu_manifold.cache import CACHE_DIR  # noqa: E402

# =============================================================================================
# Module constants -- the bar is in source before any measurement exists (D-15).
# =============================================================================================

ROLL_FLOOR = 0.65
"""D-02's absolute floor: median rho_chart over >=5 torch seeds must exceed this for the
Swiss roll step to clear. Ratified at 03-01 Task 1's blocking checkpoint, before any
Phase 3 rho_chart value existed. Under the n_charts sweep the floor applies to the BEST
swept config (D-04) -- this module measures one cell at a time and never applies the
floor itself; that judgement belongs to plan 03-02's median/read-out layer."""

RAW_BASELINE_CONTEXT = 0.6712
"""The raw-point centroid estimator's measured Spearman (k=30, d=2) on this identical
fixture (02.5-05/02.5-09). CONTEXT ONLY -- gates nothing (D-02). 02.5-09-SUMMARY.md
section 3 warns this number should not be read as a validated reference point: it missed
02.5-05's own notebook's >0.90 sanity bar. Reported here only so a reader can see where
the chart-decoder arm sits relative to a baseline that works but was never validated as
correct, and asserted against the freshly-measured raw-point Spearman to four decimal
places so any drift in that path is caught here rather than mistaken for a decoder
effect."""

N_CHARTS_SWEEP: Tuple[int, ...] = (2, 3, 5, 8)
"""D-05's ratified swept set: the measured monotone range {3, 5, 8} (02.5-09) plus one
untested lower value, 2 -- the roll's true chart-count floor for a single-sheet
manifold."""

TORCH_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
"""D-01/D-05's ratified seed set. Seed 0 is the exact 02.5-09 configuration and is this
task's reproduction anchor."""

FIXTURE_SEED = 20260807
N_POINTS = 3000
"""The original pre-registered value (03-01, commit 4dc9b05). Superseded for the default
sweep by 03-02-AMENDMENT-01.md, which found this insufficient to estimate a
second-derivative quantity (curvature) reliably at n_charts<=3 -- CLAUDE.md's "~3,000
points" protocol was written for reconstruction sanity, a zeroth-order property. This
constant itself is NOT changed by the amendment (Section 4): raw_baseline_context() still
measures RAW_BASELINE_CONTEXT against exactly this value, unconditionally. The amended
sweep passes N_POINTS_AMENDED (or --n-points) into run_cell() explicitly instead."""

N_POINTS_AMENDED = 12000
"""03-02-AMENDMENT-01.md's amended point count. Not the module default -- callers opt in
via --n-points 12000 (or any explicit override), which also routes the record to a
distinct cache key so the sealed n=3000 grid is never clobbered or resumed-into."""

CHART_DIM = 2
EMBED_DIM = 8
HIDDEN = [64, 64]
K_BASELINE = 30

def resolve_device(device_str: str) -> torch.device:
    """Parse and validate a ``--device`` CLI argument ('cpu', 'cuda', or 'cuda:N'). Fails
    with an actionable message, naming the installed torch build, if ``cuda`` is requested
    but ``torch.cuda.is_available()`` is False -- 03-07-SUPPLEMENT-01's setup notes give the
    exact install command."""
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"--device={device_str!r} requested but torch.cuda.is_available() is False. "
            f"The installed torch build is CPU-only ({torch.__version__}). Install a CUDA "
            "build matching your driver's CUDA version, e.g.:\n"
            "  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "(see https://pytorch.org/get-started/locally/ for the correct cuXXX tag), then "
            "verify with:\n"
            '  .venv/bin/python -c "import torch; print(torch.cuda.is_available())"\n'
            "See .planning/phases/03-decoder-curvature-field/03-07-SUPPLEMENT-01.md for the "
            "full setup walkthrough."
        )
    return device


BASE_CFG: Dict[str, Any] = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch=64,
    max_epochs=300,
    early_stop_patience=25,
    early_stop_min_delta=1e-4,
    fps_pretrain_epochs=20,
    lip_weight=1e-3,
    lip_every_n_steps=1,
)
"""02.5_swiss_roll_chart_curvature_check.ipynb's hyperparameters verbatim. Not adjusted --
reproducing -0.0604 at seed 0 is the whole point of this task."""


# =============================================================================================
# JSON-lines append-only record: resumability, same idiom as template_benchmark_run.py.
# =============================================================================================


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_completed(record_path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Reads every JSON-lines record already on disk, keyed by (n_charts_configured, seed)
    -- the resumability index. Missing file -> empty dict (a fresh run, not an error)."""
    completed: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not record_path.exists():
        return completed
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed[(int(rec["n_charts_configured"]), int(rec["seed"]))] = rec
    return completed


def append_record(record_path: Path, record: Dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as f:
        f.write(json.dumps(_to_jsonable(record)) + "\n")


# =============================================================================================
# Raw-point context baseline -- computed ONCE per invocation, never per cell (D-02): it
# depends only on the fixture, not on any trained decoder.
# =============================================================================================


def raw_baseline_context() -> float:
    fx = curvature_probe.make_swiss_roll_fixture(n=N_POINTS, seed=FIXTURE_SEED)
    X, H_true_norm = fx["X"], fx["H_norm"]
    H_raw = curvature_probe.centroid_mean_curvature(X, k=K_BASELINE, d=CHART_DIM)
    h_raw = curvature_probe.mean_curvature_norm(H_raw)
    rho_raw = curvature_probe.spearman_gate_statistic(h_raw, H_true_norm)
    print(
        f"raw-point centroid baseline (context, gates nothing): rho = {rho_raw:.4f}  "
        f"(RAW_BASELINE_CONTEXT = {RAW_BASELINE_CONTEXT})"
    )
    if round(rho_raw, 4) != RAW_BASELINE_CONTEXT:
        raise ValueError(
            f"raw_baseline_context: measured rho_raw={rho_raw:.6f} disagrees with "
            f"RAW_BASELINE_CONTEXT={RAW_BASELINE_CONTEXT} to four decimal places -- the "
            "raw-point path has drifted from its sealed 02.5-05 measurement; this is a "
            "regression in curvature_probe, not a decoder effect."
        )
    return rho_raw


# =============================================================================================
# Per-cell measurement: fit -> chart decoder -> torch.func curvature -> Spearman vs analytic H.
# =============================================================================================


def run_cell(
    n_charts: int,
    seed: int,
    *,
    n_points: int = N_POINTS,
    max_epochs: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """One (n_charts, seed) measurement, following
    ``02.5_swiss_roll_chart_curvature_check.ipynb``'s cell sequence exactly. ``n_points``
    and ``max_epochs`` are overridable only for ``--smoke``'s cheap wiring check; the real
    sweep always uses the module constants (``N_POINTS``, ``BASE_CFG["max_epochs"]``).

    ``device``: default ``cpu``, zero behaviour change. Both models are constructed exactly
    as before, on CPU, under ``torch.manual_seed(seed)`` -- device placement happens strictly
    AFTER construction via ``model.to(device)``, so this argument never touches RNG
    consumption order (03-07-SUPPLEMENT-01, hard constraint 2). See that supplement for the
    non-reproducibility-across-devices and float64-throughput caveats."""
    fx = curvature_probe.make_swiss_roll_fixture(n=n_points, seed=FIXTURE_SEED)
    X, t, H_true_norm, global_std = fx["X"], fx["t"], fx["H_norm"], fx["global_std"]

    # Analytic mean curvature VECTOR -- the spiral's own curvature vector in the x-z
    # plane, copied verbatim from 02.5_swiss_roll_chart_curvature_check.ipynb cell 4.
    ct, st = np.cos(t), np.sin(t)
    d1 = np.stack([ct - t * st, st + t * ct], axis=1)  # c'(t)
    d2 = np.stack([-2 * st - t * ct, 2 * ct - t * st], axis=1)  # c''(t)
    speed2 = np.sum(d1 * d1, axis=1)
    k_vec = (d2 - (np.sum(d2 * d1, axis=1) / speed2)[:, None] * d1) / speed2[:, None]
    H_true = np.zeros((n_points, 3))
    H_true[:, 0] = k_vec[:, 0] * global_std
    H_true[:, 2] = k_vec[:, 1] * global_std

    pin = float(np.abs(np.linalg.norm(H_true, axis=1) - H_true_norm).max())
    if pin >= 1e-12:
        raise ValueError(
            f"run_cell: derived analytic H vector disagrees with the fixture's sealed "
            f"H_norm by {pin:.2e} (must be < 1e-12) -- the spiral algebra has drifted "
            "from the sealed reference derivation."
        )

    # Split rng seeded by the torch seed, coupling split and initialization exactly as
    # 02.5-09 did -- a stated limitation (02.5-09-SUMMARY.md), not silently changed here.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_points)
    n_train = int(0.8 * n_points)
    train_idx, holdout_idx = perm[:n_train], perm[n_train:]
    x_all = torch.tensor(X, dtype=torch.float32).to(device)
    # Index tensors are moved to `device` explicitly rather than relying on numpy-array
    # indexing to auto-place itself against a non-CPU tensor.
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    holdout_idx_t = torch.as_tensor(holdout_idx, dtype=torch.long, device=device)
    x_train, x_holdout = x_all[train_idx_t], x_all[holdout_idx_t]

    cfg = dict(BASE_CFG)
    if max_epochs is not None:
        cfg["max_epochs"] = max_epochs

    train0 = time.time()
    torch.manual_seed(seed)
    model = cae.ChartAutoEncoder(
        in_dim=3,
        embed_dim=EMBED_DIM,
        chart_dim=CHART_DIM,
        n_charts=n_charts,
        hidden=HIDDEN,
        activation="silu",
    )
    model = model.to(device)  # AFTER construction only -- never touches RNG consumption order
    cae.train_cae(model, x_train, {**cfg, "seed": seed, "n_charts": n_charts})
    model.eval()

    torch.manual_seed(seed)
    plain = cae.PlainAutoEncoder(3, CHART_DIM, hidden=tuple(HIDDEN), activation="silu")
    plain = plain.to(device)  # AFTER construction only -- same reasoning as `model` above
    cae.train_plain_ae(plain, x_train, dict(cfg))
    plain.eval()
    train_wall_s = time.time() - train0

    with torch.no_grad():
        y_hold = model.reconstruct(x_holdout)
        y_plain = plain(x_holdout)["y"]
    cae_stats = cae.reconstruction_stats(x_holdout.double(), y_hold.double())
    plain_stats = cae.reconstruction_stats(x_holdout.double(), y_plain.double())

    model.double()  # second derivatives are exactly where float32 noise shows first; runs
    # in float64 on every device -- chart_curvature._assert_float64 is never relaxed
    curv0 = time.time()
    field = chart_curvature.chart_curvature_field(model, x_all.double())
    curv_wall_s = time.time() - curv0
    activation = chart_curvature.assert_c2_activation(model)

    H_chart = field["H_vec"].cpu().numpy()
    h_chart = field["H_norm"].cpu().numpy()
    cond = field["metric_condition_number"].cpu().numpy()

    rho_chart = curvature_probe.spearman_gate_statistic(h_chart, H_true_norm)
    mre_chart = curvature_probe.median_relative_error(h_chart, H_true_norm)
    fid = chart_curvature.curvature_fidelity_report(H_chart, H_true)

    return {
        "n_charts_configured": int(n_charts),
        "n_charts_used": int(field["n_charts_used"]),
        "seed": int(seed),
        "n_points": int(n_points),
        "rho_chart": float(rho_chart),
        "mre_chart": float(mre_chart),
        "median_cosine_similarity": fid["median_cosine_similarity"],
        "median_magnitude_ratio": fid["median_magnitude_ratio"],
        "magnitude_ratio_cv": fid["magnitude_ratio_cv"],
        "calibration_slope": fid["calibration_slope"],
        "calibration_intercept": fid["calibration_intercept"],
        "calibration_r2": fid["calibration_r2"],
        "cond_median": float(np.median(cond)),
        "cond_max": float(np.max(cond)),
        "cae_mse_per_dim": cae_stats["mse_per_dim"],
        "plain_mse_per_dim": plain_stats["mse_per_dim"],
        "curvature_convention": field["curvature_convention"],
        "train_wall_s": float(train_wall_s),
        "curv_wall_s": float(curv_wall_s),
        "activation": activation,
        "device": str(device),
        "torch_version": torch.__version__,
    }


# =============================================================================================
# Summary layer (plan 03-02): full sweep table, median-of-5-seeds statistic, floor
# application to the best swept config, D-05a stop-and-report branch. This layer only
# reads the JSONL record back and prints -- it never writes a verdict artifact (D-15).
# =============================================================================================

SUMMARY_TABLE_COLUMNS: Tuple[str, ...] = (
    "n_charts_configured",
    "n_charts_used",
    "seed",
    "rho_chart",
    "median_cosine_similarity",
    "median_magnitude_ratio",
    "magnitude_ratio_cv",
    "calibration_slope",
    "calibration_r2",
    "cond_median",
    "cond_max",
    "cae_mse_per_dim",
    "plain_mse_per_dim",
    "curv_wall_s",
)
"""The four fidelity axes (direction/magnitude/calibration/rank) stay separate columns --
no composite score is ever formed. n_charts_used sits immediately beside
n_charts_configured because D-05's monotone finding is stated in charts actually used, and
a config that collapses to fewer live charts is a different measurement from the one
requested."""


def print_full_sweep_table(records: List[Dict[str, Any]]) -> None:
    """One row per (n_charts_configured, seed) pair -- the full sweep table, so the
    best-of-N is visible rather than hidden (D-04)."""
    header = " | ".join(SUMMARY_TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for rec in sorted(records, key=lambda r: (r["n_charts_configured"], r["seed"])):
        cells = []
        for col in SUMMARY_TABLE_COLUMNS:
            val = rec.get(col)
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        print(" | ".join(cells))


def per_config_summary(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One row per n_charts value: median rho_chart over its recorded seeds (D-01 -- the
    median is THE statistic), the min and max (the full spread), the per-seed values
    listed inline, and the median n_charts_used. Deliberately no mean anywhere in this
    function -- do not print a mean, and do not print a best-seed row as if it were the
    result."""
    by_config: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        by_config.setdefault(int(rec["n_charts_configured"]), []).append(rec)

    rows: List[Dict[str, Any]] = []
    for n_charts in sorted(by_config):
        cells = sorted(by_config[n_charts], key=lambda r: r["seed"])
        rhos = [float(c["rho_chart"]) for c in cells]
        used = [int(c["n_charts_used"]) for c in cells]
        rows.append(
            {
                "n_charts": n_charts,
                "n_cells": len(cells),
                "median_rho": statistics.median(rhos),
                "min_rho": min(rhos),
                "max_rho": max(rhos),
                "per_seed_rho": [(int(c["seed"]), float(c["rho_chart"])) for c in cells],
                "median_n_charts_used": statistics.median(used),
            }
        )
    return rows


def print_per_config_summary(rows: List[Dict[str, Any]]) -> None:
    print(
        "per-config summary (median rho_chart over recorded seeds is the D-01 statistic "
        "-- never a mean, never a best-seed row):"
    )
    for row in rows:
        per_seed = ", ".join(f"seed{s}={rho:.4f}" for s, rho in row["per_seed_rho"])
        print(
            f"  n_charts={row['n_charts']:<2d}  n_cells={row['n_cells']}  "
            f"median_rho_chart={row['median_rho']:.4f}  "
            f"spread=[{row['min_rho']:.4f}, {row['max_rho']:.4f}]  "
            f"median_n_charts_used={row['median_n_charts_used']:.1f}  "
            f"per-seed: {per_seed}"
        )


def print_context_baseline(records: List[Dict[str, Any]]) -> None:
    """The raw-point 0.6712 centroid baseline, printed as context that gates nothing
    (D-02) -- with the 02.5-09-SUMMARY.md section 3 caveat that it missed 02.5-05's own
    notebook's >0.90 sanity bar, so it is a baseline that works, not one validated as
    correct."""
    rho_raw = records[0].get("rho_raw_context") if records else None
    if rho_raw is None:
        print(
            f"context: raw-point centroid baseline RAW_BASELINE_CONTEXT={RAW_BASELINE_CONTEXT} "
            "-- CONTEXT ONLY, GATES NOTHING (no recorded cell to read it from yet)."
        )
        return
    print(
        f"context: raw-point centroid baseline rho = {rho_raw:.4f} "
        f"(RAW_BASELINE_CONTEXT = {RAW_BASELINE_CONTEXT}) -- CONTEXT ONLY, GATES NOTHING. "
        "It missed 02.5-05's own notebook's >0.90 sanity bar (02.5-09-SUMMARY.md section 3), "
        "so it is a baseline that works, not one validated as correct."
    )


def print_floor_decision(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """D-02/D-04: compare the BEST swept config's median against ROLL_FLOOR. Prints
    STEP-1 GATE: CLEARS or STEP-1 GATE: DOES NOT CLEAR, then unconditionally prints the
    multiple-comparisons caveat naming the exact number of configs swept -- N configs
    give N shots at a fixed bar, and the floor was applied to the best of them."""
    if not rows:
        print("STEP-1 GATE: no cells recorded yet -- nothing to decide.")
        return None

    best = max(rows, key=lambda r: r["median_rho"])
    cleared = best["median_rho"] > ROLL_FLOOR
    if cleared:
        print(
            f"STEP-1 GATE: CLEARS -- n_charts={best['n_charts']}  "
            f"median rho_chart={best['median_rho']:.4f}  "
            f"spread=[{best['min_rho']:.4f}, {best['max_rho']:.4f}]  "
            f"> ROLL_FLOOR={ROLL_FLOOR}"
        )
    else:
        print(
            f"STEP-1 GATE: DOES NOT CLEAR -- best observed config n_charts={best['n_charts']}  "
            f"median rho_chart={best['median_rho']:.4f}  "
            f"spread=[{best['min_rho']:.4f}, {best['max_rho']:.4f}]  "
            f"<= ROLL_FLOOR={ROLL_FLOOR}"
        )
    print(
        f"multiple-comparisons caveat: {len(rows)} n_charts config(s) were swept and "
        f"ROLL_FLOOR was applied to the best of them -- {len(rows)} configs give "
        f"{len(rows)} shots at a fixed bar, which loosens the effective bar relative to a "
        "single pre-committed config. Named here, not hidden."
    )
    return {"cleared": cleared, "best": best}


def print_d05a_branch() -> None:
    """D-05a's stop-and-report branch, printed only on a non-clear -- a complete phase
    outcome, not an error to work around."""
    print("=" * 92)
    print(
        "D-05a: the roll did not clear ROLL_FLOOR at any swept n_charts. Phase 3 stops and "
        "reports. This is a complete outcome, not a failure to work around. Two concrete "
        "follow-on actions:"
    )
    print(
        '  1. Re-point plan 03-11\'s depends_on to ["03-02", "03-03", "03-04"] (drop the '
        "steps 2-3 plans it no longer needs)."
    )
    print(
        "  2. Write a Step-1-only 03-FINDINGS.md from this SUMMARY, and do not execute "
        "plans 03-05 through 03-10."
    )
    print("=" * 92)


def summarize(record_path: Path) -> None:
    """Reads the JSONL record back and prints the full sweep table, the per-config
    median-and-spread rows, the context baseline, the floor decision with its
    unconditional multiple-comparisons caveat, and (on a non-clear) the D-05a branch.
    Never raises on a partial record -- reports how many of the planned cells are
    present and summarizes whatever is there."""
    records = list(load_completed(record_path).values())
    total_planned = len(N_CHARTS_SWEEP) * len(TORCH_SEEDS)

    print("=" * 92)
    print(f"STEP-1 SWEEP SUMMARY -- {len(records)} of {total_planned} planned cells present")
    print("=" * 92)

    if not records:
        print("No cells recorded yet -- nothing to summarize.")
        return

    print_full_sweep_table(records)
    print("-" * 92)
    rows = per_config_summary(records)
    print_per_config_summary(rows)
    print("-" * 92)
    print_context_baseline(records)
    print("-" * 92)
    outcome = print_floor_decision(rows)
    if outcome is not None and not outcome["cleared"]:
        print_d05a_branch()


# =============================================================================================
# CLI
# =============================================================================================


def build_grid(n_charts_values: Tuple[int, ...], seeds: Tuple[int, ...]):
    return [(n, s) for n in n_charts_values for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast path: one cheap cell (n_charts=2, seed=0, max_epochs=5, 300 points) "
            "purely to prove the wiring executes end to end, and prints a tally. Writes "
            "nothing to the record file -- a smoke key must never collide with the real "
            "sweep's own (n_charts, seed) resumability index."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned (n_charts, seed) grid and the declared floor, then exit "
        "without running or writing anything.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any (n_charts, seed) cell already present in the record file rather "
        "than recomputing it.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help="Override the append-only record path (default: "
        "notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl).",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Run at most this many cells this invocation, then stop.",
    )
    parser.add_argument(
        "--n-charts",
        type=str,
        default=None,
        help="Comma-separated override of N_CHARTS_SWEEP, e.g. '8'.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated override of TORCH_SEEDS, e.g. '0'.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Read the JSONL record back and print the full sweep table, the "
        "per-config median-and-spread rows, the context baseline, and the D-02/D-04 "
        "floor decision. Runs no new cells. Also printed automatically as the "
        "trailing step of a full (non-smoke, non-dry-run) invocation.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="03-02-AMENDMENT-01.md: override N_POINTS for the real sweep (e.g. 12000). "
        "Does not change raw_baseline_context(), which always measures against the "
        "original N_POINTS=3000 fixture. Without --record-path, routes the record to a "
        "distinct cache key (03_swiss_roll_curvature_sweep_n<N>.jsonl) so the sealed "
        "n=3000 grid is never clobbered or resumed-into.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="03-07-SUPPLEMENT-01: 'cpu' (default, zero behaviour change), 'cuda', or "
        "'cuda:N'. Fails with an actionable message if cuda is requested but unavailable. "
        "THREE CAVEATS: (1) GPU runs do NOT reproduce CPU runs bit-for-bit -- CUDA RNG "
        "differs from CPU RNG, so a GPU cell is a different draw, not a reproduction; every "
        "record carries its own 'device' field for exactly this reason. (2) float64 "
        "throughput is hardware-dependent -- curvature is float64-only, and on a consumer "
        "GPU float64 can run at 1/32-1/64 of float32 (vs ~1/2 on a data-center GPU), so the "
        "curvature term may be SLOWER on a consumer GPU than on CPU even though training "
        "should be faster. (3) Do not mix devices within one sweep -- every cell must share "
        "one device, or the seed spread mixes two different draws.",
    )
    args = parser.parse_args()

    n_charts_values = (
        tuple(int(v) for v in args.n_charts.split(",")) if args.n_charts else N_CHARTS_SWEEP
    )
    seeds = tuple(int(v) for v in args.seeds.split(",")) if args.seeds else TORCH_SEEDS
    n_points = args.n_points if args.n_points is not None else N_POINTS
    device = resolve_device(args.device)

    print("=" * 92)
    print("Phase 3 Plan 01: Swiss roll n_charts x seed curvature sweep runner")
    print(
        f"ROLL_FLOOR = {ROLL_FLOOR}  (D-02 absolute floor on median rho_chart; "
        f"RAW_BASELINE_CONTEXT = {RAW_BASELINE_CONTEXT} is context only, gates nothing)"
    )
    print(f"device={device}  torch_version={torch.__version__}")
    if device.type == "cuda":
        print(
            "CAVEAT: GPU runs do NOT reproduce CPU runs bit-for-bit (CUDA RNG != CPU RNG); "
            "this is a different draw, recorded as such via each record's 'device' field."
        )
    print("=" * 92)

    if args.dry_run:
        grid = build_grid(n_charts_values, seeds)
        print(
            f"planned cells: {len(grid)}  (n_charts={n_charts_values}, seeds={seeds}, "
            f"{len(n_charts_values)} n_charts values x {len(seeds)} seeds)"
        )
        for n_charts, seed in grid:
            print(f"  n_charts={n_charts}  seed={seed}")
        print(f"ROLL_FLOOR = {ROLL_FLOOR}")
        print("--dry-run: nothing executed, nothing written.")
        return

    if args.smoke:
        print("--smoke: n_charts=2 seed=0 max_epochs=5 n_points=300 (wiring check only)")
        record = run_cell(n_charts=2, seed=0, n_points=300, max_epochs=5, device=device)
        print(
            f"  n_charts_used={record['n_charts_used']}  rho_chart={record['rho_chart']:.4f}  "
            f"train_wall_s={record['train_wall_s']:.1f}  curv_wall_s={record['curv_wall_s']:.2f}  "
            f"activation={record['activation']!r}  "
            f"curvature_convention={record['curvature_convention']!r}"
        )
        print("--smoke tally: 1/1 cell completed. Nothing written to the record file.")
        return

    default_record_name = (
        "03_swiss_roll_curvature_sweep.jsonl"
        if n_points == N_POINTS
        else f"03_swiss_roll_curvature_sweep_n{n_points}.jsonl"
    )
    record_path = (
        Path(args.record_path) if args.record_path else (CACHE_DIR / default_record_name)
    )

    if args.summary:
        print(f"record_path={record_path}")
        summarize(record_path)
        return

    print(f"record_path={record_path}  n_points={n_points}")

    grid = build_grid(n_charts_values, seeds)
    print(f"cells in this invocation's scope: {len(grid)}")

    completed = load_completed(record_path) if args.resume else {}
    if args.resume:
        print(f"--resume: {len(completed)} cell(s) already on record, will be skipped.")
        for rec in completed.values():
            if int(rec.get("n_points", N_POINTS)) != n_points:
                raise ValueError(
                    f"--resume: record_path={record_path} already contains a cell recorded "
                    f"at n_points={rec.get('n_points')}, but this invocation requested "
                    f"n_points={n_points} -- a mid-sweep n_points change would silently make "
                    "two halves of the table incomparable (02.6-FINDINGS-02.md section 12's "
                    "defect pattern). Use a distinct --record-path instead."
                )
            # 03-07-SUPPLEMENT-01 caveat 3: never mix devices within one sweep -- a
            # resumed cell recorded on a different device would mix two different draws.
            rec_device = rec.get("device", "cpu")
            if rec_device != str(device):
                raise ValueError(
                    f"--resume: record_path={record_path} already contains a cell recorded "
                    f"at device={rec_device!r}, but this invocation requested "
                    f"device={str(device)!r} -- mixing devices within one sweep mixes two "
                    "different RNG draws (03-07-SUPPLEMENT-01 caveat 3). Use a distinct "
                    "--record-path instead."
                )

    rho_raw = raw_baseline_context()

    n_run = 0
    for n_charts, seed in grid:
        key = (n_charts, seed)
        if args.resume and key in completed:
            print(f"  [skip, resumed] n_charts={n_charts} seed={seed}")
            continue
        if args.max_combos is not None and n_run >= args.max_combos:
            remaining = len(grid) - n_run
            print(f"--max-combos={args.max_combos} reached; stopping ({remaining} cell(s) remain).")
            break

        t0 = time.monotonic()
        record = run_cell(n_charts=n_charts, seed=seed, n_points=n_points, device=device)
        record["rho_raw_context"] = rho_raw
        append_record(record_path, record)
        elapsed = time.monotonic() - t0
        print(
            f"  n_charts={n_charts} seed={seed}  n_charts_used={record['n_charts_used']}  "
            f"rho_chart={record['rho_chart']:.4f}  mre_chart={record['mre_chart']:.4f}  "
            f"cond_max={record['cond_max']:.2f}  wall={elapsed:.1f}s"
        )
        n_run += 1

    print(f"completed {n_run} cell(s) this invocation.")
    print()
    summarize(record_path)  # the default trailing step of a full run


if __name__ == "__main__":
    main()
