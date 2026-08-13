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
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
CHART_DIM = 2
EMBED_DIM = 8
HIDDEN = [64, 64]
K_BASELINE = 30

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
) -> Dict[str, Any]:
    """One (n_charts, seed) measurement, following
    ``02.5_swiss_roll_chart_curvature_check.ipynb``'s cell sequence exactly. ``n_points``
    and ``max_epochs`` are overridable only for ``--smoke``'s cheap wiring check; the real
    sweep always uses the module constants (``N_POINTS``, ``BASE_CFG["max_epochs"]``)."""
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
    x_all = torch.tensor(X, dtype=torch.float32)
    x_train, x_holdout = x_all[train_idx], x_all[holdout_idx]

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
    cae.train_cae(model, x_train, {**cfg, "seed": seed, "n_charts": n_charts})
    model.eval()

    torch.manual_seed(seed)
    plain = cae.PlainAutoEncoder(3, CHART_DIM, hidden=tuple(HIDDEN), activation="silu")
    cae.train_plain_ae(plain, x_train, dict(cfg))
    plain.eval()
    train_wall_s = time.time() - train0

    with torch.no_grad():
        y_hold = model.reconstruct(x_holdout)
        y_plain = plain(x_holdout)["y"]
    cae_stats = cae.reconstruction_stats(x_holdout.double(), y_hold.double())
    plain_stats = cae.reconstruction_stats(x_holdout.double(), y_plain.double())

    model.double()  # second derivatives are exactly where float32 noise shows first
    curv0 = time.time()
    field = chart_curvature.chart_curvature_field(model, x_all.double())
    curv_wall_s = time.time() - curv0
    activation = chart_curvature.assert_c2_activation(model)

    H_chart = field["H_vec"].numpy()
    h_chart = field["H_norm"].numpy()
    cond = field["metric_condition_number"].numpy()

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
    }


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
    args = parser.parse_args()

    n_charts_values = (
        tuple(int(v) for v in args.n_charts.split(",")) if args.n_charts else N_CHARTS_SWEEP
    )
    seeds = tuple(int(v) for v in args.seeds.split(",")) if args.seeds else TORCH_SEEDS

    print("=" * 92)
    print("Phase 3 Plan 01: Swiss roll n_charts x seed curvature sweep runner")
    print(
        f"ROLL_FLOOR = {ROLL_FLOOR}  (D-02 absolute floor on median rho_chart; "
        f"RAW_BASELINE_CONTEXT = {RAW_BASELINE_CONTEXT} is context only, gates nothing)"
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
        record = run_cell(n_charts=2, seed=0, n_points=300, max_epochs=5)
        print(
            f"  n_charts_used={record['n_charts_used']}  rho_chart={record['rho_chart']:.4f}  "
            f"train_wall_s={record['train_wall_s']:.1f}  curv_wall_s={record['curv_wall_s']:.2f}  "
            f"activation={record['activation']!r}  "
            f"curvature_convention={record['curvature_convention']!r}"
        )
        print("--smoke tally: 1/1 cell completed. Nothing written to the record file.")
        return

    record_path = (
        Path(args.record_path)
        if args.record_path
        else (CACHE_DIR / "03_swiss_roll_curvature_sweep.jsonl")
    )
    print(f"record_path={record_path}")

    grid = build_grid(n_charts_values, seeds)
    print(f"cells in this invocation's scope: {len(grid)}")

    completed = load_completed(record_path) if args.resume else {}
    if args.resume:
        print(f"--resume: {len(completed)} cell(s) already on record, will be skipped.")

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
        record = run_cell(n_charts=n_charts, seed=seed)
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


if __name__ == "__main__":
    main()
