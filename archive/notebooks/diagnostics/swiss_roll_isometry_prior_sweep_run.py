"""Quick task 260815-e1t Part A: Swiss roll isometry-prior weight-ladder runner.

Spikes the isometry prior (``pu_manifold.decoder_priors``) on the CAE chart decoder, on the
Swiss roll only, before any PU compute is spent on it. Copies the shape, the fixture, the
protocol and the metric call sequence from
``notebooks/diagnostics/swiss_roll_curvature_sweep_run.py`` -- that file is NOT edited, and its
constants (``FIXTURE_SEED``, ``CHART_DIM``, ``EMBED_DIM``, ``HIDDEN``, ``BASE_CFG``,
``N_POINTS``, ``N_POINTS_AMENDED``, ``ROLL_FLOOR``) are imported for reuse rather than restated.

**The Swiss roll anchor is this runner's first gate, checked with ``==`` not a tolerance
(``--anchor-check``).** ``rho_chart = -0.06041003026778113`` at ``n_charts=8, seed=0,
n_points=3000`` must reproduce exactly through this new code path -- proof that this runner's
fixture, protocol and metric call sequence are a faithful copy and not a re-derivation.

Invoke:
    .venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --dry-run
    .venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --anchor-check
    .venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --probe
    .venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --max-epochs 100 --resume
    .venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --max-epochs 100 --mode conformal --resume
    .venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --summary
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import swiss_roll_curvature_sweep_run as sweep  # noqa: E402

from pu_manifold import cae, chart_curvature, curvature_probe, decoder_priors  # noqa: E402
from pu_manifold.cache import CACHE_DIR  # noqa: E402

# =============================================================================================
# The ladder is declared in source, before any number exists (same idiom as sweep.ROLL_FLOOR).
# =============================================================================================

LADDER_N_CHARTS = 8
LADDER_N_POINTS = sweep.N_POINTS_AMENDED  # 12000, 03-02-AMENDMENT-01.md's amended point count

LADDER_SEEDS: Tuple[int, ...] = (0, 3)
"""Chosen from the recorded n=12000 baseline (measured fact 4): seed 0 is the weakest baseline
of the five (rho_chart = 0.2877) and seed 3 is the median (0.8234). Spanning the range, rather
than picking the two ceiling seeds (1 and 2, at 0.9377 and 0.9743, where no improvement is
measurable), is a deliberate choice made before the ladder ran."""

LADDER_WEIGHTS: Tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1)

LADDER_MODE_DEFAULT = "isometry"
"""The developer-specified primary. ``cae.ChartEncoder`` ends in a sigmoid, so chart
coordinates live in the open unit cube (0,1)^d, and a decoder whose domain has extent 1 and
whose image must span a manifold of extent L needs ||J|| ~ L, i.e. g ~ L^2 I, not g = I. The
strict isometry penalty therefore contains a scale term fighting reconstruction directly, while
cond(g) is scale-invariant and is what the conformal variant targets. If the isometry arm fails
its mechanism check, the conformal arm is the PRE-DECLARED branch, not a post-hoc axis switch
(measured fact 5)."""

LADDER_MAX_EPOCHS_CANDIDATES: Tuple[int, ...] = (150, 100, 75, 50)
LADDER_BUDGET_S = 3000

# Early stopping is disabled for every ladder cell -- see run_cell's disable_early_stopping
# docstring for why (03-08-DEFECTS-01.md defect 2 in a new costume, headed off by construction).

RECORD_NAME = "quick_isometry_prior_sweep.jsonl"
ANCHOR_RHO_CHART = -0.06041003026778113


# =============================================================================================
# JSON-lines append-only record: resumability, same idiom as swiss_roll_curvature_sweep_run.py.
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


def cell_key(mode: str, weight: float, seed: int) -> Tuple[str, float, int]:
    """``(mode, weight, seed)`` -- except at ``weight == 0.0``, where the mode string is
    replaced by the sentinel ``"any"``. ``decoder_prior_active`` installs nothing at all at
    weight 0.0 regardless of which mode string is passed (the module-docstring guarantee), so a
    weight=0.0 cell recorded under one mode is valid, bit-identical evidence for every other
    mode too -- the conformal arm therefore never needs to re-run its own weight=0.0 cell."""
    weight = float(weight)
    mode_key = "any" if weight == 0.0 else mode
    return (mode_key, weight, int(seed))


def load_completed(record_path: Path) -> Dict[Tuple[str, float, int], Dict[str, Any]]:
    """Reads every JSON-lines record already on disk, keyed by :func:`cell_key`. Missing file
    -> empty dict (a fresh run, not an error)."""
    completed: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    if not record_path.exists():
        return completed
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = cell_key(rec["mode"], rec["weight"], rec["seed"])
            completed[key] = rec
    return completed


def append_record(record_path: Path, record: Dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as f:
        f.write(json.dumps(_to_jsonable(record)) + "\n")


# =============================================================================================
# Per-cell measurement: fit (with the prior optionally active) -> chart decoder -> torch.func
# curvature -> Spearman vs analytic H. Follows sweep.run_cell's sequence exactly with three
# differences: (a) the fit is wrapped in decoder_priors.decoder_prior_active; (b) no plain
# autoencoder is fitted; (c) cfg carries the chosen max_epochs and (by default) disabled early
# stopping.
# =============================================================================================


def run_cell(
    weight: float,
    seed: int,
    mode: str,
    max_epochs: int,
    *,
    n_charts: int = LADDER_N_CHARTS,
    n_points: int = LADDER_N_POINTS,
    device: torch.device = torch.device("cpu"),
    disable_early_stopping: bool = True,
) -> Dict[str, Any]:
    """One ``(weight, seed, mode)`` measurement. ``disable_early_stopping=True`` is the ladder
    protocol; ``--anchor-check`` passes ``False`` so that cell reproduces the sealed protocol
    (``sweep.BASE_CFG`` verbatim, early stopping as configured there), not the ladder protocol.
    """
    fx = curvature_probe.make_swiss_roll_fixture(n=n_points, seed=sweep.FIXTURE_SEED)
    X, t, H_true_norm, global_std = fx["X"], fx["t"], fx["H_norm"], fx["global_std"]

    # Analytic mean curvature VECTOR, copied VERBATIM from
    # swiss_roll_curvature_sweep_run.run_cell (lines 252-268), including its pin < 1e-12
    # assertion against the fixture's sealed H_norm -- that assertion is what makes this copy
    # safe: it fails loudly if the copy drifts from the sealed derivation, rather than silently
    # producing a plausible-looking but wrong H_true.
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

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_points)
    n_train = int(0.8 * n_points)
    train_idx, holdout_idx = perm[:n_train], perm[n_train:]
    x_all = torch.tensor(X, dtype=torch.float32).to(device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    holdout_idx_t = torch.as_tensor(holdout_idx, dtype=torch.long, device=device)
    x_train, x_holdout = x_all[train_idx_t], x_all[holdout_idx_t]

    cfg = dict(sweep.BASE_CFG)
    cfg["max_epochs"] = max_epochs
    if disable_early_stopping:
        # train_cae early-stops on TOTAL loss, and the prior is added to total, so leaving
        # early stopping on would let the prior weight change the training length and confound
        # every column of the table -- 03-08-DEFECTS-01.md defect 2 in a new costume, headed
        # off here by construction rather than caught afterwards.
        cfg["early_stop_patience"] = max_epochs + 1

    train0 = time.time()
    torch.manual_seed(seed)
    model = cae.ChartAutoEncoder(
        in_dim=3,
        embed_dim=sweep.EMBED_DIM,
        chart_dim=sweep.CHART_DIM,
        n_charts=n_charts,
        hidden=sweep.HIDDEN,
        activation="silu",
    )
    model = model.to(device)  # AFTER construction only -- never touches RNG consumption order

    # (a) the CAE fit is wrapped in decoder_priors.decoder_prior_active: at weight=0.0 this
    # installs nothing at all, so the weight=0 arm is structurally the untouched cae.train_cae
    # code path, not merely a numerically-zero one.
    with decoder_priors.decoder_prior_active(model, weight, mode):
        fit = cae.train_cae(model, x_train, {**cfg, "seed": seed, "n_charts": n_charts})
    model.eval()
    train_wall_s = time.time() - train0

    # (b) no plain autoencoder is fitted -- Part B retires that comparison, and the weight=0
    # arm is the matched control: identical architecture, identical protocol, identical seeds,
    # a strictly better match than a plain AE ever was. Measured fact 6 records why omitting the
    # plain AE leaves rho_chart bit-identical: the CAE's own parameters are fully determined by
    # its own training and never depend on whether a plain AE is later constructed.

    with torch.no_grad():
        y_hold = model.reconstruct(x_holdout)
    cae_stats = cae.reconstruction_stats(x_holdout.double(), y_hold.double())

    with torch.no_grad():
        final_penalty_value = float(
            decoder_priors.isometry_penalty(model, x_train, weight, mode).item()
        )

    model.double()  # second derivatives are exactly where float32 noise shows first
    curv0 = time.time()
    field = chart_curvature.chart_curvature_field(model, x_all.double())
    curv_wall_s = time.time() - curv0

    H_chart = field["H_vec"].cpu().numpy()
    h_chart = field["H_norm"].cpu().numpy()
    cond = field["metric_condition_number"].cpu().numpy()

    rho_chart = curvature_probe.spearman_gate_statistic(h_chart, H_true_norm)
    mre_chart = curvature_probe.median_relative_error(h_chart, H_true_norm)
    fid = chart_curvature.curvature_fidelity_report(H_chart, H_true)

    return {
        "weight": float(weight),
        "mode": mode,
        "seed": int(seed),
        "n_charts_configured": int(n_charts),
        "n_charts_used": int(field["n_charts_used"]),
        "n_points": int(n_points),
        "max_epochs": int(max_epochs),
        "epochs_run": int(fit["epochs_run"]),
        "early_stopped": bool(fit["early_stopped"]),
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
        "final_penalty_value": final_penalty_value,
        "train_wall_s": float(train_wall_s),
        "curv_wall_s": float(curv_wall_s),
        "device": str(device),
        "torch_version": torch.__version__,
    }


# =============================================================================================
# --anchor-check: the faithfulness gate. == , not a tolerance.
# =============================================================================================


def anchor_check(device: torch.device) -> None:
    print(
        "--anchor-check: weight=0.0, n_charts=8, seed=0, n_points=3000, sweep.BASE_CFG "
        "verbatim (early stopping as configured there, not disabled -- this cell reproduces "
        "the sealed protocol, not the ladder protocol). Writes nothing."
    )
    record = run_cell(
        weight=0.0,
        seed=0,
        mode=LADDER_MODE_DEFAULT,
        max_epochs=sweep.BASE_CFG["max_epochs"],
        n_charts=8,
        n_points=sweep.N_POINTS,
        device=device,
        disable_early_stopping=False,
    )
    measured = record["rho_chart"]
    print(f"measured rho_chart = {measured!r}")
    if measured != ANCHOR_RHO_CHART:
        raise ValueError(
            f"--anchor-check: measured rho_chart={measured!r} != expected "
            f"{ANCHOR_RHO_CHART!r}. This is a FAITHFULNESS FAILURE, not a tolerance "
            "question -- this runner's fixture, protocol or metric call sequence has "
            "drifted from swiss_roll_curvature_sweep_run.py."
        )
    print(f"ANCHOR OK: rho_chart == {ANCHOR_RHO_CHART!r} exactly.")


# =============================================================================================
# --probe: the budget probe, in the timing_probe idiom this repo already uses.
# =============================================================================================


def probe(device: torch.device) -> Optional[int]:
    seed = LADDER_SEEDS[0]
    print(
        f"--probe: two max_epochs=10 cells at n_charts={LADDER_N_CHARTS} "
        f"n_points={LADDER_N_POINTS} seed={seed} -- one at weight=0.0, one at "
        f"weight={LADDER_WEIGHTS[-1]}. Writes nothing."
    )
    rec0 = run_cell(
        weight=0.0, seed=seed, mode=LADDER_MODE_DEFAULT, max_epochs=10, device=device
    )
    rec1 = run_cell(
        weight=LADDER_WEIGHTS[-1], seed=seed, mode=LADDER_MODE_DEFAULT, max_epochs=10, device=device
    )

    t0 = rec0["train_wall_s"] / 10.0
    t1 = rec1["train_wall_s"] / 10.0
    c = (rec0["curv_wall_s"] + rec1["curv_wall_s"]) / 2.0

    print(
        f"measured: t0={t0:.3f}s/epoch (weight=0.0)  t1={t1:.3f}s/epoch "
        f"(weight={LADDER_WEIGHTS[-1]})  c={c:.3f}s/cell (curvature, does not scale with epochs)"
    )

    # 8 cells total in the ladder scope (2 seeds x 4 weights): 2 at weight=0.0, 6 at weight>0.0.
    projections = []
    for max_e in LADDER_MAX_EPOCHS_CANDIDATES:
        projected = 8 * c + max_e * (2 * t0 + 6 * t1)
        projections.append((max_e, projected))
        print(f"  E={max_e:<4d}  projected_total={projected:.1f}s")

    chosen = None
    for max_e, projected in projections:
        if projected <= LADDER_BUDGET_S and (chosen is None or max_e > chosen):
            chosen = max_e

    if chosen is None:
        print("BUDGET NOT MET -- even the smallest candidate exceeds LADDER_BUDGET_S:")
        for max_e, projected in projections:
            print(f"  E={max_e:<4d}  projected_total={projected:.1f}s  (budget={LADDER_BUDGET_S}s)")
        sys.exit(1)

    print(
        f"chosen LADDER_MAX_EPOCHS = {chosen}  (largest candidate at or under "
        f"LADDER_BUDGET_S={LADDER_BUDGET_S}s)"
    )
    return chosen


# =============================================================================================
# --summary: read the record back and print, with no composite score anywhere.
# =============================================================================================

FULL_TABLE_COLUMNS: Tuple[str, ...] = (
    "mode",
    "weight",
    "seed",
    "n_charts_used",
    "max_epochs",
    "epochs_run",
    "early_stopped",
    "rho_chart",
    "median_magnitude_ratio",
    "magnitude_ratio_cv",
    "calibration_slope",
    "cond_median",
    "cond_max",
    "cae_mse_per_dim",
    "final_penalty_value",
)

PER_WEIGHT_COLUMNS: Tuple[str, ...] = (
    "rho_chart",
    "median_magnitude_ratio",
    "calibration_slope",
    "cond_median",
    "cond_max",
    "cae_mse_per_dim",
)


def print_full_table(records: List[Dict[str, Any]]) -> None:
    header = " | ".join(FULL_TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for rec in sorted(records, key=lambda r: (r["mode"], r["weight"], r["seed"])):
        cells = []
        for col in FULL_TABLE_COLUMNS:
            val = rec.get(col)
            if isinstance(val, float):
                cells.append(f"{val:.6g}")
            else:
                cells.append(str(val))
        print(" | ".join(cells))


def records_for_mode(records: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """This mode's own weight>0 cells, plus the mode-independent weight=0.0 cell recorded
    under whichever mode first ran it -- see :func:`cell_key`."""
    return [r for r in records if float(r["weight"]) == 0.0 or r["mode"] == mode]


def per_weight_summary(mode_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_weight: Dict[float, List[Dict[str, Any]]] = {}
    for rec in mode_records:
        by_weight.setdefault(float(rec["weight"]), []).append(rec)

    rows = []
    for weight in sorted(by_weight):
        cells = sorted(by_weight[weight], key=lambda r: r["seed"])
        row: Dict[str, Any] = {"weight": weight, "n_cells": len(cells)}
        for col in PER_WEIGHT_COLUMNS:
            vals = [float(c[col]) for c in cells]
            row[col] = statistics.median(vals)
            row[f"{col}_per_seed"] = list(zip([int(c["seed"]) for c in cells], vals))
        rows.append(row)
    return rows


def print_per_weight_summary(rows: List[Dict[str, Any]]) -> None:
    print(
        "per-weight seed-median summary -- with two seeds the median is the MIDPOINT of the "
        "pair, not a robust statistic; stated here so a reader does not assume otherwise:"
    )
    for row in rows:
        print(f"  weight={row['weight']}  n_cells={row['n_cells']}")
        for col in PER_WEIGHT_COLUMNS:
            per_seed = ", ".join(f"seed{s}={v:.6g}" for s, v in row[f"{col}_per_seed"])
            print(f"    {col}: median={row[col]:.6g}  per-seed: {per_seed}")


def check_epochs_uniform(mode_records: List[Dict[str, Any]]) -> bool:
    bad = [
        r
        for r in mode_records
        if int(r["epochs_run"]) != int(r["max_epochs"]) or bool(r["early_stopped"])
    ]
    if bad:
        print(
            "TRAINING-LENGTH CONFOUND -- the following cells did not run for the full "
            "declared max_epochs, or early-stopped:"
        )
        for r in bad:
            print(
                f"  mode={r['mode']} weight={r['weight']} seed={r['seed']}  "
                f"epochs_run={r['epochs_run']}  max_epochs={r['max_epochs']}  "
                f"early_stopped={r['early_stopped']}"
            )
        print("refusing to print the read-outs below.")
        return False
    print("epochs_run uniformity: OK -- every cell has epochs_run == max_epochs and early_stopped == False.")
    return True


def mechanism_check(rows: List[Dict[str, Any]], mode: str) -> bool:
    weights_sorted = sorted(r["weight"] for r in rows)
    cond_vals = [next(r["cond_median"] for r in rows if r["weight"] == w) for w in weights_sorted]

    non_increasing = all(cond_vals[i] >= cond_vals[i + 1] for i in range(len(cond_vals) - 1))
    strictly_lower_at_max = cond_vals[-1] < cond_vals[0]
    demonstrated = non_increasing and strictly_lower_at_max

    verdict = "MECHANISM DEMONSTRATED" if demonstrated else "MECHANISM NOT DEMONSTRATED"
    print(f"MECHANISM CHECK ({mode}): {verdict}")
    for w, v in zip(weights_sorted, cond_vals):
        print(f"  weight={w}  cond_median={v:.6g}")
    if not demonstrated:
        print(
            "  the penalty is not doing what it claims; no other column below is read as "
            "evidence about anything."
        )
    return demonstrated


def bias_check(rows: List[Dict[str, Any]], mode: str) -> str:
    weights_sorted = sorted(r["weight"] for r in rows)
    rho = {r["weight"]: r["rho_chart"] for r in rows}
    mag = {r["weight"]: r["median_magnitude_ratio"] for r in rows}
    cal = {r["weight"]: r["calibration_slope"] for r in rows}

    rho_vals = [rho[w] for w in weights_sorted]
    mag_vals = [mag[w] for w in weights_sorted]

    rho_nondecreasing = all(rho_vals[i] <= rho_vals[i + 1] for i in range(len(rho_vals) - 1))
    mag_nonincreasing = all(mag_vals[i] >= mag_vals[i + 1] for i in range(len(mag_vals) - 1))
    mag_below_90_at_max = mag_vals[-1] < 0.90
    flattening = rho_nondecreasing and mag_nonincreasing and mag_below_90_at_max

    w0 = weights_sorted[0]
    w_star = None
    if not flattening:
        for w in weights_sorted:
            if rho[w] > rho[w0] and abs(mag[w] - 1.0) <= 0.10 and abs(cal[w] - 1.0) <= 0.20:
                w_star = w
                break

    if flattening:
        verdict = "FLATTENING SUSPECTED"
    elif w_star is not None:
        verdict = "CLEAN IMPROVEMENT"
    else:
        verdict = "NO EFFECT"

    print(f"BIAS CHECK ({mode}): {verdict}")
    if verdict == "FLATTENING SUSPECTED":
        print(
            "  rank correlation improved while scale collapsed -- this is the signature of a "
            "prior that is flattening the surface it is meant to leave alone. Adoption is NOT "
            "recommended."
        )
    elif verdict == "CLEAN IMPROVEMENT":
        print(
            f"  w*={w_star}: rho_chart above weight=0, |median_magnitude_ratio - 1| <= 0.10, "
            "|calibration_slope - 1| <= 0.20."
        )
    print("  ladder (full, printed exactly as prominently regardless of which verdict fired):")
    for w in weights_sorted:
        print(
            f"    weight={w}  rho_chart={rho[w]:.6g}  median_magnitude_ratio={mag[w]:.6g}  "
            f"calibration_slope={cal[w]:.6g}"
        )
    return verdict


def print_caveat() -> None:
    print(
        "caveat: two seeds cannot separate a small effect from seed noise, and the roll's "
        "cond(g) is already benign (1.4-8.3) -- the roll can demonstrate the mechanism and the "
        "absence of bias but CANNOT demonstrate a fix at the PU pathology of 4.9e7."
    )


def summarize(record_path: Path) -> None:
    records = list(load_completed(record_path).values())
    print("=" * 92)
    print(f"ISOMETRY PRIOR SWEEP SUMMARY -- {len(records)} cell(s) on record at {record_path}")
    print("=" * 92)

    if not records:
        print("No cells recorded yet -- nothing to summarize.")
        return

    print_full_table(records)
    print("-" * 92)

    modes_present = sorted({r["mode"] for r in records if float(r["weight"]) != 0.0})
    if not modes_present:
        modes_present = sorted({r["mode"] for r in records})

    for mode in modes_present:
        mode_records = records_for_mode(records, mode)
        rows = per_weight_summary(mode_records)
        print_per_weight_summary(rows)
        print("-" * 92)
        uniform = check_epochs_uniform(mode_records)
        print("-" * 92)
        if not uniform:
            continue
        mechanism_check(rows, mode)
        print("-" * 92)
        bias_check(rows, mode)
        print("-" * 92)

    print_caveat()


# =============================================================================================
# CLI
# =============================================================================================


def _print_dry_run() -> None:
    grid = [(w, s) for w in LADDER_WEIGHTS for s in LADDER_SEEDS]
    print(
        f"LADDER_N_CHARTS={LADDER_N_CHARTS}  LADDER_N_POINTS={LADDER_N_POINTS}  "
        f"LADDER_SEEDS={LADDER_SEEDS}  LADDER_WEIGHTS={LADDER_WEIGHTS}  "
        f"LADDER_MODE_DEFAULT={LADDER_MODE_DEFAULT!r}"
    )
    print(
        f"LADDER_MAX_EPOCHS_CANDIDATES={LADDER_MAX_EPOCHS_CANDIDATES}  "
        f"LADDER_BUDGET_S={LADDER_BUDGET_S}  early stopping disabled for every ladder cell"
    )
    print(f"planned cells ({LADDER_MODE_DEFAULT}): {len(grid)}")
    for w, s in grid:
        print(f"  mode={LADDER_MODE_DEFAULT}  weight={w}  seed={s}")
    print(
        "read-out rules: MECHANISM CHECK (cond_median strictly lower at the largest weight "
        "than at weight=0, non-increasing across the ladder -> MECHANISM DEMONSTRATED / "
        "MECHANISM NOT DEMONSTRATED); BIAS CHECK (exactly one of FLATTENING SUSPECTED / "
        "CLEAN IMPROVEMENT / NO EFFECT, printed with the full ladder underneath)."
    )
    print("--dry-run: nothing executed, nothing written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned ladder grid and the declared read-out rules. Runs nothing, writes nothing.",
    )
    parser.add_argument(
        "--anchor-check", action="store_true",
        help="Faithfulness gate: reproduce rho_chart == -0.06041003026778113 exactly at "
        "weight=0.0, n_charts=8, seed=0, n_points=3000, sweep.BASE_CFG verbatim. Writes nothing.",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Budget probe: two max_epochs=10 cells project the wall-clock cost of each "
        "LADDER_MAX_EPOCHS_CANDIDATES value and print the chosen LADDER_MAX_EPOCHS. Writes nothing.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Read the record back and print the full table, per-weight medians, the "
        "epochs_run uniformity assertion, the mechanism check and the bias check. Runs no new cells.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip any (mode, weight, seed) cell already present in the record file. "
        "weight=0.0 cells are always skipped if already present under any mode, "
        "regardless of this flag (mode-independent -- see cell_key).",
    )
    parser.add_argument("--record-path", type=str, default=None)
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="LADDER_MAX_EPOCHS chosen by --probe; required for a real ladder run.",
    )
    parser.add_argument(
        "--mode", type=str, default=LADDER_MODE_DEFAULT, choices=list(decoder_priors.PRIOR_MODES),
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Comma-separated override of LADDER_WEIGHTS.",
    )
    parser.add_argument(
        "--seeds", type=str, default=None, help="Comma-separated override of LADDER_SEEDS.",
    )
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = sweep.resolve_device(args.device)

    print("=" * 92)
    print("Quick task 260815-e1t: Swiss roll isometry-prior weight-ladder runner")
    print(f"device={device}  torch_version={torch.__version__}")
    print("=" * 92)

    if args.dry_run:
        _print_dry_run()
        return

    if args.anchor_check:
        anchor_check(device)
        return

    if args.probe:
        probe(device)
        return

    record_path = (
        Path(args.record_path) if args.record_path else (CACHE_DIR / RECORD_NAME)
    )

    if args.summary:
        print(f"record_path={record_path}")
        summarize(record_path)
        return

    if args.max_epochs is None:
        raise SystemExit(
            "--max-epochs is required for a real ladder run -- run --probe first to choose "
            "LADDER_MAX_EPOCHS, then pass it explicitly with --max-epochs."
        )

    weights = (
        tuple(float(v) for v in args.weights.split(",")) if args.weights else LADDER_WEIGHTS
    )
    seeds = tuple(int(v) for v in args.seeds.split(",")) if args.seeds else LADDER_SEEDS
    mode = args.mode

    grid = [(w, s) for w in weights for s in seeds]
    print(
        f"record_path={record_path}  mode={mode}  max_epochs={args.max_epochs}  "
        f"cells in scope: {len(grid)}"
    )

    completed = load_completed(record_path)
    if args.resume:
        print(f"--resume: {len(completed)} cell(s) already on record.")

    n_run = 0
    for weight, seed in grid:
        key = cell_key(mode, weight, seed)
        if key in completed and (args.resume or float(weight) == 0.0):
            reason = "resumed" if args.resume and float(weight) != 0.0 else "mode-independent weight=0.0 already recorded"
            print(f"  [skip, {reason}] mode={mode} weight={weight} seed={seed}")
            continue
        if args.max_combos is not None and n_run >= args.max_combos:
            remaining = len(grid) - n_run
            print(f"--max-combos={args.max_combos} reached; stopping ({remaining} cell(s) remain).")
            break

        t0 = time.monotonic()
        record = run_cell(
            weight=weight, seed=seed, mode=mode, max_epochs=args.max_epochs,
            n_charts=LADDER_N_CHARTS, n_points=LADDER_N_POINTS, device=device,
        )
        append_record(record_path, record)
        elapsed = time.monotonic() - t0
        print(
            f"  mode={mode} weight={weight} seed={seed}  epochs_run={record['epochs_run']}  "
            f"early_stopped={record['early_stopped']}  rho_chart={record['rho_chart']:.4f}  "
            f"cond_median={record['cond_median']:.2f}  wall={elapsed:.1f}s"
        )
        n_run += 1

    print(f"completed {n_run} cell(s) this invocation.")
    print()
    summarize(record_path)


if __name__ == "__main__":
    main()
