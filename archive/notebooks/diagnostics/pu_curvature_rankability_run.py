"""Does the PU manifold have RANKABLE curvature at all? A pre-Phase-4 gate.

**Why this must run before Phase 4.** Phase 4 partitions the manifold by `|H|` quantiles, so
it consumes the curvature field's ORDERING. Spike 003 established that a `rho ~ 0` against a
known answer has two readings that no single measurement separates -- the estimator failed, or
the surface had no rankable curvature in it. Every sealed control turned out to be the second
case. **That reading has never been checked for the PU embedding itself.** If the PU manifold
is near-constant-curvature, Phase 4's quantile partition is binning noise however good the
estimator is, and no amount of decoder work fixes it.

**Two diagnostics, neither of which needs a trained model.**

1. **Dynamic range** of the estimated `||H||` (`p95/p05`). Calibration from spike 003 at
   `d=20`, on fixtures whose answer is known:

       quadratic_bowl   ~1.4x    near-constant curvature, `rho ~ +0.03` -- UNRANKABLE
       cubic / ridge   ~28-34x   rankable, `rho +0.41 .. +0.61`

   This is a range of the ESTIMATE, not of the truth, so it is an upper bound on what is
   there: an estimator cannot invent spread it does not see, but noise CAN manufacture it,
   which is why diagnostic 2 is required alongside.

2. **Split-half reliability** `R_H` from `cross_split_curvature`: estimate the field twice from
   two DISJOINT halves of the cloud and take `2<H_A,H_B> / (||H_A||^2 + ||H_B||^2)` per point.
   Noise-manufactured spread does not reproduce across halves; real structure does. Both
   halves are evaluated at the SAME anchor points, drawn from neither half, so the two
   estimates are independent.

Neither is a PASS/FAIL gate on its own and this runner declares no verdict. It reports two
numbers whose meaning is calibrated against fixtures with known answers.

**Estimator choice.** `centroid_mean_curvature`, D-05's gating estimator, applied DIRECTLY to
the point cloud -- no decoder, no training. Spike 003 measured it at `rho = +0.61` at `d=20`
where the local quadric teacher scored `-0.02`; the quadric needs `d(d+1)/2 = 210` coefficients
per normal direction and is hopeless here.

    python notebooks/diagnostics/pu_curvature_rankability_run.py --smoke
    python notebooks/diagnostics/pu_curvature_rankability_run.py --k 30 60 120 231
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pu_manifold import cache
from pu_manifold import curvature_probe
from pu_manifold import cross_split_curvature as csc

DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "03.2_pu_curvature_rankability.jsonl"

# Calibration from spike 003, d=20, k=231 -- fixtures with known answers.
# REPORTED FOR CONTEXT, NOT GATED ON: the spike's own sweep measured rho +0.48 at a spread of
# 1.1x and +0.36 at 36x, so dynamic range does not predict rankability. What separates the
# rankable fixtures from the unrankable ones is a VARYING second fundamental form, not spread.
CALIBRATION = {
    "unrankable_bowl_spread": 1.4,
    "rankable_cubic_spread": 28.2,
    "rankable_ridge_spread": 34.3,
    "bowl_rho": 0.0302,
    "cubic_rho": 0.6115,
    "ridge_rho": 0.4119,
}

# D4-07's k-freeze rule -- a PRE-REGISTRATION, committed before any density-corrected R_H
# value has been measured (04-02-PLAN.md Task 1/Task 2 ordering). Chosen specifically because
# Phase 1's plateau rule for `k*=15` failed on uneven spacing: WINDOWS.md records that
# `STAGE2_K` was unevenly spaced (gaps 5, 5, 15), so that plateau was maximal in *index*
# space, not `k` space. An absolute increment compares median_R_H(k_i) against
# median_R_H(k_{i-1}) directly and never compares gaps across unevenly spaced grid points, so
# it is immune to that defect regardless of how the sweep's k values are spaced.
K_FREEZE_RULE = (
    "D4-07: freeze the curvature-field k at the smallest k in the ordered sweep grid whose "
    "median_R_H gain over the immediately preceding sweep point is strictly less than 0.03 "
    "AND whose median_R_H is greater than or equal to 0.5. The rule is evaluated from the "
    "SECOND sweep point onward, because the gain at the first point is undefined. If no k in "
    "the grid satisfies both conditions, the frozen k is the largest k actually run and the "
    "outcome is recorded as not-fired -- never adjusted post hoc."
)


def load_pu(column: str) -> Tuple[np.ndarray, str]:
    """The frozen Phase 1 10k subsample. Read-only; this runner never writes to the cache
    except its own record. Returns (X, subsample_file) so callers can echo provenance into
    every record without re-deriving the cache glob."""
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
    if not cands:
        raise FileNotFoundError("no subsample_*.npz in notebooks/.cache/")
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if column in z.files and z[column].shape[0] > best_n:
                best, best_n = c, z[column].shape[0]
    if best is None:
        raise KeyError(f"no cached subsample carries column {column!r}")
    with np.load(best) as z:
        X = np.asarray(z[column], dtype=np.float64)
    subsample_file = Path(best).name
    print(f"loaded {column} {X.shape} from {subsample_file}")
    return X, subsample_file


def measure_r_over_R(X: np.ndarray, k: int) -> Tuple[float, float, float]:
    """The locality statistic spike 003 reports alongside every k regime
    (`.claude/skills/spike-findings-effdim/sources/001-teacher-low-d-anchor/run_anchor.py`'s
    `measure_r_over_R`, reproduced verbatim here): median distance to the k-th nearest
    neighbour, divided by the median distance from the cloud centroid. Spike 003's own
    reference values are r/R = 1.0331 at k=231 and 1.0992 at k=500 -- at r/R above 1 the
    neighbourhood has grown past the cloud's own radius and is no longer local. Reported for
    every k in this runner, never gated on.

    Returns (r_knn, R_cloud, r_over_R).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    r_knn = float(np.median(dist[:, -1]))
    R_cloud = float(np.median(np.linalg.norm(X - X.mean(axis=0), axis=1)))
    return r_knn, R_cloud, r_knn / R_cloud


def freeze_k(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply K_FREEZE_RULE to a per-k record list (ascending k order) and return the freeze
    artifact: k_frozen, whether the rule fired, the verbatim rule text, the k grid it ran
    over, the per-k median_R_H table, the per-k delta table (None at the first point, since
    the gain there is undefined), and a one-line reason.

    Never adjusts either threshold based on what it sees -- if no k satisfies both
    conditions, k_frozen is simply the largest k in `records` and rule_fired is False. That
    is a recorded outcome, not a failure to be tuned away (04-02-PLAN.md flagged_assumptions).
    """
    if not records:
        raise ValueError("freeze_k: records must be non-empty.")
    k_grid = [int(r["k"]) for r in records]
    medians = [float(r["reliability"]["median_R_H"]) for r in records]

    delta_by_k: List[Optional[float]] = [None]
    for i in range(1, len(medians)):
        delta_by_k.append(medians[i] - medians[i - 1])

    k_frozen: Optional[int] = None
    rule_fired = False
    reason = ""
    for i in range(1, len(medians)):
        delta = delta_by_k[i]
        level = medians[i]
        if delta < 0.03 and level >= 0.5:
            k_frozen = k_grid[i]
            rule_fired = True
            reason = (
                f"rule fired at k={k_frozen}: median_R_H gain {delta:.4f} < 0.03 and "
                f"median_R_H {level:.4f} >= 0.5"
            )
            break

    if not rule_fired:
        k_frozen = k_grid[-1]
        reason = (
            "rule did not fire at any k in the grid -- k_frozen is the largest k actually "
            "run; neither threshold was adjusted"
        )

    return {
        "k_frozen": k_frozen,
        "rule_fired": rule_fired,
        "rule_text": K_FREEZE_RULE,
        "k_grid": k_grid,
        "median_R_H_by_k": dict(zip(k_grid, medians)),
        "delta_by_k": dict(zip(k_grid, delta_by_k)),
        "reason": reason,
    }


def run_cell(
    X: np.ndarray,
    k: int,
    d: int,
    seed: int,
    n_anchor: int,
    density_correct: bool = False,
    k_density: Optional[int] = None,
    subsample_file: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    # Anchors are held out of BOTH halves, so the two estimates at each anchor are
    # independent -- the same requirement cross_split_curvature's docstring states.
    anchors = perm[:n_anchor]
    rest = perm[n_anchor:]
    half = len(rest) // 2
    idx_A, idx_B = rest[:half], rest[half:]

    def field_at(idx):
        sub = np.concatenate([X[anchors], X[idx]], axis=0)
        H = curvature_probe.centroid_mean_curvature(
            sub, k=k, d=d, density_correct=density_correct, k_density=k_density
        )
        return H[: len(anchors)]

    H_A = field_at(idx_A)
    H_B = field_at(idx_B)

    # Full-cloud field, for the dynamic range figure.
    H_full = curvature_probe.centroid_mean_curvature(
        X, k=k, d=d, density_correct=density_correct, k_density=k_density
    )
    h = np.linalg.norm(H_full, axis=-1)
    p05, p95 = float(np.percentile(h, 5)), float(np.percentile(h, 95))

    out = csc.cross_curvature_field(H_A, H_B, independence="disjoint_data")
    rel = csc.reliability_summary(out["R_H"], threshold=0.5, min_fraction=0.5)

    r_knn, R_cloud, r_over_R = measure_r_over_R(X, k)

    return {
        "kind": "pu_curvature_rankability",
        "k": k, "d": d, "n": int(n), "n_anchor": int(n_anchor), "seed": seed,
        "h_p05": p05, "h_p50": float(np.percentile(h, 50)), "h_p95": p95,
        "h_spread": float(p95 / p05) if p05 > 0 else float("inf"),
        "reliability": rel,
        "median_r_dir": float(np.median(out["r_dir"])),
        "calibration": CALIBRATION,
        "wallclock_s": time.monotonic() - t0,
        "curvature_convention": curvature_probe.CURVATURE_CONVENTION,
        "density_correct": density_correct,
        "k_density": k_density,
        "r_knn": r_knn,
        "R_cloud": R_cloud,
        "r_over_R": r_over_R,
        "subsample_file": subsample_file,
    }


def _header() -> None:
    print(f"{'k':>5} {'spread':>9} {'medR_H':>8} {'fneg':>6} {'r_dir':>7} {'r/R':>7} {'admis':>6} {'s':>6}")


def _row(r: Dict[str, Any]) -> None:
    rel = r["reliability"]
    print(f"{r['k']:>5} {r['h_spread']:>9.2f} {rel['median_R_H']:>8.4f} "
          f"{rel['fraction_negative']:>6.3f} {r['median_r_dir']:>+7.3f} "
          f"{r['r_over_R']:>7.4f} {str(rel['admissible']):>6} {r['wallclock_s']:>6.0f}")


def summarize(records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("READ-OUT -- calibrated against spike 003 fixtures at d=20, k=231")
    print("=" * 78)
    print(f"  UNRANKABLE reference (quadratic_bowl): spread {CALIBRATION['unrankable_bowl_spread']}x,"
          f" rho {CALIBRATION['bowl_rho']:+.3f}")
    print(f"  RANKABLE   reference (cubic)         : spread {CALIBRATION['rankable_cubic_spread']}x,"
          f" rho {CALIBRATION['cubic_rho']:+.3f}")
    print()
    best = max(records, key=lambda r: r["reliability"]["median_R_H"])
    spread = best["h_spread"]
    medR = best["reliability"]["median_R_H"]
    print(f"  PU best cell: k={best['k']}  spread {spread:.2f}x  median R_H {medR:.4f}  "
          f"r/R {best['r_over_R']:.4f}  median r_dir {best['median_r_dir']:+.3f}")
    print()
    # NOTE (2026-08-22): an earlier version of this runner branched on `spread < 3.0` to
    # declare "near-constant curvature". That branch was REMOVED as unsound. The calibration
    # sweep in spike 003 measured `rho = +0.48` at a spread of 1.1x and `rho = +0.36` at 36x
    # on the same fixture family: Spearman is scale-free, so dynamic range does not predict
    # rankability. Spread is retained as a REPORTED figure and is no longer gated on.
    if medR < 0.2:
        print("  NOT REPRODUCIBLE. Two disjoint halves of the cloud do not agree on the field,")
        print("  so what range it has is consistent with noise. Phase 4 would be partitioning")
        print("  on an unreliable ordering. Raise k before concluding: reliability rose")
        print("  monotonically 0.078 -> 0.247 -> 0.428 over k = 30, 60, 120 on the PU cloud.")
    else:
        print("  REPRODUCIBLE. Two disjoint halves of the cloud agree on the field. Phase 4's")
        print("  premise survives this check -- which is NECESSARY, NOT SUFFICIENT, and the")
        print("  distinction is this spike's central finding: reliability certifies that a")
        print("  measurement reproduces, never that it is right. A bias both halves share is")
        print("  perfectly reliable. There is no ground truth on real data, so this can never")
        print("  be upgraded to a correctness claim by more of the same measurement.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, nargs="+", default=[30, 60, 120, 231])
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--column", type=str, default="legacysurvey")
    p.add_argument("--n-anchor", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--density-correct", action="store_true", default=False,
        help="D4-15: apply curvature_probe's density correction (D-06). Default False "
             "preserves this runner's pre-plan uncorrected behaviour exactly.",
    )
    p.add_argument(
        "--k-density", type=int, default=None,
        help="Required whenever --density-correct is set (mirrors "
             "centroid_mean_curvature's own flag/value pairing guard -- passing "
             "--density-correct with no --k-density propagates straight into that guard's "
             "ValueError naming k_density). D4-15's pre-registered value is 30.",
    )
    p.add_argument(
        "--freeze-out", type=str, default=None,
        help="Stem (no extension) to write the D4-07 freeze artifact to, via "
             "cache.cache_path so the containment guard applies. When given, freeze_k is "
             "applied to every accumulated record on disk at record_path matching this "
             "run's (density_correct, k_density, d), deduplicated by k.",
    )
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    X, subsample_file = load_pu(a.column)

    if a.smoke:
        print("SMOKE: 800 rows, k=30 -- proves the path runs, measures nothing.\n")
        _header()
        _row(run_cell(
            X[:800], 30, a.d, a.seed, 200,
            density_correct=a.density_correct, k_density=a.k_density,
            subsample_file=subsample_file,
        ))
        return

    record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"PU curvature rankability -- d={a.d}, column={a.column}, n={X.shape[0]}, "
          f"density_correct={a.density_correct}, k_density={a.k_density}")
    print("=" * 78)
    print(f"record_path = {record_path}\n")

    _header()
    records: List[Dict[str, Any]] = []
    with record_path.open("a") as fh:
        for k in a.k:
            r = run_cell(
                X, k, a.d, a.seed, a.n_anchor,
                density_correct=a.density_correct, k_density=a.k_density,
                subsample_file=subsample_file,
            )
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            records.append(r)
            _row(r)

    summarize(records)

    if a.freeze_out:
        # "Accumulated" per the plan: read every record ever written to record_path (not
        # only this invocation's), so a second pass that only adds k=500 still freezes
        # against the FULL grid, not a single new point.
        all_records: List[Dict[str, Any]] = []
        with record_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
        matching = [
            r for r in all_records
            if r.get("density_correct") == a.density_correct
            and r.get("k_density") == a.k_density
            and r.get("d") == a.d
        ]
        by_k = {r["k"]: r for r in matching}  # last-written row per k wins
        ordered = [by_k[k] for k in sorted(by_k)]
        freeze = freeze_k(ordered)
        out_path = cache.cache_path(a.freeze_out, "json")
        out_path.write_text(json.dumps(freeze, indent=2, default=float))
        print(f"\nfreeze written to {out_path}: k_frozen={freeze['k_frozen']} "
              f"rule_fired={freeze['rule_fired']}")
        print(f"  {freeze['reason']}")


if __name__ == "__main__":
    main()
