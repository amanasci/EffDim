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
from typing import Any, Dict, List

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np

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


def load_pu(column: str) -> np.ndarray:
    """The frozen Phase 1 10k subsample. Read-only; this runner never writes to the cache
    except its own record."""
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
    print(f"loaded {column} {X.shape} from {Path(best).name}")
    return X


def run_cell(X: np.ndarray, k: int, d: int, seed: int, n_anchor: int) -> Dict[str, Any]:
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
        H = curvature_probe.centroid_mean_curvature(sub, k=k, d=d)
        return H[: len(anchors)]

    H_A = field_at(idx_A)
    H_B = field_at(idx_B)

    # Full-cloud field, for the dynamic range figure.
    H_full = curvature_probe.centroid_mean_curvature(X, k=k, d=d)
    h = np.linalg.norm(H_full, axis=-1)
    p05, p95 = float(np.percentile(h, 5)), float(np.percentile(h, 95))

    out = csc.cross_curvature_field(H_A, H_B, independence="disjoint_data")
    rel = csc.reliability_summary(out["R_H"], threshold=0.5, min_fraction=0.5)

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
    }


def _header() -> None:
    print(f"{'k':>5} {'spread':>9} {'medR_H':>8} {'fneg':>6} {'r_dir':>7} {'admis':>6} {'s':>6}")


def _row(r: Dict[str, Any]) -> None:
    rel = r["reliability"]
    print(f"{r['k']:>5} {r['h_spread']:>9.2f} {rel['median_R_H']:>8.4f} "
          f"{rel['fraction_negative']:>6.3f} {r['median_r_dir']:>+7.3f} "
          f"{str(rel['admissible']):>6} {r['wallclock_s']:>6.0f}")


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
    print(f"  PU best cell: k={best['k']}  spread {spread:.2f}x  median R_H {medR:.4f}")
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
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    X = load_pu(a.column)

    if a.smoke:
        print("SMOKE: 800 rows, k=30 -- proves the path runs, measures nothing.\n")
        _header()
        _row(run_cell(X[:800], 30, a.d, a.seed, 200))
        return

    record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"PU curvature rankability -- d={a.d}, column={a.column}, n={X.shape[0]}")
    print("=" * 78)
    print(f"record_path = {record_path}\n")

    _header()
    records: List[Dict[str, Any]] = []
    with record_path.open("a") as fh:
        for k in a.k:
            r = run_cell(X, k, a.d, a.seed, a.n_anchor)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            records.append(r)
            _row(r)

    summarize(records)


if __name__ == "__main__":
    main()
