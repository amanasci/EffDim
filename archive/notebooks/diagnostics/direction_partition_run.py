"""D4-01 — does partitioning by curvature DIRECTION recover regions that `|H|` quantiles miss?

**The fork.** Phase 4 as originally specified bins `|H|` quantiles. D4-01 replaces that with
clustering the unit vectors `H/||H||`, on the grounds that spike 003 measured direction to
survive at `d=20` (cosine -> 1.000) where magnitude does not (~50x attenuated). This runner tests
that claim against a known answer instead of asserting it.

**Why a codimension-1 fixture cannot test this, and why every earlier fixture was one.** For a
graph with scalar `f`, the normal space is ONE-dimensional, so `H = H_scalar * n_hat` and the
direction of `H` is just the Gauss map -- clustering it clusters tangent-plane orientation, not
curvature structure. Measured on `ridge` at `d=8`: unit-`H` covariance rank 2. The cosine 1.000
result in spike 003 therefore establishes that the estimator recovers the NORMAL ORIENTATION,
which is a tangent-space problem known to converge -- NOT that it resolves `H`'s direction inside
a high-dimensional normal space.

PU is `d ~ 20` inside `D = 768`: codimension ~748. `varying_ii_controls.make_multinormal_ridge_control`
supplies the missing regime -- `f: R^d -> R^m` with `m` orthonormal ridge directions, so `H`
rotates within an `m`-dimensional normal space and carries an analytic region label
(`dominant_normal`, which component dominates). Measured at `d=20, m=4`: unit-`H` covariance rank
8 with eigenvalues 0.25 x 4, and balanced region counts.

**What is scored, and why not against an external region label.** A first version of this
runner scored both partitions against the fixture's `dominant_normal` label and got ~0 for
BOTH -- including the oracle. That was a defect in the test, not a finding: `H`'s normal
components scale with `sin(s_j)`, which take both signs, so `argmax_j |sin(s_j)|` regions are
alternating angular sectors that k-means cannot express. Using a SIGNED dominant label lifted
the oracle to ARI 0.25-0.39. But the deeper problem remains -- `H`'s direction is a CONTINUOUS
quantity and any discrete region label is a coarse proxy for it.

So the question is posed directly instead, with no external label:

    direction recovery   ARI( cluster(estimated unit H),  cluster(TRUE unit H) )
    magnitude recovery   ARI( qbins(estimated ||H||),     qbins(TRUE ||H||)    )

Each asks: **if Phase 4 built this partition from the estimated field, how closely would it
match the partition it would have built from the true field?** That is exactly what Phase 4
needs to know, it is apples-to-apples across the two schemes, and it needs no region label that
might be mismatched to the clustering geometry.

Adjusted Rand is chance-corrected, so 0.0 means "no better than random" and no separate null is
needed. `dominant_normal` is retained as a secondary sanity figure only.

    python notebooks/diagnostics/direction_partition_run.py --smoke
    python notebooks/diagnostics/direction_partition_run.py --d 20 --n-normal 4 --k 231
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from pu_manifold import curvature_probe
from pu_manifold import varying_ii_controls as vic

DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "03.2_direction_partition.jsonl"


def _unit(H: np.ndarray) -> np.ndarray:
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)


def _quantile_labels(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bins by rank, matching curvature_probe._quantile_bin_labels' construction
    (np.array_split over the sort order) so bin membership is well defined under ties."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.shape[0], dtype=np.int64)
    for b, grp in enumerate(np.array_split(order, n_bins)):
        labels[grp] = b
    return labels


def run(n: int, d: int, D: int, k: int, n_normal: int, seed: int,
        cluster_seed: int) -> Dict[str, Any]:
    t0 = time.monotonic()
    fx = vic.make_multinormal_ridge_control(n, d, D, seed, n_normal=n_normal)
    X = np.asarray(fx["X"], dtype=np.float64)
    H_true = np.asarray(fx["H_vec"], dtype=np.float64)
    truth = np.asarray(fx["dominant_normal"])

    H_est = curvature_probe.centroid_mean_curvature(X, k=k, d=d)

    def km(V, nc):
        return KMeans(n_clusters=nc, n_init=10, random_state=cluster_seed).fit_predict(V)

    n_regions = 2 * n_normal  # signed sectors: H's normal components change sign
    lab_dir_est = km(_unit(H_est), n_regions)
    lab_dir_true = km(_unit(H_true), n_regions)
    lab_mag_est = _quantile_labels(np.linalg.norm(H_est, axis=1), n_regions)
    lab_mag_true = _quantile_labels(np.linalg.norm(H_true, axis=1), n_regions)

    # Secondary sanity figure only -- see the module docstring on why this is not the target.
    S = np.sin(fx["frequency"] * (np.asarray(fx["x_param"]) @ np.asarray(fx["W"])))
    jmax = np.argmax(np.abs(S), axis=1)
    signed_dominant = jmax * 2 + (S[np.arange(len(S)), jmax] > 0).astype(int)

    return {
        "kind": "direction_partition",
        "n": n, "d": d, "D": D, "k": k, "n_normal": n_normal,
        "seed": seed, "cluster_seed": cluster_seed,
        "ii_cv": fx["ii_variation"]["hess_fro_cv"],
        "region_counts": np.bincount(truth, minlength=n_normal).tolist(),
        "n_regions": int(n_regions),
        "ari_direction_recovery": float(adjusted_rand_score(lab_dir_true, lab_dir_est)),
        "ari_magnitude_recovery": float(adjusted_rand_score(lab_mag_true, lab_mag_est)),
        "ari_direction_vs_signed_dominant": float(
            adjusted_rand_score(signed_dominant, lab_dir_est)),
        "ari_direction_oracle_vs_signed_dominant": float(
            adjusted_rand_score(signed_dominant, lab_dir_true)),
        "median_cosine_est_vs_true": float(np.median(
            (_unit(H_est) * _unit(H_true)).sum(1))),
        "wallclock_s": time.monotonic() - t0,
        "curvature_convention": vic.CURVATURE_CONVENTION,
    }


def _report(r: Dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"D4-01 DIRECTION PARTITION -- d={r['d']} D={r['D']} n={r['n']} k={r['k']} "
          f"normals={r['n_normal']}")
    print("=" * 78)
    print(f"  fixture II CV               = {r['ii_cv']:.4f}")
    print(f"  region counts (truth)       = {r['region_counts']}")
    print(f"  cosine, estimated vs true H = {r['median_cosine_est_vs_true']:+.4f}")
    print()
    print(f"  {'partition scheme':>28} | {'estimator fidelity (ARI)':>25}")
    print(f"  {'-'*28} | {'-'*25}")
    print(f"  {'DIRECTION clustering':>28} | {r['ari_direction_recovery']:>25.4f}")
    print(f"  {'MAGNITUDE quantiles':>28} | {r['ari_magnitude_recovery']:>25.4f}")
    print()
    print("  Each row: how closely the partition built from the ESTIMATED field matches the")
    print("  partition that would have been built from the TRUE field. Chance-corrected;")
    print(f"  {r['n_regions']} regions in both schemes, so they are directly comparable.")
    print()
    print(f"  secondary: direction clusters vs signed-dominant-normal label"
          f"  est {r['ari_direction_vs_signed_dominant']:+.4f}"
          f"  oracle {r['ari_direction_oracle_vs_signed_dominant']:+.4f}")
    print()

    dr, mr = r["ari_direction_recovery"], r["ari_magnitude_recovery"]
    if dr > mr + 0.15 and dr > 0.3:
        print("  D4-01 SUPPORTED. At PU's own intrinsic dimension, a direction-based partition")
        print("  is reproduced from the estimated field substantially better than a magnitude")
        print("  quantile partition is. Phase 4 should partition on direction.")
        print("  Scope limit: this fixture has codimension "
              f"{r['n_normal']}; PU's is ~748 and its normal structure is unknown.")
    elif mr > dr + 0.15:
        print("  D4-01 REFUTED on this fixture. The magnitude partition is reproduced BETTER")
        print("  than the direction partition. Phase 4's original quantile specification")
        print("  should stand and D4-01 should be revisited.")
    else:
        print("  INCONCLUSIVE. The two schemes are reproduced comparably well; neither is")
        print("  clearly preferable on this evidence. Do NOT rewrite Phase 4's success")
        print("  criteria on this margin -- vary k, n_normal and d before deciding.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--ambient", type=int, default=32)
    p.add_argument("--k", type=int, nargs="+", default=[231])
    p.add_argument("--n-normal", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--cluster-seed", type=int, default=0)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    if a.smoke:
        print("SMOKE: n=800, d=6, k=30 -- proves the path runs, measures nothing.")
        _report(run(800, 6, 12, 30, 2, a.seed, a.cluster_seed))
        return
    record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    with record_path.open("a") as fh:
        for k in a.k:
            r = run(a.n, a.d, a.ambient, k, a.n_normal, a.seed, a.cluster_seed)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            records.append(r)
            _report(r)
    print(f"\nwrote {record_path}")


if __name__ == "__main__":
    main()
