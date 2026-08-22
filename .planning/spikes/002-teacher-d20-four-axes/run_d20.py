"""Spike 002 -- the local-polynomial geometry teacher at `d = 20`, scored on the four axes.

The developer-directed question (`03.1-FINDINGS.md` §10, 2026-08-21): score a geometry-supervised
signal ALONE on the same four axes as the sealed decoder, at `d = 20`, where analytic truth
exists -- before any architecture change is proposed. Phase 03.1 established that decoder
parameterization is not the missing ingredient (`scale` drives `log10_det_g` from `-83.9` to
`+0.037` and moves rank `rho` only to `+0.116`). This asks whether there is anything at `d = 20`
that could teach a decoder geometry in the first place.

Teacher: `curvature_probe.quadric_mean_curvature`, UNMODIFIED -- `P̂` from the SVD tangent frame,
`ÎI` from minimum-norm least squares over `1 + d + d(d+1)/2 = 231` columns, `H = tr_g(ÎI)`
returned in ambient coordinates. No shrinkage dial: D-05 rejected one and the user held the ridge
variant on 2026-08-21.
Scorer: `synthetic_control_run._fidelity_axes`, UNMODIFIED -- the same four axes and the same code
path that produced the sealed row.
Fixture: `synthetic_controls.make_saddle_control`, the sealed `d=20` control, `seed = 20260816`.

Spike 001 established that this teacher and this scorer recover known curvature to
`rho = 0.9975` (saddle `d=2`) and `0.99998` (Swiss roll) at the same `n`, so anything that
happens here is about `d`, not about wiring.

THREE PARTS.

Part A -- the ambient-dimension licence. The grid runs at `D = 28` rather than the sealed
`D = 768`, because `_quadric_tangent_basis` calls `np.linalg.svd(..., full_matrices=True)`, which
materializes a `(D, D)` matrix per point -- `768 x 768` at every one of 10000 points. The claim
that this is free is not assumed: `02.5-NOTE` §2a records `D in {28, 50, 200, 768}` giving
bit-identical GATING values, and the argument transfers (`make_saddle_control` draws `X_local`
before `Q`, so `X_local` is identical across `D`; `Q` is orthogonal, so every distance, norm,
cosine and rank is invariant). Part A measures it for THIS estimator rather than inheriting it.

Part B -- the `k` ladder. `k = 30` is the sealed protocol's own neighbourhood and §1's table row.
`k = 231` is the first `k` at which the fit is DETERMINED (`1 + d + d(d+1)/2 = 231` columns). The
ladder spans the underdetermined and determined regimes, so "underdetermined" and "non-local" are
separated as explanations rather than confounded.

Part C -- the `n` lever, tested rather than argued. Spike 001 measured that tripling `n` at
`d = 4` moved `rho` from `0.738` to `0.903`. §1's law says the same lever is worth 11% at
`d = 20`. This runs `n = 30000` at the determined `k` to see whether it moves anything.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks" / "diagnostics"))

import synthetic_control_run as scr  # noqa: E402
from pu_manifold import curvature_probe, synthetic_controls  # noqa: E402

D_INTRINSIC = 20
SEED = 20260816                 # CONTROL_FIXTURE_SEED
N_SEALED = 10000                # CONTROL_N
D_CHEAP = 28                    # > d + 1 = 21, so the graph embeds exactly
D_SEALED = 768                  # PU's working ambient dimension
N_COEF = D_INTRINSIC * (D_INTRINSIC + 1) // 2          # 210
K_DETERMINED = 1 + D_INTRINSIC + N_COEF                 # 231
CV_FLOOR_Q = 0.10

# Reference values this spike is scored against, quoted from sealed records.
SEALED_DECODER_RHO = -0.015106571347065712   # 03.1-FINDINGS.md §2, exact
PHASE_031_BEST_RHO = 0.116                   # `scale`, strong rung, 03.1-FINDINGS.md §10
PHASE_031_COMBO_RHO = 0.013                  # post-hoc combination, ibid
D4_REFERENCE_RHO = 0.989                     # same fixture at chart_dim=4, 03-FINDINGS.md §6


def measure_r_over_R(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    r_knn = float(np.median(dist[:, -1]))
    R = float(np.median(np.linalg.norm(X - X.mean(axis=0), axis=1)))
    return r_knn / R


def floored_cv(H_est, H_true, q=CV_FLOOR_Q):
    """Magnitude-ratio CV above a quantile floor on ||H_true||. Spike 001 measured that the raw
    CV on this mixed-sign fixture is dominated by its cancellation set (`trace(Q) = 0` by
    construction, `MIN_TRUE_NORM = 1e-12` excludes none of it): at `d=2` the raw CV is `1.31`
    while the floored CV is `0.079` and the median ratio never leaves `0.998`. Both are reported
    here so the comparison across `d` is like-for-like."""
    ne = np.linalg.norm(H_est, axis=-1)
    nt = np.linalg.norm(H_true, axis=-1)
    keep = nt >= np.quantile(nt, q)
    r = (ne / nt)[keep]
    return float(np.std(r, ddof=1) / np.mean(r))


def cell(n, D, k, label):
    fx = synthetic_controls.make_saddle_control(n=n, d=D_INTRINSIC, D=D, seed=SEED)
    X, H_true = fx["X"], fx["H_vec"]
    t0 = time.time()
    out = curvature_probe.quadric_mean_curvature(X, k=k, d=D_INTRINSIC)
    wall = time.time() - t0
    axes = scr._fidelity_axes(out["H_vec"], H_true)
    rec = {
        "label": label, "n": n, "D": D, "k": k,
        "deficit": max(0, N_COEF - k),
        "underdetermined": bool(out["underdetermined"]),
        "r_over_R": measure_r_over_R(X, k),
        "rho": axes["rank_spearman_rho"],
        "cosine": axes["direction_median_cosine"],
        "ratio": axes["magnitude_median_ratio"],
        "cv_raw": axes["magnitude_ratio_cv"],
        "cv_floored": floored_cv(out["H_vec"], H_true),
        "slope": axes["calibration_slope"],
        "r2": axes["calibration_r2"],
        "mre": axes["median_relative_error"],
        "wall_s": wall,
    }
    print(f"  {label:<24} k={k:<4d} n={n:<6d} D={D:<4d} "
          f"r/R={rec['r_over_R']:.4f} deficit={rec['deficit']:<4d} "
          f"rho={rec['rho']:+.6f} cos={rec['cosine']:+.4f} ratio={rec['ratio']:.4f} "
          f"CV={rec['cv_floored']:.3f} R2={rec['r2']:.4f} MRE={rec['mre']:.4f} "
          f"[{wall:.1f}s]", flush=True)
    return rec


def part_a(n_probe):
    print("\n" + "=" * 100)
    print("PART A -- ambient-dimension licence: is D=28 the same measurement as D=768?")
    print("=" * 100, flush=True)
    a = cell(n_probe, D_SEALED, 30, "D=768 (sealed)")
    b = cell(n_probe, D_CHEAP, 30, "D=28 (cheap)")
    keys = ["rho", "cosine", "ratio", "cv_floored", "slope", "r2", "mre", "r_over_R"]
    worst = max(abs(a[k] - b[k]) for k in keys)
    print(f"\n  worst absolute disagreement across all axes: {worst:.3e}")
    print(f"  speedup from D=768 -> D=28: {a['wall_s'] / b['wall_s']:.1f}x")
    # 1e-6, not exact equality: the two fixtures carry DIFFERENT orthogonal rotations Q, so
    # the same geometry is expressed in different coordinates and float rounding differs. A
    # near-tie in the kNN sort can flip one neighbour, which moves a rank statistic by ~1e-6.
    # Exact bit-identity is not the claim; that the measurement is the same one is.
    ok = worst < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] axes are D-invariant to < 1e-6 -- "
          f"{'the grid may run at D=28' if ok else 'DO NOT run the grid at D=28'}")
    return ok


def part_b(n, ks):
    print("\n" + "=" * 100)
    print(f"PART B -- the k ladder at d=20, n={n}, D={D_CHEAP}")
    print(f"          fit is DETERMINED at k >= {K_DETERMINED}; 210 quadratic coefficients")
    print("=" * 100, flush=True)
    return [cell(n, D_CHEAP, k, f"k={k}") for k in ks]


def part_c(k):
    print("\n" + "=" * 100)
    print(f"PART C -- the n lever at the determined k={k}: does more data rescue d=20?")
    print("          spike 001 measured n 3000->30000 moving d=4 from rho 0.738 to 0.903")
    print("=" * 100, flush=True)
    return [cell(n, D_CHEAP, k, f"n={n}") for n in (N_SEALED, 30000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SEALED)
    ap.add_argument("--probe-n", type=int, default=600)
    ap.add_argument("--ks", type=int, nargs="+", default=[30, 100, K_DETERMINED, 500])
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()

    print("=" * 100)
    print("SPIKE 002 -- local-polynomial geometry teacher at d=20, four axes")
    print("teacher : curvature_probe.quadric_mean_curvature   (unmodified, min-norm lstsq)")
    print("scorer  : synthetic_control_run._fidelity_axes     (unmodified)")
    print(f"fixture : make_saddle_control(d=20, seed={SEED}) -- the sealed control")
    print("=" * 100)

    licensed = part_a(args.probe_n)
    if not licensed:
        print("\nABORT: D-invariance did not hold. The cheap grid is not the sealed measurement.")
        return 2

    rows = part_b(args.n, args.ks)
    if not args.skip_c:
        rows += part_c(K_DETERMINED)

    best = max(rows, key=lambda r: r["rho"])
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  best teacher rho over all cells    = {best['rho']:+.6f}   "
          f"(k={best['k']}, n={best['n']}, r/R={best['r_over_R']:.4f})")
    print(f"  sealed decoder rho (d=20 saddle)   = {SEALED_DECODER_RHO:+.6f}")
    print(f"  phase 03.1 best rho (scale prior)  = {PHASE_031_BEST_RHO:+.6f}")
    print(f"  same fixture at d=4 (reference)    = {D4_REFERENCE_RHO:+.6f}")
    print()
    beats_decoder = best["rho"] > SEALED_DECODER_RHO
    beats_031 = best["rho"] > PHASE_031_BEST_RHO
    usable = best["rho"] >= 0.90
    for label, ok in (
        ("teacher beats the sealed decoder", beats_decoder),
        ("teacher beats phase 03.1's best prior", beats_031),
        ("teacher is usable as supervision (rho >= 0.90)", usable),
    ):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print()
    if usable:
        print("A geometry-supervised objective is FEASIBLE at d=20: the teacher orders curvature")
        print("well enough to supervise a decoder that currently cannot.")
    elif beats_031:
        print("PARTIAL: the teacher carries more ordering than any decoder-side remedy tried so")
        print("far, but not enough to supervise with. It is a better signal, not a usable one.")
    else:
        print("The teacher does NOT carry usable geometry at d=20. A geometry-supervised")
        print("objective has nothing to teach with here -- the signal is absent before any")
        print("architecture is chosen, so no architecture change addresses it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
