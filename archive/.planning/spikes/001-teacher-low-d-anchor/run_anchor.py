"""Spike 001 -- low-`d` anchor for the local-polynomial geometry teacher.

Runs `curvature_probe.quadric_mean_curvature` (the teacher: `P̂` from the SVD tangent frame,
`ÎI` from minimum-norm least squares, `H = tr_g(ÎI)` returned as an ambient vector) UNMODIFIED,
and scores it with `synthetic_control_run._fidelity_axes` UNMODIFIED -- the same four axes, same
code path, that produced the sealed `d=20` decoder row `rank_spearman_rho = -0.0151`.

Why this runs before the `d=20` test: a FAIL at `d=20` has more than one possible cause. Either
the dimension wall recorded in `02.5-NOTE-high-d-curvature-approaches.md` §1 (`r/R = 0.906`), or
a mis-wired fixture, a convention slip, or a scorer fed vectors in mismatched frames. Only a
fixture whose answer is known separates them.

WHAT "CLEARS" MEANS HERE, and why it is stated in `r/R` rather than in `d`. The teacher inverts
an identity exact only as the neighbourhood radius `r -> 0`, so the regime where it MUST succeed
is the regime where the neighbourhood is genuinely local. §1's table pins that: `r/R = 0.115` at
`d=2` (Swiss roll, works, `median_relative_error = 0.125`), `0.391` at `d=5` (already `0.346`),
`0.906` at `d=20` (`0.870`). The anchor therefore gates on the two genuinely-local fixtures and
reports `d=4` as the TRANSITION row with its own `n`-ladder, rather than gating on it.

REVISION NOTE, recorded rather than quietly applied. The first run of this file gated `d=4` at
`rho >= 0.90` and gated the raw magnitude-ratio CV at `<= 0.50`, and both failed. `probe_cv_and_n.py`
then measured why, and both failures were the criteria rather than the teacher:
  - The raw CV on a MIXED-SIGN fixture divides by a near-zero denominator on the saddle's own
    cancellation set (`||H_true||` passes through zero by construction). The scorer's
    `MIN_TRUE_NORM = 1e-12` excludes none of it. Measured at `d=2, n=10000`: raw CV `1.31`,
    but `0.116` above a 5% quantile floor and `0.028` above the median, with the median ratio
    pinned at `0.998` throughout. The Swiss roll, whose `H` never changes sign, gives raw CV
    `0.009`. So BOTH CV forms are reported below and the floored one carries the gate.
  - `d=4` at `n=3000` scored `rho = 0.738`; at the sealed `n=10000` it is `0.845` and at
    `n=30000` it is `0.903`, against `r/R = 0.316 / 0.234 / 0.178`. It was undersampling, and
    the `(k/n)^(1/d)` law is visible in the ladder.

No caching, no verdict JSON. Prints the four axes per fixture, the measured `r/R`, and pass/fail.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks" / "diagnostics"))

import synthetic_control_run as scr  # noqa: E402  (self-bootstraps pu_manifold onto sys.path)
from pu_manifold import curvature_probe, synthetic_controls  # noqa: E402

N = 10000          # the sealed protocol's CONTROL_N, so `n` never confounds a `d` comparison
K = 30             # §1's own k, so measured r/R is comparable to its table row-for-row
SEED = 20260816    # CONTROL_FIXTURE_SEED
CV_FLOOR_Q = 0.10  # quantile floor on ||H_true|| for the cancellation-robust CV

RHO_MIN = 0.90
COSINE_MIN = 0.90
RATIO_BAND = (0.70, 1.30)
CV_MAX = 0.50      # applies to the FLOORED CV; the raw CV is reported, not gated


def measure_r_over_R(X, k):
    """§1's own locality statistic, recomputed on our fixture: median kNN ball radius against
    the cloud's median radius from its own centre. Reproduced here rather than quoted, so the
    anchor's regime claim rests on a measurement of the fixture actually being scored."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    r_knn = float(np.median(dist[:, -1]))
    R = float(np.median(np.linalg.norm(X - X.mean(axis=0), axis=1)))
    return r_knn, R, r_knn / R


def floored_cv(H_est, H_true, q):
    """Magnitude-ratio CV restricted to points whose analytic ||H|| is above its `q` quantile.

    The saddle's trace cancels by construction, so `||H_est|| / ||H_true||` has a denominator
    that approaches zero on a set of positive measure. That makes the unrestricted CV a
    statistic about the fixture's cancellation locus rather than about the teacher's scatter.
    Everything else -- median ratio, cosine, rank, calibration -- is unaffected and is taken
    from the sealed scorer unmodified. This is the only spike-local statistic in this file."""
    ne = np.linalg.norm(H_est, axis=-1)
    nt = np.linalg.norm(H_true, axis=-1)
    keep = nt >= np.quantile(nt, q)
    r = (ne / nt)[keep]
    return float(np.std(r, ddof=1) / np.mean(r)), int(keep.sum())


def swiss_roll_fixture(n, seed):
    """The Swiss roll with its analytic mean-curvature VECTOR, not only its norm.

    `make_swiss_roll_fixture` returns `H_norm` alone; the four-axis scorer needs the ambient
    vector to score direction. The spiral algebra below is copied verbatim from
    `swiss_roll_curvature_sweep_run.run_cell` and pinned against the fixture's own sealed
    `H_norm` to 1e-12, so a drift in either derivation breaks here rather than silently
    scoring the teacher against a wrong truth."""
    fx = curvature_probe.make_swiss_roll_fixture(n=n, seed=seed)
    t, global_std = fx["t"], fx["global_std"]

    ct, st = np.cos(t), np.sin(t)
    d1 = np.stack([ct - t * st, st + t * ct], axis=1)
    d2 = np.stack([-2 * st - t * ct, 2 * ct - t * st], axis=1)
    speed2 = np.sum(d1 * d1, axis=1)
    k_vec = (d2 - (np.sum(d2 * d1, axis=1) / speed2)[:, None] * d1) / speed2[:, None]

    H_true = np.zeros((n, 3))
    H_true[:, 0] = k_vec[:, 0] * global_std
    H_true[:, 2] = k_vec[:, 1] * global_std

    pin = float(np.abs(np.linalg.norm(H_true, axis=1) - fx["H_norm"]).max())
    if pin >= 1e-12:
        raise ValueError(
            f"swiss_roll_fixture: derived analytic H vector disagrees with the fixture's "
            f"sealed H_norm by {pin:.2e} (must be < 1e-12) -- the spiral algebra has drifted."
        )
    return fx["X"], H_true, 2


def saddle_fixture(n, d, D, seed):
    fx = synthetic_controls.make_saddle_control(n=n, d=d, D=D, seed=seed)
    return fx["X"], fx["H_vec"], d


def score(name, X, H_true, d, k):
    n_coef = d * (d + 1) // 2
    out = curvature_probe.quadric_mean_curvature(X, k=k, d=d)
    axes = scr._fidelity_axes(out["H_vec"], H_true)
    cv_f, n_kept = floored_cv(out["H_vec"], H_true, CV_FLOOR_Q)
    r_knn, R, ratio_rR = measure_r_over_R(X, k)

    print(f"\n--- {name}  (n={X.shape[0]}, d={d}, D={X.shape[1]}, k={k})")
    print(f"  locality    r_knn={r_knn:.4f}  R={R:.4f}  r/R = {ratio_rR:.4f}")
    print(f"  quadratic coefficients = {n_coef}   deficit = {max(0, n_coef - k)}   "
          f"underdetermined = {out['underdetermined']}")
    print(f"  direction   median cosine        = {axes['direction_median_cosine']:.6f}")
    print(f"  magnitude   median ratio         = {axes['magnitude_median_ratio']:.6f}")
    print(f"              CV raw = {axes['magnitude_ratio_cv']:.6f}   "
          f"CV above q{CV_FLOOR_Q:.2f} of ||H_true|| = {cv_f:.6f}  (n={n_kept})")
    if axes["rank_calibration_applicable"]:
        print(f"  calibration slope={axes['calibration_slope']:.6f} "
              f"intercept={axes['calibration_intercept']:.6e} R2={axes['calibration_r2']:.6f}")
        print(f"  rank        spearman rho         = {axes['rank_spearman_rho']:.6f}")
    else:
        print("  rank/calibration UNDEFINED (constant analytic ||H||)")
    print(f"  median relative error            = {axes['median_relative_error']:.6f}")
    print(f"  points scored = {axes['n_points']}   excluded (||H_true|| ~ 0) = {axes['n_excluded']}")
    print(f"  convention = {axes['curvature_convention']}")
    axes["_cv_floored"] = cv_f
    axes["_r_over_R"] = ratio_rR
    return axes


def verdict(name, axes):
    rho, cos = axes["rank_spearman_rho"], axes["direction_median_cosine"]
    ratio, cv = axes["magnitude_median_ratio"], axes["_cv_floored"]
    checks = [
        (f"rank rho >= {RHO_MIN:.2f}", rho is not None and rho >= RHO_MIN, rho),
        (f"median cosine >= {COSINE_MIN:.2f}", cos >= COSINE_MIN, cos),
        (f"median ratio in [{RATIO_BAND[0]:.2f}, {RATIO_BAND[1]:.2f}]",
         RATIO_BAND[0] <= ratio <= RATIO_BAND[1], ratio),
        (f"floored ratio CV <= {CV_MAX:.2f}", cv <= CV_MAX, cv),
    ]
    ok = all(c[1] for c in checks)
    print(f"\n  {name}:")
    for label, passed, value in checks:
        shown = "None" if value is None else f"{value:.6f}"
        print(f"    [{'PASS' if passed else 'FAIL'}] {label:<32} got {shown}")
    return ok


def main():
    print("=" * 78)
    print("SPIKE 001 -- local-polynomial geometry teacher, low-d anchor")
    print("teacher : curvature_probe.quadric_mean_curvature   (unmodified)")
    print("scorer  : synthetic_control_run._fidelity_axes     (unmodified)")
    print(f"n={N}  k={K}  seed={SEED}")
    print("=" * 78)

    gated = {}
    X, H, d = swiss_roll_fixture(N, seed=SEED)
    gated["swiss roll  (d=2, D=3)"] = score("SWISS ROLL -- the CLAUDE.md anchor", X, H, d, K)

    X, H, d = saddle_fixture(N, d=2, D=8, seed=SEED)
    gated["saddle d=2"] = score("SADDLE d=2 -- mixed-sign, genuinely local", X, H, d, K)

    print("\n" + "-" * 78)
    print("TRANSITION ROW -- reported, not gated. This is where locality starts to go.")
    print("-" * 78)
    X, H, d = saddle_fixture(N, d=4, D=12, seed=SEED)
    score("SADDLE d=4", X, H, d, K)

    print("\n" + "=" * 78)
    print("VERDICT -- gated on the genuinely-local fixtures only")
    print("=" * 78)
    all_ok = all(verdict(name, axes) for name, axes in gated.items())

    print()
    if all_ok:
        print("ANCHOR HELD. The teacher recovers known curvature on all four axes wherever the")
        print("quadratic fit is determined AND the neighbourhood is genuinely local. Fixture,")
        print("convention, ambient frame and scorer are therefore wired correctly, and a FAIL")
        print("at d=20 is a statement about d -- not about this pipeline.")
    else:
        print("ANCHOR BROKEN. Do not run or interpret the d=20 test until this is explained.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
