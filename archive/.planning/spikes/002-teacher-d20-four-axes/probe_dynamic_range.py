"""Spike 002, probe 3 -- is the bump fixture easier because of DYNAMIC RANGE?

Established so far at `d=20`, `k=30`, `n=10000`:
  saddle  rho = -0.0281   ||H_true|| spread p95/p05 =    33.3
  bumps   rho = +0.5934   ||H_true|| spread p95/p05 =  1095.3
and the gap is not the saddle's global quadratic form (probe 1: that explains the `k`-dependence,
not the level) and not local sampling scale (probe 2: partial rho = +0.6006, essentially the
headline).

One difference remains. Ranking is a comparison between points, so the difficulty of ranking a
field depends on how far apart its values are relative to the estimator's error. At `d=20` the
estimator carries ~87% median relative error on BOTH fixtures (0.8494 saddle, 0.8673 bumps --
near-identical). A field spanning three orders of magnitude survives 87% error in rank terms,
because most pairs of points differ by far more than the error. A field spanning 33x does not.

If that is the explanation, then restricting the BUMPS field to a saddle-like dynamic range
should drive its rho down to saddle-like values, using the same estimates already computed --
no change to the estimator, only to which points are compared.

This matters for transfer, which is the only reason to run a control at all. "The teacher works
at d=20 on some manifolds" is actionable. "The teacher works when the truth spans 1000x" is a
statement about fixtures, and PU's curvature field has no guarantee of that spread.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks"))

from pu_manifold import curvature_probe, synthetic_controls  # noqa: E402

D_INTRINSIC, SEED, N, D_AMBIENT, K = 20, 20260816, 10000, 28, 30


def windows(ne, nt, label):
    """rho within contiguous quantile windows of ||H_true||, reported with each window's own
    realized dynamic range so the comparison is spread-for-spread rather than name-for-name."""
    order = np.argsort(nt)
    print(f"\n  {label}")
    print(f"    {'window':>16} {'n':>6} {'spread p95/p05':>16} {'rank rho':>12}")
    print(f"    {'full':>16} {nt.size:>6} "
          f"{np.quantile(nt,0.95)/np.quantile(nt,0.05):>16.2f} "
          f"{float(spearmanr(ne, nt).statistic):>+12.6f}")
    for lo, hi in [(0.0, 0.5), (0.25, 0.75), (0.5, 1.0), (0.35, 0.65), (0.4, 0.6)]:
        idx = order[int(lo * nt.size):int(hi * nt.size)]
        w_nt, w_ne = nt[idx], ne[idx]
        spread = np.quantile(w_nt, 0.95) / max(np.quantile(w_nt, 0.05), 1e-300)
        print(f"    {f'q{lo:.2f}-q{hi:.2f}':>16} {idx.size:>6} {spread:>16.2f} "
              f"{float(spearmanr(w_ne, w_nt).statistic):>+12.6f}")


def get(name):
    if name == "saddle":
        fx = synthetic_controls.make_saddle_control(n=N, d=D_INTRINSIC, D=D_AMBIENT, seed=SEED)
    else:
        fx = curvature_probe.make_graph_of_function_fixture(
            n=N, d=D_INTRINSIC, D=D_AMBIENT, n_bumps=3, seed=SEED
        )
    out = curvature_probe.quadric_mean_curvature(fx["X"], k=K, d=D_INTRINSIC)
    return (np.linalg.norm(out["H_vec"], axis=-1),
            np.linalg.norm(fx["H_vec"], axis=-1))


print("=" * 88)
print("PROBE 3 -- does the bump fixture's rho survive a saddle-like dynamic range?")
print(f"d={D_INTRINSIC}  n={N}  k={K}  D={D_AMBIENT}")
print("=" * 88, flush=True)

s_ne, s_nt = get("saddle")
windows(s_ne, s_nt, "SADDLE (globally quadratic, full spread 33x)")

b_ne, b_nt = get("bumps")
windows(b_ne, b_nt, "BUMPS (not quadratic, full spread 1095x)")

print("\n" + "=" * 88)
print("READ-OUT")
print("=" * 88)
print("  Compare rows of comparable SPREAD across the two fixtures, not rows of the same name.")
print("  If the bumps' rho falls toward the saddle's once its dynamic range is cut to match,")
print("  then d=20 feasibility is a statement about the curvature field being ranked, not")
print("  about the teacher -- and it does not transfer to a manifold whose spread is unknown.")
