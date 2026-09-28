"""Spike 002, follow-up probe -- is the k=500 rank gain real geometry or a fixture artifact?

Part B measured rank `rho` rising monotonically with `k` at `d=20`: -0.028, +0.023, +0.084,
+0.393 at k = 30, 100, 231, 500. Taken at face value that is the spike's headline: `+0.393` is
3.4x phase 03.1's best decoder-side result (`+0.116`) and the largest ordering signal anything
in this project has produced at `d=20`.

Two reasons not to take it at face value.

CONFOUND 1 -- the saddle is GLOBALLY a quadratic form. `make_saddle_control` builds
`f(x) = 0.5 * x^T Q x` with `Q = diag(signs)`. A local-QUADRIC estimator whose neighbourhood has
grown to cover the entire manifold (`r/R = 1.0992` at k=500) is no longer fitting a local
approximation -- it is fitting the fixture's exact global model. If that is what the rank gain
is, it is a property of this fixture and transfers to nothing: PU is not a quadric.
  TEST: rerun the same `k` ladder on `make_graph_of_function_fixture`, a sum of Gaussian bumps,
  which has a spatially varying curvature field and is NOT globally quadratic. Everything else
  identical -- same `d`, `n`, `D`, seed, estimator, scorer. If `rho` at k=500 collapses there,
  the saddle number does not survive.

CONFOUND 2 -- the direction axis is NEGATIVE and gets worse as rank improves. Median cosine runs
-0.026, -0.151, -0.030, -0.3895 across the ladder. A rank correlation of +0.393 achieved by
vectors that point AWAY from the true mean curvature is not a curvature estimate that happens to
be imprecise; it is ordering extracted from something other than the geometry it claims. Reported
here as a distribution rather than a median, plus the spread of ||H_est||, so the shape of the
failure is visible rather than summarized.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks" / "diagnostics"))

import synthetic_control_run as scr  # noqa: E402
from pu_manifold import chart_curvature, curvature_probe, synthetic_controls  # noqa: E402

D_INTRINSIC = 20
SEED = 20260816
N = 10000
D_AMBIENT = 28          # licensed by Part A: D-invariant to 1.288e-14, 204x cheaper
KS = [30, 231, 500]


def saddle(n):
    fx = synthetic_controls.make_saddle_control(n=n, d=D_INTRINSIC, D=D_AMBIENT, seed=SEED)
    return fx["X"], fx["H_vec"]


def bumps(n):
    """Gaussian-bump graph of function -- spatially varying curvature, NOT globally quadratic.
    `n_bumps=3` keeps codimension small enough that D=28 still has room for the exact-zero
    padding the fixture relies on (d + n_bumps = 23 <= 28)."""
    fx = curvature_probe.make_graph_of_function_fixture(
        n=n, d=D_INTRINSIC, D=D_AMBIENT, n_bumps=3, seed=SEED
    )
    return fx["X"], fx["H_vec"]


def run(name, X, H_true, k):
    t0 = time.time()
    out = curvature_probe.quadric_mean_curvature(X, k=k, d=D_INTRINSIC)
    wall = time.time() - t0
    H_est = out["H_vec"]
    axes = scr._fidelity_axes(H_est, H_true)
    rep = chart_curvature.curvature_fidelity_report(H_est, H_true)

    cos = np.asarray(rep["cosine_similarity"])
    ne = np.linalg.norm(H_est, axis=-1)
    nt = np.linalg.norm(H_true, axis=-1)

    print(f"\n  {name}  k={k}")
    print(f"    rho={axes['rank_spearman_rho']:+.6f}  median cosine={axes['direction_median_cosine']:+.4f}  "
          f"median ratio={axes['magnitude_median_ratio']:.4f}  MRE={axes['median_relative_error']:.4f}  [{wall:.0f}s]")
    print(f"    cosine distribution: p05={np.quantile(cos,0.05):+.4f} p25={np.quantile(cos,0.25):+.4f} "
          f"p50={np.median(cos):+.4f} p75={np.quantile(cos,0.75):+.4f} p95={np.quantile(cos,0.95):+.4f}")
    print(f"    fraction of points with cosine < 0: {float((cos < 0).mean()):.4f}")
    print(f"    ||H_est||  p05={np.quantile(ne,0.05):.4e} p50={np.median(ne):.4e} p95={np.quantile(ne,0.95):.4e}  "
          f"spread p95/p05={np.quantile(ne,0.95)/max(np.quantile(ne,0.05),1e-300):.2f}")
    print(f"    ||H_true|| p05={np.quantile(nt,0.05):.4e} p50={np.median(nt):.4e} p95={np.quantile(nt,0.95):.4e}  "
          f"spread p95/p05={np.quantile(nt,0.95)/max(np.quantile(nt,0.05),1e-300):.2f}", flush=True)
    return axes["rank_spearman_rho"]


print("=" * 96)
print("PROBE -- does the k=500 rank gain survive a fixture that is not globally quadratic?")
print(f"d={D_INTRINSIC}  n={N}  D={D_AMBIENT}  seed={SEED}")
print("=" * 96, flush=True)

Xs, Hs = saddle(N)
Xb, Hb = bumps(N)

print("\n" + "-" * 96)
print("SADDLE -- f(x) = 0.5 x^T Q x, GLOBALLY QUADRATIC (the Part B fixture)")
print("-" * 96, flush=True)
saddle_rho = {k: run("saddle", Xs, Hs, k) for k in KS}

print("\n" + "-" * 96)
print("BUMPS -- sum of 3 Gaussian bumps, NOT globally quadratic")
print("-" * 96, flush=True)
bumps_rho = {k: run("bumps ", Xb, Hb, k) for k in KS}

print("\n" + "=" * 96)
print("HEAD TO HEAD -- rank rho by k")
print("=" * 96)
print(f"  {'k':>6} {'saddle (quadratic)':>22} {'bumps (not quadratic)':>24}")
for k in KS:
    print(f"  {k:>6} {saddle_rho[k]:>+22.6f} {bumps_rho[k]:>+24.6f}")

gain_saddle = saddle_rho[500] - saddle_rho[30]
gain_bumps = bumps_rho[500] - bumps_rho[30]
print(f"\n  rank gain from k=30 to k=500:  saddle {gain_saddle:+.6f}   bumps {gain_bumps:+.6f}")
if gain_bumps < 0.5 * gain_saddle:
    print("\n  CONFOUNDED. The rank gain is largely specific to the globally-quadratic saddle.")
    print("  A local-quadric estimator whose ball covers the manifold recovers that fixture's")
    print("  exact global model. It transfers to nothing that is not itself a quadric.")
else:
    print("\n  SURVIVES. The rank gain is present on a fixture with no global quadratic structure,")
    print("  so it is not an artifact of the saddle's own functional form.")
