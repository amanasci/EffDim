"""Spike 002, probe 2 -- is the bump fixture's rho = +0.593 curvature ordering, or local scale?

The confound probe found the teacher scoring `rho = +0.5934` on the Gaussian-bump fixture at the
sealed `k = 30`, against `-0.0281` on the saddle at the same `k`. Before that is reported as the
teacher carrying real geometry at `d = 20`, one alternative has to be excluded.

`02.5-NOTE-high-d-curvature-approaches.md` §1a records that at `d = 20` the estimator's error
"behaves like noise while being bias" -- the O(r^2) truncation is deterministic given the sample
but varies point to point, because each neighbourhood covers a large and differently-shaped chunk
of the manifold. The size and shape of that chunk is set by the LOCAL SAMPLING SCALE, and the
local sampling scale is not independent of the geometry: where a graph-of-function fixture is
steep, the induced density on the manifold thins out and the kNN ball grows.

So `||H_est||` could rank-correlate with `||H_true||` while estimating no curvature at all, if
both are separately driven by the local kNN radius. That is not a small effect to rule out: it
would make the teacher a density probe wearing a curvature convention.

TEST. Take the local kNN radius `r_i` (distance to the k-th neighbour) as a purely metric,
non-geometric proxy for local scale. Then compare:
  rho(||H_est||, ||H_true||)   -- the headline
  rho(||H_est||, r)            -- how much the estimate is just tracking scale
  rho(||H_true||, r)           -- how much the TRUTH is confounded with scale on this fixture
  partial rho(||H_est||, ||H_true|| | r)  -- the headline with scale's contribution removed

Spearman partial correlation, computed by rank-transforming all three and taking the residuals
of the two linear regressions on ranked `r`. If the partial correlation collapses toward zero,
the ordering was scale, not curvature. If it survives roughly intact, the teacher is ordering
something geometric and the finding stands.

Cheap: k=30 only, so both fixtures together run in well under a minute.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks"))

from pu_manifold import curvature_probe, synthetic_controls  # noqa: E402

D_INTRINSIC, SEED, N, D_AMBIENT, K = 20, 20260816, 10000, 28, 30


def partial_spearman(a, b, c):
    """Spearman partial correlation of `a` and `b` controlling for `c`: rank-transform all
    three, regress ranked `a` and ranked `b` each on ranked `c`, correlate the residuals."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    Xc = np.stack([rc, np.ones_like(rc)], axis=1)
    res_a = ra - Xc @ np.linalg.lstsq(Xc, ra, rcond=None)[0]
    res_b = rb - Xc @ np.linalg.lstsq(Xc, rb, rcond=None)[0]
    return float(np.corrcoef(res_a, res_b)[0, 1])


def analyse(name, X, H_true):
    nbrs = NearestNeighbors(n_neighbors=K + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    r = dist[:, -1]                                   # local kNN radius: pure local scale

    out = curvature_probe.quadric_mean_curvature(X, k=K, d=D_INTRINSIC)
    ne = np.linalg.norm(out["H_vec"], axis=-1)
    nt = np.linalg.norm(H_true, axis=-1)

    rho_headline = float(spearmanr(ne, nt).statistic)
    rho_est_r = float(spearmanr(ne, r).statistic)
    rho_true_r = float(spearmanr(nt, r).statistic)
    rho_partial = partial_spearman(ne, nt, r)

    print(f"\n  {name}  (d={D_INTRINSIC}, n={N}, k={K})")
    print(f"    rho(||H_est||, ||H_true||)            = {rho_headline:+.6f}   <- the headline")
    print(f"    rho(||H_est||, r_knn)                 = {rho_est_r:+.6f}   <- estimate vs local scale")
    print(f"    rho(||H_true||, r_knn)                = {rho_true_r:+.6f}   <- truth vs local scale")
    print(f"    partial rho(est, true | r_knn)        = {rho_partial:+.6f}   <- headline, scale removed")
    retained = rho_partial / rho_headline if abs(rho_headline) > 1e-9 else float("nan")
    print(f"    fraction of the headline retained     = {retained:.4f}")
    print(f"    local scale r_knn: p05={np.quantile(r,0.05):.4f} p50={np.median(r):.4f} "
          f"p95={np.quantile(r,0.95):.4f}  spread={np.quantile(r,0.95)/np.quantile(r,0.05):.2f}", flush=True)
    return rho_headline, rho_partial


print("=" * 92)
print("PROBE 2 -- is the teacher ordering curvature, or ordering local sampling scale?")
print("=" * 92, flush=True)

fx = synthetic_controls.make_saddle_control(n=N, d=D_INTRINSIC, D=D_AMBIENT, seed=SEED)
s_head, s_part = analyse("SADDLE (globally quadratic)", fx["X"], fx["H_vec"])

fx = curvature_probe.make_graph_of_function_fixture(
    n=N, d=D_INTRINSIC, D=D_AMBIENT, n_bumps=3, seed=SEED
)
b_head, b_part = analyse("BUMPS  (not globally quadratic)", fx["X"], fx["H_vec"])

print("\n" + "=" * 92)
print("READ-OUT")
print("=" * 92)
print(f"  bumps headline rho              = {b_head:+.6f}")
print(f"  bumps partial rho (scale out)   = {b_part:+.6f}")
if abs(b_part) < 0.5 * abs(b_head):
    print("\n  SCALE-CONFOUNDED. More than half the teacher's apparent curvature ordering is")
    print("  explained by the local kNN radius -- a purely metric quantity carrying no")
    print("  second-order information. The teacher is substantially a density probe here.")
else:
    print("\n  SURVIVES SCALE. The ordering is not merely the local sampling scale, so the")
    print("  teacher is ranking something genuinely second-order on this fixture.")
