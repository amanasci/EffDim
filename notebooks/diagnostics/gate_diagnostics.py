"""
Diagnostic triage for the Phase 2 gate FAIL. Two questions:

  T1. Is the negative eigenvalue mass an artifact of L2 normalization (unit-hypersphere
      curvature), or does it survive on the raw unnormalized embeddings?
  T2. Is the cloud well-described by a low-dimensional manifold at all, or is the
      intrinsic dimension unstable/high enough that graph geodesics are a poor metric?

Not a pre-registered gate. Diagnostic only. Thresholds from Phase 2 are reused verbatim
for comparability, never revised.
"""
import gc
import json
import time

import numpy as np
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigvalsh

CACHE = "notebooks/.cache"
R_MAX_PASS, M_MAX_PASS = 0.10, 0.05
R_MAX_MARGINAL, M_MAX_MARGINAL = 0.25, 0.15


def gate_stats(ev):
    """r and m exactly as Phase 2 defines them. Strict comparisons against zero."""
    neg, pos = ev[ev < 0], ev[ev > 0]
    r = abs(neg.min()) / pos.max()
    m = np.abs(neg).sum() / np.abs(ev).sum()
    return r, m, int(pos.size), int(neg.size), float(pos.max()), float(neg.min())


def classify(r, m):
    if r < R_MAX_PASS and m < M_MAX_PASS:
        return "PASS"
    if r < R_MAX_MARGINAL and m < M_MAX_MARGINAL:
        return "MARGINAL"
    return "FAIL"


def spectrum_from_distmatrix(D):
    """Full classical-MDS eigenspectrum by mean-form double-centring. float64 throughout."""
    D2 = np.array(D, dtype=np.float64, copy=True)   # copy=True: asarray on a memmap returns a view
    D2 **= 2
    row = D2.mean(axis=1, keepdims=True)
    col = D2.mean(axis=0, keepdims=True)
    tot = D2.mean()
    D2 -= row
    D2 -= col
    D2 += tot
    D2 *= -0.5
    sym = np.abs(D2 - D2.T).max()
    ev = eigvalsh(D2)
    del D2
    gc.collect()
    return ev, float(sym)


def twonn(X, sample=None, seed=0):
    """Facco et al. two-NN intrinsic dimension estimator. Cheap and fairly robust."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample or len(X), len(X)), replace=False)
    nn = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X)
    d, _ = nn.kneighbors(X[idx])
    r1, r2 = d[:, 1], d[:, 2]
    ok = (r1 > 0) & (r2 > r1)
    mu = r2[ok] / r1[ok]
    # ML estimator: d = N / sum(log mu)
    return float(ok.sum() / np.log(mu).sum())


def local_pca_dim(X, n_centers=600, k=60, var=0.90, seed=0):
    """Local PCA dimension: components needed for `var` of local variance, per neighbourhood."""
    rng = np.random.default_rng(seed)
    centers = rng.choice(len(X), size=n_centers, replace=False)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    _, nbr = nn.kneighbors(X[centers])
    dims = np.empty(n_centers, dtype=np.int64)
    for i, row in enumerate(nbr):
        P = X[row]
        P = P - P.mean(axis=0)
        s = np.linalg.svd(P, compute_uv=False)
        ev = s ** 2
        c = np.cumsum(ev) / ev.sum()
        dims[i] = int(np.searchsorted(c, var) + 1)
    return dims


print("=" * 78)
print("LOADING")
print("=" * 78)
z = np.load(f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz")
LS_NORM = z["legacysurvey"]            # L2-normalized, what Phase 1/2 fitted on
LS_NORMS = z["ls_norms"]               # original norms -> exact inversion
LS_RAW = LS_NORM * LS_NORMS[:, None]   # unnormalized embeddings
print(f"normalized  {LS_NORM.shape} {LS_NORM.dtype}   row-norm mean={np.linalg.norm(LS_NORM,axis=1).mean():.6f}")
print(f"raw         {LS_RAW.shape} {LS_RAW.dtype}   row-norm mean={np.linalg.norm(LS_RAW,axis=1).mean():.4f} "
      f"std={np.linalg.norm(LS_RAW,axis=1).std():.4f}")
print(f"norm spread: min={LS_NORMS.min():.4f} max={LS_NORMS.max():.4f} "
      f"cv={LS_NORMS.std()/LS_NORMS.mean():.4f}")

# ---- control: reproduce the published numbers from the cached spectrum ----
print()
print("=" * 78)
print("CONTROL -- reproduce published gate stats from the cached normalized spectrum")
print("=" * 78)
ev_pub = np.load(f"{CACHE}/mds_eigenspectrum_43cf438bc944c509.npz")["eigvals_all"]
r0, m0, p0, n0, mx0, mn0 = gate_stats(ev_pub)
print(f"  r={r0:.6f}  m={m0:.6f}  pos={p0} neg={n0}  verdict={classify(r0,m0)}")
print(f"  expected r=0.052419 m=0.412071  ->  "
      f"{'MATCH' if abs(r0-0.052419)<1e-6 and abs(m0-0.412071)<1e-6 else 'MISMATCH'}")

# ---- T1: unnormalized refit ----
print()
print("=" * 78)
print("T1 -- UNNORMALIZED REFIT  (k=15, n_components=18, dense; same rows, same seed)")
print("=" * 78)
t0 = time.perf_counter()
iso = Isomap(n_neighbors=15, n_components=18, eigen_solver="dense", n_jobs=-1)
iso.fit(LS_RAW)
t_fit = time.perf_counter() - t0
print(f"  fit: {t_fit:.1f}s")

t0 = time.perf_counter()
ev_raw, sym_raw = spectrum_from_distmatrix(iso.dist_matrix_)
t_eig = time.perf_counter() - t0
print(f"  eigensolve: {t_eig:.1f}s   symmetry max|D-D.T|={sym_raw:.3e}   len={ev_raw.shape[0]} {ev_raw.dtype}")

r1, m1, p1, n1, mx1, mn1 = gate_stats(ev_raw)
noise1 = ev_raw.shape[0] * np.finfo(np.float64).eps * mx1
print()
print(f"  {'':22s} {'NORMALIZED (published)':>24s} {'UNNORMALIZED (this run)':>26s}")
print(f"  {'r':22s} {r0:>24.6f} {r1:>26.6f}")
print(f"  {'m':22s} {m0:>24.6f} {m1:>26.6f}")
print(f"  {'positive eigenvalues':22s} {p0:>24d} {p1:>26d}")
print(f"  {'negative eigenvalues':22s} {n0:>24d} {n1:>26d}")
print(f"  {'lambda_max_pos':22s} {mx0:>24.6e} {mx1:>26.6e}")
print(f"  {'lambda_min_neg':22s} {mn0:>24.6e} {mn1:>26.6e}")
print(f"  {'verdict':22s} {classify(r0,m0):>24s} {classify(r1,m1):>26s}")
print()
print(f"  noise floor (unnorm) = {noise1:.6e}   |lambda_min_neg| = {abs(mn1):.6e}   "
      f"ratio = {abs(mn1)/noise1:.3e}")
print()
print(f"  >>> m changed by {m1-m0:+.6f} ({100*(m1-m0)/m0:+.2f}%) on removing L2 normalization")

np.savez_compressed("notebooks/.cache/unnorm_spectrum_diagnostic.npz",
                    eigvals_all=ev_raw)
del iso
gc.collect()

# ---- T2: manifold assumption ----
print()
print("=" * 78)
print("T2 -- MANIFOLD ASSUMPTION  (intrinsic dimension, both spaces)")
print("=" * 78)
for label, X in (("normalized", LS_NORM), ("unnormalized", LS_RAW)):
    t0 = time.perf_counter()
    d_twonn = twonn(X, sample=4000, seed=0)
    dims = local_pca_dim(X, n_centers=600, k=60, var=0.90, seed=0)
    dt = time.perf_counter() - t0
    q = np.percentile(dims, [5, 25, 50, 75, 95])
    print(f"\n  [{label}]  ({dt:.1f}s)")
    print(f"    TwoNN intrinsic dimension          = {d_twonn:.3f}")
    print(f"    local PCA dim (90% var, k=60):")
    print(f"      median={np.median(dims):.1f}  mean={dims.mean():.2f}  std={dims.std():.2f}")
    print(f"      percentiles  5%={q[0]:.0f}  25%={q[1]:.0f}  50%={q[2]:.0f}  "
          f"75%={q[3]:.0f}  95%={q[4]:.0f}")
    print(f"      min={dims.min()}  max={dims.max()}  spread(95-5)={q[4]-q[0]:.0f}")
    print(f"      fraction of neighbourhoods needing >18 dims = "
          f"{(dims > 18).mean():.3f}")
    print(f"      fraction needing >40 dims                   = {(dims > 40).mean():.3f}")

print()
print("=" * 78)
print("REFERENCE POINTS")
print("=" * 78)
print("  Phase 1 D_PROVISIONAL (median of 8 geometric estimators) = 18")
print("  Phase 2 residual-curve elbow (Tenenbaum, canonical)      = 5")
print("  Phase 2 n_components used for every fit                  = 18")
print("  A clean low-d manifold: local PCA dim tight around the true d, small spread.")
print("  A cloud that is not a manifold: high dim, wide spread, k-dependent.")
