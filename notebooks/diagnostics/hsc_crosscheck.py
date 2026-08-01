"""
Cross-survey check of the Phase 2 gate.

Every result in 02-FINDINGS.md rests on ONE column: `legacysurvey`. The same config
carries a paired `hsc` column -- same objects row-for-row, same DINOv3 ViT-B/16 model,
different sky survey -- already subsampled and normalized in cache.

Question: is the strong non-Euclideanity a property of the MODEL's embedding space, or
of the Legacy Survey column specifically?

  same m on HSC   -> property of the model/embedding space (or of the object population)
  much lower m    -> Legacy-Survey-specific; the LS finding does not generalize

Not pre-registered. Diagnostic. Reuses Phase 2's r/m definitions and thresholds verbatim;
revises nothing.
"""
import gc
import time

import numpy as np
from scipy.linalg import eigvalsh
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

CACHE = "notebooks/.cache"
R_MAX_PASS, M_MAX_PASS = 0.10, 0.05
R_MAX_MARGINAL, M_MAX_MARGINAL = 0.25, 0.15
K, NCOMP, SEED = 15, 18, 20260729


def gate_stats(ev):
    neg, pos = ev[ev < 0], ev[ev > 0]
    return (abs(neg.min()) / pos.max(),
            np.abs(neg).sum() / np.abs(ev).sum(),
            int(pos.size), int(neg.size), float(pos.max()), float(neg.min()))


def classify(r, m):
    if r < R_MAX_PASS and m < M_MAX_PASS:
        return "PASS"
    if r < R_MAX_MARGINAL and m < M_MAX_MARGINAL:
        return "MARGINAL"
    return "FAIL"


def spectrum(D):
    D2 = np.array(D, dtype=np.float64, copy=True)
    D2 **= 2
    D2 -= D2.mean(axis=1, keepdims=True)
    D2 -= D2.mean(axis=0, keepdims=True)
    D2 += D2.mean()
    D2 *= -0.5
    sym = float(np.abs(D2 - D2.T).max())
    ev = eigvalsh(D2)
    del D2
    gc.collect()
    return ev, sym


def twonn(X, sample=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    nn = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X)
    d, _ = nn.kneighbors(X[idx])
    r1, r2 = d[:, 1], d[:, 2]
    ok = (r1 > 0) & (r2 > r1)
    return float(ok.sum() / np.log(r2[ok] / r1[ok]).sum())


def local_pca_dim(X, n_centers=600, k=60, var=0.90, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.choice(len(X), size=n_centers, replace=False)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    _, nbr = nn.kneighbors(X[centers])
    out = np.empty(n_centers, dtype=np.int64)
    for i, row in enumerate(nbr):
        P = X[row] - X[row].mean(axis=0)
        ev = np.linalg.svd(P, compute_uv=False) ** 2
        out[i] = int(np.searchsorted(np.cumsum(ev) / ev.sum(), var) + 1)
    return out


z = np.load(f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz")
HSC, LS = z["hsc"], z["legacysurvey"]
print("=" * 78)
print("CROSS-SURVEY GATE CHECK -- HSC column, same objects, same DINOv3 ViT-B/16")
print("=" * 78)
print(f"  hsc          {HSC.shape}  row-norm mean={np.linalg.norm(HSC,axis=1).mean():.6f}")
print(f"  legacysurvey {LS.shape}  row-norm mean={np.linalg.norm(LS,axis=1).mean():.6f}")
print(f"  raw hsc norms: mean={z['hsc_norms'].mean():.4f} std={z['hsc_norms'].std():.4f} "
      f"cv={z['hsc_norms'].std()/z['hsc_norms'].mean():.4f}")
print(f"  raw ls  norms: mean={z['ls_norms'].mean():.4f} std={z['ls_norms'].std():.4f} "
      f"cv={z['ls_norms'].std()/z['ls_norms'].mean():.4f}")
print(f"  columns identical? {np.array_equal(HSC, LS)}   "
      f"mean |hsc-ls| = {np.abs(HSC-LS).mean():.6f}")

print()
print(f"FITTING Isomap on HSC (k={K}, n_components={NCOMP}, dense)...")
t0 = time.perf_counter()
iso = Isomap(n_neighbors=K, n_components=NCOMP, eigen_solver="dense", n_jobs=-1)
iso.fit(HSC)
t_fit = time.perf_counter() - t0
t0 = time.perf_counter()
ev_hsc, sym = spectrum(iso.dist_matrix_)
t_eig = time.perf_counter() - t0
print(f"  fit {t_fit:.1f}s   eigensolve {t_eig:.1f}s   symmetry {sym:.3e}   "
      f"len={ev_hsc.shape[0]} {ev_hsc.dtype}")
del iso
gc.collect()

r_h, m_h, p_h, n_h, mx_h, mn_h = gate_stats(ev_hsc)
noise_h = ev_hsc.shape[0] * np.finfo(np.float64).eps * mx_h

# published LS baseline, from cache
ev_ls = np.load(f"{CACHE}/mds_eigenspectrum_43cf438bc944c509.npz")["eigvals_all"]
r_l, m_l, p_l, n_l, mx_l, mn_l = gate_stats(ev_ls)

print()
print("=" * 78)
print("GATE RESULT")
print("=" * 78)
print(f"  {'':22s} {'legacysurvey (published)':>26s} {'hsc (this run)':>20s}")
print(f"  {'r':22s} {r_l:>26.6f} {r_h:>20.6f}")
print(f"  {'m':22s} {m_l:>26.6f} {m_h:>20.6f}")
print(f"  {'positive':22s} {p_l:>26d} {p_h:>20d}")
print(f"  {'negative':22s} {n_l:>26d} {n_h:>20d}")
print(f"  {'lambda_max_pos':22s} {mx_l:>26.6e} {mx_h:>20.6e}")
print(f"  {'lambda_min_neg':22s} {mn_l:>26.6e} {mn_h:>20.6e}")
print(f"  {'verdict':22s} {classify(r_l,m_l):>26s} {classify(r_h,m_h):>20s}")
print()
print(f"  noise floor (hsc) = {noise_h:.4e}   |lambda_min_neg| = {abs(mn_h):.4e}   "
      f"ratio = {abs(mn_h)/noise_h:.3e}")
print(f"  >>> m difference: {m_h - m_l:+.6f}  ({100*(m_h-m_l)/m_l:+.2f}% vs legacysurvey)")

print()
print("=" * 78)
print("INTRINSIC DIMENSION -- both columns")
print("=" * 78)
for label, X in (("legacysurvey", LS), ("hsc", HSC)):
    d2 = twonn(X, seed=0)
    dims = local_pca_dim(X, seed=0)
    q = np.percentile(dims, [5, 50, 95])
    print(f"  [{label:13s}] TwoNN={d2:6.3f}   localPCA median={np.median(dims):5.1f} "
          f"mean={dims.mean():6.2f} std={dims.std():5.2f}  "
          f"5/50/95={q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}  range={dims.min()}-{dims.max()}")

np.savez_compressed(f"{CACHE}/hsc_spectrum_diagnostic.npz", eigvals_all=ev_hsc)
print()
print("  spectrum saved -> notebooks/.cache/hsc_spectrum_diagnostic.npz")
