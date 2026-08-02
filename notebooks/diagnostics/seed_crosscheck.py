"""
Disjoint-sample check of the Phase 2 gate: draws a fresh 10,000 rows from the same
101,725-row config under a different seed (20260801 vs. the original 20260729) and re-runs
the gate on the legacysurvey column -- same model, same survey, only the objects change.
Exercises the pre-registration's remediation option 2 (resample with a new seed), closing
whether "property of the model's embedding geometry" and "property of these particular
10,000 objects" were confounded.

  same m on a disjoint draw -> not a property of the original 10k; sampling variance closed
  much lower m              -> the original draw was unrepresentative

Not pre-registered. Diagnostic. Reuses Phase 2's r/m definitions and thresholds verbatim.
"""
import gc
import time

import numpy as np
from scipy.linalg import eigvalsh
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

from pu_manifold import load_subsample

CACHE = "notebooks/.cache"
R_MAX_PASS, M_MAX_PASS = 0.10, 0.05
R_MAX_MARGINAL, M_MAX_MARGINAL = 0.25, 0.15
K, NCOMP = 15, 18
SEED_ORIG, SEED_NEW = 20260729, 20260801


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


BASE_CFG = {
    "dataset": "legacysurvey_dinov3_vitb16",
    "n_rows": 10000,
    "normalize": True,
}

print("=" * 78)
print(f"DISJOINT-SAMPLE GATE CHECK -- new seed {SEED_NEW} vs original {SEED_ORIG}")
print("=" * 78)

t0 = time.perf_counter()
new = load_subsample({**BASE_CFG, "seed": SEED_NEW})
print(f"  subsample loaded in {time.perf_counter()-t0:.1f}s")

orig = np.load(f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz")
idx_o, idx_n = set(orig["row_indices"].tolist()), set(new["row_indices"].tolist())
overlap = len(idx_o & idx_n)
print(f"  original rows: {len(idx_o)}   new rows: {len(idx_n)}")
print(f"  overlap: {overlap} rows ({100*overlap/len(idx_n):.2f}%)  "
      f"[chance expectation ~{100*10000/101725:.2f}% for independent draws]")

LS_NEW = new["legacysurvey"]
print(f"  legacysurvey (new) {LS_NEW.shape}  row-norm mean="
      f"{np.linalg.norm(LS_NEW,axis=1).mean():.6f}")

print()
print(f"FITTING Isomap (k={K}, n_components={NCOMP}, dense)...")
t0 = time.perf_counter()
iso = Isomap(n_neighbors=K, n_components=NCOMP, eigen_solver="dense", n_jobs=-1)
iso.fit(LS_NEW)
t_fit = time.perf_counter() - t0
t0 = time.perf_counter()
ev_new, sym = spectrum(iso.dist_matrix_)
t_eig = time.perf_counter() - t0
print(f"  fit {t_fit:.1f}s   eigensolve {t_eig:.1f}s   symmetry {sym:.3e}   "
      f"len={ev_new.shape[0]} {ev_new.dtype}")
del iso
gc.collect()

r_n, m_n, p_n, n_n, mx_n, mn_n = gate_stats(ev_new)
noise_n = ev_new.shape[0] * np.finfo(np.float64).eps * mx_n

ev_o = np.load(f"{CACHE}/mds_eigenspectrum_43cf438bc944c509.npz")["eigvals_all"]
r_o, m_o, p_o, n_o, mx_o, mn_o = gate_stats(ev_o)

print()
print("=" * 78)
print("GATE RESULT")
print("=" * 78)
print(f"  {'':22s} {'seed 20260729 (published)':>27s} {'seed 20260801':>18s}")
print(f"  {'r':22s} {r_o:>27.6f} {r_n:>18.6f}")
print(f"  {'m':22s} {m_o:>27.6f} {m_n:>18.6f}")
print(f"  {'positive':22s} {p_o:>27d} {p_n:>18d}")
print(f"  {'negative':22s} {n_o:>27d} {n_n:>18d}")
print(f"  {'lambda_max_pos':22s} {mx_o:>27.6e} {mx_n:>18.6e}")
print(f"  {'lambda_min_neg':22s} {mn_o:>27.6e} {mn_n:>18.6e}")
print(f"  {'verdict':22s} {classify(r_o,m_o):>27s} {classify(r_n,m_n):>18s}")
print()
print(f"  noise floor = {noise_n:.4e}   |lambda_min_neg| = {abs(mn_n):.4e}   "
      f"ratio = {abs(mn_n)/noise_n:.3e}")
print(f"  >>> m difference: {m_n - m_o:+.6f}  ({100*(m_n-m_o)/m_o:+.2f}%)")

print()
print("=" * 78)
print("INTRINSIC DIMENSION")
print("=" * 78)
for label, X in (("seed 20260729", orig["legacysurvey"]), ("seed 20260801", LS_NEW)):
    d2 = twonn(X, seed=0)
    dims = local_pca_dim(X, seed=0)
    q = np.percentile(dims, [5, 50, 95])
    print(f"  [{label:13s}] TwoNN={d2:6.3f}   localPCA median={np.median(dims):5.1f} "
          f"mean={dims.mean():6.2f} std={dims.std():5.2f}  "
          f"5/50/95={q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}  range={dims.min()}-{dims.max()}")

np.savez_compressed(f"{CACHE}/seed{SEED_NEW}_spectrum_diagnostic.npz", eigvals_all=ev_new)
print(f"\n  spectrum saved -> notebooks/.cache/seed{SEED_NEW}_spectrum_diagnostic.npz")
