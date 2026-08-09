#!/usr/bin/env python3
"""Density-calibrated local curvature estimation.

The problem this module exists to solve: every fixed-k local-PCA curvature
measure is confounded with density, because a fixed-k neighbourhood has a larger
physical radius in sparse regions. Two large biases fight each other:

  * noise-floor bias  - with manifold thickness sigma, dense regions have
    neighbourhood radius ~ sigma so the PCA residual fraction is high, while
    sparse regions have radius >> sigma so it is low. Drives rho(d_k, rf) < 0.
  * bending bias      - a bigger neighbourhood samples more of the bend, so the
    residual fraction rises with radius. Drives rho(d_k, rf) > 0.

Naive "scale-normalised" estimators such as kappa ~ 2||r||/||u||^2 do not fix
this: the noise floor puts a constant on ||r||, so kappa_hat ~ 2*sigma/R^2,
which is an almost perfect proxy for 1/d_k^2.

The fix used here is a *per-point permutation null*. We fit a local quadratic
(second fundamental form) to the neighbourhood, then refit the identical design
after destroying the pairing between tangent position and normal displacement.
The ratio kappa_jet / kappa_null is calibrated at each point's own scale, own
noise level and own neighbour count, so it is density-matched by construction
rather than by subsampling (which is impossible at this intrinsic dimension).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_knn",
    "quad_monomial_pairs",
    "jet_curvature_point",
    "compute_curvature_suite",
    "flatten_null",
    "synthetic_manifold",
]


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------

def build_knn(X: np.ndarray, k: int, block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) of the k nearest neighbours, self excluded.

    Neighbours come back sorted by increasing distance, so a k-column result can
    be sliced to any smaller scale without rebuilding the graph.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be < n={n}")

    try:
        import faiss

        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        d2, idx = index.search(X, k + 1)
        return np.sqrt(np.maximum(d2[:, 1:], 0.0)), idx[:, 1:]
    except ImportError:
        pass

    sq = (X ** 2).sum(axis=1)
    dists = np.empty((n, k), dtype=np.float32)
    idxs = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = sq[s:e, None] - 2.0 * (X[s:e] @ X.T) + sq[None, :]
        d2[np.arange(e - s), np.arange(s, e)] = np.inf  # drop self
        part = np.argpartition(d2, k, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        pd = d2[rows, part]
        order = np.argsort(pd, axis=1)
        idxs[s:e] = part[rows, order]
        dists[s:e] = np.sqrt(np.maximum(pd[rows, order], 0.0))
    return dists, idxs


# ---------------------------------------------------------------------------
# Local jet fit with a permutation null
# ---------------------------------------------------------------------------

def quad_monomial_pairs(p_quad: int) -> list[tuple[int, int]]:
    """Index pairs (a, b) with a <= b for the quadratic monomials u_a * u_b."""
    return [(a, b) for a in range(p_quad) for b in range(a, p_quad)]


def _hessian_norm_sq(beta_quad: np.ndarray, pairs: list[tuple[int, int]], p_quad: int) -> float:
    """||H||_F^2 summed over normal directions, from quadratic-monomial coefficients.

    A fit z_m = sum_{a<=b} beta_ab^m u_a u_b corresponds to z_m = 1/2 u^T H^m u
    with H^m_aa = 2 beta_aa^m and H^m_ab = H^m_ba = beta_ab^m for a != b. Writing
    it as a Hessian (rather than reading ||beta||_F directly) is what makes the
    estimator equal 1/R on a sphere of radius R.
    """
    total = 0.0
    for row, (a, b) in enumerate(pairs):
        coef = beta_quad[row]
        if a == b:
            total += float(((2.0 * coef) ** 2).sum())
        else:
            total += 2.0 * float((coef ** 2).sum())  # H_ab and H_ba
    return total / max(p_quad, 1)


def jet_curvature_point(
    u: np.ndarray,
    Y: np.ndarray,
    p_quad: int,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Local second-fundamental-form magnitude, calibrated by a permutation null.

    Parameters
    ----------
    u : (K, p_lin) tangential coordinates of the neighbours
    Y : (K, m_norm) normal coordinates of the neighbours
    p_quad : quadratic block width; curvature is measured in the top-p_quad
        tangent directions (a sectional-curvature proxy). The null uses the
        identical design, so the ratio stays calibrated.

    Returns kappa_jet, kappa_null, kappa_ratio, kappa_z, r2_quad.
    """
    K = u.shape[0]
    pairs = quad_monomial_pairs(p_quad)

    quad = np.empty((K, len(pairs)), dtype=np.float64)
    for j, (a, b) in enumerate(pairs):
        quad[:, j] = u[:, a] * u[:, b]

    ones = np.ones((K, 1), dtype=np.float64)
    lin = np.hstack([ones, u])
    design = np.hstack([lin, quad])

    # Linear-only fit: what the permutation null is allowed to keep.
    beta_lin, *_ = np.linalg.lstsq(lin, Y, rcond=None)
    Y_resid = Y - lin @ beta_lin

    pinv = np.linalg.pinv(design)  # (n_cols, K), reused by every replicate
    n_lin = lin.shape[1]

    beta = pinv @ Y
    kappa_jet = np.sqrt(_hessian_norm_sq(beta[n_lin:], pairs, p_quad))

    ss_res = float(((Y - design @ beta) ** 2).sum())
    ss_tot = float((Y_resid ** 2).sum())
    r2_quad = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    nulls = np.empty(n_perm, dtype=np.float64)
    for t in range(n_perm):
        Yp = Y_resid[rng.permutation(K)]
        beta_p = pinv @ Yp
        nulls[t] = np.sqrt(_hessian_norm_sq(beta_p[n_lin:], pairs, p_quad))

    null_mean = float(nulls.mean())
    null_sd = float(nulls.std())
    return {
        "kappa_jet": float(kappa_jet),
        "kappa_null": null_mean,
        "kappa_ratio": float(kappa_jet / null_mean) if null_mean > 1e-30 else np.nan,
        "kappa_z": float((kappa_jet - null_mean) / null_sd) if null_sd > 1e-30 else np.nan,
        "r2_quad": float(r2_quad),
    }


# ---------------------------------------------------------------------------
# Per-point suite at one scale
# ---------------------------------------------------------------------------

SUITE_KEYS = (
    "kappa_jet", "kappa_null", "kappa_ratio", "kappa_z", "r2_quad",
    "rf_k", "kappa_naive_ratio", "kappa_slope", "noise_floor", "R_med",
)


def _suite_chunk(
    X: np.ndarray,
    knn_idx: np.ndarray,
    rows: np.ndarray,
    k_t: int,
    p_quad: int,
    m_norm: int,
    n_perm: int,
    seed: int,
    progress_every: int,
) -> dict[str, np.ndarray]:
    """Curvature metrics for a subset of points. See compute_curvature_suite."""
    n, D = X.shape
    K = knn_idx.shape[1]
    p_lin = min(k_t, K - 1, D)
    out = {key: np.full(len(rows), np.nan, dtype=np.float64) for key in SUITE_KEYS}

    for pos, i in enumerate(rows):
        # Seeded per point rather than per chunk, so results do not depend on
        # how the work was split across workers.
        rng = np.random.default_rng([seed, int(i)])
        nb = X[knn_idx[i]]
        C = nb - nb.mean(axis=0)

        G = C @ C.T
        w, U = np.linalg.eigh(G)
        w = w[::-1]
        U = U[:, ::-1]
        w = np.maximum(w, 0.0)

        total = w.sum()
        if total <= 1e-30:
            continue

        # PCA residual fraction: identical to multiscale_curvature_probe's
        # residual_fraction, but free here since we already have the spectrum.
        k_rf = min(k_t, K - 1, D)
        out["rf_k"][pos] = 1.0 - w[:k_rf].sum() / total

        s = np.sqrt(w)
        u = U[:, :p_lin] * s[:p_lin]                        # tangential coords
        Y = U[:, p_lin:p_lin + m_norm] * s[p_lin:p_lin + m_norm]  # normal coords

        if Y.shape[1] == 0:
            continue

        # Exact norms from the spectrum: no need to reconstruct in D-space.
        sq_all = (C ** 2).sum(axis=1)
        sq_tan = (u ** 2).sum(axis=1)
        r_norm = np.sqrt(np.maximum(sq_all - sq_tan, 0.0))
        u_norm = np.sqrt(sq_tan)
        out["R_med"][pos] = float(np.median(np.sqrt(sq_all)))

        # Negative controls: the naive scale-normalised estimators.
        good = u_norm > 1e-12
        if good.sum() >= 3:
            out["kappa_naive_ratio"][pos] = float(
                np.median(2.0 * r_norm[good] / (u_norm[good] ** 2))
            )
            xh = 0.5 * (u_norm[good] ** 2)
            A = np.vstack([np.ones_like(xh), xh]).T
            coef, *_ = np.linalg.lstsq(A, r_norm[good], rcond=None)
            out["noise_floor"][pos] = float(coef[0])
            out["kappa_slope"][pos] = float(coef[1])

        res = jet_curvature_point(u, Y, p_quad, n_perm, rng)
        for key, val in res.items():
            out[key][pos] = val

        if progress_every and (pos + 1) % progress_every == 0:
            print(f"    ... {pos + 1}/{len(rows)} points in chunk", flush=True)

    return out


def compute_curvature_suite(
    X: np.ndarray,
    knn_idx: np.ndarray,
    k_t: int,
    *,
    p_quad: int = 3,
    m_norm: int = 5,
    n_perm: int = 16,
    seed: int = 0,
    progress_every: int = 1000,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """All per-point curvature metrics at the single scale K = knn_idx.shape[1].

    One symmetric eigendecomposition of the (K, K) neighbour Gram matrix yields
    everything: the tangent basis, the normal coordinates and the PCA residual
    fraction. Working through the Gram matrix rather than an SVD of the (K, D)
    displacement block is ~3.7x faster at D=768 and numerically equivalent.

    The point loop is embarrassingly parallel and each point's matrices are small
    enough that multithreaded BLAS is a large net loss (measured: ~20x slowdown
    from thread oversubscription). Callers should pin BLAS to one thread per
    worker and parallelise here instead.
    """
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    K = knn_idx.shape[1]
    p_lin = min(k_t, K - 1, D)
    if p_lin < p_quad:
        raise ValueError(f"p_lin={p_lin} must be >= p_quad={p_quad}")

    args = (k_t, p_quad, m_norm, n_perm, seed)
    if n_jobs == 1:
        merged = _suite_chunk(X, knn_idx, np.arange(n), *args, progress_every)
    else:
        from joblib import Parallel, delayed

        chunks = np.array_split(np.arange(n), n_jobs * 4)
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_suite_chunk)(X, knn_idx, c, *args, 0) for c in chunks
        )
        merged = {k: np.concatenate([p[k] for p in parts]) for k in SUITE_KEYS}

    return {k: v.astype(np.float32) for k, v in merged.items()}


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------

def flatten_null(X: np.ndarray, k_t: int, mode: str = "gauss", seed: int = 0) -> np.ndarray:
    """A perfectly FLAT surrogate carrying the real data's density structure.

    Projects X onto its global top-k_t PCA subspace and rebuilds it there, so
    the tangent-coordinate distribution (and hence the density variation that
    the experiment stratifies on) is exactly the real one while the manifold is
    a linear subspace by construction. Any density trend a metric shows here is
    pure artifact.

    mode='gauss'   isotropic Gaussian off-subspace noise matched to the real
                   off-subspace variance.
    mode='shuffle' the real off-subspace residuals, row-permuted across points,
                   which preserves their marginal distribution exactly.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu

    cov = (Xc.T @ Xc) / max(n - 1, 1)
    w, Q = np.linalg.eigh(cov)
    w = w[::-1]
    Q = Q[:, ::-1]
    k_t = min(k_t, D - 1)

    P = Q[:, :k_t]
    flat = (Xc @ P) @ P.T

    if mode == "shuffle":
        resid = Xc - flat
        return (flat + resid[rng.permutation(n)] + mu).astype(np.float32)
    if mode == "gauss":
        sigma = np.sqrt(max(w[k_t:].mean(), 0.0))
        E = rng.standard_normal((n, D - k_t)) * sigma
        return (flat + E @ Q[:, k_t:].T + mu).astype(np.float32)
    raise ValueError(f"unknown flatten_null mode: {mode!r}")


def synthetic_manifold(
    n: int,
    D: int,
    d: int,
    *,
    kind: str = "flat",
    curvature: float = 0.3,
    noise: float = 0.01,
    density_tilt: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Ground-truth control manifolds with deliberately non-uniform density.

    kind='flat'    a linear d-plane. True curvature 0. Sampled from a lognormal
                   scale mixture so density varies strongly.
    kind='sphere'  a d-sphere of radius R = 1/curvature, i.e. CONSTANT curvature
                   everywhere, with density tilted along one axis. This is the
                   strong test: a good estimator must both detect the curvature
                   and report it as flat across density quartiles.

    Returns (X, true_kappa).
    """
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.standard_normal((D, D)))[0]

    if kind == "flat":
        scale = np.exp(rng.normal(0.0, density_tilt, size=(n, 1)))
        T = rng.standard_normal((n, d)) * scale
        X = T @ A[:, :d].T
        true_kappa = 0.0
    elif kind == "sphere":
        R = 1.0 / max(curvature, 1e-9)
        V = rng.standard_normal((n * 4, d + 1))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        # Importance-resample to tilt the density along the first axis.
        logw = density_tilt * V[:, 0]
        p = np.exp(logw - logw.max())
        keep = rng.choice(len(V), size=n, replace=False, p=p / p.sum())
        X = (V[keep] * R) @ A[:, : d + 1].T
        true_kappa = curvature
    else:
        raise ValueError(f"unknown synthetic manifold kind: {kind!r}")

    X = X + rng.standard_normal((n, D)) * noise
    return X.astype(np.float32), float(true_kappa)
