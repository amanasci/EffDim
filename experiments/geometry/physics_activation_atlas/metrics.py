"""Reconstruction and geometry metrics for atlas ablation."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def weighted_mse(pred: np.ndarray, X: np.ndarray, w: np.ndarray) -> float:
    if len(X) == 0 or w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * ((pred - X) ** 2).sum(axis=1)) / np.sum(w))


def weighted_cosine(pred: np.ndarray, X: np.ndarray, w: np.ndarray) -> float:
    if len(X) == 0 or w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * (pred * X).sum(axis=1)) / np.sum(w))


def variance_normalized_mse(pred: np.ndarray, X: np.ndarray, w: np.ndarray) -> float:
    """MSE / mean weighted per-coordinate variance of X (scalar)."""
    mse = weighted_mse(pred, X, w)
    if not np.isfinite(mse) or w.sum() <= 0:
        return float("nan")
    ww = w / w.sum()
    mu = (ww[:, None] * X).sum(axis=0)
    var = float(np.sum(ww[:, None] * (X - mu) ** 2) / X.shape[1])
    return float(mse / max(var, 1e-12))


def median_knn_radius(X: np.ndarray, idx: np.ndarray, *, k: int = 16) -> float:
    if len(idx) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X[idx])
    return float(np.median(d[:, -1]))


def rmse_over_knn_radius(pred: np.ndarray, X: np.ndarray, w: np.ndarray, *, knn_radius: float) -> float:
    mse = weighted_mse(pred, X, w)
    if not np.isfinite(mse) or not np.isfinite(knn_radius) or knn_radius <= 0:
        return float("nan")
    return float(np.sqrt(mse) / knn_radius)


def fit_global_pca(X_train: np.ndarray, d: int) -> dict:
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    W = vt[:d].T
    return {"mu": mu.astype(np.float32), "basis": W.astype(np.float32)}


def global_pca_reconstruct(gpca: dict, X: np.ndarray) -> np.ndarray:
    U = (X - gpca["mu"]) @ gpca["basis"]
    Y = gpca["mu"] + U @ gpca["basis"].T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return (Y / np.maximum(n, 1e-8)).astype(np.float32)


def jacobian_stats_numpy(J: np.ndarray) -> dict:
    svals = np.linalg.svd(J, compute_uv=False)
    d = J.shape[1]
    eps = 1e-6 * float(svals.max()) if svals.size else 0.0
    rank = int(np.sum(svals > eps))
    cond = float(svals.max() / max(svals.min(), 1e-12)) if svals.size else float("inf")
    return {
        "rank": rank,
        "full_rank": bool(rank >= d),
        "condition": cond,
        "singular_values": svals.astype(np.float64).tolist(),
        "d": d,
    }
