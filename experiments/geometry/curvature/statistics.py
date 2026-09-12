"""Rank-space controls, Holm, Monte Carlo p-values. No real-data fitting."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

from .metric import holm


def p_mc(b: int, B: int) -> float:
    return float(b + 1) / float(B + 1)


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 8:
        return float("nan")
    return float(spearmanr(x[m], y[m]).statistic)


def rank_partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    if int(m.sum()) < 8:
        return float("nan")
    xr = rankdata(x[m]).astype(np.float64)
    yr = rankdata(y[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    bx, *_ = np.linalg.lstsq(A, xr, rcond=None)
    by, *_ = np.linalg.lstsq(A, yr, rcond=None)
    return spearman_safe(xr - A @ bx, yr - A @ by)


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def synchronized_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    """One permutation to share across targets or models."""
    return rng.permutation(n)


def freedman_lane_residual_perm(y: np.ndarray, Z: np.ndarray, perm: np.ndarray) -> np.ndarray:
    msk = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y2 = y.copy()
    yr = rankdata(y[msk]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[msk, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(msk.sum())), Zr])
    bhat, *_ = np.linalg.lstsq(A, yr, rcond=None)
    fit = A @ bhat
    resid = yr - fit
    idx = np.where(msk)[0]
    y2[idx] = fit + resid[np.argsort(perm[idx])]
    return y2
