"""Scoring against T1/T2/T3. Never score a pointwise estimator only against a patch target."""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress, spearmanr

from .geometry import cosine, directional_tensor_cos, rel_err


def _finite(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def rank_cal(est: np.ndarray, truth: np.ndarray) -> dict:
    e, t = _finite(est, truth)
    if len(e) < 8:
        return {"rho": float("nan"), "n": int(len(e))}
    rho = spearmanr(e, t).correlation
    slope, intercept, r_val, _, _ = linregress(t, e)
    ratio = float(np.median(e / np.clip(np.abs(t), 1e-12, None) * np.sign(t))) if np.any(np.abs(t) > 1e-12) else float("nan")
    rel = float(np.median(np.abs(e - t) / np.clip(np.abs(t) + np.abs(e), 1e-12, None)))
    return {
        "rho": float(rho) if rho is not None and np.isfinite(rho) else float("nan"),
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r_val**2),
        "magnitude_ratio": ratio,
        "median_rel_err": rel,
        "n": int(len(e)),
    }


def vector_recovery(H_est: np.ndarray, H_true: np.ndarray) -> dict:
    nrms_e = np.linalg.norm(H_est, axis=1)
    nrms_t = np.linalg.norm(H_true, axis=1)
    out = rank_cal(nrms_e, nrms_t)
    cos = [cosine(H_est[i], H_true[i]) for i in range(len(H_est))]
    out["median_cosine"] = float(np.nanmedian(cos))
    out["mean_cosine"] = float(np.nanmean(cos))
    return out


def partial_spearman(x, y, z) -> float:
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 12:
        return float("nan")
    A = np.column_stack([np.ones(m.sum()), z[m]])
    def resid(v):
        coef, *_ = np.linalg.lstsq(A, v[m], rcond=None)
        return v[m] - A @ coef
    rx, ry = resid(x), resid(y)
    r = spearmanr(rx, ry).correlation
    return float(r) if r is not None else float("nan")
