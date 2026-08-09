#!/usr/bin/env python3
"""Density-stratified statistics for the curvature experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr, t as student_t

__all__ = [
    "density_quartiles",
    "density_quartile_stats",
    "partial_spearman",
    "epsilon_ball_feasibility",
]

QUARTILE_NAMES = ["Q1", "Q2", "Q3", "Q4"]


def density_quartiles(d_k: np.ndarray) -> np.ndarray:
    """Label points 0..3 by d_k quartile. Q1 = densest (smallest kNN radius)."""
    d_k = np.asarray(d_k, dtype=np.float64)
    edges = np.nanpercentile(d_k, [25, 50, 75])
    return np.digitize(d_k, edges)


def _median_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) < 5:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[draws], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def density_quartile_stats(
    metric: np.ndarray,
    d_k: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Compare a metric across density quartiles of d_k.

    Returns per-quartile medians with bootstrap CIs, the Mann-Whitney U test for
    Q4 (sparsest) vs Q1 (densest) with a rank-biserial effect size, and the
    global Spearman rho against d_k with a bootstrap CI.

    Rank-biserial r = 2U/(n1*n2) - 1 is on [-1, 1]; positive means the sparsest
    quartile has larger values than the densest.
    """
    rng = np.random.default_rng(seed)
    metric = np.asarray(metric, dtype=np.float64)
    d_k = np.asarray(d_k, dtype=np.float64)
    ok = np.isfinite(metric) & np.isfinite(d_k)
    metric, d_k = metric[ok], d_k[ok]

    out: dict = {"n": int(ok.sum()), "quartiles": {}}
    if out["n"] < 40:
        return out

    q = density_quartiles(d_k)
    for j, name in enumerate(QUARTILE_NAMES):
        vals = metric[q == j]
        lo, hi = _median_ci(vals, n_boot, rng)
        out["quartiles"][name] = {
            "median": float(np.median(vals)) if len(vals) else float("nan"),
            "ci": [lo, hi],
            "n": int(len(vals)),
            "median_d_k": float(np.median(d_k[q == j])) if len(vals) else float("nan"),
        }

    a, b = metric[q == 0], metric[q == 3]
    if len(a) >= 10 and len(b) >= 10:
        u, p = mannwhitneyu(b, a, alternative="two-sided")
        out["mwu"] = {
            "U": float(u),
            "p": float(p),
            "rank_biserial": float(2.0 * u / (len(a) * len(b)) - 1.0),
        }

    rho, p = spearmanr(d_k, metric)
    boot = np.empty(min(n_boot, 500))
    for i in range(len(boot)):
        s = rng.integers(0, len(metric), len(metric))
        boot[i] = spearmanr(d_k[s], metric[s]).statistic
    lo, hi = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5])
    out["spearman_dk"] = {"rho": float(rho), "p": float(p), "ci": [float(lo), float(hi)]}
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Spearman correlation of x and y controlling for z.

    Ranks all three, regresses rank(x) and rank(y) on [1, rank(z)], and
    correlates the residuals. The p-value uses a t statistic on n-3 df.
    """
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(ok.sum())
    if n < 10:
        return {"rho": float("nan"), "p": float("nan"), "n": n}

    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))
    A = np.vstack([np.ones(n), rz]).T
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]

    denom = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    if denom < 1e-30:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    rho = float((ex * ey).sum() / denom)

    rho_c = min(max(rho, -0.999999), 0.999999)
    tstat = rho_c * np.sqrt((n - 3) / (1.0 - rho_c ** 2))
    p = float(2.0 * student_t.sf(abs(tstat), df=n - 3))
    return {"rho": rho, "p": p, "n": n}


def epsilon_ball_feasibility(d_k: np.ndarray, k_t: float, k_density: int) -> dict:
    """Would fixed-radius neighbourhoods have worked instead of fixed-k?

    Neighbour count inside a radius eps scales as K*(eps/d_K)^d. Calibrating eps
    so a median-density point in Q1 gets K neighbours leaves a Q4 point with
    K*(d_Q1/d_Q4)^d, which collapses fast once d is more than a handful. This is
    also why density-matched subsampling is infeasible: it would require keeping
    a fraction (d_Q1/d_Q4)^d of the sparse points.
    """
    d_k = np.asarray(d_k, dtype=np.float64)
    d_k = d_k[np.isfinite(d_k)]
    q = density_quartiles(d_k)
    m1, m4 = float(np.median(d_k[q == 0])), float(np.median(d_k[q == 3]))
    ratio = m4 / m1 if m1 > 0 else float("inf")
    predicted = float(k_density * ratio ** (-k_t))
    return {
        "k_t": float(k_t),
        "k_density": int(k_density),
        "median_d_k_Q1": m1,
        "median_d_k_Q4": m4,
        "dk_ratio_Q4_Q1": ratio,
        "predicted_Q4_neighbours_at_Q1_eps": predicted,
        "verdict": (
            "epsilon-ball infeasible" if predicted < 10 else "epsilon-ball possibly viable"
        ),
    }
