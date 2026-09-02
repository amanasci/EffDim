"""Full-rank B^S ridge ≡ unrestricted quadratic with geometry-derived anisotropic penalty."""

from __future__ import annotations

import numpy as np

from geometry.physics_quadratic_label_chart_alignment.features import phi2_frob


def min_norm_c_penalty(gamma: np.ndarray, B: np.ndarray, *, rcond: float = 1e-10) -> float:
    """|c_min|_2^2 = γ^T (B^T B)^+ γ  for γ in row(B), c_min = B (B^T B)^+ γ."""
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    BtB = np.asarray(B, dtype=np.float64).T @ np.asarray(B, dtype=np.float64)
    inv = np.linalg.pinv(BtB, rcond=rcond)
    return float(g @ inv @ g)


def fit_ridge_c(Phi: np.ndarray, y: np.ndarray, B: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge on c: min ||y - Phi B^T c||^2 + α |c|^2, unpenalized intercept."""
    Phi = np.asarray(Phi, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = Phi @ B.T  # (n, D)
    x_mean = X.mean(0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    D = Xc.shape[1]
    XtX = Xc.T @ Xc
    XtX.flat[:: D + 1] += float(alpha)
    c = np.linalg.solve(XtX, Xc.T @ yc)
    b = y_mean - float(x_mean @ c)
    return c, b


def fit_ridge_gamma_aniso(Phi: np.ndarray, y: np.ndarray, B: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge on γ with penalty α γ^T (B^T B)^+ γ, γ unrestricted in R^q (full column rank)."""
    Phi = np.asarray(Phi, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    q = Phi.shape[1]
    x_mean = Phi.mean(0)
    y_mean = float(y.mean())
    Xc = Phi - x_mean
    yc = y - y_mean
    BtB = B.T @ B
    pen = np.linalg.pinv(BtB, rcond=1e-10)
    XtX = Xc.T @ Xc + float(alpha) * pen
    g = np.linalg.solve(XtX, Xc.T @ yc)
    b = y_mean - float(x_mean @ g)
    return g, b


def equivalence_demo(*, seed: int = 0, n: int = 80, d: int = 6, D: int = 40, alpha: float = 3.0) -> dict:
    """If rank(B)=q, ridge-on-c matches anisotropic ridge-on-γ."""
    rng = np.random.default_rng(seed)
    q = d * (d + 1) // 2
    U = rng.normal(size=(n, d))
    Phi = phi2_frob(U)
    B = rng.normal(size=(D, q))
    # force full column rank
    assert np.linalg.matrix_rank(B) == q
    y = rng.normal(size=n)
    c, b_c = fit_ridge_c(Phi, y, B, alpha)
    g_from_c = B.T @ c
    g_aniso, b_g = fit_ridge_gamma_aniso(Phi, y, B, alpha)
    pred_c = Phi @ g_from_c + b_c
    pred_g = Phi @ g_aniso + b_g
    return {
        "ok": bool(np.allclose(pred_c, pred_g, atol=1e-6, rtol=1e-6) and np.allclose(g_from_c, g_aniso, atol=1e-5)),
        "pred_max_abs_diff": float(np.max(np.abs(pred_c - pred_g))),
        "gamma_max_abs_diff": float(np.max(np.abs(g_from_c - g_aniso))),
        "rank_B": int(np.linalg.matrix_rank(B)),
        "q": int(q),
        "penalty_cmin": min_norm_c_penalty(g_from_c, B),
        "norm_c2": float(c @ c),
        "full_rank": True,
    }
