"""Leakage-safe intrinsic label Hessian. QLCA Frobenius convention."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_quadratic_label_chart_alignment.features import Gamma_from_gamma, n_quad, phi2_frob

from .config import D_LAT, HESS_RIDGE_PAPER, HESS_RIDGE_STAB


def tangent_coords(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """u = g^{-1} J^T (x - x̃)."""
    return (Xloc - x0) @ (J @ ginv)


def _design(U: np.ndarray) -> np.ndarray:
    n, d = U.shape
    ones = np.ones((n, 1), dtype=np.float64)
    return np.concatenate([ones, U, phi2_frob(U)], axis=1)


def _unpack(coef: np.ndarray, d: int) -> tuple[float, np.ndarray, np.ndarray]:
    a0 = float(coef[0])
    a1 = np.asarray(coef[1 : 1 + d], dtype=np.float64)
    gamma = np.asarray(coef[1 + d :], dtype=np.float64)
    return a0, a1, Gamma_from_gamma(gamma, d)


def fit_label_hessian(
    U: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    d: int = D_LAT,
) -> dict[str, Any]:
    m = np.isfinite(y) & np.all(np.isfinite(U), axis=1)
    U, y = U[m], y[m]
    n = int(len(y))
    p = 1 + d + n_quad(d)
    if n < p + 2:
        return {"ok": False, "Hy": np.full((d, d), np.nan), "a0": float("nan"), "a1": np.full(d, np.nan), "n": n}
    Phi = _design(U)
    A = Phi.T @ Phi
    if ridge > 0:
        A = A + ridge * np.eye(p)
    try:
        coef, *_ = np.linalg.lstsq(A, Phi.T @ y, rcond=None)
        # lstsq(A, b) with A = Phi.T Phi is normal-equation solve
    except np.linalg.LinAlgError:
        return {"ok": False, "Hy": np.full((d, d), np.nan), "a0": float("nan"), "a1": np.full(d, np.nan), "n": n}
    a0, a1, Hy = _unpack(coef, d)
    yhat = Phi @ coef
    resid = y - yhat
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if sst < 1e-15 else 1.0 - float(np.sum(resid**2)) / sst
    cond = float(np.linalg.cond(A))
    rank = int(np.linalg.matrix_rank(Phi, tol=1e-8))
    return {
        "ok": bool(np.isfinite(Hy).all()),
        "Hy": Hy,
        "a0": a0,
        "a1": a1,
        "n": n,
        "rank": rank,
        "cond": cond,
        "coef_norm": float(np.linalg.norm(coef)),
        "train_r2": r2,
        "ridge": float(ridge),
        "n_quad": n_quad(d),
    }


def fit_linear_only(U: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    m = np.isfinite(y) & np.all(np.isfinite(U), axis=1)
    U, y = U[m], y[m]
    if len(y) < U.shape[1] + 3:
        return {"r2": float("nan")}
    Phi = np.concatenate([np.ones((len(y), 1)), U], axis=1)
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    yhat = Phi @ coef
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if sst < 1e-15 else 1.0 - float(np.sum((y - yhat) ** 2)) / sst
    return {"r2": r2, "a0": float(coef[0]), "a1": coef[1:]}


def predict_quad(U: np.ndarray, a0: float, a1: np.ndarray, Hy: np.ndarray) -> np.ndarray:
    return a0 + U @ a1 + 0.5 * np.einsum("ni,ij,nj->n", U, Hy, U)


def split_half_cosine(U: np.ndarray, y: np.ndarray, ginv: np.ndarray, *, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 40:
        return float("nan")
    perm = rng.permutation(n)
    h = n // 2
    fa = fit_label_hessian(U[perm[:h]], y[perm[:h]], ridge=HESS_RIDGE_PAPER)
    fb = fit_label_hessian(U[perm[h:]], y[perm[h:]], ridge=HESS_RIDGE_PAPER)
    if not (fa["ok"] and fb["ok"]):
        return float("nan")
    from geometry.physics_task_aligned_curvature.algebra import energy_g, cross_energy_g

    na = np.sqrt(max(energy_g(fa["Hy"], ginv), 1e-30))
    nb = np.sqrt(max(energy_g(fb["Hy"], ginv), 1e-30))
    return float(cross_energy_g(fa["Hy"], fb["Hy"], ginv) / (na * nb))
