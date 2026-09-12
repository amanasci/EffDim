"""Train-only local label Hessian, mismatch and reliability."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metric import cosine_g, energy_g
from .quadratic import Gamma_from_gamma, n_quad, phi2_frob


def tangent_coords(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    return (np.asarray(Xloc, dtype=np.float64) - np.asarray(x0, dtype=np.float64)) @ (J @ ginv)


def fit_label_hessian(U: np.ndarray, y: np.ndarray, *, ridge: float = 0.0, d: int | None = None) -> dict[str, Any]:
    U = np.asarray(U, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = int(U.shape[1] if d is None else d)
    m = np.isfinite(y) & np.all(np.isfinite(U), axis=1)
    U, y = U[m], y[m]
    p = 1 + d + n_quad(d)
    if len(y) < p + 2:
        return {"ok": False, "Hy": np.full((d, d), np.nan), "status": "unstable_or_underdetermined"}
    Phi = np.concatenate([np.ones((len(y), 1)), U, phi2_frob(U)], axis=1)
    A = Phi.T @ Phi
    if ridge > 0:
        A = A + ridge * np.eye(p)
    coef, *_ = np.linalg.lstsq(A, Phi.T @ y, rcond=None)
    Hy = Gamma_from_gamma(coef[1 + d :], d)
    finite = bool(np.isfinite(Hy).all())
    return {
        "ok": finite,
        "Hy": Hy,
        "a0": float(coef[0]),
        "a1": coef[1 : 1 + d],
        "status": "ok" if finite else "unstable_hessian",
        "ridge": float(ridge),
    }


def mismatch(Hy: np.ndarray, Bw: np.ndarray, ginv: np.ndarray) -> dict[str, float]:
    dlt = Hy - Bw
    return {
        "M_delta": float(np.sqrt(max(energy_g(dlt, ginv), 0.0))),
        "A_full": cosine_g(Hy, Bw, ginv),
        "Hy_norm": float(np.sqrt(max(energy_g(Hy, ginv), 0.0))),
        "Bw_norm": float(np.sqrt(max(energy_g(Bw, ginv), 0.0))),
    }


def null_hessian(d: int) -> np.ndarray:
    return np.zeros((d, d), dtype=np.float64)
