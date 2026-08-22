"""Matched synthetics for quadratic predictive-dimension calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .algebra import n_quad_features, phi2

SYNTH_KINDS = [
    "flat_d8",
    "flat_d12",
    "flat_d16",
    "flat_d20",
    "curved_d8",
    "curved_d12",
    "curved_d16",
    "curved_d12_q1",
    "curved_d12_q4",
    "curved_d12_q8",
    "cubic_d12",
    "isotropic",
    "thick_tangent_d12",
    "thick_normal_d12",
    "aniso_tail_d12",
]


def _qr(rng: np.random.Generator, D: int, k: int) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(D, max(k, 1))), mode="reduced")
    return Q[:, :k] if k else np.zeros((D, 0))


def make_predictive_synthetic(
    kind: str,
    *,
    n: int = 800,
    D: int = 36,
    seed: int = 0,
    radius: float = 0.12,
    noise: float = 0.008,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    true_d, q2, extra = 12, 0, "none"
    if kind == "flat_d8":
        true_d, q2 = 8, 0
    elif kind == "flat_d12":
        true_d, q2 = 12, 0
    elif kind == "flat_d16":
        true_d, q2 = 16, 0
    elif kind == "flat_d20":
        true_d, q2 = 20, 0
    elif kind == "curved_d8":
        true_d, q2 = 8, 1
    elif kind == "curved_d12":
        true_d, q2 = 12, 1
    elif kind == "curved_d16":
        true_d, q2 = 16, 1
    elif kind == "curved_d12_q1":
        true_d, q2 = 12, 1
    elif kind == "curved_d12_q4":
        true_d, q2 = 12, 4
    elif kind == "curved_d12_q8":
        true_d, q2 = 12, 8
    elif kind == "cubic_d12":
        true_d, q2, extra = 12, 0, "cubic"
    elif kind == "isotropic":
        true_d, q2, extra = D, 0, "isotropic"
    elif kind == "thick_tangent_d12":
        true_d, q2, extra = 12, 0, "thick_tan"
    elif kind == "thick_normal_d12":
        true_d, q2, extra = 12, 0, "thick_nor"
    elif kind == "aniso_tail_d12":
        true_d, q2, extra = 12, 1, "aniso_tail"
    else:
        raise ValueError(kind)

    true_d = min(true_d, D)
    T = _qr(rng, D, true_d)
    Nrm = _qr(rng, D, max(D - true_d, 0))
    if Nrm.size:
        Nrm = Nrm - T @ (T.T @ Nrm)
        Nrm, _ = np.linalg.qr(Nrm, mode="reduced")
    U = rng.normal(size=(n, true_d)) * radius
    # mildly anisotropic tangent spectrum
    decay = (1.0 / np.arange(1, true_d + 1) ** 0.7)
    U = U * decay[None, :]
    Z = U @ T.T
    if q2 > 0 and Nrm.shape[1] > 0:
        q_use = min(q2, Nrm.shape[1])
        Phi = phi2(U)
        for j in range(q_use):
            w = rng.normal(size=Phi.shape[1])
            quad = (Phi @ w) * 0.25
            Z = Z + quad[:, None] * Nrm[:, j][None, :]
    if extra == "cubic" and Nrm.shape[1]:
        Z = Z + 0.4 * ((U[:, 0] ** 3)[:, None] * Nrm[:, 0][None, :])
    if extra == "thick_tan":
        Z = Z + rng.normal(size=Z.shape) * (0.35 * radius)
        Z = (T @ (T.T @ Z.T)).T + rng.normal(size=Z.shape) * (0.02 * radius)
    if extra == "thick_nor" and Nrm.shape[1]:
        k = min(8, Nrm.shape[1])
        Z = Z + rng.normal(size=(n, k)) @ Nrm[:, :k].T * (0.4 * radius)
    if extra == "aniso_tail" and Nrm.shape[1]:
        k = min(8, Nrm.shape[1])
        tail = rng.normal(size=(n, k)) * radius
        tail *= (0.15 / np.arange(1, k + 1) ** 1.2)[None, :]
        Z = Z + tail @ Nrm[:, :k].T
    if extra == "isotropic":
        Z = rng.normal(size=(n, D)) * radius
        true_d = D
        q2 = 0
        T = np.eye(D)
        Nrm = np.zeros((D, 0))
    Z = Z + rng.normal(size=Z.shape) * noise
    # random ambient rotation
    Q = _qr(rng, D, D)
    Z = Z @ Q.T
    T = Q @ T
    Nrm = Q @ Nrm if Nrm.size else Nrm
    return {
        "kind": kind,
        "Z": Z.astype(np.float64),
        "true_d": int(true_d),
        "true_q2": int(q2),
        "T": T.astype(np.float64),
        "N": Nrm.astype(np.float64),
        "radii": np.linalg.norm(Z, axis=1),
        "D": D,
        "n": n,
        "extra": extra,
    }


def split_seeds(n_cal: int = 5, n_eval: int = 5, *, base: int = 7000) -> dict[str, list[int]]:
    return {
        "calibration_seeds": list(range(base, base + n_cal)),
        "evaluation_seeds": list(range(base + 200, base + 200 + n_eval)),
    }
