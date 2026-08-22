"""Matched synthetics in an R-dimensional carrier for implicit normal inversion."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_stable_tangent_dimension.sphere_coords import EPS

SYNTH_KINDS = [
    "flat_d12_c8_q0",
    "curved_d12_c8_q1",
    "curved_d12_c8_q4",
    "curved_d12_c8_q8",
    "flat_d16_c4",
    "curved_d16_c4_q1",
    "curved_d16_c4_q4",
    "d12_thickness_normal",
    "d12_weak_tangent_nuisance",
    "cubic_normal_weak_quad",
    "stratified_mixture",
    "isotropic_carrier",
    "unit_sphere_baseline",
]


def _qr(rng: np.random.Generator, R: int, k: int) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(R, max(k, 1))), mode="reduced")
    return Q[:, :k] if k else np.zeros((R, 0))


def make_implicit_synthetic(
    kind: str,
    *,
    n: int = 800,
    R: int = 20,
    seed: int = 0,
    radius: float = 0.12,
    noise: float = 0.008,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d1, cN, q2 = 12, 8, 0
    extra = "none"
    if kind == "flat_d12_c8_q0":
        d1, cN, q2 = 12, 8, 0
    elif kind == "curved_d12_c8_q1":
        d1, cN, q2 = 12, 8, 1
    elif kind == "curved_d12_c8_q4":
        d1, cN, q2 = 12, 8, 4
    elif kind == "curved_d12_c8_q8":
        d1, cN, q2 = 12, 8, 8
    elif kind == "flat_d16_c4":
        d1, cN, q2 = 16, 4, 0
    elif kind == "curved_d16_c4_q1":
        d1, cN, q2 = 16, 4, 1
    elif kind == "curved_d16_c4_q4":
        d1, cN, q2 = 16, 4, 4
    elif kind == "d12_thickness_normal":
        d1, cN, q2 = 12, 8, 0
        extra = "thickness"
    elif kind == "d12_weak_tangent_nuisance":
        d1, cN, q2 = 12, 8, 0
        extra = "weak_tangent"
    elif kind == "cubic_normal_weak_quad":
        d1, cN, q2 = 12, 8, 1
        extra = "cubic"
    elif kind == "stratified_mixture":
        d1, cN, q2 = 12, 8, 1
        extra = "mixture"
    elif kind == "isotropic_carrier":
        d1, cN, q2 = R, 0, 0
        extra = "isotropic"
    elif kind == "unit_sphere_baseline":
        d1, cN, q2 = R - 0, 0, 0
        extra = "sphere_removed"
    else:
        raise ValueError(kind)

    d1 = min(d1, R)
    cN = min(cN, R - d1) if extra != "isotropic" else 0
    T = _qr(rng, R, d1)
    Nrm = _qr(rng, R, cN)
    # orthogonalize N against T
    if cN:
        Nrm = Nrm - T @ (T.T @ Nrm)
        Nrm, _ = np.linalg.qr(Nrm, mode="reduced")
        Nrm = Nrm[:, :cN]
    U = rng.normal(size=(n, d1)) * radius
    Y = U @ T.T
    if q2 > 0 and cN > 0:
        q_use = min(q2, cN)
        for j in range(q_use):
            w = rng.normal(size=d1)
            quad = (U * w[None, :]) ** 2
            if extra == "cubic":
                Y = Y + 0.15 * ((U[:, 0:1] ** 3) @ Nrm[:, j : j + 1].T)
            coef = 0.12 if extra == "cubic" else 0.35
            Y = Y + coef * (quad.sum(axis=1, keepdims=True) @ Nrm[:, j : j + 1].T)
    if extra == "thickness" and cN:
        Y = Y + rng.normal(size=(n, cN)) * (0.5 * radius) @ Nrm.T
    if extra == "weak_tangent" and d1 >= 2:
        # leak a little first-order energy into two 'normal' slots
        leak = min(2, cN)
        if leak:
            Y = Y + rng.normal(size=(n, leak)) * (0.25 * radius) @ Nrm[:, :leak].T
    if extra == "mixture":
        mask = rng.random(n) < 0.3
        Y[mask] = Y[mask] + rng.normal(size=(int(mask.sum()), R)) * (0.4 * radius)
    if extra == "isotropic":
        Y = rng.normal(size=(n, R)) * radius
        T = np.eye(R)
        Nrm = np.zeros((R, 0))
        d1, cN, q2 = R, 0, 0
    Y = Y + rng.normal(size=Y.shape) * noise
    # random carrier rotation (the estimator must be invariant)
    Q = _qr(rng, R, R)
    Y = Y @ Q.T
    T = Q @ T
    Nrm = Q @ Nrm if Nrm.size else Nrm
    rad = np.linalg.norm(Y, axis=1)
    return {
        "kind": kind,
        "Y": Y.astype(np.float64),
        "true_d1": int(d1),
        "true_cN": int(cN),
        "true_q2": int(q2),
        "T": T.astype(np.float64),
        "N": Nrm.astype(np.float64),
        "radii": rad,
        "R": R,
        "extra": extra,
        "n": n,
    }


def split_seeds(n_cal: int = 6, n_eval: int = 6, *, base: int = 5000) -> dict[str, list[int]]:
    return {
        "calibration_seeds": list(range(base, base + n_cal)),
        "evaluation_seeds": list(range(base + 200, base + 200 + n_eval)),
    }
