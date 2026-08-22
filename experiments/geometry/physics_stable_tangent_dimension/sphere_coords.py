"""Sphere-aware displacements, projectors, and parallel transport.

Primary coordinate: spherical logarithm at the anchor. Projected chords are
a parity / sensitivity analysis only. Do not silently L2-normalize frozen
representations; callers must record the established convention.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
CLIP = 1e-15
SMALL_ANGLE = 1e-4


def _unit(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / max(n, eps)


def clip_cosine(c: np.ndarray | float, eps: float = CLIP) -> np.ndarray | float:
    """Clip to the arccos domain. eps only bites values that overflow ±1."""
    if np.isscalar(c):
        v = float(c)
        if v > 1.0:
            return 1.0 - eps if eps else 1.0
        if v < -1.0:
            return -1.0 + eps if eps else -1.0
        return v
    out = np.asarray(c, dtype=np.float64).copy()
    out = np.clip(out, -1.0, 1.0)
    return out


def sinc_theta_over_sin(theta: np.ndarray) -> np.ndarray:
    """Stable θ / sin θ, including the small-angle branch 1 + θ²/6."""
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty_like(theta)
    small = np.abs(theta) < SMALL_ANGLE
    out[small] = 1.0 + (theta[small] ** 2) / 6.0
    large = ~small
    s = np.sin(theta[large])
    out[large] = theta[large] / np.where(np.abs(s) < EPS, np.sign(s) * EPS + EPS, s)
    return out


def sphere_log_map(x: np.ndarray, y: np.ndarray, *, eps: float = CLIP) -> np.ndarray:
    """log_x(y) for unit vectors. y may be (D,) or (n, D)."""
    x = _unit(np.asarray(x, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    batched = y.ndim == 2
    if not batched:
        y = y[None, :]
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), EPS)
    c = clip_cosine(y @ x, eps)
    theta = np.arccos(c)
    scale = sinc_theta_over_sin(theta)
    z = scale[:, None] * (y - c[:, None] * x)
    # already in T_x up to roundoff; project for safety
    z = z - np.outer(z @ x, x)
    return z if batched else z[0]


def projected_chord(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(I - xx^T)(y - x). Sensitivity / parity coordinate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    batched = y.ndim == 2
    if not batched:
        y = y[None, :]
    d = y - x[None, :]
    z = d - np.outer(d @ x, x)
    return z if batched else z[0]


def sphere_exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    x = _unit(np.asarray(x, dtype=np.float64))
    v = np.asarray(v, dtype=np.float64)
    v = v - np.dot(v, x) * x
    th = float(np.linalg.norm(v))
    if th < SMALL_ANGLE:
        q = x + v
        return _unit(q)
    return np.cos(th) * x + np.sin(th) * (v / th)


def parallel_transport_yx(y: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Great-circle PT of v ∈ T_y S^{D-1} to T_x.

    PT_{y→x}(v) = v - (v^T x) / (1 + x^T y) * (x + y)
    away from antipodal degeneracy.
    """
    x = _unit(np.asarray(x, dtype=np.float64))
    y = _unit(np.asarray(y, dtype=np.float64))
    v = np.asarray(v, dtype=np.float64)
    v = v - np.dot(v, y) * y
    c = float(np.dot(x, y))
    if c <= -1.0 + 1e-6:
        # antipodal: project to T_x; direction is undefined
        return v - np.dot(v, x) * x
    out = v - (np.dot(v, x) / (1.0 + c)) * (x + y)
    return out - np.dot(out, x) * x


def parallel_transport_basis_yx(y: np.ndarray, x: np.ndarray, J: np.ndarray) -> np.ndarray:
    cols = [parallel_transport_yx(y, x, J[:, i]) for i in range(J.shape[1])]
    M = np.column_stack(cols)
    M = M - np.outer(x, x) @ M
    Q, _ = np.linalg.qr(M, mode="reduced")
    return Q[:, : J.shape[1]]


def tangent_projector(x: np.ndarray, J: np.ndarray) -> np.ndarray:
    """P_T = JJ^T with J already in T_x (sphere-tangent)."""
    return J @ J.T


def radial_projector(x: np.ndarray) -> np.ndarray:
    x = _unit(x)
    return np.outer(x, x)


def sphere_normal_apply(V: np.ndarray, x: np.ndarray, J: np.ndarray) -> np.ndarray:
    """P_{N,S} V = V - Proj_{span(x, J)} V."""
    x = _unit(x)
    Q, _ = np.linalg.qr(np.column_stack([x, J]), mode="reduced")
    if V.ndim == 1:
        return V - Q @ (Q.T @ V)
    return V - (V @ Q) @ Q.T


def angular_radii(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x = _unit(x)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), EPS)
    c = clip_cosine(Y @ x)
    return np.arccos(c)


def rms_tangent_radius(Z: np.ndarray) -> float:
    nrm = np.linalg.norm(Z, axis=1)
    return float(np.sqrt(np.mean(nrm**2))) if len(nrm) else float("nan")


def row_l2_status(X: np.ndarray, atol: float = 1e-4) -> dict:
    n = np.linalg.norm(X, axis=1)
    return {
        "n_rows": int(len(n)),
        "median_norm": float(np.median(n)),
        "min_norm": float(np.min(n)),
        "max_norm": float(np.max(n)),
        "unit_normalized": bool(np.all(np.abs(n - 1.0) < atol)),
    }
