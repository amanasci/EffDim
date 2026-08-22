"""Matched synthetics for order-stratified (d_1, q_2) calibration.

Calibration seeds freeze thresholds. Evaluation seeds stay locked until after.
Real probe associations are never used to choose ranks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_activation_atlas.quadratic import n_quad_features, quadratic_features
from geometry.physics_stable_tangent_dimension.sphere_coords import EPS

SYNTH_KINDS = [
    "flat_d12",
    "curved_d12_q1",
    "curved_d12_q4",
    "curved_d12_q8",
    "weak_flat_d16",
    "weak_curved_d16",
    "mixed_d12_q4_plus_2tangent",
    "d12_thickness4",
    "saddle_d12_q4",
    "unit_sphere_baseline",
]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), EPS)


def _rand_frame(rng: np.random.Generator, D: int, k: int) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(D, k)), mode="reduced")
    return Q[:, :k]


def _sphere_normal_frame(rng: np.random.Generator, x0: np.ndarray, J: np.ndarray, q: int) -> np.ndarray:
    D = x0.shape[0]
    Q, _ = np.linalg.qr(np.column_stack([x0, J, rng.normal(size=(D, q + 2))]), mode="reduced")
    N = Q[:, (1 + J.shape[1]) : (1 + J.shape[1] + q)]
    if N.shape[1] < q:
        extra = rng.normal(size=(D, q))
        extra = extra - np.outer(x0, extra.T @ x0)
        extra = extra - J @ (J.T @ extra)
        Qe, _ = np.linalg.qr(extra, mode="reduced")
        N = Qe[:, :q]
    return N


def _embed(Y: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return (Y / np.maximum(n, EPS)).astype(np.float64)


def _quad_from_normals(U: np.ndarray, normals: np.ndarray, forms: np.ndarray) -> np.ndarray:
    """Y_quad[i] = sum_j normals[:,j] * (u_i^T A_j u_i). forms (q, d, d) SPD or signed."""
    n, d = U.shape
    q = normals.shape[1]
    out = np.zeros((n, normals.shape[0]))
    for j in range(q):
        Aj = 0.5 * (forms[j] + forms[j].T)
        quad = np.einsum("ni,ij,nj->n", U, Aj, U)
        out += np.outer(quad, normals[:, j])
    return out


def make_order_synthetic(
    kind: str,
    *,
    n: int = 800,
    D: int = 64,
    seed: int = 0,
    k_obs: int | None = None,
    d_core: int = 12,
    radius: float = 0.12,
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    d12 = int(d_core)
    x0 = _unit(rng.normal(size=D))
    true_d1 = d12
    true_q2 = 0
    extra_kind = "none"
    notes: dict[str, Any] = {}

    J12 = _rand_frame(rng, D, d12)
    J12 = J12 - np.outer(x0, x0 @ J12)
    J12, _ = np.linalg.qr(J12, mode="reduced")
    J12 = J12[:, :d12]
    U = rng.normal(size=(n, d12)) * radius
    Y = x0[None, :] + U @ J12.T
    J_extra = None

    if kind == "flat_d12":
        extra_kind = "flat"
        true_q2 = 0
    elif kind == "curved_d12_q1":
        N = _sphere_normal_frame(rng, x0, J12, 1)
        A = np.eye(d12)
        Y = Y + 0.45 * _quad_from_normals(U, N, A[None, :, :])
        true_q2 = 1
        extra_kind = "quadratic_normal"
    elif kind == "curved_d12_q4":
        N = _sphere_normal_frame(rng, x0, J12, 4)
        forms = np.stack([np.diag(rng.normal(size=d12)) for _ in range(4)])
        Y = Y + 0.40 * _quad_from_normals(U, N, forms)
        true_q2 = 4
        extra_kind = "quadratic_normal"
    elif kind == "curved_d12_q8":
        N = _sphere_normal_frame(rng, x0, J12, 8)
        forms = np.stack([np.diag(rng.normal(size=d12)) for _ in range(8)])
        Y = Y + 0.35 * _quad_from_normals(U, N, forms)
        true_q2 = 8
        extra_kind = "quadratic_normal"
    elif kind == "weak_flat_d16":
        Jex = _rand_frame(rng, D, 4)
        Jex = Jex - np.outer(x0, x0 @ Jex) - J12 @ (J12.T @ Jex)
        Jex, _ = np.linalg.qr(Jex, mode="reduced")
        Jex = Jex[:, :4]
        Uex = rng.normal(size=(n, 4)) * (0.35 * radius)
        Y = Y + Uex @ Jex.T
        true_d1 = 16
        true_q2 = 0
        extra_kind = "weak_tangent"
        J_extra = Jex
    elif kind == "weak_curved_d16":
        Jex = _rand_frame(rng, D, 4)
        Jex = Jex - np.outer(x0, x0 @ Jex) - J12 @ (J12.T @ Jex)
        Jex, _ = np.linalg.qr(Jex, mode="reduced")
        Jex = Jex[:, :4]
        Uex = rng.normal(size=(n, 4)) * (0.35 * radius)
        N = _sphere_normal_frame(rng, x0, np.column_stack([J12, Jex]), 2)
        forms = np.stack([np.eye(d12), np.diag(np.linspace(1.0, 0.2, d12))])
        Y = Y + Uex @ Jex.T + 0.25 * _quad_from_normals(U, N, forms)
        true_d1 = 16
        true_q2 = 2
        extra_kind = "weak_tangent_plus_curve"
        J_extra = Jex
    elif kind == "mixed_d12_q4_plus_2tangent":
        Jex = _rand_frame(rng, D, 2)
        Jex = Jex - np.outer(x0, x0 @ Jex) - J12 @ (J12.T @ Jex)
        Jex, _ = np.linalg.qr(Jex, mode="reduced")
        Jex = Jex[:, :2]
        Uex = rng.normal(size=(n, 2)) * (0.30 * radius)
        N = _sphere_normal_frame(rng, x0, np.column_stack([J12, Jex]), 4)
        forms = np.stack([np.diag(rng.normal(size=d12)) for _ in range(4)])
        Y = Y + Uex @ Jex.T + 0.40 * _quad_from_normals(U, N, forms)
        true_d1 = 14
        true_q2 = 4
        extra_kind = "mixed"
        J_extra = Jex
    elif kind == "d12_thickness4":
        N = _sphere_normal_frame(rng, x0, J12, 4)
        thick = rng.normal(size=(n, 4)) * (0.45 * radius)
        Y = Y + thick @ N.T
        true_q2 = 0
        extra_kind = "thickness"
    elif kind == "saddle_d12_q4":
        N = _sphere_normal_frame(rng, x0, J12, 2)
        A1 = np.zeros((d12, d12))
        A1[0, 0], A1[1, 1] = 1.0, -1.0
        A2 = np.zeros((d12, d12))
        A2[0, 1] = A2[1, 0] = 1.0
        forms = np.stack([A1, A2])
        Y = Y + 0.50 * _quad_from_normals(U, N, forms)
        true_q2 = 2
        extra_kind = "saddle"
        notes["K_H_expected"] = "weak"
    elif kind == "unit_sphere_baseline":
        frame = _rand_frame(rng, D, d12 + 1)
        coeffs = rng.normal(size=(n, d12 + 1))
        Y = coeffs @ frame.T
        x0 = _unit(Y[0])
        J12 = None
        true_q2 = 0
        extra_kind = "geodesic_sphere"
    else:
        raise ValueError(kind)

    X = _embed(Y)
    x0 = _unit(X[0] if kind == "unit_sphere_baseline" else x0)
    sims = X @ x0
    order = np.argsort(-sims)
    k_obs = int(k_obs or min(256, n - 1))
    # skip the exact anchor if present
    neigh = order[1 : k_obs + 1] if float(sims[order[0]]) > 0.999 else order[:k_obs]
    return {
        "kind": kind,
        "X": X,
        "x0": x0.astype(np.float64),
        "neigh": neigh.astype(np.int64),
        "true_d1": int(true_d1),
        "true_q2": int(true_q2),
        "extra_kind": extra_kind,
        "notes": notes,
        "J12": None if J12 is None else J12.astype(np.float64),
        "J_extra": None if J_extra is None else J_extra.astype(np.float64),
        "n_quad": n_quad_features(d12),
    }


def split_seeds(n_cal: int = 8, n_eval: int = 8, *, base: int = 3000) -> dict[str, list[int]]:
    return {
        "calibration_seeds": list(range(base, base + n_cal)),
        "evaluation_seeds": list(range(base + 100, base + 100 + n_eval)),
    }


def closest_synthetic(real: dict[str, Any], synth_rows: list[dict[str, Any]]) -> tuple[str, float]:
    keys = ["median_q2", "overlap_E4", "r2_quad_E4", "pi_quad", "pi_lin", "delta_M16_minus_M12q"]
    r = np.array([float(real.get(k, np.nan)) for k in keys], dtype=np.float64)
    best, best_d = "unresolved", float("inf")
    for s in synth_rows:
        v = np.array([float(s.get(k, np.nan)) for k in keys], dtype=np.float64)
        m = np.isfinite(r) & np.isfinite(v)
        if int(m.sum()) < 3:
            continue
        d = float(np.linalg.norm(r[m] - v[m]))
        if d < best_d:
            best, best_d = str(s.get("kind", "unknown")), d
    return best, best_d
