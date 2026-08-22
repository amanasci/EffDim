"""Synthetic manifolds for threshold calibration and mechanism matching.

Calibration seeds freeze decision thresholds. Evaluation seeds are untouched
until after thresholds are locked. Real probe associations are never used.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .sphere_coords import EPS, sphere_exp_map, _unit

SYNTH_KINDS = [
    "linear_d12",
    "curved_d12",
    "d12_isotropic_noise4",
    "d12_stable_thickness4",
    "true_d16_weak_tangent",
    "d16_near_degenerate_block",
    "curved_d12_r4_normal",
    "stratified_mixture",
    "saddle_zero_mean",
    "unit_sphere_baseline",
]


def _rand_frame(rng: np.random.Generator, D: int, k: int) -> np.ndarray:
    A = rng.normal(size=(D, k))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q[:, :k]


def _embed_sphere(U: np.ndarray, x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    Y = x0[None, :] + U @ J.T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return (Y / np.maximum(n, EPS)).astype(np.float64)


def make_synthetic(
    kind: str,
    *,
    n: int = 800,
    D: int = 64,
    seed: int = 0,
    k_obs: int | None = None,
) -> dict[str, Any]:
    """Return unit-normalized points and ground-truth labels."""
    rng = np.random.default_rng(seed)
    d12, d16 = 12, 16
    x0 = rng.normal(size=D)
    x0 = _unit(x0)
    true_d = 12
    extra_kind = "none"
    notes = {}

    if kind == "linear_d12":
        J = _rand_frame(rng, D, d12)
        J = J - np.outer(x0, x0 @ J)
        J, _ = np.linalg.qr(J, mode="reduced")
        J = J[:, :d12]
        U = rng.normal(size=(n, d12)) * 0.12
        X = _embed_sphere(U, x0, J)
        extra_kind = "none"
    elif kind == "curved_d12":
        J = _rand_frame(rng, D, d12)
        J = J - np.outer(x0, x0 @ J)
        Q, _ = np.linalg.qr(np.column_stack([x0, J]))
        J = Q[:, 1 : 1 + d12]
        nvec = Q[:, -1]
        U = rng.normal(size=(n, d12)) * 0.12
        Phi_diag = np.sum(U**2, axis=1, keepdims=True)
        Y = x0[None, :] + U @ J.T + 0.35 * Phi_diag * nvec[None, :]
        X = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), EPS)
        extra_kind = "curvature_normal"
        notes["K_H_expected"] = "nonzero"
    elif kind == "d12_isotropic_noise4":
        J = _rand_frame(rng, D, d12)
        J = J - np.outer(x0, x0 @ J)
        Q, _ = np.linalg.qr(np.column_stack([x0, J]))
        J = Q[:, 1 : 1 + d12]
        U = rng.normal(size=(n, d12)) * 0.12
        noise = 0.04 * rng.normal(size=(n, D))
        noise = noise - (noise @ Q[:, : 1 + d12]) @ Q[:, : 1 + d12].T
        Y = x0[None, :] + U @ J.T + noise
        X = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), EPS)
        extra_kind = "isotropic_noise"
        true_d = 12
    elif kind == "d12_stable_thickness4":
        J = _rand_frame(rng, D, d16)
        J = J - np.outer(x0, x0 @ J)
        Q, _ = np.linalg.qr(np.column_stack([x0, J]))
        Jt = Q[:, 1 : 1 + d12]
        Jn = Q[:, 1 + d12 : 1 + d16]
        U = rng.normal(size=(n, d12)) * 0.12
        # scale-independent thickness: additive in four normal dirs, radius-independent
        T = 0.08 * rng.normal(size=(n, 4))
        Y = x0[None, :] + U @ Jt.T + T @ Jn.T
        X = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), EPS)
        extra_kind = "thickness"
        true_d = 12
    elif kind == "true_d16_weak_tangent":
        J = _rand_frame(rng, D, d16)
        J = J - np.outer(x0, x0 @ J)
        Q, _ = np.linalg.qr(np.column_stack([x0, J]))
        J = Q[:, 1 : 1 + d16]
        U = rng.normal(size=(n, d16)) * 0.12
        U[:, 12:] *= 0.28
        X = _embed_sphere(U, x0, J)
        extra_kind = "weak_tangent"
        true_d = 16
    elif kind == "d16_near_degenerate_block":
        J = _rand_frame(rng, D, d16)
        J = J - np.outer(x0, x0 @ J)
        Q, _ = np.linalg.qr(np.column_stack([x0, J]))
        J = Q[:, 1 : 1 + d16]
        U = rng.normal(size=(n, d16)) * 0.12
        # nearly equal energy on 13-16
        U[:, 12:] = rng.normal(size=(n, 4)) * 0.045
        X = _embed_sphere(U, x0, J)
        extra_kind = "weak_tangent_degenerate"
        true_d = 16
        notes["degenerate_block"] = (12, 15)
    elif kind == "curved_d12_r4_normal":
        J = _rand_frame(rng, D, d12)
        J = J - np.outer(x0, x0 @ J)
        Q, _ = np.linalg.qr(np.column_stack([x0, J]))
        J = Q[:, 1 : 1 + d12]
        nvec = Q[:, -1]
        U = rng.normal(size=(n, d12)) * 0.15
        # quadratic normal: displacement O(r^2) → covariance O(r^4)
        r2 = np.sum(U**2, axis=1, keepdims=True)
        Y = x0[None, :] + U @ J.T + 0.8 * r2 * nvec[None, :]
        X = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), EPS)
        extra_kind = "curvature_normal_r4"
        true_d = 12
    elif kind == "stratified_mixture":
        J16 = _rand_frame(rng, D, d16)
        J16 = J16 - np.outer(x0, x0 @ J16)
        Q, _ = np.linalg.qr(np.column_stack([x0, J16]))
        J = Q[:, 1 : 1 + d16]
        U = rng.normal(size=(n, d16)) * 0.05
        # far points use extra 4 directions
        far = rng.random(n) > 0.55
        U[far, 12:] = rng.normal(size=(int(far.sum()), 4)) * 0.18
        X = _embed_sphere(U, x0, J)
        extra_kind = "stratification"
        true_d = 12
        notes["scale_dependent"] = True
    elif kind == "saddle_zero_mean":
        J = _rand_frame(rng, D, d12)
        J = J - np.outer(x0, x0 @ J)
        Q, _ = np.linalg.qr(np.column_stack([x0, J, rng.normal(size=(D, 2))]))
        J = Q[:, 1 : 1 + d12]
        n1, n2 = Q[:, -2], Q[:, -1]
        U = rng.normal(size=(n, d12)) * 0.12
        # B(v,v) = (v1^2 - v2^2) n1  → H = 0, K_dir > 0
        quad = (U[:, 0:1] ** 2 - U[:, 1:2] ** 2) * n1[None, :] + (
            2.0 * U[:, 0:1] * U[:, 1:2]
        ) * n2[None, :]
        Y = x0[None, :] + U @ J.T + 0.45 * quad
        X = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), EPS)
        extra_kind = "saddle"
        notes["K_H_expected"] = "zero"
        notes["K_dir_expected"] = "positive"
        true_d = 12
    elif kind == "unit_sphere_baseline":
        # points on a totally geodesic S^{12} ⊂ S^{D-1}: sphere-normal B^S = 0
        frame = _rand_frame(rng, D, d12 + 1)
        coeffs = rng.normal(size=(n, d12 + 1))
        X = coeffs @ frame.T
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
        extra_kind = "geodesic_sphere"
        true_d = 12
        notes["K_dir_expected"] = "zero"
        x0 = X[0]
        J = None
    else:
        raise ValueError(kind)

    # neighbourhood around row 0
    sims = X @ X[0]
    order = np.argsort(-sims)
    k_obs = int(k_obs or min(256, n - 1))
    neigh = order[1 : k_obs + 1]
    return {
        "kind": kind,
        "X": X.astype(np.float64),
        "x0": X[0].astype(np.float64),
        "neigh": neigh.astype(np.int64),
        "true_d": int(true_d),
        "extra_kind": extra_kind,
        "notes": notes,
        "seed": int(seed),
    }


def split_seeds(n_cal: int, n_eval: int, *, base: int = 1000) -> dict[str, list[int]]:
    cal = list(range(base, base + n_cal))
    ev = list(range(base + 100, base + 100 + n_eval))
    return {"calibration_seeds": cal, "evaluation_seeds": ev}


def feature_vector_from_summary(row: dict[str, Any]) -> np.ndarray:
    keys = [
        "median_dT",
        "p_ge_12",
        "p_ge_16",
        "agree_13_16",
        "gain_13_16",
        "alpha_13_16",
        "var_share_13_16",
        "Dlin_12",
        "Dlin_16",
    ]
    return np.array([float(row.get(k, np.nan)) for k in keys], dtype=np.float64)


def closest_synthetic(
    real: dict[str, Any],
    synth_rows: list[dict[str, Any]],
) -> tuple[str, float]:
    r = feature_vector_from_summary(real)
    best, best_d = "unresolved", float("inf")
    for s in synth_rows:
        v = feature_vector_from_summary(s)
        m = np.isfinite(r) & np.isfinite(v)
        if m.sum() < 4:
            continue
        rr, vv = r[m], v[m]
        sd = np.std(vv) if np.std(vv) > EPS else 1.0
        d = float(np.linalg.norm((rr - vv) / max(sd, EPS)))
        if d < best_d:
            best, best_d = str(s.get("kind", "unknown")), d
    return best, best_d
