"""Implicit normal-space inverse: profiled constraints, Sampson, shape operators.

c_N is carrier-normal codimension. q_2 is quadratic-active rank inside that
normal space. They are different: q_2 <= c_N. The tangent is ker(A^T).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_activation_atlas.quadratic import n_quad_features, quadratic_features
from geometry.physics_stable_tangent_dimension.nested_pca import degenerate_blocks
from geometry.physics_stable_tangent_dimension.sphere_coords import EPS

RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0]


def vech_weights(d: int) -> np.ndarray:
    w = []
    for a in range(d):
        for b in range(a, d):
            w.append(1.0 if a == b else np.sqrt(2.0))
    return np.asarray(w, dtype=np.float64)


def weighted_phi(Y: np.ndarray) -> np.ndarray:
    """Metric-aware vech(yy^T): off-diagonals scaled by sqrt(2)."""
    Phi = quadratic_features(Y)
    return Phi * vech_weights(Y.shape[1])[None, :]


def unpack_h(h: np.ndarray, d: int) -> np.ndarray:
    """h stores 1/2 y^T H y = phi_weighted · h; recover symmetric H."""
    H = np.zeros((d, d), dtype=np.float64)
    idx = 0
    for a in range(d):
        for b in range(a, d):
            if a == b:
                H[a, a] = 2.0 * float(h[idx])
            else:
                H[a, b] = H[b, a] = float(h[idx]) * np.sqrt(2.0)
            idx += 1
    return H


def pack_H(H: np.ndarray) -> np.ndarray:
    """Inverse of unpack_h for 1/2 y^T H y = phi_w · h."""
    d = H.shape[0]
    h = []
    for a in range(d):
        for b in range(a, d):
            if a == b:
                h.append(0.5 * float(H[a, a]))
            else:
                h.append(float(H[a, b]) / np.sqrt(2.0))
    return np.asarray(h, dtype=np.float64)


def quadratic_form(Y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """(1/2) y^T H y for rows of Y."""
    return 0.5 * np.einsum("ni,ij,nj->n", Y, H, Y)


def intersection_rank(A: np.ndarray, B: np.ndarray, *, cos_min: float = 0.5) -> int:
    if A.size == 0 or B.size == 0 or A.shape[1] == 0 or B.shape[1] == 0:
        return 0
    from geometry.physics_activation_atlas.tangent_reliability import principal_angles

    ang = principal_angles(A, B)
    return int(np.sum(np.cos(np.asarray(ang, dtype=np.float64)) >= cos_min))


def projector_overlap(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0 or A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    k = min(A.shape[1], B.shape[1])
    M = A[:, :k].T @ B[:, :k]
    return float(np.sum(M * M) / k)


def tangent_basis(A: np.ndarray, R: int) -> np.ndarray:
    """Orthonormal basis of ker(A^T) in R^R."""
    if A.size == 0 or A.shape[1] == 0:
        return np.eye(R)
    Q, _ = np.linalg.qr(A, mode="reduced")
    P = np.eye(R) - Q @ Q.T
    U, s, _ = np.linalg.svd(P, full_matrices=False)
    return U[:, s > 0.5]


def qr_orthonormal(A: np.ndarray) -> np.ndarray:
    if A.size == 0 or A.shape[1] == 0:
        return A
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q[:, : A.shape[1]]


def profiled_K(Y: np.ndarray, Phi: np.ndarray, lam: float, w: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    """K_λ = Ỹ^T (I - P_Φ) Ỹ after profiling quadratic coefficients."""
    n, R = Y.shape
    w = np.ones(n, dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)
    sw = np.sqrt(np.maximum(w, 0.0))
    Yw = Y * sw[:, None]
    Pw = Phi * sw[:, None]
    G = Pw.T @ Pw + lam * np.eye(Pw.shape[1])
    Cyy = Yw.T @ Yw
    Cyp = Yw.T @ Pw
    try:
        GiC = np.linalg.solve(G, Cyp.T)
    except np.linalg.LinAlgError:
        GiC = np.linalg.lstsq(G, Cyp.T, rcond=None)[0]
    K = Cyy - Cyp @ GiC
    K = 0.5 * (K + K.T)
    p = Pw.shape[1]
    try:
        Ginv = np.linalg.inv(G)
        df = float(p - lam * np.trace(Ginv))
    except np.linalg.LinAlgError:
        df = float("nan")
    return {"K": K, "G": G, "df": df, "Cyy": Cyy}


def fit_h_for_a(Y: np.ndarray, Phi: np.ndarray, a: np.ndarray, lam: float, w: np.ndarray | None = None) -> np.ndarray:
    n = Y.shape[0]
    w = np.ones(n) if w is None else w
    sw = np.sqrt(np.maximum(w, 0.0))
    Pw = Phi * sw[:, None]
    rhs = -(Pw.T @ ((Y @ a) * sw))
    G = Pw.T @ Pw + lam * np.eye(Pw.shape[1])
    try:
        return np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, rhs, rcond=None)[0]


def constraint_residuals(Y: np.ndarray, a: np.ndarray, h: np.ndarray, Phi: np.ndarray) -> dict[str, np.ndarray]:
    lin = Y @ a
    quad = Phi @ h
    return {"linear": lin, "quadratic": quad, "corrected": lin + quad}


def r2_cancel(lin: np.ndarray, corrected: np.ndarray) -> float:
    den = float(np.mean(lin * lin))
    if den < EPS:
        return float("nan")
    return float(1.0 - np.mean(corrected * corrected) / den)


def sampson_distance(Y: np.ndarray, A: np.ndarray, Hs: list[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    """d_F(y)^2 for rows of Y. JF_ℓ = a_ℓ + H_ℓ y."""
    n, R = Y.shape
    q = A.shape[1]
    if q == 0:
        return np.zeros(n)
    F = np.stack([Y @ A[:, ℓ] + quadratic_form(Y, Hs[ℓ]) for ℓ in range(q)], axis=1)
    out = np.empty(n)
    Ieps = eps * np.eye(q)
    for i in range(n):
        y = Y[i]
        JF = np.stack([A[:, ℓ] + Hs[ℓ] @ y for ℓ in range(q)], axis=0)  # (q, R)
        G = JF @ JF.T + Ieps
        f = F[i]
        try:
            out[i] = float(f @ np.linalg.solve(G, f))
        except np.linalg.LinAlgError:
            out[i] = float(np.dot(f, f))
    return out


def sampson_batch(Y: np.ndarray, A: np.ndarray, Hs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Hs: (q, R, R). Vectorized-enough Sampson for moderate q,R."""
    n, R = Y.shape
    q = A.shape[1]
    if q == 0:
        return np.zeros(n)
    F = Y @ A
    for ℓ in range(q):
        F[:, ℓ] = F[:, ℓ] + quadratic_form(Y, Hs[ℓ])
    # JF[i,ℓ,:] = A[:,ℓ] + Hs[ℓ] @ y_i  → (n, q, R)
    HY = np.einsum("qij,nj->nqi", Hs, Y)
    JF = A.T[None, :, :] + HY
    G = np.einsum("nqr,nsr->nqs", JF, JF) + eps * np.eye(q)
    try:
        sol = np.linalg.solve(G, F[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = F
    return np.einsum("nq,nq->n", F, sol)


def bottom_eigh(M: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Smallest-q eigenpairs; evals ascending, vecs as columns."""
    ev, U = np.linalg.eigh(0.5 * (M + M.T))
    q = int(min(q, len(ev)))
    return ev[:q], U[:, :q]


def stiefel_qr(A: np.ndarray, G: np.ndarray, step: float) -> np.ndarray:
    """One QR retraction step on the Stiefel manifold: A ← qr(A - step G)."""
    return qr_orthonormal(A - step * G)


def implicit_shape_operators(A: np.ndarray, Hs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S_ℓ = -T^T H_ℓ T and stacked metric-weighted B: Sym^2(T) → N.

    Identity: a_ℓ^T B(v,w) = -(Tv)^T H_ℓ (Tw).
    """
    R = A.shape[0]
    T = tangent_basis(A, R)
    d = T.shape[1]
    q = A.shape[1]
    Ss = np.stack([-T.T @ Hs[ℓ] @ T for ℓ in range(q)], axis=0) if q else np.zeros((0, d, d))
    # flatten each S with sqrt2 off-diag, stack as (q, n_sym)
    cols = []
    for ℓ in range(q):
        S = Ss[ℓ]
        v = []
        for a in range(d):
            for b in range(a, d):
                v.append(S[a, a] if a == b else np.sqrt(2.0) * S[a, b])
        cols.append(v)
    Bflat = np.asarray(cols, dtype=np.float64) if cols else np.zeros((0, n_quad_features(max(d, 1))))
    return Ss, Bflat


def loglog_exponent(radii: np.ndarray, amp: np.ndarray, *, min_points: int = 4) -> dict[str, float]:
    r = np.asarray(radii, dtype=np.float64)
    a = np.asarray(amp, dtype=np.float64)
    m = np.isfinite(r) & np.isfinite(a) & (r > EPS) & (a > EPS)
    if int(m.sum()) < min_points:
        return {"alpha": float("nan"), "n": int(m.sum()), "resolved": False}
    x = np.log(r[m])
    y = np.log(a[m])
    span = float(x.max() - x.min())
    A = np.column_stack([np.ones(len(x)), x])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        alpha = float(coef[1])
    except np.linalg.LinAlgError:
        return {"alpha": float("nan"), "n": int(m.sum()), "resolved": False}
    return {"alpha": alpha, "n": int(m.sum()), "resolved": bool(span >= np.log(1.8))}


def mixed_var_nnls(radii: np.ndarray, energy: np.ndarray, cols: str = "r2_r4_c") -> dict[str, float]:
    r = np.asarray(radii, dtype=np.float64)
    v = np.asarray(energy, dtype=np.float64)
    m = np.isfinite(r) & np.isfinite(v) & (r > EPS) & (v >= 0)
    if int(m.sum()) < 4:
        return {"identifiable": False, "resolved": False}
    x, y = r[m], v[m]
    if cols == "r6_c":
        design = np.column_stack([x**6, np.ones(len(x))])
        names = ["gamma", "c"]
    else:
        design = np.column_stack([x**2, x**4, np.ones(len(x))])
        names = ["alpha", "beta", "c"]
    try:
        from scipy.optimize import nnls

        coef, _ = nnls(design, y)
    except Exception:  # noqa: BLE001
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        coef = np.maximum(coef, 0.0)
    span = float(np.log(x.max() / max(x.min(), EPS)))
    out = {names[i]: float(coef[i]) for i in range(len(names))}
    pred = design @ coef[: design.shape[1]]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out["r2_fit"] = float(1.0 - ss_res / max(ss_tot, EPS))
    out["identifiable"] = bool(span >= np.log(1.8))
    out["resolved"] = bool(out["identifiable"] and out["r2_fit"] > 0.2)
    return out


__all__ = [
    "EPS",
    "RIDGES",
    "bottom_eigh",
    "constraint_residuals",
    "degenerate_blocks",
    "fit_h_for_a",
    "implicit_shape_operators",
    "intersection_rank",
    "loglog_exponent",
    "mixed_var_nnls",
    "n_quad_features",
    "pack_H",
    "profiled_K",
    "projector_overlap",
    "qr_orthonormal",
    "quadratic_features",
    "quadratic_form",
    "r2_cancel",
    "sampson_batch",
    "sampson_distance",
    "stiefel_qr",
    "tangent_basis",
    "unpack_h",
    "vech_weights",
    "weighted_phi",
]
