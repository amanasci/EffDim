"""Order-stratified algebra: B^S SVD, overlaps, mixed scaling, odd/even, chart refine.

q_2 is the reliable rank of the sphere-normal quadratic image, not extra
intrinsic dimension. Complements are taken inside a data-supported carrier S_R.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric
from geometry.physics_activation_atlas.quadratic import n_quad_features, quadratic_features
from geometry.physics_activation_atlas.quadratic_structure import truncate_B
from geometry.physics_activation_atlas.sphere_normal_quadratic import (
    NestedChart,
    flatten_BS_for_svd,
    normalize_rows,
)
from geometry.physics_activation_atlas.tangent_reliability import principal_angles
from geometry.physics_stable_tangent_dimension.nested_pca import (
    block_agreement,
    degenerate_blocks,
    prefix_agreement,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import EPS

from geometry.physics_stable_tangent_dimension.curvature_panel import flatten_sym2


def projector_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """(1/k) tr(P_A P_B) for orthonormal frames A, B. k = min(ncols)."""
    if A.size == 0 or B.size == 0 or A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    k = min(A.shape[1], B.shape[1])
    M = A[:, :k].T @ B[:, :k]
    return float(np.sum(M * M) / k)


def intersection_rank(A: np.ndarray, B: np.ndarray, *, cos_min: float = 0.7) -> int:
    """Effective intersection: number of principal cosines >= cos_min."""
    if A.size == 0 or B.size == 0 or A.shape[1] == 0 or B.shape[1] == 0:
        return 0
    s = np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0.0, 1.0)
    return int(np.sum(s >= cos_min))


def bs_metric_matrix(BS_flat: np.ndarray, d: int) -> np.ndarray:
    """Metric-whitened vech(B^S): off-diagonals scaled by sqrt(2). Shape (D, q)."""
    if BS_flat.size == 0:
        return BS_flat
    B = unpack_BS_symmetric(BS_flat, d)
    return flatten_sym2(B)


def svd_quadratic_image(BS_flat: np.ndarray, d: int) -> dict[str, np.ndarray]:
    """Left singular vectors of metric-flattened B^S are candidate normal directions."""
    M = bs_metric_matrix(BS_flat, d)
    if M.size == 0 or min(M.shape) == 0:
        return {
            "U": np.zeros((BS_flat.shape[0], 0)),
            "s": np.zeros(0),
            "Vt": np.zeros((0, M.shape[1] if M.ndim == 2 else 0)),
            "M": M,
        }
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return {"U": U, "s": s, "Vt": Vt, "M": M}


def _sym_weights(d: int) -> np.ndarray:
    w = []
    for a in range(d):
        for b in range(a, d):
            w.append(1.0 if a == b else np.sqrt(2.0))
    return np.asarray(w, dtype=np.float64)


def truncate_bs_left(BS_flat: np.ndarray, d: int, q: int) -> np.ndarray:
    """B_{12,q}^S = U_q U_q^T B_{12}^S in the metric-flattened basis, unscaled back."""
    info = svd_quadratic_image(BS_flat, d)
    M = info["M"]
    if q <= 0 or info["U"].shape[1] == 0:
        return np.zeros_like(BS_flat)
    q = min(q, info["U"].shape[1])
    Mq = info["U"][:, :q] @ (info["U"][:, :q].T @ M)
    w = _sym_weights(d)
    return Mq / np.maximum(w[None, :], EPS)


def cross_frobenius(BA: np.ndarray, BB: np.ndarray) -> float:
    """Split-cross energy <B^A, B^B>_F (unbiased |B|_F^2 estimator)."""
    if BA.size == 0 or BB.size == 0:
        return float("nan")
    return float(np.sum(BA * BB))


def whiten_tangent(U: np.ndarray, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if scale is None:
        scale = np.sqrt(np.maximum(np.mean(U**2, axis=0), EPS))
    return U / np.maximum(scale[None, :], EPS), scale


def fit_quadratic_map(U: np.ndarray, E: np.ndarray, ridge: float) -> np.ndarray:
    """Least-squares A in e ≈ phi(u) A^T. Returns A (d_e, q)."""
    Phi = quadratic_features(U)
    G = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    R = Phi.T @ E
    try:
        return np.linalg.solve(G, R).T
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, R, rcond=None)[0].T


def predict_quadratic_map(U: np.ndarray, A: np.ndarray) -> np.ndarray:
    return quadratic_features(U) @ A.T


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    num = float(np.mean(np.sum((y - yhat) ** 2, axis=1)))
    den = float(np.mean(np.sum(y * y, axis=1)))
    if den < EPS:
        return float("nan")
    return float(1.0 - num / den)


def per_col_r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    num = np.mean((y - yhat) ** 2, axis=0)
    den = np.mean(y * y, axis=0)
    out = 1.0 - num / np.maximum(den, EPS)
    out[den < EPS] = np.nan
    return out


def mixed_scale_nnls(radii: np.ndarray, energy: np.ndarray) -> dict[str, float]:
    """Nonnegative V(r) = a r^2 + b r^4 + c. Unresolved if radius span is too small."""
    r = np.asarray(radii, dtype=np.float64)
    v = np.asarray(energy, dtype=np.float64)
    m = np.isfinite(r) & np.isfinite(v) & (r > EPS) & (v >= 0)
    if int(m.sum()) < 4:
        return _unresolved_mix(int(m.sum()))
    x = r[m]
    y = v[m]
    span = float(np.log(x.max() / max(x.min(), EPS)))
    design = np.column_stack([x**2, x**4, np.ones(len(x))])
    try:
        from scipy.optimize import nnls

        coef, resid = nnls(design, y)
    except Exception:  # noqa: BLE001
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        coef = np.maximum(coef, 0.0)
        resid = float(np.sum((design @ coef - y) ** 2))
    a, b, c = (float(v) for v in coef[:3])
    pred = design @ np.array([a, b, c])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, EPS))
    identifiable = span >= np.log(1.8) and int(m.sum()) >= 4
    return {
        "a": a,
        "b": b,
        "c": c,
        "r2_fit": r2,
        "resid": float(resid) if np.isscalar(resid) else ss_res,
        "n": int(m.sum()),
        "r_span_log": span,
        "identifiable": bool(identifiable),
        "resolved": bool(identifiable and r2 > 0.2),
    }


def mix_shares(a: float, b: float, c: float, r: float) -> dict[str, float]:
    lin = a * r * r
    quad = b * (r**4)
    thick = c
    tot = lin + quad + thick
    if tot < EPS:
        return {"pi_lin": float("nan"), "pi_quad": float("nan"), "pi_thick": float("nan")}
    return {
        "pi_lin": float(lin / tot),
        "pi_quad": float(quad / tot),
        "pi_thick": float(thick / tot),
    }


def _unresolved_mix(n: int) -> dict[str, float]:
    return {
        "a": float("nan"),
        "b": float("nan"),
        "c": float("nan"),
        "r2_fit": float("nan"),
        "resid": float("nan"),
        "n": n,
        "r_span_log": float("nan"),
        "identifiable": False,
        "resolved": False,
    }


def pair_antipodes(
    U: np.ndarray,
    radii: np.ndarray,
    *,
    cos_min: float = 0.85,
    radius_rel: float = 0.25,
) -> dict[str, Any]:
    """One-to-one radius-matched antipodal pairs in core coordinates."""
    U = np.asarray(U, dtype=np.float64)
    n = len(U)
    if n < 8:
        return {"n_pairs": 0, "plus": np.zeros(0, dtype=np.int64), "minus": np.zeros(0, dtype=np.int64), "quality": float("nan")}
    Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), EPS)
    C = Un @ Un.T
    used = np.zeros(n, dtype=bool)
    plus, minus, quals = [], [], []
    # greedy: most antipodal unused pairs (C ≈ -1) with matched radii
    score = C.copy()
    np.fill_diagonal(score, np.inf)
    order = np.argsort(score, axis=None)
    for idx in order:
        i, j = divmod(int(idx), n)
        if i >= j or used[i] or used[j]:
            continue
        if C[i, j] > -cos_min:
            break
        ri, rj = float(radii[i]), float(radii[j])
        if max(ri, rj) < EPS:
            continue
        if abs(ri - rj) / max(ri, rj, EPS) > radius_rel:
            continue
        used[i] = used[j] = True
        plus.append(i)
        minus.append(j)
        quals.append(float(-C[i, j]))  # 1 = perfect antipode
        if len(plus) >= n // 4:
            break
    return {
        "n_pairs": int(len(plus)),
        "plus": np.asarray(plus, dtype=np.int64),
        "minus": np.asarray(minus, dtype=np.int64),
        "quality": float(np.mean(quals)) if quals else float("nan"),
    }


def odd_even_displacements(Z: np.ndarray, plus: np.ndarray, minus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zp = Z[plus]
    zm = Z[minus]
    return 0.5 * (zp - zm), 0.5 * (zp + zm)


def pca_subspace(Z: np.ndarray, d: int) -> np.ndarray:
    if len(Z) < max(d + 1, 3) or d < 1:
        return np.zeros((Z.shape[1], 0))
    _, _, Vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
    return Vt[:d].T


def refine_chart_coords(
    chart: NestedChart,
    X: np.ndarray,
    U0: np.ndarray,
    *,
    n_iter: int = 4,
    fd_eps: float = 1e-4,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Gauss–Newton refine of u so decode_TRS(u) ≈ X. Batched finite differences."""
    U = np.asarray(U0, dtype=np.float64).copy()
    n, d = U.shape
    if n == 0 or d == 0:
        return U
    eye = np.eye(d)
    for _ in range(n_iter):
        pred = chart.decode_TRS(U)
        r = pred - X
        cols = []
        for a in range(d):
            Up = U.copy()
            Up[:, a] += fd_eps
            cols.append((chart.decode_TRS(Up) - pred) / fd_eps)
        J = np.stack(cols, axis=-1)
        JTJ = np.einsum("ndi,ndj->nij", J, J) + ridge * eye
        JTr = np.einsum("ndi,nd->ni", J, r)
        try:
            du = np.linalg.solve(JTJ, -JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        U = U + du
    return U


def ambient_mse(pred: np.ndarray, X: np.ndarray) -> float:
    return float(np.mean(np.sum((pred - X) ** 2, axis=1)))


def sphere_normal_residual(X: np.ndarray, x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Project displacements into N_S = span(x0, J)^perp."""
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    Q, _ = np.linalg.qr(np.column_stack([x0u, J]), mode="reduced")
    Z = X - x0u[None, :]
    return Z - (Z @ Q) @ Q.T


__all__ = [
    "EPS",
    "NestedChart",
    "ambient_mse",
    "block_agreement",
    "bs_metric_matrix",
    "cross_frobenius",
    "degenerate_blocks",
    "fit_quadratic_map",
    "intersection_rank",
    "mix_shares",
    "mixed_scale_nnls",
    "n_quad_features",
    "normalize_rows",
    "odd_even_displacements",
    "pair_antipodes",
    "pca_subspace",
    "per_col_r2",
    "prefix_agreement",
    "principal_angles",
    "predict_quadratic_map",
    "projector_overlap",
    "quadratic_features",
    "r2_score",
    "refine_chart_coords",
    "sphere_normal_residual",
    "svd_quadratic_image",
    "truncate_B",
    "truncate_bs_left",
    "whiten_tangent",
]
