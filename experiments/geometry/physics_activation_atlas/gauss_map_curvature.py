"""Sphere-corrected spatial tangent dispersion (Gauss-map style curvature).

Parallel-transport neighbouring tangent projectors on the unit sphere, then
regress ΔP² ~ β d_S² to estimate curvature energy ||B^S||_F² ≈ (d/2) β.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12


def sphere_log(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Log map on unit sphere at p of point q (tangent vector at p)."""
    p = p / max(np.linalg.norm(p), EPS)
    q = q / max(np.linalg.norm(q), EPS)
    cos_th = float(np.clip(np.dot(p, q), -1.0, 1.0))
    th = float(np.arccos(cos_th))
    if th < 1e-10:
        return np.zeros_like(p)
    v = q - cos_th * p
    nv = np.linalg.norm(v)
    if nv < EPS:
        return np.zeros_like(p)
    return v * (th / nv)


def sphere_exp(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exp map on unit sphere at p along tangent v."""
    p = p / max(np.linalg.norm(p), EPS)
    # ensure tangent
    v = v - np.dot(v, p) * p
    th = float(np.linalg.norm(v))
    if th < 1e-10:
        return p.copy()
    return np.cos(th) * p + np.sin(th) * (v / th)


def sphere_geodesic_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(np.linalg.norm(p), EPS)
    q = q / max(np.linalg.norm(q), EPS)
    return float(np.arccos(np.clip(np.dot(p, q), -1.0, 1.0)))


def parallel_transport_sphere(p: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Parallel-transport tangent vector v at p to q along the great-circle geodesic.

    Uses the closed-form S^{D-1} formula in span{p, q, v}.
    """
    p = p / max(np.linalg.norm(p), EPS)
    q = q / max(np.linalg.norm(q), EPS)
    v = v - np.dot(v, p) * p  # ensure tangent at p
    cos_th = float(np.clip(np.dot(p, q), -1.0, 1.0))
    th = float(np.arccos(cos_th))
    if th < 1e-10:
        return v - np.dot(v, q) * q
    # unit tangent of geodesic at p
    e = (q - cos_th * p) / max(np.sin(th), EPS)
    # decompose v = a e + w, w ⟂ span{p,e}
    a = float(np.dot(v, e))
    w = v - a * e
    # transport: e → -sin(θ)p + cos(θ)e ; w unchanged then project to T_q
    e_q = -np.sin(th) * p + np.cos(th) * e
    v_q = a * e_q + w
    v_q = v_q - np.dot(v_q, q) * q
    return v_q


def parallel_transport_basis(p: np.ndarray, q: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Transport orthonormal tangent basis J at p to q; re-orthonormalize in T_q."""
    cols = [parallel_transport_sphere(p, q, J[:, i]) for i in range(J.shape[1])]
    M = np.column_stack(cols)
    # project out q and QR
    M = M - np.outer(q, q) @ M
    Q, _ = np.linalg.qr(M)
    return Q[:, : J.shape[1]]


def parallel_transport_projector(p: np.ndarray, q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Transport rank-d projector P at p to q via its eigenspace basis."""
    # P = J J^T; take top eigenspace
    evals, evecs = np.linalg.eigh(P)
    d = int(np.sum(evals > 0.5))
    d = max(d, 1)
    J = evecs[:, -d:]
    Jq = parallel_transport_basis(p, q, J)
    return Jq @ Jq.T


def projector_frobenius_sq(P1: np.ndarray, P2: np.ndarray) -> float:
    return float(np.linalg.norm(P1 - P2, "fro") ** 2)


def delta_P2_sphere(
    x: np.ndarray,
    y: np.ndarray,
    Px: np.ndarray,
    Py: np.ndarray,
) -> tuple[float, float]:
    """Sphere-corrected projector discrepancy and geodesic distance."""
    Py_at_x = parallel_transport_projector(y, x, Py)
    return projector_frobenius_sq(Px, Py_at_x), sphere_geodesic_distance(x, y)


def split_half_projectors(
    Xn: np.ndarray,
    x0: np.ndarray,
    d: int,
    seed: int,
    pca_fn,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Two independent PCA projectors from disjoint halves; return P1,P2,Δ_split²."""
    rng = np.random.default_rng(seed)
    n = len(Xn)
    perm = rng.permutation(n)
    h = n // 2
    if h < d + 2:
        J, _, _ = pca_fn(Xn, x0, d)
        P = J @ J.T
        return P, P, 0.0
    J1, _, _ = pca_fn(Xn[perm[:h]], x0, d)
    J2, _, _ = pca_fn(Xn[perm[h : 2 * h]], x0, d)
    P1, P2 = J1 @ J1.T, J2 @ J2.T
    return P1, P2, projector_frobenius_sq(P1, P2)


def debiased_delta(
    delta_obs: float,
    split_i: float,
    split_j: float,
) -> float:
    """Signed debiased Δ²; do not clamp for fitting."""
    return float(delta_obs - 0.5 * split_i - 0.5 * split_j)


def regress_gauss_map(
    delta2: np.ndarray,
    dS2: np.ndarray,
) -> dict[str, Any]:
    """Fit ΔP² = α + β d_S² (ordinary LS). ||B^S||_F² ≈ (d/2)*β reported separately."""
    m = np.isfinite(delta2) & np.isfinite(dS2) & (dS2 > 0)
    if m.sum() < 4:
        return {"alpha": float("nan"), "beta": float("nan"), "n": int(m.sum()), "r2": float("nan")}
    x = dS2[m]
    y = delta2[m]
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "n": int(m.sum()),
        "r2": float(1 - ss_res / max(ss_tot, EPS)),
    }


def curvature_energy_from_beta(beta: float, d: int) -> float:
    """||B^S||_F² ≈ (d/2) β under the Gauss-map small-distance expansion."""
    return float(0.5 * d * beta)


def scale_convergence_score(
    delta2: np.ndarray,
    dS: np.ndarray,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Assess whether ΔP² / d_S² plateaus at small distance."""
    m = np.isfinite(delta2) & np.isfinite(dS) & (dS > 1e-8)
    if m.sum() < 8:
        return {"score": float("nan"), "label": "unresolved", "plateau_cv": float("nan")}
    ratio = delta2[m] / (dS[m] ** 2)
    # smallest-distance half
    order = np.argsort(dS[m])
    k = max(4, len(order) // 3)
    small = ratio[order[:k]]
    large = ratio[order[-k:]]
    cv = float(np.std(small) / max(np.mean(np.abs(small)), EPS))
    drift = float(np.abs(np.median(small) - np.median(large)) / max(np.abs(np.median(small)), EPS))
    if cv < 0.35 and drift < 0.5 and np.median(small) > 0:
        lab = "pointwise_gauss_regime"
    elif cv < 0.6 and np.median(small) > 0:
        lab = "finite_scale_tangent_heterogeneity"
    elif np.median(small) <= 0 or cv > 1.5:
        lab = "noise_dominated"
    else:
        lab = "unresolved"
    return {"score": float(1.0 / (1.0 + cv + drift)), "label": lab, "plateau_cv": cv, "drift": drift}


def estimate_anchor_gauss_map(
    x0: np.ndarray,
    Px: np.ndarray,
    sites: list[tuple[np.ndarray, np.ndarray]],
    split_x: float,
    split_sites: list[float],
    d: int,
) -> dict[str, Any]:
    """sites: list of (y, Py); split_sites aligned split Δ² at those sites."""
    deltas_obs, deltas_deb, dS, dS2 = [], [], [], []
    for (y, Py), sj in zip(sites, split_sites):
        d2, ds = delta_P2_sphere(x0, y, Px, Py)
        deltas_obs.append(d2)
        deltas_deb.append(debiased_delta(d2, split_x, sj))
        dS.append(ds)
        dS2.append(ds**2)
    deltas_obs_a = np.asarray(deltas_obs, float)
    deltas_deb_a = np.asarray(deltas_deb, float)
    dS_a = np.asarray(dS, float)
    dS2_a = np.asarray(dS2, float)
    fit = regress_gauss_map(deltas_deb_a, dS2_a)
    conv = scale_convergence_score(deltas_deb_a, dS_a)
    energy = curvature_energy_from_beta(fit["beta"], d) if np.isfinite(fit["beta"]) else float("nan")
    return {
        **fit,
        **conv,
        "curvature_energy": energy,
        "median_delta_obs": float(np.nanmedian(deltas_obs_a)) if len(deltas_obs_a) else float("nan"),
        "median_delta_deb": float(np.nanmedian(deltas_deb_a)) if len(deltas_deb_a) else float("nan"),
        "n_pairs": int(len(sites)),
        # plotting helper (clamped)
        "median_delta_deb_plot": float(np.nanmedian(np.maximum(deltas_deb_a, 0.0)))
        if len(deltas_deb_a)
        else float("nan"),
    }
