"""Metric-aware Procrustes alignment and matched-target scores."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .config import N_TANGENT_DIRS, PROJ_ERR_INVALID, DIRECTION_SEED
from .fixtures import energy_g


def axes_vec(H_est: np.ndarray, H_true: np.ndarray) -> dict:
    he = np.linalg.norm(H_est, axis=1)
    ht = np.linalg.norm(H_true, axis=1)
    num = (H_est * H_true).sum(axis=1)
    den = np.maximum(he * ht, 1e-30)
    rel = np.linalg.norm(H_est - H_true, axis=1) / np.maximum(ht, 1e-12)
    rho = float("nan")
    if np.std(ht) > 1e-12 and np.std(he) > 1e-12:
        rho = float(spearmanr(he, ht).statistic)
    return {
        "rho": rho,
        "median_cosine": float(np.median(num / den)),
        "median_ratio": float(np.median(he / np.maximum(ht, 1e-12))),
        "median_rel_err": float(np.median(rel)),
        "constant_truth": bool(np.std(ht) <= 1e-12 * max(float(np.median(ht)), 1.0)),
    }


def spearman_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or float(np.std(a[m])) < 1e-15 or float(np.std(b[m])) < 1e-15:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def principal_angles(Ja: np.ndarray, Jb: np.ndarray) -> np.ndarray:
    Qa, _ = np.linalg.qr(Ja)
    Qb, _ = np.linalg.qr(Jb)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)


def procrustes_metric(J_est: np.ndarray, J_true: np.ndarray, g_est: np.ndarray, g_true: np.ndarray):
    """Align estimated orthonormal frame to true frame. Returns R (d,d) acting on chart indices."""
    def onb(J, g):
        w, v = np.linalg.eigh(g)
        gmh = (v * (1.0 / np.sqrt(np.clip(w, 1e-15, None)))) @ v.T
        return J @ gmh

    E_est = onb(J_est, g_est)
    E_true = onb(J_true, g_true)
    M = E_est.T @ E_true
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt  # E_est @ R ≈ E_true
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    E_al = E_est @ R
    proj_err = float(np.linalg.norm(E_al @ E_al.T - E_true @ E_true.T) / max(np.linalg.norm(E_true @ E_true.T), 1e-12))
    ang = principal_angles(E_est, E_true)
    return R, proj_err, ang


def transform_B(B: np.ndarray, R: np.ndarray) -> np.ndarray:
    """B'_cd = B_ab R_ac R_bd  (chart change)."""
    return np.einsum("iab,ac,bd->icd", B, R, R)


def tensor_scores(B_est: np.ndarray, B_true: np.ndarray, ginv: np.ndarray) -> dict:
    num = float(np.tensordot(B_est, B_true, axes=([0, 1, 2], [0, 1, 2])))
    na, nb = float(np.linalg.norm(B_est)), float(np.linalg.norm(B_true))
    cos = float(num / max(na * nb, 1e-30))
    ratio = float(na / max(nb, 1e-12))
    rel = float(np.linalg.norm(B_est - B_true) / max(nb, 1e-12))
    e_est, e_true = energy_g(B_est, ginv), energy_g(B_true, ginv)
    return {
        "tensor_cosine": cos,
        "tensor_ratio": ratio,
        "tensor_rel": rel,
        "energy_ratio": float(np.sqrt(e_est) / max(np.sqrt(e_true), 1e-12)),
    }


def compare_aligned(est: dict, truth: dict, *, kind: str) -> dict:
    """kind in {full, residual}. Align est chart to truth chart; compare ambient bilinear forms."""
    keyB = "II_E" if kind == "full" else "B_S"
    keyH = "H_E" if kind == "full" else "H_S"
    R, proj_err, ang = procrustes_metric(est["J"], truth["J"], est["g"], truth["g"])
    valid = proj_err <= PROJ_ERR_INVALID
    B_al = transform_B(est[keyB], R)
    ts = tensor_scores(B_al, truth[keyB], truth["ginv"]) if valid else {k: float("nan") for k in ("tensor_cosine", "tensor_ratio", "tensor_rel", "energy_ratio")}
    H_est, H_true = est[keyH], truth[keyH]
    he, ht = float(np.linalg.norm(H_est)), float(np.linalg.norm(H_true))
    hcos = float(np.dot(H_est, H_true) / max(he * ht, 1e-30))
    hratio = float(he / max(ht, 1e-12))
    # trace-free
    d = truth["g"].shape[0]
    mean_e = np.einsum("ab,i->iab", est["g"], est[keyH] / d)
    mean_t = np.einsum("ab,i->iab", truth["g"], truth[keyH] / d)
    Btf_e = transform_B(est[keyB] - mean_e, R)
    Btf_t = truth[keyB] - mean_t
    tf = tensor_scores(Btf_e, Btf_t, truth["ginv"]) if valid else {k: float("nan") for k in ("tensor_cosine", "tensor_ratio", "tensor_rel", "energy_ratio")}
    rng = np.random.default_rng(DIRECTION_SEED)
    dirs = rng.standard_normal((N_TANGENT_DIRS, d))
    dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-15, None)
    def bend(B, v):
        return np.einsum("iab,na,nb->ni", B, v, v)
    be, bt = bend(B_al, dirs), bend(truth[keyB], dirs)
    dir_rel = float(np.median(np.linalg.norm(be - bt, axis=1) / np.maximum(np.linalg.norm(bt, axis=1), 1e-12)))
    gerr = float(np.linalg.norm(est["g"] - truth["g"]) / max(np.linalg.norm(truth["g"]), 1e-12))
    return {
        "valid_tensor": bool(valid),
        "proj_err": proj_err,
        "max_principal_angle": float(np.max(ang)),
        "metric_rel": gerr,
        "H_cosine": hcos,
        "H_ratio": hratio,
        "H_rel": float(np.linalg.norm(H_est - H_true) / max(ht, 1e-12)),
        "tf_cosine": tf["tensor_cosine"],
        "tf_ratio": tf["tensor_ratio"],
        "dir_rel": dir_rel,
        "Scal_est": est.get("Scal", float("nan")),
        "Scal_true": truth.get("Scal", float("nan")),
        **ts,
    }
