"""Q0–Q5 ablation. Q1–Q4 are residual-quadratic; Q5 is frozen production fit_quad."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.known_curvature_point_patch_fixture_audit.estimator_q import (  # noqa: E402
    fit_anchor_quadratic,
)
from geometry.known_curvature_point_patch_fixture_audit.geometry import (  # noqa: E402
    EPS,
    cosine,
    curvature_from_B,
    hess_from_bs_flat,
    rel_err,
    whiten_B,
)
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES  # noqa: E402

from .config import D_LAT, PINV_RCOND_MULT, Q_FEATURES, RIDGE_VAL_FRACTION


def quad_phi(U: np.ndarray) -> np.ndarray:
    cols = []
    for a in range(U.shape[1]):
        for b in range(a, U.shape[1]):
            cols.append(U[:, a] * U[:, b])
    return np.stack(cols, axis=1)


def chart_coords(Y: np.ndarray, x0: np.ndarray, J: np.ndarray, U_exact: np.ndarray | None = None) -> np.ndarray:
    if U_exact is not None:
        return np.asarray(U_exact, dtype=np.float64)
    g = J.T @ J
    try:
        ginv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        ginv = np.linalg.pinv(g)
    return (Y - x0[None, :]) @ J @ ginv


def _sphere_metric(x0: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """g, ginv, unit radial. Avoids forming D×D projectors."""
    Gh = x0 / max(float(np.linalg.norm(x0)), EPS)
    g = J.T @ J
    try:
        ginv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        ginv = np.linalg.pinv(g)
    return g, ginv, Gh


def _apply_pns(V: np.ndarray, Gh: np.ndarray, J: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """P_{N,S} without materialising I_D. (n, D) rows or (D, q) columns."""
    if V.shape[1] == Gh.shape[0]:
        return V - np.outer(V @ Gh, Gh) - (V @ J) @ ginv @ J.T
    return V - np.outer(Gh, Gh @ V) - J @ (ginv @ (J.T @ V))


def _design_from_svals(n: int, q: int, svals: np.ndarray, ridge: float) -> dict:
    cond = float(svals[0] / max(svals[-1], 1e-30)) if len(svals) else float("nan")
    if ridge > 0:
        edf = float(np.sum(svals**2 / (svals**2 + ridge)))
        rank = int(np.sum(svals > 1e-12 * svals[0])) if len(svals) else 0
        shrink = edf / max(q, 1)
        note = "production RIDGES val pick"
    else:
        rcond = PINV_RCOND_MULT * max(n, q) * np.finfo(np.float64).eps
        cutoff = rcond * (svals[0] if len(svals) else 1.0)
        rank = int(np.sum(svals > cutoff)) if len(svals) else 0
        edf = float(rank)
        shrink = 1.0
        note = "thin-SVD least squares, machine-eps cutoff"
    return {
        "n_obs": int(n),
        "q": int(q),
        "design_rank": rank,
        "cond": cond,
        "edf": edf,
        "ridge": float(ridge),
        "shrinkage": float(shrink),
        "rcond_note": note,
    }


def _fast_solve(Phi: np.ndarray, R: np.ndarray, ridge: float = 0.0) -> tuple[np.ndarray, dict]:
    """Solve Phi S^T ≈ R. Thin SVD when unregularized; q×q ridge otherwise."""
    n, q = Phi.shape
    U, svals, Vt = np.linalg.svd(Phi, full_matrices=False)
    if ridge > 0:
        S = (Vt.T * (svals / (svals**2 + ridge))) @ (U.T @ R)
        S = S.T
    else:
        rcond = PINV_RCOND_MULT * max(n, q) * np.finfo(np.float64).eps
        cutoff = rcond * (svals[0] if len(svals) else 1.0)
        s_inv = np.where(svals > cutoff, 1.0 / np.clip(svals, 1e-30, None), 0.0)
        S = (Vt.T * s_inv) @ (U.T @ R)
        S = S.T
    diag = _design_from_svals(n, q, svals, ridge)
    diag["resid_mse"] = float(np.mean(np.sum((R - Phi @ S.T) ** 2, axis=1)))
    return S, diag


def fit_residual_quadratic(
    Y: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    *,
    U: np.ndarray | None = None,
    ridge: float = 0.0,
    pick_ridge: bool = False,
) -> dict:
    x0 = np.asarray(x0, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    g, ginv, Gh = _sphere_metric(x0, J)
    U = chart_coords(Y, x0, J, U)
    lin = x0[None, :] + U @ J.T
    resid = _apply_pns(Y - lin, Gh, J, ginv)
    Phi = quad_phi(U)
    chosen = float(ridge)
    if pick_ridge:
        rng = np.random.default_rng(0)
        n = len(Y)
        perm = rng.permutation(n)
        n_val = max(8, int(round(RIDGE_VAL_FRACTION * n)))
        val, fit = perm[:n_val], perm[n_val:]
        Gf = Phi[fit].T @ Phi[fit]
        Rf = Phi[fit].T @ resid[fit]
        Pv = Phi[val]
        Rv = resid[val]
        best_lam, best_loss = RIDGES[0], float("inf")
        eye = np.eye(Phi.shape[1])
        for lam in RIDGES:
            S = np.linalg.solve(Gf + float(lam) * eye, Rf).T
            loss = float(np.mean(np.sum((Rv - Pv @ S.T) ** 2, axis=1)))
            if loss < best_loss:
                best_loss, best_lam = loss, float(lam)
        chosen = best_lam
    S, diag = _fast_solve(Phi, resid, ridge=chosen)
    S = _apply_pns(S, Gh, J, ginv)
    Hess = hess_from_bs_flat(S, J.shape[1])
    curv = curvature_from_B(Hess, g)
    return {
        "ok": True,
        "Hess": Hess,
        "S": S,
        "J": J,
        "x0": x0,
        "g": g,
        **curv,
        "design": diag,
    }


def procrustes_R(J_est: np.ndarray, J_true: np.ndarray) -> np.ndarray:
    M = J_est.T @ J_true
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def principal_angles(J_est: np.ndarray, J_true: np.ndarray) -> dict:
    def orth(J):
        q, _ = np.linalg.qr(J)
        return q[:, : J.shape[1]]

    A, B = orth(J_est), orth(J_true)
    s = np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0.0, 1.0)
    ang = np.arccos(s)
    return {
        "max_principal_angle": float(np.max(ang)),
        "mean_principal_angle": float(np.mean(ang)),
        "procrustes_resid": float(np.linalg.norm(A @ (A.T @ B) - B)),
    }


def align_tensor(B: np.ndarray, J_est: np.ndarray, J_true: np.ndarray) -> np.ndarray:
    R = procrustes_R(J_est, J_true)
    Bw = whiten_B(B, J_est.T @ J_est)
    return np.einsum("Dab,ai,bj->Dij", Bw, R, R)


def tensor_errors(Bhat: np.ndarray, ghat: np.ndarray, Btrue: np.ndarray | None, gtrue: np.ndarray | None, J_est=None, J_true=None) -> dict:
    out = {}
    if Btrue is None or gtrue is None:
        out.update({k: float("nan") for k in ("rel_B", "rel_H", "cos_H", "rel_Btf", "rel_Kdir", "rel_B_aligned")})
        return out
    ct = curvature_from_B(Btrue, gtrue)
    ch = curvature_from_B(Bhat, ghat)
    out["rel_B"] = rel_err(Bhat, Btrue)
    out["rel_H"] = rel_err(ch["H"], ct["H"])
    out["cos_H"] = cosine(ch["H"], ct["H"])
    out["rel_Btf"] = rel_err(ch["Btf"], ct["Btf"])
    out["rel_Kdir"] = rel_err(ch["K_dir"], ct["K_dir"])
    if J_est is not None and J_true is not None:
        Bal = align_tensor(Bhat, J_est, J_true)
        Bt_w = whiten_B(Btrue, gtrue)
        out["rel_B_aligned"] = rel_err(Bal, Bt_w)
    else:
        out["rel_B_aligned"] = out["rel_B"]
    return out


def q1_exact(Y, x0, J, U=None) -> dict:
    return fit_residual_quadratic(Y, x0, J, U=U, ridge=0.0, pick_ridge=False)


def q2_exact_ridge(Y, x0, J, U=None) -> dict:
    return fit_residual_quadratic(Y, x0, J, U=U, pick_ridge=True)


def numpy_pca_frame(Xloc: np.ndarray, d: int):
    """Same sphere-tangent PCA as nested_pca_frame, numpy SVD (CPU, no torch)."""
    x0 = Xloc.mean(0)
    nrm = float(np.linalg.norm(x0))
    x0 = x0 / max(nrm, 1e-15)
    dx = Xloc - x0[None, :]
    dx = dx - np.outer(dx @ x0, x0)
    _, S, Vt = np.linalg.svd(dx, full_matrices=False)
    J = Vt[:d].T.copy()
    J = J - np.outer(x0, x0 @ J)
    qj, _ = np.linalg.qr(J)
    J = qj[:, :d]
    ev = (S[:d] ** 2) / max(len(Xloc), 1)
    return x0, J, ev, {"backend": "numpy_svd", "eigengap": float(ev[d - 1] - S[d] ** 2 / max(len(Xloc), 1)) if len(S) > d else float("nan")}


def q3_pca(Y, d=D_LAT, frame=None) -> dict:
    if frame is None:
        x0, J, ev, diag = numpy_pca_frame(Y, d)
    else:
        x0, J, ev, diag = frame
    rec = fit_residual_quadratic(Y, x0, J[:, :d], U=None, ridge=0.0)
    rec["pca_ev"] = ev
    rec["pca_diag"] = diag
    return rec


def q4_pca_ridge(Y, d=D_LAT, frame=None) -> dict:
    if frame is None:
        x0, J, ev, diag = numpy_pca_frame(Y, d)
    else:
        x0, J, ev, diag = frame
    rec = fit_residual_quadratic(Y, x0, J[:, :d], U=None, pick_ridge=True)
    rec["pca_ev"] = ev
    rec["pca_diag"] = diag
    return rec


def q5_production(Y, d=D_LAT, seed=0, ai=0) -> dict:
    fit = fit_anchor_quadratic(Y, d, n_splits=1, seed=seed, ai=ai, device=torch.device("cpu"))
    if not fit.get("ok"):
        return {"ok": False}
    HA, HB = fit["Hess_A"], fit["Hess_B"]
    Hess = 0.5 * (HA + HB) if HB is not None else HA
    g = fit["J"][:, :d].T @ fit["J"][:, :d]
    curv = curvature_from_B(Hess, g)
    return {
        "ok": True,
        "Hess": Hess,
        "Hess_A": HA,
        "Hess_B": HB,
        "J": fit["J"][:, :d],
        "x0": fit["x0"],
        "g": g,
        "agg": fit.get("agg", {}),
        "design": {
            "n_obs": int(fit["n_loc"]),
            "q": Q_FEATURES,
            "n_half": int(fit["n_loc"] // 2),
            "ridge": "frozen RIDGES A/B",
            "shrinkage": float("nan"),
            "design_rank": float("nan"),
            "cond": float("nan"),
            "edf": float("nan"),
            "resid_mse": float("nan"),
        },
        **curv,
    }
