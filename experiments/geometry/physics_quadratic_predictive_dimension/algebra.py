"""Quadratic predictive dimension: features, ridge, closest-point projection.

f(u) = J u + B phi_2(u), f(0)=0, Df(0)=J. No intercept.
phi_2 uses sqrt(2) off-diagonals so it matches Frobenius geometry of Sym^2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from geometry.physics_stable_tangent_dimension.sphere_coords import EPS

N_QUAD = lambda d: d * (d + 1) // 2


def n_quad_features(d: int) -> int:
    return d * (d + 1) // 2


def vech_weights(d: int) -> np.ndarray:
    w = []
    for a in range(d):
        for b in range(a, d):
            w.append(1.0 if a == b else np.sqrt(2.0))
    return np.asarray(w, dtype=np.float64)


def phi2(U: np.ndarray) -> np.ndarray:
    """Metric-aware degree-two map: (u_a^2, sqrt(2) u_a u_b)."""
    U = np.asarray(U, dtype=np.float64)
    n, d = U.shape
    if d == 0:
        return np.zeros((n, 0), dtype=np.float64)
    ii, jj = np.triu_indices(d)
    Phi = U[:, ii] * U[:, jj]
    Phi[:, ii != jj] *= np.sqrt(2.0)
    return Phi


def scale_phi_train(Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column RMS from training only. No mean-centering (keeps f(0)=0)."""
    rms = np.sqrt(np.mean(Phi * Phi, axis=0))
    rms = np.maximum(rms, EPS)
    return Phi / rms[None, :], rms


def ridge_grid_from_gram(G: np.ndarray, n: int, n_grid: int = 11) -> np.ndarray:
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    smax = float(max(ev.max() if len(ev) else 1.0, EPS))
    scale = smax / max(int(n), 1)
    return np.geomspace(1e-6, 1e2, int(n_grid)) * scale


def ridge_fit(Phi: np.ndarray, Y: np.ndarray, lam: float, *, G=None, C=None) -> np.ndarray:
    """Y ≈ Phi B^T, B shape (D, p). Intercept-free."""
    if G is None:
        G = Phi.T @ Phi
    if C is None:
        C = Phi.T @ Y
    p = G.shape[0]
    A = G + lam * np.eye(p)
    try:
        B = np.linalg.solve(A, C).T
    except np.linalg.LinAlgError:
        B = np.linalg.lstsq(A, C, rcond=None)[0].T
    return B


def ridge_df(G: np.ndarray, lam: float) -> float:
    """tr(G (G + λI)^{-1})."""
    p = G.shape[0]
    try:
        inv = np.linalg.inv(G + lam * np.eye(p))
        return float(np.trace(G @ inv))
    except np.linalg.LinAlgError:
        return float("nan")


def unpack_B_to_H(B: np.ndarray, d: int) -> np.ndarray:
    """Ambient quadratic tensors H[i] with u^T H[i] u = (B phi)_i."""
    D = B.shape[0]
    H = np.zeros((D, d, d), dtype=np.float64)
    idx = 0
    s2 = np.sqrt(2.0)
    for a in range(d):
        for b in range(a, d):
            if a == b:
                H[:, a, a] = B[:, idx]
            else:
                H[:, a, b] = H[:, b, a] = B[:, idx] / s2
            idx += 1
    return H


def predict_f(U: np.ndarray, J: np.ndarray, B: np.ndarray) -> np.ndarray:
    return U @ J.T + phi2(U) @ B.T


def jacobian_batch(U: np.ndarray, J: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Df(u): shape (n, D, d). Df(0)=J."""
    return J[None, :, :] + 2.0 * np.einsum("iab,nb->nia", H, U, optimize=True)


def remove_radial_rows(Z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Project rows of Z (n, D) orthogonal to unit-ish x."""
    u = np.asarray(x, dtype=np.float64)
    nrm = float(np.linalg.norm(u))
    if nrm < EPS:
        return Z
    u = u / nrm
    return Z - np.outer(Z @ u, u)


def remove_radial_cols(B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Project columns of B (D, p) orthogonal to x."""
    u = np.asarray(x, dtype=np.float64)
    nrm = float(np.linalg.norm(u))
    if nrm < EPS:
        return B
    u = u / nrm
    return B - np.outer(u, u @ B)


def project_B_normal(B: np.ndarray, J: np.ndarray) -> np.ndarray:
    """B^N = (I - J J^T) B."""
    return B - J @ (J.T @ B)


def nmse(Z: np.ndarray, Zhat: np.ndarray) -> float:
    num = float(np.sum((Z - Zhat) ** 2))
    den = float(np.sum(Z * Z))
    if den < EPS:
        return float("nan")
    return num / den


def r2_from_nmse(val: float) -> float:
    return float(1.0 - val) if np.isfinite(val) else float("nan")


def component_r2(Z: np.ndarray, Zhat: np.ndarray, Pcols: np.ndarray) -> float:
    """R_C^2 = 1 - |P_C (z-hat)|^2 / |P_C z|^2 using orthonormal columns of C."""
    if Pcols.size == 0 or Pcols.shape[1] == 0:
        return float("nan")
    Cz = Z @ Pcols
    Cr = (Z - Zhat) @ Pcols
    den = float(np.sum(Cz * Cz))
    if den < EPS:
        return float("nan")
    return float(1.0 - np.sum(Cr * Cr) / den)


def _phi2_torch(U):
    import torch

    n, d = U.shape
    if d == 0:
        return U.new_zeros((n, 0))
    ii, jj = torch.triu_indices(d, d, device=U.device)
    Phi = U[:, ii] * U[:, jj]
    off = ii != jj
    if bool(off.any()):
        Phi = Phi.clone()
        Phi[:, off] *= np.sqrt(2.0)
    return Phi


def _predict_torch(U, J, B):
    return U @ J.T + _phi2_torch(U) @ B.T


def _remove_radial_torch(Z, x):
    nrm = torch.linalg.norm(x)
    if float(nrm) < EPS:
        return Z
    u = x / nrm
    return Z - (Z @ u)[:, None] * u[None, :]


def _closest_point_torch(
    Z: np.ndarray,
    J: np.ndarray,
    B: np.ndarray,
    u0: np.ndarray,
    *,
    u_max: float,
    max_iter: int,
    damp: float,
    x_anchor: np.ndarray | None,
    device,
) -> dict[str, Any]:
    import torch

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    Zt = torch.as_tensor(Z, dtype=torch.float64, device=dev)
    Jt = torch.as_tensor(J, dtype=torch.float64, device=dev)
    Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
    U = torch.as_tensor(u0, dtype=torch.float64, device=dev).clone()
    n, D = Zt.shape
    d = Jt.shape[1]
    xt = torch.as_tensor(x_anchor, dtype=torch.float64, device=dev) if x_anchor is not None else None
    H = torch.as_tensor(unpack_B_to_H(B, d), dtype=torch.float64, device=dev)
    Zfix = _predict_torch(U, Jt, Bt)
    if xt is not None:
        Zfix = _remove_radial_torch(Zfix, xt)
        Zt = _remove_radial_torch(Zt, xt)
    r2_fix = torch.sum((Zt - Zfix) ** 2, dim=1)
    best_U = U.clone()
    best_r2 = r2_fix.clone()
    n_iter = torch.zeros(n, dtype=torch.int16, device=dev)
    Ieps = damp * torch.eye(d, dtype=torch.float64, device=dev)
    for it in range(max_iter):
        Zhat = _predict_torch(U, Jt, Bt)
        if xt is not None:
            Zhat = _remove_radial_torch(Zhat, xt)
        R = Zt - Zhat
        Jf = Jt[None, :, :] + 2.0 * torch.einsum("iab,nb->nia", H, U)
        JTJ = torch.einsum("nia,nib->nab", Jf, Jf) + Ieps
        JTr = torch.einsum("nia,ni->na", Jf, R)
        try:
            delta = torch.linalg.solve(JTJ, JTr.unsqueeze(-1)).squeeze(-1)
        except Exception:  # noqa: BLE001
            delta = JTr
        U_try = U + delta
        nrm = torch.linalg.norm(U_try, dim=1, keepdim=True)
        hit = nrm[:, 0] > (u_max + 1e-12)
        scale = u_max / torch.clamp(nrm, min=EPS)
        U_try = torch.where(hit[:, None], U_try * scale, U_try)
        Ztry = _predict_torch(U_try, Jt, Bt)
        if xt is not None:
            Ztry = _remove_radial_torch(Ztry, xt)
        r2_try = torch.sum((Zt - Ztry) ** 2, dim=1)
        better = r2_try <= best_r2 + 1e-12
        U = torch.where(better[:, None], U_try, U)
        best_r2 = torch.where(better, r2_try, best_r2)
        best_U = torch.where(better[:, None], U_try, best_U)
        n_iter = torch.where(better, torch.tensor(it + 1, dtype=torch.int16, device=dev), n_iter)
        if (not bool(better.any())) and it > 1:
            break
    Zhat = _predict_torch(best_U, Jt, Bt)
    if xt is not None:
        Zhat = _remove_radial_torch(Zhat, xt)
    best_U_np = best_U.detach().cpu().numpy()
    Zhat_np = Zhat.detach().cpu().numpy()
    Zfix_np = Zfix.detach().cpu().numpy()
    Z_np = Zt.detach().cpu().numpy()
    best_r2_np = best_r2.detach().cpu().numpy()
    r2_fix_np = r2_fix.detach().cpu().numpy()
    n_iter_np = n_iter.detach().cpu().numpy()
    bfrac = np.linalg.norm(best_U_np, axis=1) >= u_max * 0.999
    return {
        "U": best_U_np,
        "Zhat": Zhat_np,
        "n_iter": n_iter_np,
        "boundary": bfrac,
        "improved": best_r2_np <= r2_fix_np + 1e-10,
        "fixed_nmse": nmse(Z_np, Zfix_np),
        "close_nmse": nmse(Z_np, Zhat_np),
        "fixed_r2": r2_from_nmse(nmse(Z_np, Zfix_np)),
        "close_r2": r2_from_nmse(nmse(Z_np, Zhat_np)),
        "mean_euclid": float(np.mean(np.sqrt(np.maximum(best_r2_np, 0.0)))),
        "median_euclid": float(np.median(np.sqrt(np.maximum(best_r2_np, 0.0)))),
    }


def closest_point_project(
    Z: np.ndarray,
    J: np.ndarray,
    B: np.ndarray,
    u0: np.ndarray,
    *,
    u_max: float,
    max_iter: int = 8,
    damp: float = 1e-4,
    x_anchor: np.ndarray | None = None,
    device=None,
) -> dict[str, Any]:
    """Gauss-Newton closest point. Monotone: never worse than |z - f(u0)| beyond tol.

    Constrains |u| <= u_max. Returns u_hat, z_hat, n_iter, boundary, improved.
    """
    n, D = Z.shape
    d = J.shape[1]
    U = np.asarray(u0, dtype=np.float64).copy()
    if B.size == 0 or d == 0:
        Zhat = U @ J.T
        return {
            "U": U,
            "Zhat": Zhat,
            "n_iter": np.zeros(n, dtype=np.int16),
            "boundary": np.zeros(n, dtype=bool),
            "improved": np.ones(n, dtype=bool),
            "fixed_nmse": nmse(Z, Zhat),
            "close_nmse": nmse(Z, Zhat),
            "fixed_r2": r2_from_nmse(nmse(Z, Zhat)),
            "close_r2": r2_from_nmse(nmse(Z, Zhat)),
            "mean_euclid": float(np.mean(np.linalg.norm(Z - Zhat, axis=1))),
            "median_euclid": float(np.median(np.linalg.norm(Z - Zhat, axis=1))),
        }
    use_torch = False
    if device is not None:
        import torch

        dev = torch.device(device) if not isinstance(device, torch.device) else device
        use_torch = dev.type == "cuda" and torch.cuda.is_available()
    if use_torch:
        return _closest_point_torch(
            Z, J, B, u0, u_max=u_max, max_iter=max_iter, damp=damp, x_anchor=x_anchor, device=device
        )
    H = unpack_B_to_H(B, d)
    Zfix = predict_f(U, J, B)
    if x_anchor is not None:
        Zfix = remove_radial_rows(Zfix, x_anchor)
        Z = remove_radial_rows(Z, x_anchor)
    r2_fix = np.sum((Z - Zfix) ** 2, axis=1)
    best_U = U.copy()
    best_r2 = r2_fix.copy()
    n_iter = np.zeros(n, dtype=np.int16)
    Ieps = damp * np.eye(d)
    for it in range(max_iter):
        Zhat = predict_f(U, J, B)
        if x_anchor is not None:
            Zhat = remove_radial_rows(Zhat, x_anchor)
        R = Z - Zhat
        Jf = jacobian_batch(U, J, H)
        JTJ = np.einsum("nia,nib->nab", Jf, Jf, optimize=True) + Ieps
        JTr = np.einsum("nia,ni->na", Jf, R, optimize=True)
        try:
            delta = np.linalg.solve(JTJ, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            delta = JTr
        U_try = U + delta
        nrm = np.linalg.norm(U_try, axis=1, keepdims=True)
        hit = nrm[:, 0] > u_max + 1e-12
        U_try = np.where(hit[:, None], U_try * (u_max / np.maximum(nrm, EPS)), U_try)
        Ztry = predict_f(U_try, J, B)
        if x_anchor is not None:
            Ztry = remove_radial_rows(Ztry, x_anchor)
        r2_try = np.sum((Z - Ztry) ** 2, axis=1)
        better = r2_try <= best_r2 + 1e-12
        U = np.where(better[:, None], U_try, U)
        best_r2 = np.where(better, r2_try, best_r2)
        best_U = np.where(better[:, None], U_try, best_U)
        n_iter = np.where(better, it + 1, n_iter)
        if not bool(np.any(better)) and it > 1:
            break
    Zhat = predict_f(best_U, J, B)
    if x_anchor is not None:
        Zhat = remove_radial_rows(Zhat, x_anchor)
    bfrac = np.linalg.norm(best_U, axis=1) >= u_max * 0.999
    return {
        "U": best_U,
        "Zhat": Zhat,
        "n_iter": n_iter,
        "boundary": bfrac,
        "improved": best_r2 <= r2_fix + 1e-10,
        "fixed_nmse": nmse(Z, Zfix),
        "close_nmse": nmse(Z, Zhat),
        "fixed_r2": r2_from_nmse(nmse(Z, Zfix)),
        "close_r2": r2_from_nmse(nmse(Z, Zhat)),
        "mean_euclid": float(np.mean(np.sqrt(np.maximum(best_r2, 0.0)))),
        "median_euclid": float(np.median(np.sqrt(np.maximum(best_r2, 0.0)))),
    }


def geodesic_error(x: np.ndarray, Z: np.ndarray, Zhat: np.ndarray) -> float:
    """Mean arccos(<exp_x(z), exp_x(zhat)>)."""
    from geometry.physics_stable_tangent_dimension.sphere_coords import sphere_exp_map

    th = []
    for i in range(min(len(Z), 512)):
        yi = sphere_exp_map(x, Z[i])
        yh = sphere_exp_map(x, Zhat[i])
        c = float(np.clip(np.dot(yi, yh), -1.0, 1.0))
        th.append(float(np.arccos(c)))
    return float(np.mean(th)) if th else float("nan")
