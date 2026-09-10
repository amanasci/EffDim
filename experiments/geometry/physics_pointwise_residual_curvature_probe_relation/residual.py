"""Sphere-residual decoder curvature. Differentiate through F̃ = F/||F||.

Primary: averaged H^S = (1/d) g^{ab} B^S_ab, C_H = ||H^S||.
Historical full (control): II^E = (I-P_T) D²F on raw decode, H^E = (1/d) tr_g II^E.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.func import hessian, jacrev, vmap

_NB = Path(__file__).resolve().parents[3] / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))

from pu_manifold.decoder_curvature import plain_decoder_curvature, plain_decoder_map

from .config import D_LAT, HASH_SALT, HESSIAN_CHUNK, N_TENSOR_SUBSET


class NormDecode:
    """Differentiate through output normalization. Never post-hoc-project a raw Hessian."""

    def __init__(self, model):
        self.inner = plain_decoder_map(model)

    def __call__(self, z):
        y = self.inner(z)
        return y / torch.linalg.norm(y)


def energy_g(B: np.ndarray, ginv: np.ndarray) -> float:
    Bw = np.einsum("ac,bd,iab->icd", ginv, ginv, B)
    return float(np.tensordot(B, Bw, axes=([0, 1, 2], [0, 1, 2])))


def metric_from_jacobian(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = J.T @ J
    try:
        ginv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        ginv = np.linalg.pinv(g)
    return g, ginv


def projectors_from_jets(x: np.ndarray, J: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    nrm = float(np.linalg.norm(x))
    xhat = x / max(nrm, 1e-15)
    g, ginv = metric_from_jacobian(J)
    PT = J @ ginv @ J.T
    PT = 0.5 * (PT + PT.T)
    PNS = np.eye(x.shape[0], dtype=np.float64) - np.outer(xhat, xhat) - PT
    PNS = 0.5 * (PNS + PNS.T)
    return {"g": g, "ginv": ginv, "P_T": PT, "P_NS": PNS, "xhat": xhat}


def apply_P(P: np.ndarray, Hess: np.ndarray) -> np.ndarray:
    D, d, _ = Hess.shape
    out = np.empty_like(Hess)
    for a in range(d):
        out[:, a, :] = P @ Hess[:, a, :]
    return out


def tensors_from_jets(x: np.ndarray, J: np.ndarray, Hess: np.ndarray, *, of_normalized: bool) -> dict[str, Any]:
    """Build II^E / B^S from explicit jets. H stored averaged (1/d) and unaveraged."""
    proj = projectors_from_jets(x, J)
    d = J.shape[1]
    eye = np.eye(x.shape[0], dtype=np.float64)
    PN = eye - proj["P_T"]
    II_E = apply_P(PN, Hess)
    B_S = apply_P(proj["P_NS"], Hess)
    ginv = proj["ginv"]
    H_E_un = np.einsum("ab,iab->i", ginv, II_E)
    H_S_un = np.einsum("ab,iab->i", ginv, B_S)
    eE, eS = energy_g(II_E, ginv), energy_g(B_S, ginv)
    tr_norm2 = float(np.dot(H_S_un, H_S_un))
    dscal = tr_norm2 - eS
    return {
        "x": np.asarray(x, dtype=np.float64),
        "J": np.asarray(J, dtype=np.float64),
        "Hess": np.asarray(Hess, dtype=np.float64),
        "g": proj["g"],
        "ginv": ginv,
        "P_T": proj["P_T"],
        "P_NS": proj["P_NS"],
        "xhat": proj["xhat"],
        "II_E": II_E,
        "B_S": B_S,
        "H_E_un": H_E_un,
        "H_S_un": H_S_un,
        "H_E": H_E_un / d,
        "H_S": H_S_un / d,
        "C_H": float(np.linalg.norm(H_S_un / d)),
        "C_H_un": float(np.linalg.norm(H_S_un)),
        "C_H2": float(np.dot(H_S_un / d, H_S_un / d)),
        "C_B2": eS,
        "K_dir_D": (2.0 * eS + tr_norm2) / float(d * (d + 2)),
        "delta_Scal_D": dscal,
        "Scal_D": float(d * (d - 1) + dscal),
        "energy_II_E": eE,
        "f_res": float(eS / eE) if eE > 1e-30 else float("nan"),
        "cond_g": float(np.linalg.cond(proj["g"])),
        "of_normalized": bool(of_normalized),
    }


def _pad(rows: torch.Tensor, chunk: int) -> torch.Tensor:
    missing = chunk - rows.shape[0]
    if missing <= 0:
        return rows
    return torch.cat([rows, rows[-1:].repeat(missing, 1)], dim=0)


def contracted_H(
    decode_one: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    *,
    sphere_residual: bool,
    averaged: bool = True,
    chunk: int = HESSIAN_CHUNK,
) -> dict[str, np.ndarray]:
    """Trace-first-then-project mean curvature. Does not keep 768×d×d Hessians."""
    if z.ndim != 2:
        raise ValueError(f"z must be (batch, d); got {tuple(z.shape)}")
    n, d = z.shape
    with torch.no_grad():
        x0 = decode_one(z[0])
    D = int(x0.shape[0])
    H_parts = []
    x_parts = []
    cond_parts = []
    jcond_ok = []
    for start in range(0, n, chunk):
        real = z[start : start + chunk]
        n_real = real.shape[0]
        pad = _pad(real, chunk)
        J = vmap(jacrev(decode_one))(pad)
        Hess = vmap(hessian(decode_one))(pad)
        x = vmap(decode_one)(pad)
        g = torch.einsum("boi,boj->bij", J, J)
        eye_d = torch.eye(d, dtype=g.dtype, device=g.device).expand(chunk, d, d)
        g_inv = torch.linalg.solve(g, eye_d)
        raw = torch.einsum("bjk,bojk->bo", g_inv, Hess)
        alpha = torch.linalg.solve(g, torch.einsum("boi,bo->bi", J, raw).unsqueeze(-1)).squeeze(-1)
        H = raw - torch.einsum("boi,bi->bo", J, alpha)
        if sphere_residual:
            xn = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-15)
            xhat = x / xn
            H = H - xhat * (xhat * H).sum(dim=-1, keepdim=True)
        if averaged:
            H = H / float(d)
        H_parts.append(H[:n_real].detach())
        x_parts.append(x[:n_real].detach())
        cond_parts.append(torch.linalg.cond(g)[:n_real].detach())
        del J, Hess, g, g_inv, raw, alpha, H, x, pad
    H_vec = torch.cat(H_parts, dim=0).cpu().numpy()
    X = torch.cat(x_parts, dim=0).cpu().numpy()
    cond = torch.cat(cond_parts, dim=0).cpu().numpy()
    xn = np.linalg.norm(X, axis=1, keepdims=True)
    xhat = X / np.clip(xn, 1e-15, None)
    return {
        "H": H_vec,
        "x": X,
        "xhat": xhat,
        "cond_g": cond,
        "C_H": np.linalg.norm(H_vec, axis=1),
        "C_H2": np.sum(H_vec * H_vec, axis=1),
        "shape": (n, D),
        "averaged": bool(averaged),
        "sphere_residual": bool(sphere_residual),
    }


def residual_field(model, z: torch.Tensor, *, averaged: bool = True, chunk: int = HESSIAN_CHUNK) -> dict[str, np.ndarray]:
    model.eval()
    decode = NormDecode(model)
    return contracted_H(decode, z, sphere_residual=True, averaged=averaged, chunk=chunk)


def full_field(model, z: torch.Tensor, *, averaged: bool = True) -> dict[str, np.ndarray]:
    """Historical raw-decode quantity via the sealed plain_decoder_curvature implementation."""
    model.eval()
    field = plain_decoder_curvature(model, z)
    H_un = field["H_vec"].detach().cpu().numpy()
    d = int(z.shape[1])
    H = H_un / d if averaged else H_un
    with torch.no_grad():
        x = model.decode(z).detach().cpu().numpy()
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    xhat = x / np.clip(xn, 1e-15, None)
    return {
        "H": H,
        "H_un": H_un,
        "x": x,
        "xhat": xhat,
        "cond_g": field["metric_condition_number"].detach().cpu().numpy(),
        "C_H": np.linalg.norm(H, axis=1),
        "C_H2": np.sum(H * H, axis=1),
        "averaged": bool(averaged),
        "sphere_residual": False,
        "curvature_convention": str(field["curvature_convention"]),
    }


def tensor_at_z(model, z1: torch.Tensor, *, normalized: bool) -> dict[str, Any]:
    decode = NormDecode(model) if normalized else plain_decoder_map(model)
    x = decode(z1)
    J = jacrev(decode)(z1)
    Hess = hessian(decode)(z1)
    return tensors_from_jets(
        x.detach().cpu().numpy(),
        J.detach().cpu().numpy(),
        Hess.detach().cpu().numpy(),
        of_normalized=normalized,
    )


def hash_order(sample_ids: np.ndarray, salt: bytes = HASH_SALT) -> np.ndarray:
    keys = []
    for s in np.asarray(sample_ids, dtype=np.int64):
        h = hashlib.sha256(salt + str(int(s)).encode()).digest()
        keys.append((h, int(s)))
    keys.sort()
    return np.asarray([s for _, s in keys], dtype=np.int64)


def hash_subset(sample_ids: np.ndarray, n: int = N_TENSOR_SUBSET, salt: bytes = HASH_SALT) -> np.ndarray:
    return hash_order(sample_ids, salt=salt)[: int(n)]


def radial_diagnostics(H_E: np.ndarray, H_S: np.ndarray, xhat: np.ndarray) -> dict[str, np.ndarray]:
    h_r = -xhat
    def _cos(a, b):
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        return np.sum(a * b, axis=1) / np.clip(na * nb, 1e-30, None)

    nE = np.linalg.norm(H_E, axis=1)
    nS = np.linalg.norm(H_S, axis=1)
    return {
        "cos_HE_HR": _cos(H_E, h_r),
        "cos_HS_HR": _cos(H_S, h_r),
        "f_res_vec": (nS**2) / np.clip(nE**2, 1e-30, None),
        "H_R": h_r,
    }
