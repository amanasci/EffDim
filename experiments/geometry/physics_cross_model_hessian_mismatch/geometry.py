"""Complete probe-induced Hessian via scalar autodiff. No ambient Hessian dumps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.func import hessian, jacrev

from geometry.physics_pointwise_residual_curvature_probe_relation.residual import NormDecode
from geometry.physics_task_aligned_curvature.algebra import (
    energy_g,
    metric_from_J,
    projectors,
    raw_normal_w,
    trace_g,
)
from geometry.physics_task_aligned_curvature.geometry_d import ScalarReadout

from .config import D_LAT


def jets_normalized(model, z: np.ndarray) -> dict[str, np.ndarray]:
    model.eval().double()
    decode = NormDecode(model)
    zt = torch.as_tensor(np.asarray(z, dtype=np.float64))
    x = decode(zt)
    J = jacrev(decode)(zt)
    x_np = x.detach().cpu().numpy()
    J_np = J.detach().cpu().numpy()
    proj = projectors(x_np, J_np)
    return {"x": x_np, "J": J_np, **proj, "z": np.asarray(z, dtype=np.float64)}


def Bw_at_z(model, z: np.ndarray, w: np.ndarray, proj: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """B_w = Hess_M(w^T F̃) = <w_N, II> with complete normal w_N = P_N w."""
    model.eval().double()
    decode = NormDecode(model)
    zt = torch.as_tensor(np.asarray(z, dtype=np.float64))
    if proj is None:
        x, J = decode(zt).detach().cpu().numpy(), jacrev(decode)(zt).detach().cpu().numpy()
        proj = projectors(x, J)
    wN = raw_normal_w(w, proj)
    b = hessian(ScalarReadout(decode, torch.as_tensor(wN, dtype=torch.float64)))(zt).detach().cpu().numpy()
    g, ginv = proj["g"], proj["ginv"]
    wx = float(np.dot(w, proj["xhat"]))
    bR = -wx * g
    bS = b - bR
    R = float(np.sqrt(max(energy_g(bR, ginv), 0.0)))
    S = float(np.sqrt(max(energy_g(bS, ginv), 0.0)))
    ident = abs(R - np.sqrt(D_LAT) * abs(wx))
    return {
        "Bw": b,
        "Bw_S": bS,
        "Bw_R": bR,
        "S": S,
        "R": R,
        "wN": wN,
        "wx": wx,
        "sphere_norm_err": ident,
        "sphere_norm_ok": bool(ident < 1e-5),
        "radial_tr": float(trace_g(bR, ginv)),
    }


def cos_g(A: np.ndarray, B: np.ndarray, ginv: np.ndarray) -> float:
    from geometry.physics_task_aligned_curvature.algebra import cross_energy_g

    na = float(np.sqrt(max(energy_g(A, ginv), 0.0)))
    nb = float(np.sqrt(max(energy_g(B, ginv), 0.0)))
    if na < 1e-15 or nb < 1e-15:
        return float("nan")
    return float(cross_energy_g(A, B, ginv) / (na * nb))


def mismatch_stats(Hy: np.ndarray, Bw: np.ndarray, BwS: np.ndarray, ginv: np.ndarray) -> dict[str, float]:
    dlt = Hy - Bw
    dlt_s = Hy - BwS
    return {
        "M_Delta": float(np.sqrt(max(energy_g(dlt, ginv), 0.0))),
        "M_shape": float(np.sqrt(max(energy_g(dlt_s, ginv), 0.0))),
        "A_full": cos_g(Hy, Bw, ginv),
        "A_shape": cos_g(Hy, BwS, ginv),
        "Hy_norm": float(np.sqrt(max(energy_g(Hy, ginv), 0.0))),
        "Bw_norm": float(np.sqrt(max(energy_g(Bw, ginv), 0.0))),
    }


def HS_norm(proj: dict[str, np.ndarray], Hess_S_trace_energy: float | None = None) -> float:
    return float("nan") if Hess_S_trace_energy is None else float(Hess_S_trace_energy)
