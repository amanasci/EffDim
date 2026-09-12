"""Decoder task-aligned curvature via scalar Hessian. Streams; no ambient Hessian dumps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.func import hessian, jacrev

from geometry.physics_pointwise_residual_curvature_probe_relation.residual import NormDecode

_NB = Path(__file__).resolve().parents[3] / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))
from pu_manifold.decoder_curvature import plain_decoder_map  # noqa: E402

from .algebra import energy_g, metric_from_J, projectors, radial_b, raw_normal_w, sphere_normal_w, trace_g
from .config import D_LAT


class ScalarReadout:
    """φ(z) = <w_fixed, F(z)>. w is frozen; not differentiated."""

    def __init__(self, decode_one: Callable, w: torch.Tensor):
        self.decode_one = decode_one
        self.w = w

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return (self.decode_one(z) * self.w).sum()


def _jets(decode_one, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = decode_one(z)
    J = jacrev(decode_one)(z)
    return x, J


def task_aligned_at_z(
    model,
    z: np.ndarray,
    w: np.ndarray,
    *,
    normalized: bool,
    sphere_residual: bool,
) -> dict[str, Any]:
    """One-anchor task-aligned tensor. Scalar Hessian only."""
    model.eval().double()
    decode = NormDecode(model) if normalized else plain_decoder_map(model)
    zt = torch.as_tensor(np.asarray(z, dtype=np.float64))
    x, J = _jets(decode, zt)
    x_np = x.detach().cpu().numpy()
    J_np = J.detach().cpu().numpy()
    proj = projectors(x_np, J_np)
    ww = np.asarray(w, dtype=np.float64)
    if sphere_residual:
        wN = sphere_normal_w(ww, proj)
        kind = "D_residual_sphere_normal"
    else:
        wN = raw_normal_w(ww, proj)
        kind = "D_raw_euclidean_normal"
    w_t = torch.as_tensor(wN, dtype=torch.float64)
    scalar = ScalarReadout(decode, w_t)
    b = hessian(scalar)(zt).detach().cpu().numpy()
    g, ginv = proj["g"], proj["ginv"]
    E = energy_g(b, ginv)
    T = trace_g(b, ginv)
    bR = radial_b(ww, proj["xhat"], g) if normalized else None
    trR = float(-D_LAT * np.dot(ww, proj["xhat"])) if normalized else float("nan")
    trR_num = trace_g(bR, ginv) if bR is not None else float("nan")
    return {
        "kind": kind,
        "b": b.astype(np.float64),
        "E": float(E),
        "T": float(T),
        "abs_T": float(abs(T)),
        "wN": wN,
        "cond_g": float(np.linalg.cond(g)),
        "xhat": proj["xhat"],
        "g": g,
        "ginv": ginv,
        "b_radial": bR,
        "tr_g_bR": trR_num,
        "tr_g_bR_identity": trR,
        "radial_identity_ok": bool(bR is None or abs(trR_num - trR) < 1e-6),
        "normalized": bool(normalized),
        "sphere_residual": bool(sphere_residual),
        "finite": bool(np.isfinite(E) and np.isfinite(b).all()),
    }


def eval_decoder_field(
    model,
    X_anc: np.ndarray,
    w: np.ndarray,
    *,
    normalized: bool,
    sphere_residual: bool,
) -> dict[str, np.ndarray]:
    encode = model.encode
    x = torch.as_tensor(np.asarray(X_anc, dtype=np.float64))
    with torch.no_grad():
        z = encode(x).cpu().numpy()
    n = len(z)
    E = np.full(n, np.nan)
    T = np.full(n, np.nan)
    cond = np.full(n, np.nan)
    ident = np.zeros(n, dtype=bool)
    finite = np.zeros(n, dtype=bool)
    b_stack = np.full((n, D_LAT, D_LAT), np.nan)
    for i in range(n):
        rec = task_aligned_at_z(model, z[i], w, normalized=normalized, sphere_residual=sphere_residual)
        E[i] = rec["E"]
        T[i] = rec["T"]
        cond[i] = rec["cond_g"]
        ident[i] = rec["radial_identity_ok"]
        finite[i] = rec["finite"]
        b_stack[i] = rec["b"]
    return {
        "E": E,
        "T": T,
        "abs_T": np.abs(T),
        "cond_g": cond,
        "radial_identity_ok": ident,
        "finite": finite,
        "b": b_stack,
        "z": z,
        "kind": "D_residual_sphere_normal" if sphere_residual else "D_raw_euclidean_normal",
    }
