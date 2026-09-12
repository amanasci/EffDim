"""Streamed pointwise sphere-residual H^S plus scalar/vector diagnostics. No Hessian dumps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.func import jacrev, vmap

from geometry.physics_pointwise_residual_curvature_probe_relation.residual import (
    NormDecode,
    residual_field,
)

from .config import D_LAT, HESSIAN_CHUNK
from .decoder import encode_decode, recon_row


def _jacobian_diag(model, z: torch.Tensor, chunk: int = HESSIAN_CHUNK) -> dict[str, np.ndarray]:
    decode = NormDecode(model)
    n, d = z.shape
    ranks, jconds, pns_j, pns_x, nres = [], [], [], [], []
    for start in range(0, n, chunk):
        zb = z[start : start + chunk]
        J = vmap(jacrev(decode))(zb)
        x = vmap(decode)(zb)
        xn = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-15)
        xhat = x / xn
        g = torch.einsum("boi,boj->bij", J, J)
        eye = torch.eye(d, dtype=g.dtype, device=g.device).expand(zb.shape[0], d, d)
        ginv = torch.linalg.solve(g, eye)
        # J: (B, D, d), ginv: (B, d, d), PT: (B, D, D)
        PT = torch.einsum("bai,bij,bcj->bac", J, ginv, J)
        PT = 0.5 * (PT + PT.transpose(-1, -2))
        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        xx = torch.einsum("bi,bj->bij", xhat, xhat)
        PNS = I - xx - PT
        PNS = 0.5 * (PNS + PNS.transpose(-1, -2))
        svals = torch.linalg.svdvals(J)
        rank = (svals > 1e-8).sum(dim=-1)
        jcond = (svals[:, 0] / svals[:, -1].clamp_min(1e-15))
        pns_j.append(torch.linalg.norm(torch.einsum("bik,bkj->bij", PNS, J), dim=(1, 2)).detach().cpu().numpy())
        pns_x.append(torch.linalg.norm(torch.einsum("bij,bj->bi", PNS, xhat), dim=-1).detach().cpu().numpy())
        nres.append((torch.linalg.norm(x, dim=-1) - 1.0).abs().detach().cpu().numpy())
        ranks.append(rank.detach().cpu().numpy())
        jconds.append(jcond.detach().cpu().numpy())
        del J, x, g, ginv, PT, PNS, svals
    return {
        "jac_rank": np.concatenate(ranks),
        "jac_cond": np.concatenate(jconds),
        "proj_PNS_J": np.concatenate(pns_j),
        "proj_PNS_x": np.concatenate(pns_x),
        "norm_residual": np.concatenate(nres),
    }


def eval_residual(
    model,
    X_anc: np.ndarray,
    *,
    hessian_device: str,
    chunk: int = HESSIAN_CHUNK,
) -> dict[str, np.ndarray]:
    z, y = encode_decode(model, X_anc)
    recon = recon_row(X_anc, y)
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    yhat = y / np.clip(yn, 1e-15, None)
    recon_unit = recon_row(X_anc, yhat)
    zt = torch.as_tensor(z, dtype=torch.float64)
    if hessian_device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(hessian_device)
        model = model.to(dev).double()
        zt = zt.to(dev)
    else:
        model = model.cpu().double()
        zt = zt.cpu()
    res = residual_field(model, zt, averaged=True, chunk=chunk)
    diag = _jacobian_diag(model, zt, chunk=chunk)
    model.cpu()
    H = np.asarray(res["H"], dtype=np.float64)
    finite = np.isfinite(H).all(axis=1) & np.isfinite(res["C_H"])
    return {
        "z": z,
        "y": y,
        "recon": recon,
        "recon_encoded": recon,
        "recon_unit": recon_unit,
        "H_S": H,
        "C_H": np.asarray(res["C_H"], dtype=np.float64),
        "C_H2": np.asarray(res["C_H2"], dtype=np.float64),
        "cond_g": np.asarray(res["cond_g"], dtype=np.float64),
        "xhat": np.asarray(res["xhat"], dtype=np.float64),
        "jac_rank": diag["jac_rank"],
        "jac_cond": diag["jac_cond"],
        "proj_PNS_J": diag["proj_PNS_J"],
        "proj_PNS_x": diag["proj_PNS_x"],
        "norm_residual": diag["norm_residual"],
        "finite": finite.astype(bool),
        "averaged": True,
        "convention": "H^S=(1/d) g^{ab} B^S_ab; C_H=||H^S||",
        "H_S_dtype": "float64_in_memory_float32_on_disk",
    }
