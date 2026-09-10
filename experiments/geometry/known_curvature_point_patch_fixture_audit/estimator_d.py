"""Estimator D: colleague PlainAutoEncoder, sphere-projected decode, full B^S."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import hessian, jacrev, vmap

_ROOT = Path(__file__).resolve().parents[3]
_NB = _ROOT / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))

from pu_manifold import cae  # noqa: E402
from pu_manifold.chart_curvature import VMAP_CHUNK, _assert_float64, _pad_to_chunk  # noqa: E402
from pu_manifold.decoder_curvature import assert_c2_decoder, plain_decoder_map  # noqa: E402

from .config import (
    AE_ACTIVATION,
    AE_HIDDEN,
    D_AMB,
    D_LAT,
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    TRAIN_CFG,
)
from .geometry import geometry_from_jets


class SphereProjectedDecoder(torch.nn.Module):
    """Identical wrap to run_ae_local_patch_scale_match.SphereProjectedDecoder."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.decoder = model.decoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        F = self.model.decode(z)
        return F / torch.linalg.norm(F, dim=-1, keepdim=True)


def split_indices(n: int, split_seed: int = SPLIT_SEED, holdout_fraction: float = HOLDOUT_FRACTION):
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_fraction))
    return perm[n_holdout:], perm[:n_holdout]


def train_decoder(
    X: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str = "cuda",
) -> tuple[SphereProjectedDecoder, dict]:
    torch.manual_seed(int(seed))
    model = cae.PlainAutoEncoder(D_AMB, D_LAT, hidden=AE_HIDDEN, activation=AE_ACTIVATION)
    cfg = dict(TRAIN_CFG)
    cfg["seed"] = int(seed)
    cfg["max_epochs"] = int(epochs)
    cfg["early_stop_patience"] = int(epochs) + 1
    X_train = torch.as_tensor(X[train_idx], dtype=torch.float32)
    info = cae.train_plain_ae(model, X_train, cfg)
    wrapped = SphereProjectedDecoder(model)
    if device.startswith("cuda") and torch.cuda.is_available():
        wrapped = wrapped.to(device)
    else:
        wrapped = wrapped.cpu()
    return wrapped, dict(info)


def reconstruction_stats(model: SphereProjectedDecoder, X: np.ndarray, idx: np.ndarray, device: str) -> dict:
    model.eval()
    xt = torch.as_tensor(X[idx], dtype=torch.float32, device=next(model.parameters()).device)
    with torch.no_grad():
        z = model.model.encode(xt)
        y = model.decode(z)
    ynp = y.detach().cpu().numpy()
    xnp = X[idx]
    resid = ynp - xnp
    mse = float(np.mean(np.sum(resid**2, axis=1)))
    var = float(np.mean(np.sum((xnp - xnp.mean(0)) ** 2, axis=1)))
    return {
        "mse": mse,
        "var": var,
        "r2": float(1.0 - mse / max(var, 1e-18)),
        "n": int(len(idx)),
    }


def decoder_full_BS(
    model: SphereProjectedDecoder, z: torch.Tensor
) -> dict[str, np.ndarray]:
    """Full sphere-normal second fundamental form of the normalized decoder.

    Does not call sealed plain_decoder_curvature (that path discards B^S after the
    trace). Uses the same vmap/hessian chunking as decoder_curvature.py.
    """
    assert_c2_decoder(model)
    _assert_float64(model, z)
    if z.ndim != 2:
        raise ValueError(f"z must be (batch, d); got {tuple(z.shape)}")
    decode_one = plain_decoder_map(model)
    batch, latent_dim = z.shape
    with torch.no_grad():
        probe = decode_one(z[0])
    out_dim = int(probe.shape[0])
    Gs, Js, Hs, Bs = [], [], [], []
    for start in range(0, batch, VMAP_CHUNK):
        real = z[start : start + VMAP_CHUNK]
        n_real = real.shape[0]
        chunk = _pad_to_chunk(real)
        J = vmap(jacrev(decode_one))(chunk)
        Hess = vmap(hessian(decode_one))(chunk)
        G = vmap(decode_one)(chunk)
        g = torch.einsum("boi,boj->bij", J, J)
        eye = torch.eye(latent_dim, dtype=g.dtype, device=g.device).expand_as(g)
        ginv = torch.linalg.solve(g, eye)
        PT = torch.einsum("bpi,bij,bqj->bpq", J, ginv, J)
        Gh = G / torch.clamp(G.norm(dim=-1, keepdim=True), min=1e-15)
        PNS = (
            torch.eye(out_dim, dtype=G.dtype, device=G.device).expand(VMAP_CHUNK, out_dim, out_dim)
            - torch.einsum("bi,bj->bij", Gh, Gh)
            - PT
        )
        B = torch.einsum("bij,bjkl->bikl", PNS, Hess)
        Gs.append(G[:n_real].detach())
        Js.append(J[:n_real].detach())
        Hs.append(Hess[:n_real].detach())
        Bs.append(B[:n_real].detach())
    Gcat = torch.cat(Gs).cpu().numpy().astype(np.float64)
    Jcat = torch.cat(Js).cpu().numpy().astype(np.float64)
    Hcat = torch.cat(Hs).cpu().numpy().astype(np.float64)
    Bcat = torch.cat(Bs).cpu().numpy().astype(np.float64)
    return {"G": Gcat, "J": Jcat, "Hess": Hcat, "B": Bcat}


def decoder_curvature_at_latents(model: SphereProjectedDecoder, z: np.ndarray) -> list[dict]:
    curv = copy.deepcopy(model).double()
    device = next(curv.parameters()).device
    zt = torch.as_tensor(z, dtype=torch.float64, device=device)
    raw = decoder_full_BS(curv, zt)
    out = []
    for i in range(len(z)):
        geo = geometry_from_jets(raw["G"][i], raw["J"][i], raw["Hess"][i])
        # B from geometry_from_jets uses P_NS @ Hess; should match raw B
        out.append(geo)
    return out


def encoder_latents(model: SphereProjectedDecoder, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.model.encode(xt)
    return z.detach().cpu().numpy().astype(np.float64)
