"""D-full (raw decode, historical) and D-residual (differentiate through normalize)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import hessian, jacrev

_NB = Path(__file__).resolve().parents[3] / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))

from pu_manifold import cae
from pu_manifold.decoder_curvature import plain_decoder_curvature, plain_decoder_map

from .config import AE_ACTIVATION, AE_HIDDEN, D_LAT, TORCH_INIT_SEED, TRAIN_CFG
from .fixtures import targets_from_jets


def train_decoder(X: np.ndarray, *, epochs: int | None = None) -> tuple[cae.PlainAutoEncoder, dict, float]:
    import time

    D = X.shape[1]
    torch.manual_seed(int(TORCH_INIT_SEED))
    model = cae.PlainAutoEncoder(in_dim=D, latent_dim=D_LAT, hidden=AE_HIDDEN, activation=AE_ACTIVATION)
    cfg = dict(TRAIN_CFG)
    if epochs is not None:
        cfg["max_epochs"] = int(epochs)
        cfg["early_stop_patience"] = int(epochs) + 1
    x32 = torch.tensor(X, dtype=torch.float32)
    t0 = time.time()
    model.train().float()
    info = cae.train_plain_ae(model, x32, cfg)
    t_train = time.time() - t0
    model.eval().double()
    return model, dict(info), t_train


def var_explained(x: torch.Tensor, y: torch.Tensor) -> float:
    rec = cae.reconstruction_stats(x, y)
    sig = float((torch.linalg.norm(x, dim=1) ** 2).mean())
    return 1.0 - rec["mse_total"] / max(sig, 1e-18)


def r2_centered(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean(np.sum((x - y) ** 2, axis=1)))
    xc = x - x.mean(0)
    var = float(np.mean(np.sum(xc**2, axis=1)))
    return 1.0 - mse / max(var, 1e-18)


def _jets_at_z(decode_one, z1: torch.Tensor):
    x = decode_one(z1)
    J = jacrev(decode_one)(z1)
    Qh = hessian(decode_one)(z1)
    return x.detach().cpu().numpy(), J.detach().cpu().numpy(), Qh.detach().cpu().numpy()


def estimate_D_full(model, z: torch.Tensor) -> dict:
    """Historical: raw decode, H = tr_g(II^E), no 1/d."""
    field = plain_decoder_curvature(model, z)
    decode_one = plain_decoder_map(model)
    rows = []
    for i in range(z.shape[0]):
        x, J, Qh = _jets_at_z(decode_one, z[i])
        rows.append(targets_from_jets(x, J, Qh))
    return {
        "H_vec_hist": field["H_vec"].detach().cpu().numpy(),
        "cond_g": field["metric_condition_number"].detach().cpu().numpy(),
        "rows": rows,
    }


class _NormDecode:
    def __init__(self, model):
        self.inner = plain_decoder_map(model)

    def __call__(self, z):
        y = self.inner(z)
        return y / torch.linalg.norm(y)


def estimate_D_residual(model, z: torch.Tensor) -> dict:
    """Differentiate through F̃ = F/||F||. Do not project an already-computed Hessian."""
    decode_one = _NormDecode(model)
    rows = []
    for i in range(z.shape[0]):
        x, J, Qh = _jets_at_z(decode_one, z[i])
        rows.append(targets_from_jets(x, J, Qh))
    return {"rows": rows}


def encode(model, X: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        return model.encode(x)
