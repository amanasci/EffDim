"""Seed-aware D=28 trainer. Does not edit the dual-estimator package."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_NB = Path(__file__).resolve().parents[3] / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))

from pu_manifold import cae

from geometry.known_curvature_dual_estimator_robustness.config import AE_ACTIVATION, AE_HIDDEN, D_LAT, TRAIN_CFG
from geometry.known_curvature_dual_estimator_robustness.decoder import (  # noqa: F401
    encode,
    estimate_D_full,
    estimate_D_residual,
    r2_centered,
    var_explained,
)


def train_decoder_seeded(
    X: np.ndarray,
    seed: int,
    *,
    device: str = "cpu",
) -> tuple[cae.PlainAutoEncoder, dict, float]:
    """Frozen architecture/epochs; only the init+shuffle seed changes."""
    D = X.shape[1]
    torch.manual_seed(int(seed))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = cae.PlainAutoEncoder(in_dim=D, latent_dim=D_LAT, hidden=AE_HIDDEN, activation=AE_ACTIVATION)
    cfg = dict(TRAIN_CFG)
    cfg["seed"] = int(seed)
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    x32 = torch.tensor(X, dtype=torch.float32, device=dev)
    model = model.to(dev)
    t0 = time.time()
    model.train().float()
    info = cae.train_plain_ae(model, x32, cfg)
    t_train = time.time() - t0
    model.eval().double()
    model = model.to("cpu")
    return model, dict(info), t_train
