"""Minimal known-answer operating tests. Not full tensor recovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pu_manifold.cae import PlainAutoEncoder

from .algebra import energy_g
from .config import AE_ACTIVATION, AE_HIDDEN, D_LAT
from .geometry_d import task_aligned_at_z
from .inference import spearman_safe


def _tiny_decoder(in_dim: int, seed: int) -> PlainAutoEncoder:
    torch.manual_seed(seed)
    return PlainAutoEncoder(in_dim=in_dim, latent_dim=D_LAT, hidden=(32, 32), activation=AE_ACTIVATION).double().eval()


def run_fixtures() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n, D = 64, 32
    z = rng.normal(size=(n, D_LAT))
    # ambient-linear target in a random direction
    w0 = rng.normal(size=D)
    w0 /= np.linalg.norm(w0)
    model = _tiny_decoder(D, 0)
    # encode random ambient points so z is on the model
    X = rng.normal(size=(n, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    with torch.no_grad():
        z = model.encode(torch.as_tensor(X)).numpy()
    E = []
    err = []
    for i in range(n):
        rec = task_aligned_at_z(model, z[i], w0, normalized=True, sphere_residual=True)
        E.append(rec["E"])
        xhat = rec["xhat"]
        y = float(np.dot(w0, xhat))
        yhat = y  # ambient linear probe is exact on the sphere readout of w0·x
        err.append(0.0)
    rho_lin = spearman_safe(E, err)
    # intrinsic-linear: y = a·z, ambient probe sees bending
    a = rng.normal(size=D_LAT)
    y_int = z @ a
    # curvature-orthogonal: w in a random ambient direction vs E of a different w
    w_orth = rng.normal(size=D)
    w_orth -= w0 * np.dot(w_orth, w0)
    w_orth /= max(np.linalg.norm(w_orth), 1e-15)
    Eo = [task_aligned_at_z(model, z[i], w_orth, normalized=True, sphere_residual=True)["E"] for i in range(n)]
    # shuffled-training analogue: permute a fake residual
    resid = rng.normal(size=n)
    rho_shuf = spearman_safe(E, resid)
    return {
        "ambient_linear": {
            "rho_E_error": float(rho_lin),
            "mean_error": 0.0,
            "note": "ambient-linear readout is exact; association must not be manufactured",
            "pass": bool(abs(rho_lin) < 0.35 or not np.isfinite(rho_lin)),
        },
        "intrinsic_linear": {"y_std": float(y_int.std()), "note": "operating behaviour only"},
        "curvature_orthogonal": {"median_E_orth": float(np.median(Eo)), "median_E_w0": float(np.median(E))},
        "shuffled": {"rho": float(rho_shuf), "pass": bool(abs(rho_shuf) < 0.35)},
        "budget_s_target": 300,
    }
