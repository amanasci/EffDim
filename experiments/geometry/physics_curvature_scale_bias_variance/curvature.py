"""Frozen quadratic chart on a hash-selected neighbour subset."""

from __future__ import annotations

import numpy as np
import torch

from geometry.physics_activation_atlas.effdim_curvature_metrics import cross_metric_pair, decompose_tensors
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad
from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices

from .hashing import _digest, fit_val_seed, select_m, split_ab


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_cell(
    Xloc_pool: np.ndarray,
    *,
    R: int,
    m: int,
    d: int,
    seed: int,
    sample_id: int,
    device: str = "cuda",
    d_max: int = 20,
) -> dict:
    """PCA + split-half quadratic on m neighbours drawn from the first R.

    Returns the cross-split sphere-normal mean-curvature statistic K_H_cross
    = <H_A, H_B>, not the mean-curvature vector itself.
    """
    pool = np.arange(int(R), dtype=np.int64)
    chosen = select_m(pool, int(m), seed=seed, sample_id=sample_id)
    Xloc = np.asarray(Xloc_pool[chosen], dtype=np.float64)
    A, B = split_ab(np.arange(len(chosen), dtype=np.int64), seed=seed, sample_id=sample_id)
    select_hash = _digest(f"select:{int(seed)}:{int(sample_id)}:{int(R)}:{int(m)}")
    split_hash = _digest(f"split:{int(seed)}:{int(sample_id)}:{int(m)}")
    x0, J, _ev, _diag = nested_pca_frame(Xloc, int(d_max), _device(device))
    if J.shape[1] < int(d):
        return {
            "ok": False,
            "reason": "rank_short",
            "K_H_cross": float("nan"),
            "select_hash": select_hash,
            "split_hash": split_hash,
        }
    Jd = J[:, : int(d)]
    fA, vA = _half_fit_indices(A, fit_val_seed(seed=seed, sample_id=sample_id, tag="A"))
    fB, vB = _half_fit_indices(B, fit_val_seed(seed=seed, sample_id=sample_id, tag="B"))
    chA, _, infoA = fit_quad(Xloc, x0, Jd, fA, vA, B, ridges=RIDGES, device=_device(device))
    chB, _, infoB = fit_quad(Xloc, x0, Jd, fB, vB, A, ridges=RIDGES, device=_device(device))
    if chA is None or chB is None:
        return {
            "ok": False,
            "reason": "fit_fail",
            "K_H_cross": float("nan"),
            "select_hash": select_hash,
            "split_hash": split_hash,
        }
    cross = cross_metric_pair(chA.BS_flat, chB.BS_flat, int(d))
    HA = np.asarray(decompose_tensors(chA.BS_flat, int(d))["H"], dtype=float)
    HB = np.asarray(decompose_tensors(chB.BS_flat, int(d))["H"], dtype=float)
    resid = _radial_quantile_residual(Xloc, x0, Jd, chosen)
    return {
        "ok": True,
        "K_H_cross": float(cross["K_H_cross"]),
        "R_H": float(cross["R_H"]),
        "H": 0.5 * (HA + HB),
        "n_fit": int(len(chosen)),
        "n_A": int(len(A)),
        "n_B": int(len(B)),
        "select_hash": select_hash,
        "split_hash": split_hash,
        "outer_resid": resid,
        "dS": 0.5 * (float(infoA.get("dS", np.nan)) + float(infoB.get("dS", np.nan))),
    }


def _radial_quantile_residual(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray, _idx: np.ndarray) -> dict[str, float]:
    """Held-out linear residual energy by distance quantile (diagnostic)."""
    U = (Xloc - x0) @ J
    recon = x0 + U @ J.T
    err = np.sum((Xloc - recon) ** 2, axis=1)
    dist = np.linalg.norm(Xloc - x0, axis=1)
    qs = np.quantile(dist, [0.33, 0.67])
    inner = err[dist <= qs[0]]
    outer = err[dist > qs[1]]
    return {
        "inner_mse": float(np.mean(inner)) if len(inner) else float("nan"),
        "outer_mse": float(np.mean(outer)) if len(outer) else float("nan"),
        "outer_inner_ratio": float(np.mean(outer) / max(np.mean(inner), 1e-18)) if len(inner) and len(outer) else float("nan"),
    }
