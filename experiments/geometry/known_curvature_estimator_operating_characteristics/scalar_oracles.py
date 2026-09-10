"""Matched scalar oracles for production Q statistics K_H^cross and K_dir^cross."""

from __future__ import annotations

import numpy as np

from geometry.known_curvature_point_patch_fixture_audit.geometry import curvature_from_B, kdir_from_pair


def matched_q_scalars(B: np.ndarray, g: np.ndarray) -> dict[str, float]:
    """Scalar T2/T3 oracles using the same Hessian/whitening path as production Q.

    Production:
        K_H_cross = ⟨H_A, H_B⟩ with H = (1/d) tr(whiten(Hess))
        K_dir_cross from kdir_from_pair on Hessian tensors

    For a single matched tensor the corresponding truths are the self-cross:
        K_H^* = ||H||^2 = kdir_from_pair(B, B, g)["K_H_cross"]
        K_dir^* = kdir_from_pair(B, B, g)["K_dir_cross"]
    which agrees with curvature_from_B on orthonormal charts.
    """
    B = np.asarray(B, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    curv = curvature_from_B(B, g)
    pair = kdir_from_pair(B, B, g)
    return {
        "K_H_star": float(pair["K_H_cross"]),
        "K_dir_star": float(pair["K_dir_cross"]),
        "K_H2_curvature_from_B": float(curv["K_H2"]),
        "K_dir_curvature_from_B": float(curv["K_dir"]),
        "H_norm": float(curv["H_norm"]),
    }


def residualized_full_magnitude(h_e_norm: np.ndarray, d: int) -> np.ndarray:
    """sqrt(max(||H^E||^2 − d^2, 0)), the sphere-residual part of Euclidean mean curvature."""
    h = np.asarray(h_e_norm, dtype=np.float64)
    return np.sqrt(np.maximum(h * h - float(d) ** 2, 0.0))
