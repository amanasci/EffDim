"""Decoder second-fundamental-form helpers. Torch is imported lazily."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .metric import projectors


def decoder_full_euclidean_second_fundamental_form(Hess: np.ndarray, P_N: np.ndarray) -> np.ndarray:
    """II^E = (I-P_T) D²F = P_N Hess. Raw decoder image."""
    return np.einsum("ij,jab->iab", P_N, Hess)


def decoder_normalized_full_second_fundamental_form(Hess_tilde: np.ndarray, P_N: np.ndarray) -> np.ndarray:
    """Full Euclidean II of F̃=F/‖F‖. Not the sphere-residual form."""
    return decoder_full_euclidean_second_fundamental_form(Hess_tilde, P_N)


def decoder_sphere_residual_second_fundamental_form(Hess: np.ndarray, P_NS: np.ndarray) -> np.ndarray:
    """II^S = (I - xx^T - P_T) D²F̃."""
    return np.einsum("ij,jab->iab", P_NS, Hess)


def stream_contract_probe(w: np.ndarray, Hess: np.ndarray) -> np.ndarray:
    """b_ab = <w, Hess_ab> without materialising extra (D,d,d) copies."""
    return np.einsum("i,iab->ab", w, Hess)


def christoffel_second_fundamental_form(Hess: np.ndarray, J: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """II = Hess - J Γ with Γ = g^{-1} J^T Hess."""
    Gamma = np.einsum("pq,ip,iab->qab", ginv, J, Hess)
    return Hess - np.einsum("iq,qab->iab", J, Gamma)


def sphere_radial_second_fundamental_form(xhat: np.ndarray, g: np.ndarray) -> np.ndarray:
    """II^R = -x ⊗ g."""
    return -np.einsum("i,ab->iab", xhat, g)


def unaveraged_mean_curvature(II: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """H = g^{ab} II_ab (no 1/d)."""
    return np.einsum("ab,iab->i", ginv, II)


def he_equals_hs_minus_d_x(HE: np.ndarray, HS: np.ndarray, xhat: np.ndarray, d: int) -> bool:
    return bool(np.allclose(HE, HS - d * xhat, atol=1e-6))


def _torch():
    import torch
    from torch.func import hessian, jacrev

    return torch, hessian, jacrev


def jets_of(decode: Callable, z: np.ndarray) -> dict[str, Any]:
    torch, _, jacrev = _torch()
    zt = torch.as_tensor(np.asarray(z, dtype=np.float64))
    x = decode(zt)
    J = jacrev(decode)(zt)
    return {
        "x": x.detach().cpu().numpy(),
        "J": J.detach().cpu().numpy(),
        **projectors(x.detach().cpu().numpy(), J.detach().cpu().numpy()),
    }
