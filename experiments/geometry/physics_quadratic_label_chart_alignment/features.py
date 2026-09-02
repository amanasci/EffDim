"""Frobenius-preserving quadratic features and B^S packing conversions.

Convention (preregistered):
  φ_aa = (1/2) u_a²
  φ_ab = u_a u_b / √2   (a < b)
  γ_aa = Γ_aa,  γ_ab = √2 Γ_ab
so  γᵀφ = (1/2) uᵀΓu  and  ‖γ‖₂² = ‖Γ‖_F².
"""

from __future__ import annotations

import numpy as np

from .config import PRIMARY_D


def n_quad(d: int) -> int:
    return d * (d + 1) // 2


def phi2_frob(U: np.ndarray) -> np.ndarray:
    """Frobenius-preserving degree-2 map. Shape (N, d(d+1)/2)."""
    U = np.asarray(U, dtype=np.float64)
    n, d = U.shape
    out = np.empty((n, n_quad(d)), dtype=np.float64)
    k = 0
    sqrt2 = np.sqrt(2.0)
    for a in range(d):
        out[:, k] = 0.5 * U[:, a] * U[:, a]
        k += 1
        for b in range(a + 1, d):
            out[:, k] = (U[:, a] * U[:, b]) / sqrt2
            k += 1
    return out


def gamma_from_Gamma(Gamma: np.ndarray) -> np.ndarray:
    """vech_√2 of symmetric Γ."""
    d = Gamma.shape[0]
    g = np.empty(n_quad(d), dtype=np.float64)
    k = 0
    sqrt2 = np.sqrt(2.0)
    for a in range(d):
        g[k] = Gamma[a, a]
        k += 1
        for b in range(a + 1, d):
            g[k] = sqrt2 * Gamma[a, b]
            k += 1
    return g


def Gamma_from_gamma(gamma: np.ndarray, d: int) -> np.ndarray:
    G = np.zeros((d, d), dtype=np.float64)
    k = 0
    sqrt2 = np.sqrt(2.0)
    for a in range(d):
        G[a, a] = gamma[k]
        k += 1
        for b in range(a + 1, d):
            G[a, b] = gamma[k] / sqrt2
            G[b, a] = G[a, b]
            k += 1
    return G


def bs_prod_to_frob(BS_prod: np.ndarray, d: int) -> np.ndarray:
    """Convert production NestedChart packing (φ=u_a u_b, off-diag stores 2B) to Frobenius columns.

    Production: ambient = BS_prod @ φ_prod with φ_prod_ab = u_a u_b.
    Frobenius: ambient = BS_frob @ φ_frob.
    """
    BS = np.asarray(BS_prod, dtype=np.float64)
    out = np.empty_like(BS)
    k = 0
    sqrt2 = np.sqrt(2.0)
    for a in range(d):
        # BS_f_aa * (1/2 u²) = BS_p_aa * u²  ⇒  BS_f_aa = 2 BS_p_aa
        out[:, k] = 2.0 * BS[:, k]
        k += 1
        for b in range(a + 1, d):
            # BS_f_ab * (uab/√2) = BS_p_ab * uab  ⇒  BS_f_ab = √2 BS_p_ab
            out[:, k] = sqrt2 * BS[:, k]
            k += 1
    return out


def verify_n_quad(d: int = PRIMARY_D) -> None:
    assert n_quad(d) == d * (d + 1) // 2
    if d == 16:
        assert n_quad(d) == 136
