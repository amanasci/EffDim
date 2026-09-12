"""Finite-patch quadratic charts and split-half statistics.

Frobenius-preserving convention matches QLCA features.py.
"""

from __future__ import annotations

import numpy as np

from .metric import cross_energy_g, energy_g


def n_quad(d: int) -> int:
    return d * (d + 1) // 2


def phi2_frob(U: np.ndarray) -> np.ndarray:
    """φ_aa = (1/2) u_a²; φ_ab = u_a u_b / √2 (a < b)."""
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
    """vech_√2 of symmetric Γ so ‖γ‖₂² = ‖Γ‖_F²."""
    d = int(Gamma.shape[0])
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


def decompose_Q(Q: np.ndarray, x0: np.ndarray, J: np.ndarray, g: np.ndarray) -> dict[str, np.ndarray]:
    """Q shape (D, d, d). Returns QT, QR, BS."""
    from .metric import projectors

    proj = projectors(x0, J)
    QT = np.einsum("ij,jab->iab", proj["P_T"], Q)
    QR = -np.einsum("i,ab->iab", proj["xhat"], g)
    BS = np.einsum("ij,jab->iab", proj["P_NS"], Q)
    return {"Q_T": QT, "Q_R": QR, "B_S": BS, "xhat": proj["xhat"]}


def quadratic_sphere_residual_second_fundamental_form(
    Q: np.ndarray, x0: np.ndarray, J: np.ndarray, g: np.ndarray
) -> np.ndarray:
    """Named B^S extractor. Not interchangeable with decoder II^S."""
    return decompose_Q(Q, x0, J, g)["B_S"]


def kh_cross(HA: np.ndarray, HB: np.ndarray) -> float:
    """Signed <H_A, H_B>. Not clamped."""
    return float(np.dot(HA, HB))


def kdir_cross(BA: np.ndarray, BB: np.ndarray, *, d: int) -> float:
    """(2 <BA,BB>_F + <tr BA, tr BB>) / (d(d+2))."""
    fro = float(np.tensordot(BA, BB, axes=3))
    tr = float(np.dot(np.trace(BA, axis1=1, axis2=2), np.trace(BB, axis1=1, axis2=2)))
    return (2.0 * fro + tr) / float(d * (d + 2))


def split_half_cross_energy(bA: np.ndarray, bB: np.ndarray, ginv: np.ndarray) -> float:
    """Signed <bA, bB>_g. Never average-then-square. Never clamp."""
    return cross_energy_g(bA, bB, ginv)


def assert_not_clamped(value: float) -> float:
    return float(value)


def mean_curvature_vector_averaged(B: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """H^S = (1/d) g^{ab} B_ab. Averaged convention used by some Q tables."""
    d = ginv.shape[0]
    return np.einsum("ab,iab->i", ginv, B) / float(d)


def mean_curvature_vector_unaveraged(B: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """H = g^{ab} B_ab. Unaveraged decoder-style trace."""
    return np.einsum("ab,iab->i", ginv, B)
