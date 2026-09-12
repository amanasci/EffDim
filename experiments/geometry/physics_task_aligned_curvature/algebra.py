"""Task-aligned tensor algebra. Named estimands only; no silent substitutions."""

from __future__ import annotations

import numpy as np


def metric_from_J(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = J.T @ J
    try:
        ginv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        ginv = np.linalg.pinv(g)
    return g, ginv


def projectors(x: np.ndarray, J: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    xhat = x / max(float(np.linalg.norm(x)), 1e-15)
    g, ginv = metric_from_J(J)
    PT = J @ ginv @ J.T
    PT = 0.5 * (PT + PT.T)
    eye = np.eye(x.shape[0], dtype=np.float64)
    PN = eye - PT
    PNS = eye - np.outer(xhat, xhat) - PT
    PNS = 0.5 * (PNS + PNS.T)
    return {"g": g, "ginv": ginv, "P_T": PT, "P_N": PN, "P_NS": PNS, "xhat": xhat}


def energy_g(b: np.ndarray, ginv: np.ndarray) -> float:
    """||b||_g^2 = g^{ac} g^{bd} b_ab b_cd. Does not clamp."""
    return float(np.einsum("ac,bd,ab,cd->", ginv, ginv, b, b))


def cross_energy_g(ba: np.ndarray, bb: np.ndarray, ginv: np.ndarray) -> float:
    """Signed <bA, bB>_g. Negative values are kept."""
    return float(np.einsum("ac,bd,ab,cd->", ginv, ginv, ba, bb))


def trace_g(b: np.ndarray, ginv: np.ndarray) -> float:
    return float(np.einsum("ab,ab->", ginv, b))


def contract_w_B(w: np.ndarray, B: np.ndarray) -> np.ndarray:
    """b_ab = <w, B_{ab}> for B shape (D, d, d)."""
    return np.einsum("i,iab->ab", w, B)


def sphere_normal_w(w: np.ndarray, proj: dict[str, np.ndarray]) -> np.ndarray:
    return proj["P_NS"] @ np.asarray(w, dtype=np.float64)


def raw_normal_w(w: np.ndarray, proj: dict[str, np.ndarray]) -> np.ndarray:
    return proj["P_N"] @ np.asarray(w, dtype=np.float64)


def radial_b(w: np.ndarray, xhat: np.ndarray, g: np.ndarray) -> np.ndarray:
    """b^R_ab = -<w, x> g_ab."""
    return -float(np.dot(w, xhat)) * g


def holm(ps: np.ndarray) -> np.ndarray:
    ps = np.asarray(ps, dtype=np.float64)
    m = len(ps)
    order = np.argsort(ps)
    out = np.empty(m, dtype=np.float64)
    prev = 0.0
    for rank, i in enumerate(order):
        prev = min(1.0, max(prev, (m - rank) * ps[i]))
        out[i] = prev
    return out
