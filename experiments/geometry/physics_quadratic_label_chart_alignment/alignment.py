"""Hessian–curvature alignment A_B and fold stability."""

from __future__ import annotations

import numpy as np

from .config import PRIMARY_D
from .features import Gamma_from_gamma, n_quad, phi2_frob


def alignment_AB(gamma: np.ndarray, BS_frob: np.ndarray) -> float:
    """A_B = q * (γᵀ (BᵀB) γ) / (‖γ‖² tr(BᵀB))."""
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    B = np.asarray(BS_frob, dtype=np.float64)
    q = g.size
    ng2 = float(g @ g)
    BtB = B.T @ B
    tr = float(np.trace(BtB))
    if ng2 < 1e-18 or tr < 1e-18:
        return float("nan")
    return float(q * (g @ BtB @ g) / (ng2 * tr))


def fit_uq_gamma_oof(
    U: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    *,
    alpha_lin: float = 100.0,
    alpha_quad: float = 1000.0,
) -> tuple[np.ndarray, float]:
    """Single pooled UQ gamma from all folds' training (for diagnostics): mean of foldwise gammas."""
    from .models import _ridge_block, _scalar_rms, _design_UQ

    gammas = []
    for f in sorted(set(fold.tolist())):
        tr = fold != f
        if tr.sum() < 32:
            continue
        s = max(_scalar_rms(U[tr]), 1e-8)
        Xtr = _design_UQ(U[tr] / s)
        w, b, info = _ridge_block(Xtr, y[tr], n_lin=PRIMARY_D, alpha_lin=alpha_lin, alpha_quad=alpha_quad)
        if not info.get("ok"):
            continue
        # w = [w_lin (d), gamma (q)]; features from v=u/s; γ_u for (1/2)uᵀΓu = γ_v / s²
        gammas.append(w[PRIMARY_D:] / (s * s))
    if not gammas:
        return np.full(n_quad(PRIMARY_D), np.nan), float("nan")
    G = np.stack(gammas)
    # fold stability: median pairwise cosine
    cos = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            a, b = G[i], G[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                cos.append(float(a @ b / (na * nb)))
    stab = float(np.median(cos)) if cos else float("nan")
    return np.mean(G, axis=0), stab


def random_gamma_null(BS_frob: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = BS_frob.shape[1]
    out = np.empty(n)
    for i in range(n):
        g = rng.normal(size=q)
        g /= max(np.linalg.norm(g), 1e-12)
        out[i] = alignment_AB(g, BS_frob)
    return out
