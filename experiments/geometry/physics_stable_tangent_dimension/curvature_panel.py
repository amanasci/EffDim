"""Complementary local-linearity and curvature metrics, evaluated after d_T freeze.

Mean curvature is diagnostic anatomy, not the primary measure.
K_dir^2 = K_H^2 + K_aniso^2 with aniso pref 2/(d(d+2)).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric
from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    aniso_prefactor,
    decompose_tensors,
    metric_scalars,
    monte_carlo_K_dir2,
)

from .sphere_coords import EPS, angular_radii, parallel_transport_basis_yx
from .nested_pca import reconstruction_risk


def D_lin(Z_test: np.ndarray, J_train: np.ndarray, d: int) -> float:
    """Held-out linear distortion: E||(I-P)z||^2 / E||z||^2."""
    tot = float(np.mean(np.sum(Z_test * Z_test, axis=1)))
    if tot < EPS:
        return float("nan")
    return reconstruction_risk(Z_test, J_train, d) / tot


def flatten_sym2(B: np.ndarray) -> np.ndarray:
    """B (D,d,d) symmetric → (D, q) with √2 off-diagonal so ||flat||_F = ||B||_F."""
    D, d, _ = B.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            if a == b:
                cols.append(B[:, a, a])
            else:
                cols.append(np.sqrt(2.0) * B[:, a, b])
    return np.stack(cols, axis=1)


def curvature_spectrum(BS_flat: np.ndarray, d: int) -> dict[str, float | np.ndarray]:
    B = unpack_BS_symmetric(BS_flat, d)
    M = flatten_sym2(B)
    s = np.linalg.svd(M, compute_uv=False)
    tot = float(np.sum(s**2))
    p = (s**2) / max(tot, EPS)
    csum = np.cumsum(p)
    r90 = int(np.searchsorted(csum, 0.90) + 1) if tot > 0 else 0
    r95 = int(np.searchsorted(csum, 0.95) + 1) if tot > 0 else 0
    stable = float((p.sum() ** 2) / max(np.sum(p**2), EPS))
    ent = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))) if np.any(p > 0) else 0.0
    return {
        "singular_values": s,
        "r90": r90,
        "r95": r95,
        "stable_rank": stable,
        "entropy_rank": ent,
        "fro2": tot,
    }


def k_max_directional(
    BS_flat: np.ndarray,
    d: int,
    *,
    n_starts: int = 12,
    n_mc: int = 2000,
    n_iter: int = 40,
    seed: int = 0,
) -> dict[str, float]:
    """max_{|v|=1} |B^S(v,v)| via multi-start sphere gradient + MC lower bound."""
    B = unpack_BS_symmetric(BS_flat, d)
    rng = np.random.default_rng(seed)

    def Bvv(v: np.ndarray) -> np.ndarray:
        return np.einsum("dab,a,b->d", B, v, v)

    def objective(v: np.ndarray) -> float:
        w = Bvv(v)
        return float(np.linalg.norm(w))

    def step(v: np.ndarray, lr: float = 0.15) -> np.ndarray:
        w = Bvv(v)
        # d|w|/dv ∝ sum_i (w_i / |w|) * 2 B_i v
        nw = max(float(np.linalg.norm(w)), EPS)
        g = np.einsum("d,dab,b->a", w / nw, B, v) * 2.0
        g = g - np.dot(g, v) * v
        v = v + lr * g
        n = float(np.linalg.norm(v))
        return v / max(n, EPS)

    starts = rng.normal(size=(n_starts, d))
    starts /= np.linalg.norm(starts, axis=1, keepdims=True)
    best = 0.0
    conv = []
    for s in range(n_starts):
        v = starts[s]
        hist = []
        for _ in range(n_iter):
            v = step(v)
            hist.append(objective(v))
        val = hist[-1]
        conv.append(abs(hist[-1] - hist[max(0, len(hist) - 5)]) / max(hist[-1], EPS))
        best = max(best, val)
    Vmc = rng.normal(size=(n_mc, d))
    Vmc /= np.linalg.norm(Vmc, axis=1, keepdims=True)
    mc = np.array([objective(v) for v in Vmc])
    mc_lb = float(np.max(mc))
    return {
        "K_max": float(max(best, mc_lb)),
        "K_max_opt": float(best),
        "K_max_mc": mc_lb,
        "converged": bool(np.median(conv) < 0.05),
        "median_rel_change": float(np.median(conv)),
    }


def excess_sectional(BS_flat: np.ndarray, d: int) -> dict[str, float]:
    """Gauss excess K_ab = <B_aa, B_bb> - |B_ab|^2. Ambient +1 kept separate."""
    B = unpack_BS_symmetric(BS_flat, d)
    vals = []
    for a in range(d):
        for b in range(a + 1, d):
            vals.append(float(np.dot(B[:, a, a], B[:, b, b]) - np.sum(B[:, a, b] ** 2)))
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {
            "mean_excess": float("nan"),
            "rms_excess": float("nan"),
            "mean_abs_excess": float("nan"),
        }
    return {
        "mean_excess": float(np.mean(vals)),
        "rms_excess": float(np.sqrt(np.mean(vals**2))),
        "mean_abs_excess": float(np.mean(np.abs(vals))),
        "ambient_baseline": 1.0,
    }


def tangent_rotation_stat(
    x: np.ndarray,
    Y: np.ndarray,
    Jx: np.ndarray,
    neighbour_J: list[np.ndarray],
) -> float:
    """K_rot^2 = E_j ||P_i - P_{PT(T_j)}||_F^2 / (2 θ_ij^2)."""
    th = angular_radii(x, Y)
    Px_coords = Jx  # (D,d)
    acc = []
    for j, Jy in enumerate(neighbour_J):
        if Jy is None or th[j] < 1e-8:
            continue
        Jpt = parallel_transport_basis_yx(Y[j], x, Jy)
        # ||Pa - Pb||_F^2 = 2d - 2||Ja^T Jb||_F^2
        d = min(Jx.shape[1], Jpt.shape[1])
        M = Jx[:, :d].T @ Jpt[:, :d]
        df2 = 2.0 * d - 2.0 * float(np.sum(M * M))
        acc.append(df2 / (2.0 * th[j] ** 2))
    return float(np.mean(acc)) if acc else float("nan")


def verify_kdir_identity(BS_flat: np.ndarray, d: int, *, seed: int = 0) -> dict[str, float]:
    s = metric_scalars(BS_flat, d)
    rec = s["K_H2"] + s["K_aniso2"]
    mc = monte_carlo_K_dir2(BS_flat, d, n_dir=3000, seed=seed)
    return {
        "identity_err": abs(s["K_dir2"] - rec) / max(s["K_dir2"], EPS),
        "mc_rel_err": abs(mc - s["K_dir2"]) / max(s["K_dir2"], EPS),
        "K_dir2": s["K_dir2"],
        "mc": mc,
    }


def pack_from_B(B: np.ndarray) -> np.ndarray:
    D, d, _ = B.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(B[:, a, a] if a == b else (2.0 * B[:, a, b]))
    return np.stack(cols, axis=1)
