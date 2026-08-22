"""Matched label-free nulls for stability, held-out gain, and eigengaps.

Cache sufficient statistics only. Residual isotropic nulls randomize residual
directions while preserving residual-norm distribution, train/test sizes, and
ambient residual dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .nested_pca import (
    block_agreement,
    crossfit_risk,
    eigengaps,
    incremental_gain,
    nested_uncentred_svd,
    prefix_agreement,
    reconstruction_risk,
)
from .sphere_coords import EPS


def residual_isotropic_null(
    Z: np.ndarray,
    J_prefix: np.ndarray,
    *,
    rng: np.random.Generator,
    n_draw: int = 32,
    d_extra: int = 1,
    device=None,
) -> dict[str, np.ndarray]:
    """After removing prefix P, Haar-randomize residual directions then re-PCA.

    Preserves ||(I-P)z|| per row. Returns null incremental agreement and gain
    for adding `d_extra` residual PCs — not prefix agreement, which is dominated
    by the shared lower-rank span.
    """
    n, D = Z.shape
    d0 = int(J_prefix.shape[1]) if J_prefix.size else 0
    if d0 > 0:
        U = Z @ J_prefix
        Z_res = Z - U @ J_prefix.T
        Z_pref = U @ J_prefix.T
    else:
        Z_res = Z
        Z_pref = np.zeros_like(Z)
    norms = np.linalg.norm(Z_res, axis=1, keepdims=True)
    d_tgt = d0 + d_extra
    agree_inc = []
    agree_pref = []
    gains = []
    gaps = []
    R0 = float(np.mean(np.sum(Z * Z, axis=1)))
    R_prev = reconstruction_risk(Z, J_prefix, d0) if d0 else R0

    def iso_cloud() -> np.ndarray:
        G = rng.normal(size=(n, D))
        if d0 > 0:
            G = G - (G @ J_prefix) @ J_prefix.T
        G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), EPS) * norms
        return Z_pref + G

    for _ in range(n_draw):
        ZA = iso_cloud()
        ZB = iso_cloud()
        JA, evA = nested_uncentred_svd(ZA, d_tgt, device=device)
        JB, evB = nested_uncentred_svd(ZB, d_tgt, device=device)
        agree_pref.append(prefix_agreement(JA, JB, d_tgt))
        if d0 == 0:
            agree_inc.append(prefix_agreement(JA, JB, 1) if d_extra == 1 else prefix_agreement(JA, JB, d_tgt))
        else:
            # overlap of added columns of A with span of B at rank d_tgt
            if JA.shape[1] >= d_tgt and JB.shape[1] >= d_tgt:
                Uadd = JA[:, d0:d_tgt]
                Pb = JB[:, :d_tgt]
                ov = Uadd.T @ Pb @ Pb.T @ Uadd
                agree_inc.append(float(np.trace(ov) / max(d_extra, 1)))
            else:
                agree_inc.append(np.nan)
        R_d = reconstruction_risk(Z, JA, min(d_tgt, JA.shape[1]))
        gains.append(incremental_gain(R_prev, R_d, R0))
        if len(evA) > d0:
            g = eigengaps(evA)
            gaps.append(g[d0 - 1] if d0 > 0 else g[0] if len(g) else np.nan)
        else:
            gaps.append(np.nan)
    return {
        "agreement": np.asarray(agree_pref, dtype=np.float64),
        "agreement_inc": np.asarray(agree_inc, dtype=np.float64),
        "gain": np.asarray(gains, dtype=np.float64),
        "eigengap": np.asarray(gaps, dtype=np.float64),
    }


def column_permutation_null(
    ZA: np.ndarray,
    ZB: np.ndarray,
    d_max: int,
    *,
    rng: np.random.Generator,
    n_draw: int = 24,
    device=None,
) -> dict[str, np.ndarray]:
    """Permute feature axes independently on each half: destroy shared covariance."""
    nA, D = ZA.shape
    agrees = np.zeros((n_draw, d_max))
    gains = np.zeros((n_draw, d_max))
    R0 = 0.5 * (
        float(np.mean(np.sum(ZA * ZA, axis=1))) + float(np.mean(np.sum(ZB * ZB, axis=1)))
    )
    for t in range(n_draw):
        permA = rng.permutation(D)
        permB = rng.permutation(D)
        JA, _ = nested_uncentred_svd(ZA[:, permA], d_max, device=device)
        JB, _ = nested_uncentred_svd(ZB[:, permB], d_max, device=device)
        R_prev = R0
        for d in range(1, d_max + 1):
            agrees[t, d - 1] = prefix_agreement(JA, JB, d)
            Rd = crossfit_risk(ZA[:, permA], ZB[:, permB], JA, JB, d)
            gains[t, d - 1] = incremental_gain(R_prev, Rd, R0)
            R_prev = Rd
    return {"agreement": agrees, "gain": gains}


def split_schedule_null(
    Z: np.ndarray,
    radii: np.ndarray,
    d_max: int,
    *,
    rng: np.random.Generator,
    n_draw: int = 16,
    device=None,
    split_fn=None,
) -> dict[str, np.ndarray]:
    """Agreement induced by overlapping neighbourhood structure alone.

    Random radially stratified re-splits of the *same* neighbourhood.
    High values are expected; used as a floor, not a discovery threshold.
    """
    from .nested_pca import radial_stratified_halves

    split_fn = split_fn or radial_stratified_halves
    agrees = np.zeros((n_draw, d_max))
    for t in range(n_draw):
        A, B = split_fn(radii, int(rng.integers(0, 2**31 - 1)))
        if min(len(A), len(B)) < d_max + 2:
            agrees[t] = np.nan
            continue
        JA, _ = nested_uncentred_svd(Z[A], d_max, device=device)
        JB, _ = nested_uncentred_svd(Z[B], d_max, device=device)
        for d in range(1, d_max + 1):
            agrees[t, d - 1] = prefix_agreement(JA, JB, d)
    return {"agreement": agrees}


def max_statistic(observed: np.ndarray, null_draws: np.ndarray) -> dict[str, float]:
    """Max across ranks of (obs - null_mean) / null_std; null of the max."""
    obs = np.asarray(observed, dtype=np.float64)
    nd = np.asarray(null_draws, dtype=np.float64)
    if nd.ndim == 1:
        nd = nd[:, None]
    mu = np.nanmean(nd, axis=0)
    sd = np.nanstd(nd, axis=0)
    sd = np.where(sd < EPS, 1.0, sd)
    z = (obs - mu) / sd
    z_max = float(np.nanmax(z))
    null_max = np.nanmax((nd - mu) / sd, axis=1)
    p = float(np.mean(null_max >= z_max - 1e-12)) if len(null_max) else float("nan")
    return {"z_max": z_max, "p_max": p, "z": z}


def quantile_threshold(null_draws: np.ndarray, q: float) -> float:
    x = np.asarray(null_draws, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))
