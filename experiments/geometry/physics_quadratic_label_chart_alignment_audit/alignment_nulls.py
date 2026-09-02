"""Haar right-singular randomization and matched-anchor alignment nulls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_quadratic_label_chart_alignment.alignment import alignment_AB

from .config import N_QUAD, STABILITY_THRESHOLD
from .io_util import p_mc


def haar_frame(q: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic Haar-like r-frame in R^q (QR of Gaussian with sign fix)."""
    A = rng.normal(size=(q, r))
    Q, R = np.linalg.qr(A, mode="reduced")
    diag = np.diag(R)
    Q = Q * np.sign(np.where(diag == 0, 1.0, diag))
    return Q


def alignment_from_spectrum(gamma: np.ndarray, S: np.ndarray, V: np.ndarray) -> float:
    """A_B = q (γ^T V Σ² V^T γ) / (|γ|² tr Σ²). V is q × r with orthonormal columns."""
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64)
    ng2 = float(g @ g)
    tr = float(np.sum(S * S))
    if ng2 < 1e-18 or tr < 1e-18:
        return float("nan")
    w = V.T @ g
    num = float(np.sum((S * w) ** 2))
    return float(N_QUAD * num / (ng2 * tr))


def haar_alignment(gamma: np.ndarray, S: np.ndarray, rng: np.random.Generator) -> float:
    """Spectrum-preserving orientation null (Haar r-frame), QR implementation."""
    r = int(S.size)
    q = int(np.asarray(gamma).size)
    V = haar_frame(q, r, rng)
    return alignment_from_spectrum(gamma, S, V)


def haar_alignment_fast(gamma: np.ndarray, S: np.ndarray, rng: np.random.Generator) -> float:
    """Same law as Haar r-frame A_B without forming Q: |V^T γ|^2/|γ|^2 ~ Beta(r/2,(q-r)/2)."""
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    q = int(g.size)
    r = int(S.size)
    ng2 = float(g @ g)
    tr = float(np.sum(S * S))
    if ng2 < 1e-18 or tr < 1e-18 or r < 1:
        return float("nan")
    direction = rng.normal(size=r)
    nrm = float(np.linalg.norm(direction))
    if nrm < 1e-18:
        return float("nan")
    direction = direction / nrm
    if r >= q:
        radius = np.sqrt(ng2)
    else:
        radius = np.sqrt(ng2 * float(rng.beta(r / 2.0, (q - r) / 2.0)))
    w = direction * radius
    num = float(np.sum((S * w) ** 2))
    return float(N_QUAD * num / (ng2 * tr))


def isotropic_alignment(S: np.ndarray, rng: np.random.Generator) -> float:
    q = N_QUAD
    g = rng.normal(size=q)
    g = g / max(float(np.linalg.norm(g)), 1e-12)
    r = int(S.size)
    # any orthonormal V: isotropic g makes V irrelevant; use identity frame on first r
    V = np.zeros((q, r))
    V[:r, :r] = np.eye(r)
    return alignment_from_spectrum(g, S, V)


def summarize_median_test(obs_vec: np.ndarray, null_medians: np.ndarray, *, greater: bool = True) -> dict:
    obs = float(np.nanmedian(obs_vec))
    B = int(len(null_medians))
    if greater:
        b = int(np.sum(null_medians >= obs))
    else:
        b = int(np.sum(null_medians <= obs))
    return {
        "observed_median": obs,
        "null_median": float(np.nanmedian(null_medians)),
        "null_mean": float(np.nanmean(null_medians)),
        "ci95_lo": float(np.nanpercentile(obs_vec, 2.5)),
        "ci95_hi": float(np.nanpercentile(obs_vec, 97.5)),
        "p_mc": p_mc(b, B),
        "n_null": B,
        "n_obs": int(np.isfinite(obs_vec).sum()),
    }


def matched_bins(df: pd.DataFrame) -> np.ndarray:
    """Coarse matching bins: radius × |K_H| × original rank quartiles."""
    parts = []
    for col in ("log_knn_radius", "K_H_cross", "r_original"):
        v = df[col].to_numpy(float) if col in df.columns else np.zeros(len(df))
        v = np.abs(v) if col == "K_H_cross" else v
        try:
            b = pd.qcut(pd.Series(v).rank(method="first"), 4, labels=False, duplicates="drop")
            b = np.asarray(b, dtype=float)
            b = np.nan_to_num(b, nan=0.0).astype(int)
        except (ValueError, TypeError):
            b = np.zeros(len(df), dtype=int)
        parts.append(b)
    return parts[0] * 100 + parts[1] * 10 + parts[2]


def permute_within_bins(values: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if len(idx) >= 2:
            out[idx] = rng.permutation(out[idx])
    return out


def high_stability_mask(cosine: np.ndarray) -> np.ndarray:
    return np.asarray(cosine, dtype=np.float64) >= STABILITY_THRESHOLD
