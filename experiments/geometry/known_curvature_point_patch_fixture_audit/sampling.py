"""Sampling regimes S0–S5 with recorded selection probabilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .config import D_LAT, S1_RHO_MAX
from .fixtures import GENERATORS_NP, bump_field, CLIFFORD_P
from .geometry import stereo_inv_np


def haar_sphere_chart(n: int, dim_sphere: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on S^{dim_sphere} via Gaussian in R^{dim_sphere+1}, stereographic chart."""
    g = rng.standard_normal((n, dim_sphere + 1))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return stereo_inv_np(g)


def sample_latent_S0(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name in ("F2", "F3"):
        u = haar_sphere_chart(n, CLIFFORD_P, rng)
        v = haar_sphere_chart(n, CLIFFORD_P, rng)
        return np.concatenate([u, v], axis=1)
    return haar_sphere_chart(n, D_LAT, rng)


def _softmax_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    s = scores / max(temperature, 1e-12)
    s = s - np.max(s)
    w = np.exp(s)
    w = w / np.clip(w.mean(), 1e-15, None)
    return w


def _weighted_subsample(
    z: np.ndarray, weights: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(weights, dtype=np.float64)
    p = np.clip(p, 1e-18, None)
    p = p / p.sum()
    idx = rng.choice(len(z), size=n, replace=False, p=p)
    return z[idx], p[idx]


def sample_condition(
    name: str,
    sampling: str,
    n: int,
    rng: np.random.Generator,
    Q: np.ndarray,
    k_true_fn=None,
) -> dict:
    """Return latent z, selection probability, and log-density proxy.

    Overgenerate then subsample so S1–S5 have well-defined inclusion probabilities.
    k_true_fn(z) -> (n,) K_dir used for S1 independence and S2/S3 correlation.
    """
    n_pool = max(8 * n, n + 64)
    z_pool = sample_latent_S0(name, n_pool, rng)
    ones = np.ones(n_pool, dtype=np.float64)

    if sampling == "S0":
        z, p = z_pool[:n], ones[:n]
        return {"z": z, "p_select": p, "log_density": np.log(p), "sampling": sampling}

    if sampling == "S4":
        # Compact holes: drop two seeded balls in chart coordinates.
        c1 = z_pool.mean(0)
        c2 = z_pool[0]
        d1 = np.linalg.norm(z_pool - c1, axis=1)
        d2 = np.linalg.norm(z_pool - c2, axis=1)
        keep = (d1 > np.quantile(d1, 0.12)) & (d2 > np.quantile(d2, 0.08))
        # One-sided sector: drop a half-space.
        axis = rng.standard_normal(D_LAT)
        axis /= np.linalg.norm(axis)
        keep = keep & ((z_pool @ axis) > -0.15)
        z_keep = z_pool[keep]
        if len(z_keep) < n:
            z_keep = z_pool
            keep = np.ones(n_pool, dtype=bool)
        idx = rng.choice(len(z_keep), size=n, replace=False)
        p = np.ones(n, dtype=np.float64)
        p *= float(keep.mean())
        return {
            "z": z_keep[idx],
            "p_select": p,
            "log_density": np.log(np.clip(p, 1e-18, None)),
            "sampling": sampling,
            "kept_fraction": float(keep.mean()),
        }

    if sampling == "S5":
        # Anisotropic sparsity: squeeze first latent coordinate.
        z = z_pool[:n].copy()
        z[:, 0] *= 0.20
        p = np.full(n, 0.20, dtype=np.float64)
        return {"z": z, "p_select": p, "log_density": np.log(p), "sampling": sampling}

    # S1–S3 need a curvature (or independent) field on the pool.
    if k_true_fn is None:
        raise ValueError("S1–S3 require k_true_fn")
    k_pool = np.asarray(k_true_fn(z_pool), dtype=np.float64)

    if sampling == "S1":
        # Independent smooth density: random latent direction, regenerate until |ρ|<0.05.
        accepted = None
        for t in range(32):
            direction = rng.standard_normal(D_LAT)
            direction /= np.linalg.norm(direction)
            field = z_pool @ direction
            rho = spearmanr(field, k_pool).correlation
            if np.isfinite(rho) and abs(float(rho)) < S1_RHO_MAX:
                accepted = (direction, field, float(rho), t)
                break
        if accepted is None:
            direction = rng.standard_normal(D_LAT)
            direction /= np.linalg.norm(direction)
            field = z_pool @ direction
            accepted = (direction, field, float(spearmanr(field, k_pool).correlation), 32)
        weights = _softmax_scores(field, temperature=np.std(field) + 1e-12)
        z, p = _weighted_subsample(z_pool, weights, n, rng)
        k_s = k_true_fn(z)
        if np.std(k_s) < 1e-15 or np.std(np.log(np.clip(p, 1e-18, None))) < 1e-15:
            rho_final = float("nan")
        else:
            rho_final = float(spearmanr(np.log(p), k_s).correlation)
        return {
            "z": z,
            "p_select": p,
            "log_density": np.log(np.clip(p, 1e-18, None)),
            "sampling": sampling,
            "rho_logp_K": rho_final,
            "s1_tries": accepted[3],
        }

    if sampling in ("S2", "S3"):
        sign = 1.0 if sampling == "S2" else -1.0
        field = sign * (k_pool - np.median(k_pool)) / (np.std(k_pool) + 1e-12)
        weights = _softmax_scores(field, temperature=0.5)
        z, p = _weighted_subsample(z_pool, weights, n, rng)
        k_s = k_true_fn(z)
        if np.std(k_s) < 1e-15 or np.std(np.log(np.clip(p, 1e-18, None))) < 1e-15:
            rho_final = float("nan")
        else:
            rho_final = float(spearmanr(np.log(p), k_s).correlation)
        return {
            "z": z,
            "p_select": p,
            "log_density": np.log(np.clip(p, 1e-18, None)),
            "sampling": sampling,
            "rho_logp_K": rho_final,
        }

    raise KeyError(sampling)
