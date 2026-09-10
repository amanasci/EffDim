"""Noise regimes N0–N5. The clean generator remains the truth."""

from __future__ import annotations

import numpy as np

from .config import D_AMB, D_LAT
from .geometry import sphere_normal_projectors


def _onb_normal(G: np.ndarray, J: np.ndarray, rng: np.random.Generator, n_dirs: int) -> np.ndarray:
    """Sample unit vectors in the sphere-normal space at G."""
    g, PT, PNS, Gh = sphere_normal_projectors(G, J)
    # Draw ambient gaussians and project.
    E = rng.standard_normal((n_dirs, G.shape[0]))
    E = E @ PNS.T
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    nrm = np.clip(nrm, 1e-15, None)
    return E / nrm


def energy_fractions(eps: np.ndarray, G: np.ndarray, J: np.ndarray) -> dict[str, float]:
    g, PT, PNS, Gh = sphere_normal_projectors(G, J)
    et = eps @ PT
    er = np.outer(eps @ Gh, Gh) if eps.ndim == 2 else Gh * float(np.dot(eps, Gh))
    if eps.ndim == 1:
        et = PT @ eps
        er = Gh * float(np.dot(eps, Gh))
        en = PNS @ eps
        tot = float(np.dot(eps, eps)) + 1e-18
        return {
            "frac_tangent": float(np.dot(et, et) / tot),
            "frac_radial": float(np.dot(er, er) / tot),
            "frac_sphere_normal": float(np.dot(en, en) / tot),
        }
    en = eps @ PNS
    tot = np.sum(eps * eps, axis=1) + 1e-18
    return {
        "frac_tangent": float(np.mean(np.sum(et * et, axis=1) / tot)),
        "frac_radial": float(np.mean(np.sum((eps * Gh[None, :]).sum(axis=1, keepdims=True) ** 2, axis=1) / tot)),
        "frac_sphere_normal": float(np.mean(np.sum(en * en, axis=1) / tot)),
    }


def apply_noise(
    *,
    name: str,
    noise: str,
    X_clean: np.ndarray,
    z: np.ndarray,
    Q: np.ndarray,
    generator,
    jacobian_fn,
    eta: float,
    r_med: float,
    rng: np.random.Generator,
    k_true: np.ndarray | None = None,
) -> dict:
    """eta = median||ε|| / median r_k. Returns noisy X on the unit sphere and diagnostics."""
    n, D = X_clean.shape
    mag = float(eta) * float(r_med)
    diag = {"noise": noise, "eta": float(eta), "r_med": float(r_med), "mag": mag}

    if noise == "N0" or eta <= 0.0:
        return {"X": X_clean.copy(), "diagnostics": {**diag, "frac_tangent": 0.0, "frac_radial": 0.0, "frac_sphere_normal": 0.0}}

    if noise == "N1":
        # Tangent jitter in latent coordinates; manifold not thickened.
        scale = mag / 4.0  # latent units; physical scale set by typical ||J||~O(1) on the sphere
        z2 = z + rng.standard_normal(z.shape) * scale
        X = generator(z2, Q)
        if X.ndim == 1:
            X = X[None, :]
        return {"X": X, "diagnostics": {**diag, "frac_tangent": 1.0, "frac_radial": 0.0, "frac_sphere_normal": 0.0}}

    # Need a normal frame per point for N2/N4/N5. Use a cheap FD Jacobian at each point
    # is expensive (n×d). For noise, project with the position and a local PCA-free
    # sphere-tangent from the generator Jacobian at a subsample, else use G-only radial
    # plus random ambient projected with (I-GG^T) which mixes tangent+normal.
    # Proper: autodiff J is costly. Use analytic stereo Jacobian for F0/F1/F2/F3 via FD
    # on a *batch* of random ONB in (I-GG^T), then split tangent vs normal using J.
    #
    # Fast path: sample ε in ambient, decompose with J if provided by jacobian_fn
    # for all points. jacobian_fn(z_i)->(D,d) may be slow; we compute in chunks of 64
    # using the torch map when available. Here we accept a callable jacobian_fn(z)->J.

    J_all = jacobian_fn(z)  # (n, D, d)
    if noise == "N3":
        eps = rng.standard_normal((n, D))
        eps *= mag / (np.linalg.norm(eps, axis=1, keepdims=True) + 1e-15)
        # set median ||ε|| = mag
        eps *= mag / (np.median(np.linalg.norm(eps, axis=1)) + 1e-15)
        fracs = []
        for i in range(n):
            fracs.append(energy_fractions(eps[i], X_clean[i], J_all[i]))
        X = X_clean + eps
        X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-15, None)
        return {
            "X": X,
            "diagnostics": {
                **diag,
                "frac_tangent": float(np.mean([f["frac_tangent"] for f in fracs])),
                "frac_radial": float(np.mean([f["frac_radial"] for f in fracs])),
                "frac_sphere_normal": float(np.mean([f["frac_sphere_normal"] for f in fracs])),
            },
        }

    # Sphere-normal noise (N2, N4, N5)
    thickness = np.full(n, mag, dtype=np.float64)
    if noise == "N4":
        field = z @ rng.standard_normal(D_LAT)
        field = (field - field.mean()) / (field.std() + 1e-12)
        thickness = mag * np.exp(0.5 * field)
        thickness *= mag / (np.median(thickness) + 1e-15)
    if noise in ("N5", "N5n"):
        if k_true is None:
            raise ValueError("N5 requires k_true")
        k = (k_true - np.median(k_true)) / (np.std(k_true) + 1e-12)
        sign = -1.0 if noise == "N5n" else 1.0
        thickness = mag * np.exp(0.6 * sign * k)
        thickness *= mag / (np.median(thickness) + 1e-15)

    eps = np.zeros((n, D), dtype=np.float64)
    for i in range(n):
        g, PT, PNS, Gh = sphere_normal_projectors(X_clean[i], J_all[i])
        v = rng.standard_normal(D)
        v = PNS @ v
        nv = float(np.linalg.norm(v))
        if nv < 1e-15:
            continue
        eps[i] = (thickness[i] / nv) * v
    X = X_clean + eps
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-15, None)
    return {
        "X": X,
        "diagnostics": {
            **diag,
            "frac_tangent": 0.0,
            "frac_radial": 0.0,
            "frac_sphere_normal": 1.0,
            "median_thickness": float(np.median(thickness)),
        },
    }
