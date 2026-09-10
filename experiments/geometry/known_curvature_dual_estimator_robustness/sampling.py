"""Sampling S0–S3 and noise N0–N4. Anchors are independent of training rows."""

from __future__ import annotations

import hashlib

import numpy as np

from .config import (
    ANCHOR_HASH_SEED,
    BETA_DENSITY,
    DATA_SEED,
    N1_RMS,
    N2_RMS,
    N3_RMS,
    N4_RMS_DENSE,
    N4_RMS_SPARSE,
    N_ANCHORS,
    N_MAX,
    N_SPARSE,
)
from .fixtures import GENERATORS_NP, jacobian_batch_fd, sample_latent


def _hash_order(ids: np.ndarray, seed: int) -> np.ndarray:
    keyed = []
    for sid in np.asarray(ids, dtype=np.int64):
        h = hashlib.sha256(f"{int(seed)}:{int(sid)}".encode()).hexdigest()
        keyed.append((h, int(sid)))
    keyed.sort()
    return np.asarray([s for _, s in keyed], dtype=np.int64)


def density_score(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), -1.0, 1.0)
    return np.exp(BETA_DENSITY * s)


def make_anchors(name: str, Q: np.ndarray, n: int = N_ANCHORS, seed: int = ANCHOR_HASH_SEED) -> dict:
    rng = np.random.default_rng(int(seed) + {"F0": 0, "F1": 1, "F2": 2, "F4": 4}[name])
    z, s = sample_latent(name, 256, rng)
    order = _hash_order(np.arange(len(z)), seed)
    take = order[:n]
    z_a, s_a = z[take], s[take]
    X = GENERATORS_NP[name](z_a, Q)
    w = density_score(s_a)
    ids = np.asarray([100000 + int(i) for i in take], dtype=np.int64)  # disjoint from train ids
    return {"z": z_a, "s": s_a, "w": w, "X": X, "sample_id": ids, "name": name}


def sample_training(name: str, sampling: str, Q: np.ndarray, seed: int = DATA_SEED) -> dict:
    n = N_SPARSE if sampling in ("S1", "S3") else N_MAX
    rng = np.random.default_rng(int(seed) + 17 * {"S0": 0, "S1": 1, "S2": 2, "S3": 3}[sampling] + {"F0": 0, "F1": 1, "F2": 2, "F4": 4}[name])
    if sampling in ("S0", "S1"):
        z, s = sample_latent(name, n, rng)
        w = density_score(s)
        p = np.ones(n, dtype=np.float64) / n
    else:
        n_pool = max(8 * n, n + 64)
        z_pool, s_pool = sample_latent(name, n_pool, rng)
        w_pool = density_score(s_pool)
        p = w_pool / w_pool.sum()
        idx = rng.choice(n_pool, size=n, replace=False, p=p)
        z, s, w = z_pool[idx], s_pool[idx], w_pool[idx]
        p = p[idx]
    X = GENERATORS_NP[name](z, Q)
    ids = np.arange(n, dtype=np.int64)
    return {
        "z": z,
        "s": s,
        "w": w,
        "p_select": p,
        "X_clean": X,
        "sample_id": ids,
        "sampling": sampling,
        "n": n,
        "name": name,
    }


def s_x_scale(X: np.ndarray) -> float:
    mu = X.mean(axis=0)
    return float(np.median(np.linalg.norm(X - mu[None, :], axis=1)))


def apply_noise(name: str, noise: str, train: dict, Q: np.ndarray, seed: int = DATA_SEED) -> dict:
    X = np.asarray(train["X_clean"], dtype=np.float64)
    z = train["z"]
    n, D = X.shape
    rng = np.random.default_rng(int(seed) + 101 * {"N0": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 4}[noise] + n)
    sx = s_x_scale(X)
    rec = {
        "s_x": sx,
        "noise": noise,
        "frac_tangent": 0.0,
        "frac_radial": 0.0,
        "frac_sphere_normal": 0.0,
        "rms_eps": 0.0,
    }
    if noise == "N0":
        return {**rec, "X_obs": X.copy()}

    if noise == "N4":
        w = train["w"]
        # densest → 0.01 sx, sparsest → 0.05 sx
        t = (np.log(np.clip(w, 1e-18, None)) - np.log(w.min() + 1e-18))
        t = t / max(float(t.max()), 1e-12)
        rms = sx * (N4_RMS_SPARSE + (N4_RMS_DENSE - N4_RMS_SPARSE) * t)
    elif noise == "N1":
        rms = np.full(n, N1_RMS * sx)
    elif noise == "N2":
        rms = np.full(n, N2_RMS * sx)
    else:
        rms = np.full(n, N3_RMS * sx)

    from geometry.known_curvature_point_patch_fixture_audit.geometry import sphere_normal_projectors

    G0, Jb = jacobian_batch_fd(name, z, Q)
    eps = rng.standard_normal((n, D))
    tan = rad = nrm = np.zeros(n)
    if noise == "N3":
        en = np.linalg.norm(eps, axis=1, keepdims=True)
        eps = eps / np.clip(en, 1e-15, None) * rms[:, None]
        for i in range(n):
            _, PT, PNS, Gh = sphere_normal_projectors(G0[i], Jb[i])
            e = eps[i]
            tot = float(np.dot(e, e)) + 1e-18
            et, er, en_ = PT @ e, Gh * float(np.dot(e, Gh)), PNS @ e
            tan[i], rad[i], nrm[i] = np.dot(et, et) / tot, np.dot(er, er) / tot, np.dot(en_, en_) / tot
    else:
        for i in range(n):
            _, PT, PNS, Gh = sphere_normal_projectors(G0[i], Jb[i])
            e = rng.standard_normal(D)
            e = PNS @ e
            e = e / max(float(np.linalg.norm(e)), 1e-15) * float(rms[i])
            eps[i] = e
            tot = float(np.dot(e, e)) + 1e-18
            et, er, en_ = PT @ e, Gh * float(np.dot(e, Gh)), PNS @ e
            tan[i], rad[i], nrm[i] = np.dot(et, et) / tot, np.dot(er, er) / tot, np.dot(en_, en_) / tot

    Xn = X + eps
    Xn = Xn / np.clip(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-15, None)
    rec.update(
        {
            "X_obs": Xn,
            "eps": eps,
            "frac_tangent": float(np.mean(tan)),
            "frac_radial": float(np.mean(rad)),
            "frac_sphere_normal": float(np.mean(nrm)),
            "rms_eps": float(np.sqrt(np.mean(np.sum(eps**2, axis=1)))),
            "rms_target": float(np.mean(rms)),
        }
    )
    return rec
