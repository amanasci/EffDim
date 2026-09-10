"""Production Q fit wrapper. Returns scalars only — never Hessians or B^S."""

from __future__ import annotations

import hashlib

import numpy as np

from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    cross_metric_pair,
    decompose_tensors,
)
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad
from geometry.physics_activation_atlas.nested_dimension_curvature import _fit_rank
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices

from .config import D, EXPERIMENT_SEED, K, PRODUCTION_SEED


def hash_seed(*parts) -> int:
    msg = ":".join(str(p) for p in parts)
    h = hashlib.sha256(f"{EXPERIMENT_SEED}:{msg}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1)


def partition_halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    mid = int(n) // 2
    a, b = perm[:mid], perm[mid:]
    return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)


def design_condition(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray) -> float:
    U = (np.asarray(Xloc, dtype=np.float64) - np.asarray(x0, dtype=np.float64)[None, :]) @ np.asarray(J, dtype=np.float64)
    cols = []
    d = J.shape[1]
    for a in range(d):
        for b in range(a, d):
            cols.append(U[:, a] * U[:, b])
    Phi = np.stack(cols, axis=1)
    g = Phi.T @ Phi
    w = np.clip(np.linalg.eigvalsh(g), 0.0, None)
    return float(w.max() / max(float(w.min()), 1e-18))


def fit_explicit_halves(
    Xloc: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    halfA: np.ndarray,
    halfB: np.ndarray,
    *,
    d: int = D,
    seed: int,
    device=None,
) -> dict:
    """One production split-half pair. Drops BS_flat before return."""
    halfA = np.asarray(halfA, dtype=np.int64)
    halfB = np.asarray(halfB, dtype=np.int64)
    if set(halfA).intersection(set(halfB)):
        raise ValueError("halves are not disjoint")
    fA, vA = _half_fit_indices(halfA, int(seed) + 3)
    fB, vB = _half_fit_indices(halfB, int(seed) + 7)
    Jd = np.asarray(J, dtype=np.float64)[:, :d]
    chA, _, infoA = fit_quad(Xloc, x0, Jd, fA, vA, halfB, ridges=RIDGES, device=device)
    chB, _, infoB = fit_quad(Xloc, x0, Jd, fB, vB, halfA, ridges=RIDGES, device=device)
    if chA is None or chB is None:
        return {"ok": False, "K_H_cross": float("nan"), "K_dir_cross": float("nan")}
    cross = cross_metric_pair(chA.BS_flat, chB.BS_flat, d)
    HA = decompose_tensors(chA.BS_flat, d)["H"]
    HB = decompose_tensors(chB.BS_flat, d)["H"]
    na, nb = float(np.linalg.norm(HA)), float(np.linalg.norm(HB))
    hcos = float(np.dot(HA, HB) / max(na * nb, 1e-30))
    out = {
        "ok": True,
        "K_H_cross": float(cross["K_H_cross"]),
        "K_dir_cross": float(cross["K_dir_cross"]),
        "K_aniso_cross": float(cross["K_aniso_cross"]),
        "split_H_cosine": hcos,
        "dS": 0.5 * (float(infoA.get("dS", np.nan)) + float(infoB.get("dS", np.nan))),
        "n_A": int(len(halfA)),
        "n_B": int(len(halfB)),
        "clamped": False,
    }
    del chA, chB
    return out


def production_mean_kh(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray, ai: int, k: int = K) -> dict:
    """Exact production _fit_rank average (n_splits=5, seed=0). Scalars only."""
    fits = _fit_rank(Xloc, x0, J, D, k, 5, PRODUCTION_SEED, int(ai))
    if not fits:
        return {"ok": False, "K_H_cross": float("nan"), "K_dir_cross": float("nan")}
    kh = float(np.mean([f["K_H_cross"] for f in fits]))
    kd = float(np.mean([f["K_dir_cross"] for f in fits]))
    for f in fits:
        f.pop("BS_flat_A", None)
        f.pop("BS_flat_B", None)
        f.pop("H_mean", None)
    return {"ok": True, "K_H_cross": kh, "K_dir_cross": kd, "n_splits": len(fits)}
