"""Frozen Q plus T2 (matched neighbourhood) and T3 (uniform patch) oracles."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from geometry.known_curvature_point_patch_fixture_audit.estimator_q import (
    fit_anchor_quadratic,
    fit_anchors_parallel,
    knn_indices,
)
from geometry.known_curvature_point_patch_fixture_audit.oracle import _fit_population_quadratic
from geometry.known_curvature_point_patch_fixture_audit.geometry import hess_from_bs_flat

from .config import D_LAT, K_PRIMARY, N_ORACLE_MAX, N_SPLITS_Q
from .fixtures import GENERATORS_NP, energy_g, targets_from_jets, truth_at


def run_Q(
    X_obs: np.ndarray,
    X_anchor: np.ndarray,
    k: int,
    seed: int,
    n_workers: int,
    device: str = "cpu",
) -> list[dict]:
    idx = knn_indices(X_obs, X_anchor, k, device=None)  # CPU cdist is fine at D=28
    payloads = []
    for ai in range(len(X_anchor)):
        payloads.append(
            {
                "Xloc": X_obs[idx[ai]],
                "d": D_LAT,
                "n_splits": N_SPLITS_Q,
                "seed": seed,
                "ai": int(ai),
                "neigh_idx": idx[ai],
            }
        )
    fits = fit_anchors_parallel(
        [{k: p[k] for k in ("Xloc", "d", "n_splits", "seed", "ai")} for p in payloads],
        n_workers=n_workers,
    )
    out = []
    for p, f in zip(payloads, fits):
        rec = dict(f)
        rec["neigh_idx"] = p["neigh_idx"]
        rec["k"] = k
        out.append(rec)
    return out


def oracle_T2_matched(
    name: str,
    Qrot: np.ndarray,
    z_train: np.ndarray,
    neigh_idx: np.ndarray,
    z_anchor: np.ndarray,
) -> dict:
    """Clean outputs of the actual Q neighbourhood, exact latents, true tangent at the anchor."""
    z_nb = z_train[np.asarray(neigh_idx, dtype=np.int64)]
    Y = GENERATORS_NP[name](z_nb, Qrot)
    geo = truth_at(name, z_anchor, Qrot)
    U = z_nb - z_anchor[None, :]
    w = np.ones(len(z_nb), dtype=np.float64)
    B = _fit_population_quadratic(Y, geo["G"], geo["J"], w, U=U)
    tgt = targets_from_jets(geo["G"], geo["J"], B)
    # B from the quadratic fit IS already a Hessian-convention residual tensor in the true chart
    # but _fit_population_quadratic projects with PNS, so B ≈ B^S of the patch model.
    tgt["B_S"] = B
    tgt["source"] = "T2_matched"
    tgt["n_patch"] = int(len(z_nb))
    return tgt


def oracle_T3_uniform(
    name: str,
    Qrot: np.ndarray,
    z_anchor: np.ndarray,
    radius_lat: float,
    n: int = N_ORACLE_MAX,
    seed: int = 0,
) -> dict:
    """Deterministic Sobol design in the latent ball of the same radius (shared across sampling)."""
    d = z_anchor.shape[0]
    m = max(1, int(np.ceil(np.log2(max(n, 2)))))
    directions = qmc.Sobol(d=d, scramble=True, seed=int(seed) + 1).random_base2(m)[:n]
    from scipy.special import erfinv

    gauss = np.sqrt(2.0) * erfinv(np.clip(2.0 * directions - 1.0, -1 + 1e-12, 1 - 1e-12))
    gauss /= np.clip(np.linalg.norm(gauss, axis=1, keepdims=True), 1e-15, None)
    rad_u = qmc.Sobol(d=1, scramble=True, seed=int(seed) + 2).random_base2(m)[:n]
    z = z_anchor[None, :] + gauss * (float(radius_lat) * rad_u)
    z[0] = z_anchor
    Y = GENERATORS_NP[name](z, Qrot)
    geo = truth_at(name, z_anchor, Qrot)
    U = z - z_anchor[None, :]
    w = np.ones(len(z), dtype=np.float64)
    B = _fit_population_quadratic(Y, geo["G"], geo["J"], w, U=U)
    tgt = targets_from_jets(geo["G"], geo["J"], B)
    tgt["B_S"] = B
    tgt["source"] = "T3_uniform"
    tgt["n_patch"] = int(len(z))
    tgt["radius_lat"] = float(radius_lat)
    return tgt


def q_tensors(fit: dict, d: int = D_LAT) -> dict:
    if not fit.get("ok"):
        return {"ok": False}
    HA = fit["Hess_A"]
    HB = fit["Hess_B"]
    Havg = 0.5 * (HA + HB)
    J = np.asarray(fit["J"][:, :d], dtype=np.float64)
    x0 = np.asarray(fit["x0"], dtype=np.float64)
    tgtA = targets_from_jets(x0, J, HA)
    tgtB = targets_from_jets(x0, J, HB)
    tgt = targets_from_jets(x0, J, Havg)
    # split-half intrinsic scalar, unaveraged traces, no clamp
    scal_cross = float(
        d * (d - 1)
        + np.dot(tgtA["H_S"], tgtB["H_S"])
        - energy_g_cross(tgtA["B_S"], tgtB["B_S"], tgtA["ginv"])
    )
    kh = float(fit.get("agg", {}).get("K_H_cross", np.nan))
    kdir = float(fit.get("agg", {}).get("K_dir_cross", np.nan))
    return {
        "ok": True,
        "tgt": tgt,
        "tgtA": tgtA,
        "tgtB": tgtB,
        "Scal_cross": scal_cross,
        "K_H_cross": kh,
        "K_dir_cross": kdir,
        "J": J,
        "x0": x0,
        "ridge": fit.get("agg", {}).get("ridge_note"),
        "radius_median": fit.get("radius_median"),
        "split_cos": _tensor_cos(tgtA["B_S"], tgtB["B_S"]),
    }


def energy_g_cross(A: np.ndarray, B: np.ndarray, ginv: np.ndarray) -> float:
    Bw = np.einsum("ac,bd,iab->icd", ginv, ginv, B)
    return float(np.tensordot(A, Bw, axes=([0, 1, 2], [0, 1, 2])))


def _tensor_cos(A: np.ndarray, B: np.ndarray) -> float:
    num = float(np.tensordot(A, B, axes=([0, 1, 2], [0, 1, 2])))
    na = float(np.linalg.norm(A))
    nb = float(np.linalg.norm(B))
    return float(num / max(na * nb, 1e-30))
