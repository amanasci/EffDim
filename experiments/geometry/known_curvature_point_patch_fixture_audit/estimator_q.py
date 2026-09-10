"""Estimator Q: frozen local quadratic patch curvature (exact production path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad  # noqa: E402
from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame  # noqa: E402
from geometry.physics_activation_atlas.split_half_curvature_reliability import (  # noqa: E402
    _half_fit_indices,
)
from geometry.physics_cross_model_full_curvature_reconciliation.metrics import (  # noqa: E402
    cross_metric_pair,
    decompose_tensors,
)

from .config import D_LAT, N_SPLITS_Q
from .geometry import hess_from_bs_flat, kdir_from_pair, pack_symmetric_weights, unpack_BS_symmetric


def pca_frame(Xloc: np.ndarray, d: int, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0, J, ev, diag = nested_pca_frame(Xloc, d, device)
    return x0, J, ev, diag


def knn_indices(X: np.ndarray, anchors: np.ndarray, k: int, device: torch.device | None = None) -> np.ndarray:
    """Return (n_anchors, k) neighbour indices into X (excluding the query if present)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.as_tensor(X, device=device, dtype=torch.float32)
    At = torch.as_tensor(anchors, device=device, dtype=torch.float32)
    # chunked cdist
    out = np.empty((len(anchors), k), dtype=np.int64)
    bs = 64
    for s in range(0, len(anchors), bs):
        d = torch.cdist(At[s : s + bs], Xt)
        # drop exact self matches by large diagonal-like: if a query equals a row, zero dist
        kn = torch.topk(d, k=min(k + 1, Xt.shape[0]), largest=False)
        idx = kn.indices.detach().cpu().numpy()
        for i, row in enumerate(idx):
            # drop first if essentially the point itself
            if row.size and d[i, row[0]].item() < 1e-12:
                take = row[1 : k + 1]
            else:
                take = row[:k]
            if take.size < k:
                take = row[:k]
            out[s + i] = take[:k]
    return out


def radii_from_knn(X: np.ndarray, idx: np.ndarray, x0: np.ndarray) -> np.ndarray:
    neigh = X[idx]
    return np.linalg.norm(neigh - x0[None, :], axis=1)


def fit_anchor_quadratic(
    Xloc: np.ndarray,
    d: int,
    n_splits: int,
    seed: int,
    ai: int,
    device: torch.device | None = None,
    weights: np.ndarray | None = None,
) -> dict:
    """Frozen nested_pca_frame + _fit_rank algebra. Optional inverse-density weights."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0, J, ev, diag = pca_frame(Xloc, d, device)
    k = len(Xloc)
    X_use = Xloc
    if weights is not None:
        # Weighted clone: replicate sqrt(w) by transforming rows. Not an edit of fit_quad.
        w = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, None)
        w = w / np.mean(w)
        # Pass through as a diagonal metric on residuals by scaling centred points.
        # Equivalent WLS for the linearised residual: scale (X-x0) and keep x0.
        X_use = x0[None, :] + np.sqrt(w)[:, None] * (Xloc - x0[None, :])
    rows = []
    BS_A = BS_B = None
    for s in range(n_splits):
        rng = np.random.default_rng(seed + 1009 * ai + 17 * s + d * 13 + k)
        perm = rng.permutation(k)
        halfA, halfB = perm[: k // 2], perm[k // 2 :]
        fA, vA = _half_fit_indices(halfA, seed + 3 + s)
        fB, vB = _half_fit_indices(halfB, seed + 7 + s)
        chA, _, infoA = fit_quad(X_use, x0, J[:, :d], fA, vA, halfB, ridges=RIDGES, device=device)
        chB, _, infoB = fit_quad(X_use, x0, J[:, :d], fB, vB, halfA, ridges=RIDGES, device=device)
        if chA is None or chB is None:
            continue
        cross = cross_metric_pair(chA.BS_flat, chB.BS_flat, d)
        HA = decompose_tensors(chA.BS_flat, d)["H"]
        HB = decompose_tensors(chB.BS_flat, d)["H"]
        HessA = hess_from_bs_flat(chA.BS_flat, d)
        HessB = hess_from_bs_flat(chB.BS_flat, d)
        gA = J[:, :d].T @ J[:, :d]
        hx = kdir_from_pair(HessA, HessB, gA)
        rho = float(np.median(np.linalg.norm(Xloc - x0[None, :], axis=1)))
        kdir_h = hx["K_dir_cross"]
        C_rho = float(np.sign(kdir_h) * rho * np.sqrt(abs(kdir_h))) if np.isfinite(kdir_h) else float("nan")
        rec = {
            "split": s,
            "K_H_cross_unpacked": cross["K_H_cross"],
            "K_aniso_cross_unpacked": cross["K_aniso_cross"],
            "K_dir_cross_unpacked": cross["K_dir_cross"],
            "K_H_cross": hx["K_H_cross"],
            "K_tf_cross": hx["K_tf_cross"],
            "K_dir_cross": hx["K_dir_cross"],
            "C_rho": C_rho,
            "R_H": cross["R_H"],
            "R_B0": cross["R_B0"],
            "R_BS": cross["R_BS"],
            "ridge_note": "frozen RIDGES grid; negative cross not clamped",
            "C_rho_convention": "sign(K_dir_cross)*rho*sqrt(|K_dir_cross|) on Hessian tensors",
        }
        rows.append(rec)
        if s == 0:
            BS_A, BS_B = chA.BS_flat, chB.BS_flat
    if not rows:
        return {"ok": False, "x0": x0, "J": J}
    agg = {k: float(np.mean([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])) for k in rows[0] if k not in ("split", "ridge_note", "C_rho_convention")}
    return {
        "ok": True,
        "x0": x0,
        "J": J,
        "ev": ev,
        "splits": rows,
        "agg": agg,
        "BS_flat_A": BS_A,
        "BS_flat_B": BS_B,
        "Hess_A": hess_from_bs_flat(BS_A, d) if BS_A is not None else None,
        "Hess_B": hess_from_bs_flat(BS_B, d) if BS_B is not None else None,
        "radius_median": float(np.median(np.linalg.norm(Xloc - x0[None, :], axis=1))),
        "radius_max": float(np.max(np.linalg.norm(Xloc - x0[None, :], axis=1))),
        "n_loc": int(k),
        "dS_note": "A/B split fits; production unpacked scalars retained alongside Hessian convention",
    }


def _cpu_worker_init() -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _q_worker(payload: dict) -> dict:
    """CPU quadratic fit; picklable. Avoids concurrent CUDA from many threads."""
    import torch

    try:
        return fit_anchor_quadratic(
            payload["Xloc"],
            payload["d"],
            n_splits=payload["n_splits"],
            seed=payload["seed"],
            ai=payload["ai"],
            device=torch.device("cpu"),
            weights=payload.get("weights"),
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def fit_anchors_parallel(payloads: list[dict], n_workers: int) -> list[dict]:
    if not payloads:
        return []
    workers = max(1, int(n_workers))
    if workers == 1 or len(payloads) == 1:
        return [_q_worker(p) for p in payloads]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(payloads)),
        mp_context=ctx,
        initializer=_cpu_worker_init,
    ) as ex:
        return list(ex.map(_q_worker, payloads, chunksize=1))


def knn_fixed_radius(X: np.ndarray, x0: np.ndarray, radius: float) -> np.ndarray:
    d = np.linalg.norm(X - x0[None, :], axis=1)
    idx = np.where(d <= radius)[0]
    return idx
