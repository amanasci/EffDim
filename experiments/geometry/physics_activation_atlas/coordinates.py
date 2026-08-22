"""Weighted local PCA coordinates per chart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from topology.physics_activation_density_ph.data import effective_rank_from_cov  # noqa: E402


def weighted_local_pca(
    X: np.ndarray,
    weights: np.ndarray,
    *,
    n_components: int,
    weight_floor: float = 1e-8,
) -> dict:
    w = np.asarray(weights, dtype=np.float64)
    mask = w > weight_floor
    if mask.sum() < n_components + 2:
        # fallback: top-weight points
        order = np.argsort(-w)[: max(n_components + 5, 10)]
        mask = np.zeros(len(w), dtype=bool)
        mask[order] = True
    Xw = X[mask]
    ww = w[mask]
    ww = ww / ww.sum()
    mu = (ww[:, None] * Xw).sum(axis=0)
    Xc = Xw - mu
    # weighted SVD via sqrt-weight rows
    Y = np.sqrt(ww)[:, None] * Xc
    # economy SVD
    _, s, vt = np.linalg.svd(Y, full_matrices=False)
    d = min(n_components, vt.shape[0])
    W = vt[:d].T  # (D, d)
    evals = (s[:d] ** 2).astype(np.float64)
    # standardize scales from train coords
    U = (X - mu) @ W
    # weighted std on members
    um = U[mask]
    std = np.sqrt(np.maximum((ww[:, None] * (um**2)).sum(axis=0), 1e-12))
    return {
        "mu": mu.astype(np.float32),
        "basis": W.astype(np.float32),
        "eigenvalues": evals,
        "coord_std": std.astype(np.float32),
        "n_effective": float((ww.sum() ** 2) / np.maximum((ww**2).sum(), 1e-12)),
        "n_members": int(mask.sum()),
    }


def encode_chart(X: np.ndarray, pca: dict, *, standardize: bool = True) -> np.ndarray:
    U = (X - pca["mu"]) @ pca["basis"]
    if standardize:
        U = U / np.maximum(pca["coord_std"], 1e-8)
    return U.astype(np.float32)


def local_rank_diagnostics(pca: dict) -> dict:
    ev = np.asarray(pca["eigenvalues"], dtype=np.float64)
    tot = float(ev.sum()) if ev.size else 0.0
    if tot <= 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0, "rank95": 0}
    p = ev / tot
    p_pos = p[p > 0]
    eff = float(np.exp(-np.sum(p_pos * np.log(p_pos))))
    pr = float((p.sum() ** 2) / np.maximum((p**2).sum(), 1e-30))
    csum = np.cumsum(p)
    rank95 = int(np.searchsorted(csum, 0.95) + 1)
    return {"effective_rank": eff, "participation_ratio": pr, "rank95": rank95}


def fit_all_charts(
    X: np.ndarray,
    W: sparse.csr_matrix,
    *,
    n_components: int,
    train_idx: np.ndarray,
) -> list[dict]:
    C = W.shape[1]
    out = []
    for c in range(C):
        w_all = np.asarray(W[:, c].todense()).ravel()
        w_tr = np.zeros(len(X), dtype=np.float64)
        w_tr[train_idx] = w_all[train_idx]
        pca = weighted_local_pca(X, w_tr, n_components=n_components)
        pca["chart"] = c
        pca["diagnostics"] = local_rank_diagnostics(pca)
        out.append(pca)
    return out


def save_coordinates(out: Path, pcas: list[dict], coords: dict, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for c, pca in enumerate(pcas):
        np.savez_compressed(
            out / f"pca_chart{c}.npz",
            mu=pca["mu"],
            basis=pca["basis"],
            eigenvalues=pca["eigenvalues"],
            coord_std=pca["coord_std"],
        )
    np.savez_compressed(out / "coords.npz", **coords)
    (out / "coordinates_meta.json").write_text(json.dumps(meta, indent=2, default=str))
