"""Local tangent directional-imbalance boundary scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def directional_imbalance(
    X: np.ndarray,
    idx: np.ndarray,
    basis: np.ndarray,
    *,
    k: int = 32,
    eps: float = 1e-8,
) -> dict:
    """
    For points X[idx], project neighbour offsets into chart tangent (basis: D x d)
    and compute mean unit-direction magnitude.
    """
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean").fit(X)
    dists, neigh = nn.kneighbors(X[idx])
    scores = np.zeros(len(idx), dtype=np.float64)
    radii = dists[:, -1]
    residuals = np.zeros(len(idx), dtype=np.float64)
    for t, i in enumerate(idx):
        js = neigh[t, 1:]
        V = (X[js] - X[i]) @ basis  # (k, d)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        U = V / np.maximum(norms, eps)
        # equal neighbour weights
        b = U.mean(axis=0)
        scores[t] = float(np.linalg.norm(b))
        # tangent residual in ambient
        offs = X[js] - X[i]
        proj = (offs @ basis) @ basis.T
        residuals[t] = float(np.mean(np.linalg.norm(offs - proj, axis=1)))
    return {
        "boundary_score": scores,
        "knn_radius": radii.astype(np.float64),
        "tangent_residual": residuals,
        "indices": idx.astype(np.int64),
    }


def dtm_proxy(X: np.ndarray, idx: np.ndarray, *, k: int = 32) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean").fit(X)
    dists, _ = nn.kneighbors(X[idx])
    return dists[:, 1:].mean(axis=1)


def chart_boundary_diagnostics(
    X: np.ndarray,
    pcas: list[dict],
    membership_w_dense: np.ndarray,
    *,
    max_points_per_chart: int = 200,
    k: int = 32,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    all_scores = []
    all_dtm = []
    all_rad = []
    for c, pca in enumerate(pcas):
        w = membership_w_dense[:, c]
        cand = np.where(w > 1e-4)[0]
        if len(cand) == 0:
            continue
        if len(cand) > max_points_per_chart:
            p = w[cand] / w[cand].sum()
            cand = rng.choice(cand, size=max_points_per_chart, replace=False, p=p)
        d = directional_imbalance(X, cand, pca["basis"], k=k)
        dtm = dtm_proxy(X, cand, k=k)
        # density-matched synthetic control: isotropic Gaussian in ambient then project? 
        # simpler: scramble neighbour directions via random tangent signs
        ctrl = d["boundary_score"].copy()
        # matched control: sample from local gaussian cloud around each point
        ctrl_scores = []
        for t, i in enumerate(cand):
            noise = rng.normal(0, d["knn_radius"][t] / np.sqrt(X.shape[1]), size=(k, X.shape[1]))
            V = noise @ pca["basis"]
            U = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
            ctrl_scores.append(float(np.linalg.norm(U.mean(axis=0))))
        ctrl_scores = np.asarray(ctrl_scores)
        # correlations with density proxies
        if len(cand) > 5:
            corr_dtm = float(np.corrcoef(d["boundary_score"], dtm)[0, 1])
            corr_rad = float(np.corrcoef(d["boundary_score"], d["knn_radius"])[0, 1])
        else:
            corr_dtm = corr_rad = float("nan")
        row = {
            "chart": c,
            "n": int(len(cand)),
            "boundary_mean": float(d["boundary_score"].mean()),
            "boundary_p90": float(np.percentile(d["boundary_score"], 90)),
            "control_mean": float(ctrl_scores.mean()),
            "excess_vs_control": float(d["boundary_score"].mean() - ctrl_scores.mean()),
            "corr_with_dtm": corr_dtm,
            "corr_with_knn_radius": corr_rad,
            "tangent_residual_mean": float(d["tangent_residual"].mean()),
        }
        rows.append(row)
        all_scores.append(d["boundary_score"])
        all_dtm.append(dtm)
        all_rad.append(d["knn_radius"])
    mean_excess = float(np.mean([r["excess_vs_control"] for r in rows])) if rows else 0.0
    mean_corr = float(np.nanmean([r["corr_with_dtm"] for r in rows])) if rows else float("nan")
    label = "boundary_structure_detected" if mean_excess > 0.05 and abs(mean_corr) < 0.7 else "boundary_density_confounded"
    return {
        "charts": rows,
        "mean_excess_vs_control": mean_excess,
        "mean_corr_with_dtm": mean_corr,
        "label": label,
        "note": "directional imbalance is density-confounded; not proof of manifold boundary",
    }


def save_boundary(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "boundary.json").write_text(json.dumps(result, indent=2))
