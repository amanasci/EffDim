"""Overlapping atlas: FPS / stratified centres + soft RBF memberships."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from topology.physics_activation_density_ph.landmarks import (  # noqa: E402
    density_stratified_landmarks,
    farthest_point_landmarks,
)


def select_chart_centres(
    X_train: np.ndarray,
    *,
    n_charts: int,
    method: str,
    seed: int,
    dtm: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    method = method.lower()
    if method in {"fps", "farthest_point"}:
        return farthest_point_landmarks(
            X_train,
            n_landmarks=n_charts,
            seed=seed,
            dtm=dtm,
            device=device,
        )
    if method in {"density_stratified", "stratified"}:
        if dtm is None:
            raise ValueError("density_stratified chart selection requires DTM")
        return density_stratified_landmarks(dtm, n_landmarks=n_charts, seed=seed)
    raise ValueError(method)


def estimate_bandwidths(
    X_train: np.ndarray,
    centres_local: np.ndarray,
    *,
    policy: str = "median_knn",
    knn_k: int = 32,
) -> np.ndarray:
    """Per-chart bandwidth from training distances to centre."""
    from sklearn.neighbors import NearestNeighbors

    C = X_train[centres_local]
    # distance from all train points to each centre
    # for bandwidth: median distance among k nearest train points to centre
    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(X_train)), metric="euclidean")
    nn.fit(X_train)
    dists, _ = nn.kneighbors(C)
    # drop self if centre is in train (distance ~0)
    radii = dists[:, -1]
    if policy == "median_knn":
        h = np.maximum(radii, 1e-3)
    elif policy == "mean_knn":
        h = np.maximum(dists[:, 1:].mean(axis=1), 1e-3)
    else:
        h = np.maximum(radii, 1e-3)
    return h.astype(np.float64)


def soft_memberships(
    X: np.ndarray,
    centres: np.ndarray,
    bandwidths: np.ndarray,
    *,
    charts_per_sample: int,
) -> tuple[sparse.csr_matrix, dict]:
    """RBF soft memberships to nearest r centres; rows sum to 1."""
    from sklearn.neighbors import NearestNeighbors

    r = min(charts_per_sample, len(centres))
    nn = NearestNeighbors(n_neighbors=r, metric="euclidean").fit(centres)
    dists, idxs = nn.kneighbors(X)
    n, C = len(X), len(centres)
    rows, cols, data = [], [], []
    for i in range(n):
        d2 = dists[i] ** 2
        h = bandwidths[idxs[i]]
        w = np.exp(-d2 / (2.0 * np.maximum(h, 1e-8) ** 2))
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
        for j, c in enumerate(idxs[i]):
            rows.append(i)
            cols.append(int(c))
            data.append(float(w[j]))
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, C))
    # diagnostics
    row_nnz = np.diff(W.indptr)
    pi = np.asarray(W.sum(axis=0)).ravel() / max(n, 1)
    # membership entropy per sample
    ent = []
    for i in range(n):
        s, e = W.indptr[i], W.indptr[i + 1]
        wi = W.data[s:e]
        wi = wi / max(wi.sum(), 1e-12)
        ent.append(float(-np.sum(wi * np.log(np.maximum(wi, 1e-12)))))
    ent = np.asarray(ent)
    multi = float(np.mean(row_nnz >= 2))
    meta = {
        "charts_per_sample": r,
        "frac_with_ge2_charts": multi,
        "membership_entropy_mean": float(ent.mean()),
        "membership_entropy_p10": float(np.percentile(ent, 10)),
        "membership_entropy_p90": float(np.percentile(ent, 90)),
        "pi_min": float(pi.min()),
        "pi_max": float(pi.max()),
        "pi_entropy": float(-np.sum(pi * np.log(np.maximum(pi, 1e-12)))),
        "support_sizes": [int((W[:, c].data > 1e-6).sum()) for c in range(C)],
    }
    return W, meta


def enforce_chart_population(
    W: sparse.csr_matrix,
    *,
    min_chart_samples: int,
    max_chart_samples: int | None,
) -> tuple[sparse.csr_matrix, list[int], dict]:
    """Drop tiny charts; optionally cap mass via thresholding (record only)."""
    C = W.shape[1]
    supports = np.array([(W[:, c].data > 1e-6).sum() for c in range(C)])
    keep = np.where(supports >= min_chart_samples)[0]
    dropped = [int(c) for c in range(C) if c not in set(keep.tolist())]
    if len(keep) == 0:
        keep = np.argsort(-supports)[: max(1, min(3, C))]
        dropped = [int(c) for c in range(C) if c not in set(keep.tolist())]
    W2 = W[:, keep]
    # renormalize rows
    row_sums = np.asarray(W2.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1e-12)
    W2 = sparse.diags(1.0 / row_sums) @ W2
    meta = {
        "kept_charts": [int(c) for c in keep],
        "dropped_charts": dropped,
        "supports_before": supports.tolist(),
        "min_chart_samples": min_chart_samples,
        "max_chart_samples": max_chart_samples,
    }
    return W2.tocsr(), [int(c) for c in keep], meta


def save_charts(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "chart_centres.npz",
        centres_train_local=payload["centres_train_local"],
        centres_global_local=payload["centres_global_local"],
        bandwidths=payload["bandwidths"],
        kept_original_ids=payload["kept_original_ids"],
    )
    sparse.save_npz(out / "memberships_csr.npz", payload["W"])
    (out / "charts_meta.json").write_text(json.dumps(payload["meta"], indent=2, default=str))
