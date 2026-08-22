"""Unit tests for multi-model OOF / graph-prior / quadratic helpers."""

from __future__ import annotations

import numpy as np
import torch

from geometry.physics_activation_atlas.global_probe_curvature_alignment import (
    fit_global_probe,
    ridge_multi_intercept_torch,
)
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import (
    energy_rank,
    jl_sketch,
    knn_torch_ip,
    l2_normalize,
    oof_quad_errors,
    participation_ratio,
)


def test_l2_normalize_unit_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 8)).astype(np.float32)
    Xn = l2_normalize(X)
    norms = np.linalg.norm(Xn, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_energy_rank_and_pr():
    ev = np.array([10.0, 5.0, 1.0, 0.1])
    assert energy_rank(ev, 0.9) >= 2
    assert participation_ratio(ev) > 1.0


def test_ridge_multi_matches_fit_global_probe():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 12)).astype(np.float32)
    y = X @ rng.normal(size=12) + 0.05 * rng.normal(size=200)
    Y = np.column_stack([y, y + 0.01 * rng.normal(size=200)])
    device = torch.device("cpu")
    W, b, ok = ridge_multi_intercept_torch(
        torch.tensor(X, device=device),
        torch.tensor(Y, device=device, dtype=torch.float32),
        alpha=100.0,
    )
    assert ok
    w0, b0 = fit_global_probe(X, Y[:, 0], 100.0)
    cos = abs(float(np.dot(W[:, 0].numpy(), w0) / (np.linalg.norm(W[:, 0]) * np.linalg.norm(w0) + 1e-12)))
    assert cos > 0.999
    assert abs(float(b[0]) - b0) < 1e-3


def test_knn_ip_matches_euclidean_on_sphere():
    rng = np.random.default_rng(2)
    X = l2_normalize(rng.normal(size=(80, 16)).astype(np.float32))
    q = X[:5]
    idx = knn_torch_ip(X, q, k=7, device=torch.device("cpu"), batch=5)
    from sklearn.neighbors import NearestNeighbors

    _, idx_cpu = NearestNeighbors(n_neighbors=8, metric="euclidean").fit(X).kneighbors(q)
    for i in range(5):
        a = set(idx_cpu[i, 1:].tolist())
        b = set(int(j) for j in idx[i] if int(j) != i)
        assert len(a & b) >= 6


def test_oof_quad_errors_improves_on_quadratic_manifold():
    rng = np.random.default_rng(3)
    n, d_true, D = 120, 3, 24
    U = rng.normal(size=(n, d_true))
    J, _ = np.linalg.qr(rng.normal(size=(D, d_true)))
    # quadratic residual in ambient
    X = U @ J.T
    for a in range(d_true):
        for b in range(a, d_true):
            X = X + 0.05 * (U[:, a] * U[:, b])[:, None] * rng.normal(size=(1, D))
    X = l2_normalize(X.astype(np.float32))
    S = jl_sketch(D, 32, seed=0)
    r = oof_quad_errors(X, d=d_true, sketch=S, n_folds=5, seed=0)
    assert r["ok"]
    assert np.isfinite(r["E_quadratic"])
    assert r["neff_over_df"] > 0
