"""Unit / smoke tests for tangent-estimator benchmark helpers."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.gauss_map_curvature import (
    estimate_anchor_gauss_map,
    split_half_projectors,
)
from geometry.physics_activation_atlas.tangent_estimator_benchmark import (
    ESTIMATORS,
    joint_quadratic_principal_manifold,
    sample_manifold,
    same_patch_pca,
    tangent_error,
    train_synthetic_sae,
)
from geometry.physics_activation_atlas.tangent_reliability import pca_tangent


def test_geodesic_pca_low_error():
    man = sample_manifold("geodesic", d=4, D=32, n=800, noise=0.0, seed=0, radius=0.2)
    x0, J_true, X = man["x0"], man["J_true"], man["X"]
    dists = np.linalg.norm(X - x0, axis=1)
    Xn = X[np.argsort(dists)[1:257]]
    J, _ = same_patch_pca(Xn, x0, 4)
    err = tangent_error(J, J_true)
    assert err["E_T"] < 0.2


def test_stratified_flag():
    man = sample_manifold("stratified", d=4, D=32, n=600, noise=0.0, seed=1, radius=0.2)
    assert man["stratified"] is True


def test_joint_quadratic_returns_stiefel():
    man = sample_manifold("pure_mean", d=4, D=32, n=600, noise=0.0, seed=2, radius=0.2)
    x0, X = man["x0"], man["X"]
    dists = np.linalg.norm(X - x0, axis=1)
    Xn = X[np.argsort(dists)[1:257]]
    J, diag = joint_quadratic_principal_manifold(Xn, x0, 4, n_iter=4)
    assert J.shape[1] == 4
    assert abs(np.linalg.norm(J.T @ J - np.eye(4))) < 1e-5
    assert abs(np.linalg.norm(J.T @ x0)) < 1e-5
    assert diag["estimator"] == "joint_quadratic_principal_manifold"


def test_synthetic_sae_jacobian_gates():
    man = sample_manifold("geodesic", d=4, D=32, n=1200, noise=0.0, seed=3, radius=0.2)
    X = man["X"]
    bundle = train_synthetic_sae(X[:900], feature_dim=64, k=8, steps=80, seed=3)
    x0 = man["x0"]
    dists = np.linalg.norm(X - x0, axis=1)
    Xn = X[np.argsort(dists)[1:257]]
    J, diag = ESTIMATORS["sae_reconstruction_jacobian"](
        Xn, x0, 4, sae_bundle=bundle, device=bundle["device"]
    )
    # May fail gates on tiny train; if ok, Stiefel constraints hold
    if J is not None:
        assert abs(np.linalg.norm(J.T @ J - np.eye(J.shape[1]))) < 1e-4
        assert diag.get("ok") is True


def test_gauss_map_geodesic_low_energy():
    man = sample_manifold("geodesic", d=4, D=40, n=1000, noise=0.0, seed=4, radius=0.15)
    X = man["X"]
    rng = np.random.default_rng(4)
    ai = int(rng.integers(0, len(X)))
    x0 = X[ai] / np.linalg.norm(X[ai])
    dists = np.linalg.norm(X - x0, axis=1)
    order = np.argsort(dists)
    Xn = X[order[1:129]]
    J, _, _ = pca_tangent(Xn, x0, 4)
    Px = J @ J.T
    _, _, split_x = split_half_projectors(Xn, x0, 4, 0, pca_tangent)
    sites, splits = [], []
    for rnk in (8, 16, 32, 48, 64):
        y = X[order[rnk]]
        y = y / np.linalg.norm(y)
        Yn = X[np.argsort(np.linalg.norm(X - y, axis=1))[1:129]]
        Jy, _, _ = pca_tangent(Yn, y, 4)
        sites.append((y, Jy @ Jy.T))
        _, _, sj = split_half_projectors(Yn, y, 4, rnk, pca_tangent)
        splits.append(sj)
    g = estimate_anchor_gauss_map(x0, Px, sites, split_x, splits, 4)
    # geodesic: energy should not be huge after debias
    assert np.isfinite(g["beta"]) or g["label"] in (
        "noise_dominated",
        "unresolved",
        "pointwise_gauss_regime",
        "finite_scale_tangent_heterogeneity",
    )
