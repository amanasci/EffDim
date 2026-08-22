"""Unit tests for tangent-reliability helpers."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.sphere_normal_quadratic import (
    normal_projector_apply,
    sphere_project_basis,
)
from geometry.physics_activation_atlas.tangent_reliability import (
    grassmann_dist,
    kernel_weights,
    normal_residual_scaling,
    pca_tangent,
    projector,
    run_synthetic_controls,
)


def test_sphere_project_orthogonal_to_x0():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=16)
    x0 /= np.linalg.norm(x0)
    J0, _ = np.linalg.qr(rng.normal(size=(16, 5)))
    J = sphere_project_basis(x0, J0)
    assert np.linalg.norm(J.T @ x0) < 1e-8
    assert np.allclose(J.T @ J, np.eye(J.shape[1]), atol=1e-6)


def test_normal_projector_kills_span():
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=12)
    x0 /= np.linalg.norm(x0)
    J, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    J = sphere_project_basis(x0, J)
    v = 0.3 * x0 + J @ rng.normal(size=J.shape[1]) + 0.01 * rng.normal(size=12)
    n = normal_projector_apply(v, x0, J)
    Q, _ = np.linalg.qr(np.column_stack([x0, J]))
    assert np.linalg.norm(Q.T @ n) < 1e-6


def test_pca_tangent_recovers_plane():
    rng = np.random.default_rng(2)
    D, d, n = 24, 4, 200
    Jtrue, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - Jtrue @ (Jtrue.T @ x0)
    x0 /= np.linalg.norm(x0)
    U = rng.normal(size=(n, d)) * 0.2
    X = x0 + U @ Jtrue.T
    # row normalize lightly
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    J, _, diag = pca_tangent(X, x0, d)
    dist = grassmann_dist(projector(J), projector(Jtrue), d)
    assert dist < 0.25
    assert diag["d_eff"] == d


def test_kernel_weights_peak_at_zero():
    d = np.array([0.0, 0.2, 0.9, 2.0])
    w = kernel_weights(d, bandwidth=1.0)
    assert w[0] >= w[1] >= w[2]
    assert w[3] == 0.0


def test_residual_scaling_affine_vs_rotated():
    rng = np.random.default_rng(3)
    D, d, n = 20, 3, 400
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - J @ (J.T @ x0)
    x0 /= np.linalg.norm(x0)
    U = rng.normal(size=(n, d)) * 0.3
    X = (x0 + U @ J.T)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sc_ok = normal_residual_scaling(X, x0, J, np.arange(100, n))
    R, _ = np.linalg.qr(rng.normal(size=(D, D)))
    Jbad = sphere_project_basis(x0, R[:, :d])
    sc_bad = normal_residual_scaling(X, x0, Jbad, np.arange(100, n))
    assert "slope_log" in sc_ok and "leakage_frac" in sc_bad
    assert sc_bad["n_eval"] >= 16


def test_synthetics_pass():
    res = run_synthetic_controls(0)
    assert res["pass"], res["checks"]
