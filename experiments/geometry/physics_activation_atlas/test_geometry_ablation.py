"""Tests for geometry ablation primitives."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.curvature import (
    run_curvature_unit_tests,
    sphere_tangent_decompose,
)
from geometry.physics_activation_atlas.quadratic import fit_quadratic_chart, quadratic_features


def test_quadratic_features_count():
    U = np.random.default_rng(0).normal(size=(20, 4))
    Phi = quadratic_features(U)
    assert Phi.shape == (20, 10)


def test_quadratic_improves_or_matches_linear_on_quadratic_data():
    rng = np.random.default_rng(1)
    d, D, n = 3, 16, 200
    W = rng.normal(size=(D, d))
    W, _ = np.linalg.qr(W)
    U = rng.normal(size=(n, d))
    # quadratic ambient signal then normalize
    Y = U @ W.T + 0.2 * ((U[:, :1] * U[:, 1:2]) @ rng.normal(size=(1, D)))
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    pca = {
        "mu": Y.mean(0).astype(np.float32),
        "basis": W.astype(np.float32),
        "coord_std": np.ones(d, dtype=np.float32),
    }
    # re-encode approximately
    U2 = ((Y - pca["mu"]) @ pca["basis"]).astype(np.float32)
    w = np.ones(n)
    tr, va = np.arange(160), np.arange(160, 200)
    q, info = fit_quadratic_chart(pca, U2[tr], Y[tr], w[tr], U2[va], Y[va], w[va])
    from geometry.physics_activation_atlas.decoder import pca_reconstruct
    from geometry.physics_activation_atlas.metrics import weighted_mse

    mse_q = weighted_mse(q.decode(U2[va]), Y[va], w[va])
    mse_p = weighted_mse(pca_reconstruct(pca, U2[va]), Y[va], w[va])
    assert mse_q <= mse_p + 1e-3
    assert np.isfinite(info["val_mse"])


def test_sphere_tangent_decompose_radial():
    x = np.array([1.0, 0, 0])
    H = np.array([0.5, 0.1, 0.0])
    d = sphere_tangent_decompose(H, x)
    assert abs(d["H_rad"][0] - 0.5) < 1e-9
    assert abs(d["H_sphere"][0]) < 1e-9
    assert abs(d["H_sphere"][1] - 0.1) < 1e-9


def test_analytic_curvature_still_passes():
    assert run_curvature_unit_tests(device="cpu")["all_pass"]
