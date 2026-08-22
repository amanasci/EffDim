"""Unit tests for atlas primitives (curvature analytics + memberships + priors)."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.charts import estimate_bandwidths, select_chart_centres, soft_memberships
from geometry.physics_activation_atlas.curvature import run_curvature_unit_tests
from geometry.physics_activation_atlas.priors import fit_mle_gmm, fit_standard_gaussian, weighted_loglik
from geometry.physics_activation_atlas.synthetic import validate_synthetic_atlas


def test_curvature_analytics():
    out = run_curvature_unit_tests(device="cpu")
    assert out["all_pass"], out


def test_soft_memberships_partition():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    centres = select_chart_centres(X, n_charts=6, method="fps", seed=0)
    bw = estimate_bandwidths(X, centres)
    W, meta = soft_memberships(X, X[centres], bw, charts_per_sample=3)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    assert meta["frac_with_ge2_charts"] > 0.9


def test_gmm_prior_loglik_finite():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((300, 4)).astype(np.float32)
    w = np.ones(300)
    g0 = fit_standard_gaussian(4)
    g1 = fit_mle_gmm(U, w, n_components=2, seed=0)
    assert np.isfinite(weighted_loglik(g0, U, w))
    assert np.isfinite(weighted_loglik(g1, U, w))


def test_synthetic_smoke_fast():
    out = validate_synthetic_atlas(n=120, n_charts=5, charts_per_sample=3, latent_dim=4, seed=0, ambient=16)
    assert out["all_curvature_tests_pass"]
    assert "plane" in out["manifolds"]
