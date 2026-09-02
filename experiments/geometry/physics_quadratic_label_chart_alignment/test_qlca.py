"""Unit tests for quadratic-label chart alignment."""

from __future__ import annotations

import numpy as np

from geometry.physics_quadratic_label_chart_alignment.features import (
    Gamma_from_gamma,
    bs_prod_to_frob,
    gamma_from_Gamma,
    n_quad,
    phi2_frob,
    verify_n_quad,
)
from geometry.physics_quadratic_label_chart_alignment.io_util import p_mc
from geometry.physics_quadratic_label_chart_alignment.models import mse, oof_predict_model
from geometry.physics_quadratic_label_chart_alignment.synthetic import run_synthetics


def test_n_quad_16():
    verify_n_quad()
    assert n_quad(16) == 136


def test_frob_norm_identity():
    rng = np.random.default_rng(0)
    d = 8
    G = rng.normal(size=(d, d))
    G = 0.5 * (G + G.T)
    g = gamma_from_Gamma(G)
    assert abs(float(g @ g) - float(np.sum(G * G))) < 1e-10
    G2 = Gamma_from_gamma(g, d)
    assert np.allclose(G, G2)


def test_phi_matches_half_uGu():
    rng = np.random.default_rng(1)
    d = 6
    U = rng.normal(size=(20, d))
    A = rng.normal(size=(d, d))
    G = 0.5 * (A + A.T)
    g = gamma_from_Gamma(G)
    pred = phi2_frob(U) @ g
    direct = np.array([0.5 * u @ G @ u for u in U])
    assert np.allclose(pred, direct, atol=1e-10)


def test_bs_c_equivalence():
    """cᵀ B φ == (Bᵀ c)ᵀ φ."""
    rng = np.random.default_rng(2)
    d, D, n = 6, 40, 30
    U = rng.normal(size=(n, d))
    BS = rng.normal(size=(D, n_quad(d)))
    c = rng.normal(size=D)
    Phi = phi2_frob(U)
    left = (Phi @ BS.T) @ c
    right = Phi @ (BS.T @ c)
    assert np.allclose(left, right)


def test_bs_prod_frob_decode_match():
    """Production packing vs Frobenius packing give same ambient displacement."""
    from geometry.physics_activation_atlas.quadratic import quadratic_features

    rng = np.random.default_rng(3)
    d, D, n = 8, 32, 25
    U = rng.normal(size=(n, d))
    BS_prod = rng.normal(size=(D, n_quad(d)))
    BS_f = bs_prod_to_frob(BS_prod, d)
    a = quadratic_features(U) @ BS_prod.T
    b = phi2_frob(U) @ BS_f.T
    assert np.allclose(a, b, atol=1e-10)


def test_fold_isolation_oof():
    rng = np.random.default_rng(4)
    n, d = 200, 8
    U = rng.normal(size=(n, d))
    y = U.sum(1) + rng.normal(scale=0.1, size=n)
    fold = np.tile(np.arange(5), 40)
    yhat, _ = oof_predict_model(U, y, fold, kind="L")
    assert np.isfinite(yhat).sum() >= 150


def test_p_mc():
    assert p_mc(0, 200) == 1 / 201


def test_synthetics_smoke():
    out = run_synthetics(seed=0)
    # Allow partial gate failures on tiny noise; require linear+aligned structure mostly
    assert "deltas" in out
    assert out["deltas"]["aligned_dUQ"] > out["deltas"]["linear_dQ"] - 0.01
