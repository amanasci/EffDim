"""Unit tests for global-probe curvature magnitude analysis."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.global_probe_curvature_magnitude import (
    a_h_from_w_H,
    bh_fdr,
    classify_target,
    rank_z,
)


def test_a_h_bounds_and_unstable():
    rng = np.random.default_rng(0)
    D, d = 16, 4
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    H = rng.normal(size=D) * 0.2
    H = H - T @ (T.T @ H) - x0 * np.dot(x0, H)
    w = H + 0.1 * rng.normal(size=D)
    ah, uns = a_h_from_w_H(w, T, x0, H)
    assert not uns
    assert 0 <= ah <= 1 + 1e-6
    ah0, uns0 = a_h_from_w_H(w, T, x0, np.zeros(D))
    assert uns0 and not np.isfinite(ah0)


def test_rank_z_finite():
    x = np.array([1.0, 2.0, np.nan, 3.0])
    z = rank_z(x)
    assert np.isfinite(z[[0, 1, 3]]).all()
    assert np.isnan(z[2])


def test_bh_fdr_and_classify():
    adj = bh_fdr(np.array([0.001, 0.04, 0.2]))
    assert adj[0] <= adj[1]
    lab, _ = classify_target(
        {
            "mean_local_r2": 0.3,
            "partial_A_B_C2": -0.2,
            "p_perm_A_B_C2": 0.01,
            "partial_K_traceless_C0": -0.02,
            "p_perm_K_traceless_C0": 0.5,
            "partial_K_mean_C0": 0.01,
            "p_perm_K_mean_C0": 0.8,
            "partial_PCA_given_AB": 0.05,
            "raw_A_PCA_normal": 0.05,
            "raw_A_B_normal": -0.2,
        },
        {"coef_interaction_M4": 0.01, "ci_interaction_M4": [-0.1, 0.1], "dcv_M4": 0.0},
    )
    assert lab == "orientation_mismatch"
