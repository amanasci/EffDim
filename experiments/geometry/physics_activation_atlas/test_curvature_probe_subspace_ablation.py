"""Unit tests for curvature-subspace probe ablation primitives."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.curvature_probe_subspace_ablation import (
    ambient_quadratic_form,
    haar_normal_basis,
    orthonormalize_mutually,
    phi_weighted,
    run_synthetic_ablation,
)


def test_orthonormalize_mutually():
    rng = np.random.default_rng(0)
    D, d, r = 32, 6, 8
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    UB = rng.normal(size=(D, r))
    T2, x0u, UB2 = orthonormalize_mutually(T, x0, UB)
    assert np.allclose(T2.T @ T2, np.eye(T2.shape[1]), atol=1e-6)
    assert abs(np.linalg.norm(x0u) - 1) < 1e-6
    assert np.allclose(T2.T @ x0u, 0, atol=1e-5)
    assert np.allclose(UB2.T @ x0u, 0, atol=1e-5)
    assert np.allclose(UB2.T @ T2, 0, atol=1e-5)


def test_phi_weighted_offdiag():
    z = np.array([[1.0, 2.0]])
    phi = phi_weighted(z)[0]
    # z0^2, sqrt2 z0 z1, z1^2
    assert abs(phi[0] - 1.0) < 1e-9
    assert abs(phi[1] - np.sqrt(2) * 2.0) < 1e-9
    assert abs(phi[2] - 4.0) < 1e-9


def test_ambient_quadratic_form_matches_tensor():
    rng = np.random.default_rng(1)
    D, d = 16, 3
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + B.transpose(0, 2, 1))
    z = rng.normal(size=(5, d))
    V = ambient_quadratic_form(B, z)
    for i in range(5):
        v = np.zeros(D)
        for a in range(d):
            for b in range(d):
                v += B[:, a, b] * z[i, a] * z[i, b]
        assert np.allclose(V[i], v, atol=1e-8)


def test_haar_in_normal_complement():
    rng = np.random.default_rng(2)
    D, d, r = 40, 5, 7
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    U = haar_normal_basis(D, x0, T, r, rng)
    assert U.shape == (D, r)
    assert np.allclose(U.T @ U, np.eye(r), atol=1e-6)
    assert np.allclose(U.T @ x0, 0, atol=1e-5)
    assert np.allclose(U.T @ T, 0, atol=1e-5)


def test_synthetics_pass():
    out = run_synthetic_ablation(0)
    assert out["pass"], out["checks"]
