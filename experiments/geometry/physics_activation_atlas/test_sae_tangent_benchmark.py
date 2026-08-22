"""Unit tests for SAE tangent v2 helpers."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.sae_tangent_benchmark import (
    E_T_thin,
    SynthDecoder,
    _local_isomap_coords,
    projector_frobenius_sq_thin,
    sample_latent,
    sae_code_covariance_pushforward,
    train_synthetic_sae,
)


def test_linear_decoder_exact_tangent_low_error():
    dec = SynthDecoder("linear", d=4, D=32, seed=0)
    rng = np.random.default_rng(0)
    Z = sample_latent("gaussian", 600, 4, rng)
    X = dec.embed(Z)
    z0 = Z[0]
    x0 = X[0]
    Jstar = dec.true_tangent(z0)
    # PCA on neighbours should be close
    dists = np.linalg.norm(X - x0, axis=1)
    Xn = X[np.argsort(dists)[1:129]]
    from geometry.physics_activation_atlas.tangent_reliability import pca_tangent

    J, _, _ = pca_tangent(Xn, x0, 4)
    assert E_T_thin(J, Jstar) < 0.25


def test_thin_projector_identity():
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    assert projector_frobenius_sq_thin(U, U) < 1e-10


def test_isomap_coords_shape():
    rng = np.random.default_rng(2)
    S = rng.normal(size=(40, 16))
    Z = _local_isomap_coords(S, n_comp=4, knn=8)
    assert Z.shape == (40, 4)


def test_code_pushforward_runs():
    dec = SynthDecoder("linear", d=4, D=32, seed=3)
    rng = np.random.default_rng(3)
    Z = sample_latent("gaussian", 800, 4, rng)
    X = dec.embed(Z).astype(np.float32)
    bundle = train_synthetic_sae(X[:600], feature_dim=64, k=8, steps=60, seed=3)
    x0 = X[600]
    Xn = X[601:729]
    J, diag = sae_code_covariance_pushforward(Xn, x0, 4, sae_bundle=bundle, device=bundle["device"])
    assert J is not None or diag.get("ok") is False
    if J is not None:
        assert J.shape[1] == 4
