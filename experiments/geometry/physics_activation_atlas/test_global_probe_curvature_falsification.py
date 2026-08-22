"""Unit tests for falsification helpers."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.global_probe_curvature_falsification import (
    make_null_neighborhood,
    mc_p,
    n_quad_features,
    select_smoke_anchors,
    unit_normal_component,
)
import pandas as pd


def test_n_quad():
    assert n_quad_features(8) == 36
    assert n_quad_features(16) == 136


def test_mc_p_never_zero():
    p, B = mc_p(1.0, np.zeros(10), greater=True)
    assert B == 10
    assert p >= 1 / 11


def test_unit_normal_orthogonal():
    rng = np.random.default_rng(0)
    D, d = 32, 6
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    w = rng.normal(size=D)
    wN = unit_normal_component(w, T, x0)
    assert abs(np.dot(wN, x0)) < 1e-8
    assert np.linalg.norm(T.T @ wN) < 1e-8


def test_null_preserves_shape_and_norm():
    rng = np.random.default_rng(1)
    n, D, d = 100, 32, 6
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    u = rng.normal(size=(n, d)) * 0.2
    resid = rng.normal(size=(n, D)) * 0.05
    resid = resid - (resid @ T) @ T.T - np.outer(resid @ x0, x0)
    Xn = x0 + u @ T.T + resid
    Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    Xnull = make_null_neighborhood(Xn, x0, T, rng)
    assert Xnull.shape == Xn.shape
    assert np.allclose(np.linalg.norm(Xnull, axis=1), 1.0, atol=1e-5)


def test_smoke_selection_ignores_r2():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "knn_radius": rng.random(n),
            "reconstruction_error": rng.random(n),
            "local_r2": rng.random(n),  # must not drive selection
        }
    )
    ids = select_smoke_anchors(df, 48, seed=0)
    assert len(ids) == 48
    assert len(set(ids)) == 48
