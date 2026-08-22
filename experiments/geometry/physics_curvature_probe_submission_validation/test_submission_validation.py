"""Unit tests: schema, identities, local R², projectors, whitening, parity constants."""

from __future__ import annotations

import numpy as np
import pytest

from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric
from geometry.physics_activation_atlas.effdim_curvature_metrics import cross_metric_pair, metric_scalars
from geometry.physics_activation_atlas.global_probe_curvature_alignment import local_r2_fixed_predictions, weighted_r2
from geometry.physics_activation_atlas.sphere_normal_quadratic import normal_projector_apply, sphere_project_basis
from geometry.physics_curvature_probe_rank_sweep.pipeline import kh_trace_identity
from geometry.physics_curvature_probe_submission_validation.config import FROZEN_CTL, FROZEN_DELTA_20_12, FROZEN_RAW
from geometry.physics_curvature_probe_submission_validation.schema import (
    PRIMARY,
    TargetKind,
    assert_not_catalog_vector,
    assert_probe_performance,
    kind_of,
)


def test_schema_rejects_ambiguous_and_catalog():
    assert kind_of("mag_r_desi_local_oof_r2") is TargetKind.PROBE_PERFORMANCE
    assert kind_of("mag_r_desi_catalog_value") is TargetKind.CATALOG_VALUE
    with pytest.raises(RuntimeError):
        assert_probe_performance("mag_r_desi")
    with pytest.raises(RuntimeError):
        assert_probe_performance("local_r2")
    with pytest.raises(RuntimeError):
        assert_probe_performance("mag_r_desi_catalog_value")
    assert assert_probe_performance(PRIMARY.value) == PRIMARY.value


def test_catalog_vector_rejected():
    y = np.linspace(0, 1, 32)
    assert_not_catalog_vector(y, y + 2.0)
    with pytest.raises(RuntimeError):
        assert_not_catalog_vector(y, y.copy())


def test_local_r2_matches_sse_sst():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    yhat = y + 0.1 * rng.normal(size=50)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst
    assert abs(local_r2_fixed_predictions(y, yhat) - r2) < 1e-12
    w = np.ones(len(y))
    assert abs(weighted_r2(y, yhat, w) - r2) < 1e-12


def test_projectors_and_radial():
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=8)
    x0 = x0 / np.linalg.norm(x0)
    J = rng.normal(size=(8, 3))
    J = sphere_project_basis(x0, J)
    assert np.max(np.abs(x0 @ J)) < 1e-10
    assert np.max(np.abs(J.T @ J - np.eye(3))) < 1e-8
    V = rng.normal(size=(8, 5))
    P = normal_projector_apply(V, x0, J)
    P2 = normal_projector_apply(P, x0, J)
    assert np.max(np.abs(P - P2)) < 1e-10
    assert np.max(np.abs(x0 @ P)) < 1e-10
    assert np.max(np.abs(J.T @ P)) < 1e-10


def test_whitening_rms():
    rng = np.random.default_rng(2)
    U = rng.normal(size=(200, 4)) * np.array([1.0, 2.0, 0.5, 3.0])
    sc = np.sqrt((U**2).mean(0))
    Uw = U / sc
    assert np.allclose((Uw**2).mean(0), 1.0, atol=1e-12)


def test_kh_cross_vs_norm_identity():
    rng = np.random.default_rng(3)
    d = 4
    q = d * (d + 1) // 2
    D = 6
    flat = rng.normal(size=(D, q))
    B = unpack_BS_symmetric(flat, d)
    H = B[:, np.arange(d), np.arange(d)].mean(axis=1)
    kh = float(np.sqrt(np.dot(H, H)))
    assert abs(kh - metric_scalars(flat, d)["K_H"]) < 1e-10
    assert abs(kh - kh_trace_identity(flat, d)) < 1e-10
    pair = cross_metric_pair(flat, flat, d)
    assert abs(pair["K_H_cross"] - float(np.dot(H, H))) < 1e-10


def test_frozen_constants():
    assert abs(FROZEN_CTL[20] - FROZEN_CTL[12] - FROZEN_DELTA_20_12) < 1e-12
    assert abs(FROZEN_RAW[16] + 0.412430) < 1e-6
    assert abs(FROZEN_CTL[16] + 0.240484) < 1e-6
