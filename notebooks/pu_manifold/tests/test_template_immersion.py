"""
notebooks/pu_manifold/tests/test_template_immersion.py -- D-14: the four-template immersion
generator (canonical embedding -> random orthogonal lift -> named smooth warp) and D-14's
own Jacobian rank check, verified at sampled points, never assumed.

Not collected by the core `effdim` test suite (`pyproject.toml`'s `testpaths = ["tests"]`
excludes this directory) -- run explicitly:

    .venv/bin/python -m pytest notebooks/pu_manifold/tests/test_template_immersion.py -q

Every test here pins a function against an input whose answer is known independently (a
unit-sphere row norm, a torus's own defining distance-from-tube-centre-circle identity, an
orthonormal-lift Gram-matrix identity, a rank-1-collapsed lift that MUST fail the immersion
check) or a hand-verified structural property (Duff et al.'s branchless
orthonormal-basis-from-normal construction's singularity-free poles, numerically checked
this session at 10,000 random draws and at the exact poles (0,0,+-1)) -- same discipline as
`test_persistence_probe.py` and `test_decoder_curvature.py`.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from pu_manifold import template_immersion as ti

FIXTURE_SEED = 20260807
"""Matches this phase's other fixed-seed artifacts (e.g. `template_tracer_run.py`'s
`TRACER_SEED` family), so this module's own numbers are directly comparable to theirs."""

AMBIENT_D = 768
"""Phase 02.7's production dimensionality -- D-15's required grid level, and this plan's own
production-dimensionality acceptance criterion."""


# --- Task 1: the four canonical templates, the orthogonal lift, the named warp ------------


def test_canonical_sample_s1_returns_unit_circle_with_analytic_tangent():
    rng = np.random.default_rng(FIXTURE_SEED)
    points, tangent = ti.canonical_sample("S1", 300, rng)
    assert points.shape == (300, 2) and points.dtype == np.float64
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
    assert tangent.shape == (300, 2, 1)
    dots = np.einsum("ni,nij->nj", points, tangent)
    assert np.allclose(dots, 0.0, atol=1e-12)


def test_canonical_sample_s2_is_unit_norm_via_normalized_gaussian_not_angle_chart():
    rng = np.random.default_rng(FIXTURE_SEED)
    points, tangent = ti.canonical_sample("S2", 500, rng)
    assert points.shape == (500, 3) and points.dtype == np.float64
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
    assert tangent.shape == (500, 3, 2)
    dots = np.einsum("ni,nij->nj", points, tangent)
    assert np.allclose(dots, 0.0, atol=1e-10)
    gram = np.einsum("nik,nil->nkl", tangent, tangent)
    assert np.allclose(gram, np.broadcast_to(np.eye(2), gram.shape), atol=1e-10)


def test_canonical_sample_t2_distance_from_tube_centre_circle_equals_r():
    rng = np.random.default_rng(FIXTURE_SEED)
    points, tangent = ti.canonical_sample("T2", 400, rng)
    assert points.shape == (400, 3) and points.dtype == np.float64
    assert tangent.shape == (400, 3, 2)
    R, r = ti.T2_MAJOR_RADIUS, ti.T2_MINOR_RADIUS
    rho = np.linalg.norm(points[:, :2], axis=1)
    dist_from_tube_centre = np.sqrt((rho - R) ** 2 + points[:, 2] ** 2)
    assert np.allclose(dist_from_tube_centre, r, atol=1e-10)


def test_canonical_sample_ball_points_lie_inside_unit_ball_at_requested_dimension():
    rng = np.random.default_rng(FIXTURE_SEED)
    points, tangent = ti.canonical_sample("ball", 500, rng, d=7)
    assert points.shape == (500, 7) and points.dtype == np.float64
    norms = np.linalg.norm(points, axis=1)
    assert np.all(norms <= 1.0 + 1e-12)
    assert tangent.shape == (500, 7, 7)
    assert np.allclose(tangent[0], np.eye(7), atol=1e-12)


def test_canonical_sample_ball_requires_d():
    rng = np.random.default_rng(FIXTURE_SEED)
    with pytest.raises(ValueError):
        ti.canonical_sample("ball", 10, rng)


def test_random_orthogonal_lift_is_orthonormal_at_production_dimension():
    rng = np.random.default_rng(FIXTURE_SEED)
    Q = ti.random_orthogonal_lift(2, AMBIENT_D, rng)
    assert Q.shape == (AMBIENT_D, 2) and Q.dtype == np.float64
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-12)


def test_immerse_returns_labelled_float64_cloud_at_production_dimension():
    cloud = ti.immerse(
        "T2",
        n=200,
        D=AMBIENT_D,
        noise=0.01,
        density=1.0,
        seed=0,
        warp_params={"strength": 0.1, "freq": 1.0},
    )
    assert cloud["points"].shape == (200, AMBIENT_D)
    assert cloud["points"].dtype == np.float64
    assert cloud["template"] == "T2"
    assert cloud["d_true"] == 2
    for key in ("seed", "noise", "density", "warp_params"):
        assert key in cloud


def test_immerse_no_notimplementederror_survives_for_any_template():
    for template, extra in (("S1", {}), ("S2", {}), ("T2", {}), ("ball", {"d": 5})):
        cloud = ti.immerse(
            template,
            n=40,
            D=64,
            noise=0.0,
            density=1.0,
            seed=1,
            warp_params={"strength": 0.05, "freq": 1.0},
            **extra,
        )
        assert cloud["points"].shape == (40, 64)


# --- Task 2: the Jacobian rank check, at sampled points, at D = 768 -----------------------


def test_jacobian_rank_s2_known_good_immersion_at_production_dimension():
    cloud = ti.immerse(
        "S2",
        n=120,
        D=AMBIENT_D,
        noise=0.0,
        density=1.0,
        seed=1,
        warp_params={"strength": 0.05, "freq": 1.0},
    )
    result = ti.jacobian_rank(cloud, n_check=20, rank_tol=1e-6)
    assert result["ranks"].shape == (20,)
    assert result["min_rank"] == 2
    assert result["expected_rank"] == 2
    assert result["is_immersion"] is True


def test_jacobian_rank_t2_known_good_immersion_at_production_dimension():
    cloud = ti.immerse(
        "T2",
        n=120,
        D=AMBIENT_D,
        noise=0.0,
        density=1.0,
        seed=2,
        warp_params={"strength": 0.05, "freq": 1.0},
    )
    result = ti.jacobian_rank(cloud, n_check=20, rank_tol=1e-6)
    assert result["min_rank"] == 2
    assert result["is_immersion"] is True


def test_jacobian_rank_deliberately_collapsed_lift_is_caught():
    cloud = ti.immerse(
        "S2",
        n=60,
        D=AMBIENT_D,
        noise=0.0,
        density=1.0,
        seed=3,
        warp_params={"strength": 0.05, "freq": 1.0},
    )
    collapsed = dict(cloud)
    bad_lift = np.array(cloud["lift"], copy=True)
    bad_lift[:, 1:] = 0.0
    collapsed["lift"] = bad_lift
    result = ti.jacobian_rank(collapsed, n_check=10, rank_tol=1e-6)
    assert result["min_rank"] < result["expected_rank"]
    assert result["is_immersion"] is False


def test_jacobian_rank_n_check_and_rank_tol_carry_no_default():
    p = inspect.signature(ti.jacobian_rank).parameters
    assert p["n_check"].default is inspect.Parameter.empty
    assert p["rank_tol"].default is inspect.Parameter.empty


def test_immerse_check_immersion_attaches_result_and_ground_truth_valid():
    cloud = ti.immerse(
        "S2",
        n=120,
        D=AMBIENT_D,
        noise=0.0,
        density=1.0,
        seed=1,
        warp_params={"strength": 0.05, "freq": 1.0},
        check_immersion=True,
    )
    assert cloud["immersion_check"]["is_immersion"] is True
    assert cloud["ground_truth_valid"] is True
