"""Unit tests for sphere parallel transport and Gauss-map helpers."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.gauss_map_curvature import (
    curvature_energy_from_beta,
    debiased_delta,
    parallel_transport_basis,
    parallel_transport_sphere,
    regress_gauss_map,
    sphere_exp,
    sphere_geodesic_distance,
    sphere_log,
)
from geometry.physics_activation_atlas.sphere_normal_quadratic import sphere_project_basis


def test_transport_stays_tangent_and_preserves_norm():
    rng = np.random.default_rng(0)
    D = 32
    p = rng.normal(size=D)
    p /= np.linalg.norm(p)
    # nearby q
    v = rng.normal(size=D)
    v = v - np.dot(v, p) * p
    v *= 0.3 / np.linalg.norm(v)
    q = sphere_exp(p, v)
    # random tangent at p
    t = rng.normal(size=D)
    t = t - np.dot(t, p) * p
    t *= 1.2 / np.linalg.norm(t)
    tq = parallel_transport_sphere(p, q, t)
    assert abs(np.dot(tq, q)) < 1e-8
    assert abs(np.linalg.norm(tq) - np.linalg.norm(t)) < 1e-8


def test_round_trip_transport():
    rng = np.random.default_rng(1)
    D = 24
    p = rng.normal(size=D)
    p /= np.linalg.norm(p)
    w = rng.normal(size=D)
    w = w - np.dot(w, p) * p
    w *= 0.4 / np.linalg.norm(w)
    q = sphere_exp(p, w)
    t = rng.normal(size=D)
    t = t - np.dot(t, p) * p
    tq = parallel_transport_sphere(p, q, t)
    t2 = parallel_transport_sphere(q, p, tq)
    assert np.linalg.norm(t2 - t) < 1e-7


def test_geodesic_subsphere_zero_gauss_dispersion():
    """Totally geodesic S^{d} ⊂ S^{D-1}: transported projectors agree."""
    rng = np.random.default_rng(2)
    D, d = 40, 5
    # fixed totally geodesic sphere: span{e0,...,ed} intersect unit sphere
    basis = np.eye(D)[:, : d + 1]
    # two points on that sphere
    a = rng.normal(size=d + 1)
    a /= np.linalg.norm(a)
    b = rng.normal(size=d + 1)
    b /= np.linalg.norm(b)
    p = basis @ a
    q = basis @ b
    # true tangent = orth complement of radial in the (d+1) plane
    Jp = sphere_project_basis(p, basis[:, 1:])
    Jq = sphere_project_basis(q, basis[:, 1:])
    Jq_at_p = parallel_transport_basis(q, p, Jq)
    # projectors should nearly match (same d-plane)
    Pp = Jp @ Jp.T
    Pq = Jq_at_p @ Jq_at_p.T
    assert np.linalg.norm(Pp - Pq, "fro") < 0.15


def test_log_exp_inverse():
    rng = np.random.default_rng(3)
    p = rng.normal(size=16)
    p /= np.linalg.norm(p)
    v = rng.normal(size=16)
    v = v - np.dot(v, p) * p
    v *= 0.5 / np.linalg.norm(v)
    q = sphere_exp(p, v)
    v2 = sphere_log(p, q)
    assert np.linalg.norm(v2 - v) < 1e-8
    assert sphere_geodesic_distance(p, q) == np.linalg.norm(v2)


def test_regress_and_debias():
    rng = np.random.default_rng(4)
    dS2 = rng.uniform(0.01, 0.2, size=40)
    beta_true = 2.0
    y = 0.1 + beta_true * dS2 + 0.01 * rng.normal(size=40)
    fit = regress_gauss_map(y, dS2)
    assert abs(fit["beta"] - beta_true) < 0.3
    assert curvature_energy_from_beta(fit["beta"], d=8) > 0
    assert abs(debiased_delta(1.0, 0.2, 0.2) - 0.8) < 1e-12
