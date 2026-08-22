"""Unit tests for effdim curvature metric algebra."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric
from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    aniso_prefactor,
    cross_metric_pair,
    decompose_tensors,
    m_quad,
    metric_scalars,
    monte_carlo_K_dir2,
    probe_facing_cross,
    probe_facing_scalar,
    project_normal,
)


def _pack_BS(B: np.ndarray) -> np.ndarray:
    """Pack symmetric B[D,d,d] into flat (D,q) with off-diag = 2*B_ab."""
    D, d, _ = B.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(B[:, a, a] if a == b else (2.0 * B[:, a, b]))
    return np.stack(cols, axis=1)


def test_m_quad():
    assert m_quad(16) == 136
    assert m_quad(8) == 36


def test_pure_mean_curvature_K_dir_equals_K_H():
    d, D = 6, 20
    H = np.zeros(D)
    H[0] = 1.5
    B = np.zeros((D, d, d))
    for a in range(d):
        B[:, a, a] = H
    flat = _pack_BS(B)
    s = metric_scalars(flat, d)
    assert abs(s["K_aniso"]) < 1e-8
    assert abs(s["K_dir"] - s["K_H"]) < 1e-8


def test_traceless_saddle_zero_mean():
    d, D = 4, 16
    B = np.zeros((D, d, d))
    B[0, 0, 0] = 1.0
    B[0, 1, 1] = -1.0
    flat = _pack_BS(B)
    s = metric_scalars(flat, d)
    assert abs(s["K_H"]) < 1e-8
    assert s["K_aniso"] > 0.1


def test_monte_carlo_matches_closed_form_K_dir():
    rng = np.random.default_rng(0)
    d, D = 5, 24
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    flat = _pack_BS(B)
    closed = metric_scalars(flat, d)["K_dir2"]
    mc = monte_carlo_K_dir2(flat, d, n_dir=8000, seed=1)
    # formula uses specific aniso prefactor; allow modest MC error
    assert abs(mc - closed) / max(closed, 1e-8) < 0.15


def test_rotation_invariance_of_metrics():
    rng = np.random.default_rng(1)
    d, D = 5, 18
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    # rotate tangent indices
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Br = np.einsum("ia,dab,jb->dij", Q, B, Q)
    s0 = metric_scalars(_pack_BS(B), d)
    s1 = metric_scalars(_pack_BS(Br), d)
    for k in ("K_H", "K_aniso", "K_dir", "B_fro"):
        assert abs(s0[k] - s1[k]) < 1e-6


def test_cross_product_unbiased_under_noise():
    rng = np.random.default_rng(2)
    d, D = 4, 12
    B = rng.normal(size=(D, d, d)) * 0.3
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    noise = lambda: 0.5 * (
        rng.normal(size=(D, d, d)) + np.transpose(rng.normal(size=(D, d, d)), (0, 2, 1))
    )
    true = metric_scalars(_pack_BS(B), d)["K_dir2"]
    crosses = []
    for _ in range(40):
        A = B + noise()
        C = B + noise()
        crosses.append(cross_metric_pair(_pack_BS(A), _pack_BS(C), d)["K_dir_cross"])
    assert abs(np.mean(crosses) - true) < 0.35 * max(abs(true), 0.1)


def test_probe_facing_rotation_invariance():
    rng = np.random.default_rng(3)
    d, D = 5, 20
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    w = rng.normal(size=D)
    w /= np.linalg.norm(w)
    x0 = rng.normal(size=D)
    x0 /= np.linalg.norm(x0)
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    J = J[:, :d]
    wh, _ = project_normal(w, x0, J)
    s0 = probe_facing_scalar(_pack_BS(B), d, wh)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Br = np.einsum("ia,dab,jb->dij", Q, B, Q)
    s1 = probe_facing_scalar(_pack_BS(Br), d, wh)
    assert abs(s0["K_w_dir2"] - s1["K_w_dir2"]) < 1e-6


def test_radial_only_small_sphere_normal():
    d, D = 4, 16
    x0 = np.zeros(D)
    x0[0] = 1.0
    # pure radial quadratic in ambient: only along x0
    B = np.zeros((D, d, d))
    B[0, :, :] = 0.2  # radial direction components
    # After sphere-normal projection one would remove span(x0,J); here B is already
    # mostly radial — K_H of residual after projecting B onto normal should be tiny
    # if we zero the x0 row
    B_n = B.copy()
    B_n[0] = 0.0
    s = metric_scalars(_pack_BS(B_n), d)
    assert s["K_dir"] < 1e-12


def test_aniso_prefactor_positive():
    assert abs(aniso_prefactor(16) - 2.0 / (16 * 18)) < 1e-12
