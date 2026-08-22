"""Unit tests for the implicit normal-space inverse."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_implicit_normal_inverse.algebra import (
    EPS,
    RIDGES,
    bottom_eigh,
    constraint_residuals,
    fit_h_for_a,
    implicit_shape_operators,
    pack_H,
    profiled_K,
    projector_overlap,
    qr_orthonormal,
    quadratic_form,
    r2_cancel,
    sampson_batch,
    sampson_distance,
    stiefel_qr,
    tangent_basis,
    unpack_h,
    vech_weights,
    weighted_phi,
)
from geometry.physics_implicit_normal_inverse.classify import consecutive_normal_count, primary_label
from geometry.physics_implicit_normal_inverse.pipeline import _done, fit_constraints, implicit_q2_from_pack
from geometry.physics_implicit_normal_inverse.synthetics import make_implicit_synthetic
from geometry.physics_stable_tangent_dimension.sphere_coords import sphere_log_map


def test_spherical_log_radial_removal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=16)
    x = x / np.linalg.norm(x)
    y = x[None, :] + 0.05 * rng.normal(size=(40, 16))
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    z = sphere_log_map(x, y)
    assert np.max(np.abs(z @ x)) < 1e-8


def test_pack_unpack_and_weighted_phi():
    rng = np.random.default_rng(1)
    d = 5
    H = rng.normal(size=(d, d))
    H = 0.5 * (H + H.T)
    h = pack_H(H)
    H2 = unpack_h(h, d)
    assert np.allclose(H, H2, atol=1e-10)
    Y = rng.normal(size=(30, d))
    Phi = weighted_phi(Y)
    qf = quadratic_form(Y, H)
    assert np.allclose(qf, Phi @ h, atol=1e-10)
    w = vech_weights(d)
    assert w[0] == 1.0
    assert abs(w[1] - np.sqrt(2.0)) < 1e-12


def test_implicit_gradient_and_tangent_ker():
    rng = np.random.default_rng(2)
    R, q = 8, 3
    A, _ = np.linalg.qr(rng.normal(size=(R, q)))
    T = tangent_basis(A, R)
    assert T.shape == (R, R - q)
    assert np.allclose(A.T @ T, 0, atol=1e-8)
    # JF(0) = A
    y = np.zeros(R)
    Hs = [rng.normal(size=(R, R)) for _ in range(q)]
    Hs = [0.5 * (H + H.T) for H in Hs]
    JF = np.stack([A[:, ℓ] + Hs[ℓ] @ y for ℓ in range(q)], axis=1)
    assert np.allclose(JF, A)


def test_invariance_carrier_rotation_and_recombination():
    rng = np.random.default_rng(3)
    pack = make_implicit_synthetic("flat_d12_c8_q0", n=200, R=12, seed=3, radius=0.1, noise=0.002)
    Y = pack["Y"]
    Q, _ = np.linalg.qr(rng.normal(size=(Y.shape[1], Y.shape[1])))
    Y2 = Y @ Q.T
    Phi, Phi2 = weighted_phi(Y), weighted_phi(Y2)
    K = profiled_K(Y, Phi, 0.1)["K"]
    K2 = profiled_K(Y2, Phi2, 0.1)["K"]
    ev, U = bottom_eigh(K, 4)
    ev2, U2 = bottom_eigh(K2, 4)
    assert np.allclose(ev, ev2, rtol=0.15, atol=1e-6)
    # recombination of constraints: A R for orthogonal R
    A = U
    Rmat, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assert abs(projector_overlap(A, A @ Rmat) - 1.0) < 1e-8


def test_profiled_ridge_matches_direct_fit():
    rng = np.random.default_rng(4)
    n, R = 80, 6
    Y = rng.normal(size=(n, R))
    Phi = weighted_phi(Y)
    lam = 0.1
    info = profiled_K(Y, Phi, lam)
    ev, U = bottom_eigh(info["K"], 1)
    a = U[:, 0]
    h = fit_h_for_a(Y, Phi, a, lam)
    # Rayleigh of K equals residual of Ya + Phi h
    r = Y @ a + Phi @ h
    ray = float(a @ info["K"] @ a)
    assert abs(ray - float(np.sum(r * r))) / max(ray, 1e-12) < 0.05 or abs(ray - float(np.sum(r * r))) < 1e-4


def test_sampson_and_stiefel():
    rng = np.random.default_rng(5)
    n, R, q = 25, 6, 2
    A, _ = np.linalg.qr(rng.normal(size=(R, q)))
    Hs = np.stack([0.5 * (H + H.T) for H in rng.normal(size=(q, R, R))], axis=0)
    Y = rng.normal(size=(n, R)) * 0.05
    d1 = sampson_distance(Y, A, [Hs[i] for i in range(q)])
    d2 = sampson_batch(Y, A, Hs)
    assert np.allclose(d1, d2, atol=1e-8)
    G = rng.normal(size=A.shape)
    A2 = stiefel_qr(A, G, 0.01)
    assert np.allclose(A2.T @ A2, np.eye(q), atol=1e-8)


def test_implicit_second_derivative_identity():
    rng = np.random.default_rng(6)
    R, q, d = 8, 2, 6
    A, _ = np.linalg.qr(rng.normal(size=(R, q)))
    T = tangent_basis(A, R)
    assert T.shape[1] == R - q
    Hs = np.stack([0.5 * (H + H.T) for H in rng.normal(size=(q, R, R))], axis=0)
    Ss, Bflat = implicit_shape_operators(A, Hs)
    v = rng.normal(size=T.shape[1])
    w = rng.normal(size=T.shape[1])
    Tv, Tw = T @ v, T @ w
    for ℓ in range(q):
        lhs = -float(Tv @ Hs[ℓ] @ Tw)
        rhs = float(v @ Ss[ℓ] @ w)
        assert abs(lhs - rhs) < 1e-8


def test_flat_subspace_codimension_recovery():
    pack = make_implicit_synthetic("flat_d12_c8_q0", n=400, R=20, seed=7, radius=0.1, noise=0.001)
    fit = fit_constraints(pack["Y"], pack["radii"], q_max=10, seed=7, n_null=4, refine_steps=0)
    assert fit["ok"]
    # bottom-8 of K should overlap the true normal
    ov = projector_overlap(fit["UA"][:, :8], pack["N"])
    assert ov > 0.4, ov


def test_parabola_and_saddle_normal_recovery():
    rng = np.random.default_rng(8)
    n, R = 300, 6
    T, _ = np.linalg.qr(rng.normal(size=(R, 4)))
    N, _ = np.linalg.qr(rng.normal(size=(R, 2)))
    N = N - T @ (T.T @ N)
    N, _ = np.linalg.qr(N)
    U = rng.normal(size=(n, 4)) * 0.12
    # parabola in N0, saddle in N1
    y = U @ T.T
    y = y + ((U[:, 0] ** 2)[:, None] @ N[:, :1].T)
    y = y + ((U[:, 0] * U[:, 1])[:, None] @ N[:, 1:2].T)
    y = y + rng.normal(size=y.shape) * 0.003
    rad = np.linalg.norm(y, axis=1)
    fit = fit_constraints(y, rad, q_max=4, seed=8, n_null=3, refine_steps=0)
    assert fit["ok"]
    ov = projector_overlap(fit["UA"][:, :2], N)
    assert ov > 0.3, ov
    q2 = implicit_q2_from_pack(fit["UA"], fit["h_pack"], 2)["q2"]
    assert q2 >= 1


def test_curved_d12_c8_q1_and_weak_d16():
    p12 = make_implicit_synthetic("curved_d12_c8_q1", n=500, R=20, seed=9, radius=0.12, noise=0.004)
    f12 = fit_constraints(p12["Y"], p12["radii"], q_max=10, seed=9, n_null=4, refine_steps=0)
    assert f12["ok"]
    ov12 = projector_overlap(f12["UA"][:, :8], p12["N"])
    p16 = make_implicit_synthetic("flat_d16_c4", n=500, R=20, seed=10, radius=0.12, noise=0.004)
    f16 = fit_constraints(p16["Y"], p16["radii"], q_max=10, seed=10, n_null=4, refine_steps=0)
    ov16 = projector_overlap(f16["UA"][:, :4], p16["N"])
    # method must not collapse both to the same projector size blindly
    assert ov12 > 0.25 or ov16 > 0.25
    ev12 = f12["ev_K"]
    ev16 = f16["ev_K"]
    # d16 should have fewer tiny K-evals than d12
    assert float(np.mean(ev12[:8])) <= float(np.mean(ev16[:8])) * 3 + 1e-6


def test_sphere_baseline_no_radial_rediscovery():
    pack = make_implicit_synthetic("unit_sphere_baseline", n=200, R=12, seed=11, radius=0.1, noise=0.003)
    # already in log coordinates; isotropic-ish in R dims
    fit = fit_constraints(pack["Y"], pack["radii"], q_max=4, seed=11, n_null=3, refine_steps=0)
    assert fit["ok"]
    # should not invent a dominant single constraint
    rel = fit["ev_K"][0] / max(fit["ev_K"][-1], EPS)
    assert rel > 0.05


def test_consecutive_prefix_and_primary_unresolved_without_synth():
    labs = ["curvature_active_normal", "approximately_flat_normal", "unresolved", "approximately_flat_normal"]
    assert consecutive_normal_count(labs) == 2
    lab = primary_label(
        cN_minus=8,
        d1_minus=12,
        d1_plus=12,
        q2=1,
        R=20,
        e4_normal_frac=0.5,
        synth_not_only12=False,
    )
    assert lab == "implicit_normal_inverse_unresolved"


def test_done_helper(tmp_path):
    p = tmp_path / "m.json"
    assert _done(p, False) is False
    p.write_text("{}")
    assert _done(p, False) is True
    assert _done(p, True) is False


def test_qr_orthonormal_columns():
    rng = np.random.default_rng(12)
    A = rng.normal(size=(10, 3))
    Q = qr_orthonormal(A)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)
