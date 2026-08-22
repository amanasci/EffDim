"""Unit tests for quadratic predictive dimension."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_quadratic_predictive_dimension.algebra import (
    closest_point_project,
    jacobian_batch,
    n_quad_features,
    nmse,
    phi2,
    predict_f,
    project_B_normal,
    ridge_df,
    ridge_fit,
    scale_phi_train,
    unpack_B_to_H,
)
from geometry.physics_quadratic_predictive_dimension.classify import (
    DEFAULT_THRESHOLDS,
    adequacy_ranks,
    plateau_from_curve,
    primary_label,
)
from geometry.physics_quadratic_predictive_dimension.pipeline import (
    PRESERVED,
    SOURCE_INI,
    SOURCE_OSG,
    SOURCE_STD,
    QuadPredConfig,
    _done,
    assert_not_preserved,
    fit_neighbourhood,
    write_df,
)
from geometry.physics_quadratic_predictive_dimension.synthetics import make_predictive_synthetic
from geometry.physics_stable_tangent_dimension.nested_pca import nested_uncentred_svd


def test_phi2_count_and_frobenius():
    rng = np.random.default_rng(0)
    d = 4
    U = rng.normal(size=(20, d))
    Phi = phi2(U)
    assert Phi.shape == (20, n_quad_features(d))
    # <H, uu^T>_F = h · phi: diagonal H_aa * u_a^2, off-diag 2 H_ab u_a u_b
    # = H_aa u_a^2 + sqrt(2) H_ab * sqrt(2) u_a u_b
    H = rng.normal(size=(d, d))
    H = 0.5 * (H + H.T)
    vech = []
    for a in range(d):
        for b in range(a, d):
            vech.append(H[a, b] if a == b else np.sqrt(2.0) * H[a, b])
    vech = np.asarray(vech)
    qf = np.einsum("na,ab,nb->n", U, H, U)
    assert np.allclose(qf, Phi @ vech, atol=1e-10)


def test_scale_phi_no_mean_keeps_origin():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(40, 3))
    Phi, rms = scale_phi_train(phi2(U))
    z0 = phi2(np.zeros((1, 3))) / rms[None, :]
    assert np.allclose(z0, 0.0)
    assert np.all(rms > 0)


def test_f0_and_Df0():
    rng = np.random.default_rng(2)
    d, D = 3, 8
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    B = rng.normal(size=(D, n_quad_features(d))) * 0.1
    z0 = predict_f(np.zeros((1, d)), J, B)
    assert np.allclose(z0, 0.0, atol=1e-12)
    H = unpack_B_to_H(B, d)
    Jf = jacobian_batch(np.zeros((1, d)), J, H)
    assert np.allclose(Jf[0], J, atol=1e-12)


def test_ridge_df_orthogonal():
    p = 5
    s = np.array([4.0, 3.0, 2.0, 1.0, 0.5])
    G = np.diag(s)
    lam = 1.0
    df = ridge_df(G, lam)
    expect = float(np.sum(s / (s + lam)))
    assert abs(df - expect) < 1e-10


def test_ridge_fit_no_intercept():
    rng = np.random.default_rng(3)
    Phi = rng.normal(size=(30, 4))
    Btrue = rng.normal(size=(6, 4))
    Y = Phi @ Btrue.T
    Bhat = ridge_fit(Phi, Y, 1e-8)
    assert Bhat.shape == (6, 4)
    assert np.allclose(Bhat, Btrue, atol=1e-4)


def test_closest_point_linear_is_projection():
    rng = np.random.default_rng(4)
    d, D, n = 3, 7, 25
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    U = rng.normal(size=(n, d)) * 0.1
    Z = U @ J.T + 0.02 * rng.normal(size=(n, D))
    B = np.zeros((D, n_quad_features(d)))
    u0 = Z @ J
    pack = closest_point_project(Z, J, B, u0, u_max=10.0, max_iter=5)
    assert np.allclose(pack["U"], u0, atol=1e-8)
    assert pack["close_nmse"] <= pack["fixed_nmse"] + 1e-12


def test_closest_point_parabola_recovers_on_surface():
    # f(u) = (u, 0.5 u^2) in R^2
    d = 1
    J = np.array([[1.0], [0.0]])
    B = np.array([[0.0], [0.5]])  # phi = u^2
    u_true = np.array([[0.3], [-0.2], [0.15]])
    Z = predict_f(u_true, J, B)
    u0 = Z @ J
    pack = closest_point_project(Z, J, B, u0, u_max=2.0, max_iter=12, damp=1e-8)
    assert np.allclose(pack["U"], u_true, atol=1e-4)
    assert pack["close_nmse"] < 1e-8
    assert np.all(pack["improved"])


def test_closest_point_never_worsens_fixed_coord():
    rng = np.random.default_rng(5)
    d, D, n = 4, 12, 40
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    B = rng.normal(size=(D, n_quad_features(d))) * 0.2
    U = rng.normal(size=(n, d)) * 0.08
    Z = predict_f(U, J, B) + 0.01 * rng.normal(size=(n, D))
    u0 = Z @ J
    pack = closest_point_project(Z, J, B, u0, u_max=0.5, max_iter=8)
    zfix = predict_f(u0, J, B)
    r_fix = np.sum((Z - zfix) ** 2, axis=1)
    r_cl = np.sum((Z - pack["Zhat"]) ** 2, axis=1)
    assert np.all(r_cl <= r_fix + 1e-9)
    assert pack["close_nmse"] <= pack["fixed_nmse"] + 1e-10


def test_normal_only_kills_tangential_quadratic():
    rng = np.random.default_rng(6)
    d, D = 3, 8
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    B = J @ rng.normal(size=(d, n_quad_features(d)))
    BN = project_B_normal(B, J)
    assert np.linalg.norm(J.T @ BN) < 1e-10


def test_cv_leakage_train_only_basis():
    rng = np.random.default_rng(7)
    n, D, d = 80, 16, 4
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    Z = rng.normal(size=(n, d)) @ T.T * 0.1
    # distinctive outliers that would dominate SVD if leaked into training
    Z[-8:] = 4.0 * rng.normal(size=(8, D))
    radii = np.linalg.norm(Z, axis=1)
    # force a split: small-radius train vs large-radius test-like
    order = np.argsort(radii)
    tr, te = order[:40], order[40:]
    Jtr, _ = nested_uncentred_svd(Z[tr], d, device=None, centre=False)
    Jall, _ = nested_uncentred_svd(Z, d, device=None, centre=False)
    # train-only J should align with planted T more than full-data J
    align_tr = float(np.linalg.norm(Jtr.T @ T, ord="fro"))
    align_all = float(np.linalg.norm(Jall.T @ T, ord="fro"))
    assert align_tr >= align_all - 0.25
    rows = fit_neighbourhood(
        Z,
        radii,
        np.zeros(D),
        ds=[4],
        thr=dict(DEFAULT_THRESHOLDS),
        seed=7,
        frozen_J=None,
        d_core=4,
        d_ref=4,
        R=4,
        n_inner_cp=16,
        device=None,
    )
    assert len(rows)
    # held-out NMSE must be computed; inner selection cannot be inf
    assert np.all(np.isfinite([r["quad_close_nmse"] for r in rows]))


def test_fit_neighbourhood_linear_vs_quad_on_parabola():
    pack = make_predictive_synthetic("curved_d8", n=240, D=20, seed=8, radius=0.12, noise=0.003)
    rows = fit_neighbourhood(
        pack["Z"],
        pack["radii"],
        np.zeros(pack["Z"].shape[1]),
        ds=[4, 8],
        thr=dict(DEFAULT_THRESHOLDS),
        seed=8,
        frozen_J=None,
        d_core=8,
        d_ref=8,
        R=8,
        n_inner_cp=32,
        device=None,
    )
    df = pd.DataFrame(rows)
    g = df.groupby("d").mean(numeric_only=True)
    assert g.loc[8].quad_close_nmse <= g.loc[8].lin_nmse + 0.02


def test_plateau_and_adequacy_gates():
    ds = np.arange(4, 21)
    nmse = np.exp(-0.4 * (ds - 4))
    nmse[ds >= 12] = nmse[ds == 12][0] * np.ones(np.sum(ds >= 12))
    plat = plateau_from_curve(ds, nmse, np.ones_like(ds, dtype=float) * 0.5, DEFAULT_THRESHOLDS)
    assert plat["d_plat"] <= 13
    r2 = 1.0 - nmse
    adeq = adequacy_ranks(ds, r2, DEFAULT_THRESHOLDS)
    assert adeq["d90"] != "not_reached"
    lab = primary_label(
        dQ=12,
        dL=16,
        d95="not_reached",
        r2_total=0.80,
        r2_E4=0.10,
        r2_U8=0.05,
        delta_Q_12_16=0.001,
        delta_L_12_16=0.02,
        synth_not_only12=True,
        scale_stable=True,
        thr=DEFAULT_THRESHOLDS,
    )
    assert lab == "quadratic_predictive_plateau_at_12_but_inadequate"
    lab2 = primary_label(
        dQ=12,
        dL=12,
        d95=12,
        r2_total=0.97,
        r2_E4=0.05,
        r2_U8=0.02,
        delta_Q_12_16=0.0,
        delta_L_12_16=0.0,
        synth_not_only12=True,
        scale_stable=True,
        thr=DEFAULT_THRESHOLDS,
    )
    assert lab2 == "high_total_low_tail_adequacy"


def test_done_and_preserved(tmp_path):
    p = tmp_path / "m.json"
    assert _done(p, False) is False
    p.write_text("{}")
    assert _done(p, False) is True
    assert _done(p, True) is False
    root = platonic_root()
    for rel in (SOURCE_STD, SOURCE_OSG, SOURCE_INI):
        dest = resolve_path(root, rel)
        try:
            assert_not_preserved(dest, root)
            raise AssertionError(f"should refuse {rel}")
        except RuntimeError:
            pass
    own = tmp_path / "physics_quadratic_predictive_dimension"
    own.mkdir()
    assert_not_preserved(own, root)


def test_write_df_supersede(tmp_path):
    dest = tmp_path / "a.csv"
    write_df(dest, pd.DataFrame({"x": [1]}), force=False)
    write_df(dest, pd.DataFrame({"x": [2]}), force=True)
    assert dest.exists()
    supers = list(tmp_path.glob("a.superseded.*.csv"))
    assert len(supers) == 1


def test_remove_radial_torch_has_torch():
    import torch as th

    x = th.tensor([1.0, 0.0, 0.0], dtype=th.float64)
    Z = th.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], dtype=th.float64)
    from geometry.physics_quadratic_predictive_dimension.algebra import _remove_radial_torch

    out = _remove_radial_torch(Z, x)
    assert th.allclose(out[:, 0], th.zeros(2, dtype=th.float64), atol=1e-12)


def test_thick_tangent_synthetic_shapes():
    pack = make_predictive_synthetic("thick_tangent_d12", n=40, D=20, seed=11)
    assert pack["Z"].shape == (40, 20)
    assert np.all(np.isfinite(pack["Z"]))
    names = " ".join(PRESERVED)
    assert "physics_stable_tangent_dimension" in names
    assert "physics_nested_dimension_curvature" in names
    assert "physics_order_stratified_geometry" in names
    assert "physics_implicit_normal_inverse" in names
