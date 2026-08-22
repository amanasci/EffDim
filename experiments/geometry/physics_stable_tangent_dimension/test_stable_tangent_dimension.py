"""Unit tests for stable tangent dimension geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric
from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    metric_scalars,
    monte_carlo_K_dir2,
)
from geometry.physics_stable_tangent_dimension.curvature_panel import (
    flatten_sym2,
    pack_from_B,
    verify_kdir_identity,
)
from geometry.physics_stable_tangent_dimension.dimension import dT_from_rank_flags
from geometry.physics_stable_tangent_dimension.nested_pca import (
    block_agreement,
    crossfit_risk,
    flip_signs,
    nested_uncentred_svd,
    prefix_agreement,
    reconstruction_risk,
    rotate_block,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import (
    parallel_transport_yx,
    projected_chord,
    sphere_exp_map,
    sphere_log_map,
    sphere_normal_apply,
)
from geometry.physics_stable_tangent_dimension.synthetics import make_synthetic


def test_log_map_small_angle_and_inverse():
    rng = np.random.default_rng(0)
    x = rng.normal(size=24)
    x /= np.linalg.norm(x)
    v = rng.normal(size=24)
    v = v - np.dot(v, x) * x
    v *= 1e-7 / np.linalg.norm(v)
    y = sphere_exp_map(x, v)
    z = sphere_log_map(x, y)
    assert np.linalg.norm(z - v) < 1e-12
    v2 = v * (0.4 / 1e-7)
    y2 = sphere_exp_map(x, v2)
    z2 = sphere_log_map(x, y2)
    assert np.linalg.norm(z2 - v2) < 1e-8
    assert abs(np.dot(z2, x)) < 1e-8


def test_log_map_batch_matches_scalar():
    rng = np.random.default_rng(1)
    x = rng.normal(size=16)
    x /= np.linalg.norm(x)
    Y = rng.normal(size=(7, 16))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Z = sphere_log_map(x, Y)
    for i in range(7):
        assert np.allclose(Z[i], sphere_log_map(x, Y[i]), atol=1e-10)


def test_parallel_transport_stays_tangent_preserves_norm():
    rng = np.random.default_rng(2)
    y = rng.normal(size=32)
    y /= np.linalg.norm(y)
    v = rng.normal(size=32)
    v = v - np.dot(v, y) * y
    v *= 1.1 / np.linalg.norm(v)
    w = rng.normal(size=32)
    w = w - np.dot(w, y) * y
    w *= 0.35 / np.linalg.norm(w)
    x = sphere_exp_map(y, w)
    vx = parallel_transport_yx(y, x, v)
    assert abs(np.dot(vx, x)) < 1e-8
    assert abs(np.linalg.norm(vx) - np.linalg.norm(v)) < 1e-8
    v2 = parallel_transport_yx(x, y, vx)
    assert np.linalg.norm(v2 - v) < 1e-7


def test_pt_geodesic_subsphere():
    rng = np.random.default_rng(3)
    D, d = 40, 5
    basis = np.eye(D)[:, : d + 1]
    a = rng.normal(size=d + 1)
    a /= np.linalg.norm(a)
    b = rng.normal(size=d + 1)
    b /= np.linalg.norm(b)
    y = basis @ a
    x = basis @ b
    t = rng.normal(size=d + 1)
    t = t - np.dot(t, a) * a
    v = basis @ t
    vx = parallel_transport_yx(y, x, v)
    rec = basis @ (basis.T @ vx)
    assert np.linalg.norm(vx - rec) < 1e-8


def test_tangent_and_normal_projectors():
    rng = np.random.default_rng(4)
    D, d = 20, 6
    x = rng.normal(size=D)
    x /= np.linalg.norm(x)
    J = rng.normal(size=(D, d))
    J = J - np.outer(x, x @ J)
    J, _ = np.linalg.qr(J)
    J = J[:, :d]
    v = rng.normal(size=D)
    Pt = J @ (J.T @ v)
    Pn = sphere_normal_apply(v, x, J)
    Pr = np.dot(v, x) * x
    assert np.linalg.norm(Pt + Pn + Pr - v) < 1e-8
    assert abs(np.dot(Pn, x)) < 1e-8
    assert np.linalg.norm(J.T @ Pn) < 1e-8


def test_nested_pca_equals_independent_rank_d():
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(80, 30))
    J20, ev20 = nested_uncentred_svd(Z, 20)
    for d in (4, 8, 12):
        Jd, evd = nested_uncentred_svd(Z, d)
        M = J20[:, :d].T @ Jd
        assert abs(np.sum(M * M) / d - 1.0) < 1e-8
        assert np.allclose(ev20[:d], evd[:d], rtol=1e-6, atol=1e-8)


def test_sign_and_block_rotation_invariance():
    rng = np.random.default_rng(6)
    Z = rng.normal(size=(60, 25))
    JA, _ = nested_uncentred_svd(Z[:30], 8)
    JB, _ = nested_uncentred_svd(Z[30:], 8)
    a0 = prefix_agreement(JA, JB, 8)
    signs = rng.choice([-1.0, 1.0], size=8)
    assert abs(prefix_agreement(flip_signs(JA, signs), JB, 8) - a0) < 1e-10
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    JA_r = rotate_block(JA, 2, 5, Q)
    assert abs(prefix_agreement(JA_r, JB, 8) - a0) < 1e-10
    b0 = block_agreement(JA, JA, 2, 5)
    b1 = block_agreement(JA, JA_r, 2, 5)
    assert abs(b0 - 1.0) < 1e-10
    assert abs(b1 - 1.0) < 1e-8


def test_ambient_orthogonal_invariance():
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(50, 18))
    Q, _ = np.linalg.qr(rng.normal(size=(18, 18)))
    J0, ev0 = nested_uncentred_svd(Z, 6)
    J1, ev1 = nested_uncentred_svd(Z @ Q, 6)
    assert np.allclose(ev0, ev1, rtol=1e-6, atol=1e-8)
    r0 = reconstruction_risk(Z, J0, 4)
    r1 = reconstruction_risk(Z @ Q, J1, 4)
    assert abs(r0 - r1) / max(r0, 1e-12) < 1e-6


def test_crossfit_risk_decreases_and_matches_formula():
    rng = np.random.default_rng(8)
    Z = rng.normal(size=(40, 12))
    JA, _ = nested_uncentred_svd(Z[:20], 5)
    JB, _ = nested_uncentred_svd(Z[20:], 5)
    R0 = crossfit_risk(Z[:20], Z[20:], JA, JB, 0)
    R1 = crossfit_risk(Z[:20], Z[20:], JA, JB, 1)
    R5 = crossfit_risk(Z[:20], Z[20:], JA, JB, 5)
    assert R0 >= R1 - 1e-10
    assert R1 >= R5 - 1e-10
    e = 0.5 * (np.mean(np.sum(Z[:20] ** 2, 1)) + np.mean(np.sum(Z[20:] ** 2, 1)))
    assert abs(R0 - e) / e < 1e-10


def test_sym2_sqrt2_preserves_frobenius():
    rng = np.random.default_rng(9)
    D, d = 10, 5
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    M = flatten_sym2(B)
    assert abs(np.linalg.norm(M) - np.linalg.norm(B)) < 1e-10


def _pack_BS(B: np.ndarray) -> np.ndarray:
    cols = []
    d = B.shape[1]
    for a in range(d):
        for b in range(a, d):
            cols.append(B[:, a, a] if a == b else (2.0 * B[:, a, b]))
    return np.stack(cols, axis=1)


def test_kdir_identity_and_mc():
    rng = np.random.default_rng(10)
    d, D = 5, 18
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    flat = _pack_BS(B)
    s = metric_scalars(flat, d)
    assert abs(s["K_dir2"] - (s["K_H2"] + s["K_aniso2"])) < 1e-10
    chk = verify_kdir_identity(flat, d, seed=0)
    assert chk["identity_err"] < 1e-10
    mc = monte_carlo_K_dir2(flat, d, n_dir=8000, seed=1)
    assert abs(mc - s["K_dir2"]) / max(s["K_dir2"], 1e-8) < 0.15


def test_unit_sphere_zero_sphere_normal():
    pack = make_synthetic("unit_sphere_baseline", n=200, D=32, seed=11, k_obs=80)
    X = pack["X"]
    x0 = pack["x0"]
    Z = sphere_log_map(x0, X[pack["neigh"]])
    J, _ev = nested_uncentred_svd(Z, 12)
    r = reconstruction_risk(Z, J, 12)
    tot = float(np.mean(np.sum(Z * Z, 1)))
    assert r / max(tot, 1e-12) < 0.15


def test_saddle_zero_mean_positive_dir():
    d, D = 4, 16
    B = np.zeros((D, d, d))
    B[0, 0, 0] = 1.0
    B[0, 1, 1] = -1.0
    flat = _pack_BS(B)
    s = metric_scalars(flat, d)
    assert abs(s["K_H"]) < 1e-12
    assert s["K_dir"] > 0.2


def test_split_cross_debias():
    rng = np.random.default_rng(12)
    true = rng.normal(size=30)
    naives, crosses = [], []
    for _ in range(40):
        a = true + rng.normal(size=30) * 0.4
        b = true + rng.normal(size=30) * 0.4
        naives.append(0.5 * (np.dot(a, a) + np.dot(b, b)))
        crosses.append(np.dot(a, b))
    err_n = abs(np.mean(naives) - np.dot(true, true))
    err_c = abs(np.mean(crosses) - np.dot(true, true))
    assert err_c < err_n


def test_dT_consecutive_prefix():
    flags = np.array([True, True, True, False, True])
    assert dT_from_rank_flags(flags) == 3


def test_chord_vs_log_small_angle():
    rng = np.random.default_rng(13)
    x = rng.normal(size=12)
    x /= np.linalg.norm(x)
    v = rng.normal(size=12)
    v = v - np.dot(v, x) * x
    v *= 1e-4 / np.linalg.norm(v)
    y = sphere_exp_map(x, v)
    zlog = sphere_log_map(x, y)
    zch = projected_chord(x, y)
    assert np.linalg.norm(zlog - zch) / np.linalg.norm(zlog) < 0.05


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(14)
    B = rng.normal(size=(8, 6, 6))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    B2 = unpack_BS_symmetric(pack_from_B(B), 6)
    assert np.allclose(B, B2, atol=1e-10)


def test_deterministic_resume_helper(tmp_path):
    from geometry.physics_stable_tangent_dimension.pipeline import _done

    p = tmp_path / "marker.json"
    assert _done(p, False) is False
    p.write_text("{}")
    assert _done(p, False) is True
    assert _done(p, True) is False


def test_linear_synth_recovers_core_prefix():
    from geometry.physics_stable_tangent_dimension.dimension import DEFAULT_THRESHOLDS
    from geometry.physics_stable_tangent_dimension.pipeline import _eval_synth_one, StableTangentConfig

    cfg = StableTangentConfig(smoke=True, device="cpu")
    row = _eval_synth_one("linear_d12", 1000, cfg, dict(DEFAULT_THRESHOLDS), device=None)
    assert row["median_dT"] >= 8, row
    assert row["p_ge_16"] == 0.0 or row["median_dT"] <= 14


def test_isotropic_inc_agreement_small():
    from geometry.physics_stable_tangent_dimension.nulls import residual_isotropic_null
    from geometry.physics_stable_tangent_dimension.nested_pca import nested_uncentred_svd

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(80, 40))
    J, _ = nested_uncentred_svd(Z, 4)
    iso = residual_isotropic_null(Z, J, rng=rng, n_draw=20, d_extra=1, device=None)
    assert np.nanmedian(iso["agreement_inc"]) < 0.35


def test_resolve_k_grid_not_hardcoded_below_pack():
    from geometry.physics_stable_tangent_dimension.pipeline import resolve_k_grid

    g = resolve_k_grid(2048, [1024, 2048], smoke=False)
    assert 2048 in g and 1024 in g
    assert len(g) >= 5
    gs = resolve_k_grid(256, [256], smoke=True)
    assert max(gs) <= 256
    assert len(gs) >= 4


def test_rank_flags_straddle_requires_scaling_for_extras():
    """A single bulk 0–19 block must not become d_T=20 without tangent-like extras."""
    from geometry.physics_stable_tangent_dimension.dimension import DEFAULT_THRESHOLDS, dT_from_rank_flags
    from geometry.physics_stable_tangent_dimension.pipeline import _rank_flags_for_anchor

    d_max = 20
    # slowly decaying spectrum: one block under rel_gap_min=0.15
    ev = np.linspace(1.0, 0.4, d_max)
    rows = []
    for d in range(1, d_max + 1):
        rows.append({"split": -1, "d": d, "ev_full": float(ev[d - 1]), "A": np.nan, "G": np.nan, "ev": np.nan})
        rows.append({"split": 0, "d": d, "ev_full": np.nan, "A": 0.95, "G": 0.05, "ev": float(ev[d - 1])})
    g = pd.DataFrame(rows)
    thr = dict(DEFAULT_THRESHOLDS)
    flags = _rank_flags_for_anchor(g, pd.DataFrame(), thr, d_max)
    assert int(dT_from_rank_flags(flags)) == 12


def test_frozen_primary_gates_linear_d12():
    from geometry.physics_stable_tangent_dimension.dimension import DEFAULT_THRESHOLDS
    from geometry.physics_stable_tangent_dimension.pipeline import _dT_frozen_primary_gates
    from geometry.physics_stable_tangent_dimension.synthetics import make_synthetic

    pack = make_synthetic("linear_d12", n=400, D=48, seed=7, k_obs=256)
    dT = _dT_frozen_primary_gates(
        pack["x0"],
        pack["X"][pack["neigh"][:256]],
        [64, 96, 128, 192, 256],
        256,
        16,
        7,
        dict(DEFAULT_THRESHOLDS),
        None,
        "log",
    )
    assert dT is not None
    assert 8 <= dT <= 14, dT
