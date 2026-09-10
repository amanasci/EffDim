"""Deterministic synthetics for the orthogonal component algebra and probes."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from geometry.physics_cross_model_full_curvature_reconciliation.metrics import pack_BS
from geometry.physics_quadratic_label_chart_alignment.features import (
    Gamma_from_gamma,
    gamma_from_Gamma,
    phi2_frob,
)

from .decompose import (
    alignment_AB,
    aniso_prefactor,
    bh_btf_frob,
    cross_components,
    iso_unit_gamma,
    project_gamma_iso_tf,
    split_B,
    split_Gamma,
    split_flat,
)
from geometry.physics_cross_model_full_curvature_reconciliation.metrics import aniso_prefactor as _ap


def _rand_sym_B(D: int, d: int, rng: np.random.Generator) -> np.ndarray:
    B = rng.normal(size=(D, d, d))
    return 0.5 * (B + np.transpose(B, (0, 2, 1)))


def test_pure_mean_geometry() -> None:
    d, D = 6, 18
    H = np.zeros(D)
    H[0] = 1.2
    B = np.zeros((D, d, d))
    for a in range(d):
        B[:, a, a] = H
    t = split_flat(pack_BS(B), d)
    assert t["K_H2"] > 1.0
    assert t["K_TF2"] < 1e-12
    assert abs(t["C_trace"] - 1.0) < 1e-10


def test_pure_saddle_geometry() -> None:
    d, D = 6, 18
    B = np.zeros((D, d, d))
    n = np.zeros(D)
    n[1] = 1.0
    B[:, 0, 0] = n
    B[:, 1, 1] = -n
    t = split_flat(pack_BS(B), d)
    assert t["K_H2"] < 1e-12
    assert t["K_TF2"] > 0.01
    assert t["C_trace"] < 1e-10


def test_identity_and_orthogonality() -> None:
    rng = np.random.default_rng(0)
    d, D = 5, 16
    A = _rand_sym_B(D, d, rng)
    C = _rand_sym_B(D, d, rng)
    sc = cross_components(pack_BS(A), pack_BS(C), d)
    a = split_flat(pack_BS(A), d)
    assert a["tr_TF_max"] < 1e-10
    assert abs(a["inner_BH_BTF"]) < 1e-10
    assert a["recon_err"] < 1e-12
    assert abs(sc["K_dir_cross"] - sc["K_H_cross"] - sc["K_TF_cross"]) < 1e-12


def test_fixed_total_varying_organization() -> None:
    d, D = 6, 14
    n = np.zeros(D)
    n[0] = 1.0
    dirs = []
    for frac in (0.05, 0.4, 0.9):
        H = n * np.sqrt(frac)
        B = np.zeros((D, d, d))
        for a in range(d):
            B[:, a, a] = H
        # add TF energy to hold K_dir2 ~ 1
        pref = aniso_prefactor(d)
        tf = np.sqrt((1.0 - frac) / pref)
        B[0, 0, 1] = B[0, 1, 0] = tf / np.sqrt(2)
        t = split_flat(pack_BS(B), d)
        dirs.append(t["K_H2"] + t["K_TF2"])
    assert max(dirs) - min(dirs) < 0.15


def test_conditional_mean_outcome() -> None:
    rng = np.random.default_rng(1)
    n = 400
    kh = rng.normal(size=n)
    ktf = 0.7 * kh + rng.normal(size=n)
    y = kh + 0.05 * rng.normal(size=n)
    df = pd.DataFrame(
        {
            "K_H_cross": kh,
            "K_TF_cross": ktf,
            "mse_G": y,
            "log_knn_radius": rng.normal(size=n),
            "local_label_variance": rng.normal(size=n) ** 2,
            "local_evaluation_count": np.full(n, 2048.0),
        }
    )
    from geometry.physics_curvature_probe_rank_sweep.inference import associate

    Ztf = np.column_stack(
        [df.log_knn_radius, df.local_label_variance, df.local_evaluation_count, df.K_TF_cross]
    )
    Zh = np.column_stack(
        [df.log_knn_radius, df.local_label_variance, df.local_evaluation_count, df.K_H_cross]
    )
    a_h = associate(df.K_H_cross.to_numpy(), df.mse_G.to_numpy(), Ztf)
    a_tf = associate(df.K_TF_cross.to_numpy(), df.mse_G.to_numpy(), Zh)
    assert a_h["controlled"] > 0.4
    assert abs(a_tf["controlled"]) < abs(a_h["controlled"])


def test_conditional_tf_outcome() -> None:
    rng = np.random.default_rng(2)
    n = 400
    ktf = rng.normal(size=n)
    kh = 0.7 * ktf + rng.normal(size=n)
    y = ktf + 0.05 * rng.normal(size=n)
    df = pd.DataFrame(
        {
            "K_H_cross": kh,
            "K_TF_cross": ktf,
            "mse_G": y,
            "log_knn_radius": rng.normal(size=n),
            "local_label_variance": rng.normal(size=n) ** 2,
            "local_evaluation_count": np.full(n, 2048.0),
        }
    )
    from geometry.physics_curvature_probe_rank_sweep.inference import associate

    Ztf = np.column_stack(
        [df.log_knn_radius, df.local_label_variance, df.local_evaluation_count, df.K_TF_cross]
    )
    Zh = np.column_stack(
        [df.log_knn_radius, df.local_label_variance, df.local_evaluation_count, df.K_H_cross]
    )
    a_h = associate(df.K_H_cross.to_numpy(), df.mse_G.to_numpy(), Ztf)
    a_tf = associate(df.K_TF_cross.to_numpy(), df.mse_G.to_numpy(), Zh)
    assert a_tf["controlled"] > 0.4
    assert abs(a_h["controlled"]) < abs(a_tf["controlled"])


def test_iso_quadratic_span() -> None:
    rng = np.random.default_rng(3)
    d, n = 6, 80
    U = rng.normal(size=(n, d))
    Phi = phi2_frob(U)
    e = iso_unit_gamma(d)
    e = e / np.linalg.norm(e)
    iso = Phi @ e
    tf = Phi - iso[:, None] * e[None, :]
    rec = iso[:, None] * e[None, :] + tf
    assert np.linalg.norm(rec - Phi) < 1e-10
    # isotropic label is in the iso column
    tau = 0.8
    G = (tau / d) * np.eye(d)
    y = phi2_frob(U) @ gamma_from_Gamma(G)
    # IQ feature
    feat = 0.5 * np.sum(U * U, axis=1)
    coef, *_ = np.linalg.lstsq(feat[:, None], y, rcond=None)
    pred = feat * coef[0]
    assert np.mean((y - pred) ** 2) < 1e-10


def test_tf_quadratic_label() -> None:
    rng = np.random.default_rng(4)
    d, n = 5, 100
    U = rng.normal(size=(n, d))
    G = np.zeros((d, d))
    G[0, 0] = 1.0
    G[1, 1] = -1.0
    y = phi2_frob(U) @ gamma_from_Gamma(G)
    feat = 0.5 * np.sum(U * U, axis=1)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), feat]), y, rcond=None)
    pred = coef[0] + coef[1] * feat
    assert np.mean((y - pred) ** 2) > 0.05
    gH, gTF = project_gamma_iso_tf(gamma_from_Gamma(G), d)
    assert np.linalg.norm(gH) < 1e-12
    assert np.linalg.norm(gTF) > 0.5


def test_alignment_iso_vs_tf() -> None:
    d, D = 6, 20
    H = np.zeros(D)
    H[0] = 1.0
    B = np.zeros((D, d, d))
    for a in range(d):
        B[:, a, a] = H
    from geometry.physics_quadratic_label_chart_alignment.features import bs_prod_to_frob

    BH = bs_prod_to_frob(pack_BS(B), d)
    BTF = np.zeros_like(BH)
    g = iso_unit_gamma(d)
    assert alignment_AB(g, BH) > 5.0
    # random TF gamma vs mean geometry
    G = np.zeros((d, d))
    G[0, 1] = G[1, 0] = 1.0
    gtf = gamma_from_Gamma(G)
    assert alignment_AB(gtf, BH) < 0.2


def test_mixed_quadratic_label() -> None:
    rng = np.random.default_rng(6)
    d, n = 6, 120
    U = rng.normal(size=(n, d))
    G = (0.4 / d) * np.eye(d)
    G[0, 1] = G[1, 0] = 0.35
    y = phi2_frob(U) @ gamma_from_Gamma(G)
    e = iso_unit_gamma(d)
    e = e / np.linalg.norm(e)
    Phi = phi2_frob(U)
    iso = Phi @ e
    tf = Phi - iso[:, None] * e[None, :]
    pred_iso = iso * np.linalg.lstsq(iso[:, None], y, rcond=None)[0][0]
    pred_tf = tf @ np.linalg.lstsq(tf, y, rcond=None)[0]
    pred_u = Phi @ np.linalg.lstsq(Phi, y, rcond=None)[0]
    mse_iso = float(np.mean((y - pred_iso) ** 2))
    mse_tf = float(np.mean((y - pred_tf) ** 2))
    mse_u = float(np.mean((y - pred_u) ** 2))
    assert mse_u < mse_iso and mse_u < mse_tf


def test_shuffled_no_gain_algebra() -> None:
    rng = np.random.default_rng(5)
    n, d = 120, 6
    U = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    Phi = np.column_stack([U, phi2_frob(U)])
    # ridge-ish least squares on half, evaluate other half
    tr, te = np.arange(80), np.arange(80, 120)
    w, *_ = np.linalg.lstsq(Phi[tr], y[tr], rcond=None)
    pred = Phi[te] @ w
    gain = float(np.mean((y[te] - y[tr].mean()) ** 2) - np.mean((y[te] - pred) ** 2))
    assert gain < 0.5


TESTS: list[tuple[str, Callable[[], None]]] = [
    ("pure_mean_geometry", test_pure_mean_geometry),
    ("pure_saddle_geometry", test_pure_saddle_geometry),
    ("identity_orthogonality", test_identity_and_orthogonality),
    ("fixed_total_varying_org", test_fixed_total_varying_organization),
    ("conditional_mean_outcome", test_conditional_mean_outcome),
    ("conditional_tf_outcome", test_conditional_tf_outcome),
    ("iso_quadratic_span", test_iso_quadratic_span),
    ("tf_quadratic_label", test_tf_quadratic_label),
    ("alignment_iso_vs_tf", test_alignment_iso_vs_tf),
    ("mixed_quadratic_label", test_mixed_quadratic_label),
    ("shuffled_no_gain_algebra", test_shuffled_no_gain_algebra),
]


def run_unit_tests() -> dict[str, Any]:
    rows = []
    for name, fn in TESTS:
        try:
            fn()
            rows.append({"name": name, "ok": True, "error": ""})
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"n": len(rows), "n_pass": int(sum(r["ok"] for r in rows)), "ok": all(r["ok"] for r in rows), "rows": rows}
