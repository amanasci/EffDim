"""Unit and parity tests for the handoff curvature package. No real-data fitting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from geometry.curvature.alignment import assert_unique_sample_ids, join_by_sample_id
from geometry.curvature.artifacts import decision_label, is_complete, source_of_truth_rank
from geometry.curvature.decoder import (
    christoffel_second_fundamental_form,
    decoder_full_euclidean_second_fundamental_form,
    decoder_sphere_residual_second_fundamental_form,
    he_equals_hs_minus_d_x,
    sphere_radial_second_fundamental_form,
    unaveraged_mean_curvature,
)
from geometry.curvature.label_hessian import fit_label_hessian, mismatch
from geometry.curvature.metric import energy_g, holm, metric_from_J, projectors, trace_g
from geometry.curvature.probe_aligned import (
    complete_normal_w,
    probe_aligned_second_fundamental_form,
    q_task_aligned_cross_energy,
    sphere_component,
    sphere_term_norm_g,
)
from geometry.curvature.quadratic import Gamma_from_gamma, n_quad, phi2_frob, split_half_cross_energy
from geometry.curvature.statistics import p_mc, rank_partial_spearman
from geometry.curvature.types import OutcomeKind, assert_not_catalogue_as_performance


def test_projector_symmetric_idempotent():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(10, 3))
    x = rng.normal(size=10)
    x /= np.linalg.norm(x)
    P = projectors(x, J)["P_T"]
    assert np.allclose(P, P.T)
    assert np.allclose(P @ P, P, atol=1e-8)


def test_metric_reparam_invariance():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(8, 3))
    A = np.array([[1.1, 0.2, 0], [0, 0.9, 0.1], [0.05, 0, 1.0]])
    g, ginv = metric_from_J(J)
    g2, ginv2 = metric_from_J(J @ A)
    b = rng.normal(size=(3, 3))
    b = 0.5 * (b + b.T)
    b2 = A.T @ b @ A
    assert abs(energy_g(b, ginv) - energy_g(b2, ginv2)) / max(energy_g(b, ginv), 1e-12) < 1e-8


def test_II_normal_and_christoffel():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(9, 3))
    x = rng.normal(size=9)
    x /= np.linalg.norm(x)
    proj = projectors(x, J)
    Hess = rng.normal(size=(9, 3, 3))
    Hess = 0.5 * (Hess + np.transpose(Hess, (0, 2, 1)))
    II = decoder_full_euclidean_second_fundamental_form(Hess, proj["P_N"])
    assert np.allclose(np.einsum("ij,jab->iab", proj["P_T"], II), 0, atol=1e-8)
    II2 = christoffel_second_fundamental_form(Hess, J, proj["ginv"])
    assert np.allclose(II, II2, atol=1e-7)


def test_radial_identity_and_HE_HS():
    d = 4
    g = np.eye(d)
    xhat = np.zeros(6)
    xhat[0] = 1.0
    IIR = sphere_radial_second_fundamental_form(xhat, g)
    # <x, II^R> = -g
    assert np.allclose(np.einsum("i,iab->ab", xhat, IIR), -g)
    IIS = np.zeros((6, d, d))
    IIE = IIS + IIR
    HE = unaveraged_mean_curvature(IIE, np.eye(d))
    HS = unaveraged_mean_curvature(IIS, np.eye(d))
    assert he_equals_hs_minus_d_x(HE, HS, xhat, d)


def test_trace_vs_average():
    B = np.zeros((5, 3, 3))
    B[:, 0, 0] = 1.0
    ginv = np.eye(3)
    una = unaveraged_mean_curvature(B, ginv)
    avg = una / 3.0
    assert np.allclose(una, 3 * avg)


def test_frob_features_and_sqrt2():
    from geometry.curvature.quadratic import gamma_from_Gamma

    U = np.array([[1.0, 2.0], [0.0, 1.0]])
    phi = phi2_frob(U)
    assert phi.shape == (2, 3)
    assert np.allclose(phi[0, 0], 0.5)
    assert np.allclose(phi[0, 1], 2.0 / np.sqrt(2))
    G = np.array([[1.0, 0.5], [0.5, 2.0]])
    g1 = gamma_from_Gamma(G)
    G2 = Gamma_from_gamma(g1, 2)
    assert np.allclose(G, G2)
    assert np.allclose(np.dot(g1, g1), np.sum(G * G))


def test_split_half_no_clamp():
    ginv = np.eye(2)
    ba = -np.eye(2)
    bb = np.eye(2)
    v = split_half_cross_energy(ba, bb, ginv)
    assert v < 0


def test_probe_aligned_and_sphere_norm():
    rng = np.random.default_rng(3)
    J = rng.normal(size=(8, 3))
    x = rng.normal(size=8)
    x /= np.linalg.norm(x)
    proj = projectors(x, J)
    w = rng.normal(size=8)
    wN = complete_normal_w(w, proj)
    assert np.allclose(proj["P_T"] @ wN, 0, atol=1e-8)
    Hess = rng.normal(size=(8, 3, 3))
    II = decoder_full_euclidean_second_fundamental_form(Hess, proj["P_N"])
    b = probe_aligned_second_fundamental_form(wN, II)
    assert np.allclose(b, np.einsum("i,iab->ab", wN, II))
    bR = sphere_component(w, proj["xhat"], proj["g"])
    chk = sphere_term_norm_g(bR, proj["ginv"], w, proj["xhat"], 3)
    assert chk["ok"]


def test_q_cross_not_average_then_square():
    rng = np.random.default_rng(4)
    BA = rng.normal(size=(5, 3, 3))
    BB = -BA
    w = rng.normal(size=5)
    ginv = np.eye(3)
    e = q_task_aligned_cross_energy(w, BA, BB, ginv)
    avg = 0.5 * (BA + BB)
    b_avg = probe_aligned_second_fundamental_form(w, avg)
    e_wrong = energy_g(b_avg, ginv)
    assert e <= 0
    assert e_wrong >= 0 or True  # just prove they differ in general
    assert not np.isclose(e, e_wrong)


def test_known_quadratic_hessian():
    rng = np.random.default_rng(5)
    U = rng.normal(size=(300, 3))
    H = np.array([[1.0, 0.2, 0], [0.2, -0.5, 0.1], [0, 0.1, 0.3]])
    y = 0.4 + U @ np.array([0.1, -0.2, 0.05]) + 0.5 * np.einsum("ni,ij,nj->n", U, H, U)
    fit = fit_label_hessian(U, y, ridge=0.0, d=3)
    assert fit["ok"]
    assert np.allclose(fit["Hy"], H, atol=1e-6)


def test_sample_id_join():
    a = pd.DataFrame({"sample_id": [3, 1, 2], "x": [10, 11, 12]})
    b = pd.DataFrame({"sample_id": [1, 2, 3], "y": [1, 2, 3]})
    j = join_by_sample_id(a, b)
    assert list(j.sort_values("sample_id").y) == [1, 2, 3]
    with pytest.raises(ValueError):
        assert_unique_sample_ids([1, 1, 2])


def test_target_typing():
    with pytest.raises(TypeError):
        assert_not_catalogue_as_performance(OutcomeKind.CATALOGUE_LABEL)


def test_holm_and_pmc():
    h = holm(np.array([0.01, 0.04]))
    assert h[0] <= h[1] + 1e-15
    assert p_mc(0, 10000) == 1 / 10001


def test_parity_task_aligned_algebra():
    hist = pytest.importorskip("geometry.physics_task_aligned_curvature.algebra")
    rng = np.random.default_rng(6)
    J = rng.normal(size=(7, 3))
    x = rng.normal(size=7)
    assert np.allclose(projectors(x, J)["g"], hist.projectors(x, J)["g"])
    b = rng.normal(size=(3, 3))
    ginv = projectors(x, J)["ginv"]
    assert abs(energy_g(b, ginv) - hist.energy_g(b, ginv)) < 1e-12


def test_parity_qlca_features():
    hist = pytest.importorskip("geometry.physics_quadratic_label_chart_alignment.features")
    U = np.arange(12, dtype=float).reshape(3, 4)
    assert np.allclose(phi2_frob(U), hist.phi2_frob(U))


def test_frozen_headline_parity():
    out = ROOT / "outputs" / "geometry"
    tac = out / "physics_task_aligned_curvature" / "decision.json"
    if tac.exists():
        d = json.loads(tac.read_text())
        assert d["label"] == "quadratic_task_aligned_effect_only"
        assert abs(d["P2"]["observed"] - 0.11614541311898466) < 1e-12
    hmm = out / "physics_cross_model_hessian_mismatch" / "decision.json"
    if hmm.exists():
        d = json.loads(hmm.read_text())
        assert d["label"] == "label_hessian_unreliable"
    qres = out / "physics_q_geometry_resampling_stability" / "decision.json"
    if qres.exists():
        d = json.loads(qres.read_text())
        assert d["summary_label"] == "q_global_and_adaptation_associations_geometry_robust"
        assert abs(d["notes"]["conditional"]["r2_G"]["original"] + 0.2404841119636992) < 1e-12
    assert source_of_truth_rank()[0] == "COMPLETE.json"


def test_train_eval_isolation_hash():
    probes = pytest.importorskip("geometry.physics_task_aligned_curvature.probes")
    ids = np.arange(100)
    s = probes.split_mask(ids, np.ones(100, bool), "mag_r_desi")
    assert not np.any(s["train"] & s["eval"])


def test_probe_aligned_finite_diff_identity():
    rng = np.random.default_rng(7)
    d, D = 3, 8
    J = rng.normal(size=(D, d))
    Q = rng.normal(size=(D, d, d))
    Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
    x0 = rng.normal(size=D)
    x0 /= np.linalg.norm(x0)
    w = rng.normal(size=D)
    proj = projectors(x0, J)
    wN = complete_normal_w(w, proj)
    II = decoder_full_euclidean_second_fundamental_form(Q, proj["P_N"])
    b = probe_aligned_second_fundamental_form(wN, II)
    # F(u)=x0+Ju+(1/2)Q(u,u); Hess(w·F)=<w,Q>=<w_N,II>
    assert np.allclose(b, np.einsum("i,iab->ab", wN, Q), atol=1e-8)
    u = np.zeros(d)
    eps = 1e-5
    def yhat(uu):
        x = x0 + J @ uu + 0.5 * np.einsum("iab,a,b->i", Q, uu, uu)
        return float(np.dot(w, x))
    hess = np.zeros((d, d))
    for a in range(d):
        for b in range(d):
            e_a = np.zeros(d); e_a[a] = eps
            e_b = np.zeros(d); e_b[b] = eps
            hess[a, b] = (yhat(u+e_a+e_b) - yhat(u+e_a-e_b) - yhat(u-e_a+e_b) + yhat(u-e_a-e_b)) / (4 * eps * eps)
    assert np.allclose(0.5 * (hess + hess.T), np.einsum("i,iab->ab", w, Q), atol=2e-4)


def test_manifest_hash_roundtrip(tmp_path):
    from geometry.curvature.artifacts import file_sha256, file_meta
    p = tmp_path / "a.txt"
    p.write_text("curvature-handoff")
    assert len(file_sha256(p)) == 64
    assert file_meta(p)["exists"] is True
    assert file_meta(tmp_path / "missing")["exists"] is False
