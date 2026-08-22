"""Unit tests for order-stratified geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_order_stratified_geometry.algebra import (
    mixed_scale_nnls,
    pair_antipodes,
    projector_overlap,
    svd_quadratic_image,
    truncate_bs_left,
)
from geometry.physics_order_stratified_geometry.rank import DEFAULT_Q_THRESHOLDS, classify_hypothesis, select_q2
from geometry.physics_order_stratified_geometry.synthetics import make_order_synthetic


def test_projector_overlap_identity():
    rng = np.random.default_rng(0)
    A, _ = np.linalg.qr(rng.normal(size=(20, 4)))
    assert abs(projector_overlap(A, A) - 1.0) < 1e-10
    B, _ = np.linalg.qr(rng.normal(size=(20, 4)))
    ov = projector_overlap(A, B)
    assert 0.0 <= ov <= 1.0 + 1e-8


def test_svd_recovers_planted_quadratic_rank():
    rng = np.random.default_rng(1)
    D, d, qtrue = 40, 6, 3
    U = rng.normal(size=(D, qtrue))
    U, _ = np.linalg.qr(U)
    s = np.array([5.0, 3.0, 1.5])
    V = rng.normal(size=(d * (d + 1) // 2, qtrue))
    V, _ = np.linalg.qr(V)
    M = (U * s) @ V.T
    # pack as BS_flat without extra sqrt2 (flatten_sym2 will reweight; use diagonal-only)
    nq = d * (d + 1) // 2
    BS = np.zeros((D, nq))
    # put energy on first qtrue diagonal monomials, which have weight 1
    idx = 0
    diag_idx = []
    for a in range(d):
        for b in range(a, d):
            if a == b:
                diag_idx.append(idx)
            idx += 1
    for j, di in enumerate(diag_idx[:qtrue]):
        BS[:, di] = U[:, j] * s[j]
    info = svd_quadratic_image(BS, d)
    assert info["s"][0] > info["s"][qtrue] * 2 if len(info["s"]) > qtrue else True
    assert np.sum(info["s"][:qtrue] ** 2) / max(np.sum(info["s"] ** 2), 1e-12) > 0.85


def test_truncate_zero_and_full():
    rng = np.random.default_rng(2)
    BS = rng.normal(size=(16, 10))
    z = truncate_bs_left(BS, 4, 0)
    assert np.allclose(z, 0)
    full = truncate_bs_left(BS, 4, 20)
    assert full.shape == BS.shape


def test_mixed_nnls_recovers_quad():
    r = np.linspace(0.05, 0.2, 12)
    v = 3.0 * r**4
    mix = mixed_scale_nnls(r, v)
    assert mix["identifiable"]
    assert mix["b"] > mix["a"]
    assert mix["b"] > mix["c"]


def test_antipodal_pairing_finds_negatives():
    rng = np.random.default_rng(3)
    U = rng.normal(size=(80, 6))
    U = np.vstack([U, -U])
    rad = np.linalg.norm(U, axis=1)
    pr = pair_antipodes(U, rad, cos_min=0.8)
    assert pr["n_pairs"] >= 20


def test_select_q2_consecutive_prefix():
    rng = np.random.default_rng(4)
    UA, _ = np.linalg.qr(rng.normal(size=(30, 6)))
    Q3, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    UB = UA.copy()
    UB[:, :3] = UA[:, :3] @ Q3
    sA = np.array([4.0, 3.9, 3.8, 0.25, 0.12, 0.05])
    sB = sA.copy()
    dS = np.cumsum(np.array([0.05, 0.04, 0.03, 0.0, 0.0, 0.0]))
    thr = dict(DEFAULT_Q_THRESHOLDS)
    thr["rel_gap_min"] = 0.15
    sel = select_q2(sA=sA, sB=sB, UA=UA, UB=UB, dS=dS, persist=np.ones(6), energy_null=0.01, thr=thr)
    assert 2 <= sel["q2"] <= 4, sel


def test_classify_curved12():
    lab = classify_hypothesis(
        q2=4,
        overlap_e4=0.6,
        r2_quad=0.4,
        residual_r2_linear=0.01,
        pi_lin=0.1,
        pi_quad=0.7,
        pi_thick=0.2,
        m12_vs_m16=-0.01,
        mix_resolved=True,
    )
    assert lab == "linear12_plus_quadratic_normal_modes"


def test_synth_kinds_run():
    pack = make_order_synthetic("curved_d12_q4", n=120, D=36, seed=0, k_obs=64, d_core=6)
    assert pack["true_q2"] == 4
    assert pack["X"].shape[1] == 36
    pack2 = make_order_synthetic("flat_d12", n=80, D=32, seed=1, k_obs=40, d_core=6)
    assert pack2["true_q2"] == 0


def test_done_helper(tmp_path):
    from geometry.physics_order_stratified_geometry.pipeline import _done

    p = tmp_path / "m.json"
    assert _done(p, False) is False
    p.write_text("{}")
    assert _done(p, False) is True
    assert _done(p, True) is False
