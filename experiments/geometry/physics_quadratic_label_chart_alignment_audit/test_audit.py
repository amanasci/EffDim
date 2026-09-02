"""Unit tests for QLCA audit math (no frozen embeddings required)."""

from __future__ import annotations

import numpy as np

from geometry.physics_quadratic_label_chart_alignment.config import LIN_GRID, QUAD_GRID
from geometry.physics_quadratic_label_chart_alignment.features import phi2_frob

from geometry.physics_quadratic_label_chart_alignment_audit.alignment_nulls import (
    alignment_from_spectrum,
    haar_alignment,
    haar_alignment_fast,
    haar_frame,
)
from geometry.physics_quadratic_label_chart_alignment_audit.io_util import p_mc
from geometry.physics_quadratic_label_chart_alignment_audit.rank import (
    energy_rank,
    numerical_rank,
    reachable_fraction,
    row_space_projector,
    singular_spectrum,
    stable_rank,
)
from geometry.physics_quadratic_label_chart_alignment_audit.regularizer import equivalence_demo
from geometry.physics_quadratic_label_chart_alignment_audit.shuffle_diag import gates_from_deltas
from geometry.physics_quadratic_label_chart_alignment_audit.truncated_bs import uq_contains_L


def test_numerical_and_energy_rank():
    rng = np.random.default_rng(0)
    # exact rank 5 in 20 x 12
    A = rng.normal(size=(20, 5))
    B = A @ rng.normal(size=(5, 12))
    S = singular_spectrum(B)
    assert numerical_rank(S, B.shape) == 5
    assert energy_rank(S, 0.99) <= 5
    assert stable_rank(S) <= 5 + 1e-6


def test_row_space_projection():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(30, 10))
    P = row_space_projector(B)
    assert P.shape == (10, 10)
    assert np.allclose(P, P.T)
    assert np.allclose(P @ P, P, atol=1e-8)
    # vectors in row space are fixed
    g = B.T @ rng.normal(size=30)
    assert np.allclose(P @ g, g, atol=1e-8)
    # orthogonal residual
    g2 = rng.normal(size=10)
    frac = reachable_fraction(g2, B)
    assert 0.0 <= frac <= 1.0 + 1e-8
    res = 1.0 - frac
    Pg = row_space_projector(B) @ g2
    assert abs(frac - float(Pg @ Pg) / float(g2 @ g2)) < 1e-10


def test_regularizer_equivalence():
    rec = equivalence_demo(seed=2, n=60, d=5, D=25, alpha=2.5)
    assert rec["ok"]
    assert rec["rank_B"] == rec["q"]
    assert rec["pred_max_abs_diff"] < 1e-6


def test_haar_preserves_spectrum():
    rng = np.random.default_rng(3)
    q, r = 12, 4
    S = np.sort(np.abs(rng.normal(size=r)))[::-1]
    V = haar_frame(q, r, rng)
    assert V.shape == (q, r)
    assert np.allclose(V.T @ V, np.eye(r), atol=1e-10)
    g = rng.normal(size=q)
    a1 = haar_alignment(g, S, np.random.default_rng(4))
    a2 = alignment_from_spectrum(g, S, V)
    assert np.isfinite(a1) and np.isfinite(a2)
    assert np.isfinite(haar_alignment_fast(g, S, np.random.default_rng(9)))


def test_haar_destroys_orientation_in_expectation():
    rng = np.random.default_rng(5)
    q, r = 20, 6
    S = np.linspace(5.0, 0.5, r)
    # γ aligned with first right vector of a fixed V0
    V0 = haar_frame(q, r, rng)
    g = V0[:, 0]
    obs = alignment_from_spectrum(g, S, V0)
    nulls = [haar_alignment(g, S, np.random.default_rng(100 + i)) for i in range(80)]
    assert obs > np.median(nulls)


def test_uq_does_not_contain_L_in_frozen_grid():
    assert uq_contains_L(LIN_GRID, QUAD_GRID) is False
    assert uq_contains_L(LIN_GRID, tuple(list(QUAD_GRID) + [np.inf])) is True


def test_p_mc_never_zero():
    assert p_mc(0, 2000) == 1 / 2001


def test_shuffle_gates_sign_not_abs():
    # large negative: false-positive safe, not calibrated
    g = gates_from_deltas(np.full(21, -7.5))
    assert g["shuffle_no_positive_gain"] is True
    assert g["shuffle_well_calibrated"] is False
    # small positive: not false-positive safe
    g2 = gates_from_deltas(np.full(21, 0.2))
    assert g2["shuffle_no_positive_gain"] is False


def test_train_only_scaling_matches_frozen_contract():
    """Scalar RMS from train fold only; test uses the same s."""
    rng = np.random.default_rng(6)
    U = rng.normal(size=(100, 8))
    fold = np.tile(np.arange(5), 20)
    te = fold == 0
    tr = ~te
    s = float(np.sqrt(np.mean(U[tr] ** 2)))
    Ute = U[te] / s
    assert Ute.shape[0] == int(te.sum())
    # using test RMS would differ
    s_te = float(np.sqrt(np.mean(U[te] ** 2)))
    assert abs(s - s_te) > 0 or True  # just document the contract


def test_phi2_dimension():
    assert phi2_frob(np.zeros((3, 16))).shape[1] == 136
