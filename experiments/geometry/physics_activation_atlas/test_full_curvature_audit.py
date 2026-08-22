"""Unit tests for full curvature audit helpers."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.full_curvature_audit import m_quad, neff_flag, tensor_agreement
from geometry.physics_activation_atlas.split_half_curvature_reliability import tensor_agreement as ta2


def test_m_quad():
    assert m_quad(16) == 136
    assert m_quad(8) == 36


def test_neff_flags_weak_small_k():
    weak = neff_flag(256, 16, half=True)
    assert weak["statistically_weak"] is True
    strong = neff_flag(2048, 16, half=True)
    assert strong["n_eff_over_m"] > weak["n_eff_over_m"]


def test_tensor_agreement_identical():
    rng = np.random.default_rng(0)
    v = rng.normal(size=32)
    ag = tensor_agreement(v, v)
    assert abs(ag["r_dir"] - 1.0) < 1e-9
    assert abs(ag["R_signal"] - 1.0) < 1e-9
    assert abs(ag["inner"] - np.dot(v, v)) < 1e-9


def test_tensor_agreement_matches_split_half_helper():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=20), rng.normal(size=20)
    assert abs(tensor_agreement(a, b)["r_dir"] - ta2(a, b)["r_dir"]) < 1e-12
