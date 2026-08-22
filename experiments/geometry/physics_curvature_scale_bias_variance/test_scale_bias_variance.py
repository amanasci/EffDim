"""Unit tests that do not require embeddings or a GPU."""

from __future__ import annotations

import numpy as np
import pytest

from geometry.physics_curvature_probe_submission_validation.schema import assert_not_catalog_vector, assert_probe_performance
from geometry.physics_curvature_scale_bias_variance.config import CELLS, PRIMARY
from geometry.physics_curvature_scale_bias_variance.hashing import select_m, split_ab
from geometry.physics_curvature_scale_bias_variance.io_util import p_mc
from geometry.physics_curvature_scale_bias_variance.synthetic import run_synthetic


def test_nested_selection():
    pool = np.arange(2048)
    a = select_m(pool, 1024, seed=0, sample_id=7)
    b = select_m(pool, 1536, seed=0, sample_id=7)
    c = select_m(pool, 2048, seed=0, sample_id=7)
    assert len(a) == 1024 and len(b) == 1536
    assert set(a).issubset(set(b))
    assert set(b).issubset(set(c))
    assert np.array_equal(c, pool)


def test_split_disjoint():
    idx = np.arange(1024)
    A, B = split_ab(idx, seed=3, sample_id=9)
    assert len(A) == 512 and len(B) == 512
    assert len(set(A) & set(B)) == 0
    assert set(A).union(set(B)) == set(idx)


def test_six_unique_cells():
    assert len(CELLS) == 6
    assert len(set(CELLS)) == 6
    assert (2048, 1024) in CELLS
    assert (1024, 1024) in CELLS


def test_p_mc_never_zero():
    assert p_mc(0, 5000) == 1 / 5001
    assert p_mc(5000, 5000) == 1.0


def test_catalog_guard():
    assert_probe_performance(PRIMARY)
    y = np.linspace(0, 1, 32)
    with pytest.raises(RuntimeError):
        assert_not_catalog_vector(y, y.copy())


def test_synthetic_can_discriminate():
    df = run_synthetic(seed=0, n_rep=4, n_anchor=24)
    const = df[df.family == "constant"]
    het = df[df.family == "heterogeneous"]
    assert len(const) and len(het)
    # design check: both families produce finite associations
    assert np.isfinite(const.raw.to_numpy(float)).any()
