"""Tests for ``crossmodal_curvature.split_indices``, the one function the paper runners use.

The Phase 7 pre-registration guards and the freeze-commit ancestry test that used to live here
are archived in ``archive/pu_manifold_trimmed/tests/test_crossmodal_curvature.py``.

Loads no PU data, trains nothing, reads no cache. Completes in well under 10 seconds.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import crossmodal_curvature as cc  # noqa: E402


# --- split_indices -----------------------------------------------------------------------


def test_split_indices_shape_and_disjointness():
    train_idx, holdout_idx = cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    assert len(train_idx) == 8000
    assert len(holdout_idx) == 2000
    train_set = set(train_idx.tolist())
    holdout_set = set(holdout_idx.tolist())
    assert train_set.isdisjoint(holdout_set)
    assert train_set | holdout_set == set(range(10000))


def test_split_indices_is_deterministic():
    train_a, holdout_a = cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    train_b, holdout_b = cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(holdout_a, holdout_b)
