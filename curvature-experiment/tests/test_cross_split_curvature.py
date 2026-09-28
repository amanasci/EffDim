"""Known-answer tests for :mod:`pu_manifold.cross_split_curvature`.

Every case here has an answer derivable by hand or from an independent implementation
(``scipy.stats.spearmanr``). The two statistical tests -- noise cancellation and confound
removal -- are the ones that matter: they are the reasons the module exists, and neither is
checkable by inspecting the algebra.

Not collected by the core ``effdim`` suite (``pyproject.toml``'s ``testpaths = ["tests"]``
excludes this directory) -- run explicitly:

    python -m pytest curvature-experiment/tests/test_cross_split_curvature.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from scipy.stats import spearmanr

from pu_manifold import cross_split_curvature as csc


# --- partial_spearman ---------------------------------------------------------------------


def test_partial_spearman_without_controls_matches_scipy():
    rng = np.random.default_rng(7)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(size=200)
    assert abs(csc.partial_spearman(x, y) - spearmanr(x, y).statistic) < 1e-10


def test_partial_spearman_removes_a_pure_confound():
    """``x`` and ``y`` are conditionally independent given ``c``; both are driven by it.

    The raw rank correlation is large and entirely spurious. The controlled statistic must
    collapse toward zero -- this is the transform that turns the source branch's raw
    ``-0.412`` at ``d=16`` into its reported ``-0.240``.
    """
    rng = np.random.default_rng(240)
    n = 3000
    c = rng.normal(size=n)
    x = c + 0.3 * rng.normal(size=n)
    y = c + 0.3 * rng.normal(size=n)

    raw = csc.partial_spearman(x, y)
    controlled = csc.partial_spearman(x, y, controls=c)
    assert raw > 0.85
    assert abs(controlled) < 0.10


def test_partial_spearman_keeps_a_real_association_that_is_not_the_control():
    rng = np.random.default_rng(241)
    n = 3000
    c = rng.normal(size=n)
    shared = rng.normal(size=n)
    x = c + shared + 0.2 * rng.normal(size=n)
    y = c + shared + 0.2 * rng.normal(size=n)

    controlled = csc.partial_spearman(x, y, controls=c)
    assert controlled > 0.5


def test_partial_spearman_refuses_degenerate_input():
    with pytest.raises(ValueError, match="fewer than three points"):
        csc.partial_spearman([1.0, 2.0], [1.0, 2.0])
