"""Known-answer tests for :mod:`pu_manifold.cross_split_curvature`.

Every case here has an answer derivable by hand or from an independent implementation
(``scipy.stats.spearmanr``). The two statistical tests -- noise cancellation and confound
removal -- are the ones that matter: they are the reasons the module exists, and neither is
checkable by inspecting the algebra.

Not collected by the core ``effdim`` suite (``pyproject.toml``'s ``testpaths = ["tests"]``
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_cross_split_curvature.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from scipy.stats import spearmanr

from pu_manifold import cross_split_curvature as csc


# --- tensor_agreement: hand-computable fixed points -------------------------------------


def test_identical_arms_are_perfectly_reliable():
    H = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = csc.tensor_agreement(H, H)
    np.testing.assert_allclose(out["R_signal"], [1.0, 1.0])
    np.testing.assert_allclose(out["r_dir"], [1.0, 1.0])
    # <H, H> == ||H||^2: 25 and 1.
    np.testing.assert_allclose(out["inner"], [25.0, 1.0])


def test_orthogonal_arms_score_zero():
    A = np.array([[1.0, 0.0]])
    B = np.array([[0.0, 1.0]])
    out = csc.tensor_agreement(A, B)
    np.testing.assert_allclose(out["inner"], [0.0])
    np.testing.assert_allclose(out["R_signal"], [0.0])


def test_opposing_arms_score_minus_one():
    """The diagnostic case: the two fits disagree on the sign of the curvature."""
    A = np.array([[2.0, 0.0]])
    out = csc.tensor_agreement(A, -A)
    np.testing.assert_allclose(out["R_signal"], [-1.0])
    np.testing.assert_allclose(out["r_dir"], [-1.0])


def test_R_signal_penalises_magnitude_disagreement_where_r_dir_does_not():
    """Same direction, 10x scale: r_dir = 1, R_signal = 2*10/(1+100)."""
    A = np.array([[1.0, 0.0]])
    B = np.array([[10.0, 0.0]])
    out = csc.tensor_agreement(A, B)
    np.testing.assert_allclose(out["r_dir"], [1.0])
    np.testing.assert_allclose(out["R_signal"], [20.0 / 101.0])


def test_mismatched_shapes_refuse():
    with pytest.raises(ValueError, match="identical shape"):
        csc.tensor_agreement(np.zeros((4, 3)), np.zeros((5, 3)))


def test_one_dimensional_input_refuses():
    with pytest.raises(ValueError, match=r"\(n, D\)"):
        csc.tensor_agreement(np.zeros(3), np.zeros(3))


# --- the reason the module exists ---------------------------------------------------------


def test_cross_statistic_is_unbiased_where_the_single_split_norm_is_not():
    """The central claim, stated as a measurement rather than as algebra.

    Truth ``H`` is a fixed vector per point. Each arm observes ``H + e``, ``e`` independent
    zero-mean Gaussian noise of a magnitude comparable to ``||H||`` -- the ``d = 20`` regime,
    where spike 002 measured a median magnitude ratio of 224.

    ``E<H + e_A, H + e_B> = ||H||^2`` because the cross terms and ``E<e_A, e_B>`` all vanish.
    ``E||H + e_A||`` does NOT equal ``||H||``; it is strictly larger, because the norm is
    convex and the noise adds in quadrature. So the single-split statistic inflates and the
    cross statistic does not.
    """
    rng = np.random.default_rng(20260822)
    n, D, sigma = 20000, 8, 1.0
    H = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (n, 1))
    A = H + rng.normal(0.0, sigma, size=(n, D))
    B = H + rng.normal(0.0, sigma, size=(n, D))

    out = csc.cross_curvature_field(A, B, independence="disjoint_data")
    truth_sq = 1.0

    # Cross statistic recovers ||H||^2 = 1. Standard error over n draws is ~sqrt(2)*sigma^2/sqrt(n).
    assert abs(out["K_H_cross"].mean() - truth_sq) < 0.05

    # The single-split norm is inflated far beyond ||H|| = 1 by the same noise.
    assert out["h_norm_single"].mean() > 2.5

    # And squaring it does not fix the bias -- E||H+e||^2 = ||H||^2 + D*sigma^2 = 9.
    assert abs((out["h_norm_single"] ** 2).mean() - (truth_sq + D * sigma**2)) < 0.2


def _ranking_pair(seed: int, sigma: float, n: int = 4000, D: int = 8):
    """A field whose true ``||H||`` varies, observed through independent noise of scale
    ``sigma`` in each arm. Returns ``(rho_single, rho_cross)`` against the true ``||H||``."""
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.2, 3.0, size=n)
    H = np.zeros((n, D))
    H[:, 0] = scale
    A = H + rng.normal(0.0, sigma, size=(n, D))
    B = H + rng.normal(0.0, sigma, size=(n, D))
    out = csc.cross_curvature_field(A, B, independence="disjoint_data")
    truth = np.linalg.norm(H, axis=-1)
    return (
        float(spearmanr(out["h_norm_single"], truth).statistic),
        float(spearmanr(out["K_H_cross"], truth).statistic),
    )


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003])
def test_cross_statistic_ranks_better_than_the_single_split_norm(seed):
    """At moderate noise the cross statistic recovers ordering the single-split norm loses.

    Measured over seeds 1000-1007 at ``sigma = 1.0``: single ``0.430``, cross ``0.556``,
    minimum gap ``+0.110``. The bound below is deliberately well inside that.
    """
    rho_single, rho_cross = _ranking_pair(seed, sigma=1.0)
    assert rho_cross > rho_single + 0.08


def test_cross_statistic_advantage_shrinks_as_the_estimator_degrades():
    """**The limit of this method, pinned as a test because it bounds what the port buys.**

    The cross statistic is unbiased at every noise level, but rank recovery still decays
    with noise, and the ADVANTAGE over the single-split norm decays with it. Measured means
    over eight seeds:

        sigma   single   cross    gap
        0.5     0.804    0.883   +0.080
        1.0     0.430    0.556   +0.126
        1.5     0.231    0.314   +0.083
        2.5     0.093    0.128   +0.035

    The ``sigma = 2.5`` row is the regime our ``d = 20`` measurements sit in -- single-split
    rho around 0.09, which brackets the sealed decoder's ``-0.015`` and phase 03.1's
    ``+0.116``. There, going cross-split buys roughly ``+0.035``. So removing the noise bias
    is a genuine improvement and CANNOT on its own carry a ``d = 20`` field to a usable
    ordering; anything claiming otherwise is claiming something this test refutes.
    """
    low = np.mean([_ranking_pair(s, sigma=1.0)[1] - _ranking_pair(s, sigma=1.0)[0] for s in range(1000, 1004)])
    high = np.mean([_ranking_pair(s, sigma=2.5)[1] - _ranking_pair(s, sigma=2.5)[0] for s in range(1000, 1004)])
    assert high < low
    assert high < 0.08


def test_independence_mode_is_not_defaultable():
    H = np.ones((3, 2))
    with pytest.raises(TypeError):
        csc.cross_curvature_field(H, H)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="independence must be one of"):
        csc.cross_curvature_field(H, H, independence="whatever")


def test_K_H_cross_stays_signed():
    """Clipping before a rank statistic would tie every disagreeing point together."""
    A = np.array([[1.0, 0.0], [1.0, 0.0]])
    B = np.array([[1.0, 0.0], [-1.0, 0.0]])
    out = csc.cross_curvature_field(A, B, independence="disjoint_data")
    assert out["K_H_cross"][1] < 0.0


# --- reliability gate ---------------------------------------------------------------------


def test_reliability_summary_counts_and_verdict():
    R = np.array([-0.5, 0.1, 0.6, 0.9])
    s = csc.reliability_summary(R, threshold=0.5, min_fraction=0.5)
    assert s["n_above"] == 2
    assert s["fraction_above"] == 0.5
    assert s["admissible"] is True
    assert s["fraction_negative"] == 0.25

    strict = csc.reliability_summary(R, threshold=0.5, min_fraction=0.75)
    assert strict["admissible"] is False


def test_reliability_summary_refuses_empty_field():
    with pytest.raises(ValueError, match="not a measurement"):
        csc.reliability_summary(np.zeros(0), threshold=0.5)


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
