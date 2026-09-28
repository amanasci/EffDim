"""Known-answer tests for :mod:`pu_manifold.reconstruction_truth`.

The decisive tests are the two IDENTITY tests: feeding a fixture's own ``X`` back through
the inverse map must reproduce that fixture's own ``H_norm``. Everything this module does
rests on inverting a preprocessing/rotation chain correctly, and an inversion that is subtly
wrong produces plausible numbers rather than an error -- so the identity is the only check
that can catch it.

Not collected by the core ``effdim`` suite (``pyproject.toml``'s ``testpaths = ["tests"]``
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_reconstruction_truth.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from pu_manifold import curvature_probe, synthetic_controls
from pu_manifold import reconstruction_truth as rt


# --- Swiss roll ---------------------------------------------------------------------------


def test_swiss_roll_t_at_recovers_the_generators_own_t():
    """``t = sqrt(x^2 + z^2)`` must return the generator's parameter to machine precision."""
    n, seed = 500, 0
    fx = curvature_probe.make_swiss_roll_fixture(n=n, seed=seed)
    t_hat = rt.swiss_roll_t_at(fx["X"], n=n, seed=seed)
    np.testing.assert_allclose(t_hat, fx["t"], rtol=1e-12, atol=1e-10)


def test_swiss_roll_truth_at_reproduces_fixture_on_its_own_points():
    """IDENTITY TEST. Evaluated at the fixture's own points, the re-scored truth IS the
    fixture's sealed ``H_norm``."""
    n, seed = 500, 0
    fx = curvature_probe.make_swiss_roll_fixture(n=n, seed=seed)
    H_at = rt.swiss_roll_truth_at(fx["X"], n=n, seed=seed)
    np.testing.assert_allclose(H_at, fx["H_norm"], rtol=1e-10, atol=1e-12)


def test_swiss_roll_truth_changes_when_the_point_moves():
    """A displaced point must get a DIFFERENT true curvature -- otherwise the whole concern
    this module addresses would be vacuous."""
    n, seed = 500, 0
    fx = curvature_probe.make_swiss_roll_fixture(n=n, seed=seed)
    X = np.asarray(fx["X"])
    # push every point outward in the x-z plane: a real move along the spiral
    moved = X.copy()
    moved[:, 0] *= 1.25
    moved[:, 2] *= 1.25
    H_moved = rt.swiss_roll_truth_at(moved, n=n, seed=seed)
    assert not np.allclose(H_moved, fx["H_norm"], rtol=1e-3)
    # and curvature must DROP, since ||H|| for the spiral falls as t grows
    assert np.median(H_moved) < np.median(fx["H_norm"])


def test_swiss_roll_t_at_rejects_wrong_width():
    with pytest.raises(ValueError, match=r"\(m, 3\)"):
        rt.swiss_roll_t_at(np.zeros((10, 4)), n=100, seed=0)


# --- saddle control -----------------------------------------------------------------------


@pytest.mark.parametrize("d,D", [(4, 12), (20, 28)])
def test_saddle_truth_at_reproduces_fixture_on_its_own_points(d, D):
    """IDENTITY TEST at two ``d``, including the ``d=20`` regime the sealed controls ran at.

    This is the test that validates the ``rotate_and_pad`` inversion: the zero-pad, the
    orthogonal ``Q`` rebuilt from ``(D, seed)``, the centring offset, and the ``global_std``
    rescaling of both ``X`` and ``H``. Any one of those wrong and the numbers stay finite and
    plausible while being meaningless.
    """
    n, seed = 400, 20260816
    fx = synthetic_controls.make_saddle_control(n=n, d=d, D=D, seed=seed)
    H_at = rt.saddle_truth_at(fx["X"], fx, d=d, D=D, seed=seed)
    np.testing.assert_allclose(H_at, fx["H_norm"], rtol=1e-9, atol=1e-12)


def test_saddle_truth_changes_when_the_point_moves():
    n, d, D, seed = 400, 4, 12, 20260816
    fx = synthetic_controls.make_saddle_control(n=n, d=d, D=D, seed=seed)
    rng = np.random.default_rng(3)
    moved = np.asarray(fx["X"]) + 0.3 * rng.standard_normal(np.asarray(fx["X"]).shape)
    H_moved = rt.saddle_truth_at(moved, fx, d=d, D=D, seed=seed)
    assert not np.allclose(H_moved, fx["H_norm"], rtol=1e-3)


def test_saddle_rotation_matches_synthetic_controls_own_Q():
    """``_rotation`` must rebuild the exact ``Q`` ``rotate_and_pad`` used, not merely an
    orthogonal matrix from the same seed family."""
    D, seed = 12, 20260816
    Q = rt._rotation(D, seed)
    rng = np.random.default_rng(seed)
    Q_ref, _ = np.linalg.qr(rng.standard_normal((D, D)))
    np.testing.assert_allclose(Q, Q_ref, rtol=0, atol=0)
    np.testing.assert_allclose(Q @ Q.T, np.eye(D), atol=1e-12)


# --- drift and rescore ---------------------------------------------------------------------


def test_reconstruction_drift_is_zero_for_a_perfect_reconstruction():
    X = np.random.default_rng(0).standard_normal((50, 3))
    out = rt.reconstruction_drift(X, X)
    np.testing.assert_allclose(out["drift"], 0.0, atol=0)
    assert out["median_drift_relative"] == 0.0


def test_reconstruction_drift_measures_the_offset():
    X = np.zeros((4, 2))
    X[:, 0] = 1.0
    Y = X.copy()
    Y[:, 1] = 3.0  # each point moved 3 units orthogonally
    out = rt.reconstruction_drift(X, Y)
    np.testing.assert_allclose(out["drift"], 3.0)
    np.testing.assert_allclose(out["median_drift_relative"], 3.0)


def test_reconstruction_drift_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        rt.reconstruction_drift(np.zeros((4, 3)), np.zeros((5, 3)))


def test_rescore_reports_both_scores_and_their_gap():
    rng = np.random.default_rng(11)
    n = 400
    truth_in = rng.uniform(0.5, 2.0, size=n)
    # estimate tracks the truth at the RECONSTRUCTION, not at the input
    truth_recon = truth_in + rng.normal(0, 0.4, size=n)
    est = truth_recon + rng.normal(0, 0.05, size=n)

    out = rt.rescore(est, truth_in, truth_recon)
    assert out["rho_at_recon"] > out["rho_at_input"]
    assert out["delta"] > 0.0
    assert out["curvature_convention"] == "trace"


def test_rescore_delta_is_zero_when_the_two_truths_agree():
    rng = np.random.default_rng(12)
    truth = rng.uniform(0.5, 2.0, size=300)
    est = truth + rng.normal(0, 0.1, size=300)
    out = rt.rescore(est, truth, truth)
    assert out["delta"] == pytest.approx(0.0, abs=1e-12)
    assert out["truth_rank_agreement"] == pytest.approx(1.0, abs=1e-12)


def test_rescore_accepts_a_drift_control():
    rng = np.random.default_rng(13)
    n = 300
    truth = rng.uniform(0.5, 2.0, size=n)
    drift = rng.uniform(0.0, 1.0, size=n)
    est = truth + 2.0 * drift + rng.normal(0, 0.05, size=n)
    out = rt.rescore(est, truth, truth, drift=drift)
    assert "rho_input_given_drift" in out
    assert abs(out["rho_input_given_drift"]) <= 1.0
