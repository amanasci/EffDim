"""Gates for the density-calibrated curvature estimator.

These are the checks that license reading the density-vs-curvature results at
all: if the estimator cannot recover a known curvature, or if it reports a
density trend on a manifold that is exactly flat, then any trend it reports on
real embeddings is an artifact of neighbourhood radius rather than geometry.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

_EXP = Path(__file__).resolve().parents[1] / "experiments" / "physics-probe-subspace"
sys.path.insert(0, str(_EXP))

from curvature_core import (  # noqa: E402
    build_knn,
    compute_curvature_suite,
    flatten_null,
    synthetic_manifold,
)
from density_stats import partial_spearman  # noqa: E402


def _sphere(n, d, R, D, noise, seed):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d + 1))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    X = np.zeros((n, D))
    X[:, : d + 1] = V * R
    return X + rng.standard_normal((n, D)) * noise


def _suite(X, K, k_t, p_quad=3, n_perm=8):
    _, idx = build_knn(X, K)
    return compute_curvature_suite(X, idx, k_t, p_quad=p_quad, m_norm=5,
                                   n_perm=n_perm, seed=0, progress_every=0)


@pytest.mark.parametrize("R", [1.0, 2.0])
def test_recovers_known_sphere_curvature(R):
    """kappa_jet is scaled so that a sphere of radius R reads 1/R."""
    out = _suite(_sphere(1500, 2, R, 64, 1e-4, 0), K=60, k_t=2, p_quad=2)
    kappa = np.nanmedian(out["kappa_jet"])
    assert abs(kappa - 1.0 / R) / (1.0 / R) < 0.15
    assert np.nanmedian(out["kappa_ratio"]) > 3.0


def test_plane_reads_as_uncurved():
    rng = np.random.default_rng(0)
    X = np.zeros((1500, 64))
    X[:, :2] = rng.standard_normal((1500, 2))
    X += rng.standard_normal((1500, 64)) * 1e-4
    out = _suite(X, K=60, k_t=2, p_quad=2)
    assert 0.85 < np.nanmedian(out["kappa_ratio"]) < 1.20


def test_rf_k_matches_existing_residual_fraction():
    """The suite's rf_k must be identical to the already-published estimator."""
    from multiscale_curvature_probe import residual_fraction

    X, _ = synthetic_manifold(400, 64, 5, kind="flat", noise=0.05, seed=1)
    _, idx = build_knn(X, 60)
    mine = compute_curvature_suite(X, idx, 5, p_quad=3, m_norm=5, n_perm=2,
                                   seed=0, progress_every=0)["rf_k"]
    assert np.allclose(mine, residual_fraction(X.astype(np.float64), idx, 5), atol=1e-6)


def test_calibration_removes_the_density_artifact():
    """On an exactly FLAT manifold with strongly non-uniform density, the naive
    metrics track density almost perfectly; the calibrated one must not."""
    X, true_k = synthetic_manifold(1200, 64, 3, kind="flat", noise=0.02,
                                   density_tilt=2.0, seed=0)
    assert true_k == 0.0
    dists, idx = build_knn(X, 60)
    d_k = dists[:, 29]
    out = compute_curvature_suite(X, idx, 3, p_quad=3, m_norm=5, n_perm=16,
                                  seed=0, progress_every=0)

    def rho(name):
        v = out[name]
        ok = np.isfinite(v)
        return spearmanr(d_k[ok], v[ok]).statistic

    assert abs(rho("rf_k")) > 0.8, "the artifact should be blatant on flat data"
    assert abs(rho("kappa_naive_ratio")) > 0.5
    assert abs(rho("kappa_ratio")) < 0.25, "calibration failed to remove it"
    assert 0.85 < np.nanmedian(out["kappa_ratio"]) < 1.20


def test_flatten_null_is_actually_flat():
    """The surrogate built from real-shaped data must read as uncurved."""
    X, _ = synthetic_manifold(1000, 64, 4, kind="sphere", curvature=0.5,
                              noise=0.02, seed=0)
    flat = flatten_null(X, k_t=4, mode="gauss", seed=0)
    curved_ratio = np.nanmedian(_suite(X, 60, 4)["kappa_ratio"])
    flat_ratio = np.nanmedian(_suite(flat, 60, 4)["kappa_ratio"])
    assert curved_ratio > flat_ratio
    assert 0.85 < flat_ratio < 1.25


def test_build_knn_excludes_self_and_sorts():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 16))
    dists, idx = build_knn(X, 10)
    assert not (idx == np.arange(200)[:, None]).any(), "self must be excluded"
    assert (np.diff(dists, axis=1) >= -1e-5).all(), "must be distance-sorted"


def test_partial_spearman_reduces_to_spearman_for_constant_control():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    y = 0.6 * x + rng.standard_normal(300)
    z = np.ones(300)
    assert partial_spearman(x, y, z)["rho"] == pytest.approx(
        spearmanr(x, y).statistic, abs=1e-9
    )
