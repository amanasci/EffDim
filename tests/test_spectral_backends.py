"""Tests for swappable spectral backends (streaming covariance vs SVD)."""
import numpy as np
import pytest

from effdim.api import (
    _VALID_SPECTRAL_BACKENDS,
    _do_streaming_covariance,
    _do_svd,
    compute_dim,
)


SPECTRAL_KEYS = [
    "pca_explained_variance_95",
    "participation_ratio",
    "shannon_entropy",
    "geometric_mean_eff_dimensionality",
    *[f"renyi_eff_dimensionality_alpha_{a}" for a in range(2, 6)],
]


class TestPrivateSpectralHelpers:
    def test_streaming_matches_svd_bulk_spectrum(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 32)) * np.linspace(3.0, 0.2, 32)
        X = X + 10.0  # uncentered

        evals_cov = _do_streaming_covariance(X)
        evals_svd = _do_svd(X - X.mean(axis=0))

        k = min(len(evals_cov), len(evals_svd))
        cum_cov = np.cumsum(evals_cov[:k])
        total = cum_cov[-1]
        n_keep = int(np.searchsorted(cum_cov / total, 0.999) + 1)

        np.testing.assert_allclose(
            evals_cov[:n_keep],
            evals_svd[:n_keep],
            rtol=1e-5,
            atol=1e-8,
        )

    def test_streaming_is_stable_for_large_mean_offset(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((500, 32)) + 1e9

        evals_cov = _do_streaming_covariance(X, batch_size=37)
        evals_svd = _do_svd(X - X.mean(axis=0))

        np.testing.assert_allclose(evals_cov, evals_svd, rtol=1e-7, atol=1e-8)

    def test_batching_matches_oneshot(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((1000, 40)) + 5.0
        a = _do_streaming_covariance(X, batch_size=None)
        b = _do_streaming_covariance(X, batch_size=128)
        c = _do_streaming_covariance(X, batch_size=7)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a, c, rtol=1e-10, atol=1e-12)

    def test_rank_deficient_nonnegative(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((200, 5))
        B = rng.standard_normal((5, 20))
        X = A @ B
        evals = _do_streaming_covariance(X)
        assert evals.shape == (20,)
        assert np.all(evals >= -1e-10)
        assert np.sum(evals > 1e-8) <= 5

    def test_do_svd_returns_eigenvalues(self):
        rng = np.random.default_rng(7)
        Xc = rng.standard_normal((100, 10))
        Xc = Xc - Xc.mean(axis=0)
        evals = _do_svd(Xc)
        s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
        np.testing.assert_allclose(evals, (s**2) / (Xc.shape[0] - 1), rtol=1e-10)


class TestComputeDimBackendSwap:
    def test_default_is_streaming_covariance(self):
        rng = np.random.default_rng(3)
        data = rng.standard_normal((80, 8))
        default = compute_dim(data)
        explicit = compute_dim(data, spectral_backend="streaming_covariance")
        for key in SPECTRAL_KEYS:
            assert default[key] == explicit[key]

    def test_spectral_metrics_agree_across_backends(self):
        rng = np.random.default_rng(4)
        data = rng.standard_normal((300, 16)) * np.linspace(2.0, 0.5, 16)
        data = data + 3.0

        cov = compute_dim(data, spectral_backend="streaming_covariance")
        svd = compute_dim(data, spectral_backend="svd")

        assert cov["pca_explained_variance_95"] == svd["pca_explained_variance_95"]
        assert np.isclose(cov["participation_ratio"], svd["participation_ratio"], rtol=1e-5)
        assert np.isclose(cov["shannon_entropy"], svd["shannon_entropy"], rtol=1e-5)
        for alpha in range(2, 6):
            key = f"renyi_eff_dimensionality_alpha_{alpha}"
            assert np.isclose(cov[key], svd[key], rtol=1e-5)

    def test_batch_size_does_not_change_spectral_metrics(self):
        rng = np.random.default_rng(5)
        data = rng.standard_normal((400, 12))
        a = compute_dim(data, spectral_backend="streaming_covariance", batch_size=None)
        b = compute_dim(data, spectral_backend="streaming_covariance", batch_size=64)
        for key in SPECTRAL_KEYS:
            assert np.isclose(a[key], b[key], rtol=1e-10)

    def test_same_result_keys_for_both_backends(self):
        rng = np.random.default_rng(6)
        data = rng.standard_normal((60, 5))
        cov = compute_dim(data, spectral_backend="streaming_covariance")
        svd = compute_dim(data, spectral_backend="svd")
        assert set(cov) == set(svd)

    def test_wide_data_warns_and_falls_back_to_svd(self):
        rng = np.random.default_rng(9)
        data = rng.standard_normal((20, 30))

        with pytest.warns(RuntimeWarning, match="falling back to the SVD"):
            cov = compute_dim(data, spectral_backend="streaming_covariance")
        svd = compute_dim(data, spectral_backend="svd")

        for key in SPECTRAL_KEYS:
            assert np.isclose(cov[key], svd[key], rtol=1e-12)

    def test_invalid_backend_raises(self):
        data = np.random.default_rng(0).standard_normal((20, 3))
        with pytest.raises(ValueError, match="spectral_backend"):
            compute_dim(data, spectral_backend="qr")

    def test_valid_backend_names(self):
        assert _VALID_SPECTRAL_BACKENDS == ("streaming_covariance", "svd")
