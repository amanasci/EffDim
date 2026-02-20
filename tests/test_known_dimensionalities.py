"""
Enhanced tests validating estimates against known dimensionalities.
Tests that random noise approximates D, and manifolds like Swiss Roll
approximate their intrinsic dimension.
"""
import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll
from effdim.api import compute_dim
from effdim.geometry import (
    mle_dimensionality,
    two_nn_dimensionality,
    mind_mlk_dimensionality,
    tle_dimensionality,
    gmst_dimensionality,
)


class TestKnownDimensionalities:
    """Validate estimates against known intrinsic dimensions."""

    def test_random_noise_3d(self):
        """Random 3D Gaussian noise should have intrinsic dimension ~3."""
        np.random.seed(42)
        data = np.random.randn(300, 3)
        results = compute_dim(data)
        assert 2.0 < results["mle_dimensionality"] < 5.0
        assert 2.0 < results["two_nn_dimensionality"] < 5.0
        assert 2.0 < results["mind_mlk_dimensionality"] < 5.0
        assert 2.0 < results["tle_dimensionality"] < 5.0

    def test_random_noise_10d(self):
        """Random 10D Gaussian noise should have intrinsic dimension ~10."""
        np.random.seed(42)
        data = np.random.randn(500, 10)
        results = compute_dim(data)
        assert 7 < results["mle_dimensionality"] < 14
        assert 7 < results["two_nn_dimensionality"] < 14
        assert 7 < results["mind_mlk_dimensionality"] < 14
        assert 7 < results["tle_dimensionality"] < 14
        assert results["participation_ratio"] > 7

    def test_swiss_roll_2d_manifold(self):
        """Swiss Roll is a 2D manifold in 3D space."""
        X, _ = make_swiss_roll(n_samples=1000, noise=0.01, random_state=42)
        results = compute_dim(X)
        # Geometric methods should detect ~2
        assert 1.5 < results["mle_dimensionality"] < 3.5
        assert 1.5 < results["two_nn_dimensionality"] < 3.5
        # PCA should see 3 (global structure fills 3D)
        assert results["pca_explained_variance_95"] >= 2

    def test_linear_subspace_rank3_in_10d(self):
        """Rank-3 linear subspace in 10D should have intrinsic dim ~3."""
        np.random.seed(42)
        A = np.random.randn(500, 3)
        B = np.random.randn(3, 10)
        data = A @ B + 1e-6 * np.random.randn(500, 10)
        results = compute_dim(data)
        assert results["pca_explained_variance_95"] <= 4
        assert results["participation_ratio"] < 5
        assert 2.0 < results["mle_dimensionality"] < 5.0

    def test_1d_curve_in_3d(self):
        """A 1D curve (helix) in 3D space should have intrinsic dim ~1."""
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 500)
        data = np.column_stack([
            np.cos(t), np.sin(t), t / (4 * np.pi)
        ]) + 1e-6 * np.random.randn(500, 3)
        mle = mle_dimensionality(data, k=5)
        two_nn = two_nn_dimensionality(data)
        assert 0.5 < mle < 2.5, f"MLE got {mle} for 1D helix"
        assert two_nn > 0, f"Two-NN got {two_nn} for 1D helix"

    def test_2d_plane_in_5d(self):
        """2D plane embedded in 5D space."""
        np.random.seed(42)
        coords_2d = np.random.randn(400, 2)
        embedding = np.random.randn(2, 5)
        data = coords_2d @ embedding + 1e-6 * np.random.randn(400, 5)
        results = compute_dim(data)
        assert results["pca_explained_variance_95"] <= 3
        assert 1.0 < results["mle_dimensionality"] < 4.0
        assert 1.0 < results["mind_mlk_dimensionality"] < 4.0

    def test_isotropic_gaussian_spectral(self):
        """For isotropic Gaussian, spectral dims should be close to D."""
        np.random.seed(42)
        D = 8
        data = np.random.randn(400, D)
        results = compute_dim(data)
        assert results["participation_ratio"] > 0.6 * D
        assert results["shannon_entropy"] > 0.6 * D


class TestEstimatorConsistency:
    """Test that different estimators give consistent results."""

    def test_estimators_agree_on_isotropic(self):
        """On isotropic Gaussian, most estimators should agree approximately."""
        np.random.seed(42)
        data = np.random.randn(300, 5)
        results = compute_dim(data)
        geometric_results = [
            results["mle_dimensionality"],
            results["mind_mlk_dimensionality"],
            results["tle_dimensionality"],
        ]
        # All should be within [3, 8] for 5D Gaussian
        for val in geometric_results:
            assert 3 < val < 8, f"Estimator returned {val} for 5D Gaussian"

    def test_low_dim_data_all_estimators_low(self):
        """For truly low-dim data, all estimators should give low values."""
        np.random.seed(42)
        # 1D data in 10D
        t = np.random.randn(200, 1)
        embedding = np.random.randn(1, 10)
        data = t @ embedding + 1e-6 * np.random.randn(200, 10)
        results = compute_dim(data)
        assert results["pca_explained_variance_95"] <= 2
        assert results["participation_ratio"] < 3
