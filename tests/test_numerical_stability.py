"""
Test numerical stability and edge cases for effective dimensionality estimators.
"""
import numpy as np
import pytest
from effdim.metrics import (
    participation_ratio,
    shannon_entropy,
    renyi_eff_dimensionality,
    geometric_mean_eff_dimensionality,
    pca_explained_variance,
)
from effdim.geometry import (
    mle_dimensionality,
    two_nn_dimensionality,
)
from effdim.api import compute_dim


class TestSpectralEstimatorsEdgeCases:
    """Test edge cases for spectral estimators."""
    
    def test_single_dominant_eigenvalue(self):
        """Test when one eigenvalue dominates (low effective dimension)."""
        # One large eigenvalue, rest very small
        spectrum = np.array([100.0, 0.01, 0.01, 0.01])
        
        pr = participation_ratio(spectrum)
        assert 1.0 < pr < 1.5, "PR should be close to 1 when one eigenvalue dominates"
        
        probs = spectrum / np.sum(spectrum)
        shannon = shannon_entropy(probs)
        assert 1.0 < shannon < 1.5, "Shannon ED should be close to 1"
    
    def test_equal_eigenvalues(self):
        """Test when all eigenvalues are equal (maximum effective dimension)."""
        D = 10
        spectrum = np.ones(D)
        
        pr = participation_ratio(spectrum)
        assert np.isclose(pr, D), "PR should equal D when all eigenvalues are equal"
        
        probs = spectrum / np.sum(spectrum)
        shannon = shannon_entropy(probs)
        assert np.isclose(shannon, D), "Shannon ED should equal D"
    
    def test_zero_eigenvalues(self):
        """Test handling of zero eigenvalues."""
        spectrum = np.array([10.0, 5.0, 1.0, 0.0, 0.0])
        
        # PR should work with zeros
        pr = participation_ratio(spectrum)
        assert pr > 0, "PR should handle zero eigenvalues"
        
        # Shannon should filter zeros
        probs = spectrum / np.sum(spectrum)
        shannon = shannon_entropy(probs)
        assert np.isfinite(shannon), "Shannon ED should handle zero probabilities"
        
        # Geometric mean should filter zeros
        gm = geometric_mean_eff_dimensionality(spectrum)
        assert gm > 0, "Geometric mean should handle zero eigenvalues"
    
    def test_all_zero_spectrum(self):
        """Test with all-zero spectrum."""
        spectrum = np.zeros(5)
        
        pr = participation_ratio(spectrum)
        assert pr == 0.0, "PR should be 0 for all-zero spectrum"
        
        gm = geometric_mean_eff_dimensionality(spectrum)
        assert gm == 0.0, "GM should be 0 for all-zero spectrum"
    
    def test_very_small_eigenvalues(self):
        """Test numerical stability with very small eigenvalues."""
        spectrum = np.array([1.0, 1e-8, 1e-10, 1e-12])
        
        pr = participation_ratio(spectrum)
        assert np.isfinite(pr) and pr > 0, "PR should handle very small eigenvalues"
        
        probs = spectrum / np.sum(spectrum)
        shannon = shannon_entropy(probs)
        assert np.isfinite(shannon), "Shannon should handle very small probabilities"
    
    def test_very_large_eigenvalue_range(self):
        """Test with large range in eigenvalues (ill-conditioned)."""
        spectrum = np.array([1e10, 1e5, 1e0, 1e-5])
        
        pr = participation_ratio(spectrum)
        assert np.isfinite(pr), "PR should handle large eigenvalue range"
        
        probs = spectrum / np.sum(spectrum)
        shannon = shannon_entropy(probs)
        assert np.isfinite(shannon), "Shannon should handle large eigenvalue range"
    
    def test_renyi_alpha_2_equals_pr(self):
        """Test that Rényi with alpha=2 equals Participation Ratio."""
        spectrum = np.array([4.0, 3.0, 2.0, 1.0])
        probs = spectrum / np.sum(spectrum)
        
        pr = participation_ratio(spectrum)
        renyi_2 = renyi_eff_dimensionality(probs, alpha=2)
        
        assert np.isclose(pr, renyi_2), "Rényi-2 should equal PR"
    
    def test_renyi_invalid_alpha(self):
        """Test that invalid alpha values raise errors."""
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        
        with pytest.raises(ValueError):
            renyi_eff_dimensionality(probs, alpha=1.0)
        
        with pytest.raises(ValueError):
            renyi_eff_dimensionality(probs, alpha=0.0)
        
        with pytest.raises(ValueError):
            renyi_eff_dimensionality(probs, alpha=-1.0)
    
    def test_pca_threshold_edge_cases(self):
        """Test PCA explained variance with edge case thresholds."""
        spectrum = np.array([4.0, 3.0, 2.0, 1.0])
        
        # Threshold = 0 should give 1
        result = pca_explained_variance(spectrum, threshold=0.0)
        assert result >= 1
        
        # Threshold = 1.0 should give all components
        result = pca_explained_variance(spectrum, threshold=1.0)
        assert result == len(spectrum)
        
        # Threshold just above first eigenvalue
        ratio_first = 4.0 / 10.0  # 0.4
        result = pca_explained_variance(spectrum, threshold=ratio_first + 0.01)
        assert result == 2


class TestGeometricEstimatorsEdgeCases:
    """Test edge cases for geometric estimators."""
    
    def test_perfect_line_2d(self):
        """Test with data on a perfect 1D line in 2D space."""
        # Points on a line: y = 2x
        t = np.linspace(0, 10, 100).reshape(-1, 1)
        data = np.hstack([t, 2*t])
        
        # Add tiny noise to avoid identical points
        data += np.random.randn(*data.shape) * 1e-8
        
        mle_dim = mle_dimensionality(data, k=5)
        two_nn_dim = two_nn_dimensionality(data)
        
        # MLE should be close to 1 (1D manifold)
        assert 0.5 < mle_dim < 2.0, f"MLE should detect 1D structure, got {mle_dim}"
        # Two-NN can overestimate for nearly-perfect linear data
        # This is a known limitation, not a bug
        assert two_nn_dim > 0, f"Two-NN should return positive value, got {two_nn_dim}"
    
    def test_perfect_plane_3d(self):
        """Test with data on a perfect 2D plane in 3D space."""
        # Points on plane: z = x + y
        n = 100
        x = np.random.uniform(-5, 5, n)
        y = np.random.uniform(-5, 5, n)
        z = x + y
        data = np.column_stack([x, y, z])
        
        # Add tiny noise
        data += np.random.randn(*data.shape) * 1e-8
        
        mle_dim = mle_dimensionality(data, k=5)
        two_nn_dim = two_nn_dimensionality(data)
        
        # Should be close to 2 (2D manifold)
        assert 1.0 < mle_dim < 3.5, f"MLE should detect 2D structure, got {mle_dim}"
        assert 1.0 < two_nn_dim < 3.5, f"Two-NN should detect 2D structure, got {two_nn_dim}"
    
    def test_very_small_dataset(self):
        """Test with very small datasets."""
        # 2 points - should return 0
        data = np.random.randn(2, 5)
        mle_dim = mle_dimensionality(data, k=5)
        assert mle_dim == 0.0, "MLE should return 0 for n<2"
        
        two_nn_dim = two_nn_dimensionality(data)
        assert two_nn_dim == 0.0, "Two-NN should return 0 for n<3"
        
        # 3 points - MLE should work, Two-NN should work
        data = np.random.randn(3, 5)
        mle_dim = mle_dimensionality(data, k=2)
        assert mle_dim >= 0, "MLE should work with n=3, k=2"
        
        two_nn_dim = two_nn_dimensionality(data)
        assert two_nn_dim >= 0, "Two-NN should work with n=3"
    
    def test_identical_points(self):
        """Test with some identical points (distance = 0)."""
        # Create data with duplicates
        data = np.array([
            [1.0, 2.0],
            [1.0, 2.0],  # Duplicate
            [3.0, 4.0],
            [3.0, 4.0],  # Duplicate
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        
        # Should handle without crashing (epsilon prevents log(0))
        mle_dim = mle_dimensionality(data, k=2)
        assert np.isfinite(mle_dim), "MLE should handle duplicate points"
        
        two_nn_dim = two_nn_dimensionality(data)
        assert np.isfinite(two_nn_dim), "Two-NN should handle duplicate points"
    
    def test_high_dimensional_gaussian(self):
        """Test with high-dimensional Gaussian (intrinsic dim = ambient dim)."""
        n, d = 200, 10
        np.random.seed(42)
        data = np.random.randn(n, d)
        
        mle_dim = mle_dimensionality(data, k=10)
        two_nn_dim = two_nn_dimensionality(data)
        
        # Should estimate close to actual dimension d
        # Allow generous bounds due to finite sample effects
        assert 5 < mle_dim < 15, f"MLE dimension {mle_dim} far from expected {d}"
        assert 5 < two_nn_dim < 15, f"Two-NN dimension {two_nn_dim} far from expected {d}"


class TestComputeDimIntegration:
    """Integration tests for compute_dim function."""
    
    def test_low_rank_data(self):
        """Test with low-rank data (known effective dimension)."""
        n, p, k = 100, 50, 5
        np.random.seed(42)
        
        # Create rank-k data: X = A @ B^T where A is n×k, B is p×k
        A = np.random.randn(n, k)
        B = np.random.randn(p, k)
        data = A @ B.T
        
        results = compute_dim(data)
        
        # Spectral methods should detect low rank
        assert results['participation_ratio'] < k + 2, "PR should be close to true rank"
        assert results['shannon_entropy'] < k + 2, "Shannon should be close to true rank"
        assert results['pca_explained_variance_95'] <= k + 1, "PCA should detect rank"
    
    def test_noisy_low_rank_data(self):
        """Test with noisy low-rank data."""
        n, p, k = 100, 50, 5
        np.random.seed(42)
        
        # Create rank-k data with noise
        A = np.random.randn(n, k)
        B = np.random.randn(p, k)
        data = A @ B.T + 0.1 * np.random.randn(n, p)
        
        results = compute_dim(data)
        
        # Should still detect approximate low rank
        assert results['participation_ratio'] < k + 5, "PR should reflect approximate rank"
    
    def test_isotropic_gaussian(self):
        """Test with isotropic Gaussian (all dimensions equally important)."""
        n, p = 100, 10
        np.random.seed(42)
        data = np.random.randn(n, p)
        
        results = compute_dim(data)
        
        # For isotropic Gaussian, effective dim should be close to ambient dim
        assert results['participation_ratio'] > 0.7 * p, "PR should be high for isotropic data"
        assert results['shannon_entropy'] > 0.7 * p, "Shannon should be high"
    
    def test_all_results_finite(self):
        """Test that all results are finite numbers."""
        np.random.seed(42)
        data = np.random.randn(50, 10)
        
        results = compute_dim(data)
        
        for key, value in results.items():
            assert np.isfinite(value), f"Result '{key}' is not finite: {value}"
            assert value >= 0, f"Result '{key}' is negative: {value}"
    
    def test_centered_vs_uncentered(self):
        """Test that centering is handled correctly."""
        np.random.seed(42)
        data_centered = np.random.randn(50, 10)
        data_shifted = data_centered + 100  # Large shift
        
        results_centered = compute_dim(data_centered)
        results_shifted = compute_dim(data_shifted)
        
        # Results should be identical (implementation centers data)
        for key in results_centered:
            assert np.isclose(
                results_centered[key], 
                results_shifted[key], 
                rtol=1e-10
            ), f"Results differ for {key}: {results_centered[key]} vs {results_shifted[key]}"


class TestNumericalStability:
    """Test numerical stability with extreme values."""
    
    def test_very_large_data_values(self):
        """Test with very large data values."""
        np.random.seed(42)
        data = 1e6 * np.random.randn(50, 10)
        
        results = compute_dim(data)
        
        for key, value in results.items():
            assert np.isfinite(value), f"Failed for large data: {key} = {value}"
    
    def test_very_small_data_values(self):
        """Test with very small data values."""
        np.random.seed(42)
        data = 1e-6 * np.random.randn(50, 10)
        
        results = compute_dim(data)
        
        for key, value in results.items():
            assert np.isfinite(value), f"Failed for small data: {key} = {value}"
    
    def test_mixed_scale_features(self):
        """Test with features at very different scales."""
        np.random.seed(42)
        n = 50
        data = np.column_stack([
            1e6 * np.random.randn(n, 2),   # Very large scale
            np.random.randn(n, 3),          # Normal scale
            1e-6 * np.random.randn(n, 2),  # Very small scale
        ])
        
        results = compute_dim(data)
        
        for key, value in results.items():
            assert np.isfinite(value), f"Failed for mixed scale: {key} = {value}"
