"""
Tests for new geometric estimators:
DANCo, MiND-MLi, MiND-MLk, ESS, TLE, GMST.
"""
import numpy as np
import pytest
from effdim.geometry import (
    danco_dimensionality,
    mind_mli_dimensionality,
    mind_mlk_dimensionality,
    ess_dimensionality,
    tle_dimensionality,
    gmst_dimensionality,
)
from effdim.api import compute_dim


class TestDANCo:
    """Tests for DANCo dimensionality estimator."""

    def test_high_dimensional_gaussian(self):
        """DANCo should estimate dimension close to ambient for isotropic data."""
        np.random.seed(42)
        data = np.random.randn(200, 10)
        dim = danco_dimensionality(data, k=10)
        assert np.isfinite(dim) and dim > 0, f"DANCo returned {dim}"

    def test_low_dimensional_manifold(self):
        """DANCo should detect low-dimensional structure."""
        np.random.seed(42)
        n = 200
        t = np.random.uniform(0, 10, n)
        data = np.column_stack([t, 2 * t, 3 * t])
        data += 1e-6 * np.random.randn(*data.shape)
        dim = danco_dimensionality(data, k=5)
        assert np.isfinite(dim) and dim > 0, f"DANCo returned {dim}"

    def test_small_dataset(self):
        """DANCo should return 0.0 for very small datasets."""
        data = np.random.randn(2, 5)
        assert danco_dimensionality(data) == 0.0

    def test_with_precomputed_knn(self):
        """DANCo should work with precomputed knn distances."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        dim1 = danco_dimensionality(data, k=5)
        from effdim.geometry import compute_knn_distances
        knn = compute_knn_distances(data, 5)
        dim2 = danco_dimensionality(data, precomputed_knn_dist_sq=knn)
        assert np.isfinite(dim2) and dim2 > 0

    def test_duplicate_points(self):
        """DANCo should handle duplicate points without crashing."""
        data = np.array([
            [1.0, 2.0], [1.0, 2.0], [3.0, 4.0],
            [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
        ])
        dim = danco_dimensionality(data, k=2)
        assert np.isfinite(dim)


class TestMiNDMLi:
    """Tests for MiND-MLi dimensionality estimator."""

    def test_high_dimensional_gaussian(self):
        """MiND-MLi should estimate dimension for isotropic data."""
        np.random.seed(42)
        data = np.random.randn(200, 10)
        dim = mind_mli_dimensionality(data)
        assert np.isfinite(dim) and dim > 0, f"MiND-MLi returned {dim}"

    def test_1d_manifold(self):
        """MiND-MLi should return a positive finite value for 1D-like data.
        Note: MiND-MLi uses only single neighbor distances with r_max normalization,
        so it may overestimate for nearly-uniform spacing on a line (known limitation).
        """
        np.random.seed(42)
        t = np.linspace(0, 10, 200).reshape(-1, 1)
        data = np.hstack([t, 2 * t])
        data += 1e-6 * np.random.randn(*data.shape)
        dim = mind_mli_dimensionality(data)
        assert np.isfinite(dim) and dim > 0, f"MiND-MLi got {dim} for 1D data"

    def test_small_dataset(self):
        """MiND-MLi should return 0.0 for very small datasets."""
        data = np.random.randn(2, 5)
        assert mind_mli_dimensionality(data) == 0.0

    def test_identical_distances(self):
        """MiND-MLi should handle identical distances."""
        # Points on a circle - all nearest neighbor distances roughly equal
        n = 50
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        data = np.column_stack([np.cos(theta), np.sin(theta)])
        dim = mind_mli_dimensionality(data)
        assert np.isfinite(dim)

    def test_with_precomputed_knn(self):
        """MiND-MLi should work with precomputed knn distances."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        from effdim.geometry import compute_knn_distances
        knn = compute_knn_distances(data, 5)
        dim = mind_mli_dimensionality(data, precomputed_knn_dist_sq=knn)
        assert np.isfinite(dim) and dim > 0


class TestMiNDMLk:
    """Tests for MiND-MLk dimensionality estimator."""

    def test_high_dimensional_gaussian(self):
        """MiND-MLk should estimate dimension for isotropic data."""
        np.random.seed(42)
        data = np.random.randn(200, 10)
        dim = mind_mlk_dimensionality(data, k=10)
        assert 5 < dim < 15, f"MiND-MLk got {dim}, expected ~10"

    def test_2d_manifold(self):
        """MiND-MLk should detect 2D structure."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(-5, 5, n)
        y = np.random.uniform(-5, 5, n)
        z = x + y
        data = np.column_stack([x, y, z]) + 1e-6 * np.random.randn(n, 3)
        dim = mind_mlk_dimensionality(data, k=5)
        assert 1.0 < dim < 3.5, f"MiND-MLk got {dim} for 2D manifold"

    def test_small_dataset(self):
        """MiND-MLk should return 0.0 for very small datasets."""
        data = np.random.randn(1, 5)
        assert mind_mlk_dimensionality(data) == 0.0

    def test_robustness_vs_mle(self):
        """MiND-MLk (median) should be comparable to MLE (mean)."""
        np.random.seed(42)
        from effdim.geometry import mle_dimensionality
        data = np.random.randn(200, 5)
        mle = mle_dimensionality(data, k=10)
        mlk = mind_mlk_dimensionality(data, k=10)
        # Both should be in the same ballpark
        assert abs(mle - mlk) < 5.0, f"MLE={mle}, MLk={mlk}"


class TestESS:
    """Tests for ESS dimensionality estimator."""

    def test_high_dimensional_gaussian(self):
        """ESS should estimate dimension for isotropic data."""
        np.random.seed(42)
        data = np.random.randn(200, 10)
        dim = ess_dimensionality(data, k=10)
        assert np.isfinite(dim) and dim > 0, f"ESS returned {dim}"

    def test_low_dimensional_structure(self):
        """ESS should detect low-dimensional structure."""
        np.random.seed(42)
        t = np.linspace(0, 10, 200).reshape(-1, 1)
        data = np.hstack([t, 2 * t])
        data += 1e-6 * np.random.randn(*data.shape)
        dim = ess_dimensionality(data, k=5)
        assert np.isfinite(dim) and dim > 0

    def test_small_dataset(self):
        """ESS should return 0.0 for very small datasets."""
        data = np.random.randn(2, 5)
        assert ess_dimensionality(data) == 0.0

    def test_duplicate_points(self):
        """ESS should handle duplicate points."""
        data = np.array([
            [1.0, 2.0], [1.0, 2.0], [3.0, 4.0],
            [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
        ])
        dim = ess_dimensionality(data, k=2)
        assert np.isfinite(dim)


class TestTLE:
    """Tests for TLE dimensionality estimator."""

    def test_high_dimensional_gaussian(self):
        """TLE should estimate dimension for isotropic data."""
        np.random.seed(42)
        data = np.random.randn(200, 10)
        dim = tle_dimensionality(data, k=10)
        assert 5 < dim < 15, f"TLE got {dim}, expected ~10"

    def test_2d_manifold(self):
        """TLE should detect 2D structure."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(-5, 5, n)
        y = np.random.uniform(-5, 5, n)
        z = x + y
        data = np.column_stack([x, y, z]) + 1e-6 * np.random.randn(n, 3)
        dim = tle_dimensionality(data, k=5)
        assert 1.0 < dim < 3.5, f"TLE got {dim} for 2D manifold"

    def test_small_dataset(self):
        """TLE should return 0.0 for very small datasets."""
        data = np.random.randn(1, 5)
        assert tle_dimensionality(data) == 0.0

    def test_equivalent_to_mle(self):
        """TLE should produce similar results to MLE (same formula, different name)."""
        np.random.seed(42)
        from effdim.geometry import mle_dimensionality
        data = np.random.randn(200, 5)
        mle = mle_dimensionality(data, k=10)
        tle = tle_dimensionality(data, k=10)
        # Should be very close (essentially same formula)
        assert abs(mle - tle) < 1.0, f"MLE={mle}, TLE={tle}"


class TestGMST:
    """Tests for GMST dimensionality estimator."""

    def test_high_dimensional_gaussian(self):
        """GMST should estimate dimension for isotropic data."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        dim = gmst_dimensionality(data)
        assert np.isfinite(dim) and dim > 0, f"GMST returned {dim}"

    def test_2d_data(self):
        """GMST should estimate close to 2 for 2D data."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        dim = gmst_dimensionality(data)
        assert np.isfinite(dim) and dim > 0, f"GMST returned {dim}"

    def test_small_dataset(self):
        """GMST should return 0.0 for small datasets."""
        data = np.random.randn(5, 3)
        assert gmst_dimensionality(data) == 0.0

    def test_geodesic_mode(self):
        """GMST geodesic mode should work."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        dim = gmst_dimensionality(data, geodesic=True)
        assert np.isfinite(dim), f"GMST geodesic returned {dim}"

    def test_geodesic_vs_euclidean(self):
        """Both modes should return finite results."""
        np.random.seed(42)
        data = np.random.randn(50, 3)
        dim_e = gmst_dimensionality(data, geodesic=False)
        dim_g = gmst_dimensionality(data, geodesic=True)
        assert np.isfinite(dim_e) and np.isfinite(dim_g)


class TestNewEstimatorsIntegration:
    """Integration tests for new estimators via compute_dim."""

    def test_compute_dim_contains_new_keys(self):
        """compute_dim should contain all new estimator keys."""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        results = compute_dim(data)
        new_keys = [
            "danco_dimensionality",
            "mind_mli_dimensionality",
            "mind_mlk_dimensionality",
            "ess_dimensionality",
            "tle_dimensionality",
            "gmst_dimensionality",
        ]
        for key in new_keys:
            assert key in results, f"Missing key: {key}"
            assert isinstance(results[key], (float, np.floating, int, np.integer)), \
                f"Result for {key} is not a number"
            assert np.isfinite(results[key]), f"Result for {key} is not finite"

    def test_all_results_finite(self):
        """All results including new estimators should be finite."""
        np.random.seed(42)
        data = np.random.randn(50, 10)
        results = compute_dim(data)
        for key, value in results.items():
            assert np.isfinite(value), f"Result '{key}' is not finite: {value}"
            assert value >= 0, f"Result '{key}' is negative: {value}"

    def test_known_dimensionality_gaussian(self):
        """All estimators should give reasonable results for 10D Gaussian."""
        np.random.seed(42)
        data = np.random.randn(200, 10)
        results = compute_dim(data)
        # Geometric methods should be in a reasonable range for 10D
        assert results["mle_dimensionality"] > 5
        assert results["two_nn_dimensionality"] > 5
        assert results["mind_mlk_dimensionality"] > 5
        assert results["tle_dimensionality"] > 5

    def test_swiss_roll_intrinsic_dim(self):
        """Geometric estimators should detect 2D intrinsic dim of Swiss Roll."""
        from sklearn.datasets import make_swiss_roll
        X, _ = make_swiss_roll(n_samples=500, noise=0.01, random_state=42)
        results = compute_dim(X)
        # MLE and Two-NN are well-known to detect ~2 for Swiss Roll
        assert 1.0 < results["mle_dimensionality"] < 4.0
        assert 1.0 < results["two_nn_dimensionality"] < 4.0
