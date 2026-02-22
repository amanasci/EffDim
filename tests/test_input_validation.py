"""
Tests for input validation and error handling in the effdim package.
Covers invalid input types, shapes, and edge cases.
"""
import numpy as np
import pytest
from effdim.api import compute_dim
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
    danco_dimensionality,
    mind_mli_dimensionality,
    mind_mlk_dimensionality,
    ess_dimensionality,
    tle_dimensionality,
    gmst_dimensionality,
)


class TestComputeDimInputValidation:
    """Test compute_dim with invalid and edge-case inputs."""

    def test_invalid_input_type_raises(self):
        """compute_dim should raise ValueError for non-array, non-list inputs."""
        with pytest.raises(ValueError):
            compute_dim("not an array")

        with pytest.raises(ValueError):
            compute_dim({"key": "value"})

        with pytest.raises(ValueError):
            compute_dim(42)

    def test_list_of_arrays_input(self):
        """compute_dim should accept a list of 2D arrays."""
        rng = np.random.default_rng(0)
        data = [rng.standard_normal((20, 5)) for _ in range(3)]
        results = compute_dim(data)
        assert "participation_ratio" in results
        assert results["participation_ratio"] > 0

    def test_single_array_input(self):
        """compute_dim should accept a single 2D numpy array."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 8))
        results = compute_dim(data)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_result_contains_all_expected_keys(self):
        """compute_dim result should contain all documented keys."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 5))
        results = compute_dim(data)
        expected_keys = [
            "pca_explained_variance_95",
            "participation_ratio",
            "shannon_entropy",
            "geometric_mean_eff_dimensionality",
            "mle_dimensionality",
            "two_nn_dimensionality",
            "danco_dimensionality",
            "mind_mli_dimensionality",
            "mind_mlk_dimensionality",
            "ess_dimensionality",
            "tle_dimensionality",
            "gmst_dimensionality",
            "renyi_eff_dimensionality_alpha_2",
            "renyi_eff_dimensionality_alpha_3",
            "renyi_eff_dimensionality_alpha_4",
            "renyi_eff_dimensionality_alpha_5",
        ]
        for key in expected_keys:
            assert key in results, f"Missing expected key: {key}"

    def test_all_results_are_numbers(self):
        """All values in compute_dim result should be numeric."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 5))
        results = compute_dim(data)
        for key, value in results.items():
            assert isinstance(
                value, (float, int, np.floating, np.integer)
            ), f"Result for '{key}' is not numeric: {type(value)}"

    def test_all_results_non_negative(self):
        """All dimensionality estimates should be non-negative."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 5))
        results = compute_dim(data)
        for key, value in results.items():
            assert value >= 0, f"Result for '{key}' is negative: {value}"

    def test_uncentered_data_handled(self):
        """compute_dim should internally center data with large mean shift."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 5)) + 1000
        results = compute_dim(data)
        for key, value in results.items():
            assert np.isfinite(value), f"Result for '{key}' not finite after centering"

    def test_single_feature_column(self):
        """compute_dim should handle data with a single feature."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 1))
        results = compute_dim(data)
        # With 1 feature, PCA dim should be 1
        assert results["pca_explained_variance_95"] == 1
        assert results["participation_ratio"] > 0

    def test_square_matrix(self):
        """compute_dim should work when n_samples == n_features."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((20, 20))
        results = compute_dim(data)
        assert results["participation_ratio"] > 0


class TestMetricsInputEdgeCases:
    """Test metrics functions with edge-case inputs."""

    def test_participation_ratio_single_nonzero(self):
        """PR should be 1.0 for a spectrum with only one non-zero eigenvalue."""
        spectrum = np.array([5.0, 0.0, 0.0])
        pr = participation_ratio(spectrum)
        assert np.isclose(pr, 1.0)

    def test_participation_ratio_uniform(self):
        """PR should equal D for uniform spectrum."""
        D = 7
        spectrum = np.ones(D)
        pr = participation_ratio(spectrum)
        assert np.isclose(pr, D)

    def test_shannon_entropy_uniform(self):
        """Shannon entropy ED should equal D for uniform distribution."""
        D = 6
        probs = np.ones(D) / D
        ed = shannon_entropy(probs)
        assert np.isclose(ed, D, rtol=1e-6)

    def test_shannon_entropy_concentrated(self):
        """Shannon entropy ED close to 1 when probability mass is concentrated."""
        probs = np.array([0.999, 0.0005, 0.0005])
        ed = shannon_entropy(probs)
        assert 1.0 < ed < 1.5

    def test_renyi_valid_alpha_half(self):
        """renyi_eff_dimensionality should work with alpha=0.5."""
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        result = renyi_eff_dimensionality(probs, alpha=0.5)
        assert np.isfinite(result) and result > 0

    def test_renyi_valid_integer_alphas(self):
        """renyi_eff_dimensionality should return positive finite values for alpha 2-5."""
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        for alpha in [2, 3, 4, 5]:
            result = renyi_eff_dimensionality(probs, alpha=alpha)
            assert np.isfinite(result) and result > 0, f"alpha={alpha} returned {result}"

    def test_renyi_ordering(self):
        """Rényi dimensionality should decrease as alpha increases for non-uniform dist."""
        probs = np.array([0.5, 0.3, 0.15, 0.05])
        alphas = [2, 3, 4, 5]
        values = [renyi_eff_dimensionality(probs, alpha=a) for a in alphas]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"Rényi not non-increasing: alpha={alphas[i]} gave {values[i]}, "
                f"alpha={alphas[i+1]} gave {values[i+1]}"
            )

    def test_geometric_mean_known_value(self):
        """geometric_mean_eff_dimensionality should return am/gm ratio."""
        spectrum = np.array([4.0, 1.0])
        # am = (4+1)/2 = 2.5, gm = sqrt(4*1) = 2.0, ratio = 1.25
        result = geometric_mean_eff_dimensionality(spectrum)
        expected = 2.5 / 2.0
        assert np.isclose(result, expected, rtol=1e-6)

    def test_geometric_mean_equal_values(self):
        """geometric_mean_eff_dimensionality should return 1.0 for uniform spectrum."""
        spectrum = np.array([3.0, 3.0, 3.0])
        result = geometric_mean_eff_dimensionality(spectrum)
        assert np.isclose(result, 1.0, rtol=1e-6)

    def test_pca_variance_threshold_50pct(self):
        """pca_explained_variance should work correctly for 50% threshold."""
        # Spectrum: [4, 2, 1, 1] -> total=8, cumsum=[4,6,7,8]
        # 50% = 4, so 1 component reaches 4/8=0.5
        spectrum = np.array([4.0, 2.0, 1.0, 1.0])
        result = pca_explained_variance(spectrum, threshold=0.5)
        assert result == 1

    def test_pca_variance_threshold_75pct(self):
        """pca_explained_variance at 75% threshold."""
        # Spectrum: [4, 2, 1, 1] -> total=8, cumsum=[4,6,7,8]
        # cumsum/8 = [0.5, 0.75, 0.875, 1.0]
        # 75% is reached at index 1 (0-based), so 2 components
        spectrum = np.array([4.0, 2.0, 1.0, 1.0])
        result = pca_explained_variance(spectrum, threshold=0.75)
        assert result == 2


class TestGeometricEstimatorsInputEdgeCases:
    """Test geometric estimators with edge-case inputs."""

    def test_mle_single_sample_returns_zero(self):
        """MLE should return 0.0 for single sample."""
        data = np.array([[1.0, 2.0, 3.0]])
        result = mle_dimensionality(data, k=1)
        assert result == 0.0

    def test_two_nn_two_samples_returns_zero(self):
        """Two-NN should return 0.0 for two samples."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = two_nn_dimensionality(data)
        assert result == 0.0

    def test_danco_two_samples_returns_zero(self):
        """DANCo should return 0.0 for two samples."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = danco_dimensionality(data)
        assert result == 0.0

    def test_mind_mli_two_samples_returns_zero(self):
        """MiND-MLi should return 0.0 for two samples."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = mind_mli_dimensionality(data)
        assert result == 0.0

    def test_mind_mlk_single_sample_returns_zero(self):
        """MiND-MLk should return 0.0 for a single sample."""
        data = np.array([[1.0, 2.0, 3.0]])
        result = mind_mlk_dimensionality(data)
        assert result == 0.0

    def test_ess_two_samples_returns_zero(self):
        """ESS should return 0.0 for two samples."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ess_dimensionality(data)
        assert result == 0.0

    def test_tle_single_sample_returns_zero(self):
        """TLE should return 0.0 for a single sample."""
        data = np.array([[1.0, 2.0, 3.0]])
        result = tle_dimensionality(data)
        assert result == 0.0

    def test_gmst_small_dataset_returns_zero(self):
        """GMST should return 0.0 for fewer than 10 samples."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((9, 3))
        result = gmst_dimensionality(data)
        assert result == 0.0

    def test_mle_positive_for_reasonable_data(self):
        """MLE should return a positive value for reasonable data."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 5))
        result = mle_dimensionality(data, k=5)
        assert result > 0

    def test_two_nn_positive_for_reasonable_data(self):
        """Two-NN should return a positive value for reasonable data."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 5))
        result = two_nn_dimensionality(data)
        assert result > 0
