import numpy as np
import pytest
from effdim.api import compute_dim

def test_compute_dim_small_data():
    """Test compute_dim with small data to trigger np.linalg.svd path."""
    # Create a small random dataset
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 10))

    # Compute dimensions
    results = compute_dim(data)

    # Check if results dictionary contains expected keys
    expected_keys = [
        "pca_explained_variance_95",
        "participation_ratio",
        "shannon_entropy",
        "geometric_mean_eff_dimensionality",
        "stable_rank",
        "numerical_rank",
        "cumulative_eigenvalue_ratio",
        "mle_dimensionality",
        "two_nn_dimensionality"
    ]

    for key in expected_keys:
        assert key in results, f"Missing key: {key}"
        assert isinstance(results[key], (float, np.floating, int, np.integer)), f"Result for {key} is not a number"

def test_compute_dim_list_input():
    """Ensure identical results for list of arrays vs equivalent single array."""
    rng = np.random.default_rng(42)
    data_list = [rng.standard_normal((10, 5)) for _ in range(5)]
    data_array = np.vstack(data_list)

    results_list = compute_dim(data_list)
    results_array = compute_dim(data_array)
    
    for key in results_list:
        assert np.isclose(results_list[key], results_array[key], rtol=1e-10), (
            f"Results differ for '{key}': list={results_list[key]}, "
            f"array={results_array[key]}"
        )

def test_compute_dim_centered():
    """Ensure identical results for uncentered (shifted) vs standard centered data."""
    rng = np.random.default_rng(42)
    data_standard = rng.standard_normal((50, 5))
    data_shifted = data_standard + 100  # Shift mean

    results_standard = compute_dim(data_standard)
    results_shifted = compute_dim(data_shifted)
    
    for key in results_standard:
        assert np.isclose(results_standard[key], results_shifted[key], rtol=1e-10), (
            f"Results differ for '{key}': standard={results_standard[key]}, "
            f"shifted={results_shifted[key]}"
        )
