import numpy as np
import pytest
from effdim.api import compute_dim

def test_compute_dim_small_data():
    """Test compute_dim with small data to trigger np.linalg.svd path."""
    # SETUP (parity contract): seed=42, shape=(100, 10)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 10))

    # Compute dimensions
    results = compute_dim(data)

    # Full authoritative key inventory (matches test_input_validation)
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
        assert key in results, f"Missing key: {key}"
        assert isinstance(results[key], (float, np.floating, int, np.integer)), f"Result for {key} is not a number"

def test_compute_dim_list_input():
    """Test compute_dim with list of arrays input."""
    rng = np.random.default_rng(42)
    data = [rng.standard_normal((10, 5)) for _ in range(5)]

    results = compute_dim(data)
    assert "participation_ratio" in results

def test_compute_dim_centered():
    """Test that compute_dim handles uncentered data (it should center it internally)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((50, 5)) + 100  # Shift mean

    results = compute_dim(data)
    # Just checking it runs without error and produces results
    assert results["participation_ratio"] > 0
