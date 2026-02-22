"""
Tests for the public API of the effdim package.
Covers package-level imports, version, and compute_dim reproducibility.
"""
import numpy as np
import pytest
from effdim.api import compute_dim


class TestPackagePublicAPI:
    """Test that the public API is correctly exposed."""

    def test_import_compute_dim_from_package(self):
        """compute_dim should be importable directly from effdim."""
        from effdim import compute_dim
        assert callable(compute_dim)

    def test_version_attribute_exists(self):
        """effdim package should expose __version__."""
        import effdim
        assert hasattr(effdim, "__version__")
        assert isinstance(effdim.__version__, str)
        assert len(effdim.__version__) > 0

    def test_version_is_semver_like(self):
        """__version__ should look like a semver string (X.Y or X.Y.Z)."""
        import effdim
        # Version should start with numeric major.minor components
        parts = effdim.__version__.split(".")
        assert len(parts) >= 2, f"Version '{effdim.__version__}' has fewer than 2 parts"
        # At minimum the major and minor parts should be numeric
        assert parts[0].isdigit(), f"Major version part '{parts[0]}' is not numeric"
        # Minor might include pre-release suffix (e.g. "1b0"), so strip non-digits
        assert any(c.isdigit() for c in parts[1]), (
            f"Minor version part '{parts[1]}' contains no digits"
        )

    def test_all_exports(self):
        """effdim.__all__ should list compute_dim and __version__."""
        import effdim
        assert hasattr(effdim, "__all__")
        assert "compute_dim" in effdim.__all__
        assert "__version__" in effdim.__all__

    def test_compute_dim_returns_dict(self):
        """compute_dim should return a dict."""
        from effdim import compute_dim
        rng = np.random.default_rng(1)
        data = rng.standard_normal((50, 5))
        result = compute_dim(data)
        assert isinstance(result, dict)


class TestReproducibility:
    """Test that compute_dim produces consistent results."""

    def test_same_input_same_output(self):
        """compute_dim with the same input should give identical results."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((80, 6))
        results1 = compute_dim(data)
        results2 = compute_dim(data)
        for key in results1:
            assert results1[key] == results2[key], (
                f"Inconsistent result for '{key}': {results1[key]} != {results2[key]}"
            )

    def test_copied_input_same_output(self):
        """compute_dim should produce same results for a copy of the data."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((60, 5))
        data_copy = data.copy()
        results1 = compute_dim(data)
        results2 = compute_dim(data_copy)
        for key in results1:
            assert np.isclose(results1[key], results2[key], rtol=1e-10), (
                f"Results differ for '{key}': {results1[key]} vs {results2[key]}"
            )

    def test_list_vs_array_input(self):
        """compute_dim should give the same results for list-of-arrays and equivalent ndarray."""
        rng = np.random.default_rng(7)
        chunk1 = rng.standard_normal((30, 5))
        chunk2 = rng.standard_normal((30, 5))
        data_array = np.vstack([chunk1, chunk2])
        data_list = [chunk1, chunk2]
        results_array = compute_dim(data_array)
        results_list = compute_dim(data_list)
        for key in results_array:
            assert np.isclose(results_array[key], results_list[key], rtol=1e-10), (
                f"Results differ for '{key}': array={results_array[key]}, "
                f"list={results_list[key]}"
            )


class TestRenyiDimensionalitiesInComputeDim:
    """Test that Rényi dimensionalities are computed correctly by compute_dim."""

    def test_renyi_keys_alpha_2_through_5(self):
        """compute_dim should include Rényi keys for alpha 2, 3, 4, 5."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 5))
        results = compute_dim(data)
        for alpha in range(2, 6):
            key = f"renyi_eff_dimensionality_alpha_{alpha}"
            assert key in results, f"Missing key: {key}"
            assert np.isfinite(results[key]), f"Non-finite result for {key}"
            assert results[key] > 0, f"Non-positive result for {key}"

    def test_renyi_ordering_in_compute_dim(self):
        """Rényi dimensionalities should be non-increasing with alpha."""
        rng = np.random.default_rng(0)
        # Use anisotropic data so Rényi values are distinct
        data = rng.standard_normal((100, 5)) * np.array([5, 3, 2, 1, 0.5])
        results = compute_dim(data)
        values = [results[f"renyi_eff_dimensionality_alpha_{a}"] for a in range(2, 6)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-6, (
                f"Rényi not non-increasing: alpha={i+2} gave {values[i]}, "
                f"alpha={i+3} gave {values[i+1]}"
            )

    def test_renyi_alpha_2_matches_participation_ratio(self):
        """Rényi alpha=2 should equal Participation Ratio."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 5))
        results = compute_dim(data)
        assert np.isclose(
            results["renyi_eff_dimensionality_alpha_2"],
            results["participation_ratio"],
            rtol=1e-6,
        ), (
            f"Rényi-2={results['renyi_eff_dimensionality_alpha_2']} "
            f"!= PR={results['participation_ratio']}"
        )
