"""
Phase 4 maturin/PyO3 native module tests (full-dict compute_dim smoke).

Covers importability of ``effdim._native``, float64 2-D NumPy round-trip via
``roundtrip_array``, and Phase 4 ``compute_dim`` full 16-key dict smoke (D-03, D-04).

Public ``__all__`` and ``__init__.py`` still exclude ``_native``. The full-dict
smoke fails RED (AttributeError) until plan 04-03 registers ``_native.compute_dim``.
"""

from pathlib import Path

import numpy as np

# Private extension module name (substring used in source-level D-16 guards).
_NATIVE_MODULE = "_native"

_REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "effdim"

# Full 16-key inventory matching tests/test_api.py / Phase 1 validation (D-03).
_FULL_COMPUTE_DIM_KEYS = {
    "pca_explained_variance_95",
    "participation_ratio",
    "shannon_entropy",
    "renyi_eff_dimensionality_alpha_2",
    "renyi_eff_dimensionality_alpha_3",
    "renyi_eff_dimensionality_alpha_4",
    "renyi_eff_dimensionality_alpha_5",
    "geometric_mean_eff_dimensionality",
    "mle_dimensionality",
    "two_nn_dimensionality",
    "danco_dimensionality",
    "mind_mli_dimensionality",
    "mind_mlk_dimensionality",
    "ess_dimensionality",
    "tle_dimensionality",
    "gmst_dimensionality",
}


def test_native_importable():
    """``effdim._native`` must import after ``maturin develop``."""
    import effdim._native as native

    assert native is not None


def test_roundtrip_array_same_shape_dtype():
    """float64 2-D input round-trips with equal shape, dtype, and values."""
    import effdim._native as native

    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = native.roundtrip_array(data)

    assert out.shape == data.shape
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, data)


def test_native_not_in_public_all():
    """Public ``__all__`` stays ``__version__`` + ``compute_dim`` (no ``_native``)."""
    import effdim

    assert _NATIVE_MODULE not in effdim.__all__
    assert set(effdim.__all__) == {"__version__", "compute_dim"}


def test_init_module_does_not_import_native():
    """``__init__.py`` must not reference the private native module (D-16)."""
    init_src = (_REPO_SRC / "__init__.py").read_text(encoding="utf-8")

    assert _NATIVE_MODULE not in init_src


def test_compute_dim_full_dict_keys():
    """``compute_dim`` returns the full 16-key flat dict (D-03, D-04).

    RED until plan 04-03 lands ``effdim._native.compute_dim``.
    """
    import effdim._native as native

    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 4)).astype(np.float64)
    result = native.compute_dim(data)

    assert hasattr(result, "keys")
    assert set(result.keys()) == _FULL_COMPUTE_DIM_KEYS
    assert isinstance(result["pca_explained_variance_95"], int)
