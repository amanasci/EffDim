"""
Phase 2 maturin/PyO3 skeleton tests (RUST-02 shell).

Covers importability of ``effdim._native`` and float64 2-D NumPy round-trip
via ``roundtrip_array``. Not the flat ``compute_dim`` result dict — that is
Phase 3+.

Also freezes D-06: public ``compute_dim`` path stays pure Python — ``_native``
must not appear in ``__all__`` or in ``api`` / ``__init__`` source.
"""

from pathlib import Path

import numpy as np

# Private extension module name (substring used in source-level D-06 guards).
_NATIVE_MODULE = "_native"

_REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "effdim"


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


def test_api_module_does_not_import_native():
    """``api.py`` and ``__init__.py`` must not reference the private native module."""
    api_src = (_REPO_SRC / "api.py").read_text(encoding="utf-8")
    init_src = (_REPO_SRC / "__init__.py").read_text(encoding="utf-8")

    assert _NATIVE_MODULE not in api_src
    assert _NATIVE_MODULE not in init_src
