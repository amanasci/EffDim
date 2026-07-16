"""
Phase 2 maturin/PyO3 skeleton tests (RUST-02 shell).

Covers importability of ``effdim._native`` and float64 2-D NumPy round-trip
via ``roundtrip_array``. Not the flat ``compute_dim`` result dict — that is
Phase 3+.
"""

import numpy as np


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
