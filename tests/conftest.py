"""Shared pytest fixtures for EffDim parity tests."""

from pathlib import Path

import numpy as np
import pytest

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_SWISS_ROLL_F64BIN = _FIXTURES_DIR / "swiss_roll_n1000_noise001_rs42.f64bin"


@pytest.fixture
def swiss_roll_n1000() -> np.ndarray:
    """Load Inventory B swiss-roll manifold: (1000, 3) little-endian float64.

    Bytes match crates/effdim-core/src/fixtures/swiss_roll_n1000_noise001_rs42.f64bin
    (sklearn make_swiss_roll n_samples=1000, noise=0.01, random_state=42).
    """
    data = np.fromfile(_SWISS_ROLL_F64BIN, dtype="<f8")
    assert data.size == 1000 * 3, f"expected 3000 f64 values, got {data.size}"
    return data.reshape(1000, 3)
