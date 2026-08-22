"""Unit tests for nested-dimension curvature diagnostics."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric
from geometry.physics_activation_atlas.nested_dimension_curvature import (
    block_energies,
    verify_H_partition,
)


def _pack_BS(B: np.ndarray) -> np.ndarray:
    D, d, _ = B.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(B[:, a, a] if a == b else (2.0 * B[:, a, b]))
    return np.stack(cols, axis=1)


def test_H_partition_identity():
    rng = np.random.default_rng(0)
    D, d = 32, 16
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    part = verify_H_partition(B, 12, 16)
    assert part["rel_err"] < 1e-10


def test_block_energies_sum_bound():
    rng = np.random.default_rng(1)
    D, d = 24, 16
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    en = block_energies(B, 12, 16)
    # CC + EE + 2*CE covers all entries
    approx = en["E_CC"] + en["E_EE"] + en["E_CE"]
    assert abs(approx - en["E_full"]) / max(en["E_full"], 1e-12) < 1e-8


def test_unpack_roundtrip_diag_mean():
    rng = np.random.default_rng(2)
    D, d = 16, 8
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    flat = _pack_BS(B)
    B2 = unpack_BS_symmetric(flat, d)
    assert np.allclose(B, B2, atol=1e-6)
