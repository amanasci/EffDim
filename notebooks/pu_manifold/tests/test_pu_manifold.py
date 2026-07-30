"""
Fast synthetic-array tests for the ``pu_manifold`` cache and subsample contracts.

No HuggingFace access, no torch, no fixtures beyond ``tmp_path``/``monkeypatch``. Not
collected by the core `effdim` test suite (``pyproject.toml``'s ``testpaths = ["tests"]``
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_pu_manifold.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from pu_manifold import cache as cache_mod
from pu_manifold import subsample as subsample_mod


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    """Point CACHE_DIR at a tmp_path for every test so nothing touches notebooks/.cache/."""
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    yield tmp_path


# --- config_key --------------------------------------------------------------------------


def test_config_key_length_and_insertion_order_stability():
    key = cache_mod.config_key({"a": 1, "b": 2})
    assert len(key) == cache_mod.KEY_LEN == 16
    assert all(c in "0123456789abcdef" for c in key)
    assert key == cache_mod.config_key({"b": 2, "a": 1})


def test_config_key_changes_with_any_single_field():
    base = {"a": 1, "b": 2}
    changed = {"a": 1, "b": 3}
    assert cache_mod.config_key(base) != cache_mod.config_key(changed)


# --- npz_cache -----------------------------------------------------------------------------


def test_npz_cache_round_trip_is_bit_identical_and_computes_once():
    cfg = {"seed": 1, "n": 10}
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        rng = np.random.default_rng(cfg["seed"])
        return {"x": rng.standard_normal((10, 3)), "y": rng.standard_normal(5)}

    first = cache_mod.npz_cache("stem_npz", cfg, compute)
    second = cache_mod.npz_cache("stem_npz", cfg, compute)

    assert calls["count"] == 1
    assert set(first) == set(second) == {"x", "y"}
    for key in first:
        assert np.array_equal(first[key], second[key])


# --- joblib_cache --------------------------------------------------------------------------


def test_joblib_cache_round_trip_is_bit_identical_and_computes_once():
    cfg = {"seed": 2}
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"embedding_": np.arange(12).reshape(4, 3).astype(np.float64)}

    first = cache_mod.joblib_cache("stem_joblib", cfg, compute)
    second = cache_mod.joblib_cache("stem_joblib", cfg, compute)

    assert calls["count"] == 1
    assert np.array_equal(first["embedding_"], second["embedding_"])


# --- manifest mismatch ---------------------------------------------------------------------


def test_manifest_mismatch_raises_instead_of_silently_reusing(tmp_path):
    cfg = {"seed": 3}
    cache_mod.npz_cache("stem_mismatch", cfg, lambda: {"x": np.array([1.0, 2.0])})

    manifest_path = tmp_path / "stem_mismatch.meta.json"
    manifest_path.write_text('{"seed": 999}')

    with pytest.raises(ValueError):
        cache_mod.npz_cache("stem_mismatch", cfg, lambda: {"x": np.array([1.0, 2.0])})


# --- containment guard ---------------------------------------------------------------------


def test_cache_path_rejects_traversal_stem():
    with pytest.raises(ValueError):
        cache_mod.cache_path("../escape", "npz")


# --- draw_row_indices ----------------------------------------------------------------------


def test_draw_row_indices_sorted_unique_and_reproducible():
    a = subsample_mod.draw_row_indices(100, 10, seed=42)
    b = subsample_mod.draw_row_indices(100, 10, seed=42)
    assert np.array_equal(a, b)
    assert len(a) == 10
    # Strictly increasing implies both sorted and duplicate-free.
    assert np.all(np.diff(a) > 0)


def test_draw_row_indices_rejects_n_rows_below_two():
    with pytest.raises(ValueError):
        subsample_mod.draw_row_indices(100, 1, seed=0)


def test_draw_row_indices_rejects_n_rows_above_max():
    with pytest.raises(ValueError):
        subsample_mod.draw_row_indices(100_000, subsample_mod.MAX_N_ROWS + 1, seed=0)


def test_draw_row_indices_rejects_n_total_below_n_rows():
    with pytest.raises(ValueError):
        subsample_mod.draw_row_indices(5, 10, seed=0)


# --- l2_normalize --------------------------------------------------------------------------


def test_l2_normalize_returns_unit_rows_and_correct_raw_norms():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 8))
    unit, norms = subsample_mod.l2_normalize(x)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
    assert np.allclose(norms, np.linalg.norm(x, axis=1))


def test_l2_normalize_rejects_zero_norm_row():
    x = np.zeros((3, 4))
    with pytest.raises(ValueError):
        subsample_mod.l2_normalize(x)


# --- assert_alignment ----------------------------------------------------------------------


def test_assert_alignment_passes_on_synthetic_aligned_pair():
    rng = np.random.default_rng(7)
    n = 200
    hsc_raw = rng.standard_normal((n, subsample_mod.N_FEATURES))
    ls_raw = hsc_raw + 0.01 * rng.standard_normal((n, subsample_mod.N_FEATURES))
    hsc, _ = subsample_mod.l2_normalize(hsc_raw)
    legacysurvey, _ = subsample_mod.l2_normalize(ls_raw)
    row_indices = np.arange(n)

    stats = subsample_mod.assert_alignment(hsc, legacysurvey, row_indices, seed=7)
    assert stats["z"] > subsample_mod.ALIGNMENT_MARGIN_Z
    assert "row_indices_sha256" in stats


def test_assert_alignment_raises_on_off_by_one_negative_control():
    rng = np.random.default_rng(7)
    n = 200
    hsc_raw = rng.standard_normal((n, subsample_mod.N_FEATURES))
    ls_raw = hsc_raw + 0.01 * rng.standard_normal((n, subsample_mod.N_FEATURES))
    hsc, _ = subsample_mod.l2_normalize(hsc_raw)
    legacysurvey, _ = subsample_mod.l2_normalize(ls_raw)
    legacysurvey_rolled = np.roll(legacysurvey, 1, axis=0)
    row_indices = np.arange(n)

    with pytest.raises(ValueError):
        subsample_mod.assert_alignment(hsc, legacysurvey_rolled, row_indices, seed=7)
