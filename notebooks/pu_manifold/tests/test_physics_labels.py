"""Unit tests for ``pu_manifold.physics_labels`` -- the Phase 9 row-alignment / loader module.

This suite performs no network access and opens no parquet -- every fixture is an in-memory
numpy array or a pandas ``DataFrame`` built in the test. Load-bearing tests:
``test_alignment_curve_peaks_at_zero_when_aligned`` and
``test_alignment_curve_peaks_at_true_offset_when_shifted`` (the heart of this suite -- the
known-aligned and known-offset cases, including the D9-08 SEARCH branch's input), and
``test_assert_preregistered_rejects_unset_constant`` (the freeze guard's malformed-constant
sweep).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import physics_labels as pl  # noqa: E402


def _small_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A small deterministic out-of-fold callable, independent of the sibling Phase 9
    statistics module -- these tests must not depend on the ridge implementation there."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    n = X.shape[0]
    y_hat = np.full(n, np.nan)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in kfold.split(X):
        model = Ridge(alpha=1.0).fit(X[train_idx], y[train_idx])
        y_hat[test_idx] = model.predict(X[test_idx])
    return y_hat


# --- alignment curve: the heart of this suite --------------------------------------------------


def test_alignment_curve_peaks_at_zero_when_aligned():
    rng = np.random.default_rng(1)
    n, d_in = 300, 6
    X = rng.normal(size=(n, d_in))
    w = rng.normal(size=d_in)
    y_true = X @ w + 0.05 * rng.normal(size=n)

    shifts = (-3, -2, -1, 0, 1, 2, 3)
    curve = pl.alignment_r2_curve(X, y_true, shifts, n_permutations=5, permutation_seed=1, oof_fn=_small_oof)
    shift_rows = {row["shift"]: row["r2"] for row in curve if row["alignment"] == "shift"}
    assert max(shift_rows, key=lambda s: shift_rows[s]) == 0

    verdict = pl.alignment_verdict(curve, margin=0.3)
    assert verdict["passed"] is True
    assert verdict["r2_shift0"] == shift_rows[0]


def test_alignment_curve_peaks_at_true_offset_when_shifted():
    rng = np.random.default_rng(2)
    n, d_in = 300, 6
    X = rng.normal(size=(n, d_in))
    w = rng.normal(size=d_in)
    y_true = X @ w + 0.05 * rng.normal(size=n)
    # y_input[j] = y_true[(j - 7) % n] -- so shifted_pairing(n, 7) recovers the true pairing.
    y_input = np.roll(y_true, 7)

    shifts = (-3, -2, -1, 0, 1, 2, 3, 7)
    curve = pl.alignment_r2_curve(
        X, y_input, shifts, n_permutations=5, permutation_seed=2, oof_fn=_small_oof
    )
    shift_rows = {row["shift"]: row["r2"] for row in curve if row["alignment"] == "shift"}
    assert max(shift_rows, key=lambda s: shift_rows[s]) == 7

    verdict = pl.alignment_verdict(curve, margin=0.3)
    assert verdict["passed"] is False  # shift 0 does not win
    assert verdict["clearing_alignments"] == [7]


def test_alignment_verdict_is_strict_at_the_margin():
    curve = [
        {"alignment": "shift", "shift": 0, "r2": 0.50, "n_finite": 100},
        {"alignment": "shift", "shift": 1, "r2": 0.30, "n_finite": 100},
    ]
    verdict = pl.alignment_verdict(curve, margin=0.20)  # gap is exactly 0.20
    assert verdict["gap"] == pytest.approx(0.20)
    assert verdict["passed"] is False


def test_alignment_curve_raises_on_empty_shift_set():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=50)
    with pytest.raises(ValueError, match="non-empty"):
        pl.alignment_r2_curve(X, y, (), n_permutations=1, permutation_seed=1, oof_fn=_small_oof)


# --- shifted_pairing -----------------------------------------------------------------------------


def test_shifted_pairing_wraps_and_is_a_permutation():
    for n in (5, 50):
        for shift in (0, 1, -1, n, -n, 3 * n + 2, -(3 * n + 2)):
            result = pl.shifted_pairing(n, shift)
            assert np.array_equal(np.sort(result), np.arange(n))


def test_shifted_pairing_raises_on_empty():
    with pytest.raises(ValueError):
        pl.shifted_pairing(0, 1)


# --- mask_sentinels --------------------------------------------------------------------------


def test_mask_sentinels_converts_only_sentinels_and_nonfinite():
    y = np.array([1.0, -99.0, 2.5, np.inf, -np.inf, np.nan, -99.0, 3.0])
    out = pl.mask_sentinels(y, (-99.0,))
    expected_nan_positions = np.array([False, True, False, True, True, True, True, False])
    assert np.array_equal(np.isnan(out), expected_nan_positions)
    kept = ~expected_nan_positions
    assert np.array_equal(out[kept], y[kept])

    with pytest.raises(ValueError):
        pl.mask_sentinels(y.reshape(2, 4), (-99.0,))


# --- canonical_label -------------------------------------------------------------------------


def test_canonical_label_raises_naming_column_and_revision(monkeypatch):
    monkeypatch.setattr(pl, "LABEL_REVISION", "v2.0")
    table = pd.DataFrame({"mag_r_desi": [1.0, 2.0, 3.0]})
    column_map = {"mag_r": "mag_r_desi_typo"}  # deliberately absent from the table

    with pytest.raises(KeyError) as excinfo:
        pl.canonical_label(table, "mag_r", column_map, sentinels=(-99.0,))
    message = str(excinfo.value)
    assert "mag_r" in message
    assert "mag_r_desi_typo" in message
    assert "v2.0" in message

    # Happy path: resolves and masks sentinels.
    good_map = {"mag_r": "mag_r_desi"}
    values = pl.canonical_label(table, "mag_r", good_map, sentinels=(-99.0,))
    assert np.array_equal(values, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(KeyError):
        pl.canonical_label(table, "not_a_canonical_name", good_map, sentinels=())


# --- assert_expected_rows --------------------------------------------------------------------


def test_assert_expected_rows_is_exact_integer_comparison():
    pl.assert_expected_rows(100, 100, "physics embeddings")  # must not raise
    with pytest.raises(ValueError) as excinfo:
        pl.assert_expected_rows(99, 100, "physics embeddings")
    message = str(excinfo.value)
    assert "99" in message
    assert "100" in message
    assert "physics embeddings" in message


# --- freeze guard: malformed-constant sweep ---------------------------------------------------


def _plausible_value(name: str):
    equality_map = {
        "ALIGNMENT_PASS_RULE": pl._REQUIRED_ALIGNMENT_PASS_RULE,
        "ALIGNMENT_SEARCH_RULE": pl._REQUIRED_ALIGNMENT_SEARCH_RULE,
    }
    if name in equality_map:
        return equality_map[name]
    original = getattr(pl, name)
    if isinstance(original, dict):
        return {"placeholder_key": 1}
    if isinstance(original, tuple):
        return (1,)
    return 1


def _unset_value(name: str):
    original = getattr(pl, name)
    if isinstance(original, dict):
        return {}
    if isinstance(original, tuple):
        return ()
    if isinstance(original, str):
        return ""
    return None


@pytest.mark.parametrize("name", pl._REQUIRED_CONSTANTS)
def test_assert_preregistered_rejects_unset_constant(name):
    with pytest.MonkeyPatch.context() as mp:
        for other in pl._REQUIRED_CONSTANTS:
            if other == name:
                continue
            mp.setattr(pl, other, _plausible_value(other))
        mp.setattr(pl, name, _unset_value(name))
        with pytest.raises(RuntimeError, match=re.escape(name)):
            pl.assert_preregistered()
