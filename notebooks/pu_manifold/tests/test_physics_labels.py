"""Unit tests for ``pu_manifold.physics_labels`` -- the Phase 9 row-alignment / loader module.

This suite performs no network access and opens no parquet -- every fixture is an in-memory
numpy array or a pandas ``DataFrame`` built in the test. Load-bearing tests:
``test_alignment_curve_peaks_at_zero_when_aligned`` and
``test_alignment_curve_peaks_at_true_offset_when_shifted`` (the heart of this suite -- the
known-aligned and known-offset cases, including the D9-08 SEARCH branch's input), and
``test_assert_preregistered_rejects_unset_constant`` (the freeze guard's malformed-constant
sweep).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import physics_labels as pl  # noqa: E402

_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "diagnostics" / "09_row_alignment_proof_run.py"
)


@pytest.fixture(scope="module")
def runner():
    """Loads `09_row_alignment_proof_run.py` as a module by file path -- it is not a package
    member (`notebooks/diagnostics/` is a plain directory, not a `pu_manifold` package member,
    and its module name starts with a digit), mirroring
    `test_crossmodal_curvature_run.py`'s own precedent."""
    spec = importlib.util.spec_from_file_location("row_alignment_proof_run_under_test", _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


# --- Loaders (Task 1): _shard_url, load_label_table, label_missingness_report ------------------
#
# Every test here stubs the parquet read -- monkeypatching the SAME `pyarrow.parquet` module
# object `physics_labels` resolves via its own lazy `import pyarrow.parquet as pq` (module
# caching in sys.modules means patching `pq.read_table`/`pq.read_schema` here is visible to
# physics_labels's own lazy import) -- and monkeypatches the UNSET label-source constants to
# plausible filled values, so no test performs a live network read and the loaders are
# exercisable without a freeze.


def _set_label_source_constants(monkeypatch, n_shards=2, repo="Smith42/galaxies", revision="v2.0", split="test"):
    monkeypatch.setattr(pl, "LABEL_REPO", repo)
    monkeypatch.setattr(pl, "LABEL_REVISION", revision)
    monkeypatch.setattr(pl, "LABEL_SPLIT", split)
    monkeypatch.setattr(pl, "LABEL_N_SHARDS", n_shards)


def test_shard_url_pins_revision(monkeypatch):
    _set_label_source_constants(monkeypatch, n_shards=16, repo="Smith42/galaxies", revision="v2.0", split="test")

    for index in range(16):
        url = pl._shard_url(index)
        assert url == f"hf://datasets/Smith42/galaxies@v2.0/data/test-{index:05d}-of-00016.parquet"
        assert "v2.0" in url

    with pytest.raises(ValueError, match="outside range"):
        pl._shard_url(-1)
    with pytest.raises(ValueError, match="outside range"):
        pl._shard_url(16)


def test_shard_url_raises_when_shard_count_is_unset():
    # LABEL_N_SHARDS is UNSET (None) at module scope by default in this test process (no
    # monkeypatch applied) -- _shard_url must not raise TypeError from range(None); it must
    # raise ValueError.
    with pytest.raises(ValueError, match="outside range"):
        pl._shard_url(0)


def test_load_label_table_raises_on_missing_column(monkeypatch):
    _set_label_source_constants(monkeypatch, n_shards=1)
    stub_table = pa.Table.from_pydict({"dr8_id": ["a", "b"], "mag_r_desi": [1.0, 2.0]})

    monkeypatch.setattr(pq, "read_schema", lambda url: stub_table.schema)
    monkeypatch.setattr(pq, "read_table", lambda url, columns=None: stub_table.select(columns))

    with pytest.raises(KeyError) as excinfo:
        pl.load_label_table(["mag_r_desi", "not_a_real_column"], expected_rows=2)
    message = str(excinfo.value)
    assert "not_a_real_column" in message
    assert "Smith42/galaxies" in message
    assert "v2.0" in message


def test_load_label_table_raises_on_row_count_mismatch(monkeypatch):
    _set_label_source_constants(monkeypatch, n_shards=2)
    tables_by_shard = {
        0: pa.Table.from_pydict({"mag_r_desi": [1.0, 2.0, 3.0]}),
        1: pa.Table.from_pydict({"mag_r_desi": [4.0, 5.0]}),
    }

    def fake_read_schema(url):
        return tables_by_shard[0].schema

    def fake_read_table(url, columns=None):
        for index, table in tables_by_shard.items():
            if f"{index:05d}-of-" in url:
                return table.select(columns) if columns else table
        raise AssertionError(f"unexpected url {url!r}")

    monkeypatch.setattr(pq, "read_schema", fake_read_schema)
    monkeypatch.setattr(pq, "read_table", fake_read_table)

    with pytest.raises(ValueError) as excinfo:
        pl.load_label_table(["mag_r_desi"], expected_rows=10)
    message = str(excinfo.value)
    assert "5" in message
    assert "10" in message


def test_load_label_table_raises_on_empty_read(monkeypatch):
    _set_label_source_constants(monkeypatch, n_shards=1)
    empty_table = pa.Table.from_pydict({"mag_r_desi": pa.array([], type=pa.float64())})

    monkeypatch.setattr(pq, "read_schema", lambda url: empty_table.schema)
    monkeypatch.setattr(pq, "read_table", lambda url, columns=None: empty_table.select(columns))

    with pytest.raises(ValueError, match="zero rows"):
        pl.load_label_table(["mag_r_desi"], expected_rows=5)


def test_load_label_table_concatenates_shards_in_ascending_order(monkeypatch):
    _set_label_source_constants(monkeypatch, n_shards=2)
    tables_by_shard = {
        0: pa.Table.from_pydict({"mag_r_desi": [1.0, 2.0]}),
        1: pa.Table.from_pydict({"mag_r_desi": [3.0, 4.0]}),
    }

    def fake_read_schema(url):
        return tables_by_shard[0].schema

    def fake_read_table(url, columns=None):
        for index, table in tables_by_shard.items():
            if f"{index:05d}-of-" in url:
                return table.select(columns) if columns else table
        raise AssertionError(f"unexpected url {url!r}")

    monkeypatch.setattr(pq, "read_schema", fake_read_schema)
    monkeypatch.setattr(pq, "read_table", fake_read_table)

    frame = pl.load_label_table(["mag_r_desi"], expected_rows=4)
    assert list(frame["mag_r_desi"]) == [1.0, 2.0, 3.0, 4.0]


# --- label_missingness_report -------------------------------------------------------------------


def test_missingness_report_counts_sentinels_separately():
    table = pd.DataFrame({"mag_r_desi": [1.0, -99.0, 2.5, np.nan, -99.0, 3.0, np.inf]})
    column_map = {"mag_r": "mag_r_desi"}

    report = pl.label_missingness_report(table, column_map, sentinels=(-99.0,))
    stats = report["mag_r"]

    assert stats["raw_column"] == "mag_r_desi"
    assert stats["n_total"] == 7
    assert stats["n_finite_raw"] == 5  # 1.0, -99.0, 2.5, -99.0, 3.0 are all finite BEFORE masking
    assert stats["n_sentinel"] == 2
    assert stats["n_finite_masked"] == 3  # only 1.0, 2.5, 3.0 survive sentinel masking
    assert stats["fraction_finite"] == pytest.approx(3 / 7)


def test_missingness_report_all_sentinel_column():
    table = pd.DataFrame({"stellar_mass_raw": [-99.0, -99.0, -99.0]})
    column_map = {"stellar_mass": "stellar_mass_raw"}

    report = pl.label_missingness_report(table, column_map, sentinels=(-99.0,))
    stats = report["stellar_mass"]

    assert stats["n_total"] == 3
    assert stats["n_sentinel"] == 3
    assert stats["n_finite_masked"] == 0  # reports zero rather than raising


def test_missingness_report_raises_on_missing_column():
    table = pd.DataFrame({"mag_r_desi": [1.0, 2.0]})
    column_map = {"mag_r": "not_present"}
    with pytest.raises(KeyError):
        pl.label_missingness_report(table, column_map, sentinels=(-99.0,))


# --- Runner (Task 2): --mode manifest writes dataset metadata only, no statistic ---------------


def test_manifest_mode_writes_no_statistic_key(runner, monkeypatch, tmp_path):
    """Behavioural pin on the metadata-only rule (D9-18): monkeypatch the loaders to return
    small stub arrays, run the manifest code path against a temporary record path inside a
    temporary output root, and assert every written row parses as JSON and carries no key
    named `r2`, `rho` or `p`. Holds even if the code is refactored, unlike a source grep."""
    monkeypatch.setattr(runner.pcp, "resolve_output_root", lambda: tmp_path)

    def fake_load_physics_embeddings(**kwargs):
        return {
            "X": np.zeros((5, 3)), "row_norm": np.ones(5), "n_rows": 5, "n_features": 3,
            "source_url": "stub://embeddings", "normalization": "stub",
        }

    def fake_load_label_table(columns, expected_rows=None):
        return pd.DataFrame({name: [1.0, 2.0, -99.0, np.nan, 3.0] for name in columns})

    monkeypatch.setattr(runner.pl, "load_physics_embeddings", fake_load_physics_embeddings)
    monkeypatch.setattr(runner.pl, "load_label_table", fake_load_label_table)

    record_path = tmp_path / "09_data_manifest_stub.jsonl"
    args = runner.build_arg_parser().parse_args(
        [
            "--mode", "manifest",
            "--candidate-columns", "col_a", "col_b",
            "--record-path", str(record_path),
        ]
    )

    ok = runner.run_manifest(args)
    assert ok is True

    lines = record_path.read_text().strip().splitlines()
    assert len(lines) > 0
    for line in lines:
        row = json.loads(line)
        assert "r2" not in row
        assert "rho" not in row
        assert "p" not in row
        assert "passed" not in row
