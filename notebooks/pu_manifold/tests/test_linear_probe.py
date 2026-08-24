"""
Known-answer and boundary tests for ``pu_manifold.linear_probe`` (Phase 5's probe fit/score,
seed pooling, bucketing and verdict functions).

No HuggingFace access, no CAE checkpoint, no read of ``notebooks/.cache/``, and no PU
embedding -- every fixture is generated in-test from a fixed ``np.random.default_rng`` seed.
Not collected by the core `effdim` test suite (``pyproject.toml``'s ``testpaths = ["tests"]``
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q

This is a permitted new test file: ``05-VALIDATION.md``'s Wave 0 Requirements name it
explicitly and ``05-RESEARCH.md``'s Validation Architecture lists it as the sole test-file gap
for this phase.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from scipy.stats import spearmanr

from pu_manifold import linear_probe as lp


# --- Test 1: known-answer probe fit, row alignment matters -----------------------------------


def test_fit_probe_shape_and_row_alignment():
    """D5-01: with n=400, X of shape (400, 30), a known A of shape (12, 30) and
    Y = X @ A.T + b, fit_probe returns coef_ of shape (12, 30), intercept_ of length 12, and
    predictions matching Y to within 1e-6. Shuffling X and Y by the same permutation leaves
    the fitted coef_ unchanged to within 1e-8; shuffling only Y breaks the fit (R-squared
    drops below 0.5), proving row alignment is load-bearing and not incidental."""
    rng = np.random.default_rng(1)
    n, d_in, d_out = 400, 30, 12
    X = rng.normal(size=(n, d_in))
    A = rng.normal(size=(d_out, d_in))
    b = rng.normal(size=d_out)
    Y = X @ A.T + b

    alpha_grid = (1e-6, 1e-4, 1e-2, 1.0)
    fit = lp.fit_probe(X, Y, alpha_grid=alpha_grid, alpha_per_target=False, fit_intercept=True)
    assert fit["coef_shape"] == (d_out, d_in)
    assert fit["intercept_shape"] == (d_out,)
    Y_pred = lp.predict_probe(fit, X)
    assert np.max(np.abs(Y_pred - Y)) < 1e-6

    perm = rng.permutation(n)
    fit_shuffled = lp.fit_probe(
        X[perm], Y[perm], alpha_grid=alpha_grid, alpha_per_target=False, fit_intercept=True
    )
    coef_a = np.asarray(fit["estimator"].coef_)
    coef_b = np.asarray(fit_shuffled["estimator"].coef_)
    assert np.max(np.abs(coef_a - coef_b)) < 1e-8

    perm_y_only = rng.permutation(n)
    fit_broken = lp.fit_probe(
        X, Y[perm_y_only], alpha_grid=alpha_grid, alpha_per_target=False, fit_intercept=True
    )
    Y_pred_broken = lp.predict_probe(fit_broken, X)
    r2_broken = lp.aggregate_r2(Y[perm_y_only], Y_pred_broken, multioutput="variance_weighted")
    assert r2_broken < 0.5


# --- Test 2: the RESEARCH A3 citation, verified numerically rather than trusted --------------


def test_r2_matches_per_point_residual_aggregate():
    """RESEARCH A3: on a random Y_true / Y_pred pair with unequal per-column variance,
    aggregate_r2(Y_true, Y_pred, "variance_weighted") equals
    1 - per_point_residuals(...).sum() / ((Y_true - Y_true.mean(axis=0)) ** 2).sum() to
    within 1e-12. The identity does NOT hold for "uniform_average", which is why the
    variance-weighted form is the one that will be frozen."""
    rng = np.random.default_rng(2)
    n, d = 300, 6
    scale = np.array([0.1, 1.0, 5.0, 20.0, 0.5, 8.0])
    Y_true = rng.normal(size=(n, d)) * scale[None, :]
    Y_pred = Y_true + rng.normal(size=(n, d)) * (scale[None, :] * 0.3)

    r2_weighted = lp.aggregate_r2(Y_true, Y_pred, multioutput="variance_weighted")
    residuals = lp.per_point_residuals(Y_true, Y_pred)
    denom = float(np.sum((Y_true - Y_true.mean(axis=0)) ** 2))
    identity_weighted = 1.0 - float(residuals.sum()) / denom
    assert abs(r2_weighted - identity_weighted) < 1e-12

    r2_uniform = lp.aggregate_r2(Y_true, Y_pred, multioutput="uniform_average")
    assert abs(r2_uniform - identity_weighted) > 1e-6


# --- Test 3: seed pooling normalization does real work ----------------------------------------


def _piecewise_constant(values: np.ndarray, n_levels: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, n_levels)
    out = np.empty_like(values, dtype=np.float64)
    for level, g in enumerate(groups, start=1):
        out[g] = float(level)
    return out


def test_pool_seeds_no_single_seed_dominates():
    """D5-04, RESEARCH A1: three synthetic length-500 fields reproducing the measured
    1359 / 51438 / 70794 shape -- two seeds (scaled 40x and 55x) share the SAME
    piecewise-constant pattern with only eight distinct values, while the third seed (scaled
    1x) is independent noise uncorrelated with that pattern -- exactly the shape that makes a
    magnitude-weighted average dangerous: the two large, mutually-agreeing seeds get ~99% of
    the raw weight and the disagreeing seed is almost entirely averaged away. A plain
    np.mean of the raw fields therefore has Spearman above 0.99 against the largest-magnitude
    seed alone (it reproduces that seed's pattern almost exactly), while
    pool_seed_fields(..., "per_seed_median_divide") -- which gives every seed equal footing
    after normalizing away its own scale -- has Spearman strictly below 0.99 against the same
    seed: the independent seed's disagreement now actually counts.

    (Note: with exactly four distinct piecewise levels over 500 points, the tie-corrected
    Spearman ceiling between ANY continuous comparator and the tied field itself is
    0.9682 -- provably below 0.99 regardless of how well-ordered the comparator is. Eight
    levels raises that ceiling above 0.99 while still being a small, "collapsed-metric"
    level count, so this fixture uses eight.)
    """
    rng = np.random.default_rng(3)
    n = 500
    n_levels = 8
    base = rng.uniform(0.0, 1.0, size=n)
    piecewise = _piecewise_constant(base, n_levels)
    independent_seed = rng.uniform(1.0, 5.0, size=n)  # unrelated to `base`/`piecewise`

    fields = {1: independent_seed, 2: piecewise * 40.0, 3: piecewise * 55.0}
    largest_seed = 3

    raw_mean = np.mean([fields[s] for s in sorted(fields)], axis=0)
    rho_raw, _ = spearmanr(raw_mean, fields[largest_seed])
    assert rho_raw > 0.99

    pooled_result = lp.pool_seed_fields(fields, method="per_seed_median_divide")
    rho_normalized, _ = spearmanr(pooled_result["pooled"], fields[largest_seed])
    assert rho_normalized < 0.99


def test_pool_seed_fields_requires_explicit_method():
    """Calling pool_seed_fields with only the fields argument raises TypeError; calling it
    with an unrecognized method string raises ValueError naming that string."""
    with pytest.raises(TypeError):
        lp.pool_seed_fields({1: [1.0, 2.0]})
    with pytest.raises(ValueError):
        lp.pool_seed_fields({1: [1.0, 2.0], 2: [3.0, 4.0]}, method="not_a_real_method")


# --- Test 4: convention agreement, deliberately RED until the 05-04 freeze -------------------


@pytest.mark.xfail(strict=True, reason="constant is unset until the 05-04 pre-registration freeze")
def test_curvature_convention_matches_sealed_modules():
    """D5-06: linear_probe.CURVATURE_CONVENTION equals chart_curvature.CURVATURE_CONVENTION
    equals curvature_probe.CURVATURE_CONVENTION equals the string "trace". This test is
    written now and is RED until the freeze at 05-04 sets the constant -- 05-04 Task 2 must
    REMOVE this expected-failure marker (see the decorator above), and the test must then
    pass."""
    from pu_manifold import chart_curvature, curvature_probe

    assert (
        lp.CURVATURE_CONVENTION
        == chart_curvature.CURVATURE_CONVENTION
        == curvature_probe.CURVATURE_CONVENTION
        == "trace"
    )


# --- Test 5: bucket assignment, tie rule --------------------------------------------------------


def test_bucket_assignment_known_answer():
    """D5-07 / D5-09: with values = np.arange(1, 31, dtype=float) and n_buckets=3,
    bucket_edges_from_field returns exactly two edges, assign_buckets produces exactly ten
    points per bucket, and a value equal to an edge lands in the HIGHER bucket (the
    documented tie rule)."""
    values = np.arange(1, 31, dtype=float)
    edges = lp.bucket_edges_from_field(values, n_buckets=3)
    assert len(edges) == 2
    assert edges == (11.0, 21.0)

    labels = lp.assign_buckets(values, edges)
    counts = np.bincount(labels, minlength=3)
    assert list(counts) == [10, 10, 10]

    idx_low_edge = int(np.flatnonzero(values == 11.0)[0])
    idx_high_edge = int(np.flatnonzero(values == 21.0)[0])
    assert labels[idx_low_edge] == 1
    assert labels[idx_high_edge] == 2


def test_bucket_counts_partition_exactly():
    """For any labels array, bucket_counts(labels, n_buckets)'s per-bucket counts sum
    exactly to labels.shape[0], and calling bucket_by_field twice on identical input returns
    arrays satisfying np.array_equal."""
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 3, size=200)
    counts_info = lp.bucket_counts(labels, n_buckets=3)
    assert int(counts_info["counts"].sum()) == labels.shape[0]

    values = rng.normal(size=300)
    labels1, edges1 = lp.bucket_by_field(values, n_buckets=4)
    labels2, edges2 = lp.bucket_by_field(values, n_buckets=4)
    assert np.array_equal(labels1, labels2)
    assert edges1 == edges2


# --- Test 6: the realized-test-split size-match distinction (D5-08 / Pitfall 4) --------------


def test_size_matched_check_uses_test_split_counts():
    """D5-08, RESEARCH Pitfall 4: build a full field of 3,000 points bucketed into exactly
    1,000 each, then take a deliberately unbalanced 300-point "test split" whose realized
    bucket counts are 60 / 100 / 140. size_matched_check's returned n_match is 60 (the
    smallest REALIZED test-split count), realized_bucket_counts equals (60, 100, 140), and
    n_match is NOT 1000 -- the exact artifact that undercut Phase 4's verdict."""
    rng = np.random.default_rng(5)
    field = np.concatenate(
        [
            np.zeros(1000) + rng.normal(scale=0.01, size=1000),
            np.full(1000, 1.0) + rng.normal(scale=0.01, size=1000),
            np.full(1000, 2.0) + rng.normal(scale=0.01, size=1000),
        ]
    )
    labels_full, _ = lp.bucket_by_field(field, n_buckets=3)
    counts_full = lp.bucket_counts(labels_full, n_buckets=3)
    assert list(counts_full["counts"]) == [1000, 1000, 1000]

    idx0 = np.flatnonzero(labels_full == 0)[:60]
    idx1 = np.flatnonzero(labels_full == 1)[:100]
    idx2 = np.flatnonzero(labels_full == 2)[:140]
    test_idx = np.concatenate([idx0, idx1, idx2])
    labels_test = labels_full[test_idx]
    residuals_test = rng.normal(size=test_idx.shape[0]) ** 2

    result = lp.size_matched_check(
        residuals_test, labels_test, n_repeats=10, seed=5, confidence_level=0.9
    )
    assert result["n_match"] == 60
    assert result["realized_bucket_counts"] == (60, 100, 140)
    assert result["n_match"] != 1000


# --- Test 7: the D5-10 guard, proven live in both directions ---------------------------------


def test_assert_preregistered_raises_when_absent(monkeypatch):
    """D5-10: assert_preregistered() raises RuntimeError on the module as shipped. Then with
    monkeypatch.setattr filling VERDICT_RULE with a string that omits the literal N_BUCKETS,
    it still raises RuntimeError; with every constant monkeypatched to a well-formed value it
    does not raise. Three separate assertions, so the guard is proven live in both
    directions."""
    with pytest.raises(RuntimeError):
        lp.assert_preregistered()

    monkeypatch.setattr(lp, "VERDICT_RULE", "a rule that omits the bucket-count constant name")
    with pytest.raises(RuntimeError):
        lp.assert_preregistered()

    monkeypatch.setattr(lp, "VERDICT_RULE", "well-formed rule naming N_BUCKETS explicitly")
    monkeypatch.setattr(lp, "N_BUCKETS", 3)
    monkeypatch.setattr(lp, "TRAIN_FRACTION", 0.8)
    monkeypatch.setattr(lp, "SPLIT_SEED", 20260824)
    monkeypatch.setattr(lp, "RIDGE_ALPHA_GRID", (0.1, 1.0, 10.0))
    monkeypatch.setattr(lp, "POOLING_METHOD", "per_seed_median_divide")
    monkeypatch.setattr(lp, "BUCKET_EDGES", (1.0, 2.0))
    monkeypatch.setattr(lp, "SEED_STEMS", (20260813, 20260814, 20260815))
    monkeypatch.setattr(lp, "CURVATURE_CONVENTION", "trace")
    monkeypatch.setattr(lp, "CURVATURE_SOURCE_FUNCTION", "chart_curvature.chart_curvature_field")
    lp.assert_preregistered()  # must not raise


# --- Test 8: input guards ----------------------------------------------------------------------


def test_linear_probe_input_guards():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 5))
    Y = rng.normal(size=(20, 3))

    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        lp.fit_probe(X_bad, Y, alpha_grid=(1.0,), alpha_per_target=False, fit_intercept=True)

    with pytest.raises(ValueError):
        lp.fit_probe(X, Y[:-1], alpha_grid=(1.0,), alpha_per_target=False, fit_intercept=True)

    with pytest.raises(ValueError):
        lp.bucket_edges_from_field(rng.normal(size=10), n_buckets=1)

    with pytest.raises(ValueError):
        lp.train_test_split_indices(20, train_fraction=0.0, split_seed=1)
    with pytest.raises(ValueError):
        lp.train_test_split_indices(20, train_fraction=1.0, split_seed=1)

    with pytest.raises(ValueError):
        lp.bucket_edges_from_field(rng.normal(size=2), n_buckets=5)
