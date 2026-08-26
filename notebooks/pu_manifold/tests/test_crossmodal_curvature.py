"""Phase 7 pre-registration guards for ``crossmodal_curvature.py``.

The load-bearing tests here are the parameterized malformed-constant sweep over
``_REQUIRED_CONSTANTS`` (a constant added later without a guard entry must fail this suite)
and ``test_freeze_commit_is_a_strict_ancestor_of_head`` (the freeze proof shape D7-06 actually
requires: STRICT ancestry, not merely ``--is-ancestor``, which a commit satisfies for itself).

Loads no PU data, trains nothing, reads no cache. Completes in well under 10 seconds.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import crossmodal_curvature as cc  # noqa: E402
from pu_manifold import curvature_probe  # noqa: E402
from pu_manifold import mknn  # noqa: E402


# The freeze commit SHA recorded in this plan's SUMMARY -- the commit that added
# crossmodal_curvature.py (Task 2). Every later PU number must be a descendant of this commit.
FREEZE_COMMIT_SHA = "f032745f6450068c63763993d39fa112fd36bb8c"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _freeze_commit_exists() -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{FREEZE_COMMIT_SHA}^{{commit}}"],
        cwd=_repo_root(),
        capture_output=True,
    )
    return result.returncode == 0


def _freeze_commit_is_strict_ancestor_of_head() -> bool:
    """True only once at least one commit exists after the freeze commit. Immediately after
    the freeze commit itself (HEAD == freeze commit, e.g. right before this test file's own
    commit lands), this is False and the test below is skipped rather than failed -- the
    freeze commit being HEAD is the expected state at that moment, not a defect."""
    if not _freeze_commit_exists():
        return False
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT_SHA, "HEAD"],
        cwd=_repo_root(),
    )
    if is_ancestor.returncode != 0:
        return False
    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{FREEZE_COMMIT_SHA}..HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return int(count_result.stdout.strip()) >= 1


# --- basic pass/shape ------------------------------------------------------------------


def test_assert_preregistered_passes_when_frozen():
    cc.assert_preregistered()


def test_verdict_is_terminal_accepts_every_verdict_value_and_rejects_a_near_miss():
    for value in cc.VERDICT_VALUES:
        assert cc.verdict_is_terminal(value)
    assert not cc.verdict_is_terminal("ASSOCIATION DETECTED ")
    assert not cc.verdict_is_terminal("split across d")
    assert not cc.verdict_is_terminal("HOLDS")


def test_required_constants_covers_every_frozen_constant():
    """Neither a missing guard entry nor a stale one can pass: every module-level UPPER_CASE
    name (excluding the private ``_REQUIRED_CONSTANTS`` itself) must appear in
    ``_REQUIRED_CONSTANTS``, and vice versa."""
    declared = {n for n in vars(cc) if n.isupper() and not n.startswith("_")}
    guarded = set(cc._REQUIRED_CONSTANTS)
    assert guarded == declared, (
        f"guarded-but-not-declared: {guarded - declared}; "
        f"declared-but-not-guarded: {declared - guarded}"
    )


# --- malformed-constant boundary sweep, parameterized over every required constant -------


@pytest.mark.parametrize("name", cc._REQUIRED_CONSTANTS)
def test_none_constant_raises_and_names_it(name, monkeypatch):
    monkeypatch.setattr(cc, name, None)
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert name in str(excinfo.value)


@pytest.mark.parametrize("name", cc._REQUIRED_CONSTANTS)
def test_absent_constant_raises_and_names_it(name, monkeypatch):
    monkeypatch.delattr(cc, name)
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert name in str(excinfo.value)


@pytest.mark.parametrize(
    "name",
    [n for n in cc._REQUIRED_CONSTANTS if isinstance(getattr(cc, n), str)],
)
def test_blank_string_constant_raises_and_names_it(name, monkeypatch):
    monkeypatch.setattr(cc, name, "   ")
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert name in str(excinfo.value)


@pytest.mark.parametrize(
    "name",
    [n for n in cc._REQUIRED_CONSTANTS if isinstance(getattr(cc, n), tuple)],
)
def test_empty_tuple_constant_raises_and_names_it(name, monkeypatch):
    monkeypatch.setattr(cc, name, ())
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert name in str(excinfo.value)


# --- Phase 7-specific boundary checks (D7-01 D_SWEEP, D7-02 target-rho ordering) ----------


def test_positive_control_target_rhos_must_be_strictly_increasing(monkeypatch):
    monkeypatch.setattr(cc, "POSITIVE_CONTROL_TARGET_RHOS", (0.10, 0.05, 0.20))
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert "POSITIVE_CONTROL_TARGET_RHOS" in str(excinfo.value)


def test_positive_control_target_rhos_rejects_non_increasing_ties(monkeypatch):
    monkeypatch.setattr(cc, "POSITIVE_CONTROL_TARGET_RHOS", (0.05, 0.05, 0.20))
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert "POSITIVE_CONTROL_TARGET_RHOS" in str(excinfo.value)


def test_d_sweep_rejects_a_non_positive_entry(monkeypatch):
    monkeypatch.setattr(cc, "D_SWEEP", (20, 0, 32))
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert "D_SWEEP" in str(excinfo.value)


def test_d_sweep_rejects_a_non_int_entry(monkeypatch):
    monkeypatch.setattr(cc, "D_SWEEP", (20, 25.0, 32))
    with pytest.raises(RuntimeError) as excinfo:
        cc.assert_preregistered()
    assert "D_SWEEP" in str(excinfo.value)


# --- caveat coverage in VERDICT_RULE's own text --------------------------------------------


def test_verdict_rule_carries_its_own_caveats():
    for token in (
        "INSTRUMENT_FIDELITY_RANGE",
        "Phase 4's HOLDS",
        "n = 10,000",
        "single_seed_across_d_sweep",
        "UNDERPOWERED",
    ):
        assert token in cc.VERDICT_RULE, f"VERDICT_RULE is missing {token!r}"


def test_split_across_d_is_a_reachable_terminal_outcome():
    assert "SPLIT ACROSS d" in cc.VERDICT_VALUES
    assert cc.verdict_is_terminal("SPLIT ACROSS d")


# --- the freeze-ancestry proof itself -------------------------------------------------------


@pytest.mark.skipif(
    not _freeze_commit_is_strict_ancestor_of_head(),
    reason=(
        "freeze commit is not (yet) a STRICT ancestor of HEAD -- either it is absent from "
        "this checkout's history (e.g. a shallow clone), or HEAD IS the freeze commit itself "
        "(the expected state immediately after the freeze, before this test file's own commit "
        "lands). Plan 07-04's own acceptance criteria re-run the same ancestry check "
        "unconditionally at the moment a PU number is produced, which is where it actually bites."
    ),
)
def test_freeze_commit_is_a_strict_ancestor_of_head():
    """D7-06's precision requirement: a commit is its own ancestor, so ``--is-ancestor`` alone
    would pass even if a PU number were produced in the freeze commit itself.
    ``git rev-list --count <freeze>..HEAD`` must also be at least 1."""
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT_SHA, "HEAD"],
        cwd=_repo_root(),
    )
    assert is_ancestor.returncode == 0, "freeze commit is not an ancestor of HEAD at all"

    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{FREEZE_COMMIT_SHA}..HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    strict_distance = int(count_result.stdout.strip())
    assert strict_distance >= 1, (
        "freeze commit is not a STRICT ancestor of HEAD -- HEAD IS the freeze commit "
        "(strict_distance == 0), which would mean no number-producing commit exists yet"
    )


# =============================================================================================
# Plan 07-02, Task 2 -- pin per_point_mknn, two_tailed_permutation_null, apply_verdict and
# split_indices (added by Task 1) against the sealed function each re-composes. Loads no PU
# data, trains nothing, reads no cache -- every fixture below is synthetic and small.
# =============================================================================================

_MKNN_K = 10


def _distinct_at_relative_precision(values: np.ndarray) -> int:
    """Counts distinct values after rounding at RELATIVE precision (divide by the array's
    own maximum absolute value, round to 12 decimals) rather than on raw float equality.
    05-02-SUMMARY.md reported 5,301 and 9,852 distinct values where the true counts at
    relative precision were 4 and 3 -- the retraction is on record in STATE.md. Counting raw
    float equality is that same error."""
    values = np.asarray(values, dtype=np.float64)
    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        return int(len(np.unique(values)))
    normalized = np.round(values / max_abs, 12)
    return int(len(np.unique(normalized)))


# --- per_point_mknn vs. mknn.mknn_score (D7-04 gap-fill regression) ------------------------


def test_per_point_mknn_mean_agrees_with_mknn_score():
    rng = np.random.default_rng(20260826)
    z1 = rng.normal(size=(400, 16))
    z2 = rng.normal(size=(400, 16))
    per_point = cc.per_point_mknn(z1, z2, _MKNN_K)
    assert per_point.mean() == pytest.approx(mknn.mknn_score(z1, z2, _MKNN_K))


def test_per_point_mknn_is_all_ones_against_itself():
    rng = np.random.default_rng(20260826)
    z = rng.normal(size=(400, 16))
    per_point = cc.per_point_mknn(z, z, _MKNN_K)
    assert np.allclose(per_point, 1.0)


def test_per_point_mknn_independent_clouds_lands_near_chance_floor():
    rng = np.random.default_rng(20260826)
    z1 = rng.normal(size=(400, 16))
    z2 = rng.normal(size=(400, 16))
    per_point = cc.per_point_mknn(z1, z2, _MKNN_K)
    floor = mknn.chance_floor(z1.shape[0], _MKNN_K)
    # "within a factor of three" per the plan's acceptance behavior -- a loose bound, since
    # the chance floor is itself an approximation, not an exact expectation.
    assert floor / 3.0 <= per_point.mean() <= floor * 3.0


def test_per_point_mknn_row_alignment_is_preserved_under_a_shared_permutation():
    rng = np.random.default_rng(20260826)
    n = 300
    z1 = rng.normal(size=(n, 12))
    z2 = rng.normal(size=(n, 12))
    baseline = cc.per_point_mknn(z1, z2, _MKNN_K)

    perm = rng.permutation(n)
    permuted = cc.per_point_mknn(z1[perm], z2[perm], _MKNN_K)

    np.testing.assert_array_equal(permuted, baseline[perm])


# --- per_point_mknn degenerate-input guards -------------------------------------------------


def test_per_point_mknn_raises_on_mismatched_row_counts():
    rng = np.random.default_rng(20260826)
    z1 = rng.normal(size=(50, 8))
    z2 = rng.normal(size=(40, 8))
    with pytest.raises(ValueError) as excinfo:
        cc.per_point_mknn(z1, z2, _MKNN_K)
    assert "rows" in str(excinfo.value)


def test_per_point_mknn_raises_on_non_finite_entry():
    rng = np.random.default_rng(20260826)
    z1 = rng.normal(size=(50, 8))
    z2 = rng.normal(size=(50, 8))
    z1[0, 0] = np.nan
    with pytest.raises(ValueError) as excinfo:
        cc.per_point_mknn(z1, z2, _MKNN_K)
    assert "non-finite" in str(excinfo.value)


def test_per_point_mknn_raises_on_n_less_than_two():
    z1 = np.zeros((1, 8))
    z2 = np.zeros((1, 8))
    with pytest.raises(ValueError) as excinfo:
        cc.per_point_mknn(z1, z2, _MKNN_K)
    assert "n=1" in str(excinfo.value) or "at least 2" in str(excinfo.value)


def test_per_point_mknn_raises_when_k_plus_one_exceeds_n():
    rng = np.random.default_rng(20260826)
    n = 5
    z1 = rng.normal(size=(n, 8))
    z2 = rng.normal(size=(n, 8))
    with pytest.raises(ValueError) as excinfo:
        cc.per_point_mknn(z1, z2, n)  # k = n, so k + 1 > n
    assert "exceeds" in str(excinfo.value)


def test_per_point_mknn_distinct_value_count_is_bounded_by_k_plus_one():
    rng = np.random.default_rng(20260826)
    n = 500
    z1 = rng.normal(size=(n, 20))
    z2 = z1 + rng.normal(scale=0.5, size=(n, 20))
    per_point = cc.per_point_mknn(z1, z2, _MKNN_K)
    assert _distinct_at_relative_precision(per_point) <= _MKNN_K + 1


# --- two_tailed_permutation_null -------------------------------------------------------------


def _discretized_pair(n: int, k: int, seed: int, direction: str):
    """A synthetic (h, m) pair discretized like a real per-point MKNN array against a
    uniform curvature-magnitude surrogate, matching the plan's construction: h is a uniform
    draw, m is k minus (or plus) a monotone function of h's rank divided by n, so the pair
    is strongly correlated in the requested direction and tie-dense like the real
    statistic."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(size=n)
    rank_frac = (rankdata(h) - 0.5) / n  # in (0, 1), monotone in h
    if direction == "negative":
        m = np.floor(k * (1.0 - rank_frac))
    elif direction == "positive":
        m = np.floor(k * rank_frac)
    else:
        raise ValueError(f"_discretized_pair: unknown direction {direction!r}")
    return h, m


_N_RESAMPLES_TEST = 199  # frozen N_PERMUTATIONS is for the real run only; kept small here so
# the file stays under ten seconds, passed explicitly rather than relying on any default.
_TEST_SEED = 20260826


def test_two_tailed_permutation_null_detects_negative_association():
    h, m = _discretized_pair(300, _MKNN_K, _TEST_SEED, "negative")
    result = cc.two_tailed_permutation_null(
        h, m, _N_RESAMPLES_TEST, _TEST_SEED, cc.NULL_QUANTILE_PER_TAIL
    )
    assert result["direction"] == "negative"
    assert result["clears_either"] is True


def test_two_tailed_permutation_null_detects_positive_association():
    h, m = _discretized_pair(300, _MKNN_K, _TEST_SEED, "positive")
    result = cc.two_tailed_permutation_null(
        h, m, _N_RESAMPLES_TEST, _TEST_SEED, cc.NULL_QUANTILE_PER_TAIL
    )
    assert result["direction"] == "positive"
    assert result["clears_either"] is True


def test_two_tailed_permutation_null_does_not_clear_on_independent_pair():
    rng = np.random.default_rng(_TEST_SEED)
    h = rng.uniform(size=300)
    m = rng.integers(0, _MKNN_K + 1, size=300).astype(np.float64)
    result = cc.two_tailed_permutation_null(
        h, m, _N_RESAMPLES_TEST, _TEST_SEED, cc.NULL_QUANTILE_PER_TAIL
    )
    assert result["clears_either"] is False


def test_two_tailed_permutation_null_observed_rho_matches_spearman_and_negation():
    h, m = _discretized_pair(300, _MKNN_K, _TEST_SEED, "negative")
    result = cc.two_tailed_permutation_null(
        h, m, _N_RESAMPLES_TEST, _TEST_SEED, cc.NULL_QUANTILE_PER_TAIL
    )
    assert result["observed_rho"] == pytest.approx(spearmanr(h, m).statistic)
    assert result["negative_tail"]["observed_rho"] == pytest.approx(-result["observed_rho"])


def test_single_one_sided_permutation_null_call_misses_the_negative_association():
    """The test that would have caught the one-sided defect: a single, un-mirrored
    ``curvature_probe.permutation_null(h, m, ...)`` call is one-sided (alternative='greater')
    and cannot detect a strongly NEGATIVE association -- exactly the defect
    ``two_tailed_permutation_null`` exists to close."""
    h, m = _discretized_pair(300, _MKNN_K, _TEST_SEED, "negative")
    one_sided = curvature_probe.permutation_null(
        h, m, _N_RESAMPLES_TEST, _TEST_SEED, cc.NULL_QUANTILE_PER_TAIL
    )
    assert one_sided["clears_null"] is False


# --- apply_verdict -----------------------------------------------------------------------


def test_apply_verdict_association_detected_when_every_d_clears():
    per_d = {d: True for d in cc.D_SWEEP}
    assert cc.apply_verdict(per_d, positive_control_cleared_at=0.05) == "ASSOCIATION DETECTED"


def test_apply_verdict_no_detectable_relationship_when_no_d_clears_but_control_cleared():
    per_d = {d: False for d in cc.D_SWEEP}
    assert (
        cc.apply_verdict(per_d, positive_control_cleared_at=0.05)
        == "NO DETECTABLE RELATIONSHIP"
    )


def test_apply_verdict_underpowered_when_no_d_clears_and_control_cleared_nothing():
    per_d = {d: False for d in cc.D_SWEEP}
    assert (
        cc.apply_verdict(per_d, positive_control_cleared_at=None) == "UNDERPOWERED -- NO CLAIM"
    )


def test_apply_verdict_split_across_d_on_disagreement():
    d_values = list(cc.D_SWEEP)
    per_d = {d: (i == 0) for i, d in enumerate(d_values)}
    assert cc.apply_verdict(per_d, positive_control_cleared_at=0.05) == "SPLIT ACROSS d"


def test_apply_verdict_raises_on_partial_sweep_keys():
    d_values = list(cc.D_SWEEP)
    per_d = {d: True for d in d_values[:-1]}  # missing one d
    with pytest.raises(ValueError):
        cc.apply_verdict(per_d, positive_control_cleared_at=0.05)


def test_apply_verdict_raises_on_extra_key():
    per_d = {d: True for d in cc.D_SWEEP}
    per_d[max(cc.D_SWEEP) + 1] = True  # a d outside D_SWEEP
    with pytest.raises(ValueError):
        cc.apply_verdict(per_d, positive_control_cleared_at=0.05)


# --- split_indices -----------------------------------------------------------------------


def test_split_indices_shape_and_disjointness():
    train_idx, holdout_idx = cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    assert len(train_idx) == 8000
    assert len(holdout_idx) == 2000
    train_set = set(train_idx.tolist())
    holdout_set = set(holdout_idx.tolist())
    assert train_set.isdisjoint(holdout_set)
    assert train_set | holdout_set == set(range(10000))


def test_split_indices_is_deterministic():
    train_a, holdout_a = cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    train_b, holdout_b = cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(holdout_a, holdout_b)
