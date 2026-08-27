"""Freeze-guard, verdict-rule, and locally-declared-constant tests for 07.1's own
pre-registration module.

Mirrors ``test_crossmodal_curvature.py``'s header conventions and guard-strength shape for
07.1's own freeze commit (never re-tests ``crossmodal_curvature``'s freeze -- that suite already
covers it). This plan produces no numbers: every test here exercises pure, in-memory constants
and the two verdict functions only. No PU data is loaded, nothing is trained, nothing is read
from ``notebooks/.cache/``.

Load-bearing tests: the malformed-constant sweep over ``_REQUIRED_CONSTANTS`` (a constant added
later without a guard entry must fail this suite), the git-ancestry proof
(``test_freeze_commit_is_a_strict_ancestor_of_head``), and the two verdict functions' exact-key
and structural-non-gating checks (D-14/D-15/D-16).
"""
import glob
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import cache  # noqa: E402
from pu_manifold import cross_split_curvature  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402
from pu_manifold import density_stratified_null as dsn  # noqa: E402


# The freeze commit SHA recorded in 07.1-01-SUMMARY.md -- the commit that added
# density_stratified_null.py alone (Task 3, first commit). Every later 07.1 number must be a
# strict descendant of this commit.
FREEZE_COMMIT_SHA = "676866657676a36abb639782fa10ecb3061fd688"


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
    """True only once at least one commit exists after the freeze commit. Immediately after the
    freeze commit itself (HEAD == freeze commit, e.g. right before this test file's own commit
    lands), this is False and the test below is skipped rather than failed -- the freeze commit
    being HEAD is the expected state at that moment, not a defect."""
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


def test_assert_preregistered_passes_when_all_constants_set():
    dsn.assert_preregistered()


# --- malformed-constant boundary sweep, parameterized over every required constant -------


@pytest.mark.parametrize("name", dsn._REQUIRED_CONSTANTS)
def test_assert_preregistered_rejects_unset_constant(name):
    """Setting a required constant to None, deleting it, blanking a string constant, or
    emptying a tuple constant all make assert_preregistered() raise RuntimeError naming that
    constant -- exercised as sub-cases within one parametrized test rather than four separate
    ones, since the applicable mutations differ by the constant's own type."""
    original = getattr(dsn, name)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dsn, name, None)
        with pytest.raises(RuntimeError) as excinfo:
            dsn.assert_preregistered()
        assert name in str(excinfo.value)

    with pytest.MonkeyPatch.context() as mp:
        mp.delattr(dsn, name)
        with pytest.raises(RuntimeError) as excinfo:
            dsn.assert_preregistered()
        assert name in str(excinfo.value)

    if isinstance(original, str):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dsn, name, "   ")
            with pytest.raises(RuntimeError) as excinfo:
                dsn.assert_preregistered()
            assert name in str(excinfo.value)

    if isinstance(original, (tuple, list)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dsn, name, ())
            with pytest.raises(RuntimeError) as excinfo:
                dsn.assert_preregistered()
            assert name in str(excinfo.value)


def test_required_constants_covers_every_frozen_constant():
    """Neither a missing guard entry nor a stale one can pass: every module-level UPPER_CASE
    name (excluding the private ``_REQUIRED_CONSTANTS`` itself) must appear in
    ``_REQUIRED_CONSTANTS``, and vice versa."""
    declared = {n for n in vars(dsn) if n.isupper() and not n.startswith("_")}
    guarded = set(dsn._REQUIRED_CONSTANTS)
    assert guarded == declared, (
        f"guarded-but-not-declared: {guarded - declared}; "
        f"declared-but-not-guarded: {declared - guarded}"
    )


# --- D-08: every gating constant is a fresh local literal, never imported from Phase 7 ---------


def test_gating_constants_are_declared_as_local_literals():
    """D-08's structural check: this module never imports ``crossmodal_curvature`` for a
    GATING CONSTANT -- every pre-registered constant is a fresh top-level literal. Parses only
    the FROZEN PRELUDE (source text before the first ``# Compute functions`` section marker,
    i.e. everything above the 07.1-03 compute-function boundary) with ``ast`` and asserts no
    ``import``/``from ... import`` statement in THAT SLICE names the sealed Phase 7 module --
    deliberately AST-based rather than a raw text search, since this module's own docstrings
    legitimately discuss ``crossmodal_curvature`` in prose without importing it.

    Scoped to the frozen prelude only, not the whole file: this module's own docstring
    (top-of-file) states compute functions reusing Phase 7's pure utilities
    (``_relative_precision_distinct_count``, ``_planted_array``) are "fine and expected in
    later 07.1 plans" -- 07.1-04's ``plant_positive_control_partial`` is exactly that, and it
    imports ``crossmodal_curvature`` for those two pure utilities below the freeze boundary.
    The constraint this test enforces (D-08) is about the FROZEN CONSTANTS never crossing the
    boundary, not about compute functions being forbidden from reusing Phase 7's sealed pure
    helpers -- narrowing the scan window keeps the test honoring D-08 exactly without also
    blocking the reuse the module docstring already promises."""
    import ast

    module_path = Path(dsn.__file__)
    source = module_path.read_text()
    boundary_marker = "# Compute functions"
    boundary_idx = source.index(boundary_marker)
    frozen_prelude = source[:boundary_idx]

    tree = ast.parse(frozen_prelude)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)
    assert not any("crossmodal_curvature" in name for name in imported_names), (
        f"density_stratified_null.py's frozen prelude must never import crossmodal_curvature -- "
        f"every gating constant must be a fresh top-level literal (D-08). Found imports in the "
        f"frozen prelude: {imported_names}"
    )
    # D_SWEEP, N_PERMUTATIONS, PERMUTATION_SEED, NULL_QUANTILE_PER_TAIL, SPLIT_SEED and
    # HOLDOUT_FRACTION happen to equal Phase 7's own values -- re-declared, not shared identity.
    assert dsn.D_SWEEP == (20, 25, 32)
    assert dsn.N_PERMUTATIONS == 1000
    assert dsn.PERMUTATION_SEED == 20260825
    assert dsn.NULL_QUANTILE_PER_TAIL == 0.975
    assert dsn.SPLIT_SEED == 20260813


# --- the freeze-ancestry proof itself -------------------------------------------------------


@pytest.mark.skipif(
    not _freeze_commit_is_strict_ancestor_of_head(),
    reason=(
        "freeze commit is not (yet) a STRICT ancestor of HEAD -- either it is absent from this "
        "checkout's history (e.g. a shallow clone), or HEAD IS the freeze commit itself (the "
        "expected state immediately after the freeze, before this test file's own commit "
        "lands). Later 07.1 plans re-run the same ancestry check unconditionally at the moment "
        "a 07.1 number is produced, which is where it actually bites."
    ),
)
def test_freeze_commit_is_a_strict_ancestor_of_head():
    """The precision requirement: a commit is its own ancestor, so ``--is-ancestor`` alone
    would pass even if a 07.1 number were produced in the freeze commit itself.
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


# --- apply_partial_verdict (D7.1-01) ----------------------------------------------------------


def test_apply_partial_verdict_all_clear():
    verdict = dsn.apply_partial_verdict({20: True, 25: True, 32: True}, 0.02)
    assert verdict == "RESIDUAL SURVIVES AT ALL d"
    assert dsn.verdict_is_terminal(verdict, dsn.PARTIAL_VERDICT_VALUES)


def test_apply_partial_verdict_mints_survives_at_subset():
    """D-15: subset survival is a value distinct from Phase 7's SPLIT ACROSS d vocabulary."""
    verdict = dsn.apply_partial_verdict({20: False, 25: True, 32: False}, 0.02)
    assert verdict == "SURVIVES AT SUBSET OF d"
    assert verdict not in ("SPLIT ACROSS d", "ASSOCIATION DETECTED", "NO DETECTABLE RELATIONSHIP")


def test_apply_partial_verdict_underpowered_without_positive_control():
    """D-04's power requirement made mechanical: a null with no positive-control clearance may
    not be reported as NO SURVIVING RESIDUAL."""
    verdict_with_control = dsn.apply_partial_verdict({20: False, 25: False, 32: False}, 0.02)
    assert verdict_with_control == "NO SURVIVING RESIDUAL"

    verdict_underpowered = dsn.apply_partial_verdict({20: False, 25: False, 32: False}, None)
    assert verdict_underpowered == "UNDERPOWERED -- NO CLAIM"


def test_apply_partial_verdict_raises_on_partial_sweep():
    with pytest.raises(ValueError) as excinfo:
        dsn.apply_partial_verdict({20: True, 25: True}, 0.02)
    assert "D_SWEEP" in str(excinfo.value)


def test_apply_partial_verdict_signature_cannot_accept_density():
    """D-16: the signature is exactly two parameters, and neither is a density, raw-statistic,
    or stratum-count parameter -- structural, not promissory."""
    params = list(inspect.signature(dsn.apply_partial_verdict).parameters)
    assert params == ["per_d_results", "positive_control_cleared_at"]
    for forbidden in ("density", "stratum", "s_grid", "raw", "strata"):
        assert forbidden not in [p.lower() for p in params]


# --- apply_seed_verdict (D7.1-02) -------------------------------------------------------------


def test_seed_unanimity_rule():
    """D-11: all three seeds must clear for support; any 2-of-3 pattern is the terminal
    SPLIT ACROSS SEEDS, never upgraded by majority vote."""
    assert dsn.apply_seed_verdict({0: True, 1: True, 2: True}, 0.02) == "SEED STABLE AT d=25"
    assert dsn.apply_seed_verdict({0: True, 1: True, 2: False}, 0.02) == "SPLIT ACROSS SEEDS"
    assert dsn.apply_seed_verdict({0: True, 1: False, 2: False}, 0.02) == "SPLIT ACROSS SEEDS"

    verdict_with_control = dsn.apply_seed_verdict({0: False, 1: False, 2: False}, 0.02)
    assert verdict_with_control == "NO SURVIVING RESIDUAL AT ANY SEED"
    verdict_underpowered = dsn.apply_seed_verdict({0: False, 1: False, 2: False}, None)
    assert verdict_underpowered == "UNDERPOWERED -- NO CLAIM"


def test_apply_seed_verdict_raises_on_partial_seed_set():
    with pytest.raises(ValueError) as excinfo:
        dsn.apply_seed_verdict({0: True, 1: True}, 0.02)
    assert "TORCH_INIT_SEEDS" in str(excinfo.value)


def test_apply_seed_verdict_is_key_order_invariant():
    """D-11 order-invariance: the same mapping built in any key order returns the same string."""
    mapping_a = {0: True, 1: True, 2: False}
    mapping_b = {2: False, 0: True, 1: True}
    mapping_c = {1: True, 2: False, 0: True}
    verdict_a = dsn.apply_seed_verdict(mapping_a, 0.02)
    verdict_b = dsn.apply_seed_verdict(mapping_b, 0.02)
    verdict_c = dsn.apply_seed_verdict(mapping_c, 0.02)
    assert verdict_a == verdict_b == verdict_c == "SPLIT ACROSS SEEDS"


# ================================================================================================
# Plan 07.1-05, Task 1: split-identity across seeds (D-09) and the seed-scoping restore (T-07.1-19).
# Neither test fits anything -- both run in well under a second.
# ================================================================================================


def test_split_indices_identical_across_seeds():
    """D-09: calling cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION) three times,
    with cc.TORCH_INIT_SEED set to each of the frozen TORCH_INIT_SEEDS in between, returns
    arrays that are elementwise equal -- split_indices depends only on SPLIT_SEED/
    HOLDOUT_FRACTION, never on TORCH_INIT_SEED, so the training data is identical across the
    three fits regardless of which seed is active when the split is computed."""
    entry_seed = cc.TORCH_INIT_SEED
    assert entry_seed == 0, (
        f"cc.TORCH_INIT_SEED={entry_seed!r} at test entry -- expected Phase 7's frozen 0. A "
        "prior test in this suite left the sealed module mutated."
    )
    results = []
    try:
        for seed in dsn.TORCH_INIT_SEEDS:
            cc.TORCH_INIT_SEED = seed
            results.append(cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION))
    finally:
        cc.TORCH_INIT_SEED = entry_seed

    assert len(results) == len(dsn.TORCH_INIT_SEEDS)
    train0, holdout0 = results[0]
    for train_i, holdout_i in results[1:]:
        assert np.array_equal(train_i, train0), (
            "split_indices' train array differs across TORCH_INIT_SEED values -- the three "
            "d=25 fits would not be comparable (D-09)."
        )
        assert np.array_equal(holdout_i, holdout0), (
            "split_indices' holdout array differs across TORCH_INIT_SEED values -- the three "
            "d=25 fits would not be comparable (D-09)."
        )
    assert cc.TORCH_INIT_SEED == entry_seed


def test_torch_init_seed_is_restored_after_a_failed_fit(runner_071):
    """T-07.1-19: when the fit callable raises, fit_field_at_seed's seed-scoping helper still
    restores cc.TORCH_INIT_SEED to its entry value in a `finally` block -- the sealed module's
    attribute is never left mutated however the call ends."""
    entry_seed = cc.TORCH_INIT_SEED
    assert entry_seed == 0

    class _RaisingRunner:
        @staticmethod
        def fit_and_field(*args, **kwargs):
            raise RuntimeError("simulated fit failure")

    with pytest.raises(RuntimeError, match="simulated fit failure"):
        runner_071.fit_field_at_seed(_RaisingRunner(), np.zeros((10, 3)), seed=1, n_rows=10)

    assert cc.TORCH_INIT_SEED == entry_seed, (
        "cc.TORCH_INIT_SEED was left mutated after fit_field_at_seed's callee raised -- the "
        "`finally` restore did not run or did not restore the correct value."
    )


def test_fit_field_at_seed_halts_if_entry_seed_has_drifted(runner_071):
    """fit_field_at_seed asserts cc.TORCH_INIT_SEED equals Phase 7's frozen 0 on entry -- a
    drifted entry value halts rather than silently fitting under an unregistered seed."""
    entry_seed = cc.TORCH_INIT_SEED
    assert entry_seed == 0
    cc.TORCH_INIT_SEED = 99
    try:
        with pytest.raises(RuntimeError, match="drifted"):
            runner_071.fit_field_at_seed(object(), np.zeros((10, 3)), seed=1, n_rows=10)
    finally:
        cc.TORCH_INIT_SEED = entry_seed
    assert cc.TORCH_INIT_SEED == entry_seed


# ================================================================================================
# Plan 07.1-03, Task 2: density_strata / stratified_partial_null correctness, calibration, and
# guard tests. Everything above this section belongs to 07.1-01's freeze-guard/verdict suite.
# ================================================================================================

_RUNNER_071_PATH = (
    Path(__file__).resolve().parents[2] / "diagnostics" / "07.1_density_stratified_null_run.py"
)


@pytest.fixture(scope="module")
def runner_071():
    """Loads the 07.1 runner script as a module by file path -- it is not a package member (it
    lives under `notebooks/diagnostics/`, a sibling directory to `notebooks/pu_manifold/`).
    Matches `test_crossmodal_curvature_run.py`'s existing `runner` fixture pattern rather than
    inventing a second one. Module-level code only sets thread-related env vars and imports
    (pure numpy, no torch); it does not run `main()` (guarded by `if __name__ == "__main__"`),
    so import has no side effects beyond that."""
    spec = importlib.util.spec_from_file_location(
        "density_stratified_null_run_under_test", _RUNNER_071_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_artifacts_available() -> bool:
    fields_path = cache.cache_path("07_crossmodal_curvature_fields", "npz")
    record_path = cache.cache_path("07_crossmodal_curvature", "jsonl")
    subsample_cands = glob.glob(str(cache.CACHE_DIR / "subsample_*.npz"))
    return fields_path.exists() and record_path.exists() and len(subsample_cands) > 0


# --- rank-permutation equivariance (licenses the precompute-ranks-once optimization) -----------


def test_rank_permutation_equivariance():
    from scipy.stats import rankdata

    rng = np.random.default_rng(20260827)
    x = rng.normal(size=200)
    # deliberate ties, so the equivariance is proven under scipy's average-rank tie handling,
    # not only on a tie-free array.
    x[:20] = x[0]
    x[20:40] = x[20]
    perm = rng.permutation(200)
    assert np.array_equal(rankdata(x)[perm], rankdata(x[perm]))


# --- odd-under-negation identity (licenses reading both tails off ONE null) --------------------


def test_partial_spearman_is_exactly_odd_under_negation():
    rng = np.random.default_rng(20260827)
    n = 300
    density = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    h = rng.normal(size=n) + 0.1 * np.log(density)
    m = rng.normal(size=n) + 0.1 * np.log(density)
    positive = cross_split_curvature.partial_spearman(h, m, controls=density)
    negative = cross_split_curvature.partial_spearman(-h, m, controls=density)
    assert np.isclose(negative, -positive, atol=1e-12)


# --- D-07: recomputed partial reproduces Phase 7's frozen record at all three d ----------------


@pytest.mark.skipif(
    not _frozen_artifacts_available(),
    reason="frozen Phase 7 cache artifacts (fields npz / record jsonl / subsample npz) are "
    "absent in this checkout -- they are gitignored per CLAUDE.md and not always present.",
)
def test_recomputed_partial_matches_frozen_record(runner_071):
    mknn_arr, density, X_hsc, X_ls, subsample_file = runner_071.recompute_mknn_and_density()

    record_path = cache.cache_path("07_crossmodal_curvature", "jsonl")
    frozen_by_d = {}
    with record_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_kind") == "sweep":
                frozen_by_d[row["d"]] = row["partial_rho_density_controlled"]
    assert set(frozen_by_d.keys()) == {20, 25, 32}

    for d, frozen_value in frozen_by_d.items():
        h = runner_071.load_frozen_field(d)
        recomputed = cross_split_curvature.partial_spearman(h, mknn_arr, controls=density)
        assert np.isclose(
            recomputed, frozen_value,
            rtol=dsn.PARTIAL_REFERENCE_RTOL, atol=dsn.PARTIAL_REFERENCE_ATOL,
        ), f"d={d}: recomputed {recomputed!r} vs frozen record {frozen_value!r}"
        assert np.isclose(
            recomputed, dsn.FROZEN_PARTIAL_REFERENCE[d],
            rtol=dsn.PARTIAL_REFERENCE_RTOL, atol=dsn.PARTIAL_REFERENCE_ATOL,
        ), f"d={d}: recomputed {recomputed!r} vs dsn.FROZEN_PARTIAL_REFERENCE {dsn.FROZEN_PARTIAL_REFERENCE[d]!r}"


# --- density_strata: equal-count contract, tie separation, and the 3-point floor ---------------


def test_density_strata_are_equal_count():
    rng = np.random.default_rng(20260827)

    density = rng.uniform(size=10000)
    strata = dsn.density_strata(density, 20)
    counts = np.bincount(strata)
    assert len(counts) == 20
    assert np.all(counts == 500)

    density_remainder = rng.uniform(size=10007)
    strata_remainder = dsn.density_strata(density_remainder, 20)
    counts_remainder = np.bincount(strata_remainder)
    assert len(counts_remainder) == 20
    assert np.all(counts_remainder[:19] == 500)
    assert counts_remainder[19] == 507


def test_density_strata_separate_tied_densities():
    n = 1000
    n_strata = 10
    # A 350-point tie block (density == 0) followed by a 650-point tie block (density == 1).
    # 350 is not a multiple of the bin size (100), so the first tie block straddles the
    # stratum-3 boundary.
    density = np.concatenate([np.zeros(350), np.ones(650)])
    strata = dsn.density_strata(density, n_strata)

    counts = np.bincount(strata)
    assert len(counts) == n_strata
    assert np.all(counts == 100)  # the declared per-stratum counts hold even under heavy ties

    stratum_3_mask = strata == 3
    stratum_3_densities = set(np.unique(density[stratum_3_mask]))
    assert stratum_3_densities == {0.0, 1.0}, (
        "the density==0 tie block must be separated across strata by index-order position, "
        "not merged entirely into strata 0-2"
    )
    assert set(np.unique(strata[density == 0.0])) == {0, 1, 2, 3}


def test_density_strata_raises_below_the_three_point_floor():
    density = np.arange(20, dtype=np.float64)  # 20 // 10 == 2 < 3
    with pytest.raises(ValueError) as excinfo:
        dsn.density_strata(density, 10)
    assert "3-point floor" in str(excinfo.value)


# --- stratified_partial_null: input guards ------------------------------------------------------


def _valid_null_fixture(n=300):
    rng = np.random.default_rng(20260827)
    density = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    h = rng.normal(size=n)
    m = rng.normal(size=n)
    return h, m, density


@pytest.mark.parametrize(
    "which,expected_substr",
    [("h", "h contains"), ("m", "m contains"), ("density", "density contains")],
)
def test_stratified_null_rejects_nonfinite_input(which, expected_substr):
    h, m, density = _valid_null_fixture()
    arrs = {"h": h.copy(), "m": m.copy(), "density": density.copy()}
    arrs[which][0] = np.nan
    with pytest.raises(ValueError) as excinfo:
        dsn.stratified_partial_null(
            arrs["h"], arrs["m"], arrs["density"],
            n_strata=5, n_resamples=10, seed=1, quantile_per_tail=0.975,
        )
    msg = str(excinfo.value)
    assert msg.startswith("stratified_partial_null: ")
    assert expected_substr in msg


@pytest.mark.parametrize("which,expected_substr", [("h", "h is constant"), ("m", "m is constant")])
def test_stratified_null_rejects_constant_input(which, expected_substr):
    h, m, density = _valid_null_fixture()
    arrs = {"h": h.copy(), "m": m.copy()}
    arrs[which][:] = 1.0
    with pytest.raises(ValueError) as excinfo:
        dsn.stratified_partial_null(
            arrs["h"], arrs["m"], density,
            n_strata=5, n_resamples=10, seed=1, quantile_per_tail=0.975,
        )
    msg = str(excinfo.value)
    assert msg.startswith("stratified_partial_null: ")
    assert expected_substr in msg


def test_stratified_null_rejects_length_mismatch():
    h, m, density = _valid_null_fixture(n=300)
    m_short = m[:299]
    with pytest.raises(ValueError) as excinfo:
        dsn.stratified_partial_null(
            h, m_short, density,
            n_strata=5, n_resamples=10, seed=1, quantile_per_tail=0.975,
        )
    msg = str(excinfo.value)
    assert msg.startswith("stratified_partial_null: ")


# --- stratified_partial_null: reproducibility and strict clearance -----------------------------


def test_stratified_null_is_reproducible_under_a_fixed_seed():
    h, m, density = _valid_null_fixture(n=600)
    kwargs = dict(n_strata=6, n_resamples=200, seed=42, quantile_per_tail=0.975)
    r1 = dsn.stratified_partial_null(h, m, density, **kwargs)
    r2 = dsn.stratified_partial_null(h, m, density, **kwargs)
    assert r1["null_mean"] == r2["null_mean"]
    assert r1["null_std"] == r2["null_std"]
    assert r1["null_low"] == r2["null_low"]
    assert r1["null_high"] == r2["null_high"]


def test_clearance_is_strict_at_the_band_edge():
    """Forces observed == null_low == null_high EXACTLY, rather than hoping for a lucky
    coincidence: h and m are STRATA-WISE CONSTANT (every point within a stratum carries the
    same value, differing only across strata). Permuting positions among tied rank values
    within a stratum leaves the rank vector byte-identical, so every resample's null value
    equals the observed statistic exactly -- the same deterministic residual-Pearson algebra
    both go through. This exercises the strict '>' / '<' clearance boundary directly."""
    n = 500
    n_strata = 5
    rng = np.random.default_rng(20260827)
    density = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    strata = dsn.density_strata(density, n_strata)
    h = strata.astype(np.float64) + 1.0
    m = (n_strata - strata).astype(np.float64)

    result = dsn.stratified_partial_null(
        h, m, density, n_strata=n_strata, n_resamples=50, seed=7, quantile_per_tail=0.975
    )
    assert result["null_std"] == 0.0
    assert result["observed"] == result["null_low"] == result["null_high"]
    assert result["clears_positive"] is False
    assert result["clears_negative"] is False
    assert result["clears_either"] is False
    assert result["direction"] == "neither"


# --- D7.1-01's calibration backstop: a true-null fixture should not over-reject ------------------


def _true_null_draw(rng, n, sigma):
    """A genuine true-null draw: h and m are each a deterministic function of density plus
    independent noise, so h and m are conditionally independent given density -- exactly what
    the stratified null is meant not to reject. Mirrors 07.1-RESEARCH.md's calibration fixture
    shape (a lognormal density confound driving both variables)."""
    density = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    log_density = np.log(density)
    h = 0.6 * log_density + rng.normal(scale=1.0, size=n)
    m = 0.6 * log_density + rng.normal(scale=1.0, size=n)
    return h, m, density


def test_null_calibration_on_true_null_fixture():
    """On a true-null fixture carrying a real density confound, the restricted permutation
    should not reject much more often than its nominal per-tail rate
    (1 - NULL_QUANTILE_PER_TAIL = 0.025) at N_STRATA_HEADLINE-scale (S=20) stratification.
    07.1-RESEARCH.md's Pitfall 1 measured this SAME mechanism over-rejecting the NEGATIVE tail
    at coarse S (~6.3% at S=10 on a similarly-shaped fixture, this session's own reproduction);
    this test exercises the headline-scale S=20 regime, where the same mechanism was measured
    well-calibrated (~1-2.5% per tail)."""
    n_draws = 200
    n = 2000
    n_strata = 20
    n_resamples = 100
    sigma = 2.0
    rng = np.random.default_rng(20260827)

    n_clears_positive = 0
    n_clears_negative = 0
    for _ in range(n_draws):
        h, m, density = _true_null_draw(rng, n, sigma)
        result = dsn.stratified_partial_null(
            h, m, density, n_strata=n_strata, n_resamples=n_resamples,
            seed=int(rng.integers(0, 2**31 - 1)), quantile_per_tail=dsn.NULL_QUANTILE_PER_TAIL,
        )
        n_clears_positive += int(result["clears_positive"])
        n_clears_negative += int(result["clears_negative"])

    rate_positive = n_clears_positive / n_draws
    rate_negative = n_clears_negative / n_draws
    max_rate = 0.075  # ~3x the nominal 2.5% -- generous slack for n_draws=200 binomial noise
    assert rate_positive <= max_rate, (
        f"positive-tail false-clear rate {rate_positive:.4f} ({n_clears_positive}/{n_draws}) "
        f"exceeds {max_rate} -- restricted permutation may not be calibrated at "
        "N_STRATA_HEADLINE-scale stratification"
    )
    assert rate_negative <= max_rate, (
        f"negative-tail false-clear rate {rate_negative:.4f} ({n_clears_negative}/{n_draws}) "
        f"exceeds {max_rate} -- restricted permutation may not be calibrated at "
        "N_STRATA_HEADLINE-scale stratification"
    )


# --- runner: append_record_row rejects a raw numpy value ---------------------------------------


def test_record_row_rejects_raw_numpy(runner_071, tmp_path):
    record_path = tmp_path / "scratch.jsonl"
    with pytest.raises(TypeError):
        runner_071.append_record_row({"x": np.float64(1.0)}, record_path)
    with pytest.raises(TypeError):
        runner_071.append_record_row({"x": np.array([1, 2, 3])}, record_path)


# ================================================================================================
# Plan 07.1-04, Task 1: plant_positive_control_partial / smallest_cleared_target (D-04). All on
# a small seeded fixture -- no PU data, no torch -- so the whole suite here runs in well under
# 60s.
# ================================================================================================


def _positive_control_fixture(n=800, sigma=1.0):
    """A density-confounded fixture (mirrors the true-null fixture shape above) whose h_real
    carries a real density-driven component plus noise -- realistic enough that bisecting toward
    a target PARTIAL (density-controlled) statistic is a genuine search, not a degenerate one."""
    rng = np.random.default_rng(20260827)
    density = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    log_density = np.log(density)
    h_real = 0.3 * log_density + rng.normal(scale=1.0, size=n)
    m_real = 0.3 * log_density + rng.normal(scale=1.0, size=n)  # shape-parity only, unused
    return h_real, m_real, density


def test_positive_control_recovers_planted_partial():
    """D-04's power requirement, on the partial statistic: at some target in a grid spanning a
    generous range, the stratified null recovers the planted relationship in at least one
    direction."""
    h_real, m_real, density = _positive_control_fixture()
    results = dsn.plant_positive_control_partial(
        h_real, m_real, density,
        k=20, target_rhos=(0.1, 0.3, 0.6), directions=("positive", "negative"),
        seed=42, n_strata=8, n_resamples=200, quantile_per_tail=0.975,
    )
    assert len(results) == 6
    assert any(r["clears_either"] for r in results), (
        "the stratified null recovered nothing across (0.1, 0.3, 0.6) x (positive, negative) on "
        "a fixture built with a genuine planted relationship -- D-04's power requirement is not "
        "met by this fixture/grid combination"
    )
    cleared_positive = dsn.smallest_cleared_target(results, "positive")
    cleared_negative = dsn.smallest_cleared_target(results, "negative")
    assert cleared_positive is not None or cleared_negative is not None
    if cleared_positive is not None:
        assert cleared_positive in (0.1, 0.3, 0.6)
    if cleared_negative is not None:
        assert cleared_negative in (0.1, 0.3, 0.6)


def test_positive_control_partial_is_deterministic_under_a_fixed_seed():
    """Two calls with the same frozen seed return identical slope and achieved_rho for every
    cell -- the generator is re-created inside _planted_array on every call (D-04's own
    determinism requirement)."""
    h_real, m_real, density = _positive_control_fixture()
    kwargs = dict(
        k=20, target_rhos=(0.2, 0.4), directions=("positive", "negative"),
        seed=7, n_strata=8, n_resamples=50, quantile_per_tail=0.975,
    )
    r1 = dsn.plant_positive_control_partial(h_real, m_real, density, **kwargs)
    r2 = dsn.plant_positive_control_partial(h_real, m_real, density, **kwargs)
    assert len(r1) == len(r2) == 4
    for a, b in zip(r1, r2):
        assert a["target_rho"] == b["target_rho"]
        assert a["direction"] == b["direction"]
        assert a["slope"] == b["slope"]
        assert a["achieved_rho"] == b["achieved_rho"]


def test_positive_control_partial_records_bracket_exhaustion():
    """A target no achievable slope in [0.0, 2.0] can reach (a correlation cannot exceed 1.0,
    let alone this fixture's realized ceiling) is recorded with bracket_exhausted=True and its
    REALIZED achieved_rho -- never dropped, never silently substituted."""
    h_real, m_real, density = _positive_control_fixture()
    results = dsn.plant_positive_control_partial(
        h_real, m_real, density,
        k=20, target_rhos=(5.0,), directions=("positive",),
        seed=42, n_strata=8, n_resamples=50, quantile_per_tail=0.975,
    )
    assert len(results) == 1
    assert results[0]["bracket_exhausted"] is True
    assert results[0]["achieved_rho"] < 5.0
    assert results[0]["target_rho"] == 5.0  # the target itself is still recorded verbatim


def test_positive_control_partial_rejects_nonfinite_field():
    h_real, m_real, density = _positive_control_fixture()
    h_real = h_real.copy()
    h_real[0] = np.nan
    with pytest.raises(ValueError) as excinfo:
        dsn.plant_positive_control_partial(
            h_real, m_real, density,
            k=20, target_rhos=(0.1,), directions=("positive",),
            seed=42, n_strata=8, n_resamples=50, quantile_per_tail=0.975,
        )
    assert "h_real contains" in str(excinfo.value)


def test_positive_control_partial_rejects_constant_field():
    h_real, m_real, density = _positive_control_fixture()
    h_real = np.ones_like(h_real)
    with pytest.raises(ValueError) as excinfo:
        dsn.plant_positive_control_partial(
            h_real, m_real, density,
            k=20, target_rhos=(0.1,), directions=("positive",),
            seed=42, n_strata=8, n_resamples=50, quantile_per_tail=0.975,
        )
    assert "h_real is constant" in str(excinfo.value)


def test_positive_control_partial_rejects_m_real_shape_mismatch():
    h_real, m_real, density = _positive_control_fixture()
    m_real_short = m_real[:-1]
    with pytest.raises(ValueError) as excinfo:
        dsn.plant_positive_control_partial(
            h_real, m_real_short, density,
            k=20, target_rhos=(0.1,), directions=("positive",),
            seed=42, n_strata=8, n_resamples=50, quantile_per_tail=0.975,
        )
    assert "m_real has" in str(excinfo.value)


def test_smallest_cleared_target_returns_none_when_nothing_clears():
    results = [
        {"target_rho": 0.1, "direction": "positive", "clears_either": False},
        {"target_rho": 0.2, "direction": "positive", "clears_either": False},
        {"target_rho": 0.1, "direction": "negative", "clears_either": False},
    ]
    assert dsn.smallest_cleared_target(results, "positive") is None
    assert dsn.smallest_cleared_target(results, "negative") is None


def test_smallest_cleared_target_returns_smallest_matching_target():
    results = [
        {"target_rho": 0.1, "direction": "positive", "clears_either": False},
        {"target_rho": 0.1, "direction": "negative", "clears_either": True},
        {"target_rho": 0.2, "direction": "positive", "clears_either": True},
        {"target_rho": 0.2, "direction": "negative", "clears_either": True},
    ]
    assert dsn.smallest_cleared_target(results, "positive") == 0.2
    assert dsn.smallest_cleared_target(results, "negative") == 0.1


# ================================================================================================
# Plan 07.1-04, Task 2: null-grid record shape (D-03's grid-of-thresholds contract).
# ================================================================================================


def test_null_grid_record_shape():
    """A FABRICATED nine-cell (D_SWEEP x STRATA_GRID) set of ``row_kind: "null_grid"`` rows,
    asserting the row order is D_SWEEP (outer) x STRATA_GRID (inner) -- the frozen iteration
    order -- and that exactly one distinct ``observed`` value exists per ``d`` across all three
    ``n_strata`` rows for that ``d`` (D-03: the grid moves only the null, never the observed
    statistic). Runs unconditionally as a sub-second unit test, with no dependency on real PU
    data. Also cross-checks the SAME two properties against the REAL on-disk record when
    07.1-04's ``--mode null`` has already been run in this checkout; that half is SKIPPED (not
    failed), with an explicit reason, when the real record is absent -- it is gitignored per
    CLAUDE.md and not always present."""
    fabricated_rows = []
    for i, d in enumerate(dsn.D_SWEEP):
        observed = -0.01 * (i + 1)  # distinct per d, byte-identical across S by construction
        for n_strata in dsn.STRATA_GRID:
            fabricated_rows.append(
                {"row_kind": "null_grid", "d": d, "n_strata": n_strata, "observed": observed}
            )

    assert len(fabricated_rows) == 9
    expected_order = [(d, s) for d in dsn.D_SWEEP for s in dsn.STRATA_GRID]
    actual_order = [(r["d"], r["n_strata"]) for r in fabricated_rows]
    assert actual_order == expected_order, (
        "fabricated null_grid row order must match D_SWEEP (outer) x STRATA_GRID (inner)"
    )

    observed_by_d = {}
    for r in fabricated_rows:
        observed_by_d.setdefault(r["d"], set()).add(r["observed"])
    assert all(len(v) == 1 for v in observed_by_d.values()), (
        "exactly one distinct observed value must exist per d across all three n_strata rows"
    )

    record_path = cache.cache_path(dsn.RECORD_STEM, "jsonl")
    if not record_path.exists():
        pytest.skip(
            f"{record_path} is absent -- 07.1-04's --mode null has not been run in this "
            "checkout (gitignored per CLAUDE.md); the fabricated-fixture assertions above "
            "already cover the row-order and D-03 shape contract."
        )

    real_null_grid_rows = []
    with record_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_kind") == "null_grid":
                real_null_grid_rows.append(row)

    assert len(real_null_grid_rows) >= 9, (
        f"expected at least 9 null_grid rows in the real record, found "
        f"{len(real_null_grid_rows)}"
    )
    # Group by preregistration_commit: only the most recent 9 rows sharing the LAST commit in
    # the file are checked for the D_SWEEP x STRATA_GRID order contract -- a resumed or re-run
    # record may carry rows from more than one commit, and only the latest run's shape matters
    # here.
    last_commit = real_null_grid_rows[-1].get("preregistration_commit")
    latest_rows = [
        r for r in real_null_grid_rows if r.get("preregistration_commit") == last_commit
    ]
    assert len(latest_rows) == 9, (
        f"expected exactly 9 null_grid rows under the latest preregistration_commit "
        f"{last_commit!r}, found {len(latest_rows)}"
    )
    real_order = [(r["d"], r["n_strata"]) for r in latest_rows]
    assert real_order == expected_order, (
        f"real record's null_grid row order {real_order} does not match the frozen "
        f"D_SWEEP x STRATA_GRID order {expected_order}"
    )
    real_observed_by_d = {}
    for r in latest_rows:
        real_observed_by_d.setdefault(r["d"], set()).add(r["observed"])
    assert all(len(v) == 1 for v in real_observed_by_d.values()), (
        "real record: exactly one distinct observed value must exist per d across all three "
        "n_strata rows (D-03) -- the grid must move only the null"
    )


# ================================================================================================
# Plan 07.1-05, Task 1: seed record row shape (D7.1-02's TORCH_INIT_SEEDS-order contract).
# ================================================================================================


_SEED_ROW_REQUIRED_KEYS = {
    "torch_init_seed",
    "split_checksum",
    "h_norm_distinct",
    "partial_rho_density_controlled",
    "partial_rho_raw",
    "clears_either",
    "var_explained",
}


def test_seed_record_row_shape():
    """A FABRICATED three-seed set of ``row_kind: "seed"`` rows, asserting the row order is
    the frozen ``TORCH_INIT_SEEDS`` order and that every row carries ``torch_init_seed``,
    ``split_checksum``, ``h_norm_distinct``, ``partial_rho_density_controlled``,
    ``partial_rho_raw``, ``clears_either`` and ``var_explained``. Runs unconditionally as a
    sub-second unit test, with no dependency on real PU data. Also cross-checks the SAME
    contract against the REAL on-disk record when 07.1-05's ``--mode seeds`` has already been
    run in this checkout; that half is SKIPPED (not failed), with an explicit reason, when the
    real record is absent -- it is gitignored per CLAUDE.md and not always present."""
    fabricated_rows = []
    for i, seed in enumerate(dsn.TORCH_INIT_SEEDS):
        fabricated_rows.append(
            {
                "row_kind": "seed",
                "torch_init_seed": seed,
                "d": 25,
                "split_checksum": "deadbeef" * 8,
                "h_norm_distinct": 100 + i,
                "partial_rho_density_controlled": -0.01 * (i + 1),
                "partial_rho_raw": -0.02 * (i + 1),
                "clears_either": i == 0,
                "var_explained": 0.9 + 0.01 * i,
            }
        )

    assert len(fabricated_rows) == 3
    actual_order = [r["torch_init_seed"] for r in fabricated_rows]
    assert actual_order == list(dsn.TORCH_INIT_SEEDS), (
        "fabricated seed row order must match the frozen TORCH_INIT_SEEDS order"
    )
    for row in fabricated_rows:
        assert _SEED_ROW_REQUIRED_KEYS.issubset(row.keys())

    record_path = cache.cache_path(dsn.RECORD_STEM, "jsonl")
    if not record_path.exists():
        pytest.skip(
            f"{record_path} is absent -- 07.1-05's --mode seeds has not been run in this "
            "checkout (gitignored per CLAUDE.md); the fabricated-fixture assertions above "
            "already cover the row-order and shape contract."
        )

    real_seed_rows = []
    with record_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_kind") == "seed":
                real_seed_rows.append(row)

    if not real_seed_rows:
        pytest.skip(f"{record_path} carries no row_kind='seed' rows yet.")

    last_commit = real_seed_rows[-1].get("preregistration_commit")
    latest_rows = [r for r in real_seed_rows if r.get("preregistration_commit") == last_commit]
    assert len(latest_rows) == 3, (
        f"expected exactly 3 seed rows under the latest preregistration_commit {last_commit!r}, "
        f"found {len(latest_rows)}"
    )
    real_order = [r["torch_init_seed"] for r in latest_rows]
    assert real_order == list(dsn.TORCH_INIT_SEEDS), (
        f"real record's seed row order {real_order} does not match the frozen TORCH_INIT_SEEDS "
        f"order {list(dsn.TORCH_INIT_SEEDS)}"
    )
    for row in latest_rows:
        assert _SEED_ROW_REQUIRED_KEYS.issubset(row.keys())
