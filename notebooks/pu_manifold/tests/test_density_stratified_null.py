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
    constant. Parses the source with ``ast`` and asserts no ``import``/``from ... import``
    statement names the sealed Phase 7 module -- deliberately AST-based rather than a raw
    text search, since this module's own docstrings legitimately discuss
    ``crossmodal_curvature`` in prose without importing it."""
    import ast

    module_path = Path(dsn.__file__)
    tree = ast.parse(module_path.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)
    assert not any("crossmodal_curvature" in name for name in imported_names), (
        f"density_stratified_null.py must never import crossmodal_curvature -- every gating "
        f"constant must be a fresh top-level literal (D-08). Found imports: {imported_names}"
    )
    assert "crossmodal_curvature" not in sys.modules or not hasattr(dsn, "crossmodal_curvature")
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
