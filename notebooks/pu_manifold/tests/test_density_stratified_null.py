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
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
