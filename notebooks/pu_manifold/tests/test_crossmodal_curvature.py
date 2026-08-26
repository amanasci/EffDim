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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import crossmodal_curvature as cc  # noqa: E402


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
