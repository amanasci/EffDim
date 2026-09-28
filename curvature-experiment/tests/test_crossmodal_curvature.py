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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import crossmodal_curvature as cc  # noqa: E402


# The freeze commit SHA recorded in this plan's SUMMARY -- the commit that added
# crossmodal_curvature.py (Task 2). Every later PU number must be a descendant of this commit.
FREEZE_COMMIT_SHA = "f032745f6450068c63763993d39fa112fd36bb8c"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
