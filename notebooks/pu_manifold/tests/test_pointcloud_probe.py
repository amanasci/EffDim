"""Phase 6 pre-registration guards.

The load-bearing test here is :func:`test_inherited_constants_match_phase_5_exactly`: Phase 6's
entire claim is that it changes the curvature field and NOTHING else, and that claim is only
worth as much as a mechanical check of it. Comparing the two constant blocks by eye is exactly
how an "inherited" value silently drifts.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import linear_probe as lp  # noqa: E402
from pu_manifold import pointcloud_probe as pp  # noqa: E402


INHERITED = (
    "TRAIN_FRACTION", "SPLIT_SEED", "RIDGE_ALPHA_GRID", "ALPHA_PER_TARGET", "FIT_INTERCEPT",
    "EMBEDDING_PREPROCESSING", "RESIDUAL_METRIC", "N_BUCKETS", "N_BOOTSTRAP", "BOOTSTRAP_SEED",
    "CONFIDENCE_LEVEL", "SIZE_MATCH_N_REPEATS", "SIZE_MATCH_SEED", "K_DENSITY", "FIELD_D",
    "CURVATURE_CONVENTION",
)


@pytest.mark.parametrize("name", INHERITED)
def test_inherited_constants_match_phase_5_exactly(name):
    """Every constant Phase 6 declares as inherited equals Phase 5's, with `==` and no
    tolerance. If this fails, Phase 6 changed more than the field and its comparison to Phase 5
    is void."""
    assert getattr(pp, name) == getattr(lp, name), (
        f"{name} differs: Phase 6 has {getattr(pp, name)!r}, Phase 5 has {getattr(lp, name)!r}. "
        f"Phase 6 must change the curvature field and nothing else (D6-04)."
    )


def test_phase_6_declares_a_different_curvature_source():
    """The one thing that MUST differ."""
    assert pp.CURVATURE_SOURCE_FUNCTION != lp.CURVATURE_SOURCE_FUNCTION
    assert pp.CURVATURE_SOURCE_FUNCTION == "curvature_probe.centroid_mean_curvature"
    assert lp.CURVATURE_SOURCE_FUNCTION == "chart_curvature.chart_curvature_field"


def test_split_seed_freeze_is_what_makes_the_two_phases_comparable():
    """Same n, same fraction, same seed -> byte-identical held-out row indices."""
    a_train, a_test = lp.train_test_split_indices(10000, lp.TRAIN_FRACTION, lp.SPLIT_SEED)
    b_train, b_test = lp.train_test_split_indices(10000, pp.TRAIN_FRACTION, pp.SPLIT_SEED)
    assert (a_train == b_train).all()
    assert (a_test == b_test).all()
    assert a_test.shape[0] == 3000


def test_no_seed_combination_machinery_is_carried_over():
    """D6-03: a single deterministic field has no seed split, so the phase-level combination
    rule and its verdict vocabulary must be ABSENT, not present-and-unused."""
    assert not hasattr(pp, "SEED_VERDICT_COMBINATION_RULE")
    assert not hasattr(pp, "PHASE_VERDICT_VALUES")
    assert pp.SEED_HANDLING_RULE == "single_field_no_seeds"
    for value in pp.VERDICT_VALUES:
        assert "SPLIT" not in value


def test_split_across_seeds_is_not_a_reachable_outcome():
    """D6-03. The phrase may appear in prose explaining WHY it is unreachable, but it must
    never be a value this phase can emit or a criterion its rule can apply."""
    assert "SPLIT ACROSS SEEDS" not in pp.VERDICT_VALUES
    assert not pp.verdict_is_terminal("SPLIT ACROSS SEEDS")
    # the rule may cite Phase 5's outcome by name when disclaiming it, but must not define a
    # combination step of its own
    assert "combine" not in pp.VERDICT_RULE.lower()
    assert "majority" not in pp.VERDICT_RULE.lower()


def test_assert_preregistered_passes_when_frozen():
    pp.assert_preregistered()


def test_assert_preregistered_names_every_unset_constant(monkeypatch):
    """The guard must refuse, and must name what is missing, so an unfrozen run cannot
    produce a number quietly (D6-05)."""
    monkeypatch.setattr(pp, "VERDICT_RULE", "")
    monkeypatch.setattr(pp, "K_FROZEN", None)
    with pytest.raises(RuntimeError) as excinfo:
        pp.assert_preregistered()
    msg = str(excinfo.value)
    assert "VERDICT_RULE" in msg and "K_FROZEN" in msg and "D6-05" in msg


def test_verdict_rule_carries_its_own_caveats():
    """G6-01/G6-03/G6-04 must live in the rule's own text, not only beside it -- so a reader
    who quotes the rule cannot quote it stripped of its conditions."""
    for token in ("G6-01", "G6-03", "G6-04", "R_H = 0.990", "rho = 0.469", "rule_fired: false"):
        assert token in pp.VERDICT_RULE, f"VERDICT_RULE is missing {token!r}"


def test_verdict_is_terminal_rejects_anything_else():
    assert pp.verdict_is_terminal("HOLDS")
    assert pp.verdict_is_terminal("NO DETECTABLE RELATIONSHIP")
    assert not pp.verdict_is_terminal("SPLIT ACROSS SEEDS")
    assert not pp.verdict_is_terminal("PARTIAL")
