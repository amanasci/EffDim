"""
notebooks/pu_manifold/tests/test_template_decision.py -- D-01, D-03, D-04, D-13 regression
tests for `notebooks/pu_manifold/template_decision.py`.

Phase 02.7 manifold-template-inference-front-end-inserted, plan `02.7-06`. Pins the joint
`(Betti vector, d_hat)` lookup against its known-answer rows, proves each of the four named
abstain conditions fires ALONE (never merely that the expected one fired -- the other three
are asserted absent too), proves D-13's three-way tally invariant, and proves the
metric-label invariant `assert_labelled` enforces.

Not collected by the core `effdim` test suite (`pyproject.toml`'s `testpaths = ["tests"]`
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_template_decision.py -q

Every test here pins a function against an input whose answer is known independently --
same discipline as `test_persistence_probe.py` and `test_geodesic_graph.py`. Fixtures below
are the input dicts `decide` documents itself as consuming (a
`local_dimension.plateau_consensus`-shaped `dimension_readout`, a
`geodesic_graph.component_readout`-shaped `connectivity_readout`) rather than full
end-to-end immersion + PH pipelines -- that keeps this file fast and deterministic, exactly
as this module's own docstring recommends ("pure decision logic over already-computed
Betti vectors"); the pipeline wiring itself is exercised by `template_tracer_run.py` and
the benchmark runner, not this test file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from pu_manifold import template_decision as td

# --- fixture builders ------------------------------------------------------------------


def _arms(betti_euclidean, betti_geodesic):
    """A minimal `arms` dict `decide` accepts: one Betti vector per metric label, no bands
    (`decide` reads only `"betti"`)."""
    return {
        "euclidean": {"betti": betti_euclidean},
        "geodesic": {"betti": betti_geodesic},
    }


def _clean_dimension_readout(d_hat, spread_range=0.05, dispersion_range=0.02):
    """A `local_dimension.plateau_consensus`-shaped `dimension_readout` with a resolved
    `d_hat` and both threshold-bearing numbers set well inside the loose bounds
    `_loose_bounds()` supplies, so this fixture trips no abstain condition on its own."""
    return {
        "d_hat": d_hat,
        "reason": "",
        "spread": {"range": spread_range, "min": 0.0, "max": spread_range},
        "local_dispersion": {
            "mle": {"range": dispersion_range, "iqr": dispersion_range / 2, "n_anchors": 5},
            "two_nn": {"range": dispersion_range / 2, "iqr": dispersion_range / 4, "n_anchors": 5},
        },
    }


def _no_consensus_dimension_readout(reason="no integer reached the required majority"):
    """A `local_dimension.plateau_consensus`-shaped `dimension_readout` carrying
    `d_hat is None` -- D-10's own "no consensus, no fallback aggregate" outcome, which
    `decide` must abstain on with condition (d), checked before `"spread"`/
    `"local_dispersion"` are ever read."""
    return {"d_hat": None, "reason": reason}


def _clean_connectivity_readout(n_components=1, dropped_fraction=0.0):
    """A `geodesic_graph.component_readout`-shaped `connectivity_readout`; `decide` reads
    only `"n_components"` and `"dropped_fraction"`."""
    return {"n_components": n_components, "dropped_fraction": dropped_fraction}


def _loose_bounds(dispersion_bound=1.0, spread_bound=1.0):
    """Every required `bounds` key (`td._REQUIRED_BOUNDS`), set loose enough that neither
    threshold trips against `_clean_dimension_readout`'s default numbers."""
    return {"dispersion_bound": dispersion_bound, "spread_bound": spread_bound}


def _assert_only(result, expected_condition):
    """Assert `result` abstained with exactly `expected_condition` fired -- and, since
    `decide` returns a single `abstain_condition` letter rather than four independent
    flags, explicitly assert every OTHER letter did not fire too, per this plan's own
    acceptance criteria ("asserts the three non-expected conditions did not fire, not
    merely that the expected one did")."""
    assert result["abstained"] is True
    assert result["abstain_condition"] == expected_condition
    for other in td.ABSTAIN_CONDITIONS:
        if other != expected_condition:
            assert result["abstain_condition"] != other
    assert result["template"] is None
    assert result["abstain_detail"]


# --- Task 3: known-answer lookup table -------------------------------------------------


def test_lookup_table_known_answers():
    assert td.lookup((1, 1), 1) == "S1"
    assert td.lookup((1, 0, 1), 2) == "S2"
    assert td.lookup((1, 2, 1), 2) == "T2"

    ball_d2 = td.lookup((1, 0, 0), 2)
    ball_d18 = td.lookup((1, 0, 0), 18)
    assert ball_d2 is not None
    assert ball_d18 is not None
    assert ball_d2 != ball_d18, "the ball row must name which ball -- B^2 and B^18 distinct"

    # off-library: no row's betti matches this vector at any d
    assert td.lookup((1, 3, 1), 2) is None

    # a wrong-length Betti vector raises ValueError before any matching happens --
    # D-11 caps homology at H_2, so a 4th-or-later entry is not a valid length.
    with pytest.raises(ValueError):
        td.lookup((1, 1, 1, 1), 2)
    with pytest.raises(ValueError):
        td.lookup((), 2)


# --- Task 3: each abstain condition fires alone -----------------------------------------


def test_abstain_conditions():
    bounds = _loose_bounds()

    # (a) off-library: both arms agree on a Betti vector with a clean d_hat plateau that
    # matches no TEMPLATE_TABLE row at any d (so it cannot be a (d) contradiction either).
    result_a = td.decide(
        _arms((1, 3, 1), (1, 3, 1)),
        _clean_dimension_readout(2),
        _clean_connectivity_readout(),
        bounds,
    )
    _assert_only(result_a, "a")

    # (b) a connectivity read-out with two surviving components, otherwise clean -- the
    # Betti vector (S1-shaped, matching its own d) would be a confident label if the
    # cloud connected, so only the connectivity condition can be responsible for firing.
    result_b = td.decide(
        _arms((1, 1, 0), (1, 1, 0)),
        _clean_dimension_readout(1),
        _clean_connectivity_readout(n_components=2, dropped_fraction=0.1),
        bounds,
    )
    _assert_only(result_b, "b")

    # (c) two arms carrying different Betti vectors, everything else clean.
    result_c = td.decide(
        _arms((1, 1, 0), (1, 0, 0)),
        _clean_dimension_readout(1),
        _clean_connectivity_readout(),
        bounds,
    )
    _assert_only(result_c, "c")

    # (d) a plateau_consensus result carrying d_hat is None.
    result_d_none = td.decide(
        _arms((1, 1, 0), (1, 1, 0)),
        _no_consensus_dimension_readout(),
        _clean_connectivity_readout(),
        bounds,
    )
    _assert_only(result_d_none, "d")

    # (d), fifth case: a Betti/dimension contradiction -- beta_2 = 1 at d_hat = 1 matches
    # S2's betti (1, 0, 1) but not S2's own d = 2 -- routes to abstain (d), not a
    # confident template, per D-04's stated reason for the joint key.
    result_d_contradiction = td.decide(
        _arms((1, 0, 1), (1, 0, 1)),
        _clean_dimension_readout(1),
        _clean_connectivity_readout(),
        bounds,
    )
    _assert_only(result_d_contradiction, "d")


def test_decide_raises_on_missing_bound_key():
    """`decide` raises `ValueError` when any required `bounds` key is absent -- no
    threshold this module compares against ever carries a default."""
    with pytest.raises(ValueError):
        td.decide(
            _arms((1, 1), (1, 1)),
            _clean_dimension_readout(1),
            _clean_connectivity_readout(),
            {"dispersion_bound": 1.0},  # spread_bound missing
        )
    with pytest.raises(ValueError):
        td.decide(
            _arms((1, 1), (1, 1)),
            _clean_dimension_readout(1),
            _clean_connectivity_readout(),
            {"spread_bound": 1.0},  # dispersion_bound missing
        )
    with pytest.raises(ValueError):
        td.decide(
            _arms((1, 1), (1, 1)),
            _clean_dimension_readout(1),
            _clean_connectivity_readout(),
            {},  # both missing
        )


# --- Task 3: the ball row is a positive call, not an abstention ------------------------


def test_ball_row_is_a_positive_call_not_an_abstention():
    """beta_0 = 1, no persistent beta_1, a clean d_hat plateau, both arms agreeing: this
    matches the ball row and must be a positive call, not an abstention -- the likely PU
    answer per the ROADMAP prior, and it must be reachable only as a success."""
    result = td.decide(
        _arms((1, 0, 0), (1, 0, 0)),
        _clean_dimension_readout(18),
        _clean_connectivity_readout(),
        _loose_bounds(),
    )
    assert result["abstained"] is False
    assert result["abstain_condition"] is None
    assert result["template"] == td.lookup((1, 0, 0), 18)
    assert result["template"] is not None


# --- Task 3: the D-13 tally invariant ---------------------------------------------------


def test_tally_sums_to_total():
    results = [
        {"outcome": "correct"},
        {"outcome": "correct"},
        {"outcome": "wrong"},
        {"outcome": "abstained", "abstain_condition": "d"},
        {"outcome": "abstained", "abstain_condition": "d"},
        {"outcome": "abstained", "abstain_condition": "a"},
        {"outcome": "abstained", "abstain_condition": "b"},
    ]
    t = td.tally(results)

    assert t["correct"] == 2
    assert t["wrong"] == 1
    assert t["abstained"] == 4
    assert t["correct"] + t["wrong"] + t["abstained"] == t["total"] == 7

    assert abs(t["accuracy"] - 2 / 7) < 1e-15
    assert abs(t["abstention_rate"] - 4 / 7) < 1e-15
    assert abs(t["error_rate"] - 1 / 7) < 1e-15

    assert sum(t["by_condition"].values()) == t["abstained"]
    assert t["by_condition"]["d"] == 2
    assert t["by_condition"]["a"] == 1
    assert t["by_condition"]["b"] == 1
    assert t["by_condition"]["c"] == 0


def test_tally_raises_on_abstained_without_valid_condition():
    with pytest.raises(ValueError):
        td.tally([{"outcome": "correct"}, {"outcome": "abstained"}])
    with pytest.raises(ValueError):
        td.tally([{"outcome": "abstained", "abstain_condition": "z"}])


# --- Task 3: the metric-label invariant --------------------------------------------------


def test_no_readout_without_metric_label():
    """`assert_labelled` raises on a read-out containing a per-metric value with no metric
    label attached, and a well-formed `decide` output passes it -- this is the invariant
    test for this phase's generalized identity (the primary reporting unit is a (metric,
    read-out) pair, never a bare read-out): it must go red the instant a later phase
    reintroduces a single unlabelled value."""
    with pytest.raises(ValueError):
        td.assert_labelled({"betti": (1, 1, 0)})
    with pytest.raises(ValueError):
        td.assert_labelled({"betti": {"euclidean": (1, 1, 0), "unknown_metric": (1, 1, 0)}})

    # a bare-value dict lacking "betti" entirely is not a per-metric violation -- no
    # tracked key is present at all, so nothing to raise on.
    td.assert_labelled({"template": "S1", "d_hat": 1})

    good = td.decide(
        _arms((1, 1), (1, 1)),
        _clean_dimension_readout(1),
        _clean_connectivity_readout(),
        _loose_bounds(),
    )
    td.assert_labelled(good)  # must not raise
    assert good["betti"] == {"euclidean": (1, 1, 0), "geodesic": (1, 1, 0)}
