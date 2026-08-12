"""
notebooks/pu_manifold/tests/test_local_dimension.py -- D-09, D-10, D-12 regression tests
for `notebooks/pu_manifold/local_dimension.py`.

Phase 02.7 manifold-template-inference-front-end-inserted. Pins the module's two verified
planning-time corrections against the frozen `src/effdim/geometry.py` -- `tle` is
bit-identical to `mle` at the seeds pinned below (`test_tle_is_identical_to_mle`), and
`two_nn`/`mind_mli` are invariant in `k` by construction while `mle` is not
(`test_two_nn_and_mind_mli_are_k_invariant`) -- plus the no-aggregation rule (D-09) and
local-vs-global agreement on a fixture whose answer is known by construction (D-12).

Not collected by the core `effdim` test suite (`pyproject.toml`'s `testpaths = ["tests"]`
excludes this directory) -- run explicitly:

    .venv/bin/python -m pytest notebooks/pu_manifold/tests/test_local_dimension.py -q

Every test here pins a function against an input whose answer is known independently or
against a directly measured structural fact, same discipline as
`test_persistence_probe.py` and `test_geodesic_graph.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from effdim import geometry
from pu_manifold import local_dimension as ld

FIXTURE_SEED = 20260812
"""This plan's own fixed seed, distinct from `persistence_probe`/`decoder_curvature`'s
20260807, since this module's fixtures are unrelated point clouds, not the Swiss roll."""


def _plane_fixture(n: int = 300, d: int = 2, D: int = 10, seed: int = FIXTURE_SEED) -> np.ndarray:
    """A seeded `d`-dimensional Gaussian linearly embedded in `R^D` -- every neighbourhood
    has the same true intrinsic dimension `d` by construction, since a linear map cannot
    locally distort dimension."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    A = rng.normal(size=(d, D))
    return Z @ A


# --- Task 1: eight estimators over a k sweep, no aggregation -------------------------------


def test_constants_shapes():
    assert len(ld.ESTIMATOR_NAMES) == 8
    assert set(ld.K_INVARIANT_ESTIMATORS) == {"two_nn", "mind_mli", "gmst"}
    assert ld.DUPLICATE_ESTIMATOR_PAIRS == (("mle", "tle"),)


def test_global_estimates_eight_finite_values_at_every_k():
    data = _plane_fixture(n=300, d=2, D=10)
    k_values = [5, 10, 20]
    estimates_by_k, k_applicable = ld.global_estimates(data, k_values)

    assert set(estimates_by_k.keys()) == set(k_values)
    assert set(k_applicable.keys()) == set(ld.ESTIMATOR_NAMES)
    assert k_applicable["gmst"] is False
    for name in ld.ESTIMATOR_NAMES:
        if name != "gmst":
            assert k_applicable[name] is True

    for k in k_values:
        values = estimates_by_k[k]
        assert set(values.keys()) == set(ld.ESTIMATOR_NAMES)
        for name, value in values.items():
            assert np.isfinite(value), f"{name} at k={k} is not finite: {value!r}"

    # gmst's value is repeated identically across every swept k -- it has no k to vary with.
    gmst_values = {estimates_by_k[k]["gmst"] for k in k_values}
    assert len(gmst_values) == 1


def test_no_reduction_keys_in_estimates_at_a_single_k():
    data = _plane_fixture(n=300, d=2, D=10)
    estimates_by_k, _ = ld.global_estimates(data, [10])
    forbidden = {"mean", "median", "consensus", "mode", "average"}
    assert forbidden.isdisjoint(estimates_by_k[10].keys())


def test_spread_is_descriptive_only():
    estimates_at_k = {
        "mle": 2.0,
        "two_nn": 2.1,
        "danco": 2.0,
        "mind_mli": 5.0,
        "mind_mlk": 2.0,
        "ess": 2.2,
        "tle": 2.0,
        "gmst": 2.0,
    }
    result = ld.spread(estimates_at_k)
    assert result["min"] == 2.0
    assert result["max"] == 5.0
    assert result["range"] == 3.0
    assert result["distinct_count"] == len({2.0, 2.1, 5.0, 2.2})
    assert result["values"] == estimates_at_k


def _make_estimates_by_k(per_estimator_series):
    """Build an `{k: {name: value}}` fixture from `{name: [values per swept k]}`, k values
    0..len-1."""
    n_k = len(next(iter(per_estimator_series.values())))
    return {
        k: {name: series[k] for name, series in per_estimator_series.items()}
        for k in range(n_k)
    }


def test_plateau_consensus_majority_reaches_consensus():
    # Six of eight estimators sit at 2 across all four swept k's; two do not.
    series = {
        "mle": [2, 2, 2, 2],
        "danco": [2, 2, 2, 2],
        "mind_mlk": [2, 2, 2, 2],
        "ess": [2, 2, 2, 2],
        "tle": [2, 2, 2, 2],
        "gmst": [2, 2, 2, 2],
        "two_nn": [10, 10, 11, 11],
        "mind_mli": [8, 9, 8, 9],
    }
    estimates_by_k = _make_estimates_by_k(series)
    result = ld.plateau_consensus(
        estimates_by_k, majority=5, min_run=3, tolerance=0.5, count_distinct=False
    )
    assert result["d_hat"] == 2
    assert result["reason"] == ""
    assert set(result["supporting_estimators"]) == {
        "mle",
        "danco",
        "mind_mlk",
        "ess",
        "tle",
        "gmst",
    }


def test_plateau_consensus_bimodal_split_abstains():
    # 4/4 split -- neither value reaches a majority of 5.
    series = {
        "mle": [5, 5, 5, 5],
        "danco": [5, 5, 5, 5],
        "tle": [5, 5, 5, 5],
        "two_nn": [5, 5, 5, 5],
        "mind_mlk": [20, 20, 20, 20],
        "ess": [20, 20, 20, 20],
        "gmst": [20, 20, 20, 20],
        "mind_mli": [20, 20, 20, 20],
    }
    estimates_by_k = _make_estimates_by_k(series)
    result = ld.plateau_consensus(
        estimates_by_k, majority=5, min_run=3, tolerance=0.5, count_distinct=False
    )
    assert result["d_hat"] is None
    assert result["reason"] != ""
    assert result["supporting_estimators"] == []


def test_plateau_consensus_signature_has_no_defaults():
    import inspect

    params = inspect.signature(ld.plateau_consensus).parameters
    for name in ("majority", "min_run", "tolerance", "count_distinct"):
        assert params[name].default is inspect.Parameter.empty, name
