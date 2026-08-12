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


# --- Task 2: local d_hat by anchor-neighbourhood slicing, gmst's separate path -------------


def test_local_estimates_smoke_one_anchor_all_eight_present():
    data = _plane_fixture(n=200, d=2, D=10, seed=FIXTURE_SEED + 1)
    dist_sq_global = geometry.compute_knn_distances(data, 15)
    result = ld.local_estimates(
        data,
        k=15,
        anchor_indices=[0, 50],
        neighbourhood_size=40,
        precomputed_knn_dist_sq=dist_sq_global,
    )
    assert result["anchor_indices"] == [0, 50]
    assert set(result["by_anchor"].keys()) == {0, 50}
    for anchor_id in (0, 50):
        entry = result["by_anchor"][anchor_id]
        assert set(entry.keys()) == set(ld.ESTIMATOR_NAMES)
        for name, cell in entry.items():
            assert np.isfinite(cell["value"])
            if name == "gmst":
                assert cell["provenance"] == "recomputed"
            else:
                assert cell["provenance"] == "sliced"


def test_local_estimates_neighbourhood_below_gmst_floor_raises():
    data = _plane_fixture(n=50, d=2, D=10, seed=FIXTURE_SEED + 2)
    dist_sq_global = geometry.compute_knn_distances(data, 5)
    with pytest.raises(ValueError):
        ld.local_estimates(
            data,
            k=5,
            anchor_indices=[0],
            neighbourhood_size=9,
            precomputed_knn_dist_sq=dist_sq_global,
        )


def test_dispersion_per_estimator_no_aggregate_across_estimators():
    data = _plane_fixture(n=200, d=2, D=10, seed=FIXTURE_SEED + 3)
    dist_sq_global = geometry.compute_knn_distances(data, 15)
    local = ld.local_estimates(
        data,
        k=15,
        anchor_indices=[0, 20, 60, 90],
        neighbourhood_size=30,
        precomputed_knn_dist_sq=dist_sq_global,
    )
    disp = ld.dispersion(local)
    assert set(disp.keys()) == set(ld.ESTIMATOR_NAMES)
    for name, stats in disp.items():
        assert set(stats.keys()) == {"min", "max", "range", "iqr", "n_anchors"}
        assert stats["n_anchors"] == 4
        assert stats["range"] == pytest.approx(stats["max"] - stats["min"])
    # no key aggregates across estimators -- the dict is strictly per-estimator.
    forbidden = {"mean", "median", "overall", "combined"}
    assert forbidden.isdisjoint(disp.keys())


# --- D-12 amendment (02.7-SCREENING-RULE-AMENDMENT-01.md): gating_dispersion ---------------


def test_gating_dispersion_excludes_exactly_gmst_as_a_consequence():
    """The provenance-match predicate is structural (every anchor's "provenance" tag), not a
    hardcoded name list -- this test asserts the CONSEQUENCE (gmst excluded) on the module's
    own fixture, not the mechanism itself."""
    data = _plane_fixture(n=200, d=2, D=10, seed=FIXTURE_SEED + 3)
    dist_sq_global = geometry.compute_knn_distances(data, 15)
    local = ld.local_estimates(
        data,
        k=15,
        anchor_indices=[0, 20, 60, 90],
        neighbourhood_size=30,
        precomputed_knn_dist_sq=dist_sq_global,
    )
    gating = ld.gating_dispersion(local)
    assert set(gating.keys()) == set(ld.ESTIMATOR_NAMES) - {"gmst"}
    assert "gmst" not in gating


def test_gating_dispersion_values_match_full_dispersion_report():
    """gating_dispersion restricts dispersion()'s report -- it never recomputes or perturbs
    the underlying per-estimator statistics."""
    data = _plane_fixture(n=200, d=2, D=10, seed=FIXTURE_SEED + 4)
    dist_sq_global = geometry.compute_knn_distances(data, 15)
    local = ld.local_estimates(
        data,
        k=15,
        anchor_indices=[0, 20, 60, 90],
        neighbourhood_size=30,
        precomputed_knn_dist_sq=dist_sq_global,
    )
    full = ld.dispersion(local)
    gating = ld.gating_dispersion(local)
    for name, stats in gating.items():
        assert stats == full[name]


def test_gating_dispersion_raises_when_no_estimator_is_provenance_matched():
    """A hand-constructed degenerate input (every estimator tagged "recomputed") -- this
    module's real eight estimators never produce this, but gating_dispersion must not
    silently return an empty gate."""
    fake_local = {
        "by_anchor": {
            0: {name: {"value": 2.0, "provenance": "recomputed"} for name in ld.ESTIMATOR_NAMES},
            1: {name: {"value": 2.1, "provenance": "recomputed"} for name in ld.ESTIMATOR_NAMES},
        }
    }
    with pytest.raises(ValueError):
        ld.gating_dispersion(fake_local)


# --- Task 3: pin both planning-time corrections and local/global agreement -----------------


def test_local_matches_global_on_uniform_fixture():
    """D-12: local d_hat is a row-slice of the IDENTICAL global precomputed array for the
    seven sliceable estimators, so on a fixture whose intrinsic dimension is uniform by
    construction (a d-dimensional Gaussian linearly embedded in R^D -- a linear map cannot
    locally distort dimension, so every neighbourhood shares the same true dimension), the
    median local estimate should track the global estimate at the same k. Tolerances are
    stated, not derived: measured max abs diff across 5 independent trials at this fixture
    shape was 0.42 for the seven sliceable estimators and 0.54 for gmst -- both bounds
    below carry real margin above that measured ceiling, not a loosened "just pass" value.

    gmst is the verified exception (module docstring): it recomputes its own kNN structure
    on each local subset rather than slicing a precomputed array, so its provenance is
    `"recomputed"` (not `"sliced"`) and its own looser tolerance is asserted separately --
    this test documents the asymmetry rather than hiding it.
    """
    rng_data = np.random.default_rng(FIXTURE_SEED + 10)
    n, d, D = 600, 3, 15
    Z = rng_data.normal(size=(n, d))
    A = rng_data.normal(size=(d, D))
    data = Z @ A

    k = 15
    estimates_by_k, _ = ld.global_estimates(data, [k])
    global_vals = estimates_by_k[k]

    dist_sq_global = geometry.compute_knn_distances(data, k)
    anchor_indices = list(np.random.default_rng(0).choice(n, size=40, replace=False))
    local = ld.local_estimates(
        data,
        k=k,
        anchor_indices=anchor_indices,
        neighbourhood_size=100,
        precomputed_knn_dist_sq=dist_sq_global,
    )

    sliceable = [name for name in ld.ESTIMATOR_NAMES if name != "gmst"]
    for name in sliceable:
        local_vals = [local["by_anchor"][a][name]["value"] for a in anchor_indices]
        median_local = float(np.median(local_vals))
        diff = abs(median_local - global_vals[name])
        print(f"{name}: median_local={median_local!r} global={global_vals[name]!r} diff={diff!r}")
        assert diff < 1.0, (
            f"{name}: median_local={median_local!r} global={global_vals[name]!r} "
            f"diff={diff!r}"
        )
        for a in anchor_indices:
            assert local["by_anchor"][a][name]["provenance"] == "sliced"

    gmst_local_vals = [local["by_anchor"][a]["gmst"]["value"] for a in anchor_indices]
    gmst_median_local = float(np.median(gmst_local_vals))
    gmst_diff = abs(gmst_median_local - global_vals["gmst"])
    print(f"gmst: median_local={gmst_median_local!r} global={global_vals['gmst']!r} diff={gmst_diff!r}")
    assert gmst_diff < 1.5, (
        f"gmst: median_local={gmst_median_local!r} global={global_vals['gmst']!r} "
        f"diff={gmst_diff!r}"
    )
    for a in anchor_indices:
        assert local["by_anchor"][a]["gmst"]["provenance"] == "recomputed"


def test_tle_is_identical_to_mle():
    """Pins planning-time correction 1: `tle_dimensionality` is mathematically identical
    to `mle_dimensionality` in the FROZEN `src/effdim/geometry.py` -- same expression,
    same epsilon, same `np.mean` reduction (see `local_dimension.py`'s module docstring).
    This is a property of frozen code this phase may not fix; its consequence is that any
    majority vote over the eight estimators receives two guaranteed-correlated votes
    unless `plateau_consensus`'s `count_distinct=True` collapses the pair.

    Exact bit-for-bit equality (`==`) in floating point is DATA-DEPENDENT, not a universal
    mathematical guarantee: `geometry.compute_knn_distances` casts through FAISS's float32
    path, and `log(a) - log(b)` is not bit-identical to `-log(b/a)` in floating point in
    general (a 20-seed scan at the first regime below found 15/20 exactly equal, 5/20
    differing by 1-2 float32 ULPs). The three seeds below were chosen because they
    reproduce exact equality, matching the planning record's own measurement discipline --
    the assertion is about a real, reproduced fact at these seeds, not a universal claim.
    """
    regimes = [
        (500, 3, 10, 1),
        (400, 2, 768, 0),  # D=768: the plan's production-dimensionality criterion
        (300, 5, 20, 0),
    ]
    for n, d, D, seed in regimes:
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n, d))
        A = rng.normal(size=(d, D))
        X = Z @ A
        dist_sq = geometry.compute_knn_distances(X, 10)
        mle = geometry.mle_dimensionality(X, precomputed_knn_dist_sq=dist_sq)
        tle = geometry.tle_dimensionality(X, precomputed_knn_dist_sq=dist_sq)
        print(f"n={n} d={d} D={D} seed={seed}: mle={mle!r} tle={tle!r}")
        assert mle == tle, f"n={n} d={d} D={D} seed={seed}: mle={mle!r} tle={tle!r}"


def test_two_nn_and_mind_mli_are_k_invariant():
    """Pins planning-time correction 2: `two_nn_dimensionality` reads only columns 0-1 of
    `precomputed_knn_dist_sq` (`geometry.py:111-113`) and `mind_mli_dimensionality` reads
    only column 0 (`geometry.py:227`), so slicing the array to different widths returns
    EXACTLY the same value regardless of k -- an invariance that has nothing to do with
    the data. `mle` is asserted to NOT be k-invariant on the same fixture -- the contrast
    is what makes the invariance a finding rather than an artefact of the fixture.
    """
    rng = np.random.default_rng(FIXTURE_SEED + 20)
    n, d, D = 300, 5, 20
    Z = rng.normal(size=(n, d))
    A = rng.normal(size=(d, D))
    X = Z @ A
    k_widths = [5, 10, 20, 30]
    dist_sq_full = geometry.compute_knn_distances(X, max(k_widths))

    two_nn_vals: set = set()
    mind_mli_vals: set = set()
    mle_vals: set = set()
    for k in k_widths:
        d_sq = dist_sq_full[:, :k]
        two_nn_vals.add(geometry.two_nn_dimensionality(X, precomputed_knn_dist_sq=d_sq))
        mind_mli_vals.add(geometry.mind_mli_dimensionality(X, precomputed_knn_dist_sq=d_sq))
        mle_vals.add(geometry.mle_dimensionality(X, precomputed_knn_dist_sq=d_sq))

    print(f"two_nn across k={k_widths}: {two_nn_vals!r}")
    print(f"mind_mli across k={k_widths}: {mind_mli_vals!r}")
    print(f"mle across k={k_widths}: {mle_vals!r}")

    assert len(two_nn_vals) == 1, f"two_nn varied across k: {two_nn_vals!r}"
    assert len(mind_mli_vals) == 1, f"mind_mli varied across k: {mind_mli_vals!r}"
    assert len(mle_vals) > 1, f"mle unexpectedly did not vary across k: {mle_vals!r}"


def test_no_aggregation_in_global_estimates():
    """D-09: the structure `global_estimates` returns contains no key that reduces the
    eight estimators to one number -- disagreement is the signal, and Phase 2's
    `d_frozen = 5` is the record of what averaging it away produces."""
    data = _plane_fixture(n=200, d=2, D=8, seed=FIXTURE_SEED + 30)
    estimates_by_k, k_applicable = ld.global_estimates(data, [10, 15])
    forbidden = {"mean", "median", "consensus", "mode", "average", "std", "aggregate"}
    for values in estimates_by_k.values():
        assert forbidden.isdisjoint(values.keys())
    assert forbidden.isdisjoint(k_applicable.keys())
    assert set(estimates_by_k[10].keys()) == set(ld.ESTIMATOR_NAMES)
