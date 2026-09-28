"""
notebooks/pu_manifold/tests/test_geodesic_graph.py -- D-05, D-07 regression tests for
`notebooks/pu_manifold/geodesic_graph.py`.

Phase 02.7 manifold-template-inference-front-end-inserted. Pins the verified
mutual-neighbour symmetrization bug (02.7-RESEARCH.md Pitfall 3): the closest structural
analog in this repository, the frozen `src/effdim/geometry.py:gmst_dimensionality`'s
`geodesic=True` branch, symmetrizes a directed kNN graph by summing it with its own
transpose, which DOUBLES every mutual-neighbour edge weight. `geodesic_graph.py` fixes
this with the element-wise maximum instead; the test below asserts the fix holds AND
records what the bug produces, not merely that the fix passes.

Not collected by the core `effdim` test suite (`pyproject.toml`'s `testpaths = ["tests"]`
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_geodesic_graph.py -q

Every test here pins a function against an input whose answer is known independently --
same discipline as `test_persistence_probe.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import kneighbors_graph

from pu_manifold import geodesic_graph as gg
from pu_manifold import persistence_probe as pp


def _mutual_neighbour_fixture(n: int = 20, seed: int = 20270711) -> np.ndarray:
    """A seeded 2-d point cloud, `(n, 2)`, with a guaranteed mutual-nearest-neighbour
    pair at indices 0 and 1: placed close together and pushed well clear of every other
    point, so both directions of the directed kNN graph carry this exact pair -- required
    for both the bug and the fix to be exercised at all.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 10.0, size=(n, 2))
    points[0] = np.array([0.0, 0.0])
    points[1] = np.array([0.3, 0.4])  # true Euclidean distance 0.5 from point 0
    points[2:] += 5.0  # push every other point well away from the (0, 1) pair
    return points


def test_maximum_symmetrization_not_sum():
    points = _mutual_neighbour_fixture()
    k = 5

    true_dist = float(pdist(points[[0, 1]])[0])

    directed = kneighbors_graph(points, k, mode="distance")
    # Confirm points 0 and 1 are mutual nearest neighbours -- both directions must be
    # present in the directed graph for the bug (and its fix) to be exercised.
    assert directed[0, 1] > 0.0
    assert directed[1, 0] > 0.0

    additive = directed + directed.T
    buggy_value = float(additive[0, 1])

    symmetric = gg.build_symmetric_knn_graph(points, k)
    fixed_value = float(symmetric[0, 1])

    print(
        f"true_dist={true_dist!r} buggy_additive_value={buggy_value!r} "
        f"fixed_maximum_value={fixed_value!r}"
    )

    # The fix: element-wise maximum recovers the exact true distance.
    assert abs(fixed_value - true_dist) < 1e-12

    # What the bug produces: additive symmetrization doubles the mutual-neighbour edge.
    assert abs(buggy_value - 2.0 * true_dist) < 1e-12


def test_geodesic_distance_matrix_accepted_by_persistence_diagram():
    """`geodesic_distance_matrix`'s output is square, exactly symmetric, all-finite, and
    `persistence_probe.persistence_diagram` accepts it with no adapter, returning a
    diagram list of length 3 at D-11's `maxdim=2` cap.
    """
    rng = np.random.default_rng(20270712)
    n = 40
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    D_geo, readout = gg.geodesic_distance_matrix(points, k=6)

    assert D_geo.shape[0] == D_geo.shape[1]
    assert np.allclose(D_geo, D_geo.T)
    assert np.all(np.isfinite(D_geo))
    assert readout["n_components"] >= 1

    dgms = pp.persistence_diagram(D_geo, maxdim=2)
    assert len(dgms) == 3


# --- D-06/D-07 regression tests: k-sweep component curve, no bridging ----------------------


def _single_blob_fixture(n: int = 60, seed: int = 20270713) -> np.ndarray:
    """A seeded, tightly-clustered single Gaussian blob, `(n, 3)` -- the positive control:
    a well-sampled single blob should read `n_components == 1` and `dropped_fraction == 0.0`
    at every swept `k`, so a module that always reported disconnection would not pass this
    test.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.3, size=(n, 3))


def _two_blob_fixture(
    n_large: int = 30, n_small: int = 10, seed: int = 20270714, gap: float = 50.0
):
    """Two well-separated, seeded Gaussian blobs of UNEQUAL size, `(n_large + n_small, 3)`
    total. Unequal sizes so "largest component" is unambiguous and `dropped_fraction` has a
    known exact value (`n_small / (n_large + n_small)`) rather than a coin-flip tie. `gap`
    (centre-to-centre offset along one axis) is far larger than either blob's own spread
    (`scale=0.3`), so no `k`-nearest edge at the tested `k` can cross it.

    Returns `(points, n_large, n_small)`.
    """
    rng = np.random.default_rng(seed)
    large = rng.normal(loc=0.0, scale=0.3, size=(n_large, 3))
    small = rng.normal(loc=0.0, scale=0.3, size=(n_small, 3)) + np.array([gap, 0.0, 0.0])
    points = np.vstack([large, small])
    return points, n_large, n_small


def test_k_sweep_components_single_blob_stays_connected():
    """Positive control for the sweep: a well-sampled single blob returns `n_components ==
    1` and `dropped_fraction == 0.0` at every `k` in the swept range."""
    blob = _single_blob_fixture()
    k_values = [3, 5, 8, 10]

    curve = gg.k_sweep_components(blob, k_values)

    assert [entry["k"] for entry in curve] == k_values
    for entry in curve:
        assert entry["n_components"] == 1
        assert entry["dropped_fraction"] == 0.0
        assert "dropped_fraction" in entry


def test_disconnected_graph_no_bridging():
    """D-07's no-repair guarantee, proven numerically rather than assumed: two well-
    separated, unequal-size blobs yield exactly two components, a geodesic matrix sized to
    the larger blob only (never the total), an exact `dropped_fraction` matching the smaller
    blob's share, an all-finite matrix, and -- the direct evidence no fabricated edge
    crossed the gap -- a maximum matrix entry strictly below the true inter-blob
    separation.
    """
    points, n_large, n_small = _two_blob_fixture()
    k = 3

    D_geo, readout = gg.geodesic_distance_matrix(points, k)

    assert readout["n_components"] == 2
    assert D_geo.shape == (n_large, n_large)

    expected_dropped_fraction = n_small / (n_large + n_small)
    assert readout["dropped_fraction"] == expected_dropped_fraction

    assert np.all(np.isfinite(D_geo))

    large_points = points[:n_large]
    small_points = points[n_large:]
    true_inter_blob_separation = float(cdist(large_points, small_points).min())
    assert D_geo.max() < true_inter_blob_separation


def test_k_sweep_components_disconnected_at_small_k():
    """The sweep's own curve reports disconnection directly: at a `k` small enough that no
    edge crosses the gap, the two-blob fixture reads `n_components == 2` and a strictly
    positive `dropped_fraction` in `k_sweep_components`'s output, not merely in the
    single-`k` `geodesic_distance_matrix` call."""
    points, n_large, n_small = _two_blob_fixture()

    curve = gg.k_sweep_components(points, [3])
    entry = curve[0]

    assert entry["n_components"] == 2
    assert entry["dropped_fraction"] > 0.0


def test_geodesic_matrices_over_k_returns_per_k_pairs():
    """`geodesic_matrices_over_k` returns exactly what `geodesic_distance_matrix` returns
    for each swept `k`, keyed by `k`."""
    blob = _single_blob_fixture(n=40)
    k_values = [4, 6]

    matrices = gg.geodesic_matrices_over_k(blob, k_values)

    assert set(matrices.keys()) == set(k_values)
    for k in k_values:
        D_geo, readout = matrices[k]
        expected_D, expected_readout = gg.geodesic_distance_matrix(blob, k)
        assert np.array_equal(D_geo, expected_D)
        assert readout["n_components"] == expected_readout["n_components"]


def test_contiguous_stable_range_finds_the_longest_run():
    result = gg.contiguous_stable_range([1, 1, 2, 2, 2, 2, 2, 3])
    assert result == {"start_index": 2, "end_index": 6, "length": 5, "value": 2}


def test_contiguous_stable_range_no_repeats_returns_length_one():
    result = gg.contiguous_stable_range([1, 2, 3, 4])
    assert result["length"] == 1


def test_geodesic_distance_matrix_min_component_size_guard_raises():
    """T-02.7-05's guard: a caller-supplied `min_component_size` above the actual largest
    component size raises `ValueError` naming the observed `dropped_fraction`, rather than
    returning a matrix over the shard."""
    points, n_large, n_small = _two_blob_fixture()

    try:
        gg.geodesic_distance_matrix(points, k=3, min_component_size=n_large + 1)
    except ValueError as exc:
        assert "dropped_fraction" in str(exc)
    else:
        raise AssertionError(
            "geodesic_distance_matrix did not raise ValueError when the largest "
            "component fell below min_component_size"
        )
