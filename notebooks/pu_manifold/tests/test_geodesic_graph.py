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
from scipy.spatial.distance import pdist
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
