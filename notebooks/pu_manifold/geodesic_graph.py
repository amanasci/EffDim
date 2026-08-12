"""
notebooks/pu_manifold/geodesic_graph.py -- D-05, D-07: the symmetric kNN-graph geodesic
distance matrix.

Phase 02.7 manifold-template-inference-front-end-inserted. This is D-05's second distance
arm: Isomap's distance without Isomap's embedding step. Build a symmetric kNN graph over
the point cloud, edges weighted by ambient Euclidean length, then take the all-pairs
shortest path. Not diffusion distance, not Euclidean-in-a-spectral-embedding.

**Symmetrization uses the element-wise maximum of the directed graph against its own
transpose, never the sum.** The closest structural analog in this repository --
`src/effdim/geometry.py:434-442`, the frozen `gmst_dimensionality`'s `geodesic=True`
branch -- symmetrizes by adding the directed graph to its own transpose. Verified this
phase's research session on a mutual-nearest-neighbour edge: the additive form returns
twice the true Euclidean distance (measured `2.914` against a true `1.457245873756325`,
matching `scipy.spatial.distance.pdist` to full float64 precision), because both
directions of a mutual-neighbour pair already hold the identical distance value before
the addition. That bug lives in frozen code and is not fixed there (`src/effdim/` stays
untouched for the whole v1.1 milestone); this module's route is the same shape,
corrected. Pinned as a regression test:
`notebooks/pu_manifold/tests/test_geodesic_graph.py::test_maximum_symmetrization_not_sum`.

**This module does not call the frozen squared-distance kNN helper elsewhere in
`src/effdim/geometry.py`.** That helper discards neighbour indices and returns squared
distances -- the wrong contract for a graph, which needs both real (non-squared)
distances and index structure. `sklearn.neighbors.kneighbors_graph` supplies both in one
call and is used here instead; the other helper stays reserved for feeding the frozen
dimensionality estimators' precomputed-distance argument, a different phase 02.7 plan's
concern.

**D-07 forbids repairing disconnection.** Component count at each `k` is a first-class
read-out, not a defect to fix: no spanning-tree bridging of any kind is imported or used
here, and `k` is never silently raised until the graph connects. Geodesic distance is
computed on the largest connected component only; the dropped-point fraction is reported
alongside it, never hidden.

Arrays in, dicts and arrays out -- no file I/O, no cache handling; the runner under
`notebooks/diagnostics/` owns paths and constants.
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph


def build_symmetric_knn_graph(data: np.ndarray, k: int) -> csr_matrix:
    """A symmetric sparse kNN graph over `data`, real (not squared) Euclidean edge
    weights -- D-05's distance. Symmetrized via the element-wise maximum of the directed
    graph against its own transpose (see module docstring for why the additive form is
    forbidden here).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"build_symmetric_knn_graph: data must be a 2-d (n, d) array, got shape "
            f"{data.shape!r}"
        )
    n = data.shape[0]
    if not (isinstance(k, (int, np.integer)) and k > 0):
        raise ValueError(f"build_symmetric_knn_graph: k must be a positive int, got {k!r}")
    if k >= n:
        raise ValueError(
            f"build_symmetric_knn_graph: k={k} must be strictly less than n={n} points"
        )

    directed = kneighbors_graph(data, k, mode="distance")
    graph = directed.maximum(directed.T)
    return graph


def component_readout(graph: csr_matrix) -> Dict[str, Any]:
    """D-07's first-class disconnection read-out: component count, per-point labels, the
    largest component's label/mask/size, and the fraction of points dropped by restricting
    to it. Computed via `scipy.sparse.csgraph.connected_components`. Never repairs
    disconnection -- reads it.

    Returns `{"n_components", "labels", "largest_label", "largest_mask", "largest_size",
    "dropped_fraction"}`.
    """
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest_label = int(np.argmax(sizes))
    largest_mask = labels == largest_label
    largest_size = int(sizes[largest_label])
    dropped_fraction = 1.0 - largest_size / labels.shape[0]
    return {
        "n_components": int(n_components),
        "labels": labels,
        "largest_label": largest_label,
        "largest_mask": largest_mask,
        "largest_size": largest_size,
        "dropped_fraction": float(dropped_fraction),
    }


def geodesic_distance_matrix(data: np.ndarray, k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The graph geodesic distance matrix D-05 needs, restricted to the largest connected
    component only (D-07 -- no bridging, no raising `k` until connected). Returns
    `(D_geo, readout)`: `D_geo` is square, exactly symmetric, all-finite float64, accepted
    by `persistence_probe.persistence_diagram` with no adapter; `readout` is
    `component_readout`'s dict, so the dropped-point fraction always travels with the
    matrix.
    """
    graph = build_symmetric_knn_graph(data, k)
    readout = component_readout(graph)

    largest_mask = readout["largest_mask"]
    sub_graph = graph[largest_mask][:, largest_mask]
    D_geo = np.asarray(shortest_path(sub_graph, directed=False, method="D"), dtype=np.float64)

    if D_geo.ndim != 2 or D_geo.shape[0] != D_geo.shape[1]:
        raise ValueError(
            f"geodesic_distance_matrix: shortest_path returned a non-square array, shape "
            f"{D_geo.shape!r}"
        )
    if not np.allclose(D_geo, D_geo.T):
        raise ValueError("geodesic_distance_matrix: D_geo is not symmetric")
    if not np.all(np.isfinite(D_geo)):
        raise ValueError(
            "geodesic_distance_matrix: D_geo carries non-finite entries after restricting "
            "to the largest connected component -- this should not happen; investigate "
            "component_readout's largest_mask"
        )

    return D_geo, readout
