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
alongside it, never hidden. `geodesic_distance_matrix` additionally guards against
returning a matrix over a shard: if the largest component holds fewer than a
caller-supplied `min_component_size`, it raises `ValueError` naming the observed
`dropped_fraction` rather than returning a matrix nobody asked for.

**D-06: `k` is not a single chosen value.** Phase 1's frozen `k* = 15` is explicitly NOT
inherited here -- it was frozen for `n = 10,000` PU points at `D = 768`, and this phase's
benchmark varies `n`, `D` and density by design, so a fixed `k` means a different effective
neighbourhood radius in every cell. `k_sweep_components` and `geodesic_matrices_over_k`
report the **whole curve** over a caller-supplied `k_values` sequence -- never a single
chosen `k` -- and `contiguous_stable_range` finds the longest run of a stable value in any
such curve without applying a bar of its own: the "non-trivial contiguous range" minimum
that decides whether a curve counts as stable is a ratified constant plan `02.7-08` fixes,
supplied by the caller, never hard-coded here.

Arrays in, dicts and arrays out -- no file I/O, no cache handling; the runner under
`notebooks/diagnostics/` owns paths and constants.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def geodesic_distance_matrix(
    data: np.ndarray, k: int, min_component_size: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The graph geodesic distance matrix D-05 needs, restricted to the largest connected
    component only (D-07 -- no bridging, no raising `k` until connected). Returns
    `(D_geo, readout)`: `D_geo` is square, exactly symmetric, all-finite float64, accepted
    by `persistence_probe.persistence_diagram` with no adapter; `readout` is
    `component_readout`'s dict, so the dropped-point fraction always travels with the
    matrix.

    `min_component_size`, if given, guards T-02.7-05 (silent scope loss onto a shard):
    when the largest connected component holds fewer points than this, raises `ValueError`
    naming the observed `dropped_fraction` instead of returning a matrix over the shard.
    `None` (the default) applies no guard -- this module picks no numeric minimum of its
    own; the caller who needs one supplies it.
    """
    graph = build_symmetric_knn_graph(data, k)
    readout = component_readout(graph)

    if min_component_size is not None and readout["largest_size"] < min_component_size:
        raise ValueError(
            f"geodesic_distance_matrix: largest connected component holds "
            f"{readout['largest_size']} points, fewer than the required "
            f"min_component_size={min_component_size}; dropped_fraction="
            f"{readout['dropped_fraction']!r}"
        )

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


def k_sweep_components(data: np.ndarray, k_values: Sequence[int]) -> List[Dict[str, Any]]:
    """D-06's sweep over `k_values`: the whole component curve, one entry per `k`, each
    carrying its own `k` plus every key `component_readout` already returns
    (`n_components`, `labels`, `largest_label`, `largest_mask`, `largest_size`,
    `dropped_fraction`). Never a chosen `k` and never a summary statistic over the curve --
    the curve itself, so `contiguous_stable_range` or any other caller can judge stability
    over the full swept range.
    """
    results: List[Dict[str, Any]] = []
    for k in k_values:
        graph = build_symmetric_knn_graph(data, k)
        readout = component_readout(graph)
        entry: Dict[str, Any] = {"k": int(k)}
        entry.update(readout)
        results.append(entry)
    return results


def geodesic_matrices_over_k(
    data: np.ndarray, k_values: Sequence[int]
) -> Dict[int, Tuple[np.ndarray, Dict[str, Any]]]:
    """D-06's sweep, one geodesic distance matrix per `k`: `{k: (D_geo, readout)}` so a
    caller can build one persistence diagram per `k` -- feeding
    `persistence_probe.persistence_diagram` once per sweep point -- without hand-rolling
    the graph-build/component-readout/shortest-path sequence themselves at each `k`. Each
    `(D_geo, readout)` pair is exactly what `geodesic_distance_matrix` returns for that `k`:
    largest-component-only, `dropped_fraction` always attached (D-07), no bridging.
    """
    results: Dict[int, Tuple[np.ndarray, Dict[str, Any]]] = {}
    for k in k_values:
        D_geo, readout = geodesic_distance_matrix(data, k)
        results[int(k)] = (D_geo, readout)
    return results


def contiguous_stable_range(values: Sequence[Any]) -> Dict[str, Any]:
    """The longest run of adjacent equal entries in `values`, as `{"start_index",
    "end_index", "length", "value"}` (both indices inclusive).

    Applies NO minimum-length bar of its own -- takes no such argument -- because D-06's
    "non-trivial contiguous range" threshold is a ratified constant plan `02.7-08` fixes
    and the caller supplies; measuring the run and deciding whether it counts as stable are
    kept separate, the same measure-and-print discipline `persistence_probe.py`'s own
    docstring states for its readout cells.

    On a tie between runs of equal maximal length, the first (lowest `start_index`) wins.
    Raises `ValueError` on an empty `values`.
    """
    if len(values) == 0:
        raise ValueError("contiguous_stable_range: values must be non-empty")

    best_start = 0
    best_length = 1
    best_value = values[0]

    run_start = 0
    for i in range(1, len(values) + 1):
        if i < len(values) and values[i] == values[i - 1]:
            continue
        run_length = i - run_start
        if run_length > best_length:
            best_length = run_length
            best_start = run_start
            best_value = values[run_start]
        run_start = i

    return {
        "start_index": best_start,
        "end_index": best_start + best_length - 1,
        "length": best_length,
        "value": best_value,
    }
