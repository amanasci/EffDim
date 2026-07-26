#!/usr/bin/env python3
"""Estimate manifold dimension with CAGRA Landmark Isomap."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from bench_effdim_stability import _strip_self
from bench_manifold_recovery import (
    DEFAULT_MANIFOLDS,
    _add_noise,
    _base_manifold,
    _embed,
)


def _cagra_graph(data: np.ndarray, k: int) -> tuple[csr_matrix, float]:
    import cupy as cp
    from cuvs.neighbors import cagra

    device_data = cp.asarray(data, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    index = cagra.build(
        cagra.IndexParams(metric="sqeuclidean", build_algo="ivf_pq"),
        device_data,
    )
    distances_device, neighbors_device = cagra.search(
        cagra.SearchParams(itopk_size=max(128, 4 * k)),
        index,
        device_data,
        k=k + 1,
    )
    cp.cuda.get_current_stream().synchronize()
    neighbors, distances_sq = _strip_self(
        cp.asnumpy(cp.asarray(neighbors_device)),
        cp.asnumpy(cp.asarray(distances_device)),
        k,
    )
    elapsed = time.perf_counter() - started
    del distances_device, neighbors_device, index, device_data
    cp.get_default_memory_pool().free_all_blocks()

    rows = np.repeat(np.arange(len(data)), k)
    graph = csr_matrix(
        (
            np.sqrt(np.maximum(distances_sq.ravel(), 0.0)),
            (rows, neighbors.ravel()),
        ),
        shape=(len(data), len(data)),
    )
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    return graph, elapsed


def _residual_variance_dimension(
    landmark_distances: np.ndarray,
    max_dimension: int,
    pair_count: int,
    random: np.random.Generator,
) -> tuple[int, list[float], list[float]]:
    landmark_count = landmark_distances.shape[0]
    squared = landmark_distances**2
    centering = np.eye(landmark_count) - np.ones(
        (landmark_count, landmark_count)
    ) / landmark_count
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(1e-12, eigenvalues[0] * 1e-10)
    eigenvalues = eigenvalues[positive][:max_dimension]
    eigenvectors = eigenvectors[:, positive][:, :max_dimension]
    coordinates = eigenvectors * np.sqrt(eigenvalues)

    first = random.integers(0, landmark_count, pair_count)
    second = random.integers(0, landmark_count, pair_count)
    different = first != second
    first, second = first[different], second[different]
    geodesic = landmark_distances[first, second]
    cumulative_sq = np.zeros(len(first), dtype=np.float64)
    residuals = []
    for dimension in range(coordinates.shape[1]):
        difference = (
            coordinates[first, dimension] - coordinates[second, dimension]
        )
        cumulative_sq += difference**2
        embedded = np.sqrt(cumulative_sq)
        correlation = float(np.corrcoef(geodesic, embedded)[0, 1])
        residuals.append(max(0.0, 1.0 - correlation**2))

    if len(residuals) <= 2 or residuals[0] - residuals[-1] < 1e-12:
        estimate = max(1, len(residuals))
        scores = [0.0] * len(residuals)
    else:
        x = np.linspace(0.0, 1.0, len(residuals))
        normalized = (np.asarray(residuals) - residuals[-1]) / (
            residuals[0] - residuals[-1]
        )
        scores_array = (1.0 - x) - normalized
        estimate = int(np.argmax(scores_array)) + 1
        scores = scores_array.tolist()
    return estimate, residuals, scores


def _estimate(
    data: np.ndarray,
    k: int,
    landmarks: int,
    max_dimension: int,
    pair_count: int,
    random: np.random.Generator,
) -> dict:
    graph, graph_s = _cagra_graph(data, k)
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    largest_label = int(np.argmax(component_sizes))
    largest = np.flatnonzero(labels == largest_label)
    component_fraction = len(largest) / len(data)
    subgraph = graph[largest][:, largest]
    landmark_count = min(landmarks, len(largest))
    landmark_local = np.sort(
        random.choice(len(largest), size=landmark_count, replace=False)
    )

    started = time.perf_counter()
    distances = dijkstra(
        subgraph,
        directed=False,
        indices=landmark_local,
        return_predecessors=False,
    )
    landmark_distances = distances[:, landmark_local]
    geodesic_s = time.perf_counter() - started
    if not np.all(np.isfinite(landmark_distances)):
        raise RuntimeError("largest connected component contains infinite distances")

    started = time.perf_counter()
    estimate, residuals, elbow_scores = _residual_variance_dimension(
        landmark_distances,
        max_dimension,
        pair_count,
        random,
    )
    mds_s = time.perf_counter() - started
    return {
        "estimate": estimate,
        "component_count": int(component_count),
        "largest_component_fraction": component_fraction,
        "landmarks_used": landmark_count,
        "residual_variance": residuals,
        "elbow_scores": elbow_scores,
        "timing_s": {
            "cagra_graph": graph_s,
            "landmark_geodesics": geodesic_s,
            "mds_and_elbow": mds_s,
            "total": graph_s + geodesic_s + mds_s,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--ambient-dimension", type=int, default=256)
    parser.add_argument("--snr-db", nargs="+", default=["none", "30", "20", "10"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--landmarks", type=int, default=512)
    parser.add_argument("--max-dimension", type=int, default=50)
    parser.add_argument("--pair-count", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snr_levels = [
        None if value.lower() == "none" else float(value) for value in args.snr_db
    ]

    output = {
        "configuration": {
            "manifolds": [
                {"shape": shape, "intrinsic_dimension": dimension}
                for shape, dimension in DEFAULT_MANIFOLDS
            ],
            "samples": args.samples,
            "ambient_dimension": args.ambient_dimension,
            "snr_db": snr_levels,
            "repeats": args.repeats,
            "k": args.k,
            "landmarks": args.landmarks,
            "max_dimension": args.max_dimension,
            "pair_count": args.pair_count,
            "knn_backend": "cuVS CAGRA",
            "dimension_rule": "maximum normalized residual-variance elbow score",
            "seed": args.seed,
        },
        "conditions": [],
    }
    for manifold_index, (shape, dimension) in enumerate(DEFAULT_MANIFOLDS):
        for snr_index, snr_db in enumerate(snr_levels):
            condition = {
                "shape": shape,
                "intrinsic_dimension": dimension,
                "snr_db": snr_db,
                "trials": [],
            }
            for repeat in range(args.repeats):
                random = np.random.default_rng(
                    args.seed
                    + manifold_index * 100_000
                    + snr_index * 10_000
                    + repeat
                )
                base = _base_manifold(shape, dimension, args.samples, random)
                clean = _embed(base, args.ambient_dimension, random)
                data, _ = _add_noise(clean, snr_db, random)
                print(
                    f"shape={shape}, d={dimension}, snr={snr_db}, "
                    f"repeat={repeat + 1}/{args.repeats}",
                    flush=True,
                )
                trial = _estimate(
                    data,
                    args.k,
                    args.landmarks,
                    args.max_dimension,
                    args.pair_count,
                    random,
                )
                trial["repeat"] = repeat
                condition["trials"].append(trial)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(output, indent=2))
            output["conditions"].append(condition)
            args.output.write_text(json.dumps(output, indent=2))

    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
