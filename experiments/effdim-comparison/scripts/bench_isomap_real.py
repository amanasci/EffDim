#!/usr/bin/env python3
"""Run CAGRA Landmark Isomap on real embedding matrices."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra

from bench_isomap_recovery import (
    _cagra_graph,
    _residual_variance_dimension,
)


def _run_dataset(
    path: Path,
    k: int,
    landmarks: int,
    max_dimension: int,
    pair_count: int,
    repeats: int,
    seed: int,
) -> dict:
    data = np.load(path, mmap_mode="r")
    print(f"{path.stem}: building CAGRA graph for N={len(data):,}", flush=True)
    graph, graph_s = _cagra_graph(data, k)
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    largest_label = int(np.argmax(component_sizes))
    largest = np.flatnonzero(labels == largest_label)
    component_fraction = len(largest) / len(data)
    subgraph = graph[largest][:, largest]
    trials = []
    for repeat in range(repeats):
        random = np.random.default_rng(seed + repeat)
        landmark_count = min(landmarks, len(largest))
        landmark_local = np.sort(
            random.choice(len(largest), size=landmark_count, replace=False)
        )
        print(
            f"  landmarks {repeat + 1}/{repeats}; "
            f"largest component={component_fraction:.3%}",
            flush=True,
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
            raise RuntimeError("largest component contains infinite distances")
        started = time.perf_counter()
        estimate, residuals, elbow_scores = _residual_variance_dimension(
            landmark_distances,
            max_dimension,
            pair_count,
            random,
        )
        mds_s = time.perf_counter() - started
        trials.append(
            {
                "repeat": repeat,
                "estimate": estimate,
                "residual_variance": residuals,
                "elbow_scores": elbow_scores,
                "timing_s": {
                    "landmark_geodesics": geodesic_s,
                    "mds_and_elbow": mds_s,
                    "total_excluding_shared_graph": geodesic_s + mds_s,
                },
            }
        )
    return {
        "dataset": path.stem,
        "path": str(path),
        "shape": list(data.shape),
        "component_count": int(component_count),
        "largest_component_fraction": component_fraction,
        "graph_build_s": graph_s,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--landmarks", type=int, default=512)
    parser.add_argument("--max-dimension", type=int, default=150)
    parser.add_argument("--pair-count", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "configuration": {
            "k": args.k,
            "landmarks": args.landmarks,
            "max_dimension": args.max_dimension,
            "pair_count": args.pair_count,
            "repeats": args.repeats,
            "knn_backend": "cuVS CAGRA",
            "dimension_rule": "maximum normalized residual-variance elbow score",
            "seed": args.seed,
        },
        "datasets": [],
    }
    for index, dataset in enumerate(args.datasets):
        output["datasets"].append(
            _run_dataset(
                dataset,
                args.k,
                args.landmarks,
                args.max_dimension,
                args.pair_count,
                args.repeats,
                args.seed + index * 10_000,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
