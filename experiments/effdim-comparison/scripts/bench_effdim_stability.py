#!/usr/bin/env python3
"""Measure EffDim stability under bootstrap, sparse sampling, and row regions."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


SPECTRAL_METHODS = (
    "pca_explained_variance_95",
    "participation_ratio",
    "shannon_entropy",
    "renyi_eff_dimensionality_alpha_2",
    "renyi_eff_dimensionality_alpha_3",
    "renyi_eff_dimensionality_alpha_4",
    "renyi_eff_dimensionality_alpha_5",
    "geometric_mean_eff_dimensionality",
)


def _strip_self(neighbors: np.ndarray, distances: np.ndarray, k: int):
    self_mask = neighbors == np.arange(len(neighbors), dtype=neighbors.dtype)[:, None]
    order = np.argsort(self_mask, axis=1, kind="stable")
    return (
        np.ascontiguousarray(
            np.take_along_axis(neighbors, order, axis=1)[:, :k], dtype=np.int64
        ),
        np.ascontiguousarray(
            np.take_along_axis(distances, order, axis=1)[:, :k], dtype=np.float32
        ),
    )


def _exact_gpu_knn(data: np.ndarray, k: int):
    import cupy as cp
    from cuvs.neighbors import brute_force

    device_data = cp.asarray(data, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    index = brute_force.build(device_data, metric="sqeuclidean")
    distances_device, neighbors_device = brute_force.search(
        index, device_data, k=k + 1
    )
    cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - started
    neighbors, distances = _strip_self(
        cp.asnumpy(cp.asarray(neighbors_device)),
        cp.asnumpy(cp.asarray(distances_device)),
        k,
    )
    del distances_device, neighbors_device, index, device_data
    cp.get_default_memory_pool().free_all_blocks()
    return distances, neighbors, elapsed


def _spectral_values(data: np.ndarray, chunk_size: int) -> tuple[dict, float]:
    from effdim import _native

    started = time.perf_counter()
    eigenvalues = np.asarray(
        _native.spectral_eigenvalues_streaming(data, chunk_size), dtype=np.float64
    )
    total = float(np.sum(eigenvalues))
    probabilities = (
        np.zeros_like(eigenvalues) if total <= 0.0 else eigenvalues / total
    )
    values = {
        "pca_explained_variance_95": float(
            _native.pca_explained_variance(eigenvalues, 0.95)
        ),
        "participation_ratio": float(_native.participation_ratio(eigenvalues)),
        "shannon_entropy": float(_native.shannon_entropy(probabilities)),
        "renyi_eff_dimensionality_alpha_2": float(
            _native.renyi_eff_dimensionality(probabilities, 2.0)
        ),
        "renyi_eff_dimensionality_alpha_3": float(
            _native.renyi_eff_dimensionality(probabilities, 3.0)
        ),
        "renyi_eff_dimensionality_alpha_4": float(
            _native.renyi_eff_dimensionality(probabilities, 4.0)
        ),
        "renyi_eff_dimensionality_alpha_5": float(
            _native.renyi_eff_dimensionality(probabilities, 5.0)
        ),
        "geometric_mean_eff_dimensionality": float(
            _native.geometric_mean_eff_dimensionality(probabilities)
        ),
    }
    return values, time.perf_counter() - started


def _compute_all(data: np.ndarray, k: int, chunk_size: int) -> dict:
    from effdim._native import compute_geometry_precomputed

    contiguous = np.ascontiguousarray(data, dtype=np.float64)
    spectral, spectral_s = _spectral_values(contiguous, chunk_size)
    distances, neighbors, knn_s = _exact_gpu_knn(contiguous, k)
    started = time.perf_counter()
    geometry = dict(compute_geometry_precomputed(contiguous, distances, neighbors))
    geometry_s = time.perf_counter() - started
    values = spectral | {name: float(value) for name, value in geometry.items()}
    return {
        "n_samples": len(contiguous),
        "values": values,
        "timing_s": {
            "spectral": spectral_s,
            "knn": knn_s,
            "geometry": geometry_s,
            "total": spectral_s + knn_s + geometry_s,
        },
    }


def _bootstrap_sample(
    data: np.ndarray,
    random: np.random.Generator,
) -> tuple[np.ndarray, int]:
    indices = random.integers(0, len(data), size=len(data))
    unique_indices = np.unique(indices)
    sample = np.ascontiguousarray(data[unique_indices], dtype=np.float64)
    return sample, len(indices) - len(unique_indices)


def _run_dataset(
    path: Path,
    bootstrap_iterations: int,
    sparse_iterations: int,
    sparse_fraction: float,
    regions: int,
    k: int,
    chunk_size: int,
    seed: int,
) -> dict:
    data = np.load(path, mmap_mode="r")
    random = np.random.default_rng(seed)

    print(f"{path.stem}: full N={len(data):,}", flush=True)
    full = _compute_all(data, k, chunk_size)
    trials = []

    for iteration in range(bootstrap_iterations):
        sample, duplicate_count = _bootstrap_sample(data, random)
        print(
            f"  bootstrap {iteration + 1}/{bootstrap_iterations}", flush=True
        )
        result = _compute_all(sample, k, chunk_size)
        result.update(
            {
                "scheme": "bootstrap_deduplicated",
                "iteration": iteration,
                "duplicate_draws_removed": duplicate_count,
            }
        )
        trials.append(result)
        del sample
        gc.collect()

    sparse_size = max(k + 2, round(len(data) * sparse_fraction))
    for iteration in range(sparse_iterations):
        indices = random.choice(len(data), size=sparse_size, replace=False)
        sample = np.ascontiguousarray(data[indices], dtype=np.float64)
        print(f"  sparse {iteration + 1}/{sparse_iterations}", flush=True)
        result = _compute_all(sample, k, chunk_size)
        result.update({"scheme": "sparse_10pct", "iteration": iteration})
        trials.append(result)
        del sample
        gc.collect()

    for region, indices in enumerate(np.array_split(np.arange(len(data)), regions)):
        sample = np.ascontiguousarray(data[indices], dtype=np.float64)
        print(f"  region {region + 1}/{regions}", flush=True)
        result = _compute_all(sample, k, chunk_size)
        result.update(
            {
                "scheme": "contiguous_region",
                "iteration": region,
                "start_row": int(indices[0]),
                "stop_row": int(indices[-1]) + 1,
            }
        )
        trials.append(result)
        del sample
        gc.collect()

    return {
        "dataset": path.stem,
        "path": str(path),
        "shape": list(data.shape),
        "full": full,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=25)
    parser.add_argument("--sparse-iterations", type=int, default=25)
    parser.add_argument("--sparse-fraction", type=float, default=0.10)
    parser.add_argument("--regions", type=int, default=12)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = {
        "configuration": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap": "with replacement; repeated draws collapsed to one row",
            "sparse_iterations": args.sparse_iterations,
            "sparse_fraction": args.sparse_fraction,
            "regions": args.regions,
            "regions_definition": "contiguous stored-row-order chunks",
            "knn_backend": "exact cuVS brute force",
            "gmst_excluded": True,
            "k": args.k,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
        },
        "datasets": [],
    }
    for index, dataset in enumerate(args.datasets):
        output["datasets"].append(
            _run_dataset(
                dataset,
                args.bootstrap_iterations,
                args.sparse_iterations,
                args.sparse_fraction,
                args.regions,
                args.k,
                args.chunk_size,
                args.seed + index * 10_000,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
