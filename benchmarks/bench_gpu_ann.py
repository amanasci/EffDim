#!/usr/bin/env python3
"""Benchmark cuVS CAGRA for EffDim's all-points k-NN workload."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force, cagra, ivf_flat


class GpuMonitor:
    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.peak_memory_mib = 0.0
        self.peak_utilization = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                memory, utilization = (float(value.strip()) for value in output.split(","))
                self.peak_memory_mib = max(self.peak_memory_mib, memory)
                self.peak_utilization = max(self.peak_utilization, utilization)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def _synchronize() -> None:
    cp.cuda.get_current_stream().synchronize()


def _timed(operation):
    _synchronize()
    started = time.perf_counter()
    result = operation()
    _synchronize()
    return time.perf_counter() - started, result


def _without_self(
    neighbors: np.ndarray, distances: np.ndarray, query_ids: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    out_neighbors = np.empty((len(query_ids), k), dtype=np.int64)
    out_distances = np.empty((len(query_ids), k), dtype=np.float32)
    for row, query_id in enumerate(query_ids):
        keep = neighbors[row] != query_id
        selected_neighbors = neighbors[row, keep][:k]
        selected_distances = distances[row, keep][:k]
        if len(selected_neighbors) != k:
            raise RuntimeError(f"query {query_id} returned fewer than {k} non-self neighbors")
        out_neighbors[row] = selected_neighbors
        out_distances[row] = selected_distances
    return out_neighbors, out_distances


def _recall(approx: np.ndarray, exact: np.ndarray) -> float:
    matches = sum(
        len(set(approx[row]).intersection(exact[row])) for row in range(len(exact))
    )
    return matches / exact.size


def run_size(
    n_samples: int,
    n_features: int,
    k: int,
    query_count: int,
    itopk_values: list[int],
) -> dict:
    rng = np.random.default_rng(0)
    host_data = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    device_data = cp.asarray(host_data)
    del host_data
    _synchronize()

    query_count = min(query_count, n_samples)
    query_ids = np.arange(query_count, dtype=np.int64)
    query_data = device_data[:query_count]

    with GpuMonitor() as monitor:
        exact_index = brute_force.build(device_data, metric="sqeuclidean")
        exact_s, (exact_distances_dev, exact_neighbors_dev) = _timed(
            lambda: brute_force.search(exact_index, query_data, k + 1)
        )
        exact_neighbors, exact_distances = _without_self(
            cp.asnumpy(cp.asarray(exact_neighbors_dev)),
            cp.asnumpy(cp.asarray(exact_distances_dev)),
            query_ids,
            k,
        )

        build_s, index = _timed(
            lambda: cagra.build(
                cagra.IndexParams(metric="sqeuclidean", build_algo="ivf_pq"),
                device_data,
            )
        )

        searches = []
        for itopk_size in itopk_values:
            params = cagra.SearchParams(itopk_size=itopk_size)
            validation_s, (distances_dev, neighbors_dev) = _timed(
                lambda: cagra.search(params, index, query_data, k=k + 1)
            )
            approx_neighbors, approx_distances = _without_self(
                cp.asnumpy(cp.asarray(neighbors_dev)),
                cp.asnumpy(cp.asarray(distances_dev)),
                query_ids,
                k,
            )
            recall = _recall(approx_neighbors, exact_neighbors)
            distance_ratio = float(np.mean(approx_distances) / np.mean(exact_distances))

            all_search_trials = []
            for _ in range(2):
                all_search_s, all_results = _timed(
                    lambda: cagra.search(params, index, device_data, k=k + 1)
                )
                all_search_trials.append(all_search_s)
                del all_results
            all_search_s = min(all_search_trials)
            searches.append(
                {
                    "itopk_size": itopk_size,
                    "recall_at_k": recall,
                    "mean_distance_ratio": distance_ratio,
                    "validation_queries": query_count,
                    "validation_search_s": validation_s,
                    "all_points_search_trials_s": all_search_trials,
                    "all_points_search_s": all_search_s,
                    "all_points_per_second": n_samples / all_search_s,
                }
            )

    return {
        "algorithm": "cuVS CAGRA",
        "n_samples": n_samples,
        "n_features": n_features,
        "k": k,
        "exact_validation_search_s": exact_s,
        "build_s": build_s,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "peak_gpu_utilization_percent": monitor.peak_utilization,
        "searches": searches,
    }


def run_ivf_size(
    n_samples: int,
    n_features: int,
    k: int,
    query_count: int,
    n_probes_values: list[int],
) -> dict:
    rng = np.random.default_rng(0)
    host_data = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    device_data = cp.asarray(host_data)
    del host_data
    _synchronize()

    query_count = min(query_count, n_samples)
    query_ids = np.arange(query_count, dtype=np.int64)
    query_data = device_data[:query_count]
    n_lists = max(1, round(n_samples**0.5))

    with GpuMonitor() as monitor:
        exact_index = brute_force.build(device_data, metric="sqeuclidean")
        exact_s, (exact_distances_dev, exact_neighbors_dev) = _timed(
            lambda: brute_force.search(exact_index, query_data, k + 1)
        )
        exact_neighbors, exact_distances = _without_self(
            cp.asnumpy(cp.asarray(exact_neighbors_dev)),
            cp.asnumpy(cp.asarray(exact_distances_dev)),
            query_ids,
            k,
        )

        build_s, index = _timed(
            lambda: ivf_flat.build(
                ivf_flat.IndexParams(n_lists=n_lists, metric="sqeuclidean"),
                device_data,
            )
        )
        searches = []
        for n_probes in n_probes_values:
            params = ivf_flat.SearchParams(n_probes=min(n_probes, n_lists))
            validation_s, (distances_dev, neighbors_dev) = _timed(
                lambda: ivf_flat.search(params, index, query_data, k=k + 1)
            )
            approx_neighbors, approx_distances = _without_self(
                cp.asnumpy(cp.asarray(neighbors_dev)),
                cp.asnumpy(cp.asarray(distances_dev)),
                query_ids,
                k,
            )
            recall = _recall(approx_neighbors, exact_neighbors)
            distance_ratio = float(np.mean(approx_distances) / np.mean(exact_distances))

            all_search_trials = []
            for _ in range(2):
                all_search_s, all_results = _timed(
                    lambda: ivf_flat.search(params, index, device_data, k=k + 1)
                )
                all_search_trials.append(all_search_s)
                del all_results
            all_search_s = min(all_search_trials)
            searches.append(
                {
                    "n_probes": min(n_probes, n_lists),
                    "recall_at_k": recall,
                    "mean_distance_ratio": distance_ratio,
                    "validation_queries": query_count,
                    "validation_search_s": validation_s,
                    "all_points_search_trials_s": all_search_trials,
                    "all_points_search_s": all_search_s,
                    "all_points_per_second": n_samples / all_search_s,
                }
            )

    return {
        "algorithm": "cuVS IVF-Flat",
        "n_samples": n_samples,
        "n_features": n_features,
        "k": k,
        "n_lists": n_lists,
        "exact_validation_search_s": exact_s,
        "build_s": build_s,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "peak_gpu_utilization_percent": monitor.peak_utilization,
        "searches": searches,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "n_samples",
                "n_features",
                "k",
                "build_s",
                "n_lists",
                "itopk_size",
                "n_probes",
                "recall_at_k",
                "mean_distance_ratio",
                "validation_queries",
                "validation_search_s",
                "all_points_search_s",
                "all_points_per_second",
                "peak_gpu_memory_mib",
                "peak_gpu_utilization_percent",
            ],
        )
        writer.writeheader()
        for result in results:
            for search in result["searches"]:
                writer.writerow(
                    {
                        key: value
                        for key, value in (result | search).items()
                        if key in writer.fieldnames
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10000,50000,100000")
    parser.add_argument("--n-features", type=int, default=768)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--algorithm", choices=("cagra", "ivf_flat"), default="cagra")
    parser.add_argument("--itopk", default="32,64,128")
    parser.add_argument("--probes", default="8,16,32,64")
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("benchmark-results/gpu_ann_cagra")
    )
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes.split(",")]
    itopk_values = [int(value) for value in args.itopk.split(",")]
    n_probes_values = [int(value) for value in args.probes.split(",")]
    results = []
    for n_samples in sizes:
        print(
            f"{args.algorithm} N={n_samples:,}, D={args.n_features}, k={args.k}",
            flush=True,
        )
        if args.algorithm == "cagra":
            result = run_size(
                n_samples, args.n_features, args.k, args.query_count, itopk_values
            )
        else:
            result = run_ivf_size(
                n_samples,
                args.n_features,
                args.k,
                args.query_count,
                n_probes_values,
            )
        results.append(result)
        print(
            f"  build={result['build_s']:.3f}s, "
            f"peak VRAM={result['peak_gpu_memory_mib'] / 1024:.2f} GiB",
            flush=True,
        )
        for search in result["searches"]:
            tuning = (
                f"itopk={search['itopk_size']:4d}"
                if "itopk_size" in search
                else f"nprobe={search['n_probes']:4d}"
            )
            print(
                f"  {tuning}: "
                f"recall@{args.k}={search['recall_at_k']:.4f}, "
                f"all-search={search['all_points_search_s']:.3f}s, "
                f"{search['all_points_per_second']:,.0f} points/s",
                flush=True,
            )

    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))
    write_csv(csv_path, results)
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
