#!/usr/bin/env python3
"""Head-to-head Python main versus Rust using shared CAGRA neighbors."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SPECTRAL_NAMES = (
    "pca_explained_variance_95",
    "participation_ratio",
    "shannon_entropy",
    "renyi_eff_dimensionality_alpha_2",
    "renyi_eff_dimensionality_alpha_3",
    "renyi_eff_dimensionality_alpha_4",
    "renyi_eff_dimensionality_alpha_5",
    "geometric_mean_eff_dimensionality",
)


def _timed(operation):
    started = time.perf_counter()
    value = operation()
    return time.perf_counter() - started, value


def _streaming_covariance_python(data, chunk_size: int):
    import numpy as np

    count = 0
    mean = np.zeros(data.shape[1], dtype=np.float64)
    m2 = np.zeros((data.shape[1], data.shape[1]), dtype=np.float64)
    for start in range(0, len(data), chunk_size):
        chunk = np.asarray(data[start : start + chunk_size], dtype=np.float64)
        chunk_mean = np.mean(chunk, axis=0)
        centered = chunk - chunk_mean
        chunk_m2 = centered.T @ centered
        chunk_count = len(chunk)
        if count == 0:
            mean = chunk_mean
            m2 = chunk_m2
            count = chunk_count
            continue
        combined = count + chunk_count
        delta = chunk_mean - mean
        m2 += chunk_m2 + np.outer(delta, delta) * (
            count * chunk_count / combined
        )
        mean += delta * (chunk_count / combined)
        count = combined
    eigenvalues = np.linalg.eigvalsh(m2 / (count - 1))[::-1]
    return np.maximum(eigenvalues, 0.0)


def _spectral_metrics_python(eigenvalues):
    import numpy as np
    from effdim import metrics

    total = float(np.sum(eigenvalues))
    probabilities = (
        np.zeros_like(eigenvalues) if total == 0 else eigenvalues / total
    )
    calls = {
        "pca_explained_variance_95": lambda: metrics.pca_explained_variance(
            eigenvalues, threshold=0.95
        ),
        "participation_ratio": lambda: metrics.participation_ratio(eigenvalues),
        "shannon_entropy": lambda: metrics.shannon_entropy(probabilities),
        "renyi_eff_dimensionality_alpha_2": lambda: metrics.renyi_eff_dimensionality(
            probabilities, alpha=2
        ),
        "renyi_eff_dimensionality_alpha_3": lambda: metrics.renyi_eff_dimensionality(
            probabilities, alpha=3
        ),
        "renyi_eff_dimensionality_alpha_4": lambda: metrics.renyi_eff_dimensionality(
            probabilities, alpha=4
        ),
        "renyi_eff_dimensionality_alpha_5": lambda: metrics.renyi_eff_dimensionality(
            probabilities, alpha=5
        ),
        "geometric_mean_eff_dimensionality": lambda: metrics.geometric_mean_eff_dimensionality(
            probabilities
        ),
    }
    output = {}
    for name, call in calls.items():
        elapsed, value = _timed(call)
        output[name] = {"time_s": elapsed, "value": float(value)}
    return output


def _spectral_metrics_rust(eigenvalues):
    import numpy as np
    from effdim import _native

    total = float(np.sum(eigenvalues))
    probabilities = (
        np.zeros_like(eigenvalues) if total == 0 else eigenvalues / total
    )
    calls = {
        "pca_explained_variance_95": lambda: _native.pca_explained_variance(
            eigenvalues, 0.95
        ),
        "participation_ratio": lambda: _native.participation_ratio(eigenvalues),
        "shannon_entropy": lambda: _native.shannon_entropy(probabilities),
        "renyi_eff_dimensionality_alpha_2": lambda: _native.renyi_eff_dimensionality(
            probabilities, 2.0
        ),
        "renyi_eff_dimensionality_alpha_3": lambda: _native.renyi_eff_dimensionality(
            probabilities, 3.0
        ),
        "renyi_eff_dimensionality_alpha_4": lambda: _native.renyi_eff_dimensionality(
            probabilities, 4.0
        ),
        "renyi_eff_dimensionality_alpha_5": lambda: _native.renyi_eff_dimensionality(
            probabilities, 5.0
        ),
        "geometric_mean_eff_dimensionality": lambda: _native.geometric_mean_eff_dimensionality(
            probabilities
        ),
    }
    output = {}
    for name, call in calls.items():
        elapsed, value = _timed(call)
        output[name] = {"time_s": elapsed, "value": float(value)}
    return output


def _danco_python(data, indices):
    import numpy as np

    vectors = data[indices] - data[:, np.newaxis, :]
    norms = np.linalg.norm(vectors, axis=2, keepdims=True) + 1e-10
    unit_vectors = vectors / norms
    cos_matrix = np.einsum("nik,njk->nij", unit_vectors, unit_vectors)
    triangle = np.triu_indices(indices.shape[1], k=1)
    mean_cos_sq = np.mean(cos_matrix[:, triangle[0], triangle[1]] ** 2)
    return 0.0 if mean_cos_sq < 1e-10 else float(1.0 / mean_cos_sq)


def _ess_python(data, indices):
    import numpy as np

    vectors = data[indices] - data[:, np.newaxis, :]
    norms = np.linalg.norm(vectors, axis=2, keepdims=True) + 1e-10
    centroid = np.mean(vectors / norms, axis=1)
    average = np.mean(np.sum(centroid**2, axis=1))
    return 0.0 if average < 1e-10 else float(1.0 / (indices.shape[1] * average))


def _geometry_python(data, distances, indices):
    import numpy as np
    from effdim import geometry

    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    calls = {
        "mle_dimensionality": lambda: geometry.mle_dimensionality(
            data_f32, precomputed_knn_dist_sq=distances
        ),
        "two_nn_dimensionality": lambda: geometry.two_nn_dimensionality(
            data_f32, precomputed_knn_dist_sq=distances
        ),
        "danco_dimensionality": lambda: _danco_python(data_f32, indices),
        "mind_mli_dimensionality": lambda: geometry.mind_mli_dimensionality(
            data_f32, precomputed_knn_dist_sq=distances
        ),
        "mind_mlk_dimensionality": lambda: geometry.mind_mlk_dimensionality(
            data_f32, precomputed_knn_dist_sq=distances
        ),
        "ess_dimensionality": lambda: _ess_python(data_f32, indices),
        "tle_dimensionality": lambda: geometry.tle_dimensionality(
            data_f32, precomputed_knn_dist_sq=distances
        ),
    }
    output = {}
    for name, call in calls.items():
        elapsed, value = _timed(call)
        output[name] = {"time_s": elapsed, "value": float(value)}
    return output


def _geometry_rust(data, distances, indices):
    from effdim import _native

    calls = {
        "mle_dimensionality": lambda: _native.mle_dimensionality(
            data, 10, distances
        ),
        "two_nn_dimensionality": lambda: _native.two_nn_dimensionality(
            data, distances
        ),
        "danco_dimensionality": lambda: _native.danco_dimensionality(
            data, 10, distances, indices
        ),
        "mind_mli_dimensionality": lambda: _native.mind_mli_dimensionality(
            data, distances
        ),
        "mind_mlk_dimensionality": lambda: _native.mind_mlk_dimensionality(
            data, 10, distances
        ),
        "ess_dimensionality": lambda: _native.ess_dimensionality(
            data, 10, distances, indices
        ),
        "tle_dimensionality": lambda: _native.tle_dimensionality(
            data, 10, distances
        ),
    }
    output = {}
    for name, call in calls.items():
        elapsed, value = _timed(call)
        output[name] = {"time_s": elapsed, "value": float(value)}
    return output


def _worker(
    language: str,
    dataset: Path,
    distances_path: Path,
    indices_path: Path,
    chunk_size: int,
) -> None:
    import numpy as np

    data = np.load(dataset, mmap_mode="r")
    distances = np.load(distances_path, mmap_mode="r")
    indices = np.load(indices_path, mmap_mode="r")

    if language == "python":
        from effdim.api import _do_svd, _ensure_centered

        regular_s, singular_values = _timed(
            lambda: _do_svd(_ensure_centered(np.asarray(data)))
        )
        regular_eigenvalues = singular_values**2 / (len(data) - 1)
        streaming_s, streaming_eigenvalues = _timed(
            lambda: _streaming_covariance_python(data, chunk_size)
        )
        spectral = _spectral_metrics_python(regular_eigenvalues)
        streaming_metrics = _spectral_metrics_python(streaming_eigenvalues)
        geometry_total_s, geometry = _timed(
            lambda: _geometry_python(data, distances, indices)
        )
        geometry_bundle_s = geometry_total_s
    else:
        from effdim import _native

        regular_s, regular_eigenvalues = _timed(
            lambda: np.asarray(_native.spectral_eigenvalues_exact(data))
        )
        streaming_s, streaming_eigenvalues = _timed(
            lambda: np.asarray(
                _native.spectral_eigenvalues_streaming(data, chunk_size)
            )
        )
        spectral = _spectral_metrics_rust(regular_eigenvalues)
        streaming_metrics = _spectral_metrics_rust(streaming_eigenvalues)
        geometry_total_s, geometry = _timed(
            lambda: _geometry_rust(data, distances, indices)
        )
        geometry_bundle_s, _ = _timed(
            lambda: _native.compute_geometry_precomputed(data, distances, indices)
        )

    print(
        json.dumps(
            {
                "language": language,
                "shape": list(data.shape),
                "regular_pca_s": regular_s,
                "streaming_pca_s": streaming_s,
                "chunk_size": chunk_size,
                "regular_eigenvalues": regular_eigenvalues.tolist(),
                "streaming_eigenvalues": streaming_eigenvalues.tolist(),
                "spectral": spectral,
                "streaming_spectral": streaming_metrics,
                "geometry": geometry,
                "geometry_individual_total_s": geometry_total_s,
                "geometry_bundle_s": geometry_bundle_s,
            }
        ),
        flush=True,
    )


def _strip_self(neighbors, distances, k: int):
    import numpy as np

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


def _prepare_cagra(dataset: Path, cache_dir: Path, k: int) -> dict:
    import cupy as cp
    import numpy as np
    from cuvs.neighbors import brute_force, cagra

    cache_dir.mkdir(parents=True, exist_ok=True)
    distances_path = cache_dir / f"{dataset.stem}_cagra_distances.npy"
    indices_path = cache_dir / f"{dataset.stem}_cagra_indices.npy"
    metadata_path = cache_dir / f"{dataset.stem}_cagra.json"
    if distances_path.exists() and indices_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())

    data = np.load(dataset, mmap_mode="r")
    device_data = cp.asarray(data, dtype=cp.float32)
    n_samples = len(data)
    if n_samples <= 10_000:
        itopk_size = 256
    elif n_samples <= 50_000:
        itopk_size = 1024
    else:
        itopk_size = 2048
    cp.cuda.get_current_stream().synchronize()
    build_started = time.perf_counter()
    index = cagra.build(
        cagra.IndexParams(metric="sqeuclidean", build_algo="ivf_pq"), device_data
    )
    cp.cuda.get_current_stream().synchronize()
    build_s = time.perf_counter() - build_started
    params = cagra.SearchParams(itopk_size=itopk_size)
    search_started = time.perf_counter()
    distances_device, neighbors_device = cagra.search(
        params, index, device_data, k=k + 1
    )
    cp.cuda.get_current_stream().synchronize()
    search_s = time.perf_counter() - search_started
    neighbors, distances = _strip_self(
        cp.asnumpy(cp.asarray(neighbors_device)),
        cp.asnumpy(cp.asarray(distances_device)),
        k,
    )
    np.save(distances_path, distances)
    np.save(indices_path, neighbors)

    query_count = min(1000, n_samples)
    exact = brute_force.build(device_data, metric="sqeuclidean")
    exact_distances_device, exact_neighbors_device = brute_force.search(
        exact, device_data[:query_count], k + 1
    )
    exact_neighbors, _ = _strip_self(
        cp.asnumpy(cp.asarray(exact_neighbors_device)),
        cp.asnumpy(cp.asarray(exact_distances_device)),
        k,
    )
    recall = sum(
        len(set(neighbors[row]).intersection(exact_neighbors[row]))
        for row in range(query_count)
    ) / (query_count * k)
    metadata = {
        "dataset": str(dataset),
        "distances": str(distances_path),
        "indices": str(indices_path),
        "itopk_size": itopk_size,
        "build_s": build_s,
        "search_s": search_s,
        "recall_at_k": recall,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def _rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def _run_worker(
    language: str,
    dataset: Path,
    cagra: dict,
    chunk_size: int,
    main_source: Path,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--language",
        language,
        "--dataset",
        str(dataset),
        "--distances",
        cagra["distances"],
        "--indices",
        cagra["indices"],
        "--chunk-size",
        str(chunk_size),
    ]
    environment = os.environ.copy()
    if language == "python":
        environment["PYTHONPATH"] = str(main_source)
    else:
        environment.pop("PYTHONPATH", None)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    peak_rss_mib = 0.0
    while proc.poll() is None:
        rss = _rss_mib(proc.pid)
        if rss is not None:
            peak_rss_mib = max(peak_rss_mib, rss)
        time.sleep(0.1)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"{language} worker failed:\n{stderr}")
    result = json.loads(stdout.strip().splitlines()[-1])
    result["peak_rss_mib"] = peak_rss_mib
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--distances", type=Path)
    parser.add_argument("--indices", type=Path)
    parser.add_argument("--language", choices=("python", "rust"))
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark-results/cache"))
    parser.add_argument("--main-source", type=Path, default=Path.home() / "EffDim-main/src")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/python_rust_head_to_head.json"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(
            args.language,
            args.dataset,
            args.distances,
            args.indices,
            args.chunk_size,
        )
        return

    results = []
    for dataset in args.datasets:
        print(f"Preparing CAGRA: {dataset.name}", flush=True)
        cagra = _prepare_cagra(dataset, args.cache_dir, args.k)
        entry = {
            "dataset": str(dataset),
            "cagra": cagra,
            "implementations": {},
        }
        for language in ("python", "rust"):
            print(f"  Running {language}...", flush=True)
            result = _run_worker(
                language, dataset, cagra, args.chunk_size, args.main_source
            )
            entry["implementations"][language] = result
            print(
                f"    regular PCA={result['regular_pca_s']:.3f}s, "
                f"streaming PCA={result['streaming_pca_s']:.3f}s, "
                f"peak RSS={result['peak_rss_mib'] / 1024:.2f} GiB",
                flush=True,
            )
        results.append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
