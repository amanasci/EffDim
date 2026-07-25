#!/usr/bin/env python3
"""Compare Python and Rust streaming PCA kernels at matched thread counts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def python_streaming(data: np.ndarray, chunk_size: int) -> np.ndarray:
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
            mean, m2, count = chunk_mean, chunk_m2, chunk_count
            continue
        combined = count + chunk_count
        delta = chunk_mean - mean
        m2 += chunk_m2 + np.outer(delta, delta) * (
            count * chunk_count / combined
        )
        mean += delta * (chunk_count / combined)
        count = combined
    return np.maximum(np.linalg.eigvalsh(m2 / (count - 1))[::-1], 0.0)


def timed(operation, repeats: int) -> tuple[float, np.ndarray]:
    times = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = np.asarray(operation())
        times.append(time.perf_counter() - started)
    assert result is not None
    return min(times), result


def pca_95_dimension(eigenvalues: np.ndarray) -> int:
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(eigenvalues) / total, 0.95) + 1)


def rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def worker(
    dataset: Path,
    method: str,
    chunk_size: int,
    thread_count: int,
    repeats: int,
) -> None:
    from effdim import _native

    data = np.load(dataset, mmap_mode="r")
    if method == "python_numpy_blas":
        operation = lambda: python_streaming(data, chunk_size)
    elif method in {"rust_ndarray", "rust_ndarray_openblas"}:
        operation = lambda: _native.spectral_eigenvalues_streaming(data, chunk_size)
    else:
        operation = lambda: _native.spectral_eigenvalues_streaming_faer(
            data, chunk_size, thread_count
        )
    with threadpool_limits(limits=thread_count):
        elapsed, eigenvalues = timed(operation, repeats)
    print(
        json.dumps(
            {
                "best_time_s": elapsed,
                "eigenvalues": eigenvalues.tolist(),
                "pca_95_dimension": pca_95_dimension(eigenvalues),
            }
        ),
        flush=True,
    )


def run_worker(
    dataset: Path,
    method: str,
    chunk_size: int,
    thread_count: int,
    repeats: int,
    interval: float,
) -> tuple[dict, float]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        str(dataset),
        "--method",
        method,
        "--chunk-size",
        str(chunk_size),
        "--thread-count",
        str(thread_count),
        "--repeats",
        str(repeats),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    peak_rss_mib = 0.0
    while process.poll() is None:
        current = rss_mib(process.pid)
        if current is not None:
            peak_rss_mib = max(peak_rss_mib, current)
        time.sleep(interval)
    stdout, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(f"{method} failed for {dataset}:\n{stderr}")
    return json.loads(stdout.strip().splitlines()[-1]), peak_rss_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--threads", default="1,16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.005)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--thread-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(
            args.dataset,
            args.method,
            args.chunk_size,
            args.thread_count,
            args.repeats,
        )
        return
    if not args.datasets or args.output_prefix is None:
        parser.error("datasets and --output-prefix are required")

    threads = [int(value) for value in args.threads.split(",")]
    rows = []
    for path in args.datasets:
        data = np.load(path, mmap_mode="r")
        reference = None
        for thread_count in threads:
            for method in (
                "python_numpy_blas",
                "rust_ndarray_openblas",
                "rust_faer_gemm",
            ):
                payload, peak_rss_mib = run_worker(
                    path,
                    method,
                    args.chunk_size,
                    thread_count,
                    args.repeats,
                    args.interval,
                )
                eigenvalues = np.asarray(payload.pop("eigenvalues"))
                if reference is None:
                    reference = eigenvalues
                denominator = max(float(np.max(np.abs(reference))), 1e-30)
                elapsed = payload["best_time_s"]
                rows.append(
                    {
                        "dataset": path.stem,
                        "n_samples": data.shape[0],
                        "n_features": data.shape[1],
                        "method": method,
                        "threads": thread_count,
                        "best_time_s": elapsed,
                        "samples_per_s": data.shape[0] / elapsed,
                        "pca_95_dimension": payload["pca_95_dimension"],
                        "peak_rss_mib": peak_rss_mib,
                        "max_abs_error": float(np.max(np.abs(eigenvalues - reference))),
                        "max_relative_to_leading_eigenvalue": float(
                            np.max(np.abs(eigenvalues - reference)) / denominator
                        ),
                    }
                )
                print(rows[-1], flush=True)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
