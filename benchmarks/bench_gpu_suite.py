#!/usr/bin/env python3
"""Benchmark EffDim without GMST using GPU k-NN backends."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def _memory(pid: int) -> tuple[float | None, float | None]:
    rss_mib = None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_mib = int(line.split()[1]) / 1024
                break
    except (FileNotFoundError, ProcessLookupError):
        pass
    vram_mib = None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        vram_mib = float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return rss_mib, vram_mib


def _strip_self(neighbors, distances, k: int):
    import numpy as np

    self_mask = neighbors == np.arange(len(neighbors), dtype=neighbors.dtype)[:, None]
    order = np.argsort(self_mask, axis=1, kind="stable")
    neighbors = np.take_along_axis(neighbors, order, axis=1)[:, :k]
    distances = np.take_along_axis(distances, order, axis=1)[:, :k]
    return np.ascontiguousarray(neighbors, dtype=np.int64), np.ascontiguousarray(
        distances, dtype=np.float32
    )


def _recall(approx, exact) -> float:
    matches = sum(
        len(set(approx[row]).intersection(exact[row])) for row in range(len(exact))
    )
    return matches / exact.size


def _worker(backend: str, n_samples: int, n_features: int, k: int) -> None:
    import cupy as cp
    import numpy as np
    from cuvs.neighbors import brute_force, cagra, ivf_flat
    from effdim._native import compute_geometry_precomputed, compute_spectral

    def timed(operation):
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        value = operation()
        cp.cuda.get_current_stream().synchronize()
        return time.perf_counter() - started, value

    total_started = time.perf_counter()
    generation_started = time.perf_counter()
    data = np.random.default_rng(0).standard_normal(
        (n_samples, n_features), dtype=np.float64
    )
    generation_s = time.perf_counter() - generation_started
    device_data = cp.asarray(data, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()

    tuning: dict[str, int | float] = {}
    if backend == "exact":
        build_s, index = timed(
            lambda: brute_force.build(device_data, metric="sqeuclidean")
        )
        search = lambda queries: brute_force.search(index, queries, k + 1)
    elif backend == "cagra":
        if n_samples <= 10_000:
            itopk_size = 256
        elif n_samples <= 50_000:
            itopk_size = 1024
        else:
            itopk_size = 2048
        tuning["itopk_size"] = itopk_size
        build_s, index = timed(
            lambda: cagra.build(
                cagra.IndexParams(metric="sqeuclidean", build_algo="ivf_pq"),
                device_data,
            )
        )
        params = cagra.SearchParams(itopk_size=itopk_size)
        search = lambda queries: cagra.search(params, index, queries, k=k + 1)
    else:
        n_lists = max(1, round(math.sqrt(n_samples)))
        n_probes = min(n_lists, math.ceil(0.8 * n_lists))
        tuning.update({"n_lists": n_lists, "n_probes": n_probes})
        build_s, index = timed(
            lambda: ivf_flat.build(
                ivf_flat.IndexParams(n_lists=n_lists, metric="sqeuclidean"),
                device_data,
            )
        )
        params = ivf_flat.SearchParams(n_probes=n_probes)
        search = lambda queries: ivf_flat.search(params, index, queries, k=k + 1)

    knn_s, (distances_device, neighbors_device) = timed(lambda: search(device_data))
    transfer_started = time.perf_counter()
    neighbors, distances = _strip_self(
        cp.asnumpy(cp.asarray(neighbors_device)),
        cp.asnumpy(cp.asarray(distances_device)),
        k,
    )
    transfer_s = time.perf_counter() - transfer_started

    query_count = min(1000, n_samples)
    if backend == "exact":
        recall_at_k = 1.0
    else:
        exact_index = brute_force.build(device_data, metric="sqeuclidean")
        _, (exact_distances_device, exact_neighbors_device) = timed(
            lambda: brute_force.search(exact_index, device_data[:query_count], k + 1)
        )
        exact_neighbors, _ = _strip_self(
            cp.asnumpy(cp.asarray(exact_neighbors_device)),
            cp.asnumpy(cp.asarray(exact_distances_device)),
            k,
        )
        recall_at_k = _recall(neighbors[:query_count], exact_neighbors)

    spectral_started = time.perf_counter()
    spectral = dict(compute_spectral(data))
    spectral_s = time.perf_counter() - spectral_started
    geometry_started = time.perf_counter()
    geometry = dict(compute_geometry_precomputed(data, distances, neighbors))
    geometry_s = time.perf_counter() - geometry_started
    results = spectral | geometry

    print(
        json.dumps(
            {
                "backend": backend,
                "n_samples": n_samples,
                "n_features": n_features,
                "k": k,
                "metrics": len(results),
                "recall_at_k": recall_at_k,
                "generation_s": generation_s,
                "index_build_s": build_s,
                "knn_search_s": knn_s,
                "transfer_s": transfer_s,
                "spectral_s": spectral_s,
                "geometry_s": geometry_s,
                "total_s": time.perf_counter() - total_started,
                "tuning": tuning,
                "results": results,
            }
        ),
        flush=True,
    )


def _run_worker(
    backend: str, n_samples: int, n_features: int, k: int, interval_s: float
) -> tuple[dict, list[tuple[float, float, float]]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--n-samples",
        str(n_samples),
        "--n-features",
        str(n_features),
        "--k",
        str(k),
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = []
    while proc.poll() is None:
        rss_mib, vram_mib = _memory(proc.pid)
        if rss_mib is not None and vram_mib is not None:
            samples.append((time.perf_counter() - started, rss_mib, vram_mib))
        time.sleep(interval_s)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(
            f"{backend} N={n_samples:,} failed ({proc.returncode}):\n{stderr}"
        )
    result = json.loads(stdout.strip().splitlines()[-1])
    result["peak_rss_mib"] = max((sample[1] for sample in samples), default=0.0)
    result["peak_vram_mib"] = max((sample[2] for sample in samples), default=0.0)
    return result, samples


def _write_summary(path: Path, results: list[dict]) -> None:
    fields = [
        "backend",
        "n_samples",
        "n_features",
        "metrics",
        "recall_at_k",
        "index_build_s",
        "knn_search_s",
        "spectral_s",
        "geometry_s",
        "total_s",
        "peak_rss_mib",
        "peak_vram_mib",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def _write_memory(
    path: Path, runs: list[tuple[dict, list[tuple[float, float, float]]]]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "n_samples", "elapsed_s", "rss_mib", "vram_mib"])
        for result, samples in runs:
            for elapsed_s, rss_mib, vram_mib in samples:
                writer.writerow(
                    [
                        result["backend"],
                        result["n_samples"],
                        elapsed_s,
                        rss_mib,
                        vram_mib,
                    ]
                )


def _write_memory_svg(csv_path: Path, svg_path: Path) -> None:
    grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["backend"], int(row["n_samples"]))
            grouped.setdefault(key, []).append(
                (
                    float(row["elapsed_s"]),
                    float(row["rss_mib"]) / 1024,
                    float(row["vram_mib"]) / 1024,
                )
            )

    width, height = 1100, 680
    left, right, top, bottom = 85, 35, 55, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    max_t = max(value[0] for samples in grouped.values() for value in samples)
    max_memory = max(
        max(value[1], value[2]) for samples in grouped.values() for value in samples
    )
    colors = {"exact": "#2563eb", "cagra": "#dc2626", "ivf_flat": "#16a34a"}
    dashes = {10_000: "", 50_000: "8 4", 100_000: "2 3"}

    def x(value: float) -> float:
        return left + value / max_t * plot_w

    def y(value: float) -> float:
        return top + plot_h - value / max_memory * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">EffDim GPU suite memory profile (GMST excluded)</text>',
    ]
    for tick in range(6):
        elapsed = max_t * tick / 5
        memory = max_memory * tick / 5
        lines.extend(
            [
                f'<line x1="{x(elapsed):.1f}" y1="{top}" x2="{x(elapsed):.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>',
                f'<text x="{x(elapsed):.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{elapsed:.0f}</text>',
                f'<line x1="{left}" y1="{y(memory):.1f}" x2="{left + plot_w}" y2="{y(memory):.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y(memory) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{memory:.1f}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 82}" text-anchor="middle" font-family="sans-serif" font-size="14">Elapsed time per sequential run (s)</text>',
            f'<text x="20" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2:.1f})" font-family="sans-serif" font-size="14">Memory (GiB)</text>',
        ]
    )
    for (backend, n_samples), samples in grouped.items():
        color = colors[backend]
        dash = dashes.get(n_samples, "")
        for value_index, opacity in ((1, "1.0"), (2, "0.45")):
            points = " ".join(
                f"{x(sample[0]):.1f},{y(sample[value_index]):.1f}" for sample in samples
            )
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-opacity="{opacity}" stroke-width="2" stroke-dasharray="{dash}"/>'
            )
    legend_y = height - 50
    for index, backend in enumerate(("exact", "cagra", "ivf_flat")):
        legend_x = left + index * 190
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{colors[backend]}" stroke-width="3"/>',
                f'<text x="{legend_x + 36}" y="{legend_y + 4}" font-family="sans-serif" font-size="13">{backend}</text>',
            ]
        )
    lines.append(
        f'<text x="{left + 600}" y="{legend_y + 4}" font-family="sans-serif" font-size="13">Solid/dashed/dotted: 10k/50k/100k · opaque: RSS · faint: VRAM</text>'
    )
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="exact,cagra,ivf_flat")
    parser.add_argument("--sizes", default="10000,50000,100000")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-features", type=int, default=768)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmark-results/gpu_suite_no_gmst"),
    )
    parser.add_argument("--backend", choices=("exact", "cagra", "ivf_flat"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args.backend, args.n_samples, args.n_features, args.k)
        return

    runs = []
    for n_samples in (int(value) for value in args.sizes.split(",")):
        for backend in args.backends.split(","):
            print(f"{backend}: N={n_samples:,}, D={args.n_features}", flush=True)
            result, samples = _run_worker(
                backend, n_samples, args.n_features, args.k, args.interval
            )
            runs.append((result, samples))
            print(
                f"  total={result['total_s']:.3f}s, "
                f"kNN={result['knn_search_s']:.3f}s, "
                f"recall@{args.k}={result['recall_at_k']:.4f}, "
                f"RSS={result['peak_rss_mib'] / 1024:.2f} GiB, "
                f"VRAM={result['peak_vram_mib'] / 1024:.2f} GiB",
                flush=True,
            )

    results = [result for result, _ in runs]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    memory_path = args.output_prefix.with_name(args.output_prefix.name + "_memory.csv")
    graph_path = args.output_prefix.with_name(args.output_prefix.name + "_memory.svg")
    json_path.write_text(json.dumps(results, indent=2))
    _write_summary(summary_path, results)
    _write_memory(memory_path, runs)
    _write_memory_svg(memory_path, graph_path)
    print(f"JSON: {json_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Memory: {memory_path.resolve()}")
    print(f"Graph: {graph_path.resolve()}")


if __name__ == "__main__":
    main()
