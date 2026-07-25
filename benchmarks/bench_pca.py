#!/usr/bin/env python3
"""Benchmark Rust dimensionality metrics while sampling process RSS."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def _rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def _worker(n_samples: int, n_features: int, suite: str) -> None:
    import numpy as np

    started = time.perf_counter()
    data = np.random.default_rng(0).standard_normal(
        (n_samples, n_features), dtype=np.float64
    )
    generated = time.perf_counter()
    if suite == "full":
        from effdim import compute_dim

        result = dict(compute_dim(data))
    else:
        from effdim._native import compute_spectral

        result = dict(compute_spectral(data))
    finished = time.perf_counter()
    print(
        json.dumps(
            {
                "suite": suite,
                "n_samples": n_samples,
                "n_features": n_features,
                "generation_s": generated - started,
                "compute_s": finished - generated,
                "total_s": finished - started,
                "result": result,
            }
        ),
        flush=True,
    )


def _run_one(
    n_samples: int, n_features: int, interval_s: float, suite: str
) -> tuple[dict, list[tuple[float, float]]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--n-samples",
        str(n_samples),
        "--n-features",
        str(n_features),
        "--suite",
        suite,
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples: list[tuple[float, float]] = []
    while proc.poll() is None:
        rss = _rss_mib(proc.pid)
        if rss is not None:
            samples.append((time.perf_counter() - started, rss))
        time.sleep(interval_s)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(
            f"PCA worker for N={n_samples:,} failed ({proc.returncode}):\n{stderr}"
        )
    payload = json.loads(stdout.strip().splitlines()[-1])
    return payload, samples


def _write_csv(
    path: Path, runs: list[tuple[dict, list[tuple[float, float]]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_samples", "elapsed_s", "rss_mib"])
        for payload, samples in runs:
            for elapsed_s, rss_mib in samples:
                writer.writerow([payload["n_samples"], f"{elapsed_s:.6f}", f"{rss_mib:.3f}"])


def _write_svg(
    path: Path, runs: list[tuple[dict, list[tuple[float, float]]]], suite: str
) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 90, 30, 50, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    max_t = max((t for _, samples in runs for t, _ in samples), default=1.0)
    max_rss = max((rss for _, samples in runs for _, rss in samples), default=1.0)
    max_t = max(max_t, 1.0)
    max_rss = max(max_rss, 1.0)
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")

    def x(value: float) -> float:
        return left + value / max_t * plot_w

    def y(value: float) -> float:
        return top + plot_h - value / max_rss * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="500" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Rust {suite} suite memory usage</text>',
    ]
    for tick in range(6):
        t_value = max_t * tick / 5
        rss_value = max_rss * tick / 5
        tx, ry = x(t_value), y(rss_value)
        lines.extend(
            [
                f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>',
                f'<text x="{tx:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{t_value:.1f}</text>',
                f'<line x1="{left}" y1="{ry:.1f}" x2="{left + plot_w}" y2="{ry:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{ry + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{rss_value:.0f}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">Elapsed time (s)</text>',
            f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.1f})" font-family="sans-serif" font-size="14">RSS (MiB)</text>',
        ]
    )
    for index, (payload, samples) in enumerate(runs):
        color = colors[index % len(colors)]
        points = " ".join(f"{x(t):.1f},{y(rss):.1f}" for t, rss in samples)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend_x = left + index * 170
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 48}" x2="{legend_x + 24}" y2="{height - 48}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{height - 44}" font-family="sans-serif" font-size="13">N={payload["n_samples"]:,}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-features", type=int, default=768)
    parser.add_argument("--sizes", default="10000,50000,100000")
    parser.add_argument("--suite", choices=("spectral", "full"), default="spectral")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--output-prefix", type=Path, default=Path("pca_memory"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args.n_samples, args.n_features, args.suite)
        return

    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    runs: list[tuple[dict, list[tuple[float, float]]]] = []
    for n_samples in sizes:
        print(
            f"Running Rust {args.suite} suite for ({n_samples:,}, {args.n_features})...",
            flush=True,
        )
        payload, samples = _run_one(
            n_samples, args.n_features, args.interval, args.suite
        )
        runs.append((payload, samples))
        peak_mib = max((rss for _, rss in samples), default=0.0)
        print(
            f"  compute {payload['compute_s']:.3f}s, total {payload['total_s']:.3f}s, "
            f"peak RSS {peak_mib:.1f} MiB, PCA-95 {payload['result']['pca_explained_variance_95']}",
            flush=True,
        )

    csv_path = args.output_prefix.with_suffix(".csv")
    svg_path = args.output_prefix.with_suffix(".svg")
    _write_csv(csv_path, runs)
    _write_svg(svg_path, runs, args.suite)
    print(f"CSV: {csv_path.resolve()}")
    print(f"Graph: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
