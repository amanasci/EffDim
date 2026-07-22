#!/usr/bin/env python3
"""Time full ``compute_dim`` — single shape or N sweep at fixed D.

``compute_dim`` runs on CPU (exact k-NN is O(n²d)). For notebook-style runs,
use D=768 (BGE-base) and sweep N.

    .venv/bin/python benchmarks/bench_compute_dim.py --sweep-n --compare /tmp/effdim-py/.venv/bin/python
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SWEEP_N = (500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000)


def _backend_label() -> str:
    try:
        import effdim._native  # noqa: F401

        return "rust (_native)"
    except ImportError:
        return "python (main)"


def run_bench(n_samples: int, n_features: int, repeats: int, warmup: int) -> dict:
    from effdim import compute_dim

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_samples, n_features), dtype=np.float64)

    for _ in range(warmup):
        compute_dim(data)

    times: list[float] = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = compute_dim(data)
        times.append(time.perf_counter() - t0)

    return {
        "backend": _backend_label(),
        "n_samples": n_samples,
        "n_features": n_features,
        "repeats": repeats,
        "warmup": warmup,
        "times": times,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "mean_s": statistics.mean(times),
        "n_keys": len(result) if result is not None else 0,
    }


def print_result(stats: dict) -> None:
    print(f"backend:  {stats['backend']}")
    print(f"shape:    ({stats['n_samples']}, {stats['n_features']})")
    print(f"warmup:   {stats['warmup']}  repeats: {stats['repeats']}")
    print(f"keys:     {stats['n_keys']}")
    print(f"median:   {stats['median_s']:.4f} s")
    print(f"min:      {stats['min_s']:.4f} s")
    print(f"mean:     {stats['mean_s']:.4f} s")
    print(f"all:      {[f'{t:.4f}' for t in stats['times']]}")


def _run_other_median(other_python: str, script: str, n_samples: int, n_features: int, repeats: int, warmup: int) -> tuple[float, str]:
    args = [
        "--n-samples",
        str(n_samples),
        "--n-features",
        str(n_features),
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--machine",
    ]
    proc = subprocess.run(
        [other_python, script, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("MEDIAN "):
            parts = line.split()
            median = float(parts[1])
            backend = " ".join(parts[3:]) if len(parts) > 3 else "other"
            return median, backend
    raise RuntimeError(f"could not parse median from other interpreter:\n{proc.stdout}")


def run_sweep(
    n_values: list[int],
    n_features: int,
    repeats: int,
    warmup: int,
    other_python: str | None,
    max_median_s: float,
) -> list[dict]:
    script = str(Path(__file__).resolve())
    local_backend = _backend_label()
    rows: list[dict] = []

    print(f"sweep: N in {n_values}, D={n_features}, repeats={repeats}, max_median={max_median_s}s")
    print(f"this:   {sys.executable} ({local_backend})")
    if other_python:
        print(f"other:  {other_python}")
    print()

    for n in n_values:
        print(f"--- N={n:,} ---")
        local = run_bench(n, n_features, repeats, warmup)
        row: dict = {
            "n": n,
            "d": n_features,
            "rust_median_s": local["median_s"],
            "rust_backend": local["backend"],
        }
        print(f"  {local['backend']:18s}  median {local['median_s']:.3f}s  min {local['min_s']:.3f}s")

        if other_python:
            other_median, other_backend = _run_other_median(
                other_python, script, n, n_features, repeats, warmup
            )
            row["python_median_s"] = other_median
            row["python_backend"] = other_backend
            ratio = other_median / local["median_s"] if local["median_s"] > 0 else float("inf")
            row["speedup"] = ratio
            print(f"  {other_backend:18s}  median {other_median:.3f}s  ({ratio:.2f}x vs rust)")
        print()

        rows.append(row)

        stop_median = max(local["median_s"], row.get("python_median_s", 0.0))
        if stop_median > max_median_s:
            print(f"Stopping sweep: median {stop_median:.1f}s > {max_median_s}s cap")
            break

    return rows


def print_sweep_table(rows: list[dict]) -> None:
    if not rows:
        return
    has_compare = "python_median_s" in rows[0]
    print("=" * 72)
    print("SUMMARY")
    if has_compare:
        print(f"{'N':>8}  {'Rust (s)':>10}  {'Python (s)':>12}  {'Speedup':>8}")
        print("-" * 72)
        for r in rows:
            print(
                f"{r['n']:8,d}  {r['rust_median_s']:10.3f}  {r['python_median_s']:12.3f}  {r['speedup']:7.2f}x"
            )
    else:
        print(f"{'N':>8}  {'median (s)':>12}")
        print("-" * 72)
        for r in rows:
            print(f"{r['n']:8,d}  {r['rust_median_s']:12.3f}")
    print("=" * 72)


def run_compare(other_python: str, n_samples: int, n_features: int, repeats: int, warmup: int) -> None:
    script = str(Path(__file__).resolve())
    args = [
        "--n-samples",
        str(n_samples),
        "--n-features",
        str(n_features),
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--machine",
    ]

    print("=== this interpreter ===")
    print(f"python: {sys.executable}")
    local = run_bench(n_samples, n_features, repeats, warmup)
    print_result(local)
    print()

    print("=== other interpreter ===")
    print(f"python: {other_python}")
    proc = subprocess.run(
        [other_python, script, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr)
    print()

    # Parse median from machine line: MEDIAN <seconds> BACKEND <label>
    other_median = None
    other_backend = "other"
    for line in proc.stdout.splitlines():
        if line.startswith("MEDIAN "):
            parts = line.split()
            other_median = float(parts[1])
            other_backend = " ".join(parts[3:]) if len(parts) > 3 else "other"
            break

    if other_median is None or other_median <= 0:
        print("Could not parse other median; skip speedup.")
        return

    faster = local if local["median_s"] <= other_median else None
    if faster is local:
        ratio = other_median / local["median_s"]
        print(
            f"speedup: {local['backend']} is {ratio:.2f}x faster than {other_backend} "
            f"({local['median_s']:.4f}s vs {other_median:.4f}s median)"
        )
    else:
        ratio = local["median_s"] / other_median
        print(
            f"speedup: {other_backend} is {ratio:.2f}x faster than {local['backend']} "
            f"({other_median:.4f}s vs {local['median_s']:.4f}s median)"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument(
        "--n-features",
        type=int,
        default=768,
        help="Feature dimension (768 = BGE-base embeddings; exact k-NN is O(n^2 d))",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--sweep-n",
        nargs="?",
        const=",".join(str(n) for n in DEFAULT_SWEEP_N),
        metavar="N1,N2,...",
        help=f"Sweep sample counts at fixed D (default: {','.join(str(n) for n in DEFAULT_SWEEP_N)})",
    )
    p.add_argument(
        "--max-median-s",
        type=float,
        default=180.0,
        help="Stop sweep when either backend median exceeds this (seconds)",
    )
    p.add_argument(
        "--compare",
        metavar="PYTHON",
        help="Path to the other install's python (e.g. main worktree .venv/bin/python)",
    )
    p.add_argument(
        "--machine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = p.parse_args()

    if args.sweep_n:
        n_values = [int(x.strip()) for x in args.sweep_n.split(",") if x.strip()]
        repeats = min(args.repeats, 2) if len(n_values) > 1 else args.repeats
        rows = run_sweep(
            n_values,
            args.n_features,
            repeats,
            args.warmup,
            args.compare,
            args.max_median_s,
        )
        print_sweep_table(rows)
        return

    if args.compare:
        run_compare(args.compare, args.n_samples, args.n_features, args.repeats, args.warmup)
        return

    stats = run_bench(args.n_samples, args.n_features, args.repeats, args.warmup)
    if args.machine:
        print(f"MEDIAN {stats['median_s']:.6f} BACKEND {stats['backend']}")
    print_result(stats)


if __name__ == "__main__":
    main()
