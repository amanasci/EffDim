#!/usr/bin/env python3
"""Create a deterministic chunked NumPy matrix for large PCA benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-features", type=int, default=768)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.float64,
        shape=(args.n_samples, args.n_features),
    )
    random = np.random.default_rng(args.seed)
    for start in range(0, args.n_samples, args.chunk_size):
        stop = min(start + args.chunk_size, args.n_samples)
        matrix[start:stop] = random.standard_normal(
            (stop - start, args.n_features), dtype=np.float64
        )
    matrix.flush()
    print(
        f"{args.output}: shape={matrix.shape}, dtype={matrix.dtype}, "
        f"size={matrix.nbytes / 1024**3:.2f} GiB"
    )


if __name__ == "__main__":
    main()
