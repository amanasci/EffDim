#!/usr/bin/env python3
"""Validate EffDim estimators on linear manifolds with known dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench_effdim_stability import _compute_all


def _orthonormal_embedding(
    intrinsic_dimension: int,
    ambient_dimension: int,
    random: np.random.Generator,
) -> np.ndarray:
    matrix = random.standard_normal((ambient_dimension, intrinsic_dimension))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--ambient-dimension", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if max(args.dimensions) > args.ambient_dimension:
        raise ValueError("intrinsic dimensions must not exceed ambient dimension")

    output = {
        "configuration": {
            "dimensions": args.dimensions,
            "samples": args.samples,
            "ambient_dimension": args.ambient_dimension,
            "repeats": args.repeats,
            "distribution": "isotropic standard Gaussian in a random linear subspace",
            "knn_backend": "exact cuVS brute force",
            "k": args.k,
            "seed": args.seed,
        },
        "trials": [],
    }
    for dimension in args.dimensions:
        for repeat in range(args.repeats):
            random = np.random.default_rng(
                args.seed + dimension * 10_000 + repeat
            )
            basis = _orthonormal_embedding(
                dimension, args.ambient_dimension, random
            )
            latent = random.standard_normal((args.samples, dimension))
            data = np.ascontiguousarray(latent @ basis.T, dtype=np.float64)
            print(
                f"dimension={dimension}, repeat={repeat + 1}/{args.repeats}",
                flush=True,
            )
            result = _compute_all(data, args.k, args.chunk_size)
            result.update(
                {
                    "true_dimension": dimension,
                    "repeat": repeat,
                }
            )
            output["trials"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2))

    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
