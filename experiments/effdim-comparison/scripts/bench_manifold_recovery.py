#!/usr/bin/env python3
"""Benchmark EffDim recovery on noisy nonlinear manifolds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench_effdim_stability import _compute_all


DEFAULT_MANIFOLDS = (
    ("linear", 2),
    ("linear", 5),
    ("linear", 10),
    ("linear", 20),
    ("sphere", 2),
    ("sphere", 5),
    ("sphere", 10),
    ("sphere", 20),
    ("torus", 2),
    ("torus", 5),
    ("torus", 10),
    ("torus", 20),
    ("chain", 1),
    ("swiss_roll", 2),
)


def _base_manifold(
    shape: str,
    dimension: int,
    samples: int,
    random: np.random.Generator,
) -> np.ndarray:
    if shape == "linear":
        return random.standard_normal((samples, dimension))
    if shape == "sphere":
        data = random.standard_normal((samples, dimension + 1))
        return data / np.linalg.norm(data, axis=1, keepdims=True)
    if shape == "torus":
        angles = random.uniform(0.0, 2.0 * np.pi, (samples, dimension))
        data = np.empty((samples, 2 * dimension), dtype=np.float64)
        data[:, 0::2] = np.cos(angles)
        data[:, 1::2] = np.sin(angles)
        return data / np.sqrt(dimension)
    if shape == "chain":
        parameter = random.uniform(-1.0, 1.0, samples)
        return np.column_stack(
            [
                parameter,
                parameter**2,
                parameter**3,
                np.sin(np.pi * parameter),
                np.cos(np.pi * parameter),
                np.sin(2.0 * np.pi * parameter),
                np.cos(2.0 * np.pi * parameter),
                np.sin(3.0 * np.pi * parameter),
                np.cos(3.0 * np.pi * parameter),
            ]
        )
    if shape == "swiss_roll":
        angle = random.uniform(1.5 * np.pi, 4.5 * np.pi, samples)
        height = random.uniform(-1.0, 1.0, samples)
        return np.column_stack(
            [angle * np.cos(angle), 4.0 * height, angle * np.sin(angle)]
        )
    raise ValueError(f"unknown manifold: {shape}")


def _embed(
    base: np.ndarray,
    ambient_dimension: int,
    random: np.random.Generator,
) -> np.ndarray:
    basis_seed = random.standard_normal((ambient_dimension, base.shape[1]))
    basis, _ = np.linalg.qr(basis_seed, mode="reduced")
    centered = base - np.mean(base, axis=0, keepdims=True)
    return np.ascontiguousarray(centered @ basis.T, dtype=np.float64)


def _add_noise(
    clean: np.ndarray,
    snr_db: float | None,
    random: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if snr_db is None:
        return clean, 0.0
    signal_variance = float(np.sum(np.var(clean, axis=0, ddof=1)))
    noise_variance = (
        signal_variance
        / clean.shape[1]
        / (10.0 ** (snr_db / 10.0))
    )
    noisy = clean + random.standard_normal(clean.shape) * np.sqrt(noise_variance)
    return np.ascontiguousarray(noisy, dtype=np.float64), noise_variance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--ambient-dimension", type=int, default=256)
    parser.add_argument("--snr-db", nargs="+", default=["none", "30", "20", "10"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snr_levels = [
        None if value.lower() == "none" else float(value) for value in args.snr_db
    ]

    output = {
        "configuration": {
            "manifolds": [
                {"shape": shape, "intrinsic_dimension": dimension}
                for shape, dimension in DEFAULT_MANIFOLDS
            ],
            "samples": args.samples,
            "ambient_dimension": args.ambient_dimension,
            "snr_db": snr_levels,
            "repeats": args.repeats,
            "k": args.k,
            "knn_backend": "exact cuVS brute force",
            "seed": args.seed,
            "noise_definition": "total clean variance / total ambient noise variance",
        },
        "conditions": [],
    }
    for manifold_index, (shape, dimension) in enumerate(DEFAULT_MANIFOLDS):
        for snr_index, snr_db in enumerate(snr_levels):
            condition = {
                "shape": shape,
                "intrinsic_dimension": dimension,
                "snr_db": snr_db,
                "observed_support_dimension": (
                    args.ambient_dimension if snr_db is not None else None
                ),
                "trials": [],
            }
            for repeat in range(args.repeats):
                random = np.random.default_rng(
                    args.seed
                    + manifold_index * 100_000
                    + snr_index * 10_000
                    + repeat
                )
                base = _base_manifold(
                    shape, dimension, args.samples, random
                )
                clean = _embed(base, args.ambient_dimension, random)
                data, noise_variance = _add_noise(clean, snr_db, random)
                print(
                    f"shape={shape}, d={dimension}, snr={snr_db}, "
                    f"repeat={repeat + 1}/{args.repeats}",
                    flush=True,
                )
                result = _compute_all(data, args.k, args.chunk_size)
                result.update(
                    {
                        "repeat": repeat,
                        "noise_variance": noise_variance,
                        "base_embedding_dimension": base.shape[1],
                    }
                )
                condition["trials"].append(result)
            output["conditions"].append(condition)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2))

    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
