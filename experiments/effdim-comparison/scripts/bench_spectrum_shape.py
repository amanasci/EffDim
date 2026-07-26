#!/usr/bin/env python3
"""Validate spectral EffDim metrics under eigenvalue decay and ambient noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench_effdim_stability import _spectral_values


METHODS = (
    "pca_explained_variance_95",
    "participation_ratio",
    "shannon_entropy",
    "renyi_eff_dimensionality_alpha_2",
    "renyi_eff_dimensionality_alpha_3",
    "renyi_eff_dimensionality_alpha_4",
    "renyi_eff_dimensionality_alpha_5",
)


def _signal_spectrum(profile: str, rank: int) -> np.ndarray:
    index = np.arange(1, rank + 1, dtype=np.float64)
    if profile == "flat":
        return np.ones(rank, dtype=np.float64)
    if profile.startswith("power_"):
        exponent = float(profile.removeprefix("power_"))
        return index ** (-exponent)
    if profile.startswith("exp_"):
        scale = float(profile.removeprefix("exp_"))
        return np.exp(-(index - 1.0) / scale)
    raise ValueError(f"unknown profile: {profile}")


def _population_spectrum(
    signal: np.ndarray,
    ambient_dimension: int,
    snr_db: float | None,
) -> tuple[np.ndarray, float]:
    noise_variance = (
        0.0
        if snr_db is None
        else float(np.sum(signal))
        / ambient_dimension
        / (10.0 ** (snr_db / 10.0))
    )
    spectrum = np.full(ambient_dimension, noise_variance, dtype=np.float64)
    spectrum[: len(signal)] += signal
    return spectrum, noise_variance


def _theoretical_metrics(eigenvalues: np.ndarray) -> dict[str, float]:
    eigenvalues = np.sort(eigenvalues)[::-1]
    probabilities = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(probabilities)
    positive = probabilities[probabilities > 0.0]
    output = {
        "pca_explained_variance_95": float(
            np.searchsorted(cumulative, 0.95, side="left") + 1
        ),
        "participation_ratio": float(1.0 / np.sum(probabilities**2)),
        "shannon_entropy": float(np.exp(-np.sum(positive * np.log(positive)))),
    }
    for alpha in (2, 3, 4, 5):
        output[f"renyi_eff_dimensionality_alpha_{alpha}"] = float(
            np.sum(probabilities**alpha) ** (1.0 / (1.0 - alpha))
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["flat", "power_0.5", "power_1.0", "power_2.0", "exp_10", "exp_25"],
    )
    parser.add_argument("--snr-db", nargs="+", default=["none", "30", "20", "10"])
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--ambient-dimension", type=int, default=256)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snr_levels = [
        None if value.lower() == "none" else float(value) for value in args.snr_db
    ]

    output = {
        "configuration": {
            "profiles": args.profiles,
            "snr_db": snr_levels,
            "rank": args.rank,
            "ambient_dimension": args.ambient_dimension,
            "samples": args.samples,
            "repeats": args.repeats,
            "seed": args.seed,
            "noise_definition": "total signal variance / total ambient noise variance",
        },
        "conditions": [],
    }
    for profile_index, profile in enumerate(args.profiles):
        signal = _signal_spectrum(profile, args.rank)
        for snr_index, snr_db in enumerate(snr_levels):
            population, noise_variance = _population_spectrum(
                signal, args.ambient_dimension, snr_db
            )
            condition = {
                "profile": profile,
                "snr_db": snr_db,
                "signal_eigenvalues": signal.tolist(),
                "noise_variance": noise_variance,
                "population_metrics": _theoretical_metrics(population),
                "trials": [],
            }
            for repeat in range(args.repeats):
                random = np.random.default_rng(
                    args.seed
                    + profile_index * 100_000
                    + snr_index * 10_000
                    + repeat
                )
                data = np.zeros(
                    (args.samples, args.ambient_dimension), dtype=np.float64
                )
                data[:, : args.rank] = random.standard_normal(
                    (args.samples, args.rank)
                ) * np.sqrt(signal)
                if noise_variance > 0.0:
                    data += random.standard_normal(data.shape) * np.sqrt(
                        noise_variance
                    )
                print(
                    f"profile={profile}, snr={snr_db}, "
                    f"repeat={repeat + 1}/{args.repeats}",
                    flush=True,
                )
                values, elapsed = _spectral_values(data, args.chunk_size)
                condition["trials"].append(
                    {
                        "repeat": repeat,
                        "values": {
                            method: values[method]
                            for method in METHODS
                        },
                        "time_s": elapsed,
                    }
                )
            output["conditions"].append(condition)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2))

    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
