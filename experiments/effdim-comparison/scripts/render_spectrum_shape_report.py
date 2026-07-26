#!/usr/bin/env python3
"""Render spectral-decay and ambient-noise benchmark results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

from render_effdim_stability_report import LABELS, _fmt


METRICS = (
    "pca_explained_variance_95",
    "shannon_entropy",
    "participation_ratio",
    "renyi_eff_dimensionality_alpha_5",
)
METRIC_LABELS = {
    "pca_explained_variance_95": "PCA-95",
    "shannon_entropy": "Shannon",
    "participation_ratio": "Participation ratio",
    "renyi_eff_dimensionality_alpha_5": "Rényi α=5",
}
PROFILE_LABELS = {
    "flat": "Flat",
    "power_0.5": "Power 0.5",
    "power_1.0": "Power 1.0",
    "power_2.0": "Power 2.0",
    "exp_10": "Exponential 10",
    "exp_25": "Exponential 25",
}
COLORS = ("#2563eb", "#16a34a", "#ea580c", "#9333ea")


def _condition(data: dict, profile: str, snr_db: float | None) -> dict:
    return next(
        condition
        for condition in data["conditions"]
        if condition["profile"] == profile and condition["snr_db"] == snr_db
    )


def _write_profile_chart(data: dict, path: Path) -> None:
    profiles = data["configuration"]["profiles"]
    width, height = 1000, 600
    left, right, top, bottom = 80, 25, 65, 120
    plot_w, plot_h = width - left - right, height - top - bottom

    def y(value: float) -> float:
        return top + (math.log10(256.0) - math.log10(value)) / math.log10(
            256.0
        ) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="390" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="22">Effective dimensions under eigenvalue decay</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="13">Exact population values; rank 100 in ambient dimension 256; no added noise; logarithmic y-axis</text>',
    ]
    for tick in (1, 2, 5, 10, 20, 50, 100, 200):
        lines.extend(
            [
                f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + plot_w}" y2="{y(tick):.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{y(100):.1f}" x2="{left + plot_w}" y2="{y(100):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="7,5"/>',
            f'<text x="{left + plot_w - 4}" y="{y(100) - 6:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">actual latent dimension = covariance rank = 100</text>',
        ]
    )
    group_w = plot_w / len(profiles)
    bar_w = group_w * 0.16
    for profile_index, profile in enumerate(profiles):
        condition = _condition(data, profile, None)
        center = left + (profile_index + 0.5) * group_w
        for metric_index, metric in enumerate(METRICS):
            value = condition["population_metrics"][metric]
            bar_x = center + (metric_index - 1.5) * bar_w
            lines.append(
                f'<rect x="{bar_x - bar_w * 0.42:.1f}" y="{y(value):.1f}" width="{bar_w * 0.84:.1f}" height="{y(1.0) - y(value):.1f}" fill="{COLORS[metric_index]}"/>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 19}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(PROFILE_LABELS[profile])}</text>'
        )
    for index, metric in enumerate(METRICS):
        legend_x = left + index * 215
        lines.extend(
            [
                f'<rect x="{legend_x}" y="{height - 52}" width="16" height="10" fill="{COLORS[index]}"/>',
                f'<text x="{legend_x + 23}" y="{height - 43}" font-family="sans-serif" font-size="12">{html.escape(METRIC_LABELS[metric])}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Effective dimension</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines))


def _write_noise_chart(data: dict, path: Path) -> None:
    profile = "power_2.0"
    levels = (None, 30.0, 20.0, 10.0)
    level_labels = ("No noise", "30 dB", "20 dB", "10 dB")
    width, height = 1000, 600
    left, right, top, bottom = 80, 25, 65, 120
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(index: int) -> float:
        return left + index / (len(levels) - 1) * plot_w

    def y(value: float) -> float:
        return top + (math.log10(300.0) - math.log10(value)) / math.log10(
            300.0
        ) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="390" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="22">Weak ambient noise inflates PCA-95 selectively</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="13">Power-law exponent 2; exact population values; lower SNR means more noise; logarithmic y-axis</text>',
    ]
    for tick in (1, 2, 5, 10, 20, 50, 100, 200):
        lines.extend(
            [
                f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + plot_w}" y2="{y(tick):.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}</text>',
            ]
        )
    latent_points = " ".join(f"{x(index):.1f},{y(100):.1f}" for index in range(4))
    rank_values = (100, 256, 256, 256)
    rank_points = " ".join(
        f"{x(index):.1f},{y(value):.1f}"
        for index, value in enumerate(rank_values)
    )
    lines.extend(
        [
            f'<polyline points="{latent_points}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="7,5"/>',
            f'<polyline points="{rank_points}" fill="none" stroke="#6b7280" stroke-width="2" stroke-dasharray="2,4"/>',
        ]
    )
    for index, label in enumerate(level_labels):
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>'
        )
    for metric_index, metric in enumerate(METRICS):
        values = [
            _condition(data, profile, level)["population_metrics"][metric]
            for level in levels
        ]
        points = " ".join(
            f"{x(index):.1f},{y(value):.1f}"
            for index, value in enumerate(values)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[metric_index]}" stroke-width="3"/>'
        )
        for index, value in enumerate(values):
            lines.append(
                f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="5" fill="{COLORS[metric_index]}"/>'
            )
        legend_x = left + metric_index * 215
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 47}" x2="{legend_x + 20}" y2="{height - 47}" stroke="{COLORS[metric_index]}" stroke-width="3"/>',
                f'<text x="{legend_x + 27}" y="{height - 43}" font-family="sans-serif" font-size="12">{html.escape(METRIC_LABELS[metric])}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{height - 22}" x2="{left + 20}" y2="{height - 22}" stroke="#111827" stroke-width="2" stroke-dasharray="7,5"/>',
            f'<text x="{left + 27}" y="{height - 18}" font-family="sans-serif" font-size="12">Actual latent dimension (100)</text>',
            f'<line x1="{left + 250}" y1="{height - 22}" x2="{left + 270}" y2="{height - 22}" stroke="#6b7280" stroke-width="2" stroke-dasharray="2,4"/>',
            f'<text x="{left + 277}" y="{height - 18}" font-family="sans-serif" font-size="12">Observed covariance rank (100 or 256)</text>',
        ]
    )
    lines.extend(
        [
            f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Effective dimension</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    accuracy = {}
    methods = tuple(data["conditions"][0]["population_metrics"])
    for method in methods:
        errors = []
        for condition in data["conditions"]:
            target = condition["population_metrics"][method]
            values = [trial["values"][method] for trial in condition["trials"]]
            mean = float(np.mean(values))
            error = abs(mean / target - 1.0) * 100.0
            errors.append(error)
            rows.append(
                {
                    "profile": condition["profile"],
                    "snr_db": condition["snr_db"],
                    "method": method,
                    "population_value": target,
                    "sample_mean": mean,
                    "sample_std": float(np.std(values, ddof=1)),
                    "relative_error_percent": error,
                }
            )
        accuracy[method] = (float(np.median(errors)), float(np.max(errors)))
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    _write_profile_chart(data, prefix.with_name(prefix.name + "_profiles.svg"))
    _write_noise_chart(data, prefix.with_name(prefix.name + "_noise.svg"))

    lines = [
        "# EffDim under eigenvalue decay and ambient noise",
        "",
        "Synthetic covariance spectra used rank 100 in 256 ambient dimensions,",
        "10,000 observations, and five trials per condition. Population targets are",
        "known exactly and compared with the sample covariance implementation.",
        "",
        "## Main findings",
        "",
        "1. **The actual latent dimension is 100 in every condition.** Without noise,",
        "   covariance rank is also 100. Any nonzero isotropic ambient noise makes the",
        "   population covariance full-rank at 256.",
        "2. **The earlier agreement was a special property of the flat spectrum.**",
        "   With power-law exponent 1 and no noise, PCA-95 is 78 while Shannon is",
        "   39.68, participation ratio is 16.46, and Rényi α=5 is 7.76.",
        "3. **Steeper decay increases the disagreement dramatically.** At exponent 2,",
        "   the same metrics are 11, 4.81, 2.47, and 1.85 even though algebraic rank",
        "   remains 100.",
        "4. **A weak, broad noise floor can make PCA-95 large without materially changing",
        "   dominant-direction metrics.** For exponent 2, moving from no noise to",
        "   10 dB raises PCA-95 from 11 to 116, but participation ratio only from",
        "   2.47 to 2.98 and Rényi α=5 from 1.85 to 2.08.",
        "5. **This reproduces the qualitative real-data pattern:** a few dominant",
        "   directions plus many weak directions can yield high PCA-95 and low",
        "   Shannon/Rényi dimensions simultaneously.",
        "6. **Sample estimates closely track exact population definitions.** The largest",
        "   discrepancy is 6.9% for PCA-95 under the noisiest condition; most",
        "   continuous effective-rank errors are around 0–2.5%.",
        "",
        "## Spectrum-shape effect",
        "",
        f"![]({prefix.name}_profiles.svg)",
        "",
        "## Noise-floor effect",
        "",
        f"![]({prefix.name}_noise.svg)",
        "",
        "## Exact population dimensions without noise",
        "",
        "| Spectrum | Actual latent d | Covariance rank | PCA-95 | Shannon | Participation ratio | Rényi α=5 |",
        "|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in data["configuration"]["profiles"]:
        values = _condition(data, profile, None)["population_metrics"]
        lines.append(
            f"| {PROFILE_LABELS[profile]} | 100 | 100 | "
            + " | ".join(_fmt(values[metric]) for metric in METRICS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Actual dimension versus effective dimensions under noise",
            "",
            "| SNR | Actual latent d | Covariance rank | PCA-95 | Shannon | Participation ratio | Rényi α=5 |",
            "|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for snr_db, snr_label in ((None, "No noise"), (30.0, "30 dB"), (20.0, "20 dB"), (10.0, "10 dB")):
        values = _condition(data, "power_2.0", snr_db)["population_metrics"]
        covariance_rank = 100 if snr_db is None else 256
        lines.append(
            f"| {snr_label} | 100 | {covariance_rank} | "
            + " | ".join(_fmt(values[metric]) for metric in METRICS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Sample-to-population accuracy across all 24 conditions",
            "",
            "| Method | Median relative error | Maximum relative error |",
            "|:---:|---:|---:|",
        ]
    )
    for method in methods:
        median, maximum = accuracy[method]
        lines.append(
            f"| {LABELS[method]} | {median:.2f}% | {maximum:.2f}% |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
