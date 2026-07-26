#!/usr/bin/env python3
"""Render known-dimension EffDim validation results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

from render_effdim_stability_report import LABELS, SHORT, _fmt


SPECTRAL = (
    "pca_explained_variance_95",
    "participation_ratio",
    "shannon_entropy",
    "renyi_eff_dimensionality_alpha_2",
    "renyi_eff_dimensionality_alpha_3",
    "renyi_eff_dimensionality_alpha_4",
    "renyi_eff_dimensionality_alpha_5",
)
GEOMETRY = (
    "mle_dimensionality",
    "two_nn_dimensionality",
    "danco_dimensionality",
    "mind_mli_dimensionality",
    "mind_mlk_dimensionality",
    "ess_dimensionality",
    "tle_dimensionality",
)
COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4b5563",
)


def _aggregates(data: dict) -> dict:
    methods = list(data["trials"][0]["values"])
    dimensions = data["configuration"]["dimensions"]
    output = {}
    for method in methods:
        output[method] = {}
        for dimension in dimensions:
            values = np.asarray(
                [
                    trial["values"][method]
                    for trial in data["trials"]
                    if trial["true_dimension"] == dimension
                ],
                dtype=float,
            )
            output[method][dimension] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
    return output


def _write_chart(
    aggregates: dict,
    dimensions: list[int],
    methods: tuple[str, ...],
    title: str,
    path: Path,
) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 80, 25, 65, 145
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [
        aggregates[method][dimension]["mean"]
        for method in methods
        for dimension in dimensions
        if aggregates[method][dimension]["mean"] > 0.0
    ] + dimensions
    log_min = math.floor(math.log10(min(all_values))) - 0.1
    log_max = math.ceil(math.log10(max(all_values))) + 0.1

    def x(value: float) -> float:
        return left + (
            (math.log10(value) - math.log10(min(dimensions)))
            / (math.log10(max(dimensions)) - math.log10(min(dimensions)))
            * plot_w
        )

    def y(value: float) -> float:
        return top + (log_max - math.log10(max(value, 10.0**log_min))) / (
            log_max - log_min
        ) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="403" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="22">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="13">Mean of five trials; both axes logarithmic; dashed line is perfect recovery</text>',
    ]
    for dimension in dimensions:
        lines.extend(
            [
                f'<line x1="{x(dimension):.1f}" y1="{top}" x2="{x(dimension):.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>',
                f'<text x="{x(dimension):.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="12">{dimension}</text>',
            ]
        )
    for exponent in range(math.ceil(log_min), math.floor(log_max) + 1):
        tick = 10.0**exponent
        lines.extend(
            [
                f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + plot_w}" y2="{y(tick):.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:g}</text>',
            ]
        )
    perfect = " ".join(f"{x(d):.1f},{y(d):.1f}" for d in dimensions)
    lines.append(
        f'<polyline points="{perfect}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="7,5"/>'
    )
    for index, method in enumerate(methods):
        color = COLORS[index]
        points = " ".join(
            f"{x(dimension):.1f},{y(aggregates[method][dimension]['mean']):.1f}"
            for dimension in dimensions
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for dimension in dimensions:
            lines.append(
                f'<circle cx="{x(dimension):.1f}" cy="{y(aggregates[method][dimension]["mean"]):.1f}" r="4" fill="{color}"/>'
            )
        legend_x = left + (index % 4) * 220
        legend_y = height - 45 + (index // 4) * 20
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 20}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 26}" y="{legend_y}" font-family="sans-serif" font-size="12">{html.escape(SHORT[method])}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 45}" text-anchor="middle" font-family="sans-serif" font-size="14">True intrinsic dimension</text>',
            f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Estimated dimension</text>',
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
    dimensions = data["configuration"]["dimensions"]
    aggregates = _aggregates(data)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, by_dimension in aggregates.items():
        for dimension, stats in by_dimension.items():
            rows.append(
                {
                    "method": method,
                    "true_dimension": dimension,
                    **stats,
                    "relative_error_percent": abs(stats["mean"] / dimension - 1.0)
                    * 100.0,
                }
            )
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    _write_chart(
        aggregates,
        dimensions,
        SPECTRAL,
        "Known-dimension validation: spectral estimators",
        prefix.with_name(prefix.name + "_spectral.svg"),
    )
    _write_chart(
        aggregates,
        dimensions,
        GEOMETRY,
        "Known-dimension validation: local geometry estimators",
        prefix.with_name(prefix.name + "_geometry.svg"),
    )

    methods = list(aggregates)
    lines = [
        "# EffDim validation on known-dimensional manifolds",
        "",
        "Five independent isotropic Gaussian datasets were generated at each true",
        "dimension (5, 10, 20, 50, and 100), with 10,000 samples embedded into a",
        "random 256-dimensional linear subspace. Exact GPU neighbours used `k=10`.",
        "",
        "## Main findings",
        "",
        "1. **Participation ratio, Shannon effective rank, and Rényi dimensions recover",
        "   the known linear rank almost exactly.** Even at dimension 100 their means",
        "   lie between 97.6 and 99.5.",
        "2. **PCA-95 behaves according to its definition:** 5, 10, 19, 48, and 94.2.",
        "   It is a 95%-variance rank rather than an unbiased intrinsic-dimension estimate.",
        "3. **MLE, TLE, Two-NN, and MiND-MLk work through dimension 20 but underestimate",
        "   high dimensions severely.** At true dimension 100 they return about 51–55,",
        "   showing finite-sample and `k=10` limitations.",
        "4. **DANCo saturates near 5–7, MiND-MLi remains far too low, and the simplified",
        "   ESS statistic decreases toward 0.2.** These implementations should not be",
        "   selected as general dimension estimators from stability alone.",
        "5. **Geometric-mean ED is numerically invalid on rank-deficient embeddings.**",
        "   Null covariance directions drive enormous values instead of the known rank.",
        "",
        "## Spectral estimators",
        "",
        f"![]({prefix.name}_spectral.svg)",
        "",
        "## Local geometry estimators",
        "",
        f"![]({prefix.name}_geometry.svg)",
        "",
        "## Mean estimates",
        "",
        "| Method | d=5 | d=10 | d=20 | d=50 | d=100 |",
        "|:---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        estimates = [aggregates[method][dimension]["mean"] for dimension in dimensions]
        lines.append(
            f"| {LABELS[method]} | "
            + " | ".join(_fmt(value) for value in estimates)
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
