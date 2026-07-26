#!/usr/bin/env python3
"""Render noisy nonlinear-manifold EffDim recovery results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

from render_effdim_stability_report import DATASET_LABELS, LABELS, _fmt


SELECTED = (
    "pca_explained_variance_95",
    "shannon_entropy",
    "participation_ratio",
    "renyi_eff_dimensionality_alpha_5",
    "mle_dimensionality",
    "two_nn_dimensionality",
    "danco_dimensionality",
    "mind_mlk_dimensionality",
    "isomap_dimensionality",
)
SHORT = {
    "pca_explained_variance_95": "PCA-95",
    "shannon_entropy": "Shannon",
    "participation_ratio": "PR",
    "renyi_eff_dimensionality_alpha_5": "Rényi-5",
    "mle_dimensionality": "MLE",
    "two_nn_dimensionality": "Two-NN",
    "danco_dimensionality": "DANCo",
    "mind_mlk_dimensionality": "MiND-k",
    "isomap_dimensionality": "Isomap",
}
SHAPE_LABELS = {
    "linear": "Linear",
    "sphere": "Sphere",
    "torus": "Torus",
    "chain": "Chain",
    "swiss_roll": "Swiss roll",
}
COLORS = (
    "#2563eb",
    "#16a34a",
    "#ea580c",
    "#9333ea",
    "#0891b2",
    "#dc2626",
    "#65a30d",
    "#4b5563",
    "#be123c",
)


def _aggregates(data: dict, isomap_data: dict) -> dict:
    output = {}
    for condition in data["conditions"]:
        key = (
            condition["shape"],
            condition["intrinsic_dimension"],
            condition["snr_db"],
        )
        output[key] = {}
        methods = condition["trials"][0]["values"]
        for method in methods:
            values = np.asarray(
                [trial["values"][method] for trial in condition["trials"]]
            )
            output[key][method] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
    for condition in isomap_data["conditions"]:
        key = (
            condition["shape"],
            condition["intrinsic_dimension"],
            condition["snr_db"],
        )
        values = np.asarray([trial["estimate"] for trial in condition["trials"]])
        output[key]["isomap_dimensionality"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    return output


def _ordered_manifolds(data: dict) -> list[tuple[str, int]]:
    return [
        (item["shape"], item["intrinsic_dimension"])
        for item in data["configuration"]["manifolds"]
    ]


def _error_color(error: float) -> str:
    if error <= 10.0:
        return "#bbf7d0"
    if error <= 25.0:
        return "#fef08a"
    if error <= 50.0:
        return "#fed7aa"
    return "#fecaca"


def _write_heatmap(data: dict, aggregates: dict, path: Path) -> None:
    manifolds = _ordered_manifolds(data)
    width, height = 1500, 940
    left, top, right, bottom = 230, 95, 20, 30
    cell_w = (width - left - right) / len(SELECTED)
    cell_h = (height - top - bottom) / len(manifolds)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="407" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="22">Noiseless recovery of latent manifold dimension</text>',
        f'<text x="{width / 2}" y="51" text-anchor="middle" font-family="sans-serif" font-size="13">Cell text is mean estimated dimension; color is absolute relative error: green ≤10%, yellow ≤25%, orange ≤50%, red &gt;50%</text>',
    ]
    for column, method in enumerate(SELECTED):
        center = left + (column + 0.5) * cell_w
        lines.append(
            f'<text x="{center:.1f}" y="{top - 13}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">{html.escape(SHORT[method])}</text>'
        )
    for row, (shape, dimension) in enumerate(manifolds):
        y = top + row * cell_h
        lines.append(
            f'<text x="{left - 10}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{html.escape(SHAPE_LABELS[shape])}, true d={dimension}</text>'
        )
        for column, method in enumerate(SELECTED):
            estimate = aggregates[(shape, dimension, None)][method]["mean"]
            error = abs(estimate / dimension - 1.0) * 100.0
            x = left + column * cell_w
            lines.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{_error_color(error)}" stroke="#ffffff"/>',
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(_fmt(estimate))}</text>',
                ]
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def _median_errors(
    data: dict, aggregates: dict, method: str, snr_db: float | None
) -> float:
    errors = []
    for shape, dimension in _ordered_manifolds(data):
        estimate = aggregates[(shape, dimension, snr_db)][method]["mean"]
        errors.append(abs(estimate / dimension - 1.0) * 100.0)
    return float(np.median(errors))


def _write_noise_chart(data: dict, aggregates: dict, path: Path) -> None:
    levels = (None, 30.0, 20.0, 10.0)
    labels = ("No noise", "30 dB", "20 dB", "10 dB")
    width, height = 1000, 630
    left, right, top, bottom = 80, 25, 65, 145
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(index: int) -> float:
        return left + index / (len(levels) - 1) * plot_w

    def y(value: float) -> float:
        value = max(1.0, value)
        return top + (math.log10(3000.0) - math.log10(value)) / math.log10(
            3000.0
        ) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="410" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="22">Latent-dimension recovery degrades with ambient noise</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="13">Median absolute relative error across 14 manifolds; lower is better; logarithmic y-axis</text>',
    ]
    for tick in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        lines.extend(
            [
                f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + plot_w}" y2="{y(tick):.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}%</text>',
            ]
        )
    for index, label in enumerate(labels):
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>'
        )
    for method_index, method in enumerate(SELECTED):
        values = [
            _median_errors(data, aggregates, method, level) for level in levels
        ]
        points = " ".join(
            f"{x(index):.1f},{y(value):.1f}"
            for index, value in enumerate(values)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[method_index]}" stroke-width="2.5"/>'
        )
        for index, value in enumerate(values):
            lines.append(
                f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="4" fill="{COLORS[method_index]}"/>'
            )
        legend_x = left + (method_index % 4) * 215
        legend_y = height - 48 + (method_index // 4) * 21
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 20}" y2="{legend_y - 4}" stroke="{COLORS[method_index]}" stroke-width="3"/>',
                f'<text x="{legend_x + 27}" y="{legend_y}" font-family="sans-serif" font-size="12">{html.escape(SHORT[method])}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Median absolute relative error</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines))


def _write_real_chart(
    stability: dict,
    isomap: dict,
    isomap_k15: dict,
    isomap_k20: dict,
    path: Path,
) -> None:
    methods = (
        "pca_explained_variance_95",
        "shannon_entropy",
        "participation_ratio",
        "renyi_eff_dimensionality_alpha_5",
        "mle_dimensionality",
        "mind_mlk_dimensionality",
        "isomap_k10",
        "isomap_k15",
        "isomap_k20",
    )
    labels = (
        "PCA-95",
        "Shannon",
        "PR",
        "Rényi-5",
        "MLE",
        "MiND-k",
        "Isomap k=10",
        "Isomap k=15",
        "Isomap k=20",
    )
    datasets = stability["datasets"]
    width, height = 1000, 600
    left, right, top, bottom = 80, 25, 65, 120
    plot_w, plot_h = width - left - right, height - top - bottom

    def y(value: float) -> float:
        return top + (math.log10(200.0) - math.log10(value)) / math.log10(
            200.0 / 1.0
        ) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="390" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="22">Landmark Isomap on real UniverseTBD embeddings</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="13">Whole-dataset estimates; Isomap is the mean of five landmark selections; logarithmic y-axis</text>',
    ]
    for tick in (1, 2, 5, 10, 20, 50, 100, 200):
        lines.extend(
            [
                f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + plot_w}" y2="{y(tick):.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}</text>',
            ]
        )
    group_w = plot_w / len(datasets)
    bar_w = group_w * 0.10
    for dataset_index, dataset in enumerate(datasets):
        center = left + (dataset_index + 0.5) * group_w
        isomap_dataset = next(
            item for item in isomap["datasets"] if item["dataset"] == dataset["dataset"]
        )
        isomap_mean = float(
            np.mean([trial["estimate"] for trial in isomap_dataset["trials"]])
        )
        isomap_k15_dataset = next(
            item
            for item in isomap_k15["datasets"]
            if item["dataset"] == dataset["dataset"]
        )
        isomap_k15_mean = float(
            np.mean(
                [trial["estimate"] for trial in isomap_k15_dataset["trials"]]
            )
        )
        isomap_k20_dataset = next(
            item
            for item in isomap_k20["datasets"]
            if item["dataset"] == dataset["dataset"]
        )
        isomap_k20_mean = float(
            np.mean(
                [trial["estimate"] for trial in isomap_k20_dataset["trials"]]
            )
        )
        for method_index, method in enumerate(methods):
            if method == "isomap_k10":
                value = isomap_mean
            elif method == "isomap_k15":
                value = isomap_k15_mean
            elif method == "isomap_k20":
                value = isomap_k20_mean
            else:
                value = float(dataset["full"]["values"][method])
            bar_x = center + (method_index - 4.0) * bar_w
            lines.append(
                f'<rect x="{bar_x - bar_w * 0.42:.1f}" y="{y(value):.1f}" width="{bar_w * 0.84:.1f}" height="{y(1.0) - y(value):.1f}" fill="{COLORS[method_index]}"/>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(DATASET_LABELS[dataset["dataset"]])}</text>'
        )
    for index, label in enumerate(labels):
        legend_x = left + (index % 4) * 215
        legend_y = height - 48 + (index // 4) * 21
        lines.extend(
            [
                f'<rect x="{legend_x}" y="{legend_y - 10}" width="16" height="10" fill="{COLORS[index]}"/>',
                f'<text x="{legend_x + 23}" y="{legend_y}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Reported dimension</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--isomap-results", type=Path, required=True)
    parser.add_argument("--real-stability-results", type=Path, required=True)
    parser.add_argument("--real-isomap-results", type=Path, required=True)
    parser.add_argument("--real-isomap-k15-results", type=Path, required=True)
    parser.add_argument("--real-isomap-k20-results", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    isomap_data = json.loads(args.isomap_results.read_text())
    real_stability = json.loads(args.real_stability_results.read_text())
    real_isomap = json.loads(args.real_isomap_results.read_text())
    real_isomap_k15 = json.loads(args.real_isomap_k15_results.read_text())
    real_isomap_k20 = json.loads(args.real_isomap_k20_results.read_text())
    aggregates = _aggregates(data, isomap_data)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (shape, dimension, snr_db), methods in aggregates.items():
        for method, stats in methods.items():
            rows.append(
                {
                    "shape": shape,
                    "intrinsic_dimension": dimension,
                    "snr_db": snr_db,
                    "observed_support_dimension": (
                        dimension
                        if snr_db is None
                        else data["configuration"]["ambient_dimension"]
                    ),
                    "method": method,
                    **stats,
                    "absolute_relative_error_percent": abs(
                        stats["mean"] / dimension - 1.0
                    )
                    * 100.0,
                }
            )
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    _write_heatmap(data, aggregates, prefix.with_name(prefix.name + "_clean.svg"))
    _write_noise_chart(
        data, aggregates, prefix.with_name(prefix.name + "_noise.svg")
    )
    _write_real_chart(
        real_stability,
        real_isomap,
        real_isomap_k15,
        real_isomap_k20,
        prefix.with_name(prefix.name + "_real.svg"),
    )

    methods = list(data["conditions"][0]["trials"][0]["values"]) + [
        "isomap_dimensionality"
    ]
    lines = [
        "# EffDim recovery on noisy embedded manifolds",
        "",
        "Linear subspaces, spheres, tori, a nonlinear chain, and a Swiss roll were",
        "sampled with 10,000 points, randomly embedded into 256 dimensions, and",
        "evaluated over five replicates. Core estimators used exact GPU neighbours;",
        "Landmark Isomap used a CAGRA `k=10` graph with 512 landmarks.",
        "",
        "## Main findings",
        "",
        "1. **On clean manifolds, MiND-MLk is best overall by median relative error",
        "   (4.6%), followed by DANCo (7.3%), Two-NN (11.6%), and MLE/TLE (11.8%).**",
        "   No local method is uniformly best: DANCo saturates on higher dimensions,",
        "   while Two-NN badly underestimates the nonlinear chain.",
        "2. **Landmark Isomap exactly recovers clean linear subspaces and the Swiss",
        "   roll, but its median clean error is 15%.** Closed topology prevents a",
        "   globally isometric Euclidean unfolding: spheres return about d+1 and",
        "   tori about 2d. The nonlinear chain's `k=10` graph retains only about 46%",
        "   of points in its largest component, invalidating that estimate.",
        "3. **Spectral metrics measure global embedding span, not nonlinear latent",
        "   dimension.** A sphere of intrinsic dimension d spans d+1 linear axes;",
        "   a d-torus spans 2d; and the one-dimensional chain has PCA-95 of 6.",
        "4. **At 30 dB, Two-NN has the lowest selected-method median error (12.2%),**",
        "   followed by participation ratio (15.1%) and MiND-MLk (19.5%). Local",
        "   neighbourhood geometry is already measurably distorted.",
        "5. **At 20 dB, participation ratio is best by median error (17.3%).** Isomap",
        "   follows at 22.5%; selected local estimators rise to approximately 30–47%.",
        "6. **At 10 dB, none of the methods reliably recovers the clean latent",
        "   dimension.** Gaussian ambient noise makes support dimension 256; PCA-95",
        "   is particularly inflated, with median error around 1,890%.",
        "7. **There is no universal winner.** MiND-MLk is strongest for clean manifold",
        "   recovery, participation ratio is comparatively robust to moderate noise,",
        "   Isomap is useful for unfoldable connected manifolds, and PCA-95 remains",
        "   a global variance/compression dimension.",
        "8. **On the real embeddings, Landmark Isomap is stable from k=10 to k=20.**",
        "   JWST estimates are 9.6, 9.4, and 9.0; DESI 6.6, 6.2, and 6.2; and",
        "   Legacy Survey 7.8, 8.4, and 8.2. Every graph remains fully connected,",
        "   and each dataset spans at most 0.6 dimensions across neighbourhood sizes.",
        "   These values align more closely with participation ratio and higher-order",
        "   Rényi dimensions than PCA-95, but no ground truth is available.",
        "",
        "## Clean-manifold recovery matrix",
        "",
        f"![]({prefix.name}_clean.svg)",
        "",
        "## Noise sensitivity",
        "",
        f"![]({prefix.name}_noise.svg)",
        "",
        "## Median absolute relative error across all shapes",
        "",
        "| Method | No noise | 30 dB | 20 dB | 10 dB |",
        "|:---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        errors = [
            _median_errors(data, aggregates, method, level)
            for level in (None, 30.0, 20.0, 10.0)
        ]
        lines.append(
            f"| {LABELS.get(method, 'Landmark Isomap')} | "
            + " | ".join(f"{error:.1f}%" for error in errors)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Mean noiseless estimates",
            "",
            "| Shape | True d | "
            + " | ".join(SHORT[method] for method in SELECTED)
            + " |",
            "|:---:|---:|" + "|".join("---:" for _ in SELECTED) + "|",
        ]
    )
    for shape, dimension in _ordered_manifolds(data):
        estimates = [
            aggregates[(shape, dimension, None)][method]["mean"]
            for method in SELECTED
        ]
        lines.append(
            f"| {SHAPE_LABELS[shape]} | {dimension} | "
            + " | ".join(_fmt(value) for value in estimates)
            + " |"
        )
    real_methods = (
        "pca_explained_variance_95",
        "shannon_entropy",
        "participation_ratio",
        "renyi_eff_dimensionality_alpha_5",
        "mle_dimensionality",
        "mind_mlk_dimensionality",
        "isomap_k10",
        "isomap_k15",
        "isomap_k20",
    )
    lines.extend(
        [
            "",
            "## Landmark Isomap on real embeddings",
            "",
            f"![]({prefix.name}_real.svg)",
            "",
            "All CAGRA `k=10`, `k=15`, and `k=20` graphs formed one connected component",
            "containing 100% of the data. Isomap entries are mean ± SD across five",
            "independent landmark selections. Other methods are deterministic.",
            "",
            "| Method | JWST | DESI | Legacy Survey |",
            "|:---:|---:|---:|---:|",
        ]
    )
    for method in real_methods:
        values = []
        for dataset in real_stability["datasets"]:
            if method in {"isomap_k10", "isomap_k15", "isomap_k20"}:
                source = {
                    "isomap_k10": real_isomap,
                    "isomap_k15": real_isomap_k15,
                    "isomap_k20": real_isomap_k20,
                }[method]
                matching = next(
                    item
                    for item in source["datasets"]
                    if item["dataset"] == dataset["dataset"]
                )
                estimates = np.asarray(
                    [trial["estimate"] for trial in matching["trials"]],
                    dtype=float,
                )
                values.append(
                    f"{np.mean(estimates):.1f} ± {np.std(estimates, ddof=1):.1f}"
                )
            else:
                values.append(_fmt(float(dataset["full"]["values"][method])))
        lines.append(
            f"| {LABELS.get(method, 'Landmark Isomap ' + method.removeprefix('isomap_').replace('k', 'k='))} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "The accompanying CSV contains mean, standard deviation, and relative error",
            "for all 15 non-GMST methods plus Isomap under every synthetic shape and",
            "noise condition.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
