#!/usr/bin/env python3
"""Render the UniverseTBD EffDim stability experiment as CSV, SVG, and Markdown."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np


LABELS = {
    "pca_explained_variance_95": "PCA-95",
    "participation_ratio": "Participation ratio",
    "shannon_entropy": "Shannon ED",
    "renyi_eff_dimensionality_alpha_2": "Rényi α=2",
    "renyi_eff_dimensionality_alpha_3": "Rényi α=3",
    "renyi_eff_dimensionality_alpha_4": "Rényi α=4",
    "renyi_eff_dimensionality_alpha_5": "Rényi α=5",
    "geometric_mean_eff_dimensionality": "Geometric mean ED",
    "mle_dimensionality": "MLE",
    "two_nn_dimensionality": "Two-NN",
    "danco_dimensionality": "DANCo",
    "mind_mli_dimensionality": "MiND-MLi",
    "mind_mlk_dimensionality": "MiND-MLk",
    "ess_dimensionality": "ESS",
    "tle_dimensionality": "TLE",
}
SHORT = {
    "pca_explained_variance_95": "PCA95",
    "participation_ratio": "PR",
    "shannon_entropy": "Shannon",
    "renyi_eff_dimensionality_alpha_2": "R2",
    "renyi_eff_dimensionality_alpha_3": "R3",
    "renyi_eff_dimensionality_alpha_4": "R4",
    "renyi_eff_dimensionality_alpha_5": "R5",
    "geometric_mean_eff_dimensionality": "GeoMean",
    "mle_dimensionality": "MLE",
    "two_nn_dimensionality": "TwoNN",
    "danco_dimensionality": "DANCo",
    "mind_mli_dimensionality": "MiND-i",
    "mind_mlk_dimensionality": "MiND-k",
    "ess_dimensionality": "ESS",
    "tle_dimensionality": "TLE",
}
DATASET_LABELS = {
    "jwst_dinov3_vitl16": "JWST",
    "desi_dinov3_small_vitl16": "DESI",
    "legacysurvey_dinov3_vitl16": "Legacy Survey",
}
SCHEMES = ("bootstrap_deduplicated", "sparse_10pct", "contiguous_region")
SCHEME_LABELS = {
    "bootstrap_deduplicated": "Bootstrap, duplicates merged",
    "sparse_10pct": "Sparse 10%",
    "contiguous_region": "12 regions",
}


def _summary(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if not len(finite):
        return {key: math.nan for key in ("mean", "std", "median", "q025", "q975", "min", "max", "cv")}
    mean = float(np.mean(finite))
    return {
        "mean": mean,
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "median": float(np.median(finite)),
        "q025": float(np.quantile(finite, 0.025)),
        "q975": float(np.quantile(finite, 0.975)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "cv": float(np.std(finite, ddof=1) / abs(mean))
        if len(finite) > 1 and mean != 0.0
        else 0.0,
    }


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "invalid"
    absolute = abs(value)
    if absolute == 0:
        return "0"
    if absolute >= 1000 or absolute < 0.001:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _mean_std(summary: dict) -> str:
    return f"{_fmt(summary['mean'])} ± {_fmt(summary['std'])}"


def _collect(dataset: dict, method: str, scheme: str) -> list[float]:
    return [
        float(trial["values"][method])
        for trial in dataset["trials"]
        if trial["scheme"] == scheme
    ]


def _write_csvs(data: dict, prefix: Path) -> None:
    summary_rows = []
    raw_rows = []
    for dataset in data["datasets"]:
        for method, full_value in dataset["full"]["values"].items():
            for scheme in SCHEMES:
                values = _collect(dataset, method, scheme)
                stats = _summary(values)
                summary_rows.append(
                    {
                        "dataset": dataset["dataset"],
                        "method": method,
                        "scheme": scheme,
                        "full_value": full_value,
                        "iterations": len(values),
                        **stats,
                        "median_ratio_to_full": stats["median"] / full_value
                        if full_value
                        else math.nan,
                    }
                )
                matching = [
                    trial
                    for trial in dataset["trials"]
                    if trial["scheme"] == scheme
                ]
                for trial, value in zip(matching, values):
                    raw_rows.append(
                        {
                            "dataset": dataset["dataset"],
                            "scheme": scheme,
                            "iteration": trial["iteration"],
                            "n_samples": trial["n_samples"],
                            "method": method,
                            "value": value,
                            "ratio_to_full": value / full_value
                            if full_value
                            else math.nan,
                        }
                    )
    for path, rows in (
        (prefix.with_name(prefix.name + "_summary.csv"), summary_rows),
        (prefix.with_name(prefix.name + "_raw.csv"), raw_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def _write_svg(dataset: dict, scheme: str, path: Path) -> None:
    methods = list(dataset["full"]["values"])
    width, height = 1500, 1900
    columns, rows = 2, 8
    margin_x, top, bottom = 35, 75, 25
    gap_x, gap_y = 20, 22
    panel_w = (width - 2 * margin_x - (columns - 1) * gap_x) / columns
    panel_h = (height - top - bottom - (rows - 1) * gap_y) / rows
    colors = {
        "bootstrap_deduplicated": "#dc2626",
        "sparse_10pct": "#2563eb",
        "contiguous_region": "#16a34a",
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="823" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="22">{html.escape(DATASET_LABELS[dataset["dataset"]])}: {html.escape(SCHEME_LABELS[scheme])}</text>',
        f'<text x="{width / 2}" y="53" text-anchor="middle" font-family="sans-serif" font-size="13">Histogram and rug marks show every raw observation; dashed line: whole-dataset estimate; logarithmic dimensionality axis</text>',
    ]

    for index, method in enumerate(methods):
        row, column = divmod(index, columns)
        panel_x = margin_x + column * (panel_w + gap_x)
        panel_y = top + row * (panel_h + gap_y)
        plot_left = panel_x + 45
        plot_right = panel_x + panel_w - 12
        plot_top = panel_y + 27
        plot_bottom = panel_y + panel_h - 32
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        values = np.asarray(_collect(dataset, method, scheme), dtype=float)
        values = values[np.isfinite(values)]
        full = float(dataset["full"]["values"][method])
        use_log = bool(np.all(values > 0.0) and full > 0.0)
        transformed = np.log10(values) if use_log else values.copy()
        full_transformed = math.log10(full) if use_log else full

        observed_min = float(np.min(transformed))
        observed_max = float(np.max(transformed))
        if observed_max - observed_min < 1e-10:
            observed_min -= 0.05
            observed_max += 0.05
        bin_count = min(8, max(4, math.ceil(math.sqrt(len(transformed)))))
        counts, edges = np.histogram(
            transformed,
            bins=bin_count,
            range=(observed_min, observed_max),
        )
        domain_min = min(float(edges[0]), full_transformed)
        domain_max = max(float(edges[-1]), full_transformed)
        padding = max((domain_max - domain_min) * 0.05, 1e-6)
        domain_min -= padding
        domain_max += padding
        max_count = max(1, int(np.max(counts)))

        def x(value: float) -> float:
            return plot_left + (value - domain_min) / (domain_max - domain_min) * plot_w

        def y(count: float) -> float:
            return plot_bottom - count / max_count * plot_h

        lines.extend(
            [
                f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#d1d5db"/>',
                f'<text x="{panel_x + panel_w / 2:.1f}" y="{panel_y + 19:.1f}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">{html.escape(SHORT[method])} (n={len(values)})</text>',
                f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" x2="{plot_right:.1f}" y2="{plot_bottom:.1f}" stroke="#374151"/>',
                f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" stroke="#374151"/>',
                f'<text x="{plot_left - 6:.1f}" y="{plot_top + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{max_count}</text>',
                f'<text x="{plot_left - 6:.1f}" y="{plot_bottom + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">0</text>',
            ]
        )
        for count, edge_left, edge_right in zip(counts, edges[:-1], edges[1:]):
            bar_left = x(float(edge_left))
            bar_right = x(float(edge_right))
            bar_top = y(float(count))
            lines.append(
                f'<rect x="{bar_left + 1:.1f}" y="{bar_top:.1f}" width="{max(1.0, bar_right - bar_left - 2):.1f}" height="{max(0.0, plot_bottom - bar_top):.1f}" fill="{colors[scheme]}" opacity="0.78"/>'
            )
        for value in transformed:
            observation_x = x(float(value))
            lines.append(
                f'<line x1="{observation_x:.1f}" y1="{plot_bottom - 7:.1f}" x2="{observation_x:.1f}" y2="{plot_bottom:.1f}" stroke="#111827" stroke-width="1"/>'
            )
        full_x = x(full_transformed)
        lines.append(
            f'<line x1="{full_x:.1f}" y1="{plot_top:.1f}" x2="{full_x:.1f}" y2="{plot_bottom:.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="6,4"/>'
        )
        axis_min = 10.0**domain_min if use_log else domain_min
        axis_max = 10.0**domain_max if use_log else domain_max
        lines.extend(
            [
                f'<text x="{plot_left:.1f}" y="{plot_bottom + 17:.1f}" text-anchor="start" font-family="sans-serif" font-size="10">{html.escape(_fmt(axis_min))}</text>',
                f'<text x="{plot_right:.1f}" y="{plot_bottom + 17:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{html.escape(_fmt(axis_max))}</text>',
                f'<text x="{panel_x + panel_w / 2:.1f}" y="{panel_y + panel_h - 5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">reported dimensionality{" (log)" if use_log else ""}</text>',
            ]
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def _region_tables(dataset: dict, methods: list[str]) -> list[str]:
    regions = [
        trial for trial in dataset["trials"] if trial["scheme"] == "contiguous_region"
    ]
    output = []
    for start, stop in ((0, 6), (6, 12)):
        selected = regions[start:stop]
        headers = ["Method"]
        if start == 0:
            headers.append("Whole")
        headers.extend(f"R{trial['iteration'] + 1}" for trial in selected)
        output.extend(
            [
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(":---:" for _ in headers) + "|",
            ]
        )
        for method in methods:
            row = [LABELS[method]]
            if start == 0:
                row.append(_fmt(float(dataset["full"]["values"][method])))
            row.extend(_fmt(float(trial["values"][method])) for trial in selected)
            output.append("| " + " | ".join(row) + " |")
        output.append("")
    return output


def _write_report(data: dict, prefix: Path) -> None:
    methods = list(data["datasets"][0]["full"]["values"])
    stability = {}
    for method in methods:
        scheme_spreads = {}
        all_spreads = []
        all_biases = []
        for scheme in SCHEMES:
            spreads = []
            for dataset in data["datasets"]:
                full = float(dataset["full"]["values"][method])
                values = np.asarray(_collect(dataset, method, scheme))
                finite = values[np.isfinite(values)]
                if len(finite) < 2 or full == 0.0:
                    continue
                spreads.append(float(np.std(finite, ddof=1) / abs(full) * 100.0))
                all_biases.append(
                    float(abs(np.median(finite) / full - 1.0) * 100.0)
                )
            scheme_spreads[scheme] = float(np.median(spreads))
            all_spreads.extend(spreads)
        stability[method] = {
            "scheme_spreads": scheme_spreads,
            "overall_spread": float(np.median(all_spreads)),
            "overall_bias": float(np.median(all_biases)),
        }
    stability_order = sorted(methods, key=lambda method: stability[method]["overall_spread"])
    configuration = data["configuration"]
    ess_spreads = stability["ess_dimensionality"]["scheme_spreads"]

    lines = [
        "# EffDim stability across UniverseTBD embedding samples",
        "",
        "## Experiment",
        "",
        f"Each dataset was evaluated four ways: the complete matrix; {configuration['bootstrap_iterations']} row-bootstrap",
        f"replicates; {configuration['sparse_iterations']} uniform 10% samples without replacement; and {configuration['regions']} contiguous,",
        "nearly equal stored-order chunks. Repeated draws in each bootstrap replicate",
        "were merged into one unique row. All 15 non-GMST methods used streaming covariance and",
        "exact cuVS GPU neighbours (`k=10`).",
        "",
        "The contiguous chunks test sensitivity to stored row order. They are not",
        "physical sky regions because the embedding matrices contain no coordinates.",
        "",
        "## Main findings",
        "",
        "1. **The simplified ESS proxy has the lowest normalized spread, but it is not",
        "   a calibrated dimension estimate.** Its median relative standard deviations",
        f"   are {ess_spreads['bootstrap_deduplicated']:.3f}% for bootstrap samples,",
        f"   {ess_spreads['sparse_10pct']:.3f}% for 10% samples, and",
        f"   {ess_spreads['contiguous_region']:.3f}% for regions. This indicates repeatability only.",
        "2. **Large-dataset spectral estimates are stable.** On DESI and Legacy Survey,",
        "   participation ratio and all Rényi dimensions remain close to the complete",
        "   dataset under bootstrap, 10% sampling, and region splits. Region CVs are",
        "   approximately 2–5%.",
        "3. **JWST is too small for 12-way high-dimensional spectral comparisons.**",
        "   Each region has only about 125 rows but 1,024 features, imposing a rank",
        "   ceiling. PCA-95 falls from 145 to 47–55 and geometric-mean ED becomes",
        "   numerically dominated by the null spectrum.",
        "4. **Deduplicated bootstrap avoids artificial zero-distance clouds.** It",
        "   behaves as a random unique subset containing about 63% of the rows, so",
        "   remaining differences primarily reflect sample-size sensitivity.",
        "5. **ESS and DANCo are the most region-stable geometry methods.** ESS region",
        "   CV is 0.7–1.9% and DANCo is 1.2–3.4% across the three datasets.",
        "6. **Agreement across regions does not imply agreement with the whole set.**",
        "   MLE/TLE regions have low internal CV but systematically return only about",
        "   75–81% of the whole-dataset estimate, demonstrating sample-size bias.",
        "7. **Two-NN is sensitive to ties already present in these embeddings.** Its",
        "   Legacy Survey region CV is 53.5%, and the whole JWST estimate (0.115)",
        "   is inconsistent with its region estimates (about 4–7).",
        "",
        "## Stability ranking",
        "",
        "Lower relative standard deviation means better repeatability. The bias column",
        "is also required: a method can have low spread while being consistently far",
        "from its whole-dataset estimate. Values are medians across datasets, and the",
        "overall columns are medians across all three schemes and datasets.",
        "",
        "| Method | Bootstrap RSD | Sparse RSD | Region RSD | Overall RSD | Overall absolute bias |",
        "|:---:|---:|---:|---:|---:|---:|",
    ]
    for method in stability_order:
        row = stability[method]
        scheme_spreads = row["scheme_spreads"]
        lines.append(
            f"| {LABELS[method]} | "
            f"{scheme_spreads['bootstrap_deduplicated']:.3f}% | "
            f"{scheme_spreads['sparse_10pct']:.3f}% | "
            f"{scheme_spreads['contiguous_region']:.3f}% | "
            f"{row['overall_spread']:.3f}% | "
            f"{row['overall_bias']:.2f}% |"
        )
    lines.extend(
        [
        "",
        "## Whole-dataset estimates",
        "",
        "| Method | JWST | DESI | Legacy Survey |",
        "|:---:|:---:|:---:|:---:|",
        ]
    )
    for method in methods:
        values = [
            _fmt(float(dataset["full"]["values"][method]))
            for dataset in data["datasets"]
        ]
        lines.append(f"| {LABELS[method]} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
        "## Sampled dimensionality distributions",
            "",
        "Each panel is a histogram of the raw reported dimensionalities. Short rug",
        "marks at the baseline show every individual observation, and the dashed",
        "vertical line is the whole-dataset estimate. Horizontal axes are logarithmic.",
        "The tables separately provide compact mean ± SD summaries.",
            "",
        ]
    )
    for dataset in data["datasets"]:
        lines.extend(
            [
                f"### {DATASET_LABELS[dataset['dataset']]}",
                "",
            ]
        )
        for scheme in SCHEMES:
            lines.extend(
                [
                f"![]({prefix.name}_{dataset['dataset']}_{scheme}.svg)",
                "",
                ]
            )
        lines.extend(
            [
                "| Method | Whole dataset | Bootstrap mean ± SD | Sparse mean ± SD | Regions mean ± SD |",
                "|:---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            full = float(dataset["full"]["values"][method])
            summaries = {
                scheme: _summary(_collect(dataset, method, scheme))
                for scheme in SCHEMES
            }
            lines.append(
                f"| {LABELS[method]} | "
                f"{_fmt(full)} | "
                f"{_mean_std(summaries['bootstrap_deduplicated'])} | "
                f"{_mean_std(summaries['sparse_10pct'])} | "
                f"{_mean_std(summaries['contiguous_region'])} |"
            )
        lines.extend(
            [
                "",
                "#### Individual contiguous-region estimates",
                "",
            ]
        )
        lines.extend(_region_tables(dataset, methods))

    prefix.with_suffix(".md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csvs(data, args.output_prefix)
    for dataset in data["datasets"]:
        for scheme in SCHEMES:
            _write_svg(
                dataset,
                scheme,
                args.output_prefix.with_name(
                    f"{args.output_prefix.name}_{dataset['dataset']}_{scheme}.svg"
                ),
            )
    _write_report(data, args.output_prefix)


if __name__ == "__main__":
    main()
