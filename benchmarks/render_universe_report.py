#!/usr/bin/env python3
"""Append the UniverseTBD Python/Rust comparison to the consolidated report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SECTION_MARKER = "## 5. Python main versus Rust on UniverseTBD embeddings"
METRIC_LABELS = {
    "pca_explained_variance_95": "PCA explained variance (95%)",
    "participation_ratio": "Participation ratio",
    "shannon_entropy": "Shannon entropy",
    "renyi_eff_dimensionality_alpha_2": "Rényi α=2",
    "renyi_eff_dimensionality_alpha_3": "Rényi α=3",
    "renyi_eff_dimensionality_alpha_4": "Rényi α=4",
    "renyi_eff_dimensionality_alpha_5": "Rényi α=5",
    "geometric_mean_eff_dimensionality": "Geometric mean",
    "mle_dimensionality": "MLE",
    "two_nn_dimensionality": "Two-NN",
    "danco_dimensionality": "DANCo",
    "mind_mli_dimensionality": "MiND-MLi",
    "mind_mlk_dimensionality": "MiND-MLk",
    "ess_dimensionality": "ESS",
    "tle_dimensionality": "TLE",
}


def _time(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000:.1f} µs"
    if value < 1:
        return f"{value * 1000:.2f} ms"
    return f"{value:.3f} s"


def _advantage(python_s: float, rust_s: float) -> str:
    if python_s == 0 or rust_s == 0:
        return "—"
    ratio = python_s / rust_s
    if ratio >= 1:
        return f"{ratio:.2f}× Rust"
    return f"{1 / ratio:.2f}× Python"


def _relative_difference(python_value: float, rust_value: float) -> float:
    return abs(python_value - rust_value) / max(abs(python_value), 1e-12)


def _max_metric_difference(left: dict, right: dict) -> float:
    return max(
        _relative_difference(left[name]["value"], right[name]["value"])
        for name in left
    )


def _dataset_name(path: str) -> str:
    return {
        "jwst_dinov3_vitl16": "JWST",
        "desi_dinov3_small_vitl16": "DESI",
        "legacysurvey_dinov3_vitl16": "Legacy Survey",
    }[Path(path).stem]


def _pca_chart(path: Path, results: list[dict]) -> None:
    width, height = 1000, 500
    left, right, top, bottom = 80, 30, 55, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    series = [
        ("Python regular", "#2563eb", "python", "regular_pca_s"),
        ("Python streaming", "#60a5fa", "python", "streaming_pca_s"),
        ("Rust regular", "#dc2626", "rust", "regular_pca_s"),
        ("Rust streaming", "#f87171", "rust", "streaming_pca_s"),
    ]
    values = [
        entry["implementations"][language][field]
        for entry in results
        for _, _, language, field in series
    ]
    minimum = min(values) / 1.5
    maximum = max(values) * 1.5

    def y(value: float) -> float:
        fraction = (math.log10(value) - math.log10(minimum)) / (
            math.log10(maximum) - math.log10(minimum)
        )
        return top + plot_h * (1 - fraction)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Regular vs streaming PCA on UniverseTBD embeddings</text>',
    ]
    for exponent in range(math.floor(math.log10(minimum)), math.ceil(math.log10(maximum)) + 1):
        value = 10**exponent
        py = y(value)
        lines.extend(
            [
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:g}s</text>',
            ]
        )
    group_width = plot_w / len(results)
    bar_width = group_width / 6
    for group, entry in enumerate(results):
        center = left + group_width * (group + 0.5)
        for index, (_, color, language, field) in enumerate(series):
            value = entry["implementations"][language][field]
            x = center + (index - 1.5) * bar_width
            bar_top = y(value)
            lines.append(
                f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{bar_width - 3:.1f}" height="{top + plot_h - bar_top:.1f}" fill="{color}"/>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{_dataset_name(entry["dataset"]).split()[0]}</text>'
        )
    for index, (label, color, _, _) in enumerate(series):
        x = left + index * 210
        lines.extend(
            [
                f'<rect x="{x}" y="{height - 42}" width="16" height="12" fill="{color}"/>',
                f'<text x="{x + 22}" y="{height - 32}" font-family="sans-serif" font-size="12">{label}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    metadata = json.loads(args.metadata.read_text())
    metadata_by_name = {item["name"]: item for item in metadata}
    chart_path = args.report.with_name("python_rust_universe_pca.svg")
    _pca_chart(chart_path, results)

    lines = [
        SECTION_MARKER,
        "",
        "### Scope",
        "",
        "This comparison uses the current pure-Python implementation from `main`",
        "at commit `d1e7af9` and the Rust migration based on commit `d60ef97`.",
        "Both implementations consume the **same CAGRA distances and neighbor",
        "indices**. GMST is excluded. Streaming PCA means chunked Chan covariance",
        "accumulation followed by covariance eigendecomposition, using float64",
        "accumulation and a 4,096-row chunk size in both languages.",
        "",
        "All runs were sequential and used one measured pass. Timings therefore",
        "include normal run-to-run noise; sub-millisecond scalar metric timings",
        "should be treated as directional rather than as stable microbenchmarks.",
        "",
        "### Real datasets",
        "",
        "| Dataset | Shape | CAGRA build | CAGRA search | Recall@10 |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for entry in results:
        name = Path(entry["dataset"]).stem
        item = metadata_by_name[name]
        cagra = entry["cagra"]
        lines.append(
            f"| {_dataset_name(entry['dataset'])} | {item['shape'][0]:,} × {item['shape'][1]:,} | "
            f"{cagra['build_s']:.3f} s | {cagra['search_s']:.3f} s | {cagra['recall_at_k'] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "The Legacy Survey input is capped at 100,000 rows. Only the selected",
            "embedding columns were streamed from Hugging Face; the full 108 GB",
            "repository was not downloaded.",
            "",
            "### PCA comparison",
            "",
            "![Python and Rust PCA comparison](python_rust_universe_pca.svg)",
            "",
            "| Dataset | Python regular | Rust regular | Advantage | Python streaming | Rust streaming | Advantage |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for entry in results:
        py = entry["implementations"]["python"]
        rust = entry["implementations"]["rust"]
        lines.append(
            f"| {_dataset_name(entry['dataset'])} | {py['regular_pca_s']:.3f} s | "
            f"{rust['regular_pca_s']:.3f} s | {_advantage(py['regular_pca_s'], rust['regular_pca_s'])} | "
            f"{py['streaming_pca_s']:.3f} s | {rust['streaming_pca_s']:.3f} s | "
            f"{_advantage(py['streaming_pca_s'], rust['streaming_pca_s'])} |"
        )

    lines.extend(
        [
            "",
            "| Dataset | Python PCA-95 | Rust PCA-95 | Python peak RSS | Rust peak RSS |",
            "|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for entry in results:
        py = entry["implementations"]["python"]
        rust = entry["implementations"]["rust"]
        lines.append(
            f"| {_dataset_name(entry['dataset'])} | "
            f"{py['streaming_spectral']['pca_explained_variance_95']['value']:.0f} | "
            f"{rust['streaming_spectral']['pca_explained_variance_95']['value']:.0f} | "
            f"{py['peak_rss_mib'] / 1024:.2f} GiB | {rust['peak_rss_mib'] / 1024:.2f} GiB |"
        )

    lines.extend(
        [
            "",
            "> **PCA result:** Rust regular SVD wins decisively on the small JWST",
            "> and large Legacy Survey matrices, but NumPy wins on the medium DESI",
            "> shape. Python streaming covariance is currently faster on all three",
            "> datasets. On the 100k × 1,024 matrix, the OpenBLAS-backed Rust path",
            "> completed in 1.31 s versus 0.90 s for Python.",
            "",
            "Python `main` switches to randomized SVD when both matrix dimensions",
            "are at least 1,000, while Rust regular PCA remains exact. Streaming",
            "covariance provides an exact common reference and avoids that algorithm",
            "difference.",
            "",
            "#### PCA numerical agreement",
            "",
            "| Dataset | Python regular vs streaming | Rust regular vs streaming | Python vs Rust streaming |",
            "|:---:|:---:|:---:|:---:|",
        ]
    )
    for entry in results:
        py = entry["implementations"]["python"]
        rust = entry["implementations"]["rust"]
        lines.append(
            f"| {_dataset_name(entry['dataset'])} | "
            f"{_max_metric_difference(py['spectral'], py['streaming_spectral']):.2e} | "
            f"{_max_metric_difference(rust['spectral'], rust['streaming_spectral']):.2e} | "
            f"{_max_metric_difference(py['streaming_spectral'], rust['streaming_spectral']):.2e} |"
        )

    lines.extend(
        [
            "",
            "Values are the maximum relative difference across the eight spectral",
            "dimensionality outputs. The larger Python regular-path differences on",
            "the 1,024-feature datasets come from its randomized-SVD branch and",
            "omission of the final component.",
            "",
            "### End-to-end non-GMST pipeline",
            "",
            "These totals add shared CAGRA build/search, PCA, scalar spectral",
            "metrics, and the seven geometry estimators. Rust uses its bundled",
            "geometry path so data conversion and neighbor arrays are shared once.",
            "",
            "| Dataset | Python regular total | Rust regular total | Python streaming total | Rust streaming total |",
            "|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for entry in results:
        py = entry["implementations"]["python"]
        rust = entry["implementations"]["rust"]
        common = entry["cagra"]["build_s"] + entry["cagra"]["search_s"]
        py_spectral = sum(item["time_s"] for item in py["spectral"].values())
        py_stream_spectral = sum(
            item["time_s"] for item in py["streaming_spectral"].values()
        )
        rust_spectral = sum(item["time_s"] for item in rust["spectral"].values())
        rust_stream_spectral = sum(
            item["time_s"] for item in rust["streaming_spectral"].values()
        )
        py_regular = common + py["regular_pca_s"] + py_spectral + py["geometry_bundle_s"]
        rust_regular = (
            common + rust["regular_pca_s"] + rust_spectral + rust["geometry_bundle_s"]
        )
        py_streaming = (
            common
            + py["streaming_pca_s"]
            + py_stream_spectral
            + py["geometry_bundle_s"]
        )
        rust_streaming = (
            common
            + rust["streaming_pca_s"]
            + rust_stream_spectral
            + rust["geometry_bundle_s"]
        )
        lines.append(
            f"| `{Path(entry['dataset']).stem}` | {py_regular:.3f} s | {rust_regular:.3f} s | "
            f"{py_streaming:.3f} s | {rust_streaming:.3f} s |"
        )

    lines.extend(["", "### Per-estimator comparison", ""])
    for entry in results:
        py = entry["implementations"]["python"]
        rust = entry["implementations"]["rust"]
        dataset_name = Path(entry["dataset"]).stem
        lines.extend(
            [
                f"#### `{dataset_name}`",
                "",
                "| Estimator | Python | Rust | Speed advantage | Relative value difference |",
                "|:---:|:---:|:---:|:---:|:---:|",
            ]
        )
        for group in ("spectral", "geometry"):
            for name, py_item in py[group].items():
                rust_item = rust[group][name]
                difference = _relative_difference(
                    py_item["value"], rust_item["value"]
                )
                lines.append(
                    f"| {METRIC_LABELS[name]} | {_time(py_item['time_s'])} | "
                    f"{_time(rust_item['time_s'])} | "
                    f"{_advantage(py_item['time_s'], rust_item['time_s'])} | "
                    f"{difference:.2e} |"
                )
        lines.append("")

    lines.extend(
        [
            "### Interpretation",
            "",
            "1. **CAGRA is accurate on real embeddings.** Recall@10 ranges from",
            "   99.04% to 99.99%, substantially better than on isotropic Gaussian",
            "   vectors at comparable settings.",
            "2. **Rust regular PCA is shape-dependent but strong at scale.** It is",
            "   about 4.5× faster than Python regular PCA on the 100k × 1024 Legacy",
            "   Survey matrix, while NumPy is faster on the 20k × 768 DESI matrix.",
            "3. **Python streaming PCA remains faster in isolation, but Rust wins the",
            "   optimized pipeline.** On the 100k × 1,024 matrix, streaming PCA took",
            "   0.90 s in Python and 1.31 s in Rust. Faster Rust geometry reduced the",
            "   complete non-GMST pipeline to 18.32 s versus 23.47 s for Python.",
            "4. **Estimator values agree closely.** Geometry outputs use identical",
            "   CAGRA neighbors and agree to numerical precision. Differences in",
            "   regular spectral results primarily reflect Python's randomized-SVD",
            "   branch, not a language-level discrepancy.",
            "5. **Cheap per-method Rust calls pay conversion overhead.** Several",
            "   simple geometry estimators look slower through individual PyO3",
            "   calls because each call converts the full input to float32 and",
            "   copies precomputed distances. The bundled Rust geometry path shares",
            "   those conversions and is the relevant end-to-end implementation.",
            "",
        ]
    )

    existing = args.report.read_text()
    if SECTION_MARKER in existing:
        existing = existing.split(SECTION_MARKER, 1)[0].rstrip()
    args.report.write_text(existing + "\n\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
