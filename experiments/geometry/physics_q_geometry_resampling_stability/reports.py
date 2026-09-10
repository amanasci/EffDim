"""METHODS / REPORT / MANUSCRIPT_RECOMMENDATION. No manuscript edits."""

from __future__ import annotations

from pathlib import Path


def write_reports(out: Path, *, decision: dict, parity: dict, runtime: dict, summary: dict) -> None:
    (out / "METHODS.md").write_text(_methods(runtime))
    (out / "REPORT.md").write_text(_report(decision, parity, runtime, summary))
    (out / "MANUSCRIPT_RECOMMENDATION.md").write_text(_manuscript(decision))


def _methods(runtime: dict) -> str:
    return (
        "# Methods — Q geometry-resampling stability\n\n"
        "This audit resamples the observations used to estimate the frozen ViT-B "
        "finite-patch statistic K_H^cross, then recomputes the scientific associations "
        "with frozen probe outcomes and frozen primary controls. It is not an ordinary "
        "anchor bootstrap: those treat the curvature field as fixed.\n\n"
        "Production estimator: nested_dimension_curvature._fit_rank → fit_quad (RIDGES "
        "[1e-4 … 3]) → unpacked cross_metric_pair. Negative cross-products are not clamped. "
        "Frozen frames (x0, J) and d=16 are reused. No decoder, probe, or label-model refit.\n\n"
        "Scheme A keeps the frozen k=2048 neighbour IDs and repartitions them into disjoint "
        "1024/1024 halves with a hash of (experiment seed, replicate, sample_id).\n\n"
        "Scheme B draws one shared 80% inclusion mask over the embedding table, recomputes "
        "k'=1638 neighbours, and splits into 819/819. Query anchors remain queries; they "
        "are excluded from their own neighbour set; other anchors follow the mask.\n\n"
        "Primary associations use the frozen controlled-rank residualization "
        "(log kNN radius, local label variance, evaluation count). Replicate-specific "
        "radius is a marked sensitivity only.\n\n"
        f"Pilot then adaptive replicate count. Runtime: {runtime}.\n"
    )


def _report(decision, parity, runtime, summary) -> str:
    return (
        "# Report — Q geometry resampling\n\n"
        f"## Runtime\n{runtime}\n\n"
        f"## Parity\n{parity}\n\n"
        f"## Decision\n{decision.get('summary_label')}\n\n"
        f"{decision.get('distinction')}\n\n"
        f"Prior labels not overwritten: {decision.get('prior_labels_not_overwritten')}\n\n"
        f"{summary.get('headline_md', '')}\n"
    )


def _manuscript(decision) -> str:
    return (
        "# Manuscript recommendation\n\n"
        "Do not edit the manuscript in this experiment.\n\n"
        f"If a later draft mentions geometry uncertainty of Q, use `{decision.get('summary_label')}` "
        "and keep it distinct from `quadratic_chart_link_unresolved`, "
        "`neither_estimator_validated`, and "
        "`q_moderately_informative_sampling_dependent_statistic`.\n"
    )
