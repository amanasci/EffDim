"""METHODS / REPORT / MANUSCRIPT_RECOMMENDATION. No manuscript file edits."""

from __future__ import annotations

from pathlib import Path


def write_reports(out: Path, *, decision: dict, parity: dict, runtime: dict, summary: dict) -> None:
    (out / "METHODS.md").write_text(_methods(runtime))
    (out / "REPORT.md").write_text(_report(decision, parity, runtime, summary))
    (out / "MANUSCRIPT_RECOMMENDATION.md").write_text(_manuscript(decision, summary))


def _methods(runtime: dict) -> str:
    wall = runtime.get("wall_s")
    return (
        "# Methods — estimator operating characteristics\n\n"
        "This audit does not reverse frozen exact-recovery labels. It scores two historical\n"
        "measures plus one diagnostic on the already-computed D=28 fixture panel.\n\n"
        "## Estimands\n\n"
        "- **D-full** (historical): H_D^E = g^{ab} II^E_{D,ab} with II^E=(I-P_T)D^2 F "
        "from raw `decode`. Scored against analytic full Euclidean truth only.\n"
        "- **Q** (historical): split-half K_H^{cross} and frozen K_{dir}^{cross} from "
        "`estimator_q.fit_anchor_quadratic`. Primary utility is rank recovery of "
        "K_H^{cross} against the sampling-matched finite-patch scalar "
        "K_{H,T2}^* = ||H_{patch,T2}^S||^2 using the same Hessian/whitening "
        "normalization as production (`kdir_from_pair` self-cross).\n"
        "- **D-residual** (diagnostic only): H_D^S = g^{ab} B^S_{D,ab} through F̃=F/||F||.\n\n"
        "On unit-sphere fixtures ||H^E||^2 = d^2 + ||H^S||^2. Rank targets with negligible "
        "dynamic range are marked `rank_target_degenerate`; Spearman is not a pass/fail metric "
        "there. Vector cosine, residualized magnitude sqrt(max(||H^E||^2-d^2,0)), and "
        "calibration are used instead.\n\n"
        "## Reuse\n\n"
        "Existing dual-estimator per-anchor parquet, cell-level metric CSVs, and the "
        "pointwise decoder reproduction R2/R3 cubic/ridge cells are read-only. Clouds, "
        "anchors, T2/T3 oracles, and Q neighbourhoods are regenerated from frozen seeds "
        "(DATA_SEED=20260816, ANCHOR_HASH_SEED=20260907) without retraining those 12 decoders.\n\n"
        "## Repeat panel\n\n"
        "F4 only, d=16, D=28, n_dense=5000, n_sparse=1500, the same 64 anchors. Two "
        "independent observation draws A/B (seeds 20260911 / 20260912) on clean uniform "
        "and on S3+N4. Q is rerun with the frozen k=1024, 3 splits, RIDGES grid. At most "
        "two decoder initialization seeds {0,1} per draw and condition (≤8 new AEs). "
        "Architecture, epochs, rank, bandwidth and regularization are not swept.\n\n"
        "## Metrics\n\n"
        "Rank: Spearman, Kendall tau, pairwise ordering accuracy, rank RMSE. "
        "Quartiles: truth-defined top/bottom groups with explicit chance baselines. "
        "Calibration: one global multiplicative factor fit on a frozen half of the 64 "
        "anchors (seed 20260907) and applied to the complement across conditions. "
        "Reliability: r_rel = rho(Khat_A, Khat_B); attenuation ceiling sqrt(r_rel) only "
        "when r_rel>0. Fraction of ceiling is stored unclamped; display is clamped at 1. "
        "200-bootstrap intervals on anchors. Density quintiles use the analytic sampling "
        "weight w=exp(beta s).\n\n"
        "## Noise-scale comparator\n\n"
        "Synthetic RMS is compared to existing ViT-B decoder reconstruction residual / "
        "global signal scale from the physics D-residual run, and to any already stored "
        "quadratic residual / neighbourhood radius. These are empirical scale comparators, "
        "not identifications of observational noise. No new embeddings or augmentations.\n\n"
        f"Runtime wall: {wall} s. New AE cap: 8.\n"
    )


def _report(decision: dict, parity: dict, runtime: dict, summary: dict) -> str:
    return f"""# Report — operating characteristics of D-full and Q

## Runtime

- actual_s: {runtime.get("runtime_s")}
- wall_s: {runtime.get("wall_s")}
- new_decoder_fits: {runtime.get("n_ae")}
- reused_cells: {runtime.get("reused_cells")}
- new_cells: {runtime.get("new_cells")}
- stopped_before_cap: {runtime.get("stopped_before_cap")}

## Parity

- ok: {parity.get("ok")}
- details: `{parity}`

## Estimator labels (this audit only)

- D-full: **{decision.get("d_full_summary_label")}**
- Q: **{decision.get("q_summary_label")}**
- D-residual diagnostic: **{decision.get("d_residual_summary_label")}**
- Prior exact-recovery label **{decision.get("prior_exact_recovery_label")}** was not overwritten.

{decision.get("distinction")}

## Headline operating facts

{summary.get("headline_md", "")}

## Dimensions

{summary.get("dimensions_md", "")}
"""


def _manuscript(decision: dict, summary: dict) -> str:
    return f"""# Manuscript recommendation

Do not edit the manuscript in this experiment.

Exact-recovery validation and practical operating utility are different questions.
The frozen dual-estimator label remains `{decision.get("prior_exact_recovery_label")}`.

If a methods or results sentence is added later (not in this run), keep D-full, Q,
and D-residual as separate instruments:

- D-full: {decision.get("d_full_summary_label")}
- Q: {decision.get("q_summary_label")}
- D-residual (diagnostic): {decision.get("d_residual_summary_label")}

Do not collapse them into a single winner. Do not treat accuracy drop under noise
as automatic invalidation; report retained correlation, repeat reliability, and
fraction of the attenuation ceiling instead.

{summary.get("manuscript_note", "")}
"""
