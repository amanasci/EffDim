"""METHODS / REPORT / MANUSCRIPT_RECOMMENDATION."""

from __future__ import annotations

from pathlib import Path

from .io_util import write_text


def write_markdowns(out: Path, *, decision, p1, p2, cell_df, parity, tests, runtime, hy_med, man) -> None:
    write_text(
        out / "METHODS.md",
        "\n".join(
            [
                "# Methods",
                "",
                "Fresh implementation of the readout-curvature Hessian-mismatch method from the written spec.",
                "Paper PDF was not located in the repository; Table 2 values are the stated approximations.",
                "Decoders: reused frozen PlainAutoEncoder checkpoints (250×3 SiLU, 400 epochs, anchors excluded).",
                "No new autoencoders. Probe split: `task_aligned_curvature_v1` 60/40, ridge α=100.",
                r"Primary: \(M_\Delta=\|\mathcal H_y-B_w\|_g\) vs held-out MSE (P1 > 0) and \(A_{\mathrm{full}}\) vs MSE (P2 < 0).",
                r"Shape \(S=\|B_w^S\|_g\) has no preregistered sign.",
                "Label Hessian: QLCA Frobenius features, train neighbours only, paper OLS with frozen ridge=1 fallback.",
                "Q tensors were not serialized; Q panel marked unavailable.",
                "Manuscript not edited. Prior trees read-only.",
                "",
            ]
        ),
        force=True,
    )
    lines = [
        "# Report",
        "",
        f"Primary label: `{decision['label']}`.",
        f"Reason: {decision.get('reason')}.",
        "",
        f"Wall **{runtime:.0f} s**. Tests {tests.get('n_passed')}/{tests.get('n_tests')}. New decoders trained: 0.",
        f"Paper PDF found: {parity.get('paper_pdf_found')}.",
        f"Label-Hessian median split-half cosine: {hy_med}.",
        "",
        "## P1 / P2",
        "",
        f"P1 {p1.get('observed')} CI {p1.get('ci95')} Holm p={p1.get('p_holm')} pass={p1.get('pass_holm')}",
        f"P2 {p2.get('observed')} CI {p2.get('ci95')} Holm p={p2.get('p_holm')} pass={p2.get('pass_holm')}",
        f"P1 per-model {p1.get('per_model')}",
        f"P2 per-model {p2.get('per_model')}",
        "",
        "## ViT-B paper-table recovery (historical local R², raw Spearman)",
        "",
        str(parity.get("recovered")),
        "",
    ]
    if cell_df is not None and len(cell_df):
        lines += ["## Per model/target", "", cell_df.to_string(index=False), ""]
    write_text(out / "REPORT.md", "\n".join(lines) + "\n", force=True)
    write_text(
        out / "MANUSCRIPT_RECOMMENDATION.md",
        "\n".join(
            [
                "# Manuscript recommendation",
                "",
                "Do not automatically rewrite the manuscript.",
                f"Frozen label: `{decision['label']}`.",
                "This tree does not overwrite prior task-aligned or D-residual labels.",
                r"Keep mismatch \(M_\Delta\) distinct from shape energy \(S\) and from generic \(K_H\).",
                "",
            ]
        ),
        force=True,
    )
    _ = man
