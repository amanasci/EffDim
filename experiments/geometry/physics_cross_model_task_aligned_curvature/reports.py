"""METHODS / REPORT / MANUSCRIPT_RECOMMENDATION. No paper edits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io_util import write_text


def write_markdowns(out: Path, *, decision, r1, rd, model_p2, per_target, kh_parity, runtime, tests) -> None:
    write_text(
        out / "METHODS.md",
        "\n".join(
            [
                "# Methods",
                "",
                "Leakage-safe confirmatory task-aligned curvature on ViT-B, DINOv3, CLIP, ConvNeXt-B and ViT-L.",
                "Same hash split `task_aligned_curvature_v1:{target}:{sample_id}` as the ViT-B experiment.",
                "Ridge α=100, unpenalized intercept, train-only. Evaluate eval neighbours only; min eval 128.",
                "Primary Q statistic: signed split-half sphere-normal cross energy. No clamp. No average-then-square.",
                "R1 is the equal-model-weight mean of equal-target-weight ρ_ctl(E_Q, MSE) over the four replication encoders.",
                "ViT-B is the reference and is excluded from R1. Decoder energy is secondary and may be skipped for wall time.",
                "No new autoencoders. Prior trees read-only. Manuscript not edited.",
                "",
            ]
        ),
        force=True,
    )
    r1o = r1 or {}
    lines = [
        "# Report",
        "",
        f"Primary label: `{decision['label']}`.",
        f"Reason: {decision.get('reason')}.",
        "",
        f"Wall time **{runtime:.0f} s**.",
        f"Unit tests: {tests.get('n_passed')}/{tests.get('n_tests')}.",
        "",
        "## R1 (replication models, Q vs held-out MSE)",
        "",
        f"Estimate {r1o.get('observed')}, CI {r1o.get('ci95')}, Holm p={r1o.get('p_holm')}, pass={r1o.get('pass_holm')}.",
        f"Per-model bars: {r1o.get('per_model')}.",
        f"Models with predicted sign: {r1o.get('n_models_positive')}/{r1o.get('n_models')}.",
        "",
        "## Per-model P2",
        "",
    ]
    for m, rec in (model_p2 or {}).items():
        lines.append(f"- {m}: bar={rec.get('observed')} CI={rec.get('ci95')} p={rec.get('p_mc')} targets={rec.get('per_target')}")
    lines += [
        "",
        f"KHcross recon median |diff|: {kh_parity}",
        "",
        "RD (decoder) is secondary and may be absent if the wall cap forced a skip.",
        "",
    ]
    if rd:
        lines.append(f"RD estimate {rd.get('observed')} Holm p={rd.get('p_holm')} pass={rd.get('pass_holm')}.")
    if per_target is not None and len(per_target):
        lines += ["", "## Per-target confirmatory ρ_Q", "", per_target.to_string(index=False), ""]
    write_text(out / "REPORT.md", "\n".join(lines) + "\n", force=True)
    write_text(
        out / "MANUSCRIPT_RECOMMENDATION.md",
        "\n".join(
            [
                "# Manuscript recommendation",
                "",
                "Do not automatically rewrite the manuscript.",
                f"Frozen label from this tree: `{decision['label']}`.",
                "Do not overwrite the ViT-B-only `quadratic_task_aligned_effect_only` label.",
                "Keep Q residual task-aligned energy distinct from generic K_H and from decoder residual energy.",
                "",
            ]
        ),
        force=True,
    )
