"""METHODS, REPORT, MANUSCRIPT_RECOMMENDATION. No manuscript edits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_util import write_text


def write_markdowns(
    out: Path,
    *,
    decision: dict[str, Any],
    seed_rel: dict[str, Any],
    primary: dict | None,
    parity: dict,
    runtime: dict,
    tests: dict,
    train_manifest: dict,
    extra: dict | None = None,
) -> None:
    extra = extra or {}
    methods = """# METHODS

Bounded real-data test of fixture-validated **pointwise sphere-residual decoder curvature**
(`D-residual`) against frozen ViT-B global and patch probe outcomes.

## Estimand

Raw decoder `F: R^{16} → R^{768}`. Differentiate through `F̃ = F/||F||`.

`H^S = (1/d) g^{ab} B^S_{ab}` with `B^S = P_{N,S} D²F̃` and `P_{N,S} = I - xx^T - P_T`.
Primary scalar `C_H = ||H^S||`. This is **not** intrinsic curvature. Intrinsic scalar
departure is `ΔScal_D = ||tr_g B^S||² - ||B^S||_g²` on a hash-selected 128-anchor subset.

Historical full curvature `H^E = (1/d) tr_g (I-P_T)D²F` on raw `decode` is a **control only**.

The local quadratic statistic `K_H^cross` is an empirical finite-patch quantity, not geometric
ground truth.

## Decoder

`PlainAutoEncoder` hidden (250,250,250) SiLU, AdamW lr=1e-3, weight decay 1e-4, batch 128,
400 epochs, no early stopping. Seeds `{0,1,2}` change initialization and minibatch order.
The 512 evaluation anchors are excluded from training; neighbouring objects may remain.
Label-blind: no labels, probe risks, or curvature correlations enter training or checkpointing.

## Inference

Frozen outcomes: local OOF `R_G^2`, `R_P^2`, `Δ_adapt = MSE_G - MSE_P` on the same valid objects.
Controls: log kNN radius, local label variance, evaluation count.
Rank-space Freedman–Lane, B_perm=10000, B_boot=2000, Holm over P1–P3.
P3 permutes the curvature residual once per replicate (joint in `R_G^2`,`R_P^2`).
"""
    write_text(out / "METHODS.md", methods)

    p = primary or {}
    lines = [
        "# REPORT",
        "",
        f"decision_label: `{decision.get('label')}`",
        f"seed_reliability_passed: {seed_rel.get('passed')}",
        f"median_rho_CH: {seed_rel.get('median_rho_CH')}",
        f"median_cos_HS: {seed_rel.get('median_cos_HS')}",
        f"tests: {tests.get('n_passed')}/{tests.get('n_tests')}",
        f"parity_ok: {parity.get('ok')}",
        f"runtime_s: {runtime.get('wall_s')}",
        f"n_new_decoders: {train_manifest.get('n_new_trained')}",
        "",
        "## Primary (D-residual consensus C_H, if seed gate passed)",
        "",
    ]
    for k in ("P1", "P2", "P3"):
        rec = p.get(k) or decision.get(k)
        if not rec:
            lines.append(f"{k}: not computed")
            continue
        lines.append(
            f"{k} {rec.get('name')}: ρ={rec.get('observed')} "
            f"CI95={rec.get('ci95')} p_mc={rec.get('p_mc')} p_holm={rec.get('p_holm')}"
        )
    lines += [
        "",
        f"signs_differ: {decision.get('signs_differ')}",
        f"rho_ctl_delta_adapt: {decision.get('rho_ctl_delta_adapt')}",
        "",
        "## Notes",
        "",
        extra.get("notes", "D-residual is the primary estimator. Historical D-full is a radial-removal control. Q is descriptive."),
        "",
    ]
    write_text(out / "REPORT.md", "\n".join(lines) + "\n")

    lab = decision.get("label")
    if lab == "decoder_residual_seed_unstable":
        rec = (
            "Do not report a confirmatory D-residual vs probe association. "
            "The pointwise residual field failed the frozen seed-reliability gate."
        )
        action = "do_not_include_as_main_result"
    elif lab == "stable_pointwise_residual_null":
        rec = (
            "D-residual is seed-stable but does not predict frozen G/P outcomes after correction. "
            "Do not recycle the Q-curvature narrative."
        )
        action = "do_not_include_as_main_result"
    elif lab == "stable_global_penalty_and_absolute_local_reversal":
        rec = (
            "Seed-stable D-residual C_H is negatively associated with global R_G^2 and positively "
            "with patch R_P^2; the paired contrast survives Holm. This is a D-residual result, not a Q result."
        )
        action = "consider_for_appendix_or_revision_after_review"
    elif lab == "stable_global_penalty_with_relative_local_adaptation":
        rec = (
            "Seed-stable D-residual tracks a global performance penalty with relative (not absolute) "
            "patch adaptation. Keep G vs P sign language exact; patch probes remain worse on average."
        )
        action = "consider_for_appendix"
    elif lab == "stable_positive_pointwise_decodability_link":
        rec = "Higher D-residual C_H associates with better probe performance without a global penalty."
        action = "consider_for_appendix"
    else:
        rec = "Leave the manuscript unchanged pending a cleaner D-residual result."
        action = "do_not_edit_manuscript"
    recmd = f"""# MANUSCRIPT_RECOMMENDATION

This experiment writes no manuscript edits.

- action: `{action}`
- label: `{lab}`

{rec}

Do not label historical Q statistics as validated curvature. Do not treat D-full as the primary estimator.
"""
    write_text(out / "MANUSCRIPT_RECOMMENDATION.md", recmd)
