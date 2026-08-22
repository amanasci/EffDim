"""Assign final audit label and manuscript recommendation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DECISION_LABELS


def decide(
    *,
    parity_ok: bool,
    primary_rho: float,
    primary_p: float,
    primary_ci_excludes: bool,
    mean_dm: float,
    delta_rho_mse: dict,
    ctrl_models: Any,
    beats_c: bool,
    tangent_ok: bool,
    pathway_supports: bool,
    pathway_reliable: bool,
    shuffle: dict,
    insample_artifact: bool,
) -> dict[str, Any]:
    align_survives = True
    if len(ctrl_models):
        base = ctrl_models[ctrl_models.model == "model_A_baseline"].iloc[0]["controlled"]
        comb = ctrl_models[ctrl_models.model == "model_D_both"].iloc[0]["controlled"]
        if np.isfinite(base) and np.isfinite(comb):
            align_survives = bool(comb > 0 and comb >= 0.5 * base)

    delta_sig = bool(
        np.isfinite(delta_rho_mse.get("delta_rho", float("nan")))
        and delta_rho_mse.get("ci95_lo", 0) > 0
    )

    primary_ok = bool(primary_ci_excludes and primary_rho > 0 and primary_p <= 0.05)
    shuffle_ok = bool(not shuffle.get("skipped") and shuffle.get("pass", False))
    direction_pathway = bool(pathway_supports and pathway_reliable)

    if insample_artifact and not primary_ok:
        label = "in_sample_patch_probe_artifact"
    elif not parity_ok:
        label = "local_probe_result_unresolved"
    elif (
        primary_ok
        and align_survives
        and beats_c
        and tangent_ok
        and direction_pathway
        and delta_sig
        and shuffle_ok
    ):
        label = "curvature_predicts_local_direction_adaptation"
    elif primary_ok and align_survives and beats_c and tangent_ok and shuffle_ok:
        label = "curvature_predicts_relative_local_adaptation"
    elif primary_ok and not beats_c:
        label = "curvature_predicts_local_calibration_gain"
    elif primary_ok:
        label = "curvature_predicts_relative_local_adaptation"
    else:
        label = "local_probe_result_unresolved"

    assert label in DECISION_LABELS
    ms = manuscript_action(label)
    return {
        "label": label,
        "manuscript": ms,
        "mean_dMSE_GP": mean_dm,
        "rho_dMSE": primary_rho,
        "delta_rho_mse_GP": delta_rho_mse,
        "checks": {
            "parity_ok": parity_ok,
            "primary_ci_ok": primary_ci_excludes,
            "primary_p_ok": primary_p <= 0.05,
            "align_survives": align_survives,
            "beats_C": beats_c,
            "tangent_ok": tangent_ok,
            "delta_rho_GP_excludes_zero": delta_sig,
            "shuffle_ok": shuffle_ok,
            "pathway_reliable": pathway_reliable,
            "pathway_supports": direction_pathway,
        },
        "alignment_models": {r["model"]: r for _, r in ctrl_models.iterrows()} if len(ctrl_models) else {},
    }


def manuscript_action(label: str) -> dict[str, str]:
    if label == "curvature_predicts_local_direction_adaptation":
        return {
            "action": "include_as_main_result",
            "paragraph": (
                "Patch-specific probes were worse than the frozen global probe on average, but their relative "
                "disadvantage decreased with sphere-normal curvature: K_H^cross was positively associated with "
                "MSE_G - MSE_P. This association persisted relative to a locally calibrated global prediction and "
                "under tangent-coordinate probing, with alignment-controlled and end-to-end shuffle checks supporting "
                "local direction adaptation in high-curvature regions."
            ),
        }
    if label == "curvature_predicts_relative_local_adaptation":
        return {
            "action": "include_as_main_result",
            "paragraph": (
                "Patch-specific probes were worse than the frozen global probe on average, but their relative "
                "disadvantage decreased with sphere-normal curvature: $K_H^{\\mathrm{cross}}$ was positively "
                "associated with $\\mathrm{MSE}_G-\\mathrm{MSE}_P$. This association persisted relative to a "
                "locally calibrated global prediction and under tangent-coordinate probing, indicating that "
                "high-curvature regions benefit disproportionately from local adaptation. Because the analysis "
                "was post hoc and did not produce a positive average patch advantage, we interpret it as relative "
                "adaptation rather than uniformly superior local decodability."
            ),
        }
    if label == "curvature_predicts_local_calibration_gain":
        return {
            "action": "include_as_exploratory_appendix",
            "paragraph": "Curvature-linked gains appear explained by local calibration of the global probe rather than patch-specific directions.",
        }
    return {
        "action": "do_not_include",
        "paragraph": "The local-probe adaptation audit did not support main-text inclusion.",
    }
