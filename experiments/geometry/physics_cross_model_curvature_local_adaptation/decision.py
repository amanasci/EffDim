"""Assign exactly one frozen decision label. Thresholds are not tuned after outcomes."""

from __future__ import annotations

from typing import Any

from .config import DECISION_LABELS, POSITIVE_CONTROL


def _pos(stat: dict | None) -> bool:
    if not stat:
        return False
    obs = float(stat.get("observed", float("nan")))
    p = float(stat.get("p_holm", stat.get("p_mc", 1.0)))
    ci_ok = bool(stat.get("ci_excludes_zero"))
    return bool(obs > 0 and (p <= 0.05 or ci_ok))


def decide(
    *,
    inventory: dict,
    reliable_models: list[str],
    per_model: dict[str, dict],
    aggregate: dict,
    calibration: dict[str, dict],
    parity_ok: bool,
) -> dict[str, Any]:
    n_eligible = int(inventory.get("n_eligible", 0))
    diversity_flag = bool(inventory.get("insufficient_model_diversity"))
    new_reliable = [m for m in reliable_models if m != POSITIVE_CONTROL]
    both = [
        m
        for m in reliable_models
        if _pos(per_model.get(m, {}).get("C_G")) and _pos(per_model.get(m, {}).get("C_A"))
    ]
    new_both = [m for m in both if m != POSITIVE_CONTROL]
    cg_models = [m for m in reliable_models if _pos(per_model.get(m, {}).get("C_G"))]
    ca_models = [m for m in reliable_models if _pos(per_model.get(m, {}).get("C_A"))]
    n_rel = len(reliable_models)
    majority_cg = n_rel > 0 and len(cg_models) > n_rel / 2
    majority_ca = n_rel > 0 and len(ca_models) > n_rel / 2
    agg_cg = _pos(aggregate.get("C_G_bar"))
    agg_ca = _pos(aggregate.get("C_A_bar"))

    cal_only = False
    n_dir = 0
    for m in reliable_models:
        cal = calibration.get(m, {})
        da = cal.get("delta_adapt", {}).get("controlled", float("nan"))
        dff = cal.get("delta_affine", {}).get("controlled", float("nan"))
        dd = cal.get("delta_direction", {}).get("controlled", float("nan"))
        if dd == dd and dd > 0:
            n_dir += 1
        if da == da and dff == dff and dff >= da - 0.02 and not (dd == dd and dd > 0):
            cal_only = True
    direction_exceeds_cal = n_dir > n_rel / 2 if n_rel else False

    if not parity_ok:
        label = "cross_model_result_unresolved"
        reason = "vitb_parity_failed"
    elif diversity_flag or n_eligible < 3:
        label = "insufficient_model_diversity"
        reason = f"n_eligible={n_eligible}"
    elif n_rel <= 1 and POSITIVE_CONTROL in reliable_models:
        # only the control is geometrically reliable
        if POSITIVE_CONTROL in both:
            label = "geometry_unreliable_across_models"
            reason = "only_vit_base_reliable_geometry"
        else:
            label = "geometry_unreliable_across_models"
            reason = "insufficient_reliable_geometry"
    elif len(new_reliable) == 0:
        label = "geometry_unreliable_across_models"
        reason = "no_non_control_reliable_geometry"
    elif (
        len(new_both) >= 2
        and agg_cg
        and agg_ca
        and majority_cg
        and majority_ca
        and direction_exceeds_cal
        and not cal_only
    ):
        label = "cross_model_global_penalty_and_local_adaptation"
        reason = "aggregate_and_new_models_support_both"
    elif majority_cg and agg_cg and not majority_ca:
        label = "global_penalty_replicates_local_adaptation_model_dependent"
        reason = "C_G_generalizes_C_A_heterogeneous"
    elif majority_ca and agg_ca and not majority_cg:
        label = "local_adaptation_replicates_global_penalty_model_dependent"
        reason = "C_A_generalizes_C_G_heterogeneous"
    elif POSITIVE_CONTROL in both and len(new_both) == 0:
        label = "representation_specific_effect"
        reason = "pattern_confined_to_vit_base"
    else:
        label = "cross_model_result_unresolved"
        reason = "mixed_without_stronger_label"

    assert label in DECISION_LABELS
    return {
        "label": label,
        "reason": reason,
        "reliable_models": reliable_models,
        "new_reliable_models": new_reliable,
        "models_with_both": both,
        "new_models_with_both": new_both,
        "models_C_G": cg_models,
        "models_C_A": ca_models,
        "majority_C_G": majority_cg,
        "majority_C_A": majority_ca,
        "aggregate_C_G": agg_cg,
        "aggregate_C_A": agg_ca,
        "direction_exceeds_calibration": direction_exceeds_cal,
        "calibration_only_flag": cal_only,
        "n_eligible": n_eligible,
        "n_reliable": n_rel,
    }
