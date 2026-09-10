"""Mechanical decision labels. Rules are frozen before inspecting real-data results."""

from __future__ import annotations

from typing import Any

import numpy as np


def _sig(rec: dict | None) -> bool:
    if not rec:
        return False
    if rec.get("ci_excludes_zero"):
        return True
    p = rec.get("p_holm", rec.get("p_mc", 1.0))
    return bool(np.isfinite(p) and float(p) <= 0.05)


def _pos(rec: dict | None) -> bool:
    return bool(rec and np.isfinite(rec.get("observed", np.nan)) and float(rec["observed"]) > 0)


def _neg(rec: dict | None) -> bool:
    return bool(rec and np.isfinite(rec.get("observed", np.nan)) and float(rec["observed"]) < 0)


def decide(
    *,
    parity_ok: bool,
    tests_ok: bool,
    joint_mseg: dict,
    per_model_mseg: dict[str, dict],
    vitb_dq: dict,
    hessian: dict,
    alignment: dict,
    probes: dict,
    org: dict,
) -> dict[str, Any]:
    if not parity_ok:
        return {
            "label": "curvature_component_audit_blocked",
            "reason": "parity_failed",
            "rules_version": 1,
        }
    if not tests_ok:
        return {
            "label": "curvature_component_audit_blocked",
            "reason": "unit_or_synthetic_failed",
            "rules_version": 1,
        }

    kh_j = joint_mseg.get("unique_KH_bar", {})
    tf_j = joint_mseg.get("unique_KTF_bar", {})
    kh_sig, tf_sig = _sig(kh_j), _sig(tf_j)
    n_pos_kh = int(joint_mseg.get("sign", {}).get("n_pos_KH", 0))
    n_pos_tf = int(joint_mseg.get("sign", {}).get("n_pos_TF", joint_mseg.get("sign", {}).get("n_pos_KTF", 0)))
    n_models = len(per_model_mseg) or 5
    loo = joint_mseg.get("leave_one_encoder_out", {})
    kh_loo_sign = [np.sign(v.get("unique_KH_bar", 0.0)) for v in loo.values()]
    tf_loo_sign = [np.sign(v.get("unique_KTF_bar", 0.0)) for v in loo.values()]
    loo_flip_kh = bool(kh_loo_sign) and (min(kh_loo_sign) * max(kh_loo_sign) < 0)
    loo_flip_tf = bool(tf_loo_sign) and (min(tf_loo_sign) * max(tf_loo_sign) < 0)

    vitb = per_model_mseg.get("vit_base", {})
    kh_vitb = _sig(vitb.get("unique_KH"))
    tf_vitb = _sig(vitb.get("unique_KTF"))

    dq_kh = _sig(vitb_dq.get("unique_KH"))
    dq_tf = _sig(vitb_dq.get("unique_KTF"))

    hess_tf = bool(hessian.get("primarily_traceless"))
    hess_iso = bool(hessian.get("primarily_isotropic"))
    hess_unstable = bool(hessian.get("component_unstable"))

    align_mean = bool(alignment.get("driven_by_mean"))
    align_tf = bool(alignment.get("driven_by_traceless"))
    align_inter = bool(alignment.get("driven_by_interaction"))

    iq_explains = bool(probes.get("iq_explains_uq2"))
    tq_explains = bool(probes.get("tq_explains_uq2"))
    bstf_explains_bs = bool(probes.get("bstf_explains_bs"))

    mean_predicts = kh_sig or kh_vitb or dq_kh
    tf_predicts = tf_sig or tf_vitb or dq_tf
    tf_accounts_structure = align_tf or tq_explains or (hess_tf and not hess_unstable)
    mean_accounts_structure = align_mean or iq_explains or hess_iso

    mixed_encoders = (n_pos_kh in range(1, n_models) and not kh_sig) or (
        n_pos_tf in range(1, n_models) and not tf_sig
    )
    hetero = mixed_encoders or loo_flip_kh or loo_flip_tf
    within_reliable = kh_vitb or tf_vitb or dq_kh or dq_tf or align_mean or align_tf

    kdir_sig = bool(org.get("kdir_mseg_sig"))
    components_separable = kh_sig != tf_sig or (mean_predicts and not tf_predicts) or (tf_predicts and not mean_predicts)

    label = "curvature_component_link_unresolved"
    reason = "mixed_or_unstable"

    if hess_unstable and not (mean_predicts or tf_predicts):
        label = "curvature_component_link_unresolved"
        reason = "hessian_component_below_stability_threshold"
    elif hetero and not (kh_sig or tf_sig) and within_reliable:
        label = "representation_specific_curvature_component_effects"
        reason = "encoder_heterogeneous_unique_associations"
    elif mean_predicts and tf_accounts_structure and (not tf_predicts or align_tf or tq_explains):
        label = "distinct_mean_and_traceless_predictive_roles"
        reason = "mean_predicts_error_or_gain_and_tf_accounts_for_hessian_or_tq"
    elif tf_predicts and mean_accounts_structure and not mean_predicts:
        label = "distinct_mean_and_traceless_predictive_roles"
        reason = "tf_predicts_error_and_mean_accounts_for_alignment_or_iq"
    elif mean_predicts and not tf_predicts:
        label = "mean_bending_specific_decodability_link"
        reason = "unique_KH_survives_unique_KTF_does_not"
    elif tf_predicts and not mean_predicts:
        label = "traceless_bending_specific_decodability_link"
        reason = "unique_KTF_survives_unique_KH_does_not"
    elif kdir_sig and not components_separable:
        label = "total_bending_magnitude_link"
        reason = "total_curvature_predicts_but_components_not_separable"
    elif hetero and within_reliable:
        label = "representation_specific_curvature_component_effects"
        reason = "no_common_unique_winner"
    else:
        label = "curvature_component_link_unresolved"
        reason = "reliability_collinearity_or_mixed_results"

    return {
        "label": label,
        "reason": reason,
        "rules_version": 1,
        "exploratory": True,
        "flags": {
            "unique_KH_mseg": kh_sig,
            "unique_KTF_mseg": tf_sig,
            "unique_KH_vitb": kh_vitb,
            "unique_KTF_vitb": tf_vitb,
            "unique_KH_delta_Q": dq_kh,
            "unique_KTF_delta_Q": dq_tf,
            "loo_flip_KH": loo_flip_kh,
            "loo_flip_KTF": loo_flip_tf,
            "n_pos_KH": n_pos_kh,
            "n_pos_KTF": n_pos_tf,
            "hessian_isotropic": hess_iso,
            "hessian_traceless": hess_tf,
            "align_mean": align_mean,
            "align_tf": align_tf,
            "align_interaction": align_inter,
            "iq_explains_uq2": iq_explains,
            "tq_explains_uq2": tq_explains,
            "bstf_explains_bs": bstf_explains_bs,
        },
    }
