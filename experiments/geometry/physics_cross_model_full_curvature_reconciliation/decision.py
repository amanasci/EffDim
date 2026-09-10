"""Assign exactly one new audit label. Prior trace label is not overwritten."""

from __future__ import annotations

from typing import Any

from .config import DECISION_LABELS, TRACE_DECISION_LABEL


def _supported(stat: dict | None, *, want: str) -> bool:
    if not stat:
        return False
    obs = float(stat.get("observed", float("nan")))
    if not np_ok(obs):
        return False
    ci_ok = bool(stat.get("ci_excludes_zero"))
    p = float(stat.get("p_mc", 1.0))
    sign_ok = (obs < 0) if want == "neg" else (obs > 0)
    return bool(sign_ok and (ci_ok or p <= 0.05))


def np_ok(x: float) -> bool:
    return x == x


def _loo_sign_flip(agg: dict, key: str) -> bool:
    obs = float(agg.get(key, {}).get("observed", float("nan")))
    if not np_ok(obs) or obs == 0:
        return False
    loo = agg.get("leave_one_encoder_out") or {}
    for rec in loo.values():
        v = rec.get(key)
        if v is None:
            continue
        if float(v) * obs < 0:
            return True
    return False


def decide(
    *,
    audit_ok: bool,
    tests_ok: bool,
    parity_ok: bool,
    historical_parity_ok: bool,
    full_agg: dict,
    kh_agg: dict,
    n_models: int,
) -> dict[str, Any]:
    if not audit_ok:
        label = "full_curvature_reconciliation_blocked"
        reason = "historical_definition_or_artifact_unrecovered"
    elif not tests_ok or not parity_ok or not historical_parity_ok:
        label = "full_curvature_reconciliation_unresolved"
        reason = "tests_or_parity_failed"
    else:
        cr2 = full_agg.get("C_R2_bar") or {}
        cg = full_agg.get("C_G_bar") or {}
        cr2p = full_agg.get("C_R2P_bar") or {}
        cp = full_agg.get("C_P_bar") or {}
        ca = full_agg.get("C_A_bar") or {}
        a = full_agg.get("A_bar") or {}
        sign = full_agg.get("sign") or {}
        n_neg_r2 = int(sign.get("C_R2_bar", {}).get("n_neg", 0))
        n_pos_g = int(sign.get("C_G_bar", {}).get("n_pos", 0))
        n_pos_r2p = int(sign.get("C_R2P_bar", {}).get("n_pos", 0))
        n_neg_p = int(sign.get("C_P_bar", {}).get("n_neg", 0))

        global_penalty = _supported(cr2, want="neg") or _supported(cg, want="pos")
        majority_global = (n_neg_r2 >= 4) or (n_pos_g >= 4)
        if not global_penalty and majority_global:
            global_penalty = True
        patch_reverse = _supported(cr2p, want="pos") or _supported(cp, want="neg")
        majority_patch = (n_pos_r2p >= 4) or (n_neg_p >= 4)
        if not patch_reverse and majority_patch:
            patch_reverse = True
        adapt = _supported(ca, want="pos")
        a_ok = _supported(a, want="pos")
        partial = _loo_sign_flip(full_agg, "C_R2_bar") or _loo_sign_flip(full_agg, "C_G_bar")
        if global_penalty and not majority_global:
            partial = True

        kh_cg = kh_agg.get("C_G_bar") or {}
        kh_global = _supported(kh_cg, want="pos") or (
            int((kh_agg.get("sign") or {}).get("C_G_bar", {}).get("n_pos", 0)) >= 4
        )
        conclusions_differ = bool(global_penalty) != bool(kh_global)
        remembered_recovered = bool(global_penalty and majority_global and not partial)

        if global_penalty and patch_reverse and a_ok:
            label = "full_curvature_global_to_local_sign_reversal"
            reason = "full_curvature_global_penalty_and_absolute_patch_reversal"
        elif global_penalty and adapt and not patch_reverse:
            label = "full_curvature_cross_model_global_penalty_with_relative_adaptation"
            reason = "full_curvature_global_penalty_relative_adapt_only"
        elif global_penalty and not adapt and not patch_reverse and not partial:
            label = "full_curvature_cross_model_global_penalty_only"
            reason = "full_curvature_global_penalty_without_local_claim"
        elif global_penalty and partial:
            label = "full_curvature_partial_cross_model_replication"
            reason = "full_curvature_global_signal_model_sensitive"
        elif conclusions_differ and not remembered_recovered:
            label = "trace_full_curvature_estimand_divergence"
            reason = "trace_and_full_disagree_remembered_result_not_recovered"
        else:
            label = "full_curvature_reconciliation_unresolved"
            reason = "no_stable_cross_model_full_curvature_claim"

    assert label in DECISION_LABELS
    return {
        "label": label,
        "reason": reason,
        "prior_trace_label_untouched": TRACE_DECISION_LABEL,
        "prior_trace_label_scope": "K_H_cross mean-curvature-trace estimand only",
        "new_label_scope": "historical K_dir_cross full sphere-normal curvature",
        "n_models": n_models,
        "audit_ok": audit_ok,
        "tests_ok": tests_ok,
        "parity_ok": parity_ok,
        "historical_parity_ok": historical_parity_ok,
        "questions": {
            "1_historical_full_vs_global": "Does K_dir_cross reproduce a cross-model global-performance penalty?",
            "2_full_tensor_KB": "Does K_B_cross support the same conclusion?",
            "3_absolute_patch_reverses": "Does absolute patch performance reverse sign vs global?",
            "4_relative_adaptation": "Does relative adaptation increase with full curvature?",
            "5_trace_label_specific": "Is representation_specific_effect specific to the trace statistic?",
            "6_traceless_share": "How much of the full curvature is traceless bending?",
        },
    }
