"""Mechanical decision. Thresholds frozen before seeing results."""

from __future__ import annotations

from typing import Any

from .config import DECISION_LABELS


def decide(
    *,
    tests_ok: bool,
    parity_ok: bool,
    coverage_ok: bool,
    fixtures_ok: bool,
    p1: dict | None,
    p2: dict | None,
    signs_D: list[float],
    signs_Q: list[float],
    agree: list[float],
    historical_strong: bool,
    confirmatory_weak: bool,
    radial_dominates: bool,
    resource_capped: bool,
) -> dict[str, Any]:
    if not tests_ok or not coverage_ok or resource_capped or p1 is None or p2 is None:
        lab, reason = "task_aligned_curvature_unresolved", "tests_coverage_or_incomplete"
    elif radial_dominates:
        lab, reason = "radial_or_prediction_coupling_dominates", "radial_or_prediction"
    elif historical_strong and confirmatory_weak:
        lab, reason = "historical_effect_not_leakage_safe", "H_strong_C_weak"
    else:
        p1p = bool(p1.get("pass_holm"))
        p2p = bool(p2.get("pass_holm"))
        all_d = bool(signs_D) and all(s > 0 for s in signs_D)
        all_q = bool(signs_Q) and all(s > 0 for s in signs_Q)
        agree_pos = bool(agree) and all(a > 0 for a in agree)
        hetero = (not all_d) or (not all_q) or (not agree_pos)
        if p1p and p2p and all_d and all_q and agree_pos and fixtures_ok:
            lab, reason = "cross_instrument_task_aligned_curvature_supported", "P1_P2_signs_agree"
        elif p1p and p2p:
            lab, reason = "task_aligned_curvature_supported_with_target_heterogeneity", "P1_P2_hetero"
        elif p1p and not p2p:
            lab, reason = "decoder_task_aligned_effect_only", "P1_only"
        elif p2p and not p1p:
            lab, reason = "quadratic_task_aligned_effect_only", "P2_only"
        else:
            lab, reason = "task_aligned_curvature_unresolved", "neither_primary"
        _ = hetero
    assert lab in DECISION_LABELS
    return {
        "label": lab,
        "reason": reason,
        "P1": p1,
        "P2": p2,
        "vitb_only": True,
        "historical_not_confirmatory": True,
        "prior_labels_not_overwritten": True,
    }
