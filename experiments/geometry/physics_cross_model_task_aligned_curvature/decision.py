"""Mechanical cross-model labels. Thresholds frozen before seeing results."""

from __future__ import annotations

from typing import Any

from .config import DECISION_LABELS


def decide(
    *,
    tests_ok: bool,
    parity_ok: bool,
    coverage_ok: bool,
    r1: dict | None,
    rd: dict | None,
    vitb_q_bar: float | None,
    n_rep_q: int,
    resource_capped: bool,
    q_ran: bool,
) -> dict[str, Any]:
    if not tests_ok or not coverage_ok or not q_ran or r1 is None:
        lab, reason = "task_aligned_cross_model_unresolved", "tests_coverage_or_incomplete"
    elif resource_capped and n_rep_q < 3:
        lab, reason = "task_aligned_cross_model_unresolved", "resource_or_incomplete"
    else:
        r1p = bool(r1.get("pass_holm"))
        n_pos = int(r1.get("n_models_positive", 0))
        rdp = bool(rd and rd.get("pass_holm"))
        vitb_pos = bool(vitb_q_bar is not None and vitb_q_bar > 0)
        if r1p and rdp and n_pos >= 3:
            lab, reason = "q_and_d_task_aligned_replicate", "R1_RD_majority"
        elif r1p and n_pos >= 3:
            lab, reason = "q_task_aligned_replicates_across_models", "R1_majority"
        elif r1p:
            lab, reason = "q_task_aligned_heterogeneous_across_models", "R1_pass_minority_sign"
        elif (not r1p) and rdp:
            lab, reason = "d_task_aligned_replicates_q_does_not", "RD_only"
        elif vitb_pos and not r1p:
            lab, reason = "q_task_aligned_vitb_specific", "vitb_positive_R1_fail"
        else:
            lab, reason = "task_aligned_cross_model_unresolved", "neither_primary"
    assert lab in DECISION_LABELS
    return {
        "label": lab,
        "reason": reason,
        "R1": r1,
        "RD": rd,
        "vitb_q_bar": vitb_q_bar,
        "n_replication_models_with_Q": int(n_rep_q),
        "prior_labels_not_overwritten": True,
        "vitb_excluded_from_R1": True,
    }
