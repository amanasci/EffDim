"""Mechanical labels. Thresholds frozen before seeing results."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DECISION_LABELS


def decide(
    *,
    tests_ok: bool,
    coverage_ok: bool,
    p1: dict | None,
    p2: dict | None,
    n_models: int,
    n_unreliable: int,
    hy_median_split_cos: float | None,
    shape_beats_mismatch: bool,
    resource_capped: bool,
    shuffle_survives: bool,
) -> dict[str, Any]:
    if not tests_ok or not coverage_ok or p1 is None or p2 is None:
        lab, reason = "cross_model_hessian_mismatch_unresolved", "tests_coverage_or_incomplete"
    elif resource_capped and n_models < 5:
        lab, reason = "cross_model_hessian_mismatch_unresolved", "resource_or_incomplete"
    elif n_unreliable >= 3:
        lab, reason = "decoder_geometry_unreliable_across_models", "seed_or_radial_fail"
    elif hy_median_split_cos is not None and hy_median_split_cos < 0.20:
        lab, reason = "label_hessian_unreliable", "split_half_cos"
    elif shuffle_survives:
        lab, reason = "cross_model_hessian_mismatch_unresolved", "shuffle_null_failed"
    else:
        p1p = bool(p1.get("pass_holm"))
        p2p = bool(p2.get("pass_holm"))
        p1_models = p1.get("per_model", {})
        p2_models = p2.get("per_model", {})
        n_p1_pos = int(sum(v > 0 for v in p1_models.values()))
        n_p2_neg = int(sum(v < 0 for v in p2_models.values()))
        loo1 = p1.get("loo_model_all_same_sign", True)
        loo2 = p2.get("loo_model_all_same_sign", True)
        vitb_ok = bool(p1_models.get("vit_base", 0) > 0 and p2_models.get("vit_base", 0) < 0)
        others_p1 = [v for m, v in p1_models.items() if m != "vit_base"]
        others_support = bool(others_p1) and (sum(v > 0 for v in others_p1) >= 3)
        if p1p and p2p and loo1 and loo2 and n_p1_pos >= 4 and n_p2_neg >= 4:
            lab, reason = "cross_model_hessian_mismatch_replication", "P1_P2_LOO_majority"
        elif p1p and p2p:
            lab, reason = "partial_cross_model_hessian_mismatch_replication", "P1_P2_hetero"
        elif p1p or p2p:
            lab, reason = "partial_cross_model_hessian_mismatch_replication", "one_primary"
        elif (not p1p) and (not p2p) and shape_beats_mismatch:
            lab, reason = "task_aligned_curvature_without_label_mismatch", "S_not_Delta"
        elif vitb_ok and not others_support:
            lab, reason = "vitb_specific_hessian_mismatch_effect", "vitb_only"
        elif abs(p1.get("observed", 0)) < 0.03 and abs(p2.get("observed", 0)) < 0.03:
            lab, reason = "hessian_mismatch_null", "aggregate_near_zero"
        else:
            lab, reason = "cross_model_hessian_mismatch_unresolved", "mixed"
    assert lab in DECISION_LABELS
    return {"label": lab, "reason": reason, "P1": p1, "P2": p2, "prior_labels_not_overwritten": True, "vitb_paper_not_overwritten": True}
