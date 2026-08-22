"""Phase 9: root-cause classification and audit outcome label."""

from __future__ import annotations

from typing import Any

from .config import KH_EXACT_ATOL


def classify_root_causes(parity: dict[str, Any], desi: dict[str, Any], controls: dict[str, Any]) -> list[str]:
    causes = []
    anc = parity.get("anchors") or {}
    if anc.get("adaptive_chose_new_hash_subset"):
        causes.append("anchor_selection_mismatch")
    if not parity.get("embedding", {}).get("same_artifact_both_pipelines", True):
        causes.append("embedding_or_preprocessing_mismatch")
    if parity.get("neighbours", {}).get("exact_id_agreement") is False:
        causes.append("neighbourhood_mismatch")
    if not parity.get("kh_identical", False):
        causes.append("curvature_estimator_mismatch")
    if parity.get("probe_quantity_mismatch"):
        causes.append("probe_label_alignment_failure")
    if controls.get("sign_reversal_control"):
        causes.append("control_specification_change")
    causes.append("inference_or_summary_bug")
    if not desi.get("proved"):
        causes.append("desi_alignment_unproven")
    causes.append("multiple_testing_calibration_problem")
    return causes


def audit_label(causes: list[str], parity: dict[str, Any], desi: dict[str, Any]) -> str:
    if "curvature_estimator_mismatch" in causes and not parity.get("kh_identical"):
        return "curvature_estimator_parity_failure"
    if "probe_label_alignment_failure" in causes:
        return "probe_label_alignment_failure"
    if "desi_alignment_unproven" in causes and not desi.get("proved"):
        return "dataset_alignment_unresolved"
    if "multiple_testing_calibration_problem" in causes:
        return "multiple_testing_results_revised"
    if parity.get("kh_identical") and not parity.get("probe_quantity_mismatch"):
        return "parity_restored_results_revised"
    return "adaptive_dataset_audit_unresolved"


def scale_blocks_complete(scale_pending: bool, discovery_parity: bool, joins_proven: bool) -> bool:
    return bool(discovery_parity and joins_proven and not scale_pending)
