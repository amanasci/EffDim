"""Frozen audit constants. Not retuned after seeing associations."""

from __future__ import annotations

from typing import Any

PRESERVED = [
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_stable_tangent_dimension",
    "outputs/geometry/physics_order_stratified_geometry",
    "outputs/geometry/physics_implicit_normal_inverse",
    "outputs/geometry/physics_quadratic_predictive_dimension",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_adaptive_dataset_curvature_probe",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "outputs/geometry/physics_effdim_curvature_metrics",
    "outputs/geometry/physics_cross_model_probe_curvature_coverage",
]

SOURCE_ADCP = "outputs/geometry/physics_adaptive_dataset_curvature_probe"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_QPD = "outputs/geometry/physics_quadratic_predictive_dimension"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"

DISCOVERY_DATASET = "physics_vit_base"
DISCOVERY_LABEL = "mag_r_desi"
DISCOVERY_PROBE_COLUMN = "local_r2"
PARITY_RANKS = (12, 16, 20)
FROZEN_D80 = 12
FROZEN_D85 = 20

# Frozen discovery curve (complete-case K_H vs local_r2, rank sweep).
FROZEN_RAW = {12: -0.038426, 16: -0.412430, 20: -0.392251}
FROZEN_CTL = {12: 0.142990, 16: -0.240484, 20: -0.233325}

SHARED_CORE_CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")
MIN_VALID_ANCHORS = 64
TARGET_ANCHORS = 512
N_PERM = 10000
N_BOOT = 2000
SEED = 0
R_H_FAIL = 0.20
R_H_STRICT = (0.40, 0.50, 0.60)
ALPHA = 0.05
KH_EXACT_ATOL = 1e-12
KH_TOL = 1e-8

AUDIT_LABELS = (
    "parity_restored_results_revised",
    "parity_restored_no_material_change",
    "probe_label_alignment_failure",
    "curvature_estimator_parity_failure",
    "dataset_alignment_unresolved",
    "multiple_testing_results_revised",
    "adaptive_dataset_audit_unresolved",
)

ROOT_CAUSES = (
    "anchor_selection_mismatch",
    "embedding_or_preprocessing_mismatch",
    "neighbourhood_mismatch",
    "curvature_estimator_mismatch",
    "probe_label_alignment_failure",
    "control_specification_change",
    "inference_or_summary_bug",
    "desi_alignment_unproven",
    "multiple_testing_calibration_problem",
    "unresolved",
)

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "alpha": ALPHA,
    "min_valid_anchors": MIN_VALID_ANCHORS,
    "n_perm": N_PERM,
    "n_boot": N_BOOT,
    "r_h_fail": R_H_FAIL,
    "kh_exact_atol": KH_EXACT_ATOL,
    "kh_tol": KH_TOL,
}
