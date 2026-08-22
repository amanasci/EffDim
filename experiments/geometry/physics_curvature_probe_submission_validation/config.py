"""Frozen constants. Chart positions are geometry-only and not correlation-maximised."""

from __future__ import annotations

from typing import Any

PRESERVED = [
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_quadratic_predictive_dimension",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_adaptive_dataset_curvature_probe_audit",
    "outputs/geometry/physics_adaptive_dataset_curvature_probe",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "outputs/geometry/physics_effdim_curvature_metrics",
    "outputs/geometry/physics_cross_model_probe_curvature_coverage",
]

SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_QPD = "outputs/geometry/physics_quadratic_predictive_dimension"
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_AUDIT = "outputs/geometry/physics_adaptive_dataset_curvature_probe_audit"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"

MODEL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
PRIMARY_K = 2048
PARITY_RANKS = (12, 16, 20)
DS = tuple(range(8, 21))
# Geometry-only positions in the held-out variance/risk-supported family (τ=0.80 at 12, τ=0.85 at 20).
# Middle is the midpoint of that family, not the |ρ| peak.
PREDECLARED_D = {"lower": 12, "middle": 16, "upper": 20}
SCALE_KS = (512, 1024, 1536, 2048)
N_SCALE_ANCHORS = 128
N_PERM = 10000
N_BOOT = 2000
SEED = 0
ALPHA = 0.05
KH_ATOL = 1e-8
PARITY_ATOL = 1e-6
FROZEN_RAW = {12: -0.038426, 16: -0.412430, 20: -0.392251}
FROZEN_CTL = {12: 0.142990, 16: -0.240484, 20: -0.233325}
FROZEN_DELTA_20_12 = FROZEN_CTL[20] - FROZEN_CTL[12]
CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")
R_H_FAIL = 0.20
DECISION_LABELS = (
    "submission_claim_supported",
    "claim_supported_but_scale_dependent",
    "local_r2_denominator_driven",
    "submission_claim_unresolved",
)

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "alpha": ALPHA,
    "parity_atol": PARITY_ATOL,
    "kh_atol": KH_ATOL,
    "r_h_fail": R_H_FAIL,
    "min_error_abs_rho": 0.10,
    "scale_sign_frac": 0.75,
}
