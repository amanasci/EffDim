"""Frozen constants for the bounded instrument failure-localization diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

PRESERVED = (
    "experiments/geometry/known_curvature_point_patch_fixture_audit",
    "outputs/geometry/known_curvature_point_patch_fixture_audit",
    "experiments/geometry/physics_cross_model_full_curvature_reconciliation",
    "experiments/geometry/physics_cross_model_curvature_local_adaptation",
    "experiments/geometry/physics_quadratic_label_chart_alignment",
    "experiments/geometry/physics_quadratic_label_chart_alignment_audit",
    "experiments/geometry/physics_curvature_component_predictive_decomposition",
    "experiments/geometry/physics_local_probe_adaptation",
    "experiments/geometry/physics_local_probe_adaptation_audit",
    "experiments/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_cross_model_full_curvature_reconciliation",
    "outputs/geometry/physics_cross_model_curvature_local_adaptation",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment_audit",
    "outputs/geometry/physics_curvature_component_predictive_decomposition",
    "outputs/geometry/physics_ae_local_patch_scale_match",
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "outputs/geometry/physics_effdim_curvature_metrics",
    "submissions/neurreps_2026",
    "submissions/neurreps_2026_lpa_revision",
    "submissions/ml4ps_2026",
    "papers/curvature_photometric_decoding",
    "papers/interpscience_semantic_fields",
)

FROZEN_HASH_PATHS = (
    "experiments/geometry/physics_activation_atlas/nested_dimension_curvature.py",
    "experiments/geometry/physics_activation_atlas/full_curvature_audit.py",
    "experiments/geometry/physics_activation_atlas/sphere_normal_quadratic.py",
    "experiments/geometry/known_curvature_point_patch_fixture_audit/estimator_q.py",
    "experiments/geometry/known_curvature_point_patch_fixture_audit/estimator_d.py",
    "experiments/geometry/known_curvature_point_patch_fixture_audit/geometry.py",
    "experiments/geometry/known_curvature_point_patch_fixture_audit/oracle.py",
    "notebooks/pu_manifold/decoder_curvature.py",
)

OUT_REL = "outputs/geometry/known_curvature_instrument_failure_localization"
AUDIT_OUT_REL = "outputs/geometry/known_curvature_point_patch_fixture_audit"

D_LAT = 16
D_AMB = 768
Q_FEATURES = D_LAT * (D_LAT + 1) // 2  # 136
N_ANCHORS = 64
PRIMARY_N = 16384
K_VALUES = (2048, 512)
WALL_S = 45 * 60
RESERVE_WRITE_S = 90.0
FIXTURE_SEED = 20260907
ANCHOR_HASH_SEED = 20260907
SPLIT_SEED = 20260813
PINV_RCOND_MULT = 1.0  # machine-eps multiplier for SVD cutoff: rcond = max(n,q)*eps
RIDGE_VAL_FRACTION = 0.20
N_BOOT = 50
N_FD_ANCHORS = 16
FD_STEP = 1e-5

# Frozen before inspecting this diagnostic's scores.
Q1_REL_OK = 0.25
Q1_RHO_OK = 0.70
MATERIAL_REL_DELTA = 0.15
MATERIAL_RHO_DROP = 0.20
F0_FALSE_MAX = 1e-3
F1_TF_FRAC_MAX = 0.10
F2_H_FRAC_MAX = 0.10
SEED_SPEARMAN_OK = 0.50
T2_VS_T1_REL_STRONG = 0.30

DECISION_LABELS = (
    "quadratic_oracle_or_convention_mismatch",
    "quadratic_tangent_estimation_failure",
    "quadratic_ridge_attenuation_failure",
    "quadratic_split_variance_failure",
    "quadratic_patch_model_bias",
    "decoder_learned_surface_hessian_nonidentifiability",
    "multiple_instrument_failure_sources",
    "bounded_failure_localization_unresolved",
)

PRIMARY_CELLS = (
    ("F0", "S0", "N0", 0.0, "A_F0_S0_N0_n16384_k2048_eta0.00"),
    ("F1", "S0", "N0", 0.0, "A_F1_S0_N0_n16384_k2048_eta0.00"),
    ("F2", "S0", "N0", 0.0, "A_F2_S0_N0_n16384_k2048_eta0.00"),
    ("F4", "S0", "N0", 0.0, "A_F4_S0_N0_n16384_k2048_eta0.00"),
)
STRESS_CELLS = (
    ("F4", "S1", "N0", 0.0, "B_F4_S1_N0_n16384_k2048_eta0.00"),
    ("F4", "S0", "N2", 0.10, "C_F4_S0_N2_n16384_k2048_eta0.10"),
)


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    audit_dir: str = AUDIT_OUT_REL
    device: str = "cpu"
    n_workers: int = 8
    wall_s: float = WALL_S
    force: bool = False
