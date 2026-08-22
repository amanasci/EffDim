"""Frozen constants. Not retuned per dataset after seeing associations."""

from __future__ import annotations

from typing import Any

# Completed trees — read-only.
PRESERVED = [
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_stable_tangent_dimension",
    "outputs/geometry/physics_order_stratified_geometry",
    "outputs/geometry/physics_implicit_normal_inverse",
    "outputs/geometry/physics_quadratic_predictive_dimension",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "outputs/geometry/physics_effdim_curvature_metrics",
    "outputs/geometry/physics_cross_model_probe_curvature_coverage",
]

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_QPD = "outputs/geometry/physics_quadratic_predictive_dimension"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_STD = "outputs/geometry/physics_stable_tangent_dimension"

FREEZE_HASH_EXPECTED = "d9e8616bcc9fe790"
PARITY_D16_RHO = -0.423283
PARITY_D12_RHO = -0.036315
PARITY_TOL = 0.03

DISCOVERY_DATASET = "physics_vit_base"
DISCOVERY_LABEL = "mag_r_desi"
PRIMARY_ENCODER_FAMILY = "vit_base"

# Neighbourhood scale rule (sample-size only; frozen before labels).
K_PRESET = (256, 512, 768, 1024, 1536, 2048)
K_FRAC_OF_N = 0.125  # primary k = largest preset with k <= this fraction of n

# Minimum valid labelled anchors. Spearman |ρ|=0.35, α=0.05, ~80% power ≈ 64.
# Rank-sweep used 512; 64 is the underpowered floor, not the analysis target.
MIN_VALID_ANCHORS = 64
TARGET_ANCHORS = 512

# Held-out linear variance thresholds. Do not extrapolate missing crossings.
TAU_GRID = (0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.95)
VAR_COMPARE_GRID = tuple(round(0.70 + 0.025 * i, 3) for i in range(11))

# Spectral expansion. Do not stop at 20 because that was the discovery carrier.
SPECTRAL_START = 32
SPECTRAL_STEP = 32
SPECTRAL_HARD_CAP = 128

# Shared-core confounders (ViT-B discovery triple).
SHARED_CORE_CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

# Practical plateau (same as quadratic predictive dimension).
PLATEAU_REL_TOL = 0.02
DELTA_PRACTICAL = 0.004
N_LOOKAHEAD = 3
DF_COLLAPSE = 0.12

# Noise-tail narrower-interval rule (synthetic-calibrated, frozen).
# If a high-τ crossing sits more than this many ranks past the linear plateau
# and incremental R² is below DELTA_PRACTICAL, keep the full spectral result
# but allow a narrower primary curvature interval.
NOISE_TAIL_RANKS_PAST_PLATEAU = 4

# Curvature reliability (same gates as the rank sweep).
R_H_FAIL = 0.20
VALID_FRAC_FAIL = 0.85
DF_RATIO_WARN = 0.85

N_PERM = 10000
N_BOOT = 2000
N_SCALE_ANCHORS = 128
SEED = 0
HASH_PREFIX = "adcp"

PRIMARY_LABELS = (
    "cross_dataset_curvature_transition_replicated",
    "curvature_probe_association_replicates_without_common_transition",
    "dataset_specific_curvature_probe_associations",
    "curvature_probe_replication_not_supported",
    "adaptive_curvature_sweeps_underidentified",
    "cross_dataset_curvature_replication_unresolved",
)

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "alpha": 0.05,
    "min_valid_anchors": MIN_VALID_ANCHORS,
    "target_anchors": TARGET_ANCHORS,
    "k_frac": K_FRAC_OF_N,
    "tau_grid": list(TAU_GRID),
    "plateau_rel_tol": PLATEAU_REL_TOL,
    "delta_practical": DELTA_PRACTICAL,
    "n_lookahead": N_LOOKAHEAD,
    "df_collapse": DF_COLLAPSE,
    "noise_tail_ranks": NOISE_TAIL_RANKS_PAST_PLATEAU,
    "r_h_fail": R_H_FAIL,
    "valid_frac_fail": VALID_FRAC_FAIL,
    "n_perm": N_PERM,
    "n_boot": N_BOOT,
    "ridge_n_grid": 11,
    "u_bound_q": 0.99,
    "gn_max_iter": 8,
    "gn_damp": 1e-4,
    "n_inner_cp": 96,
}
