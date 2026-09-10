"""Frozen constants. Do not retune gates after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass

OUT_REL = "outputs/geometry/physics_q_geometry_resampling_stability"
WALL_S = 45 * 60
RESERVE_WRITE_S = 150.0
DISK_CAP_BYTES = 2 * 1024**3
PILOT_REPS = 4
TARGET_REPS = 32
MAX_REPS = 64
MIN_VALID_REPS = 16
PROJECTION_BUDGET_S = 40 * 60

MODEL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
D = 16
K = 2048
K_PRIME = 2 * int((0.8 * 2048) // 2)  # 1638
HALF_PRIME = K_PRIME // 2  # 819
KEEP_FRAC = 0.80
N_ANCHORS = 512
N_SPLITS_PRODUCTION = 5
PRODUCTION_SEED = 0
EXPERIMENT_SEED = 20260911
N_BOOT_COMBINED = 2000
COMBINED_SEED = 20260911
ANCHOR_BOOT_SEED = 0
N_PARITY_REFIT = 8

CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")
PARITY_R2 = -0.240
PARITY_MSE_G = 0.227
PARITY_DMSE = 0.153
PARITY_ATOL = 0.005

GATE_SIGN_FRAC = 0.90
GATE_MEDIAN_TOL = 0.05

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_FCR = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_LPA = "outputs/geometry/physics_local_probe_adaptation"
SOURCE_QLCA = "outputs/geometry/physics_quadratic_label_chart_alignment"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"

PRESERVED_LABELS = (
    "quadratic_chart_link_unresolved",
    "neither_estimator_validated",
    "q_moderately_informative_sampling_dependent_statistic",
)


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cpu"
    n_workers: int = 12
    n_parity_refit: int = N_PARITY_REFIT
    force_reps: int | None = None
