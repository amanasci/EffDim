"""Frozen constants. Do not retune after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass, field

OUT_REL = "outputs/geometry/physics_cross_model_task_aligned_curvature"
WALL_S = 90 * 60
PROJECTED_CAP_S = 85 * 60
RESERVE_WRITE_S = 180.0

REFERENCE = "vit_base"
REPLICATION = ("dinov3", "clip_base", "convnext_base", "vit_large")
ALL_MODELS = (REFERENCE, *REPLICATION)

TARGETS = ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass")
D_LAT = 16
K = 2048
N_ANCHORS = 512
N_SMOKE = 8
N_FOLDS = 5
PROBE_ALPHA = 100.0
TRAIN_FRAC = 0.60
MIN_EVAL = 128
COVERAGE_FAIL_FRAC = 0.10
SPLIT_SALT = "task_aligned_curvature_v1"

DECODER_SEEDS = (0, 1, 2)
PROTOCOL = "reproduction_plain_ae_400"
SEED_RHO_GATE = 0.70

CONTROLS = ("log_knn_radius", "local_eval_label_variance", "local_evaluation_count")

N_PERM = 10000
N_BOOT = 2000
INFER_SEED = 0
N_Q_SPLITS = 5
Q_PRODUCTION_SEED = 0

PARITY_ATOL = 0.008
FROZEN_Q = {
    "vit_base": -0.2404841119636992,
    "dinov3": -0.001887656498552317,
    "clip_base": 0.27038674382874994,
    "convnext_base": 0.23140873373502246,
    "vit_large": -0.1896303079330747,
}
VITB_P2 = 0.11614541311898466
VITB_P1 = -0.0016449827122028832

NATIVE_D = {
    "vit_base": 768,
    "dinov3": 768,
    "clip_base": 512,
    "convnext_base": 1024,
    "vit_large": 1024,
}

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_FCR = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
SOURCE_PRCR = "outputs/geometry/physics_pointwise_residual_curvature_probe_relation"
SOURCE_CMPR = "outputs/geometry/physics_cross_model_pointwise_residual_curvature"
SOURCE_TAC = "outputs/geometry/physics_task_aligned_curvature"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"

DECISION_LABELS = (
    "q_task_aligned_replicates_across_models",
    "q_task_aligned_heterogeneous_across_models",
    "q_task_aligned_vitb_specific",
    "q_and_d_task_aligned_replicate",
    "d_task_aligned_replicates_q_does_not",
    "task_aligned_cross_model_unresolved",
)

PRESERVED = (
    SOURCE_MM,
    SOURCE_CMCLA,
    SOURCE_FCR,
    SOURCE_PRCR,
    SOURCE_CMPR,
    SOURCE_TAC,
    SOURCE_NDC,
    "outputs/geometry/physics_q_geometry_resampling_stability",
    "outputs/geometry/physics_global_probe_curvature_alignment",
    "paper",
    "papers",
    "submissions",
)


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cpu"
    hessian_device: str = "cpu"
    smoke: bool = False
    skip_decoder: bool = False
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    n_q_splits: int = N_Q_SPLITS
    models_override: list[str] | None = field(default=None)

    def models(self) -> tuple[str, ...]:
        if self.models_override:
            return tuple(self.models_override)
        return ALL_MODELS

    def n_anc(self) -> int:
        return N_SMOKE if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 80 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 80 if self.smoke else self.n_boot
