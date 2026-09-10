"""Frozen constants. Do not retune after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass

CURVATURE_EXPERIMENTS_SHA = "97efb2eb6cd7dec7f2c568f53c534752ff3c32c8"
REPRODUCTION_REL = "outputs/geometry/pointwise_decoder_curvature_reproduction"
DUAL_REL = "outputs/geometry/known_curvature_dual_estimator_robustness"

OUT_REL = "outputs/geometry/physics_pointwise_residual_curvature_probe_relation"
WALL_S = 75 * 60
RESERVE_WRITE_S = 120.0
MAX_NEW_DECODERS = 3
DECODER_SEEDS = (0, 1, 2)

D_LAT = 16
N_ANCHORS = 512
N_SMOKE = 32
N_TENSOR_SUBSET = 128
HESSIAN_CHUNK = 8

AE_HIDDEN = (250, 250, 250)
AE_ACTIVATION = "silu"
MAX_EPOCHS = 400
TRAIN_CFG_TEMPLATE = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 128,
    "max_epochs": MAX_EPOCHS,
    "early_stop_patience": MAX_EPOCHS + 1,
    "early_stop_min_delta": 1e-9,
    "lip_weight": 0.0,
    "fps_pretrain_epochs": 0,
    "wallclock_ceiling_s": float("inf"),
}

MODEL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
PRIMARY_K = 2048
PRIMARY_D = 16
N_FOLDS = 5
CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_LPA = "outputs/geometry/physics_local_probe_adaptation"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_FCR = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
SOURCE_QLCA = "outputs/geometry/physics_quadratic_label_chart_alignment"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"

PARITY_R2 = -0.240
PARITY_MSE_G = 0.227
PARITY_DMSE = 0.153
PARITY_ATOL = 0.005
HISTORICAL_DECODER_R2_RHO = 0.328
HISTORICAL_DECODER_RHO_ATOL = 0.02

SEED_RHO_GATE = 0.70
SEED_COS_GATE = 0.80

N_PERM = 10000
N_BOOT = 2000
INFER_SEED = 0
HASH_SALT = b"physics_pointwise_residual_curvature_probe_relation:v1"

OTHER_ENCODERS = ("dinov3", "clip_base", "convnext_base", "vit_large")

PRESERVED = (
    "outputs/geometry/pointwise_decoder_curvature_reproduction",
    "outputs/geometry/known_curvature_dual_estimator_robustness",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_cross_model_curvature_local_adaptation",
    "outputs/geometry/physics_cross_model_full_curvature_reconciliation",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_nested_dimension_curvature",
    "paper",
    "submissions",
)

DECISION_LABELS = (
    "stable_global_penalty_and_absolute_local_reversal",
    "stable_global_penalty_with_relative_local_adaptation",
    "stable_global_penalty_without_local_relief",
    "stable_positive_pointwise_decodability_link",
    "stable_pointwise_residual_null",
    "decoder_residual_seed_unstable",
    "pointwise_residual_probe_relation_unresolved",
)


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cuda"
    hessian_device: str = "cpu"
    smoke: bool = False
    force: bool = False
    skip_train: bool = False
    n_perm: int = N_PERM
    n_boot: int = N_BOOT

    def n_anc(self) -> int:
        return N_SMOKE if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 200 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 200 if self.smoke else self.n_boot
