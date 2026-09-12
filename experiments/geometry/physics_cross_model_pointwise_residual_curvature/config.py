"""Frozen constants. Do not retune after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass, field

OUT_REL = "outputs/geometry/physics_cross_model_pointwise_residual_curvature"
WALL_S = 90 * 60
PROJECTED_CAP_S = 85 * 60
Q_PANEL_ELAPSED_MAX_S = 55 * 60
RESERVE_WRITE_S = 180.0
MAX_NEW_DECODERS = 12
DECODER_SEEDS = (0, 1, 2)

D_LAT = 16
N_ANCHORS = 512
N_SMOKE = 32
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
PROTOCOL = "reproduction_plain_ae_400"

REFERENCE = "vit_base"
REPLICATION = ("dinov3", "clip_base", "convnext_base", "vit_large")
ALL_MODELS = (REFERENCE, *REPLICATION)
CATALOG_FIELD = "mag_r_desi"
PRIMARY_K = 2048
PRIMARY_D = 16
N_FOLDS = 5
CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_FCR = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
SOURCE_PRCR = "outputs/geometry/physics_pointwise_residual_curvature_probe_relation"
SOURCE_QRES = "outputs/geometry/physics_q_geometry_resampling_stability"

PARITY_ATOL = 0.008
FROZEN_Q = {
    "vit_base": {"C_R2": -0.2404841119636992, "C_G": 0.22704789227635297, "C_A": 0.15334238492921803},
    "dinov3": {"C_R2": -0.001887656498552317, "C_G": -0.12981029199711608, "C_A": -0.04471491943328641},
    "clip_base": {"C_R2": 0.27038674382874994, "C_G": -0.27375793579267804, "C_A": -0.17898726196198259},
    "convnext_base": {"C_R2": 0.23140873373502246, "C_G": -0.18663614658793098, "C_A": -0.2548593766093315},
    "vit_large": {"C_R2": -0.1896303079330747, "C_G": -0.03830513069202687, "C_A": 0.18130031843306899},
}
FROZEN_VITB_DRES = {
    "C_R2": 0.02970575697233952,
    "C_P": -0.21959123336308806,
    "C_A": -0.2073438626532083,
    "median_rho_seed": 0.881358737168263,
    "median_cos_seed": 0.8780326144977072,
}
SEED_RHO_GATE = 0.70
SEED_COS_GATE = 0.80
MIN_FINITE_FRAC = 0.95
EQUIV_ABS = 0.10

N_PERM = 10000
N_BOOT = 2000
INFER_SEED = 0

NATIVE_D = {
    "vit_base": 768,
    "dinov3": 768,
    "clip_base": 512,
    "convnext_base": 1024,
    "vit_large": 1024,
}
ARCH_FAMILY = {
    "vit_base": {"family": "transformer", "contrastive": False},
    "dinov3": {"family": "transformer", "contrastive": False},
    "clip_base": {"family": "transformer", "contrastive": True},
    "convnext_base": {"family": "convolutional", "contrastive": False},
    "vit_large": {"family": "transformer", "contrastive": False},
}

DECISION_LABELS = (
    "cross_model_pointwise_local_degradation_replication",
    "cross_model_pointwise_patch_degradation_only",
    "vitb_specific_pointwise_local_degradation",
    "heterogeneous_pointwise_residual_probe_relation",
    "cross_model_pointwise_residual_null",
    "pointwise_residual_seed_unstable_cross_model",
    "cross_model_pointwise_residual_unresolved",
)
Q_LABELS = (
    "q_cross_model_heterogeneity_resampling_stable",
    "q_cross_model_heterogeneity_resampling_sensitive",
    "q_cross_model_resampling_exploratory",
    "q_cross_model_resampling_not_run",
)

PRESERVED = (
    "outputs/geometry/physics_pointwise_residual_curvature_probe_relation",
    "outputs/geometry/physics_cross_model_curvature_local_adaptation",
    "outputs/geometry/physics_cross_model_full_curvature_reconciliation",
    "outputs/geometry/physics_q_geometry_resampling_stability",
    "outputs/geometry/known_curvature_dual_estimator_robustness",
    "outputs/geometry/known_curvature_estimator_operating_characteristics",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "paper",
    "papers",
    "submissions",
)


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cuda"
    hessian_device: str = "cpu"
    smoke: bool = False
    force: bool = False
    skip_q_panel: bool = False
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    models_override: list[str] | None = field(default=None)

    def models(self) -> tuple[str, ...]:
        if self.models_override:
            return tuple(self.models_override)
        return ALL_MODELS

    def n_anc(self) -> int:
        return 8 if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 80 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 80 if self.smoke else self.n_boot
