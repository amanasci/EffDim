"""Frozen constants. Do not retune after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass, field

OUT_REL = "outputs/geometry/physics_task_aligned_curvature"
WALL_S = 60 * 60
PROJECTED_CAP_S = 55 * 60
RESERVE_WRITE_S = 180.0
Q_PANEL_SKIP = True

MODEL = "vit_base"
TARGETS = ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass")
D_LAT = 16
K = 2048
N_ANCHORS = 512
N_SMOKE = 32
N_FOLDS = 5
PROBE_ALPHA = 100.0
TRAIN_FRAC = 0.60
MIN_EVAL = 128
COVERAGE_FAIL_FRAC = 0.10
SPLIT_SALT = "task_aligned_curvature_v1"

DECODER_SEEDS = (0, 1, 2)
PROTOCOL = "reproduction_plain_ae_400"
AE_HIDDEN = (250, 250, 250)
AE_ACTIVATION = "silu"

CONTROLS = ("log_knn_radius", "local_eval_label_variance", "local_evaluation_count")
MULTISCALE_K = (16, 64, 256, 1024, 2048)

N_PERM = 10000
N_BOOT = 2000
N_BOOT_DEP = 2000
INFER_SEED = 0
N_Q_SPLITS = 5
Q_PRODUCTION_SEED = 0

PARITY_ATOL = 0.008
FROZEN_Q_R2 = -0.2404841119636992
FROZEN_CH_R2 = 0.02970575697233952
HISTORICAL_RAW_RHO = 0.347
HISTORICAL_AMEND01_RHO = 0.328

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_FCR = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
SOURCE_PRCR = "outputs/geometry/physics_pointwise_residual_curvature_probe_relation"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_ALIGN = "outputs/geometry/physics_global_probe_curvature_alignment"
SOURCE_AE = "outputs/geometry/physics_ae_local_patch_scale_match"
SOURCE_AUSTIN = "outputs/geometry/pointwise_decoder_curvature_reproduction"
SOURCE_MULTI = "outputs/geometry/physics_multilabel_chart_screen"

DECISION_LABELS = (
    "cross_instrument_task_aligned_curvature_supported",
    "task_aligned_curvature_supported_with_target_heterogeneity",
    "decoder_task_aligned_effect_only",
    "quadratic_task_aligned_effect_only",
    "historical_effect_not_leakage_safe",
    "radial_or_prediction_coupling_dominates",
    "task_aligned_curvature_unresolved",
)

PRESERVED = (
    SOURCE_MM,
    SOURCE_CMCLA,
    SOURCE_FCR,
    SOURCE_PRCR,
    SOURCE_NDC,
    SOURCE_ALIGN,
    SOURCE_AE,
    SOURCE_AUSTIN,
    SOURCE_MULTI,
    "outputs/geometry/physics_cross_model_pointwise_residual_curvature",
    "outputs/geometry/physics_q_geometry_resampling_stability",
    "outputs/geometry/known_curvature_dual_estimator_robustness",
    "outputs/geometry/known_curvature_estimator_operating_characteristics",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
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
    q_device: str = "cpu"
    smoke: bool = False
    skip_patch: bool = False
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    n_q_splits: int = N_Q_SPLITS
    models_override: list[str] | None = field(default=None)

    def n_anc(self) -> int:
        return N_SMOKE if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 80 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 80 if self.smoke else self.n_boot
