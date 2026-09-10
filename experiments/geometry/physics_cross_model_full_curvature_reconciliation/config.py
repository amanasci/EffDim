"""Frozen full-curvature reconciliation protocol. Writes only into new trees."""

from __future__ import annotations

from dataclasses import dataclass, field

PRESERVED = (
    "experiments/geometry/physics_curvature_probe_submission_validation",
    "experiments/geometry/physics_local_probe_adaptation",
    "experiments/geometry/physics_local_probe_adaptation_audit",
    "experiments/geometry/physics_quadratic_label_chart_alignment",
    "experiments/geometry/physics_quadratic_label_chart_alignment_audit",
    "experiments/geometry/physics_nested_dimension_curvature",
    "experiments/geometry/physics_order_stratified_geometry",
    "experiments/geometry/physics_cross_model_curvature_local_adaptation",
    "experiments/geometry/physics_activation_atlas",
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_local_probe_adaptation_audit",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment_audit",
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_order_stratified_geometry",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "outputs/geometry/physics_cross_model_curvature_local_adaptation",
    "outputs/geometry/physics_effdim_curvature_metrics",
    "outputs/geometry/physics_full_curvature_audit",
    "outputs/geometry/physics_split_half_curvature_reliability",
    "outputs/geometry/physics_cross_model_probe_curvature_coverage",
    "submissions/neurreps_2026",
    "submissions/neurreps_2026_lpa_revision",
    "submissions/ml4ps_2026",
    "papers/interpscience_semantic_fields",
    "papers/curvature_photometric_decoding",
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"

POSITIVE_CONTROL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
PRIMARY_K = 2048
PRIMARY_D = 16
N_ANCHORS = 512
N_SPLITS = 5
SEED = 0

# Historical primary full-curvature estimand (Phase 0).
# K_dir^cross = <H_A, H_B> + [2/(d(d+2))] <B0_A, B0_B>_F
#             = (2 <B_A,B_B>_F + <tr B_A, tr B_B>) / (d(d+2))
HISTORICAL_FULL_METRIC = "K_dir_cross"
PRIMARY_METRIC = "K_dir_cross"

CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

# Completed trace-based cross-model table (do not overwrite that label).
TRACE_DECISION_LABEL = "representation_specific_effect"
PARITY_VITB_R2 = -0.240
PARITY_VITB_MSE_G = 0.227
PARITY_VITB_DMSE = 0.153
PARITY_ATOL = 0.008
TRACE_PARITY_ATOL = 1e-8
HISTORICAL_KDIR_ATOL = 1e-8

TRACE_EXPECTED = {
    "vit_base": {
        "C_G": 0.22704789227635297,
        "C_A": 0.15334238492921803,
        "C_P": 0.17477573070804867,
        "C_R2": -0.2404841119636992,
        "A": 0.052272161568304304,
        "mean_delta_adapt": -0.10119904692121723,
    },
    "dinov3": {
        "C_G": -0.12981029199711608,
        "C_A": -0.04471491943328641,
        "C_P": -0.1232513718657374,
        "C_R2": -0.001887656498552317,
        "A": -0.006558920131378682,
        "mean_delta_adapt": -0.09256759848524429,
    },
    "clip_base": {
        "C_G": -0.27375793579267804,
        "C_A": -0.17898726196198259,
        "C_P": -0.38406706334901175,
        "C_R2": 0.27038674382874994,
        "A": 0.11030912755633371,
        "mean_delta_adapt": -0.054483005101683774,
    },
    "convnext_base": {
        "C_G": -0.18663614658793098,
        "C_A": -0.2548593766093315,
        "C_P": -0.1829223458188851,
        "C_R2": 0.23140873373502246,
        "A": -0.003713800769045872,
        "mean_delta_adapt": -0.07090636865046433,
    },
    "vit_large": {
        "C_G": -0.03830513069202687,
        "C_A": 0.18130031843306899,
        "C_P": -0.14876777059658278,
        "C_R2": -0.1896303079330747,
        "A": 0.1104626399045559,
        "mean_delta_adapt": -0.07119383488297938,
    },
}

R_H_FAIL = 0.20
N_BOOT = 2000
N_PERM = 10000

MODELS = ("vit_base", "dinov3", "clip_base", "convnext_base", "vit_large")

CURVATURE_COLS = (
    "K_dir_cross",
    "K_B_cross",
    "K_H_cross",
    "K_aniso_cross",
)

OUTCOME_COLS = (
    "r2_G",
    "mse_G",
    "r2_P",
    "mse_P",
    "delta_adapt",
)

DECISION_LABELS = (
    "full_curvature_global_to_local_sign_reversal",
    "full_curvature_cross_model_global_penalty_with_relative_adaptation",
    "full_curvature_cross_model_global_penalty_only",
    "full_curvature_partial_cross_model_replication",
    "trace_full_curvature_estimand_divergence",
    "full_curvature_reconciliation_unresolved",
    "full_curvature_reconciliation_blocked",
)

MODEL_SPECS: tuple[dict, ...] = (
    {"model_id": "vit_base", "architecture": "ViT-B", "positive_control": True},
    {"model_id": "dinov3", "architecture": "DINOv3", "positive_control": False},
    {"model_id": "clip_base", "architecture": "CLIP", "positive_control": False},
    {"model_id": "convnext_base", "architecture": "ConvNeXt-B", "positive_control": False},
    {"model_id": "vit_large", "architecture": "ViT-L", "positive_control": False},
)


@dataclass
class ExpConfig:
    output_dir: str = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
    seed: int = SEED
    n_boot: int = N_BOOT
    n_perm: int = N_PERM
    smoke: bool = False
    force: bool = False
    stage: str = "all"
    device: str = "cuda"
    n_anchors_override: int | None = None
    models_override: list[str] | None = field(default=None)
    skip_geometry: bool = False

    def n_anc(self) -> int:
        if self.n_anchors_override is not None:
            return int(self.n_anchors_override)
        return 8 if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 200 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 200 if self.smoke else self.n_boot

    def n_splits(self) -> int:
        return 1 if self.smoke else N_SPLITS
