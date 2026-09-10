"""Frozen cross-model local-adaptation protocol. Does not write into preserved trees."""

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
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_local_probe_adaptation_audit",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment_audit",
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_order_stratified_geometry",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "submissions/neurreps_2026",
    "submissions/neurreps_2026_lpa_revision",
    "submissions/ml4ps_2026",
    "papers/interpscience_semantic_fields",
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_LPA = "outputs/geometry/physics_local_probe_adaptation"
SOURCE_LPA_AUDIT = "outputs/geometry/physics_local_probe_adaptation_audit"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_QLCA = "outputs/geometry/physics_quadratic_label_chart_alignment"
SOURCE_QLCA_AUDIT = "outputs/geometry/physics_quadratic_label_chart_alignment_audit"

POSITIVE_CONTROL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
PRIMARY_K = 2048
PRIMARY_D = 16
N_ANCHORS = 512
N_FOLDS = 5
PROBE_ALPHA = 100.0  # frozen LPA confirmatory estimator (fixed, not nested)
N_SPLITS_KH = 5
MIN_COMMON_ANCHORS = 256
MIN_TRAIN_PER_FOLD = 32
MIN_TEST_PER_FOLD = 8

CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

PARITY_R2 = -0.240
PARITY_MSE_G = 0.227
PARITY_DMSE = 0.153
PARITY_MSE_P = 0.175
PARITY_DM_MEAN = -0.10
PARITY_ATOL = 0.008

R_H_FAIL = 0.20  # frozen scale-bias-variance reliability fail threshold
WEIGHT_COS_RELIABLE = 0.85  # frozen LPA-audit fold-stability gate (set before inspecting)

N_BOOT = 2000
N_PERM = 10000
N_SHUFFLE = 64
SEED = 0

# Holm family, per model, frozen before outcomes.
PRIMARY_FAMILY = ("C_G", "C_A", "A")

QLCA_MEDIAN_DELTA_Q = 0.020582
QLCA_RHO_KH_DQ = 0.111249
QLCA_A_B = 2.427
QLCA_PARTIAL_ADAPT = 0.205180

DECISION_LABELS = (
    "cross_model_global_penalty_and_local_adaptation",
    "global_penalty_replicates_local_adaptation_model_dependent",
    "local_adaptation_replicates_global_penalty_model_dependent",
    "representation_specific_effect",
    "insufficient_model_diversity",
    "geometry_unreliable_across_models",
    "cross_model_result_unresolved",
)

# Artifact-level identity only. Eligibility must not inspect scientific outcomes.
MODEL_SPECS: tuple[dict, ...] = (
    {
        "model_id": "vit_base",
        "source_parquet": "data_hf/physics/vit_base_test.parquet",
        "embedding_column": "vit_base_galaxies",
        "architecture": "ViT-B",
        "training_objective": "frozen physics-table ViT-B galaxy embedding",
        "representation_layer": "pooled embedding column vit_base_galaxies",
        "positive_control": True,
    },
    {
        "model_id": "dinov3",
        "source_parquet": "data_hf/physics/dinov3_vitb16_test.parquet",
        "embedding_column": "dinov3_vitb16_galaxies",
        "architecture": "DINOv3 ViT-B/16",
        "training_objective": "self-supervised (DINOv3) frozen physics-table embedding",
        "representation_layer": "pooled embedding column dinov3_vitb16_galaxies",
        "positive_control": False,
    },
    {
        "model_id": "clip_base",
        "source_parquet": "data_hf/physics/clip_base_test.parquet",
        "embedding_column": "clip_base_galaxies",
        "architecture": "CLIP ViT-B",
        "training_objective": "contrastive (CLIP) frozen physics-table embedding",
        "representation_layer": "pooled embedding column clip_base_galaxies",
        "positive_control": False,
    },
    {
        "model_id": "convnext_base",
        "source_parquet": "data_hf/physics/convnext_base_test.parquet",
        "embedding_column": "convnext_base_galaxies",
        "architecture": "ConvNeXt-B",
        "training_objective": "frozen physics-table ConvNeXt-B galaxy embedding",
        "representation_layer": "pooled embedding column convnext_base_galaxies",
        "positive_control": False,
    },
    {
        "model_id": "vit_large",
        "source_parquet": "data_hf/physics/vit_large_test.parquet",
        "embedding_column": "vit_large_galaxies",
        "architecture": "ViT-L",
        "training_objective": "frozen physics-table ViT-L galaxy embedding",
        "representation_layer": "pooled embedding column vit_large_galaxies",
        "positive_control": False,
    },
)


@dataclass
class ExpConfig:
    output_dir: str = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
    seed: int = SEED
    n_boot: int = N_BOOT
    n_perm: int = N_PERM
    n_shuffle: int = N_SHUFFLE
    smoke: bool = False
    force: bool = False
    stage: str = "all"
    skip_shuffle: bool = False
    skip_geometry: bool = False
    device: str = "cuda"
    n_anchors_override: int | None = None
    models_override: list[str] | None = field(default=None)

    def n_anc(self) -> int:
        if self.n_anchors_override is not None:
            return int(self.n_anchors_override)
        return 8 if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 200 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 200 if self.smoke else self.n_boot

    def n_shuffle_eff(self) -> int:
        return 8 if self.smoke else self.n_shuffle

    def n_splits_kh(self) -> int:
        return 1 if self.smoke else N_SPLITS_KH
