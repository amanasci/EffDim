"""Exploratory component decomposition. Writes only into the new trees."""

from __future__ import annotations

from dataclasses import dataclass, field

PRESERVED = (
    "experiments/geometry/physics_cross_model_full_curvature_reconciliation",
    "experiments/geometry/physics_cross_model_curvature_local_adaptation",
    "experiments/geometry/physics_quadratic_label_chart_alignment",
    "experiments/geometry/physics_quadratic_label_chart_alignment_audit",
    "experiments/geometry/physics_local_probe_adaptation",
    "experiments/geometry/physics_local_probe_adaptation_audit",
    "experiments/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_cross_model_full_curvature_reconciliation",
    "outputs/geometry/physics_cross_model_curvature_local_adaptation",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment_audit",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_local_probe_adaptation_audit",
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

SOURCE_FCR = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_QLCA = "outputs/geometry/physics_quadratic_label_chart_alignment"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"

MODELS = ("vit_base", "dinov3", "clip_base", "convnext_base", "vit_large")
POSITIVE_CONTROL = "vit_base"
PRIMARY_D = 16
PRIMARY_K = 2048
N_ANCHORS = 512
N_QUAD = 136
CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

# Frozen parity
FCR_DIR_R2 = -0.025
FCR_DIR_R2P = -0.060
FCR_DIR_ADAPT = -0.073
FCR_ATOL = 0.008
PARITY_VITB_R2 = -0.240
PARITY_VITB_MSE_G = 0.227
PARITY_VITB_DMSE = 0.153
QLCA_MED_DQ = 0.020582
QLCA_RHO_KH_DQ = 0.111249
QLCA_AB = 2.427
QLCA_ATOL = 0.002
STAB_MIN = 0.5

N_BOOT = 2000
N_PERM = 10000
N_HAAR = 2000
SEED = 0

DECISION_LABELS = (
    "distinct_mean_and_traceless_predictive_roles",
    "mean_bending_specific_decodability_link",
    "traceless_bending_specific_decodability_link",
    "total_bending_magnitude_link",
    "representation_specific_curvature_component_effects",
    "curvature_component_link_unresolved",
    "curvature_component_audit_blocked",
)

PRIMARY_FAMILY = ("unique_KH_MSEG", "unique_KTF_MSEG")


@dataclass
class ExpConfig:
    output_dir: str = "outputs/geometry/physics_curvature_component_predictive_decomposition"
    seed: int = SEED
    n_boot: int = N_BOOT
    n_perm: int = N_PERM
    n_haar: int = N_HAAR
    smoke: bool = False
    force: bool = False
    stage: str = "all"
    n_anchors_override: int | None = None
    models_override: list[str] | None = field(default=None)
    n_workers: int = 8
    skip_probes: bool = False

    def n_anc(self) -> int:
        if self.n_anchors_override is not None:
            return int(self.n_anchors_override)
        return 8 if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 200 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 200 if self.smoke else self.n_boot

    def n_haar_eff(self) -> int:
        return 64 if self.smoke else self.n_haar
