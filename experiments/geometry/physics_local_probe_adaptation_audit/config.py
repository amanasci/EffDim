"""Final audit of local-probe-adaptation. Read-only inputs only."""

from __future__ import annotations

from dataclasses import dataclass

PRESERVED = (
    "experiments/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_curvature_scale_bias_variance",
    "submissions/neurreps_2026",
)

SOURCE_LPA = "outputs/geometry/physics_local_probe_adaptation"
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_ALIGN = "outputs/geometry/physics_global_probe_curvature_alignment"
SOURCE_GEOM = "outputs/geometry/physics_curvature_probe_multitarget/geometry_cache"
SOURCE_H = "outputs/geometry/physics_nested_dimension_curvature/H_vectors"

MODEL = "vit_base"
TARGET = "mag_r_desi"
PRIMARY_K = 2048
PRIMARY_D = 16
PROBE_ALPHA = 100.0
N_ANCHORS = 512
N_PROBE_EVAL = 2048

CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

# parity targets
PARITY_R2 = -0.240
PARITY_MSE = 0.227
PARITY_DMH = 0.153
PARITY_DM_MEAN = -0.10
PARITY_MSE_G = 0.227
PARITY_MSE_P = 0.175
PARITY_ATOL = 0.008

N_BOOT = 10000
N_SHUFFLE = 200
N_SHUFFLE_ANCHORS = 128
SEED = 0

WEIGHT_COS_RELIABLE = 0.85  # fold-stability gate for direction claims

DECISION_LABELS = (
    "curvature_predicts_local_direction_adaptation",
    "curvature_predicts_relative_local_adaptation",
    "curvature_predicts_local_calibration_gain",
    "in_sample_patch_probe_artifact",
    "local_probe_result_unresolved",
)


@dataclass
class AuditConfig:
    output_dir: str = "outputs/geometry/physics_local_probe_adaptation_audit"
    seed: int = SEED
    n_boot: int = N_BOOT
    n_shuffle: int = N_SHUFFLE
    smoke: bool = False
    force: bool = False
    skip_shuffle: bool = False
    stage: str = "all"

    def n_boot_eff(self) -> int:
        return 500 if self.smoke else self.n_boot

    def n_shuffle_eff(self) -> int:
        return 20 if self.smoke else self.n_shuffle

    def n_shuffle_anc(self) -> int:
        return 16 if self.smoke else N_SHUFFLE_ANCHORS
