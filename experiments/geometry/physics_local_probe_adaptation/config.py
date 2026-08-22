"""Frozen local-probe adaptation vs curvature. Does not write into preserved trees."""

from __future__ import annotations

from dataclasses import dataclass, field

PRESERVED = (
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_curvature_scale_bias_variance",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "submissions/neurreps_2026",
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_SCREEN = "outputs/geometry/physics_curvature_probe_screen"

MODEL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
PRIMARY_K = 2048
PRIMARY_D = 16
N_ANCHORS = 512
N_FOLDS = 5
PROBE_ALPHA = 100.0  # frozen global ridge; sum-of-squares objective
ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)

CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

PARITY_R2 = -0.240
PARITY_MSE = 0.227
PARITY_ATOL = 0.005

N_BOOT = 2000
N_PERM = 10000
N_SHUFFLE = 200
N_SHUFFLE_ANCHORS = 128
SEED = 0

MIN_TRAIN_PER_FOLD = 32
MIN_TEST_PER_FOLD = 8

DECISION_LABELS = (
    "curvature_predicts_local_direction_adaptation",
    "curvature_predicts_local_calibration_gain",
    "in_sample_patch_probe_artifact",
    "local_probe_result_unresolved",
)

# Secondary family for Holm (fixed before inspecting results)
SECONDARY_FAMILY = (
    "rho_KH_R2_P",
    "rho_KH_MSE_P",
    "rho_KH_dMAE_GP",
    "rho_KH_dR2_GP",
    "rho_KH_dMSE_CP",
)


@dataclass
class ExpConfig:
    output_dir: str = "outputs/geometry/physics_local_probe_adaptation"
    stage: str = "all"
    seed: int = SEED
    n_boot: int = N_BOOT
    n_perm: int = N_PERM
    n_shuffle: int = N_SHUFFLE
    smoke: bool = False
    force: bool = False
    skip_tangent: bool = False
    skip_nested_alpha: bool = False
    skip_shuffle: bool = False
    device: str = "cpu"

    def n_anc(self) -> int:
        return 8 if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 200 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 200 if self.smoke else self.n_boot

    def n_shuffle_eff(self) -> int:
        return 20 if self.smoke else self.n_shuffle
