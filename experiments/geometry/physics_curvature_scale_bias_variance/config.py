"""Fixed objects. Does not write into preserved geometry trees."""

from __future__ import annotations

from dataclasses import dataclass, field

PRESERVED = (
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_nested_dimension_curvature",
    "submissions/neurreps_2026",
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_VAL = "outputs/geometry/physics_curvature_probe_submission_validation"

MODEL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
PRIMARY = "mag_r_desi_local_oof_r2"
MSE_FIELD = "mag_r_desi_oof_mse"
PRIMARY_K = 2048
PRIMARY_D = 16
PRIMARY_D = PRIMARY_D
SECONDARY_DS = (12, 20)
SECONDARY_DS = SECONDARY_DS
N_ANCHORS = 128
SEED = 0
N_REP_MAIN = 10
N_REP_SECONDARY = 5
N_BOOT = 2000
N_PERM = 5000
R_H_FAIL = 0.20
PARITY_ATOL = 0.005

COUNT_MS = (512, 1024, 1536, 2048)
RADIUS_RS = (1024, 1536, 2048)
FIXED_R = 2048
FIXED_M = 1024

PARITY_COMMON128_D16 = {1024: -0.027, 1536: -0.080, 2048: -0.171}
PARITY_FULL512_D16 = -0.240

CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

CELLS = (
    (2048, 512),
    (2048, 1024),
    (2048, 1536),
    (2048, 2048),
    (1024, 1024),
    (1536, 1024),
)

DECISION_LABELS = (
    "finite_sample_attenuation_supported",
    "geometric_washout_supported",
    "mixed_bias_variance",
    "mechanism_unresolved",
)


@dataclass
class ExpConfig:
    output_dir: str = "outputs/geometry/physics_curvature_scale_bias_variance"
    stage: str = "all"
    seed: int = SEED
    n_replicates: int = N_REP_MAIN
    n_secondary_replicates: int = N_REP_SECONDARY
    n_boot: int = N_BOOT
    n_perm: int = N_PERM
    n_anchors: int = N_ANCHORS
    smoke: bool = False
    force: bool = False
    extended_radius: bool = False
    skip_secondary: bool = False
    device: str = "cuda"

    def n_rep(self) -> int:
        return 2 if self.smoke else int(self.n_replicates)

    def n_rep_sec(self) -> int:
        return 2 if self.smoke else int(self.n_secondary_replicates)

    def n_anc(self) -> int:
        return 8 if self.smoke else int(self.n_anchors)
