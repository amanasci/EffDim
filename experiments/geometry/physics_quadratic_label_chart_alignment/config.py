"""Preregistered: does quadratic label structure in the fitted chart explain LPA?"""

from __future__ import annotations

from dataclasses import dataclass

PRESERVED = (
    "experiments/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_order_stratified_geometry",
    "outputs/geometry/physics_quadratic_predictive_dimension",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "submissions/neurreps_2026",
    "submissions/neurreps_2026_lpa_revision",
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_LPA = "outputs/geometry/physics_local_probe_adaptation"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"

MODEL = "vit_base"
CATALOG_FIELD = "mag_r_desi"
PRIMARY_K = 2048
PRIMARY_D = 16
N_ANCHORS = 512
N_QUAD = PRIMARY_D * (PRIMARY_D + 1) // 2  # 136
assert N_QUAD == 136

CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")

PARITY_R2 = -0.240
PARITY_MSE = 0.227
PARITY_DMSE = 0.153
PARITY_ATOL = 0.008

PROBE_ALPHA = 100.0
# Matched log-spaced grids for linear / quadratic blocks (nested CV)
LIN_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
QUAD_GRID = (1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)

N_BOOT = 2000
N_PERM = 10000
SEED = 0
MIN_TRAIN = 32
MIN_TEST = 8

# Synthetic gates (frozen before ViT-B)
SYNTH_LIN_MAX_DELTA = 0.02
SYNTH_ALIGN_MIN_DELTA = 0.05
SYNTH_ORTH_BS_MAX_FRAC = 0.35  # BS gain / UQ gain when orthogonal
SYNTH_SHUFFLE_MAX_ABS_RHO = 0.15

DECISION_LABELS = (
    "curvature_aligned_quadratic_decoding",
    "generic_quadratic_local_decoding",
    "tangent_rotation_preferred",
    "quadratic_probe_unstable_or_overfit",
    "quadratic_chart_link_unresolved",
)


@dataclass
class ExpConfig:
    output_dir: str = "outputs/geometry/physics_quadratic_label_chart_alignment"
    seed: int = SEED
    n_boot: int = N_BOOT
    n_perm: int = N_PERM
    smoke: bool = False
    force: bool = False
    stage: str = "all"
    n_anchors_override: int | None = None

    def n_anc(self) -> int:
        if self.n_anchors_override is not None:
            return int(self.n_anchors_override)
        return 16 if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 200 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 200 if self.smoke else self.n_boot
