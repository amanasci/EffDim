"""Audit of frozen QLCA. Never write into original trees."""

from __future__ import annotations

from dataclasses import dataclass

PRESERVED = (
    "experiments/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "experiments/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_local_probe_adaptation_audit",
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_nested_dimension_curvature",
    "outputs/geometry/physics_order_stratified_geometry",
    "outputs/geometry/physics_quadratic_predictive_dimension",
    "outputs/geometry/physics_curvature_probe_rank_sweep",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "submissions/neurreps_2026",
    "submissions/neurreps_2026_lpa_revision",
)

SOURCE_QLCA = "outputs/geometry/physics_quadratic_label_chart_alignment"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_LPA = "outputs/geometry/physics_local_probe_adaptation"

ORIGINAL_LABEL = "quadratic_chart_link_unresolved"
ORIGINAL_N_COMP_CAP = 48  # frozen models._bs_basis hard cap
ORIGINAL_ENERGY_FRAC = 0.99
STABILITY_THRESHOLD = 0.5  # frozen from original pipeline; not tuned on A_B

PRIMARY_D = 16
N_QUAD = 136
N_ANCHORS = 512
ENERGY_FRACS = (0.90, 0.95, 0.99)
TRUNC_RULES = ("e90", "e95", "e99", "original_rule")

# numpy.linalg.matrix_rank convention: s > max(shape) * eps * smax
RANK_EPS_MULT = 1.0

N_HAAR = 2000
N_BOOT = 2000
N_PERM = 10000
N_SHUFFLE_SYNTH_SEEDS = 21
N_REAL_SHUFFLE_ANCHORS = 16
N_REAL_SHUFFLE_SEEDS = 12
SEED = 0

# Phase 0 reproduction tolerances (absolute, from frozen JSON/CSV)
PARITY_ATOL = {
    "median_delta_Q": 0.001,
    "rho_KH_delta_Q": 0.002,
    "median_delta_BS": 0.001,
    "frac_UQ_captured_by_BS": 0.01,
    "median_delta_FQ": 0.001,
    "A_B_median": 0.05,
    "A_B_null_median": 0.05,
    "gamma_fold_cosine_median": 0.02,
    "rho_r2": 0.008,
    "rho_mse": 0.008,
    "rho_dmse": 0.008,
    "rho_dmse_adj": 0.01,
    "shuffle_dQ": 0.5,
}


@dataclass
class AuditConfig:
    output_dir: str = "outputs/geometry/physics_quadratic_label_chart_alignment_audit"
    seed: int = SEED
    smoke: bool = False
    force: bool = False
    skip_truncated: bool = False
    skip_haar: bool = False
    n_anchors_override: int | None = None

    def n_anc(self) -> int:
        if self.n_anchors_override is not None:
            return int(self.n_anchors_override)
        return 8 if self.smoke else N_ANCHORS

    def n_haar(self) -> int:
        return 40 if self.smoke else N_HAAR

    def n_boot(self) -> int:
        return 200 if self.smoke else N_BOOT

    def n_real_anc(self) -> int:
        return 4 if self.smoke else N_REAL_SHUFFLE_ANCHORS

    def n_real_seeds(self) -> int:
        return 3 if self.smoke else N_REAL_SHUFFLE_SEEDS

    def n_synth_seeds(self) -> int:
        return 5 if self.smoke else N_SHUFFLE_SYNTH_SEEDS
