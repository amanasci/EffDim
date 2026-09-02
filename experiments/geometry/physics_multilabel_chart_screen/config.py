"""Multi-label curvature / quadratic-chart screen.

Reuses frozen ViT-B charts, neighbourhoods, and global OOF probes.
Does not overwrite NeurReps or frozen QLCA / submission-validation trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRESERVED = (
    "experiments/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment_audit",
    "outputs/geometry/physics_curvature_probe_submission_validation",
    "outputs/geometry/physics_local_probe_adaptation",
    "outputs/geometry/physics_multimodel_graph_prior_quadratic",
    "submissions/neurreps_2026",
    "submissions/neurreps_2026_lpa_revision",
    "submissions/ml4ps_2026",
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CPRS = "outputs/geometry/physics_curvature_probe_rank_sweep"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"

MODEL = "vit_base"
PRIMARY_K = 2048
PRIMARY_D = 16
N_ANCHORS = 512
MIN_FINITE_NEIGH = 64
MIN_ANALYZED_ANCHORS = 64

# Proven physics sample_id join only. DESI spectroscopic / DESI imaging
# catalog joins are excluded (identity unproven).
ELIGIBLE = (
    {
        "field": "mag_r_desi",
        "family": "photometric_magnitude",
        "role": "confirmatory_parity",
        "display": "apparent r-band magnitude",
    },
    {
        "field": "photo_z",
        "family": "photometric_redshift",
        "role": "secondary_screen",
        "display": "photometric redshift",
    },
    {
        "field": "smooth_fraction",
        "family": "morphology",
        "role": "secondary_screen",
        "display": "smooth fraction",
    },
    {
        "field": "stellar_mass",
        "family": "stellar_population_proxy",
        "role": "secondary_screen",
        "display": "catalog stellar-mass proxy",
    },
)

EXCLUDED = (
    {"field": "sfr", "reason": "underpowered (≈45 labelled anchors; n<64 rule)"},
    {"field": "desi_spec_z", "reason": "object-level identity join unproven"},
    {"field": "desi_mag_r", "reason": "object-level identity join unproven"},
)

PARITY_R2 = -0.240
PARITY_MSE = 0.227
PARITY_ATOL = 0.008

N_BOOT = 2000
N_PERM = 10000
SEED = 0
CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")


@dataclass
class ScreenConfig:
    output_dir: str = "outputs/geometry/physics_multilabel_chart_screen"
    smoke: bool = False
    force: bool = False
    seed: int = SEED
    n_anchors_override: int | None = None
    labels: list[str] = field(default_factory=lambda: [r["field"] for r in ELIGIBLE])
    skip_quadratic: bool = False

    def n_anc(self) -> int:
        if self.n_anchors_override is not None:
            return int(self.n_anchors_override)
        return 16 if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 200 if self.smoke else N_PERM

    def n_boot_eff(self) -> int:
        return 200 if self.smoke else N_BOOT
