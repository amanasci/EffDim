"""Frozen constants for the known-curvature point/patch fixture audit.

Numerical tolerances and bump parameters are frozen here *before* inspecting
estimator scores. Do not retune after looking at Suite A–F results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRESERVED = (
    "experiments/geometry/physics_cross_model_full_curvature_reconciliation",
    "experiments/geometry/physics_cross_model_curvature_local_adaptation",
    "experiments/geometry/physics_quadratic_label_chart_alignment",
    "experiments/geometry/physics_quadratic_label_chart_alignment_audit",
    "experiments/geometry/physics_curvature_component_predictive_decomposition",
    "experiments/geometry/physics_local_probe_adaptation",
    "experiments/geometry/physics_local_probe_adaptation_audit",
    "experiments/geometry/physics_curvature_probe_submission_validation",
    "experiments/geometry/run_ae_local_patch_scale_match.py",
    "outputs/geometry/physics_cross_model_full_curvature_reconciliation",
    "outputs/geometry/physics_cross_model_curvature_local_adaptation",
    "outputs/geometry/physics_quadratic_label_chart_alignment",
    "outputs/geometry/physics_quadratic_label_chart_alignment_audit",
    "outputs/geometry/physics_curvature_component_predictive_decomposition",
    "outputs/geometry/physics_ae_local_patch_scale_match",
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

FROZEN_ESTIMATOR_PATHS = (
    "experiments/geometry/physics_activation_atlas/nested_dimension_curvature.py",
    "experiments/geometry/physics_activation_atlas/full_curvature_audit.py",
    "experiments/geometry/physics_activation_atlas/sphere_normal_quadratic.py",
    "experiments/geometry/physics_activation_atlas/split_half_curvature_reliability.py",
    "experiments/geometry/physics_cross_model_full_curvature_reconciliation/metrics.py",
    "notebooks/pu_manifold/cae.py",
    "notebooks/pu_manifold/decoder_curvature.py",
    "notebooks/pu_manifold/chart_curvature.py",
    "experiments/geometry/run_ae_local_patch_scale_match.py",
)

OUT_REL = "outputs/geometry/known_curvature_point_patch_fixture_audit"

D_LAT = 16
D_AMB = 768
AMBIENT_ROTATION_SEED = 20260907
AMBIENT_ROTATION_HASH16 = "251f756d4adcce56"
FIXTURE_SEED = 20260907
ANCHOR_HASH_SEED = 20260907
SPLIT_SEED = 20260813
HOLDOUT_FRACTION = 0.2
TORCH_INIT_SEED = 0

# F1 latitude: G = (r φ, c) with r^2 + c^2 = 1.
F1_C = 0.6

# F3 nonminimal Clifford. Frozen before any estimator run.
F3_R2 = 0.7
F3_S2 = 0.3

# Bumped-sphere field (F4/F5). Centers from seed 20260905, then hardcoded.
# Do not retune after inspecting estimator performance.
F4_BUMP_SCALE = 0.8
F4_AMPLITUDES = (0.85, -0.70, 0.60, -0.50)
F4_WIDTHS = (0.80, 0.95, 0.70, 1.05)
F5_WIDTHS = (0.22, 0.18, 0.25, 0.20)
F4_CENTERS = (
    (
        0.6554975015546735, -0.523529394177129, 0.070503783212324, 0.1285679118433932,
        0.1763531876411883, 0.5664305600220579, -0.3992841493111468, -0.4791686552728127,
        0.0884049877469623, 0.4539735367877165, -0.019583153713226, -0.5701379946624442,
        0.3986891119736022, -0.0219427236815099, -0.2949744972657849, -0.562365480263261,
    ),
    (
        -0.2194601032411319, -1.3176041794136746, -0.7259616270721948, -0.3551205421731211,
        0.4910933697746437, 0.1955606151655251, -0.6559465853079872, 0.3431168177057484,
        -0.1138683587058425, 0.2958042506433989, -0.1702086361246476, -0.7586385825127746,
        -0.758542963790235, -0.2923113575699286, -0.0885149819427515, 0.2071702628939527,
    ),
    (
        -0.1294132673308467, -0.1986588413606185, 0.345153803024326, 0.3086747613037822,
        -1.1969646847319737, -0.5696954364410566, -0.0031542507345659, -0.2295998422227863,
        -0.328180276116154, -0.474532493440982, 0.0488860799285252, -0.3380500884350144,
        0.5713729802142972, -0.4137204047786925, 0.0356144405067917, -0.3618902838375533,
    ),
    (
        -0.3481821665540672, -0.6074072782802246, 0.0375553176078436, 0.4537031657531641,
        -0.1618543610254836, -0.2814702787043342, 0.1476463321983474, 0.5030402951523673,
        0.3532802173466812, 0.1040286585779806, 0.5002010759459415, -0.2312376261463009,
        0.4174851898124352, -0.0156933908771821, 0.2048566500914245, 0.9037811944949826,
    ),
)

# Colleague AE protocol.
AE_HIDDEN = (250, 250, 250)
AE_ACTIVATION = "silu"
MAX_EPOCHS = 600
TRAIN_CFG = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 128,
    "lip_weight": 0.0,
    "fps_pretrain_epochs": 0,
    "early_stop_patience": MAX_EPOCHS + 1,
    "early_stop_min_delta": 1e-9,
    "wallclock_ceiling_s": float("inf"),
    "seed": 0,
}
DECODER_SEEDS = (0, 1, 2)
N_SPLITS = 3
RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0]

PRIMARY_N = 16384
PRIMARY_K = 2048
N_ANCHORS_UNIT = 64
N_ANCHORS_SMOKE = 128
N_ANCHORS_PRIMARY = 512
N_COLLEAGUE = 86471
K_GRID_SMALL_N = (256, 512, 1024, 2048)
DIRECTION_SEED = 20260907
N_TANGENT_DIRS = 32
N_SPLITS_Q = 3

# Independent truth-validation tolerances. Frozen before estimator scoring.
TOL_F0_BS = 1e-8
TOL_ANALYTIC_REL = 1e-5
TOL_AUTODIFF_FD_REL = 5e-3
TOL_ORTH = 1e-7
TOL_RADIAL = 1e-7
TOL_INVARIANT = 1e-6
TOL_MC_KDIR = 2e-2
TOL_PACK_FRO = 1e-12
TOL_ORACLE_MED_REL = 0.01
TOL_ORACLE_RHO = 0.01
S1_RHO_MAX = 0.05
FD_STEP = 1e-5

ETA_LEVELS = (0.0, 0.05, 0.10, 0.25)
PRIMARY_ETA = 0.10

# Mechanical decision thresholds. Frozen after 64/128 smoke, before primary scores.
RHO_POINTWISE_OK = 0.70
COS_POINTWISE_OK = 0.70
SEED_SPEARMAN_OK = 0.50
RHO_PATCH_OK = 0.70
RHO_SHRINK_OK = 0.50
FALSE_F0_KDIR_MAX = 1e-3
FALSE_F1_TF_FRAC_MAX = 0.10
FALSE_F2_H_FRAC_MAX = 0.10
ROBUST_DROP_MAX = 0.30

DECISION_LABELS = (
    "both_instruments_valid_at_distinct_scales",
    "decoder_pointwise_valid_quadratic_patch_unreliable",
    "quadratic_patch_valid_decoder_pointwise_unstable",
    "both_valid_only_in_clean_uniform_regime",
    "density_conditioned_instrument_divergence",
    "noise_conditioned_instrument_divergence",
    "mean_vs_full_curvature_divergence",
    "known_curvature_fixture_audit_unresolved",
    "known_curvature_fixture_audit_blocked",
)

FIXTURES = ("F0", "F1", "F2", "F3", "F4", "F5")
SAMPLING = ("S0", "S1", "S2", "S3", "S4", "S5")
NOISE = ("N0", "N1", "N2", "N3", "N4", "N5")


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    seed: int = FIXTURE_SEED
    smoke: bool = False
    factorial_smoke: bool = False
    force: bool = False
    stage: str = "all"
    device: str = "cuda"
    n_anchors_override: int | None = None
    n_points_override: int | None = None
    k_override: int | None = None
    epochs_override: int | None = None
    decoder_seeds: tuple[int, ...] = DECODER_SEEDS
    skip_decoder: bool = False
    skip_oracle: bool = False
    skip_n86471: bool = False
    skip_suite_f: bool = False
    bounded: bool = False
    n_workers: int = 12
    oracle_n_qmc: int | None = None
    suites: list[str] | None = field(default=None)

    def n_anc(self) -> int:
        if self.n_anchors_override is not None:
            return int(self.n_anchors_override)
        if self.smoke:
            return N_ANCHORS_UNIT
        if self.factorial_smoke:
            return N_ANCHORS_SMOKE
        return N_ANCHORS_PRIMARY

    def n_points(self) -> int:
        if self.n_points_override is not None:
            return int(self.n_points_override)
        if self.smoke:
            return 2048
        if self.factorial_smoke:
            return 4096
        return PRIMARY_N

    def k_primary(self) -> int:
        if self.k_override is not None:
            return int(self.k_override)
        if self.smoke:
            return 256
        if self.factorial_smoke:
            return 512
        return PRIMARY_K

    def epochs(self) -> int:
        if self.epochs_override is not None:
            return int(self.epochs_override)
        if self.smoke:
            return 8
        if self.factorial_smoke:
            return 20
        return MAX_EPOCHS

    def seeds(self) -> tuple[int, ...]:
        if self.smoke or self.factorial_smoke:
            return (int(self.decoder_seeds[0]),)
        return tuple(self.decoder_seeds)

    def oracle_qmc(self) -> int:
        if self.oracle_n_qmc is not None:
            return int(self.oracle_n_qmc)
        if self.smoke:
            return 256
        if self.factorial_smoke:
            return 512
        return 4096
