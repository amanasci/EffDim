"""Frozen operating-characteristics constants. Do not retune after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass

from geometry.known_curvature_dual_estimator_robustness.config import (  # noqa: F401
    AE_ACTIVATION,
    AE_HIDDEN,
    AMBIENT_ROTATION_SEED,
    ANCHOR_HASH_SEED,
    CELLS,
    DATA_SEED,
    D_AMB,
    D_LAT,
    F4_AMPLITUDES,
    F4_BUMP_SCALE,
    F4_CENTERS,
    F4_WIDTHS,
    K_PRIMARY,
    MAX_EPOCHS,
    N_ANCHORS,
    N_MAX,
    N_SPARSE,
    N_SPLITS_Q,
    TRAIN_CFG,
    TRAIN_CFG_SEED,
    TORCH_INIT_SEED,
)

CURVATURE_EXPERIMENTS_SHA = "97efb2eb6cd7dec7f2c568f53c534752ff3c32c8"
OUT_REL = "outputs/geometry/known_curvature_estimator_operating_characteristics"
WALL_S = 45 * 60
RESERVE_WRITE_S = 120.0
MAX_NEW_AES = 8
MAX_ANCHORS = 64
N_BOOT = 200
BOOT_SEED = 20260911
CAL_SPLIT_SEED = 20260907
DRAW_SEEDS = {"A": 20260911, "B": 20260912}
DECODER_INIT_SEEDS = (0, 1)
REPEAT_FIXTURE = "F4"
REPEAT_CONDITIONS = (("S0", "N0"), ("S3", "N4"))

# Rank-target numerical tolerance: F0 residual energy gate was 1e-6; relative floor
# follows decoder reproduction float64 jets on unit-sphere immersions.
TOL_TRUTH_ABS = 1e-6
TOL_TRUTH_REL = 1e-4
RANK_DEGENERATE_IQR_MULT = 10.0
MIN_CLEAN_RHO_FOR_RETAINED = 0.10
EPS = 1e-12

# Frozen operating-characteristic bands (not universal validity thresholds).
BAND_RANK_STRONG = 0.75
BAND_RANK_MODERATE = 0.40
BAND_PAIR_STRONG = 0.80
BAND_PAIR_MODERATE = 0.65
BAND_REL_STRONG = 0.75
BAND_REL_MODERATE = 0.50
BAND_CEIL_STRONG = 0.80
BAND_CEIL_MODERATE = 0.50

HEADLINE_TOL_RHO = 0.02
HEADLINE_TOL_ENERGY = 5e-4

# Reused read-only trees.
DUAL_OUT = "outputs/geometry/known_curvature_dual_estimator_robustness"
REPRO_OUT = "outputs/geometry/pointwise_decoder_curvature_reproduction"
PATCH_OUT = "outputs/geometry/known_curvature_point_patch_fixture_audit"
FAIL_OUT = "outputs/geometry/known_curvature_instrument_failure_localization"
PHYS_OUT = "outputs/geometry/physics_pointwise_residual_curvature_probe_relation"
FCR_VIT = "outputs/geometry/physics_cross_model_full_curvature_reconciliation/tables/vit_base_per_anchor_curvature.parquet"

PRIOR_DECISION_LABEL = "neither_estimator_validated"

OPERATING_F4_CELLS = (
    "F4_S0_N0",  # clean uniform
    "F4_S1_N0",  # uniform sparse
    "F4_S2_N0",  # smoothly non-uniform dense
    "F4_S0_N1",  # low normal noise
    "F4_S0_N2",  # moderate normal noise
    "F4_S0_N3",  # moderate isotropic noise
    "F4_S3_N4",  # non-uniform sparse + heteroscedastic noise
)

STRUCTURAL_CELLS = (
    "F0_S0_N0",
    "F1_S0_N0",
    "F2_S0_N0",
    "F0_S3_N4",
    "F2_S3_N4",
)

CONDITION_LABELS = {
    "F4_S0_N0": "clean uniform",
    "F4_S1_N0": "uniform sparse",
    "F4_S2_N0": "smoothly non-uniform dense",
    "F4_S0_N1": "low normal noise",
    "F4_S0_N2": "moderate normal noise",
    "F4_S0_N3": "moderate isotropic noise",
    "F4_S3_N4": "non-uniform sparse + heteroscedastic noise",
}

HEADLINES = {
    "d_full_cubic_rho": {"path": REPRO_OUT, "cell": "R2", "expect": 0.9392535779861431},
    "d_full_ridge_rho": {"path": REPRO_OUT, "cell": "R3", "expect": 0.9897837249033489},
    "d_res_f0_energy_frac": {"expect": 0.0009638843433637421},
    "d_res_f4_clean_rho": {"expect": 0.8193223443223443},
    "d_res_f4_scal_rho": {"expect": 0.8617673992673993},
    "d_res_f4_combined_stress_rho": {"expect": 0.7369047619047618},
    "q_f4_t2_tensor_cos": {"expect": 0.6176322563326069},
    "t2_t3_tensor_cos": {"expect": 0.7481855145541307},
}


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cpu"
    q_device: str = "cpu"
    n_workers: int = 8
    skip_repeats: bool = False
    skip_oracles: bool = False
