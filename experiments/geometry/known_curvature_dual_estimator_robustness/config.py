"""Frozen constants. Do not retune after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass

CURVATURE_EXPERIMENTS_SHA = "97efb2eb6cd7dec7f2c568f53c534752ff3c32c8"
FIXTURE_VALIDITY_AUDIT_SHA = "dcd2208803f27224ee182cce47fdee21c1bc6ba5"
REPRODUCTION_COMMIT = CURVATURE_EXPERIMENTS_SHA

OUT_REL = "outputs/geometry/known_curvature_dual_estimator_robustness"
WALL_S = 60 * 60
RESERVE_WRITE_S = 90.0

D_LAT = 16
D_AMB = 28  # primary ambient; 768 training forbidden unless cached .pt exists
N_MAX = 5000
N_SPARSE = 1500
N_ANCHORS = 64
K_PRIMARY = 1024
K_SECONDARY = 512
N_SPLITS_Q = 3
N_ORACLE_MAX = 2048
MAX_AES = 12
MAX_EPOCHS = 400  # reproduction R2–R4 protocol
TORCH_INIT_SEED = 0
TRAIN_CFG_SEED = 20260816
DATA_SEED = 20260816
ANCHOR_HASH_SEED = 20260907
AMBIENT_ROTATION_SEED = 20260907
DIRECTION_SEED = 20260907
N_TANGENT_DIRS = 32

AE_HIDDEN = (250, 250, 250)
AE_ACTIVATION = "silu"
TRAIN_CFG = {
    "seed": TRAIN_CFG_SEED,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 128,
    "max_epochs": MAX_EPOCHS,
    "early_stop_patience": MAX_EPOCHS + 1,
    "early_stop_min_delta": 1e-9,
    "lip_weight": 0.0,
    "fps_pretrain_epochs": 0,
    "wallclock_ceiling_s": float("inf"),
}

BETA_DENSITY = float(__import__("math").log(10.0) / 2.0)  # 10:1 ratio
N1_RMS = 0.01
N2_RMS = 0.05
N3_RMS = 0.05
N4_RMS_DENSE = 0.01
N4_RMS_SPARSE = 0.05

# Gates (frozen)
GATE_D_FULL_CLEAN_RHO = 0.90
GATE_D_FULL_CLEAN_COS = 0.95
GATE_D_FULL_CLEAN_RATIO = (0.8, 1.2)
GATE_D_FULL_STRESS_RHO = 0.75
GATE_D_FULL_STRESS_COS = 0.90
GATE_D_FULL_STRESS_RATIO = (0.7, 1.3)
GATE_F0_RES_ENERGY_FRAC = 0.05
GATE_F1_H_REL = 0.25
GATE_F4_RES_RHO = 0.75
GATE_Q_T2_COS = 0.80
GATE_Q_T2_RATIO = (0.7, 1.3)
GATE_Q_T2_SCAL_RHO = 0.75
GATE_Q_PW_RHO = 0.75
GATE_Q_CONST_CAL = 0.30
GATE_SAMP_RHO_DROP = 0.15
GATE_SAMP_COS_DROP = 0.10
PROJ_ERR_INVALID = 0.25  # mark tensor comparison invalid

CELLS = (
    ("F0", "S0", "N0"),
    ("F1", "S0", "N0"),
    ("F2", "S0", "N0"),
    ("F4", "S0", "N0"),
    ("F4", "S1", "N0"),
    ("F4", "S2", "N0"),
    ("F4", "S0", "N1"),
    ("F4", "S0", "N2"),
    ("F4", "S0", "N3"),
    ("F4", "S3", "N4"),
    ("F0", "S3", "N4"),
    ("F2", "S3", "N4"),
)

F1_C = 0.6
F4_BUMP_SCALE = 0.8
F4_AMPLITUDES = (0.85, -0.70, 0.60, -0.50)
F4_WIDTHS = (0.80, 0.95, 0.70, 1.05)

# Exact bump centers from the sealed known-curvature fixture audit (read-only import).
from geometry.known_curvature_point_patch_fixture_audit.config import F4_CENTERS  # noqa: E402


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cpu"
    q_device: str = "cpu"
    n_workers: int = 8
    skip_k512: bool = False
