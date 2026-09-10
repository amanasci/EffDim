"""Frozen recovered constants. Do not retune after seeing scores."""

from __future__ import annotations

CURVATURE_EXPERIMENTS_SHA = "97efb2eb6cd7dec7f2c568f53c534752ff3c32c8"
FIXTURE_VALIDITY_AUDIT_SHA = "dcd2208803f27224ee182cce47fdee21c1bc6ba5"
COLLEAGUE = {
    "name": "Austin Lutterbach",
    "email": "rmhslutterbach@gmail.com",
    "role": (
        "Author of notebooks/pu_manifold/decoder_curvature.py (phase 02.6) and of the "
        "d=16 cubic/ridge fidelity table 09-FIXTURE-FIDELITY-D16.md on origin/fixture-validity-audit. "
        "Not the nested-chart K_H^cross / local-quadratic estimator on curvature-experiments."
    ),
}

OUT_REL = "outputs/geometry/pointwise_decoder_curvature_reproduction"
WALL_S = 60 * 60
RESERVE_WRITE_S = 90.0

# 07_instrument_fixture_sweep_run.py (R2–R4 / optional R5)
SWEEP_N = 5000
SWEEP_SEED = 20260816
SWEEP_K_CLOUD = 231  # point-cloud arm only; decoder does not use k
SWEEP_EPOCHS = 400
SWEEP_TORCH_INIT = 0
AE_HIDDEN = (250, 250, 250)
AE_ACTIVATION = "silu"
SWEEP_CFG = {
    "seed": SWEEP_SEED,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 128,
    "max_epochs": SWEEP_EPOCHS,
    "early_stop_patience": SWEEP_EPOCHS + 1,
    "early_stop_min_delta": 1e-9,
    "lip_weight": 0.0,
    "fps_pretrain_epochs": 0,
    "wallclock_ceiling_s": float("inf"),
}

# 09_instrument_adjudication_run.py --mode swiss-roll (R1)
SWISS_N = 3000
SWISS_RANDOM_STATE = 0
SWISS_D = 2
SWISS_K = 256
SWISS_N_ANCHORS = 256
SWISS_EPOCHS = 300
SWISS_SPLIT_SEED = 20260813
SWISS_HOLDOUT_FRACTION = 0.2
SWISS_ANCHOR_DRAW_SEED = 20260902
SWISS_TORCH_INIT = 0
SWISS_CFG = {
    "seed": 0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 128,
    "max_epochs": SWISS_EPOCHS,
    "early_stop_patience": 600 + 1,
    "early_stop_min_delta": 1e-9,
    "lip_weight": 0.0,
    "fps_pretrain_epochs": 0,
    "wallclock_ceiling_s": float("inf"),
}

TOL = {
    "var_explained": 0.002,
    "rho": 0.03,
    "rho_r1": 0.05,  # R1: historical env noted as orchestrator-brief, not JSONL
    "cosine": 0.01,
    "ratio": 0.05,
}

HISTORICAL = {
    "R1": {
        "fixture": "swiss_roll",
        "d": 2,
        "D": 3,
        "n": 3000,
        "var_explained": None,
        "rho": 0.553,
        "cosine": 0.9998,
        "ratio": 1.046,
        "source": "09-SUPPLEMENT-02-INSTRUMENT-ADJUDICATION.md §3 (orchestrator brief, not JSONL)",
        "note": (
            "Quoted there as ours H_tan (d=2, k=256, 256 anchors). The swiss-roll runner "
            "actually scores ambient H_vec (no sphere projection) via _fidelity_axes. "
            "SWISS_EPOCHS=300. k is for the Q arm / knn panel, not the decoder."
        ),
    },
    "R2": {
        "fixture": "cubic",
        "d": 16,
        "D": 28,
        "n": 5000,
        "var_explained": 0.9989458270763205,
        "rho": 0.9422766741710669,
        "cosine": 0.9932794885913858,
        "ratio": 0.9610396277738668,
        "source": "09-FIXTURE-FIDELITY-D16.md table, plain-decoder arm",
    },
    "R3": {
        "fixture": "ridge",
        "d": 16,
        "D": 28,
        "n": 5000,
        "var_explained": 0.9994405376046311,
        "rho": 0.987230725569229,
        "cosine": 0.9995659766206267,
        "ratio": 0.9923055974100696,
        "source": "09-FIXTURE-FIDELITY-D16.md table, plain-decoder arm",
    },
    "R4": {
        "fixture": "ridge",
        "d": 16,
        "D": 768,
        "n": 5000,
        "var_explained": 0.9993710455021906,
        "rho": 0.9881738890309554,
        "cosine": 0.9994735427208619,
        "ratio": 0.9853165903890537,
        "source": "09-FIXTURE-FIDELITY-D16.md table, plain-decoder arm",
    },
    "R5": {
        "fixture": "cubic",
        "d": 16,
        "D": 768,
        "n": 5000,
        "var_explained": 0.9985989197885459,
        "rho": 0.8376129732485188,
        "cosine": 0.9719196818369312,
        "ratio": 0.9972002312554775,
        "source": "09-FIXTURE-FIDELITY-D16.md table, plain-decoder arm",
        "optional": True,
    },
}

# Historical convention recovered from decoder_curvature.py
H_IS_AVERAGED = False  # H = g^{ab} II_ab, NOT (1/d) times that
II_REMOVES_SPHERE_RADIAL = False  # II = (I-P_T) D²F, not (I-xx^T-P_T) D²F
F_DIFFERENTIATED_AFTER_NORMALIZE = False  # raw model.decode
