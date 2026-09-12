"""Frozen constants. Do not retune after seeing scores."""

from __future__ import annotations

from dataclasses import dataclass, field

OUT_REL = "outputs/geometry/physics_cross_model_hessian_mismatch"
WALL_S = 90 * 60
PROJECTED_CAP_S = 85 * 60
RESERVE_WRITE_S = 180.0

REFERENCE = "vit_base"
REPLICATION = ("dinov3", "clip_base", "convnext_base", "vit_large")
ALL_MODELS = (REFERENCE, *REPLICATION)

TARGETS = ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass")
D_LAT = 16
K = 2048
N_ANCHORS = 512
N_SMOKE = 32
PROBE_ALPHA = 100.0
TRAIN_FRAC = 0.60
MIN_EVAL = 128
SPLIT_SALT = "task_aligned_curvature_v1"

DECODER_SEEDS = (0, 1, 2)
PROTOCOL = "reproduction_plain_ae_400"
SEED_RHO_GATE = 0.70
MAX_NEW_DECODERS = 12

MULTISCALE_K = (16, 64, 256, 1024, 2048)
PRIMARY_CONTROLS = tuple(f"log_radius_k{k}" for k in MULTISCALE_K) + (
    "local_eval_label_variance",
    "local_evaluation_count",
)
PAPER_CONTROLS = ("log_radius_k2048", "local_eval_label_variance", "local_evaluation_count")

# Frozen before outcomes. Paper OLS is primary; this ridge is the sole fallback.
HESS_RIDGE_PAPER = 0.0
HESS_RIDGE_STAB = 1.0

N_PERM = 10000
N_BOOT = 2000
INFER_SEED = 0

PAPER_TABLE2 = {
    "mag_r_desi": {"shape": 0.12, "sphere": -0.26, "mismatch": -0.39, "alignment": 0.35},
    "photo_z": {"shape": -0.06, "sphere": -0.33, "mismatch": -0.45, "alignment": 0.26},
    "smooth_fraction": {"shape": 0.07, "sphere": -0.13, "mismatch": -0.09, "alignment": 0.19},
    "stellar_mass": {"shape": 0.00, "sphere": -0.35, "mismatch": -0.04, "alignment": 0.01},
}

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_CMCLA = "outputs/geometry/physics_cross_model_curvature_local_adaptation"
SOURCE_FCR = "outputs/geometry/physics_cross_model_full_curvature_reconciliation"
SOURCE_PRCR = "outputs/geometry/physics_pointwise_residual_curvature_probe_relation"
SOURCE_CMPR = "outputs/geometry/physics_cross_model_pointwise_residual_curvature"
SOURCE_TAC = "outputs/geometry/physics_task_aligned_curvature"
SOURCE_ALIGN = "outputs/geometry/physics_global_probe_curvature_alignment"

DECISION_LABELS = (
    "cross_model_hessian_mismatch_replication",
    "partial_cross_model_hessian_mismatch_replication",
    "vitb_specific_hessian_mismatch_effect",
    "task_aligned_curvature_without_label_mismatch",
    "label_hessian_unreliable",
    "decoder_geometry_unreliable_across_models",
    "hessian_mismatch_null",
    "cross_model_hessian_mismatch_unresolved",
)

PRESERVED = (
    SOURCE_MM,
    SOURCE_CMCLA,
    SOURCE_FCR,
    SOURCE_PRCR,
    SOURCE_CMPR,
    SOURCE_TAC,
    SOURCE_ALIGN,
    "outputs/geometry/physics_cross_model_task_aligned_curvature",
    "paper",
    "papers",
    "submissions",
)


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cpu"
    hessian_device: str = "cpu"
    smoke: bool = False
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    models_override: list[str] | None = field(default=None)

    def models(self) -> tuple[str, ...]:
        if self.models_override:
            return tuple(self.models_override)
        return ALL_MODELS

    def n_anc(self) -> int:
        return N_SMOKE if self.smoke else N_ANCHORS

    def n_perm_eff(self) -> int:
        return 80 if self.smoke else self.n_perm

    def n_boot_eff(self) -> int:
        return 80 if self.smoke else self.n_boot
