"""Exact named-field definitions for inspect-curvature-definition."""

from __future__ import annotations

FIELDS = {
    "KHcross": {
        "name": "KHcross",
        "estimator": "Q finite-patch split-half",
        "formula": "K_H^{cross} = <H_A^S, H_B^S> with H^S = (1/d) tr(whiten B^S) in production Q tables",
        "normalization": "averaged 1/d mean-curvature vectors; signed; not clamped",
        "spatial_scale": "k=2048 neighbourhood, d=16 local PCA frame unless a tree says otherwise",
        "not": "not intrinsic curvature; not full B^S energy; not interchangeable with Kdircross or D-residual",
        "source_trees": [
            "outputs/geometry/physics_curvature_probe_submission_validation",
            "outputs/geometry/physics_cross_model_curvature_local_adaptation",
            "outputs/geometry/physics_q_geometry_resampling_stability",
        ],
    },
    "Kdircross": {
        "name": "Kdircross",
        "estimator": "Q finite-patch split-half full directional",
        "formula": "K_dir^{cross} = (2 <B_A,B_B>_F + <tr B_A, tr B_B>) / (d(d+2))",
        "normalization": "includes traceless energy; signed; not clamped",
        "spatial_scale": "same frozen k=2048, d=16 charts as KHcross when taken from FCR",
        "not": "not KHcross; most residual bending is trace-free",
        "source_trees": ["outputs/geometry/physics_cross_model_full_curvature_reconciliation"],
    },
    "D_full": {
        "name": "D_full",
        "estimator": "pointwise raw decoder",
        "formula": "II^E = (I-P_T) D^2 F; H^E = g^{ab} II^E_ab (unaveraged)",
        "normalization": "raw decoder image; includes off-sphere and radial behaviour",
        "spatial_scale": "pointwise at the decoded latent of an embedding",
        "not": "not D-residual; not Q",
        "source_trees": ["outputs/geometry/pointwise_decoder_curvature_reproduction"],
    },
    "D_residual": {
        "name": "D_residual",
        "estimator": "pointwise normalized decoder sphere-residual",
        "formula": "F̃=F/||F||; II^S = (I-xx^T-P_T) D^2 F̃; C_H = ||(1/d) tr_g II^S|| in the ViT-B probe-relation tree",
        "normalization": "differentiate through normalization; sphere-normal residual only",
        "spatial_scale": "pointwise",
        "not": "not D-full; not Q KHcross",
        "source_trees": [
            "outputs/geometry/physics_pointwise_residual_curvature_probe_relation",
            "outputs/geometry/known_curvature_dual_estimator_robustness",
        ],
    },
    "E_Q_task_aligned": {
        "name": "E_Q_task_aligned",
        "estimator": "Q residual probe-aligned split-half energy",
        "formula": "E_Q^{S,cross} = <<w_N, B_A^S>, <w_N, B_B^S>>_g",
        "normalization": "signed; never average-then-square; never clamp",
        "spatial_scale": "k=2048, d=16, leakage-safe w from task_aligned_curvature_v1 split",
        "not": "not generic KHcross; not D task-aligned energy",
        "source_trees": [
            "outputs/geometry/physics_task_aligned_curvature",
            "outputs/geometry/physics_cross_model_task_aligned_curvature",
        ],
    },
    "M_delta": {
        "name": "M_delta",
        "estimator": "label-Hessian mismatch magnitude",
        "formula": "M_Δ = ||H_y - B_w||_g",
        "normalization": "invariant g-norm; H_y from train-only quadratic",
        "spatial_scale": "same anchors/k/d; H_y needs 136 quadratic coefficients at d=16",
        "not": "decision label is label_hessian_unreliable; do not treat P1 as confirmed mechanism",
        "source_trees": ["outputs/geometry/physics_cross_model_hessian_mismatch"],
    },
}
