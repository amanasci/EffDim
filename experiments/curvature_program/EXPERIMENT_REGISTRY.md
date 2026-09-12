# Frozen experiment registry

Machine-readable twin: [`EXPERIMENT_REGISTRY.json`](EXPERIMENT_REGISTRY.json).
Numbers below are copied from frozen `COMPLETE.json` / `decision.json` / `summary.json`.
Trees that were host-only were pulled on 2026-09-12; hashes match
[`outputs/geometry/curvature_program_synthesis/SOURCE_MANIFEST.json`](../../../outputs/geometry/curvature_program_synthesis/SOURCE_MANIFEST.json).
A manuscript must never override these files.

Completion rule: an experiment without `COMPLETE.json` is **incomplete** unless its
own protocol uses another marker (submission-validation uses `decision.json`).

| Experiment | Code | Outputs locally | Complete | Label | Class | d, k, n | Runtime |
|---|---|---|---|---|---|---|---|
| `physics_curvature_probe_submission_validation` | `experiments/geometry/physics_curvature_probe_submission_validation` | yes | decision only | `claim_supported_but_scale_dependent` | confirmatory | d=16 | — |
| `physics_local_probe_adaptation` | `experiments/geometry/physics_local_probe_adaptation` | yes (pulled from host) | COMPLETE | `curvature_predicts_local_direction_adaptation` | confirmatory | 16, 2048, 512 | 1134 s |
| `physics_local_probe_adaptation_audit` | `experiments/geometry/physics_local_probe_adaptation_audit` | yes (pulled from host) | COMPLETE | `curvature_predicts_relative_local_adaptation` | secondary | 16, 2048, 512 | 22975 s |
| `physics_quadratic_label_chart_alignment` | `experiments/geometry/physics_quadratic_label_chart_alignment` | yes (pulled from host) | COMPLETE | `quadratic_chart_link_unresolved` | exploratory | 16, 2048, 512 | 160 s |
| `physics_quadratic_label_chart_alignment_audit` | `experiments/geometry/physics_quadratic_label_chart_alignment_audit` | yes (pulled from host) | COMPLETE | `geometry_regularized_quadratic_decoding` (COMPLETE only) | secondary | 16 / 136 quad | 8273 s |
| `physics_cross_model_curvature_local_adaptation` | `experiments/geometry/physics_cross_model_curvature_local_adaptation` | yes (pulled from host) | COMPLETE | `representation_specific_effect` | confirmatory | 16, 2048, 512 | 46602 s |
| `physics_cross_model_full_curvature_reconciliation` | `experiments/geometry/physics_cross_model_full_curvature_reconciliation` | yes (pulled from host) | COMPLETE | `full_curvature_partial_cross_model_replication` | secondary | 16, 2048, 512 | 9673 s |
| `pointwise_decoder_curvature_reproduction` | `experiments/geometry/pointwise_decoder_curvature_reproduction` | yes | COMPLETE | `colleague_decoder_results_reproduced` | mechanical | d=16 fixtures | 834 s |
| `known_curvature_point_patch_fixture_audit` | `experiments/geometry/known_curvature_point_patch_fixture_audit` | yes | COMPLETE | `mean_vs_full_curvature_divergence` | mechanical | fixtures | 4 s |
| `known_curvature_dual_estimator_robustness` | `experiments/geometry/known_curvature_dual_estimator_robustness` | yes | COMPLETE | `neither_estimator_validated` | confirmatory | fixtures | 612 s |
| `known_curvature_estimator_operating_characteristics` | `experiments/geometry/known_curvature_estimator_operating_characteristics` | yes | COMPLETE | utility labels; prior gate kept | secondary | fixtures | — |
| `known_curvature_instrument_failure_localization` | `experiments/geometry/known_curvature_instrument_failure_localization` | yes (COMPLETE/decision filled from host) | COMPLETE | `quadratic_tangent_estimation_failure` | mechanical | fixtures | — |
| `physics_pointwise_residual_curvature_probe_relation` | `experiments/geometry/physics_pointwise_residual_curvature_probe_relation` | yes | COMPLETE | `pointwise_residual_probe_relation_unresolved` | confirmatory | 16, 512 | 731 s |
| `physics_q_geometry_resampling_stability` | `experiments/geometry/physics_q_geometry_resampling_stability` | yes | COMPLETE | `q_global_and_adaptation_associations_geometry_robust` | confirmatory | 16, 2048, 512 | 1679 s |
| `physics_task_aligned_curvature` | `experiments/geometry/physics_task_aligned_curvature` | yes | COMPLETE | `quadratic_task_aligned_effect_only` | confirmatory | 16, 2048, 512 | 2196 s |
| `physics_cross_model_task_aligned_curvature` | `experiments/geometry/physics_cross_model_task_aligned_curvature` | yes | COMPLETE | `q_task_aligned_replicates_across_models` | confirmatory | 16, 2048, 512 | 4194 s |
| `physics_cross_model_pointwise_residual_curvature` | `experiments/geometry/physics_cross_model_pointwise_residual_curvature` | yes | COMPLETE | `cross_model_pointwise_patch_degradation_only` | confirmatory | 16, 512 | 3084 s |
| `physics_cross_model_hessian_mismatch` | `experiments/geometry/physics_cross_model_hessian_mismatch` | yes | COMPLETE | `label_hessian_unreliable` | exploratory | 16, 2048, 512 | 5157 s |
| `physics_curvature_component_predictive_decomposition` | `experiments/geometry/physics_curvature_component_predictive_decomposition` | yes | COMPLETE | `distinct_mean_and_traceless_predictive_roles` | exploratory | 16, 2048, 512 | 3673 s |
| `curvature_program_synthesis` | `experiments/geometry/curvature_program_synthesis` | yes | COMPLETE | documentation only | mechanical | — | — |
| `physics_ae_local_patch_scale_match` | `run_ae_local_patch_scale_match.py` | yes | **incomplete** | none | exploratory | 600-epoch AE | 259 s |

## Headline numbers (artifact-backed)

**ViT-B Q / probes.** `rho_ctl(KHcross, R_G^2) = -0.2404841119636992`;
`rho_ctl(KHcross, MSE_G) = +0.22704789227635297`;
`rho_ctl(KHcross, Δ_adapt) = +0.15334238492921803`;
mean `Δ_adapt = -0.1011990469212172` (patches worse on average).
Resampling 32+32 retains sign (conditional `frac_original_sign = 1.0`).

**Cross-model Q.** Label `representation_specific_effect`. Only ViT-B has both sides
(`C_G=0.22704789227635297`, `C_A=0.15334238492921803`).
DINOv3 `C_G=-0.12981029199711608`; CLIP `-0.27375793579267804`;
ConvNeXt `-0.18663614658793098`; ViT-L `C_G=-0.03830513069202687`, `C_A=+0.18130031843306899`.

**K_dir vs K_H.** Most residual bending is trace-free (aniso share 0.943).
ViT-B `ρ(K_dir, Δ_adapt) = -0.1594762627172955` vs `ρ(K_H, Δ_adapt) = +0.15334238492921803`.

**QLCA.** Held-out `median Δ_Q = 0.020581617601622228`. Audit: full rank 136 in quadratic-feature space → geometry-regularized quadratic decoding, not a low-dimensional chart constraint. Parent label remains `quadratic_chart_link_unresolved`.

**D-residual ViT-B.** `ρ_ctl(C_H, R_G^2) = +0.02970575697233952` (null). Patch association negative. Does not reproduce the Q story.

**Task-aligned ViT-B.** `P2 ρ̄_Q = +0.11614541311898466` Holm pass; `P1 ρ̄_D = -0.0016449827122028832` fail. Positive Q cells: photo_z, smooth_fraction, stellar_mass; `mag_r_desi` opposite.

**Task-aligned cross-model.** `R1 = +0.05794641314866791` (ViT-B excluded). CLIP negative.

**Hessian mismatch.** `P1 = +0.10804301439243084` Holm pass; `P2 = -0.011993127617845984` fail; **label `label_hessian_unreliable`**.

Git commits are **not recorded** in these COMPLETE files. Worktree HEAD at handoff: `dabe5e2`.
