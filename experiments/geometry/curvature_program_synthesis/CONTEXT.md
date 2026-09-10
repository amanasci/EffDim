# Curvature programme synthesis — experiment context

Documentation-only index for the curvature–probe research programme. This folder does **not** re-run estimators, train models, refit probes, overwrite historical `decision.json` labels, or edit manuscripts.

The authoritative write-up lives in the output tree:

```text
outputs/geometry/curvature_program_synthesis/CURVATURE_PROGRAM_SUMMARY.md
outputs/geometry/curvature_program_synthesis/SOURCE_MANIFEST.json
outputs/geometry/curvature_program_synthesis/CLAIM_MATRIX.csv
outputs/geometry/curvature_program_synthesis/COMPLETE.json
```

Canonical numerical artifacts remain on the science host under `/home/angus/platonic-universe/outputs/geometry/`. This worktree is a partial mirror. Frozen output trees were read, not written.

## What this is

A single synthesis of already-completed geometry experiments. It separates estimands that were historically discussed as one “curvature”:

| Name | Role |
|---|---|
| D-full | Historical pointwise full Euclidean decoder mean curvature (raw decode, unaveraged trace). |
| D-residual | Pointwise sphere-residual decoder curvature (differentiate through normalization). |
| Q \(K_H^{\mathrm{cross}}\) | Split-half finite-patch sphere-normal quadratic-bending statistic (trace). |
| Q \(K_{\mathrm{dir}}^{\mathrm{cross}}\) | Same family, full directional / tensor statistic. |
| Intrinsic Scal | Gauss-type scalar from \(B^S\); not \(K_H\) or \(K_{\mathrm{dir}}\). |
| \(\Delta_Q\) | Held-out unrestricted quadratic **label** gain; not a curvature estimator. |

Q is not a validated pointwise manifold-curvature estimator.

## Programme map (read-only sources)

Paths are relative to the repository root. `(host)` means present on the Ubuntu host and absent from this worktree.

### Real-data Q and probes

| Tree | Purpose | Frozen label |
|---|---|---|
| `outputs/geometry/physics_curvature_probe_submission_validation/` | NeurReps-era \(K_H\) vs global OOF error at \(d=16\) | `claim_supported_but_scale_dependent` |
| `outputs/geometry/physics_local_probe_adaptation/` `(host)` | ViT-B \(K_H\) vs global \(R^2\)/MSE and \(\Delta_{\mathrm{adapt}}\) | `curvature_predicts_local_direction_adaptation` |
| `outputs/geometry/physics_local_probe_adaptation_audit/` `(host)` | Relative-not-absolute restatement | `curvature_predicts_relative_local_adaptation` |
| `outputs/geometry/physics_quadratic_label_chart_alignment/` `(host)` | Held-out \(\Delta_Q\) and Hessian–bending alignment | `quadratic_chart_link_unresolved` |
| `outputs/geometry/physics_quadratic_label_chart_alignment_audit/` `(host)` | Geometry-regularizer interpretation | `geometry_regularized_quadratic_decoding` (COMPLETE only; **not** in `decision.json`) |
| `outputs/geometry/physics_cross_model_curvature_local_adaptation/` `(host)` | Five-encoder \(K_H\) table | `representation_specific_effect` |
| `outputs/geometry/physics_curvature_component_predictive_decomposition/` | Mean vs traceless Q predictive roles | `distinct_mean_and_traceless_predictive_roles` |
| `outputs/geometry/physics_cross_model_full_curvature_reconciliation/` `(host)` | Recovered \(K_{\mathrm{dir}}\) vs \(K_H\) | `full_curvature_partial_cross_model_replication` |
| `outputs/geometry/physics_q_geometry_resampling_stability/` | 32+32 ViT-B Q refits | `q_global_and_adaptation_associations_geometry_robust` |
| `outputs/geometry/physics_adaptive_dataset_curvature_probe/` | Dataset-sweep predecessor | `dataset_specific_curvature_probe_associations` (COMPLETE/`summary.json`; no `decision.json`) |
| `outputs/geometry/physics_multimodel_graph_prior_quadratic/` `(host)` | Frozen embeddings / neighbourhoods / Q features | data source; no COMPLETE |

### Known-answer geometry

| Tree | Purpose | Frozen label |
|---|---|---|
| `outputs/geometry/known_curvature_point_patch_fixture_audit/` | Mixed-estimand first fixture audit | `mean_vs_full_curvature_divergence` |
| `outputs/geometry/known_curvature_instrument_failure_localization/` | Q ablation; PCA is principal loss | `quadratic_tangent_estimation_failure` (COMPLETE on host; worktree docs diverge) |
| `outputs/geometry/pointwise_decoder_curvature_reproduction/` | Austin D-full formula/protocol | `colleague_decoder_results_reproduced` |
| `outputs/geometry/known_curvature_dual_estimator_robustness/` | Matched-target D-full / D-residual / Q gates | `neither_estimator_validated` |
| `outputs/geometry/known_curvature_estimator_operating_characteristics/` | Rank/reliability under stress | `d_full_useful_on_non_spherical_clean_geometry`; `q_moderately_informative_sampling_dependent_statistic`; `d_residual_useful_but_stress_sensitive` |

### Pointwise residual on real data

| Tree | Purpose | Frozen label |
|---|---|---|
| `outputs/geometry/physics_ae_local_patch_scale_match/` | 600-epoch AE \(H\), **not** the D-residual protocol | none (no COMPLETE) |
| `outputs/geometry/physics_pointwise_residual_curvature_probe_relation/` | Fixture-validated \(C_H\) vs frozen ViT-B probes | `pointwise_residual_probe_relation_unresolved` |

## Corresponding experiment packages

Code for the later audits lives under `experiments/geometry/` with matching directory names (for example `known_curvature_estimator_operating_characteristics/`, `physics_q_geometry_resampling_stability/`, `physics_pointwise_residual_curvature_probe_relation/`). Those packages are frozen relative to this synthesis: do not launch them from this folder.

This directory contains only `CONTEXT.md`. It is not a runnable pipeline.

## Headlines (see the summary for numbers and paths)

1. D-residual is the strongest validated **pointwise** geometric instrument.
2. Q \(K_H^{\mathrm{cross}}\) is the statistic that most robustly predicts the original ViT-B global-\(R^2\)/MSE and relative-adaptation associations, including geometry-resampling of the **same** representation population.
3. Those are not the same quantity.
4. The joint ViT-B Q pattern is not a five-encoder law. Cross-model D-residual has not been run.
5. Historical decision labels remain in force side by side. None is a master label.

## Explicitly not done here

- No new experiments, decoder training, probe refits, or curvature re-estimation.
- No manuscript edits.
- No overwrite of frozen `decision.json` files.
- The proposed cross-model D-residual run (DINOv3, CLIP, ConvNeXt-B, ViT-L; three seeds; frozen \(d=16\); same 512 anchors) is described in §14 of the summary and is **not** launched.
