# REPORT — known curvature point/patch fixture audit

## Decision

**mean_vs_full_curvature_divergence**

mean curvature discards traceless bending that full K_dir recovers

Decoder pointwise mean-curvature accuracy: {'rho_H': -0.13101995971433802, 'rho_Kdir': -0.12532617652444608, 'median_cosine_H': 0.2194926928163374, 'median_tensor_cos': 0.1324564870829827, 'rel_Kdir_F1': 0.979950841271189, 'rel_H_F1': 0.9255551365222245, 'rel_Kdir_F4F5': 1994455108222.8364}
Decoder pointwise full \(B^S\) accuracy: {'rho_H': -0.13101995971433802, 'rho_Kdir': -0.12532617652444608, 'median_cosine_H': 0.2194926928163374, 'median_tensor_cos': 0.1324564870829827, 'rel_Kdir_F1': 0.979950841271189, 'rel_H_F1': 0.9255551365222245, 'rel_Kdir_F4F5': 1994455108222.8364}
Decoder cross-seed stability: {'spearman_Kdir': nan}
Quadratic pointwise convergence: {'rho_Kdir_T1_smallest_k': nan, 'rho_Kdir_T1_largest_k': nan, 'rho_improves': nan}
Quadratic matched finite-patch accuracy: {'rho_Kdir_T2': 0.6547619047619048, 'rho_Kdir_T3': 0.6547619047619048, 'rho_Kdir_T1': 0.17673903589086246, 'rel_Kdir_T1_const': 0.934506882802829}
False curvature on residual-flat F0: {'F0_decoder_Kdir': 4.071133369724607, 'F0_quadratic_Kdir': 1.0271251738111967e-14, 'F1_decoder_tf_frac': 0.7231779674038372, 'F2_decoder_H_frac': 3.301100590494526}
Tests passed: 17/17
Runtime s: 811.4245042800903
Peak RSS MB: 14886.078125

## What the two instruments may be called

See `MANUSCRIPT_RECOMMENDATION.md`. This audit does not pick a winner against
an unmatched target. Pointwise estimators were scored against pointwise truth;
finite-patch estimators against matched finite-patch truth.

## Fixture definitions

See `fixture_definitions.json`. Ambient rotation hash `251f756d4adcce56`.

## Outputs

All artifacts under this tree. No manuscript and no prior experiment tree was modified.
