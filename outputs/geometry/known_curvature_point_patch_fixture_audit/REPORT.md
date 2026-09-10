# REPORT — known curvature point/patch fixture audit

## Decision

**mean_vs_full_curvature_divergence**

mean curvature discards traceless bending that full K_dir recovers

Decoder pointwise mean-curvature accuracy: {'rho_H': 0.948527050359918, 'rho_Kdir': 0.9444003666414895, 'median_cosine_H': 0.9993176801285808, 'median_tensor_cos': 0.7232806320742441, 'rel_Kdir_F1': 0.9999840511156917, 'rel_H_F1': 0.997280983555233, 'rel_Kdir_F4F5': 0.06652025611476747}
Decoder pointwise full \(B^S\) accuracy: {'rho_H': 0.948527050359918, 'rho_Kdir': 0.9444003666414895, 'median_cosine_H': 0.9993176801285808, 'median_tensor_cos': 0.7232806320742441, 'rel_Kdir_F1': 0.9999840511156917, 'rel_H_F1': 0.997280983555233, 'rel_Kdir_F4F5': 0.06652025611476747}
Decoder cross-seed stability: {'spearman_Kdir': 0.15770537216328492}
Quadratic pointwise convergence: {'rho_Kdir_T1_smallest_k': 0.10977840577471074, 'rho_Kdir_T1_largest_k': 0.2795769210411875, 'rho_improves': -0.16979851526647674}
Quadratic matched finite-patch accuracy: {'rho_Kdir_T2': 0.2526994163424124, 'rho_Kdir_T3': 0.2526994163424124, 'rho_Kdir_T1': 0.33884243285153526, 'rel_Kdir_T1_const': 0.9315326127772189}
False curvature on residual-flat F0: {'F0_decoder_Kdir': 8.467652068555033, 'F0_quadratic_Kdir': 3.368999765911218e-15, 'F1_decoder_tf_frac': 0.5348961970293244, 'F2_decoder_H_frac': 2.130696000022515}
Tests passed: 17/17
Runtime s: 4.234332323074341
Peak RSS MB: 979.8515625
Bounded: True skipped=['n=86471', 'suite_F']

## What the two instruments may be called

See `MANUSCRIPT_RECOMMENDATION.md`. This audit does not pick a winner against
an unmatched target. Pointwise estimators were scored against pointwise truth;
finite-patch estimators against matched finite-patch truth.

## Fixture definitions

See `fixture_definitions.json`. Ambient rotation hash `251f756d4adcce56`.

## Outputs

All artifacts under this tree. No manuscript and no prior experiment tree was modified.
