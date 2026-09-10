# REPORT — known curvature point/patch fixture audit

## Decision

**mean_vs_full_curvature_divergence**

mean curvature discards traceless bending that full K_dir recovers

Decoder pointwise mean-curvature accuracy: {'rho_H': 0.09624542124542122, 'rho_Kdir': 0.12298534798534796, 'median_cosine_H': 0.4779270604052711, 'median_tensor_cos': 0.17404812172402456, 'rel_Kdir_F1': 0.8406178061620233, 'rel_H_F1': 0.27052449664071015, 'rel_Kdir_F4F5': 354.3332017355268}
Decoder pointwise full \(B^S\) accuracy: {'rho_H': 0.09624542124542122, 'rho_Kdir': 0.12298534798534796, 'median_cosine_H': 0.4779270604052711, 'median_tensor_cos': 0.17404812172402456, 'rel_Kdir_F1': 0.8406178061620233, 'rel_H_F1': 0.27052449664071015, 'rel_Kdir_F4F5': 354.3332017355268}
Decoder cross-seed stability: {'spearman_Kdir': nan}
Quadratic pointwise convergence: {'rho_Kdir_T1_smallest_k': nan, 'rho_Kdir_T1_largest_k': nan, 'rho_improves': nan}
Quadratic matched finite-patch accuracy: {'rho_Kdir_T2': 0.3571428571428572, 'rho_Kdir_T3': 0.3571428571428572, 'rho_Kdir_T1': 0.2478479853479853, 'rel_Kdir_T1_const': 0.9900860141386199}
False curvature on residual-flat F0: {'F0_decoder_Kdir': 2.8166032809445856, 'F0_quadratic_Kdir': 8.999926624654492e-15, 'F1_decoder_tf_frac': 0.7083373611378952, 'F2_decoder_H_frac': 2.2327745487176878}
Tests passed: 17/17
Runtime s: 13.643901586532593
Peak RSS MB: 3398.24609375

## What the two instruments may be called

See `MANUSCRIPT_RECOMMENDATION.md`. This audit does not pick a winner against
an unmatched target. Pointwise estimators were scored against pointwise truth;
finite-patch estimators against matched finite-patch truth.

## Fixture definitions

See `fixture_definitions.json`. Ambient rotation hash `251f756d4adcce56`.

## Outputs

All artifacts under this tree. No manuscript and no prior experiment tree was modified.
