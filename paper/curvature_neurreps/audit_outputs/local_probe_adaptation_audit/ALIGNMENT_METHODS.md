# ALIGNMENT_METHODS

## Global (A_H^G, A_B^G)

- Geometry packs: `physics_curvature_probe_multitarget/geometry_cache/k2048_aiXXXX.npz`
  (`T`, `x0u`, `UB`, `UNPCA`).
- Mean-curvature vector H: `physics_nested_dimension_curvature/H_vectors/{sample_id}.npz` field `H16`.
- Global probe weight: pooled `w_mag_r_desi` from `physics_global_probe_curvature_alignment/global_probe_weights.npz`.
- A_B^G: `projection_energies(w,T,x0u,UB,UN)["A_B_normal"]` (`global_probe_curvature_alignment.py`).
- A_H^G: `a_h_from_w_H(w,T,x0u,H)` (`global_probe_curvature_magnitude.py`).

## Patch (A_H^P, A_B^P)

- Patch weights reconstructed by strict global-fold OOF ambient ridge (α=100, no scaler), one w per outer fold.
- Fold-weighted mean of A_B^P and A_H^P across patch folds.
- Direction reliability: median pairwise cosine of fold weights ≥ 0.85.

## Pathway

- D_PG = arccos cosine between pooled global w and mean patch w (descriptive only).
- Not used as ordinary confounders in the primary adjustment set.
