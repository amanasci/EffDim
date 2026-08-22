# Physical probe field geometry — validation / audit

Stages V0–V5. Same Physics holdout-20 smoke models; no expansion.

- gate_pass: **False**
- elapsed_s: 7435.9

## Gate

- heldout_cka_high: False
- train_test_cka_close: False
- beats_indep_perm_null: False
- beats_shared_perm_null: False
- distance_residual_positive: False
- sparse_beats_lowrank: True

## Highlights

- mean_train_cross_model_cka: 0.9358843942292192
- mean_test_cross_model_cka: 0.6410169178838919
- test_minus_train_cka: -0.2948674763453273
- mean_distance_residual_pearson: 0.15670557215572997
- mean_sparse_minus_lowrank_k16: 0.2653905923024886
- mean_abs_test_pearson: 0.06130973147060548
- mean_adjacent_similarity_test: -0.0776612390707461
- smoke_reported_W_stable_rank: 3.8966322114091954
- audited_smoke_object_stable_rank: 3.9071135165763278
- full_sae_prestd_W_stable_rank: 3.907113490273913
- probe_pca_W_stable_rank: 36.557768335233945
- mean_field_restricted_W_stable_rank: 3.317432746416531
- gate_pass: False

## Interpretation

Held-out field agreement is weak relative to the smoke claim; treat prior cross-model CKA with caution. Sparse top-k transport preserves held-out probe pullback better than parameter-matched low-rank transport. Audited full SAE Ridge map (α=1.0) has low stable rank (3.91); the smoke value ≈3.90 traced to W_sae_full_raw_with_internal_scaler — not a high-rank unregularized map.

## Research questions

1. **Does cross-model field CKA remain high on held-out test activations?**
   - mean test CKA=0.641 (train=0.936)

2. **How much does train-kernel CKA differ from test-kernel CKA?**
   - test−train=-0.295

3. **Does true field agreement exceed the conditioned label-permutation null?**
   - indep null gate=False; see conditioned_permutation_null.parquet

4. **Does it exceed the shared-permutation null?**
   - shared null gate=False

5. **Does field agreement survive removal of generic physical-distance structure?**
   - mean residual Pearson=0.157; residual_ok=False

6. **Are the fields physically predictive despite high mutual CKA?**
   - mean |test Pearson|=0.061 (typically small in smoke)

7. **Useful physical readout geometry or reproducible conditioning geometry?**
   - Held-out field agreement is weak relative to the smoke claim; treat prior cross-model CKA with caution. Sparse top-k transport preserves held-out probe pullback better than parameter-matched low-rank transport. Audited full SAE Ridge map (α=1.0) has low stable rank (3.91); the smoke value ≈3.90 traced to W_sae_full_raw_with_internal_scaler — not a high-rank unregularized map.

8. **What exact matrix produced the previous stable-rank ≈3.90?**
   - W_sae_full_raw_with_internal_scaler (Ridge α=1.0 on C_raw with fit-internal StandardScaler); audited SR=3.907

9. **Stable rank of true full 2048×2048 SAE transport?**
   - W_sae_full_prestandardized SR=3.907

10. **Stable rank of PCA/probe-space transport?**
   - W_probe_pca SR=36.558

11. **Rank of transport restricted to physical probe-field subspace?**
   - mean W_field SR=3.317

12. **Does sparse transport preserve held-out readout functions better than matched low-rank?**
   - mean Δ(topk16−lowrank)=0.265; sparse_ok=True

13. **Does sparse transport also preserve mKNN better where available?**
   - Not re-measured here; Stage 0 smoke mKNN was vit↔dino shared-basis only. See prior SAE shared-basis pipeline for mKNN tables.

14. **Is the original curved-local-field hypothesis still supported?**
   - No — adjacent similarity (test)=-0.078; local−global remains ≤0.

15. **Strongest scientifically defensible headline after these controls?**
   - Held-out field agreement is weak relative to the smoke claim; treat prior cross-model CKA with caution. Sparse top-k transport preserves held-out probe pullback better than parameter-matched low-rank transport. Audited full SAE Ridge map (α=1.0) has low stable rank (3.91); the smoke value ≈3.90 traced to W_sae_full_raw_with_internal_scaler — not a high-rank unregularized map.

## Outputs

See `heldout_kernel_results.parquet`, `conditioned_permutation_null.parquet`,
`distance_residual_results.parquet`, `transport_rank_audit.csv`,
`sparse_lowrank_frontier.parquet`, and `figures/`.
