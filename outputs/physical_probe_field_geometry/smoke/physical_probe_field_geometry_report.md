# Physical probe field geometry

Inspired by arXiv:2602.02315. Dual/readout geometry of local physical probes.

## Corpus note

Official Legacy↔HSC embeddings lack physics labels. This run uses the
Physics (Smith42) multi-model corpus with the frozen holdout-20 split
(n=16384, test_size=0.2) and pipeline SAE checkpoints (F2048_k20).

## Smoke gate

- gate_pass: **True**
- n_models: 4
- variables: ['photo_z', 'mag_r_desi', 'smooth_fraction']
- representations: ['dense', 'sae', 'sae_sqrt_idf']

## Aggregate highlights

- mean_local_pearson: 0.02814832420579558
- mean_local_minus_global: -0.02138969323464116
- mean_frac_local_better: 0.37745098039215685
- mean_field_stable_rank: 4.2900743219275865
- mean_adjacent_similarity: -0.062373157018274845
- mean_bootstrap_stability: 0.8952969443793161
- mean_cross_model_cka: 0.8906167477284436
- mean_normalized_cka: 0.9941425312301013
- mean_dual_minus_primal: -0.10595076576375484
- pullback_topk16: 0.28331907530583256
- pullback_lowrank_match16: 0.004380320940028818
- global_W_stable_rank_vit_dino: 3.8966322114091954
- mean_field_vs_W_rank_ratio: 1.1009697834366834
- gate_pass: True

## Research questions

1. **Do local readout directions vary across physical parameter space?**
   - mean frac centres where local>global=0.377; mean local−global=-0.021

2. **Is the variation smooth and bootstrap-stable?**
   - adjacent similarity=-0.062; bootstrap CKA=0.895

3. **Does local probe transfer decay with physical distance?**
   - See transfer heatmaps and median transfer lengths in field_rank_results.csv / figures.

4. **Is a global probe inadequate in high-curvature regions?**
   - See figures/local_advantage_vs_curvature.png and field_curvature_results.parquet.

5. **What is the effective rank of each physical readout field?**
   - mean field stable rank (SAE)=4.29

6. **Is field rank much lower than the global SAE transport rank?**
   - field/W rank ratio=1.101; W stable rank=3.9

7. **Do independently trained models induce similar field kernels?**
   - mean cross-model CKA=0.891; normalized=0.994

8. **Is cross-model field agreement greater than generic physical-distance smoothness?**
   - gate cka_above_distance=True

9. **Is dual/readout geometry more reproducible than primal activation geometry?**
   - mean dual−primal CKA=-0.106

10. **Does the existing global transport map align corresponding local probes?**
   - full-W pullback agreement=0.283 (see transport_probe_pullback_results for full W row)

11. **Does sparse high-rank transport preserve field geometry better than low-rank transport?**
   - topk16=0.283 vs lowrank=0.004; gate=True

12. **Are local transport maps simpler than the global map?**
   - Stage D not run in this smoke.

13. **Can global high rank be explained as the union of rotating local low-rank maps?**
   - Stage D not run.

14. **Which physical variables produce the strongest shared field geometry?**
   - {'mag_r_desi': 0.8066114339576681, 'photo_z': 0.926896165624986, 'smooth_fraction': 0.9383426436026765}

15. **Does joint physical geometry factorize into marginal fields?**
   - Stage E not run in this smoke.

16. **What is the best-supported scientific statement about rank collapse and curvature?**
   - Global probes perform similarly to local probes — field curvature may be weak for these variables.


## Interpretation

Global probes perform similarly to local probes — field curvature may be weak for these variables. Sparse high-rank W preserves readout-field geometry better than parameter-matched low-rank maps.
