# Methods — estimator operating characteristics

This audit does not reverse frozen exact-recovery labels. It scores two historical
measures plus one diagnostic on the already-computed D=28 fixture panel.

## Estimands

- **D-full** (historical): H_D^E = g^{ab} II^E_{D,ab} with II^E=(I-P_T)D^2 F from raw `decode`. Scored against analytic full Euclidean truth only.
- **Q** (historical): split-half K_H^{cross} and frozen K_{dir}^{cross} from `estimator_q.fit_anchor_quadratic`. Primary utility is rank recovery of K_H^{cross} against the sampling-matched finite-patch scalar K_{H,T2}^* = ||H_{patch,T2}^S||^2 using the same Hessian/whitening normalization as production (`kdir_from_pair` self-cross).
- **D-residual** (diagnostic only): H_D^S = g^{ab} B^S_{D,ab} through F̃=F/||F||.

On unit-sphere fixtures ||H^E||^2 = d^2 + ||H^S||^2. Rank targets with negligible dynamic range are marked `rank_target_degenerate`; Spearman is not a pass/fail metric there. Vector cosine, residualized magnitude sqrt(max(||H^E||^2-d^2,0)), and calibration are used instead.

## Reuse

Existing dual-estimator per-anchor parquet, cell-level metric CSVs, and the pointwise decoder reproduction R2/R3 cubic/ridge cells are read-only. Clouds, anchors, T2/T3 oracles, and Q neighbourhoods are regenerated from frozen seeds (DATA_SEED=20260816, ANCHOR_HASH_SEED=20260907) without retraining those 12 decoders.

## Repeat panel

F4 only, d=16, D=28, n_dense=5000, n_sparse=1500, the same 64 anchors. Two independent observation draws A/B (seeds 20260911 / 20260912) on clean uniform and on S3+N4. Q is rerun with the frozen k=1024, 3 splits, RIDGES grid. At most two decoder initialization seeds {0,1} per draw and condition (≤8 new AEs). Architecture, epochs, rank, bandwidth and regularization are not swept.

## Metrics

Rank: Spearman, Kendall tau, pairwise ordering accuracy, rank RMSE. Quartiles: truth-defined top/bottom groups with explicit chance baselines. Calibration: one global multiplicative factor fit on a frozen half of the 64 anchors (seed 20260907) and applied to the complement across conditions. Reliability: r_rel = rho(Khat_A, Khat_B); attenuation ceiling sqrt(r_rel) only when r_rel>0. Fraction of ceiling is stored unclamped; display is clamped at 1. 200-bootstrap intervals on anchors. Density quintiles use the analytic sampling weight w=exp(beta s).

## Noise-scale comparator

Synthetic RMS is compared to existing ViT-B decoder reconstruction residual / global signal scale from the physics D-residual run, and to any already stored quadratic residual / neighbourhood radius. These are empirical scale comparators, not identifications of observational noise. No new embeddings or augmentations.

Runtime wall: 2700.0 s. New AE cap: 8.
