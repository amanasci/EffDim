# Ridge scaling amplification + map geometry

**Status:** analysis complete. Manuscript not edited.  
**Namespace:** `outputs/ridge_scaling_geometry/`  
**Script:** `experiments/SAE-shared-basis/run_ridge_scaling_geometry.py`  
**Protocol (frozen):** Legacy→HSC, `n=16384`, `train_test_split(test_size=0.2, random_state=0)` → 13107/3277, test-only gallery, `k=10`, `StandardScaler` + `Ridge(α=1, fit_intercept=True)`. Seeds recorded in `run_meta.json`.

Existing `outputs/paper_alignment_controls/` files were **not** overwritten.

---

## A. Statistical support for scaling amplification

Primary pre-registered statistic:

\[
T=\frac15\sum_F\bigl(\beta_{\mathrm{Dense+Ridge},F}-\beta_{\mathrm{Dense},F}\bigr).
\]

### Family Δβ (observed)

| Family | β Dense | β Dense+Ridge | Δβ | n rungs |
|--------|---------|---------------|-----|---------|
| AstroPT | 0.00496 | 0.01010 | **+0.00513** | 3 |
| ConvNeXt | 0.00276 | 0.00568 | **+0.00292** | 4 |
| DINOv2 | 0.00004 | 0.00317 | **+0.00313** | 4 |
| ViT | 0.01297 | 0.02326 | **+0.01029** | 3 |
| I-JEPA | 0.06615 | 0.08075 | **+0.01460** | 2 (two-point) |

- \(T_{\mathrm{real}}=0.00721\)
- Median family Δβ = 0.00513
- Adjacent \(D^{\mathrm{align}}\): mean **+0.00262**, **9/11** positive

### Exact family sign test

All five families have \(\Delta\beta_F>0\).

Under a fair coin null on sign:

\[
p=(1/2)^5=0.03125.
\]

Caveats (explicit): **n = 5 architecture families**; **I-JEPA slope uses only 2 rungs**. This is an assumption-light directional test at the architecture level, not a substitute for the permutation or object bootstrap.

### Synchronized shuffle-refit permutation (primary causal null for scaling)

One permutation seed \(b\) shuffles the **train** Legacy↔HSC object correspondence and is applied to **all 16 rungs** (same train-row permutation; all official parquets share row index = object). Then Ridge is refit; Dense slopes stay fixed; \(T_b=\mathrm{mean}_F(\beta_{\mathrm{shuffled\ Ridge},F}^{(b)}-\beta_{\mathrm{Dense},F})\).

| Quantity | Value |
|----------|-------|
| \(B\) | 200 |
| \(T_{\mathrm{real}}\) | 0.00721 |
| null mean | **−0.01727** |
| null sd | 0.00060 |
| null 2.5% / 50% / 97.5% | −0.01836 / −0.01726 / −0.01601 |
| \(p_{\mathrm{perm}}=(1+\#\{T_b\ge T_{\mathrm{real}}\})/(B+1)\) | **0.004975** (0 of 200 nulls ≥ real) |

Secondary under the same null: mean adjacent \(D\) real = +0.00262 vs null mean ≈ −0.00334; # positive \(D\) real = 9 vs null mean ≈ 1.65.

**Read:** shuffled correspondences do **not** amplify size slopes; they produce systematically **negative** \(T\). True pairing is required for the Ridge-induced scaling amplification.

Files: `scaling_permutation_null.csv`, `scaling_permutation_summary.csv`.

### Object bootstrap (conditional on these families / maps)

Resample the **same** held-out test object IDs with replacement across all rungs/methods; recompute mean mKNN, family slopes, and \(T\). \(B_{\mathrm{boot}}=5000\). Per-query scores cached in `per_query_mknn.npz`.

| | |
|--|--|
| \(T_{\mathrm{real}}\) | 0.00721 |
| bootstrap mean | 0.00720 |
| 95% percentile CI | **[0.00336, 0.01105]** |

CI excludes 0 → amplification is robust to which held-out objects are scored (given fixed maps / families).

### How to read the three tests (not interchangeable)

| Test | Question |
|------|----------|
| Family sign | Is slope amplification **directionally consistent** across architecture families? |
| Object bootstrap | Is the measured \(T\) **stable to which test objects** are evaluated? |
| Shuffle-refit perm | Does stronger scaling **require true Legacy↔HSC correspondence** when fitting Ridge? |

---

## B. Alignment geometry

SVD of the **effective** end-to-end Ridge map after undoing train-only `StandardScaler`:
\(A=\mathrm{diag}(\sigma_y)W\mathrm{diag}(1/\sigma_x)\) with \(W=\)`coef_` (not SVD of scaled \(W\) alone).
Active singular values: \(\sigma_i\ge 10^{-8}\sigma_{\max}\). Shape normalized by geometric mean:

\[
\tilde\sigma_i=\sigma_i\big/\exp\bigl(\tfrac1r\sum_j\log\sigma_j\bigr).
\]

Primary anisotropy: \(A_{\log}=\mathrm{std}_i(\log\tilde\sigma_i)\). Also \(D_{\mathrm{sim}}=|\Sigma-cI|_F/|\Sigma|_F\) with \(c=\mathrm{mean}(\sigma)\), and robust \(\kappa_{95/5}\).
Full spectrum tables/figures: `paper_working/singular_spectrum_report.md` and `outputs/ridge_scaling_geometry/effective_map_spectrum/`.

| Family | model | \(\log_{10}P\) | \(A_{\log}\) | \(D_{\mathrm{sim}}\) | \(\kappa_{95/5}\) | \(H_{\mathrm{norm}}\) | Ridge mKNN |
|--------|-------|----------------|--------------|----------------------|-------------------|---------------------:|------------|
| AstroPT | 15m | 7.18 | 1.76 | 0.899 | 284 | 0.537 | 0.0320 |
| AstroPT | 95m | 7.98 | 1.84 | 0.902 | 338 | 0.556 | 0.0482 |
| AstroPT | 850m | 8.93 | 1.93 | 0.914 | 529 | 0.569 | 0.0502 |
| ConvNeXt | nano | 7.18 | 1.83 | 0.911 | 401 | 0.547 | 0.0411 |
| ConvNeXt | tiny | 7.45 | 1.86 | 0.917 | 405 | 0.537 | 0.0421 |
| ConvNeXt | base | 7.95 | 1.87 | 0.922 | 419 | 0.543 | 0.0454 |
| ConvNeXt | large | 8.30 | 1.85 | 0.928 | 375 | 0.526 | 0.0472 |
| DINOv2 | small | 7.34 | 1.82 | 0.891 | 377 | 0.577 | 0.0345 |
| DINOv2 | base | 7.93 | 1.80 | 0.905 | 328 | 0.560 | 0.0329 |
| DINOv2 | large | 8.48 | 1.76 | 0.997 | 274 | 0.015 | 0.0376 |
| DINOv2 | giant | 9.04 | 1.91 | 0.933 | 473 | 0.529 | 0.0390 |
| ViT | base | 7.93 | 1.86 | 0.954 | 392 | 0.385 | 0.0269 |
| ViT | large | 8.49 | 1.77 | 0.910 | 264 | 0.572 | 0.0453 |
| ViT | huge | 8.80 | 1.78 | 0.990 | 240 | 0.211 | 0.0459 |
| I-JEPA | huge | 8.80 | 2.62 | 1.000 | 3928 | ≈0 | 0.0399 |
| I-JEPA | giant | 9.00 | 3.15 | 0.977 | 24062 | 0.346 | 0.0561 |

Mean \(A_{\log}\approx 1.93\), mean \(D_{\mathrm{sim}}\approx 0.93\) — maps are **far** from similarity transforms. A few rungs become energy-near-degenerate once feature scales enter \(A\) (still anisotropic).

### Orthogonal Procrustes

Same split/gallery. Mean Procrustes lift vs Dense ≈ **+0.00007**; mean Ridge lift ≈ **+0.0197**. Procrustes recovers **~0.3%** of the Ridge lift. On every rung, Procrustes ≈ Dense and Ridge ≫ Procrustes.

### Anisotropy vs size

Family OLS of \(A_{\log}\) on \(\log_{10}P\): AstroPT / ConvNeXt / DINOv2 / I-JEPA **positive**; ViT **negative** (I-JEPA two-point / scale-dominated). No consistent simplification toward isometry with scale. \(H_{\mathrm{norm}}\) size signs: **2↑ / 3↓**.

### Lift vs spectrum shape (descriptive)

Across 16 rungs, Spearman(lift, \(A_{\log}\)) ≈ **0.38**; Spearman(\(R\), \(D_{\mathrm{sim}}\)) ≈ **0.15**. Within-family trends mixed; rungs are not independent. Slope amplification occurs while spectrum shape stays highly anisotropic (not “stable near isometry”).

---

## C. Global-linearity diagnostic

Residual \(r_i=y_i-(Ax_i+b)\) on test objects; source-space (Legacy Dense) kNN with \(k\in\{10,25,50\}\).

\[
R_{\mathrm{local}}=\frac1n\sum_i\frac1k\sum_{j\in N_k(i)}\cos(r_i,r_j),
\]

null: permute residual vectors across test objects (\(B=200\) per rung/k).

| k | mean \(R_{\mathrm{local}}\) | mean null | all 16 rungs |
|---|----------------------------|-----------|--------------|
| 10 | 0.0138 | ~0.0015 | \(p=1/201\) |
| 25 | 0.0105 | ~0.0014 | \(p=1/201\) |
| 50 | 0.0086 | ~0.0014 | \(p=1/201\) |

Local residual cosine similarity is **~7×** the permutation null — statistically clear — but **absolute** values are small (~0.01).

### Trend vs size

Most families show **flat or weakly decreasing** \(R_{\mathrm{local}}\) with \(\log_{10}P\) at \(k=10\). I-JEPA (2 points) increases at small \(k\). No strong evidence that global linearity improves or worsens systematically with scale.

**Stop rule:** residual locality is above null and real, so piecewise/local Ridge is **motivated as a follow-up**, but absolute structure is modest. Per prompt: **do not fit local Ridge maps now**.

---

## Figures

| Figure | Path |
|--------|------|
| 1 Scaling amplification | `paper_working/figures/ridge_scaling_amplification.png` |
| 2 Permutation null for \(T\) | `paper_working/figures/ridge_scaling_perm_null.png` |
| 3 \(A_{\log}\) vs size | `paper_working/figures/ridge_scaling_anisotropy.png` |
| 4 Residual locality vs size | `paper_working/figures/ridge_scaling_residual_locality.png` |

---

## Answers to Q1–Q6

### Q1 — Is Dense+Ridge slope amplification distinguishable from incorrect-correspondence Ridge?

**Yes.** Synchronized shuffle-refit null: \(T_{\mathrm{real}}=0.00721\) vs null mean \(−0.01727\); \(p_{\mathrm{perm}}=0.005\) (0/200 nulls ≥ real). True pairing is required for the scaling amplification, not only for level.

### Q2 — Does the required linear map become closer to a similarity/isometry as size grows?

**No clear yes.** \(A_{\log}\) and \(D_{\mathrm{sim}}\) stay large. Family trends are mixed (3/5 slopes of \(A_{\log}\) vs \(\log_{10}P\) positive). There is **no** general geometric simplification with scale.

### Q3 — How much of the Ridge gain is achievable with orthogonal Procrustes alone?

**Essentially none.** Procrustes ≈ Dense on all 16 rungs; mean lift ~0 vs Ridge lift ~0.020 (~0.3% of Ridge lift). The recoverable alignment is **not** a change-of-basis / rotation.

### Q4 — Does a single global Ridge leave spatially structured residuals?

**Yes, statistically.** All 16×3 (\(k\)) residual-locality tests beat the residual-permutation null at \(p=1/201\). Absolute \(R_{\mathrm{local}}\) is small (~0.01).

### Q5 — Does residual locality increase or decrease with model size?

**Mostly flat / weakly decreasing**; not a strong monotonic law. AstroPT decreases most clearly; I-JEPA (2-point) increases at \(k=10,25\).

### Q6 — Best-supported interpretation

**B (primary), with an important negative on geometric simplification:**

> Larger models contain more **linearly recoverable** shared structure under the true correspondence, but **substantial anisotropic deformation remains necessary**. Orthogonal Procrustes fails. Maps do **not** become similarity-like with scale. Weak residual locality exists but is secondary to the global anisotropic linear effect.

Not **A** (Procrustes fails). Not primarily **C** (locality is real but small; scaling story is carried by global Ridge). Not bare **D** alone — pairing-specific slope amplification is a real phenomenon beyond “Ridge just helps measurement,” even though spectrum shape does not simplify.

Optional precise **E** if one wants a one-liner:

> **E′:** Platonic-style size scaling of cross-survey neighbourhood agreement is **amplified by a globally anisotropic linear map that requires true object pairing**; the map stays far from isometry at all scales, and Procrustes contributes nothing.

---

## Artifacts checklist

```text
outputs/ridge_scaling_geometry/
  run_meta.json
  real_scaling_summary.json
  family_delta_beta.csv
  rung_scores.csv
  scaling_permutation_null.csv
  scaling_permutation_summary.csv
  object_bootstrap_scaling.csv
  object_bootstrap_summary.json
  ridge_singular_values.csv          # legacy; prefer effective_map_spectrum/
  ridge_geometry_metrics.csv         # geometry on effective A (post-StandardScaler undo)
  geometry_vs_size_slopes.csv
  lift_vs_anisotropy_spearman.csv
  procrustes_scores.csv
  residual_locality.csv
  residual_locality_vs_size_slopes.csv
  per_query_mknn.npz
  fig_*.png
  effective_map_spectrum/            # primary singular-spectrum analysis on A
```
