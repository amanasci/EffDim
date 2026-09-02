# Final PCA rank sweep

**Status:** complete. **No further experimental branches.** Next step: manuscript revision.  
**Artifacts:** `outputs/final_pca_rank_sweep/`  
**Script:** `experiments/SAE-shared-basis/run_final_pca_rank_sweep.py`

## Protocol (exact PCA256 convention)

| Item | Choice |
|------|--------|
| PCA fitting set | `inner_tr` = 80% of `train_idx` (`train_test_split`, seed 0) — same as `run_alignment_controls.py` |
| Centering | sklearn `PCA` default (mean-center) |
| Whitening | **No** |
| Legacy / HSC bases | **Independent** PCA on Legacy (\(X_2\)) and HSC (\(X_1\)) |
| StandardScaler | **After** PCA, inside Ridge (`StandardScaler` on \(Z_x,Z_y\)) |
| Ridge | \(\alpha=1\), `fit_intercept=True` |
| Eval | test-only gallery, mKNN@10, Legacy→HSC |
| Rank grid | 8, 16, 32, 64, 128, 256, 512, full (skip if \(d <\) rank) |

**Reproducibility:** mean PCA256 mKNN = **0.04954** (prior controls: **0.04957**).

Two rungs skip 512 (embedding dim 384): AstroPT-15m, DINOv2-small → best feasible PCA rank = 256 there.

---

## A. Rank sweep

| Rank | Mean mKNN | Δ vs Dense+Ridge | Positive rungs / N |
| ---: | --------: | ---------------: | -----------------: |
| 8 | 0.0164 | −0.0252 | 0 / 16 |
| 16 | 0.0241 | −0.0175 | 0 / 16 |
| 32 | 0.0315 | −0.0101 | 0 / 16 |
| 64 | 0.0380 | −0.0035 | 5 / 16 |
| 128 | 0.0443 | **+0.0027** | 13 / 16 |
| 256 | 0.0495 | **+0.0080** | **16 / 16** |
| 512 | 0.0551 | **+0.0124** | **14 / 14** |
| full (Dense+Ridge) | 0.0415 | 0 | — |

Raw Dense mean ≈ 0.0218 (horizontal reference on Fig. A).

Family-clustered bootstrap 95% CI for mean \(S_{\mathrm{PCA}}\):

| Rank | CI |
| ---: | --- |
| 128 | [−0.0002, 0.0057] (touches 0) |
| 256 | **[0.0054, 0.0103]** |
| 512 | **[0.0099, 0.0154]** |

---

## B. Best-rank behaviour

- **Aggregate best PCA rank:** **512** (mean 0.055).  
- **Per-rung best:** 512 on 14/16; 256 on the two \(d=384\) rungs.  
- **Per-family best:** 512 for all five families (among available ranks).

**Curve shape:** monotone **increase** through the PCA ladder (8→512). Full Dense+Ridge sits **below** PCA128–512.

Closest shape label: **E** — not a sharp low-rank peak (rules out classic intermediate optimum **B**); not “full wins” (**C**); not a flat plateau (**A**).  
Precise: *performance rises with PCA rank up to the largest feasible bottleneck; unconstrained full-dim Ridge underperforms high-rank PCA.*

PCA256 is **not** an isolated lucky spike (**interpretation C** for the project’s final choice is rejected). Advantage is robust across **128–512**, strongest at the top of the ladder.

---

## C. Scaling vs Dense+Ridge

\[
T_{\mathrm{PCA}}(d)=\tfrac15\sum_F\bigl(\beta_{d,F}-\beta_{\mathrm{Dense+Ridge},F}\bigr).
\]

| Rank | mean Δβ \(T_{\mathrm{PCA}}\) | positive families / 5 |
| ---: | ---------------------------: | --------------------: |
| 8 | −0.0146 | 0 |
| 16 | −0.0139 | 0 |
| 32 | −0.0084 | 0 |
| 64 | −0.0071 | 0 |
| 128 | −0.0031 | 1 |
| 256 | **+0.0020** | **3** |
| 512 | **+0.0027** | **3** |
| full | 0 | 0 |

Low ranks **flatten** size slopes relative to Dense+Ridge. At 256/512 there is a **mild** positive mean Δβ (3/5 families); effect size is much smaller than the Dense+Ridge-vs-Dense amplification (\(T=0.00721\)). I-JEPA remains two-point.

**Read:** PCA’s clear effect is **raising the level**; it does **not** systematically and strongly steepen scaling beyond Dense+Ridge.

---

## D. Relation to truncated-Ridge spectrum

| Quantity | Typical scale |
|----------|---------------|
| \(\tilde k_{90}^{\mathrm{transfer}}\) (map SVD) | ~**6%** of map rank |
| Best PCA rank / embedding dim | ~**0.33–0.80** (mean ~0.5) |

Both say transferable structure is **not** full-dimensional, but they measure different things:

- **PCA:** how much **representation** dimension to keep *before* fitting.  
- **Truncated Ridge SVD:** how many **fitted map** directions are needed *after* fitting.

They are consistent with a broadly low-/mid-dimensional alignment phenomenon, not interchangeable. PCA wants hundreds of embedding PCs; the map still concentrates functional transfer into a small leading singular subspace (with strong shearing).

**MSE diagnostic (limited):** train/test MSE fall as PCA rank rises, but PCA MSE is in \(Z\)-space and full Ridge MSE is in original \(Y\) — not directly comparable. Full Ridge has the lowest raw MSE yet worse mKNN than PCA256/512, so the PCA mKNN gain is **not** “just lower regression error.” Soft support for a neighbourhood-geometry / regularization story rather than pure fit quality.

---

## E. Final interpretation

### **A** (primary)

> **PCA provides a robust mid/high-rank alignment advantage over Dense+Ridge, but does not further steepen scaling in a clear, systematic way.**

Supporting:

1. PCA256 fully reproduced; PCA128–512 beat Dense+Ridge on nearly all rungs.  
2. Curve rises with rank through 512; full is worse → bottleneck helps, but the useful bottleneck is **large** (hundreds), not tiny.  
3. \(T_{\mathrm{PCA}}\) only mildly positive at 256/512 (3/5 families).  
4. Not interpretation **C** (isolated 256). Not **B** (no strong extra scaling). Not mainly **D** (all families prefer top available PCA rank).

Optional precise **E′:** *Generic PCA before Ridge is a strong regularizer for cross-survey mKNN at ranks ≳128; the previous PCA256 headline sits on a rising limb, with 512 better when dimension allows.*

---

## Figures

| | Path |
|--|------|
| A | `paper_working/figures/final_pca_mean_mknn_vs_rank.png` |
| B | `paper_working/figures/final_pca_family_curves.png` |
| C | `paper_working/figures/final_pca_scaling_T_vs_rank.png` |

---

## Surviving paper claims (experiment freeze)

1. Supervised linear recoverability increases with model size (Dense+Ridge vs Dense).  
2. True pairing is required for that scaling amplification.  
3. Alignment is strongly anisotropic, not Procrustean.  
4. Useful **map** transfer concentrates in relatively few singular directions (~6% for 90% lift).  
5. **PCA before Ridge** adds a robust **level** advantage at mid/high ranks (≳128); it is not a sharp ultra-low-rank effect and does not clearly further steepen scaling.  
6. SAE/BSF do not beat the properly controlled Dense+Ridge baseline.

**Stop experiments.** Manuscript revision next.
