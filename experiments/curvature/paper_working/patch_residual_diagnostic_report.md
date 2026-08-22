# Patchwise residual diagnostic (Dense+Ridge)

**Status:** complete. Manuscript not revised. No further experiments.  
**Artifacts:** `outputs/patch_residual_diagnostic/`  
**Script:** `experiments/SAE-shared-basis/run_patch_residual_diagnostic.py`  
**Figure:** `paper_working/figures/patch_residual_diagnostic.png`

## Protocol

Frozen Dense→Dense Ridge (Legacy→HSC, seed 0, StandardScaler + Ridge α=1).  
Patches: MiniBatchKMeans on **Legacy train** only; test assigned to nearest centroid.  
\(K\in\{4,8,16,32\}\).  
Patch intercept \(c_p\) = mean **train** residual in patch; never estimated from test.  
Local linear: source PCA→32 then Ridge(α=1) on train residuals within patch.  
Permutation null: shuffle train patch labels (\(B=200\)); primary \(G=\mathrm{MSE}_{\mathrm{global}}-\mathrm{MSE}_{\mathrm{patch}}\).

---

## Aggregate by \(K\) (16 rungs)

| \(K\) | mean rel. MSE red. (patch) | median | mean cos\((r,c_p)\) | frac \(p_G\le0.05\) | mean Δ local vs patch |
| ----: | -------------------------: | -----: | ------------------: | ------------------: | --------------------: |
| 4 | +0.00016 | −0.00002 | 0.003 | 0.31 | −0.00037 |
| 8 | +0.00028 | −0.00005 | 0.009 | 0.69 | −0.0013 |
| 16 | +0.00081 | −0.00030 | 0.015 | 0.69 | −0.0037 |
| 32 | +0.00077 | −0.00093 | 0.017 | 0.75 | −0.010 |

Relative MSE reduction \(\approx(E_{\mathrm{global}}-E_{\mathrm{patch}})/E_{\mathrm{global}}\).  
Local-linear correction **beats** patch intercept on only **13/64** rung×\(K\) cells; mean Δ is negative (local overfits).

Patch intercept beats global (\(G>0\)) on **23/64** cells; medians of relative reduction are ≤0.

---

## Answers

### Q1 — Does patch identity predict held-out residuals?

**Weakly / sometimes.** Mean cosine of test residual with patch mean vector is only **0.003–0.017**. At higher \(K\), a majority of rungs beat a label-shuffle null for \(G\), but the absolute structure is tiny.

### Q2 — How large is the effect?

**Practically negligible.** Mean relative held-out MSE reduction from patch intercept is **≪0.1%** (order \(10^{-4}\)). Residual-direction predictability (cosine) is similarly tiny.

### Q3 — Intercept vs local linear?

**Patch intercept already captures whatever little is there.** Local linear (PCA-32 + Ridge) **usually worsens** held-out MSE relative to the intercept (mean Δ negative; only 13/64 improvements). No evidence for substantial local slope variation.

### Q4 — Above permutation null?

**Often statistically, not practically.** Fraction of rungs with \(p_G\le0.05\): 31% (\(K=4\)) → ~69–75% (\(K=8\)–\(32\)). Detectable spatial organization of residuals does not translate into useful MSE reduction—consistent with earlier residual-locality findings (significant, absolute cosine ~0.01).

### Q5 — Change with model size?

**Mostly flat / mixed, leaning weakly decreasing** for relative reduction vs \(\log_{10}P\) (family slopes: more negative than positive; AstroPT decreases most). No clear law that larger models leave more (or less) patch-predictable residual structure in a practically meaningful way.

### Q6 — Conclusion

### **A**

> Residual structure is statistically detectable but practically tiny; global Ridge is adequate.

Optional nuance: **B** is not supported as a “meaningful region-dependent bias” claim—the intercept effect is too small. **C** is rejected (local linear adds nothing useful). Finish the current paper; record piecewise local-affine alignment only as optional future work if desired.

---

## Implication for the manuscript

No change to the main narrative is required. Prior residual-locality appendix wording remains accurate: weak location dependence after global linear alignment, not a reason to center the paper on local maps.
