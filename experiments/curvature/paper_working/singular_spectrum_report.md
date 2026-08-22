# Singular-value spectrum of Dense→Dense Ridge maps

**Status:** corrected — primary analysis uses the **effective end-to-end linear map** after undoing `StandardScaler`, not `Ridge.coef_` alone.  
**Artifacts:** `outputs/ridge_scaling_geometry/effective_map_spectrum/`  
**Script:** `experiments/SAE-shared-basis/run_singular_spectrum_analysis.py`  

**Map analyzed (primary):** with train-only `StandardScaler` on \(X,Y\) and Ridge weight \(W=\)`coef_` in scaled space,

\[
A=\mathrm{diag}(\sigma_y)\,W\,\mathrm{diag}(1/\sigma_x),
\qquad
\hat y = A(x-\mu_x)+\mathrm{diag}(\sigma_y)b+\mu_y.
\]

SVD and truncated-SVD transfer use \(A\) in original embedding coordinates.  
**Diagnostic only:** SVD of scaled \(W\) (previous report); still saved as `H_norm_scaled_W` / `A_log_scaled_W` in `ridge_map_metadata.csv`.  
**Primary active threshold:** \(\epsilon=10^{-8}\sigma_{\max}\).

Conceptual separation (kept throughout):

- \(R=M_{\mathrm{Dense+Ridge}}\) — recoverable cross-survey correspondence  
- \(G=H_{\mathrm{norm}}\) — spectral isotropy / flatness of the **alignment map** (not representation rank)

---

## 1. Spectrum summaries (effective \(A\))

Active set ≈ full rank on most rungs (`active_fraction` \(\ge 0.88\); typically \(\ge 0.999\)).  
\(H_{\mathrm{norm}}=1\) would be a perfectly flat energy spectrum; observed values remain **far below 1**, and a few rungs become nearly rank-1 in energy once feature scales are restored.

| Family | Model | \(P\) (approx) | \(H_{\mathrm{norm}}\) | \(r_{\mathrm{eff}}/r\) | \(A_{\log}\) | \(D_{\mathrm{sim}}\) | \(r_{90}/r\) | \(r_{95}/r\) | \(H_{\mathrm{norm}}(W)\)† |
|--------|-------|---------------:|---------------------:|-----------------------:|-------------:|---------------------:|-------------:|-------------:|--------------------------:|
| AstroPT | 15m | 15M | 0.537 | 0.064 | 1.762 | 0.899 | 0.099 | 0.172 | 0.595 |
| AstroPT | 95m | 95M | 0.556 | 0.053 | 1.837 | 0.902 | 0.107 | 0.173 | 0.616 |
| AstroPT | 850m | 850M | 0.569 | 0.037 | 1.933 | 0.914 | 0.093 | 0.150 | 0.620 |
| ConvNeXt | nano | 15M | 0.547 | 0.053 | 1.831 | 0.911 | 0.080 | 0.142 | 0.574 |
| ConvNeXt | tiny | 28M | 0.537 | 0.046 | 1.863 | 0.917 | 0.073 | 0.132 | 0.568 |
| ConvNeXt | base | 89M | 0.543 | 0.042 | 1.871 | 0.922 | 0.067 | 0.125 | 0.555 |
| ConvNeXt | large | 198M | 0.526 | 0.031 | 1.851 | 0.928 | 0.063 | 0.120 | 0.556 |
| DINOv2 | small | 22M | 0.577 | 0.080 | 1.820 | 0.891 | 0.104 | 0.169 | 0.600 |
| DINOv2 | base | 86M | 0.560 | 0.054 | 1.801 | 0.905 | 0.091 | 0.159 | 0.586 |
| DINOv2 | large | 300M | **0.015** | 0.001 | 1.759 | 0.997 | 0.001 | 0.001 | 0.588 |
| DINOv2 | giant | 1.1B | 0.529 | 0.032 | 1.908 | 0.933 | 0.056 | 0.107 | 0.553 |
| ViT | base | 86M | 0.385 | 0.017 | 1.861 | 0.954 | 0.030 | 0.066 | 0.567 |
| ViT | large | 307M | 0.572 | 0.051 | 1.766 | 0.910 | 0.081 | 0.151 | 0.670 |
| ViT | huge | 632M | 0.211 | 0.004 | 1.777 | 0.990 | 0.003 | 0.003 | 0.631 |
| I-JEPA | huge | 630M | **≈0** | 0.001 | 2.623 | 1.000 | 0.001 | 0.001 | 0.450 |
| I-JEPA | giant | 1.0B | 0.346 | 0.009 | 3.153 | 0.977 | 0.018 | 0.035 | 0.498 |

† diagnostic: scaled-space \(W\) (previous primary).

**Overall (effective \(A\)):**  
- \(A_{\log}\) range **1.76–3.15** (typical non-extreme rungs ≈ **1.76–1.93**; I-JEPA / scale-dominated rungs go higher).  
- \(D_{\mathrm{sim}}\) range **0.89–1.00**.  
- \(H_{\mathrm{norm}}\) range ≈ **0–0.58** (mean ≈ 0.44); excluding the two near-degenerate energy cases, most rungs sit ≈ **0.21–0.58**.  
- Mean \(r_{90}/r\approx 0.060\).

Undoing StandardScaler can **absorb heterogeneous feature scales into \(A\)**, concentrating energy in a few singular directions (DINOv2-large, I-JEPA-huge, ViT-huge). That still implies **strong anisotropy** of the prediction map — if anything more extreme than scaled-\(W\) mid-band \(H_{\mathrm{norm}}\approx0.45\)–\(0.67\).

Normalized spectra (figures, recomputed on \(A\)):  
`paper_working/figures/singular_spectrum_{AstroPT,ConvNeXt,DINOv2,ViT,IJEPA}.png`

Visually, curves remain **steep**. Within families, size mainly shifts / mildly reshapes the curve; there is no common flattening toward a plateau.

---

## 2. Trends with model size

Family OLS slopes vs \(\log_{10}P\) (primary flatness = \(H_{\mathrm{norm}}\) of \(A\)):

| Family | slope \(H_{\mathrm{norm}}\) | slope \(A_{\log}\) | slope \(D_{\mathrm{sim}}\) | slope \(r_{90}/r\) | n |
|--------|---------------------------:|-------------------:|---------------------------:|-------------------:|--:|
| AstroPT | **+0.018** | +0.098 | +0.009 | −0.004 | 3 |
| ConvNeXt | **−0.014** | +0.016 | +0.014 | −0.015 | 4 |
| DINOv2 | **−0.121** | +0.039 | +0.038 | −0.041 | 4 |
| ViT | **−0.139** | −0.105 | +0.028 | −0.017 | 3 |
| I-JEPA | **+1.726** | +2.639 | −0.113 | +0.084 | 2† |

† two-point only; I-JEPA slope is dominated by the near-degenerate huge rung.

**Directional consistency for \(H_{\mathrm{norm}}\):** **2/5 positive, 3/5 negative** — **mixed**. Exact sign test for “all flatten” fails.

**Strong geometric-convergence prediction** (\(H_{\mathrm{norm}}\uparrow\), \(A_{\log}\downarrow\), \(D_{\mathrm{sim}}\downarrow\) in all families) is **not** supported.

---

## 3. Relation to recoverability

Across all 16 rungs (descriptive Spearman; rungs not independent):

| Comparison | ρ |
|------------|--:|
| \(R=M_{\mathrm{Dense+Ridge}}\) vs \(H_{\mathrm{norm}}\) | **0.04** |
| Ridge lift \(L\) vs \(H_{\mathrm{norm}}\) | **0.12** |
| \(R\) vs \(A_{\log}\) | 0.43 |
| \(L\) vs \(A_{\log}\) | 0.38 |

Within-family Spearman of \(R\) vs \(H_{\mathrm{norm}}\) remains heterogeneous.

**Answer:** better recoverability does **not** require a flatter alignment spectrum. \(R(P)\) rises while \(G=H_{\mathrm{norm}}\) stays anisotropic (often mid-band, sometimes near-degenerate from feature scales) without a shared size law.

Scatter: `paper_working/figures/recoverability_vs_spectral_entropy.png`

---

## 4. Truncated-SVD transfer (no refit)

Rank-\(k\) maps \(A_k=U_k\Sigma_k V_k^\top\) of the **effective** \(A\), evaluated on the same held-out test gallery (original coordinates).

| Family | Model | \(k_{90}^{\mathrm{transfer}}\) | \(\tilde k_{90}=k_{90}/r\) | full Ridge mKNN |
|--------|-------|-------------------------------:|---------------------------:|----------------:|
| AstroPT | 15m / 95m / 850m | 16 / 32 / 128 | 0.042 / 0.042 / 0.063 | 0.032–0.050 |
| ConvNeXt | nano→large | 32 | 0.021–0.050 | 0.041–0.047 |
| DINOv2 | all | 64 | 0.042–0.167 | 0.033–0.039 |
| ViT | all | 64 | 0.050–0.083 | 0.027–0.046 |
| I-JEPA | huge / giant | 32 / 64 | 0.025 / 0.045 | 0.040–0.056 |

- Mean \(\tilde k_{90}^{\mathrm{transfer}}\approx 0.057\) (median ≈ 0.048): **~5–6% of singular directions** recover ≥90% of Ridge lift.  
- Slope of \(\tilde k_{90}\) vs \(\log_{10}P\): **−** in 3/5 families (AstroPT / I-JEPA **+**).

Files: `truncated_svd_mknn.csv`, `transfer_complexity_k90.csv`, `truncated_svd_transfer_efficiency.png`.

**Do not equate** high \(H_{\mathrm{norm}}\) with better transfer: truncated-SVD shows transfer is **functionally low-rank** even while the full spectrum remains highly anisotropic.

---

## 5. Bottom line

Closest interpretation:

### **B** (primary), with architecture / scale heterogeneity noted

> **Recoverability increases with model size, but spectral anisotropy of the required Dense→Dense Ridge map — in original coordinates after undoing StandardScaler — remains large and does not systematically flatten.**

Supporting facts:

1. Effective-\(A\) spectra are strongly anisotropic (\(A_{\log}\gtrsim 1.76\); \(H_{\mathrm{norm}}\) far below 1, sometimes near 0 when feature scales dominate).  
2. Family slopes of \(H_{\mathrm{norm}}\) vs \(\log_{10}P\) are **mixed** (2↑ / 3↓) — not geometric convergence (**A** fails).  
3. \(R\) and \(H_{\mathrm{norm}}\) are essentially uncorrelated across rungs (\(\rho\approx 0.04\)).  
4. Procrustes already showed orientation alone fails; spectra confirm **strong anisotropic stretch** of the true prediction map.  
5. Truncated SVD of \(A\): most Ridge mKNN gain lives in a **small leading fraction** of singular directions (\(\tilde k_{90}\sim 5\%\)).

Scaled-\(W\) diagnostics still occupy a mid-anisotropic band (\(H_{\mathrm{norm}}\approx0.45\)–\(0.67\)); composing out the scalers does not rescue an isotropic / Procrustes-like story — it usually preserves, and sometimes exaggerates, anisotropy.

Optional precise **E**:

> Shared structure becomes more recoverable under a **stable, strongly anisotropic, functionally low-rank** linear alignment in original coordinates; size does not buy a similarity transform or a flat singular spectrum.

---

## Artifact index

```text
outputs/ridge_scaling_geometry/effective_map_spectrum/
  singular_spectra_full.csv
  singular_spectra_normalized.csv
  singular_spectrum_summary.csv
  singular_spectrum_eps_robustness.csv
  singular_spectrum_vs_size_slopes.csv
  recoverability_vs_spectrum_spearman.csv
  adjacent_Dalign_vs_spectrum_delta.csv
  truncated_svd_mknn.csv
  transfer_complexity_k90.csv
  ridge_map_metadata.csv          # includes scaled-W diagnostic columns
  singular_spectrum_meta.json
  H_norm_size_sign_summary.json

paper_working/figures/
  singular_spectrum_AstroPT.png
  singular_spectrum_ConvNeXt.png
  singular_spectrum_DINOv2.png
  singular_spectrum_ViT.png
  singular_spectrum_IJEPA.png
  recoverability_vs_spectral_entropy.png
  truncated_svd_transfer_efficiency.png
```
