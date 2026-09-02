# Raw PCA control (pooled shared basis, no Ridge)

**Status:** complete. **No further experiments.** Manuscript not revised yet.  
**Artifacts:** `outputs/raw_pca_control/`  
**Script:** `experiments/SAE-shared-basis/run_raw_pca_control.py`

## Conceptual question

Does PCA improve native cross-survey correspondence by itself, or only when followed by a learned alignment map?

## Method (important)

| Item | Choice |
|------|--------|
| PCA basis | **Single pooled** PCA on stacked \([X_{\mathrm{Legacy,train}}; X_{\mathrm{HSC,train}}]\) |
| Pairing for PCA | **Not used** |
| Independent per-survey PCA | **Not used** (incomparable orientations for raw cosine) |
| Centering | sklearn PCA default |
| Whitening | **False** |
| StandardScaler before PCA | **No** |
| Post-projection | row-wise **l2** inside `knn_cos` (same as raw Dense) |
| Eval | test-only gallery, \(k=10\), frozen split |

PCA+Ridge numbers come from the prior sweep (`final_pca_rank_sweep`), which uses **independent** Legacy/HSC PCAs + Ridge — documented as a different construction, appropriate when Ridge realigns bases.

---

## A. Raw PCA rank sweep

| Rank | Raw PCA mean | Δ vs Raw Dense | Positive rungs |
| ---: | -----------: | -------------: | -------------: |
| 8 | 0.0152 | −0.0066 | 0 / 16 |
| 16 | 0.0195 | −0.0023 | 1 / 16 |
| 32 | 0.0218 | +0.0001 | 9 / 16 |
| 64 | 0.0234 | +0.0016 | 14 / 16 |
| 128 | 0.0239 | +0.0021 | 15 / 16 |
| 256 | 0.0241 | +0.0023 | 13 / 16 |
| 512 | 0.0245 | +0.0025 | 13 / 14 |
| full (= Dense) | 0.0218 | 0 | — |

Raw Dense mean ≈ **0.022**. Best raw PCA ≈ **0.024–0.025** — a **small** native lift (~+0.002), nowhere near Dense+Ridge (0.042) or PCA+Ridge (0.050–0.055).

---

## B. Comparison with PCA+Ridge

Same valid rung sets:

| Rank | N | Raw Dense | Raw PCA | Dense+Ridge | PCA+Ridge | Ridge gain after PCA |
| ---: | -: | --------: | ------: | ----------: | --------: | -------------------: |
| 256 | 16 | 0.0218 | 0.0241 | 0.0415 | **0.0495** | 0.0254 |
| 512 | 14 | 0.0220 | 0.0245 | 0.0427 | **0.0551** | 0.0306 |

Almost all of the PCA headline advantage is **conditional on Ridge**, not present in raw geometry.

---

## C. PCA×Ridge interaction

\[
I=(M_{\mathrm{PCA+Ridge}}-M_{\mathrm{rawPCA}})-(M_{\mathrm{Dense+Ridge}}-M_{\mathrm{rawDense}}).
\]

| Rank | Mean \(I\) | Positive rungs |
| ---: | ---------: | -------------: |
| 8 | −0.0185 | 0 / 16 |
| 16 | −0.0152 | 0 / 16 |
| 32 | −0.0101 | 0 / 16 |
| 64 | −0.0051 | 2 / 16 |
| 128 | +0.0006 | 9 / 16 |
| **256** | **+0.0057** | **14 / 16** |
| **512** | **+0.0099** | **14 / 14** |
| full | 0 | — |

At mid/high ranks, \(I>0\): PCA makes embeddings **especially amenable to linear alignment**, beyond additive PCA and Ridge effects. At low ranks, PCA hurts the Ridge increment (\(I<0\)).

---

## D. Scaling

Raw PCA vs raw Dense family Δβ:

| Rank | mean Δβ | positive families / 5 |
| ---: | ------: | --------------------: |
| 8–512 | **negative** (~−0.002 to −0.006) | 0–2 |

Dimensional reduction **does not** steepen native Platonic slopes; if anything it mildly flattens them.

Mean family β (descriptive) at rank 256: Dense 0.017 → raw PCA 0.016 → Dense+Ridge 0.025 → PCA+Ridge 0.027. The large scaling step remains **Ridge**; PCA+Ridge adds a small further mean-β bump consistent with earlier \(T_{\mathrm{PCA}}\) results.

---

## E. Final interpretation

### **B** (primary), with a positive-interaction refinement

> **PCA has little effect on native cross-survey correspondence, but substantially improves supervised linear recoverability.**

At ranks ≳64, raw PCA is only ~+0.002 above Dense, while PCA+Ridge is ~+0.008–+0.012 above Dense+Ridge. Interaction \(I>0\) at 256/512 shows the PCA benefit is specifically about **making shared structure easier to align**, not revealing it directly in cosine neighbourhoods.

Not **A** (raw PCA does not broadly close the gap to aligned methods).  
Not **C** (raw PCA is not systematically *below* Dense at useful ranks; tiny positive).  
Mild family heterogeneity (e.g. AstroPT raw-PCA lift ≈ 0/− at 256) does not overturn **B**.

---

## Boxed answer

\[
\boxed{
\text{PCA does not reveal much shared structure by itself;}
\quad
\text{it makes shared structure easier to recover with Ridge.}
}
\]

## Figures

- `paper_working/figures/raw_pca_rank_curve.png`
- `paper_working/figures/pca_ridge_interaction.png`

## Experiment freeze

Do not start CCA / PLS / kernel PCA / local maps / new sweeps.  
**Next step: manuscript revision.**
