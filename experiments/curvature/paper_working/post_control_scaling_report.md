# Post-control scaling report

Strict protocol throughout: test-only gallery, k=10, n_test=3277, Legacy→HSC, Ridge α=1 + intercept + StandardScaler. I-JEPA slopes are two-point. B=20 correspondence shuffles (min p=1/21). Family-clustered bootstrap B=10,000. Manuscript not edited.

## 1. Alignment-control result

| Method | Mean mKNN@10 |
| --- | ---: |
| Dense (unmapped) | 0.0218 |
| SAE+Ridge | 0.0331 |
| BSF+Ridge | 0.0354 |
| Dense+Ridge | **0.0415** |

Residual over Dense+Ridge \(S_R = M_R - M_{\mathrm{Dense+Ridge}}\):

| | Mean | Median | >0 | <0 | Family-bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| \(S_{\mathrm{SAE}}\) | −0.0085 | −0.0091 | **1/16** | 15/16 | **[−0.0120, −0.0043]** |
| \(S_{\mathrm{BSF}}\) | −0.0062 | −0.0087 | **4/16** | 12/16 | **[−0.0104, −0.0006]** |

Dense+Ridge beats SAE on **15/16** rungs (exception: I-JEPA huge). Beats BSF on **12/16**.

Family means of \(S\):

| Family | n | mean \(S_{\mathrm{SAE}}\) | SAE pos | mean \(S_{\mathrm{BSF}}\) | BSF pos |
| --- | ---: | ---: | ---: | ---: | ---: |
| astropt | 3 | −0.0138 | 0/3 | −0.0100 | 1/3 |
| convnext | 4 | −0.0119 | 0/4 | −0.0116 | 0/4 |
| dinov2 | 4 | −0.0058 | 0/4 | −0.0035 | 1/4 (~0 on giant) |
| vit | 3 | −0.0087 | 0/3 | −0.0079 | 0/3 |
| ijepa | 2 | +0.0015 | 1/2 | +0.0077 | 2/2 |

The overall means do **not** hide a sparse win in most families. I-JEPA is the only family where BSF (and once SAE) beat Dense+Ridge.

**Outcome 1.** The old \(M_{\mathrm{SAE/BSF}}-M_{\mathrm{dense}}\) mixed representation change with supervised alignment. After matching supervision, sparse residuals are negative; CIs exclude 0.

## 2. Refit-shuffle result

B=20; every real fit > all 20 nulls; p=1/21 on all 16×3 cells. True pairing is required.

| Method | Mean real | Mean shuffle-refit null | Real − null |
| --- | ---: | ---: | ---: |
| Dense+Ridge | 0.0415 | 0.0047 | 0.037 |
| BSF+Ridge | 0.0354 | 0.0047 | 0.031 |
| SAE+Ridge | 0.0331 | **0.0112** | 0.022 |

SAE’s shuffled null sits ~2× above Dense/BSF (~0.011 vs ~0.005), in **every** family (highest in I-JEPA, ~0.015). Null vs \(\log_{10}P\) correlation for SAE ≈ +0.49. Wrong pairing still leaves more generic neighbourhood overlap in IDF-weighted SAE codes than in dense or BSF — likely sparsity/IDF/hubs, not correspondence. That does **not** give SAE a real-fit advantage: its real−null gap is the smallest of the three.

Did not raise B to 100: B=20 already saturates the qualitative test (all 20 nulls lost). Finer p-values would not change Outcome 1.

## 3. Size scaling

Family slopes \(\beta\) of mKNN vs \(\log_{10}P\) (aligned methods):

| Family | n | β Dense+Ridge | β SAE | β BSF | Δβ SAE | Δβ BSF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| astropt | 3 | +0.0101 | +0.0064 | −0.0010 | **−0.0037** | **−0.0111** |
| convnext | 4 | +0.0057 | +0.0029 | +0.0063 | **−0.0028** | +0.0007 |
| dinov2 | 4 | +0.0032 | +0.0039 | +0.0074 | +0.0007 | +0.0042 |
| vit | 3 | +0.0232 | +0.0160 | +0.0146 | **−0.0072** | **−0.0086** |
| ijepa* | 2 | +0.0806 | +0.0351 | +0.0140 | **−0.0455** | **−0.0666** |

\*two-point slope.

Δβ SAE: 4/5 families negative (only DINOv2 slightly positive).  
Δβ BSF: 3/5 negative; ConvNeXt ≈0; DINOv2 positive.

Family-bootstrap 95% CI on mean Δβ: SAE **[−0.029, −0.0016]** (excludes 0, inflated by I-JEPA); BSF **[−0.042, +0.0004]** (includes 0).

Adjacent \(D^{\mathrm{aligned}}_R=\Delta M_R-\Delta M_{\mathrm{Dense+Ridge}}\) (11 steps):

| | Mean | Median | Range | + / − |
| --- | ---: | ---: | --- | --- |
| SAE | −0.0021 | −0.0011 | [−0.0106, +0.0050] | 4 / 7 |
| BSF | −0.0029 | ~0 | [−0.0243, +0.0055] | 6 / 5 |

Family-bootstrap CI for mean \(D^{\mathrm{aligned}}\): SAE [−0.0049, −0.0003]; BSF [−0.0093, +0.0009].

Dense+Ridge itself still rises with size in every family. Sparse maps do **not** consistently steepen that rise. Language: **no consistent positive representation×scale interaction.** Not “the interaction is weak” as a precision claim.

## 4. Bottom line

**E, with a directional lean toward C on slopes.**

- Sparse SAE/BSF **reduce recoverable aligned correspondence** relative to Dense+Ridge (a). Family CIs for \(S\) sit below zero. I-JEPA is the exception, not the rule.
- **Scaling with size remains positive** for Dense+Ridge. SAE/BSF also usually rise, but they do **not** strengthen Platonic scaling once supervision is matched. Slope differences are mixed-to-negative (b).
- Not A (sparse does not strengthen scaling). Not a clean B (Δβ is not just noise around zero for SAE). Not a uniform C if I-JEPA is set aside. Hence E: **matched alignment, not sparsity, is what recovers shared structure; sparsity does not buy a more Platonic size law.**

Suggested paper pivot (not applied): cross-survey **alignment** vs raw dense, then a negative/null result that SAE/BSF do not improve level or scaling under that control. Title should not advertise sparse representations as the source of extra shared structure.

Did not rerun unpaired DualEncoder. Did not expand PCA grid. Did not edit `paper/main.tex`.
