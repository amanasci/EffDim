# Alignment-control results (test-only gallery)

Protocol: n=16384, train=13107, test=3277, seed=0, k=10.
Direction: Legacy→HSC. Ridge α=1, intercept, StandardScaler on X and Y.
Gallery = test IDs only. Chance ≈ 10/3276 = 0.00305.
Existing SAE/BSF size-scaling outputs were not overwritten.

Source: host `outputs/paper_alignment_controls/test_only_gallery_scores.csv`.

| Family | Size | Dense | Dense+Ridge | SAE | BSF | SAE−DenseRidge | BSF−DenseRidge |
| ------ | ---: | ----: | ----------: | --: | --: | -------------: | -------------: |
| astropt | 15m | 0.0175 | 0.0320 | 0.0241 | 0.0369 | −0.0080 | +0.0049 |
| astropt | 95m | 0.0219 | 0.0482 | 0.0297 | 0.0288 | −0.0185 | −0.0194 |
| astropt | 850m | 0.0262 | 0.0502 | 0.0353 | 0.0347 | −0.0149 | −0.0155 |
| convnext | nano | 0.0180 | 0.0411 | 0.0306 | 0.0291 | −0.0105 | −0.0120 |
| convnext | tiny | 0.0185 | 0.0421 | 0.0305 | 0.0302 | −0.0116 | −0.0119 |
| convnext | base | 0.0198 | 0.0454 | 0.0342 | 0.0347 | −0.0112 | −0.0107 |
| convnext | large | 0.0211 | 0.0472 | 0.0329 | 0.0356 | −0.0143 | −0.0116 |
| dinov2 | small | 0.0229 | 0.0345 | 0.0263 | 0.0264 | −0.0083 | −0.0081 |
| dinov2 | base | 0.0195 | 0.0329 | 0.0296 | 0.0302 | −0.0033 | −0.0026 |
| dinov2 | large | 0.0222 | 0.0376 | 0.0322 | 0.0342 | −0.0054 | −0.0034 |
| dinov2 | giant | 0.0221 | 0.0390 | 0.0327 | 0.0390 | −0.0063 | +0.0000 |
| vit | base | 0.0153 | 0.0269 | 0.0225 | 0.0242 | −0.0044 | −0.0027 |
| vit | large | 0.0244 | 0.0453 | 0.0336 | 0.0335 | −0.0117 | −0.0118 |
| vit | huge | 0.0261 | 0.0459 | 0.0359 | 0.0366 | −0.0100 | −0.0092 |
| ijepa | huge | 0.0199 | 0.0400 | 0.0460 | 0.0543 | +0.0060 | +0.0144 |
| ijepa | giant | 0.0332 | 0.0561 | 0.0531 | 0.0572 | −0.0031 | +0.0010 |

## Outcome 1 — Dense+Ridge explains the apparent structured-representation lift

Distinguish:

- \(M_{\mathrm{dense}}\): unmapped cosine (no supervised alignment)
- \(M_{\mathrm{dense+Ridge}}\): same Ridge protocol as SAE/BSF
- \(M_{\mathrm{SAE+Ridge}}\), \(M_{\mathrm{BSF+Ridge}}\): representation + alignment

The old quantity \(M_{\mathrm{SAE/BSF}}-M_{\mathrm{dense}}\) confounds representation change with supervised alignment.

Meaningful residual: \(S_R=M_{R+\mathrm{Ridge}}-M_{\mathrm{dense+Ridge}}\).

Dense+Ridge wins on **16/16** rungs vs unmapped dense, **15/16** vs SAE, **12/16** vs BSF. Mean \(S_{\mathrm{SAE}}=-0.0085\) (1/16 positive); mean \(S_{\mathrm{BSF}}=-0.0062\) (4/16 positive). Family-bootstrap 95% CIs both exclude 0. I-JEPA is the only family with a sparse residual win (BSF both rungs; SAE huge only). Per-rung table above; CSV: `outputs/paper_alignment_controls/residual_sparse_lift.csv`.

Correspondence shuffle (B=20): all real fits beat all nulls (p=1/21). SAE shuffled-null mean is elevated (~0.011 vs ~0.005). True pairing matters; sparsity still does not beat Dense+Ridge.

Scaling after this control: `paper_working/post_control_scaling_report.md`.

Mean raw SAE lift over Dense: **+0.0113**
Mean raw BSF lift over Dense: **+0.0136**
Mean Dense+Ridge lift over Dense: **+0.0197** (16/16 positive)
Mean SAE residual over Dense+Ridge: **−0.0085** (family bootstrap 95% CI [−0.0120, −0.0043]); **1/16** rungs positive
Mean BSF residual over Dense+Ridge: **−0.0062** (family bootstrap 95% CI [−0.0104, −0.0007]); **4/16** rungs positive

PCA+Ridge (train-only PCA, rank chosen on an inner train split): selected rank is 256 on every rung. Mean mKNN@10 = **0.0496**, above Dense+Ridge (0.0415) and above SAE/BSF.

Figures: `paper_working/alignment_control_figures/absolute_alignment.png`, `residual_representation_lift.png`.
