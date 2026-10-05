<!--
Draft rebuttal to the ML4PS review (Borderline Accept, confidence 3/5).
Evidence behind every number, all under curvature-experiment/results/:
  tensor-fidelity/REPORT.md      concern 1
  review-robustness/REPORT.md    concerns 2-5
  scaling/SCALING_REPORT.md      10 galaxy encoders, DINOv3 size ladder
  qm9/QM9_REPORT.md              QM9 molecules, 8 encoders
The "Changes to the manuscript" list is a proposal. paper/latex/main.tex is untouched.
-->

# Response to the reviewer

We thank the reviewer for a careful report. The two points you said would move your score most were full-tensor validation and robustness to probe tuning, so we ran both, and we also ran the three remaining concerns as separate experiments. We then repeated the whole battery on ten galaxy encoders and on a second domain (QM9 molecules) to test the "practical usefulness" question. Some results came back weaker than our claims, and we say where.

## 1. Validation of the full tensor

We built a synthetic problem with a known decoder and a known label Hessian and checked the full probe contraction <w_N, II> against the truth, not just the trace. On the unit sphere this contraction splits into a radial term, which any estimate with the right tangent plane recovers, and an in-sphere part. The radial term dominates for most labels, so we also report the in-sphere part on its own, since that part actually tests the estimator.

On the small fixture (d = 4, D = 64, n = 64,000, noise-free) the cosine with the truth is at least 0.999 for the full contraction and at least 0.997 for the in-sphere part, and the mismatch ranks correlate with the true mismatch at Spearman 0.93 to 0.995. All six synthetic labels pass the line we fixed before running (on the full contraction), and they also pass the same line on the in-sphere part, which we added after seeing that the radial term dominates.

At the paper's scale (d = 16, D = 768, n = 86,471) the in-sphere cosine is 0.963 to 0.965 without noise and 0.68 to 0.84 at noise 0.25, and the mismatch rank correlation stays between 0.88 and 0.99. The estimate gets the direction right but overstates the magnitude of the label Hessian by about a factor of two at this scale. Our partials are rank-based, so this does not change them, but it rules out any claim about mismatch magnitudes, and we will say so. We read the low split-half cosines on real labels (0.19 to 0.35) as label noise at fixed n, since the same estimator recovers a known Hessian at the same n and D.

## 2. A validation-tuned probe

We chose ridge strength by leave-one-out cross-validation (alpha* between 0.001 and 0.1) and reran the mismatch, alignment and counterfactual analyses with that probe. Out-of-fold R² rises from 0.48 to 0.59 at alpha = 100 to 0.64 to 0.73 at alpha*, which matches the gain you pointed to.

With the tuned probe the results get stronger, not weaker. The mismatch partial is negative with p < 0.05 in 20 of 20 encoder-label pairs (14 of 20 at alpha = 100), ranging from -0.10 to -0.72. Alignment is positive and significant in 18 of 20. In the counterfactual, bending along the model's normal gives help of 0.93 to 0.99 and hurt of 1.00, against a random-direction help of 0.05 to 0.11, and the thinned sign test gives p < 0.001 in all 20. One detail stood out. The median best bend scale t* falls from 1.5 to 3.4 at alpha = 100 to 0.96 to 1.36 at alpha*, so the tuned probe wants roughly the decoder's own curvature, while the heavily regularised probe wanted more.

## 3. Value beyond local target difficulty

We added the label-Hessian norm and a local roughness score (one minus the label's linear R² in the decoder's tangent chart) as extra controls. The roughness score shares the Jacobian and metric with our Hessian estimate, so it over-controls if anything. With the tuned probe the mismatch partial under these controls is still negative with p < 0.001 in 20 of 20 pairs (-0.32 to -0.74).

For the held-out comparison we fit on half of the overlap blocks and scored on the other half, 20 splits. Adding mismatch and alignment to the extended controls raises out-of-sample R² of local probe error in all 20 pairs with the tuned probe (median gain +0.015 to +0.26), and the 5th percentile across splits stays above zero in 18 of 20. At alpha = 100 the gain is smaller, the median is positive in 19 of 20, and the 5th percentile clears zero in only 7. For several labels the added value is a few hundredths of R², and we will report it at that size.

## 4. Dependence across anchors

We replaced unrestricted permutation inference with a cluster bootstrap that resamples whole blocks of overlapping neighbourhoods (32 blocks, with 16 and 64 as sensitivity), and we added a thinned set of anchors whose neighbourhoods overlap by at most 10%. Neighbouring blocks still share boundary points, so we treat these intervals as more honest than before, not as exact.

Under the bootstrap, the mismatch interval excludes zero for magnitude and redshift in 10 of 10 encoder-label pairs with the tuned probe (9 of 10 at alpha = 100). For morphology and stellar mass it excludes zero in only 4 of 10. We will narrow the main claim to magnitude and redshift and describe morphology and stellar mass as inconclusive, which also fits the weaker effects the paper already reported for them.

## 5. Surrogate definition and fidelity

You are right that §4 and Appendix D disagree in wording. Every counterfactual number in the paper uses the Appendix D construction, where we keep the probe's tangent and radial readout exact on the neighbours and only scale the decoder's second-order in-sphere term. We will rewrite §4 to say this.

To measure fidelity we compared, at t = 1, the local R² of the surrogate with the local R² of the actual probe on each of 512 anchors. At alpha = 100 the median absolute gap is 0.035 to 0.093 and the per-anchor rank correlation is 0.35 to 0.82. At alpha* the gap grows to 0.10 to 0.28 and the rank correlation drops to 0.16 to 0.66. The surrogate tracks the probe reasonably at the published ridge strength but loosely for the tuned probe, and we will report both rather than only the better one.

## Breadth (significance)

To test whether the pattern holds outside five galaxy encoders, we ran the full battery on ten galaxy encoders, including a DINOv3 family spanning 22M to 6.7B parameters, and on QM9 with eight SMILES encoders (ChemBERTa-2, MoLFormer-XL, ChemFM 1B and 3B). For QM9 we set each encoder's chart dimension from intrinsic-dimension estimates (8 to 11) before running any probe, and also ran d = 16. Every run reproduces its own reference to the last digit, and our GPU rerun of the published five agrees with the paper in 39 of 40 cells, the last one borderline.

On galaxies the mismatch partial is negative and significant for magnitude and redshift in 10 of 10 encoders, for morphology in 8 and stellar mass in 6. The counterfactual pattern (help above random, help above 0.5, hurt, thinned sign test) holds in all 10 for every label. Along the DINOv3 ladder the mismatch partial for magnitude and redshift stays negative at every size, but with six models and the embedding width changing alongside size it cannot show a trend, and we do not claim one.

On molecules the mismatch partial is negative and significant for HOMO-LUMO gap, polarisability and heat capacity in 8 of 8 encoders and for dipole moment in 6 (both ChemFM models fail on dipole moment), with the same counts at d = 16. Bending along the model's normal beats a random bend in 32 of 32 encoder-label pairs and reversing it hurts in 32 of 32. Two parts do not transfer. Help exceeds 0.5 for only 3 or 4 of 8 encoders per label (mostly MoLFormer and ChemFM), and the thinned sign test reaches p < 0.05 for only 1 or 2 of 8. We read this as the direction of the effect generalising across domains while its size depends on the encoder. We also found that the ChemBERTa-2 tokenizer drops charges and hydrogen counts, so 2.4% of molecules share an embedding with another one, and we will state that alongside the results.

## Changes to the manuscript

- A new appendix with the full-tensor synthetic validation, including the factor-of-two magnitude bias.
- Results with the validation-tuned probe next to alpha = 100 in the main tables.
- The extended-control partials and the held-out comparison.
- Cluster-bootstrap intervals in place of permutation p-values for the main correlations, and a main claim narrowed to magnitude and redshift.
- §4 rewritten to match Appendix D, with the surrogate fidelity numbers.
- A short section on the ten-encoder galaxy sweep and the QM9 replication, with the parts that did not transfer.
