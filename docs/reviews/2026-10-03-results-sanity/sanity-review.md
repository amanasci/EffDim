# Sanity review of the ML4PS rebuttal (2026-10-02 draft)

Reviewer stance: skeptical, read-only. Every number below was recounted from the report tables and, where records exist locally, recomputed from the raw jsonl/npz. Scratch scripts are in `/home/akagi/.claude/jobs/56761008/tmp/opus-check/` (`cf_check.py`, `cf_check2.py`, `qm9_check.py`, `qm9_robust.py`, `scaling_boot.py`). Abbreviations: R = `docs/rebuttal/2026-10-02-ml4ps-rebuttal.md`, TF = `curvature-experiment/results/tensor-fidelity/REPORT.md`, RR = `curvature-experiment/results/review-robustness/REPORT.md`, SC = `curvature-experiment/results/scaling/SCALING_REPORT.md`, QM = `curvature-experiment/results/qm9/QM9_REPORT.md`. "Cache" = `notebooks/.cache/` (what `curvature-experiment/.cache` symlinks to).

## Verdict

The rebuttal's arithmetic is clean. I recounted roughly 45 numbers against the four reports and found one rounding slip (-0.32 should be -0.31) and two counts that undersell the result. Every report cell I checked against the raw records matched, and the reproduction guards really do show max |diff| of 0 to 1e-15. The problems are in what the numbers are taken to show, and in what was left out. (1) Section 2 and the breadth paragraphs quote anchor-level permutation p-values, which Section 4 retires. Under the block bootstrap the tuned-probe mismatch excludes zero in 14 of 20 pairs, not 20 of 20, and alignment in 10 of 20, not 18 of 20. (2) The label-Hessian norm on its own gives the same partial as the mismatch, to within 0.05, in 36 of 40 galaxy cells (median gap 0.024). The reviewer asked for exactly this comparison (target-Hessian norm alone), and the rebuttal never reports it. The neighbourhood-residual baseline the reviewer also asked for was swapped for a label-roughness control. (3) The counterfactual's "hurt" is close to automatic by construction (hurt holds whenever t* > -0.5, and random directions are also hurt at 64 to 95 percent of anchors). "t* near 1 under alpha*" is the ridge normal-equation identity, not evidence about curvature. And the decoder-free data-side variant shows the same help pattern, so the random-direction null does not isolate the geometry. (4) On QM9 the surrogate's per-anchor agreement with the actual probe is about zero (Spearman -0.30 to +0.23). The tuned probe weakens or reverses the molecular results, and the rebuttal mentions neither. (5) The tensor-fidelity fixture has a one-dimensional in-sphere normal space, so the "full contraction" direction does not depend on the probe. The synthetic test validates the decoder mismatch column, while every real table uses the empirical one. And the "label noise" reading of low split-half cosines was never tested, because the synthetic labels carry no noise. The magnitude-and-redshift mismatch result itself is robust (bootstrap 19 of 20 at alpha = 100 and 20 of 20 at alpha* across ten galaxy encoders). But the rebuttal currently claims more about geometry than the evidence supports, and several of these gaps are visible to a reviewer who reads the tables the authors promise to add.

Finding counts: Critical 3, Important 13, Minor 12.

## 1. Number mismatches (rebuttal vs report)

Recounted from the report tables. "OK" lines are listed briefly so the coverage is visible.

| R line | Rebuttal says | Source | Recount | Status |
|---|---|---|---|---|
| R:19 | full-contraction cosine >= 0.999, in-sphere >= 0.997, mismatch Spearman 0.93 to 0.995 | TF:13-18 | 0.999-1.000, 0.997-0.998, 0.932-0.995 | OK |
| R:21 | in-sphere 0.963-0.965 (noise 0), 0.68-0.84 (noise 0.25), rho 0.88-0.99 | TF:105-116 | 0.963-0.965, 0.675-0.843, 0.882-0.991 | OK |
| R:21 | "overstates ... by about a factor of two" | TF:100-101 (relerr hess_y 0.289-1.177) | relerr is unsigned (`10_tensor_fidelity_run.py:82-90`). With the paired cosines the implied norm ratio is about 1.3x (lam labels) to 2.0x (nonlin), and could be understatement for lin/lam | Overstated and unsourced (see I7) |
| R:25 | alpha* 0.001-0.1; OOF R2 0.48-0.59 to 0.64-0.73 | RR:17-36 | 0.477-0.588, 0.640-0.727 | OK |
| R:27 | mismatch p<0.05 in 20/20 (14/20 at 100), -0.10 to -0.72 | RR:17-36 | 20/20, 14/20, -0.104 to -0.720 | Counts match, but these are permutation p's (C1) |
| R:27 | alignment positive significant 18/20 | RR:17-36 | 18/20 (dinov3 stellar 0.078, clip stellar -0.073) | Count matches, permutation p (C1) |
| R:27 | help 0.93-0.99, hurt 1.00, random help 0.05-0.11, sign test p<0.001 in 20 | RR:17-36; records | help 0.930-0.994, hurt 0.998-1.000 (rounds to 1.00), random 0.049-0.113, sign p 4.8e-7 to 7.4e-4 | OK (hurt is a ceiling, I1) |
| R:27 | t* 1.5-3.4 at 100, 0.96-1.36 at alpha* | RR | 1.52-3.44, 0.96-1.36 | OK (interpretation wrong, I2) |
| R:31 | extended-control mismatch p<0.001 in 20/20, -0.32 to -0.74 | RR:71-90 | 20/20 at p floor 0.0005, range -0.312 (vit_large smooth) to -0.737 | **Rounding slip, should be -0.31 to -0.74** (M1) |
| R:33 | held-out median gain +0.015 to +0.26 in all 20 (alpha*), p05 > 0 in 18/20 | RR:71-90 | medians 0.015-0.258, p05 > 0 in 18/20 | OK, but "raises ... in all 20 pairs" means the median is positive. Per-split fraction > 0 is 0.75-1.00 (M4) |
| R:33 | alpha = 100, median positive 19/20, p05 > 0 in only 7 | RR:44-63; records | median > 0 in 19/20. **p05 strictly > 0 in 9/20** (vit_base smooth +0.00010 and clip photo +0.00029 print as +0.000) | Undercount, conservative (M2) |
| R:39 | bootstrap mag/z 10/10 alpha*, 9/10 at 100 | RR:98-177 | 10/10, 9/10 (convnext mag fails at 32 blocks) | OK |
| R:39 | morphology and stellar mass "only 4 of 10" | RR | 4/10 **at alpha* only. At alpha = 100 it is 0 of 10** | Omits the weaker number (M3) |
| R:45 | fidelity gap 0.035-0.093 / rho 0.35-0.82 (100), 0.10-0.28 / 0.16-0.66 (alpha*) | RR:185-224 | 0.0353-0.0929 / 0.354-0.817, 0.0989-0.2786 / 0.162-0.663 | OK |
| R:49 | 39 of 40 cells agree, one borderline | SC:132 | 39 agree, 1 borderline | OK |
| R:51 | galaxy mismatch mag/z 10/10, morph 8, stellar 6; counterfactual all 10 | SC:6-9, 22-39 | same | Counts match, permutation p (C1) |
| R:53 | QM9 mismatch 8/8 gap, alpha, cv; mu 6/8; same at d=16 | QM:66-76 | same | Counts match. chemfm_3b mu is significantly *positive* (+0.111, p 0.015), not just a fail (I13) |
| R:53 | beats random 32/32, hurt 32/32, help > 0.5 for 3-4 of 8, sign test 1-2 of 8 | QM:80-97 | same | OK. Pre-stated claim (c) is conjunctive and passes 15/32. (d) passes 6/32 (M10) |
| R:53 | ChemBERTa duplicates 2.4% | QM:53 | 3,144/130,744 = 2.40% | OK |
| R:49 | QM9 d 8 to 11 | QM:10-17 | 8-11 | OK |

No cherry-picked ranges were found inside the reported tables. The cherry-picking is by omission (C1, C2, C3, I10, I13).

## 2. Record spot-checks

| Report | What I checked | Source | Result |
|---|---|---|---|
| TF | Medians over 3 seeds at n=64000, noise 0 for lin, lam0.5, lam2 (cos pf_full, cos pf_tan, rho mismatch, relerr hess_y); n=4000 noise 0.5 lam0.5 rho | `results/tensor-fidelity/records/*.jsonl` | All match TF to 3 dp (e.g. lam0.5 rho 0.9318 vs +0.932; n4000 z0.5 lam0.5 rho 0.490) |
| TF | Paper scale, noise 0 and 0.25, all 6 labels | `full_noise0.jsonl`, `full_noise025.jsonl` | Match (lin pf_tan 0.9649, relerr hess_y 0.880; nonlin 0.9626 / 1.177; noise 0.25 lin pf_tan 0.843) |
| TF | **Red flag** cos pf_tan identical across labels: lin 0.9975990, lam0.5 0.9975991, lam2 0.9975991 at n=64000 | same | Explained by the fixture geometry (I5). The contracted tensor's direction does not depend on w |
| TF | n=4000 noise 0 has seeds 1,2 plus seed 0 in `small.jsonl` | records dir | Complete. Paper-scale cells are one seed each (M8) |
| RR | vit_base mag_r alpha=100: help 0.975, hurt 1.000, random help 0.260, t* 2.30, mismatch -0.387, p 0.0005 | `results/review-robustness/records/11_review_robustness_vit_base.jsonl`, and independently `notebooks/.cache/09_physics_normal_scaling_vit_base_d16.npz` | Match |
| RR | All 20 pairs at alpha=100, recomputed help/hurt/random/t* from the published cf npz | `notebooks/.cache/09_physics_normal_scaling_*_d16.npz` | All match RR. Identity check: help ⇔ t* > 0.5 and hurt ⇔ t* > -0.5 hold at 100% of anchors in every cell |
| RR | Bootstrap and thinned counts at 16/32/64 blocks | RR records | Mismatch alpha* 13/14/14 of 20; alignment alpha* 9/10/13 of 20; thinned mismatch 5/20 (100), 9/20 (alpha*) |
| RR | Held-out p05 exact values | RR records | Two "+0.000" cells are +0.00010 and +0.00029 (M2) |
| RR | Permutation floor | RR records | p = 0.00049975 = 1/2001 (n_perm 2000). "<0.001" is the floor, presented correctly as "<0.001" |
| SC | vit_base main_xfit multiscale mismatch_emp/dec/hess_label/align; the published split record | `notebooks/.cache/scaling/records/scaling__vit_base__main_xfit.jsonl`, `notebooks/.cache/09_physics_probe_facing_split.jsonl` | emp -0.387 (= paper -0.39), dec -0.304, hess_label -0.277, align +0.353. Published = emp, not the decoder mismatch (I6) |
| SC | Bootstrap counts over all 10 encoders | `notebooks/.cache/scaling/records/*__robust.jsonl` | mismatch mag 9/10 & 10/10, photo 10/10 & 10/10, smooth 3/10 & 7/10, stellar 4/10 & 7/10 (alpha=100 & alpha*); alignment mag 4/10 & 8/10, photo 5/10 & 7/10 |
| SC | Selection check: stray `scaling__llava_15_13b__main_xfit.jsonl` | scaling records | Environment row only (a `_test.parquet` memory probe). The 10-encoder set is fixed in `docs/superpowers/specs/2026-09-30-c1-ladder-design.md:30-43` before the runs. No selection found |
| QM | All 32 cells: help, random help, hurt, thinned p_help, multiscale mismatch and alignment at d_run and d=16 | `notebooks/.cache/qm9/arrays/*__cf.npz`, `*__thin.npz`, `records/*__main_xfit.jsonl`, `*__main_d16.jsonl` | All match QM (e.g. chemberta_77m_mlm cv -0.299 vs -0.30; molformer cv -0.659 vs -0.66; chemfm_3b mu +0.111 p 0.015). Thinned n is 30-36 |
| QM | Robust records (bootstrap, surrogate, tuned) not summarised in QM | `notebooks/.cache/qm9/records/*__robust.jsonl` | Surrogate rho -0.30 to +0.23 at alpha=100. Tuned help drops (ChemBERTa 0.12-0.41). ChemFM tuned alpha/cv mismatch +0.06 to +0.13 with CIs spanning 0 (C3) |
| QM | Help/hurt definition parity with galaxies | `curvature-experiment/sweep/extract.py:57-73` used for both | Identical definitions. QM9 random-direction null is the same `random_qmatched` |

## 3. Statistical and interpretation concerns

### Critical

**C1. Section 2 and the breadth paragraphs rely on the inference Section 4 retires.**
RR:13 says the Concern 2 p-values "are anchor-level permutation p's; Concern 4 gives dependence-aware intervals, which are the ones to quote". R:27 still quotes "p < 0.05 in 20 of 20", "Alignment ... significant in 18 of 20". R:31 quotes "p < 0.001 in 20 of 20". R:51 and R:53 quote permutation counts for 10 galaxy encoders and 8 QM9 encoders. Recomputed under the 32-block bootstrap (RR records, scaling and QM9 robust records):
- five encoders, alpha*: mismatch 14/20 (magnitude and redshift 10/10, morphology and stellar mass 4/10), alignment 10/20 (vs 18/20 claimed). At alpha=100: mismatch 9/20, alignment 9/20.
- ten galaxy encoders, alpha=100: smooth_fraction 3/10 and stellar_mass 4/10 (vs "8" and "6" in R:51). Alignment mag 4/10, photo 5/10.
- QM9, alpha=100: mu 5/8 (vs 6/8 in R:53). Alignment 1/8 (gap), 4/8 (mu), 3/8 (alpha), 1/8 (cv).
- thinned partials (n about 30, the only analysis with near-independent anchors): mismatch mag/z 5/10 at alpha=100 and 6/10 at alpha* in the five encoders.
A reviewer who reads R:37 ("We replaced unrestricted permutation inference ...") and then R:27 will notice at once. Also, across encoders the 20 or 40 tests share the same 86,471 galaxies, the same anchor draw and the same labels, so "N of 10" is not N independent replications (I12).

**C2. The mismatch partial is, in rank, almost the label-Hessian norm, and the two baselines the reviewer asked for are not reported.**
Reviewer concern 3 asks for "comparisons against target-Hessian norm alone and simple neighborhood residual estimates". Multiscale partials from `notebooks/.cache/scaling/records/*__main_xfit.jsonl` (column `hess_label` = |Hess_M y|_g, `09_physics_probe_facing_split_run.py:184,190`):

| encoder | mag_r emp / dec / \|Hy\| | photo_z emp / dec / \|Hy\| |
|---|---|---|
| vit_base | -0.39 / -0.30 / -0.28 | -0.45 / -0.39 / -0.37 |
| dinov3_vitb16 | -0.57 / -0.58 / -0.54 | -0.61 / -0.60 / -0.59 |
| vit_large | -0.44 / -0.45 / **-0.46** | -0.46 / -0.44 / -0.42 |
| dinov3_vit7b16 | -0.65 / -0.64 / -0.65 | -0.37 / -0.38 / -0.38 |
| convnext_base | -0.32 / -0.33 / -0.33 | -0.48 / -0.46 / -0.45 |

Across all 40 galaxy cells (smooth and stellar included) |emp - |Hy|| is at most 0.05 in 36, at most 0.03 in 27, with a median of 0.024. The largest gaps are vit_base mag_r (0.11) and photo_z (0.08). ViT-B is the paper's headline encoder, so its main table happens to show the mismatch at its most separated from |Hy|. The decoder's probe-facing term pf_full on its own has inconsistent signs (+0.13 to -0.40). So the paper's marginal evidence ("mismatch predicts local accuracy") cannot be told apart from "target curvature predicts local accuracy". The rebuttal uses |Hy| only as a control (R:31). Because mismatch ≈ |Hy| in rank, a partial that controls |Hy| measures a small residual difference. Several partials *grow* once |Hy| is controlled (13 of 20 at alpha*, e.g. clip stellar -0.154 to -0.470, dinov3 stellar -0.104 to -0.460, RR:71-90), which is the suppression signature RR:67 itself warns about. The honest "added value" number is the held-out Delta R2 over the extended controls. At the paper's alpha that is a median of 0.001 to 0.147, with p05 > 0 in 9 of 20.
The neighbourhood-residual baseline was never run. R:31 substitutes "one minus the label's linear R2", which is about the label alone and not the probe's residual. A residual baseline would be nearly decisive, because the outcome (local R2) is computed from the OOF residuals of those same 2,048 neighbours (`11_review_robustness_run.py:99-100`). Even cross-fit, half-A residual variance would predict half-B local R2 far better than any curvature term. The paper's own pitch (main.tex:306, "estimated from labelled neighbours without knowing an object's target value") applies equally to that trivial baseline. The rebuttal needs to concede that the mismatch explains where local error comes from, not that it predicts error better than the neighbours' own residuals.

**C3. QM9 results the rebuttal does not mention undermine the molecular counterfactual and the "stronger with tuning" generalisation.**
From `notebooks/.cache/qm9/records/*__robust.jsonl` (QM does not summarise these records):
- Surrogate fidelity at alpha=100: Spearman(S_model, S) between -0.30 and +0.23 in all 32 pairs, with median gap 0.05-0.48. At alpha* the gap is 0.19-0.84. The molecular surrogate does not track the actual probe per anchor, so "bending along the model's normal beats a random bend in 32 of 32" (R:53) says little about the probe. The data-side variant S, with no decoder involved, helps at 0.88-1.00 of anchors on every QM9 pair (`qm9_check.py` output), while S_model helps at 0.28-0.74. The non-transfer is decoder or surrogate fidelity, not "size depends on the encoder" (R:53).
- Tuned probe on QM9. Help falls (ChemBERTa 0.12-0.41, MoLFormer 0.56-0.60, ChemFM 0.38-0.54). t* is 0.11-0.60, not about 1. The ChemFM 1B/3B mismatch for alpha and cv turns positive (+0.13, +0.06, +0.09, +0.11) with bootstrap CIs spanning 0, and the held-out gain goes to about 0 (-0.010 to +0.013). R:27 ("With the tuned probe the results get stronger") and R:27's t* remark do not survive the second domain.
- alpha* sits at the grid edge 0.001 for 6 of 8 encoders on nearly every label. At alpha=100 the ChemFM probes are badly over-regularised (OOF R2 0.27-0.63, vs 0.54-0.90 tuned; D = 2048/3072 at a fixed alpha). So the QM9 alpha=100 ChemFM numbers describe an underfit probe (I14).

### Important

**I1. "Hurt" is close to automatic, and the random null does not isolate geometry.**
Since SSE(t) = |r0 - t q_c|^2 (main.tex:531; `09_physics_normal_scaling_run.py:101`), help ⇔ t* > 0.5 and hurt ⇔ t* > -0.5 (verified on 100% of anchors in all 20 published npz). Hurt ⊇ help always. The rebuttal gives the model hurt (1.00) next to random *help*, but random *hurt* is 0.64-0.80 at alpha=100 and 0.86-0.95 at alpha* (RR records, `cf.random_qmatched.hurt`). Neither RR's table nor R reports it. The random direction is not a component of the fitted w (`09_physics_normal_scaling_run.py:77-79`), while w_S is. The data-side variant S, which uses no decoder geometry, helps at 0.94-1.00 of anchors with hurt 1.000 everywhere (`cf_check.py`). So "the probe's own normal component helps" is expected of any least-squares fit. What S_model vs random actually shows is that the decoder's second-order model correlates with how the fitted probe uses its normal component. A null that isolates the decoder would keep w_S and use a mismatched II (for example II from another anchor, or a rotated normal frame).

**I2. "t* near 1 under alpha*" is the ridge identity.**
In-sample ridge satisfies X^T r = alpha w. So for the global analogue of the counterfactual on the component w_S, t*_global = 1 + alpha |w_S|^2 / |X w_S|^2 >= 1. It tends to 1 as alpha goes to 0, and is large when X has little variance along w_S, which is the case for normal directions by definition. That is exactly 1.5-3.4 at alpha=100 and about 1 at alpha*. The paper's own caption already attributes t* > 1 to shrinkage ("globally ridge-regularized probe under-using a locally beneficial term", main.tex Table cf caption). R:27's reading ("the tuned probe wants roughly the decoder's own curvature") should go. On QM9 tuned t* is 0.11-0.60 (C3), which also argues against a general law.

**I3. The tuned-probe comparison is not like for like.**
alpha* sets w, and so w_N, the mismatch_emp Hessian of p = Xw, alignment and the counterfactual (`11_review_robustness_run.py:328-333`; RR:13). Specific issues:
- alpha* is at the grid edge 0.001 in 5 of 20 galaxy pairs (RR:18,19,26,27,28). For vit_base photo_z the global LOO pick is 0.001 while all five folds pick 0.1 (RR:18), so the LOO curve is flat across two decades. Prediction is insensitive there, but w_N lives in the low-variance normal directions, where ridge matters most. The geometry columns at 0.001 vs 0.1 can differ while R2 does not. There is no sensitivity check.
- The random-help drop from 0.22-0.36 to 0.05-0.11 is partly mechanical. q_rm is rescaled to the model term's centred amplitude (`09_physics_normal_scaling_run.py:92`), and a less shrunk w_S gives a larger q, so t*_random shrinks.
- The tensor-fidelity validation used alpha = 100 (`10_tensor_fidelity_run.py:235`, `physics_curvature_probe.py:185`). The near-OLS w_N used at alpha* is unvalidated.
- The robustness Hessians are not cross-fitted (I11).

**I4. The "label noise" reading of real split-half cosines is untested.**
R:21 says the low split-half cosines (0.19-0.35) are label noise because "the same estimator recovers a known Hessian at the same n and D". The synthetic labels are exact smooth functions of the latent (`10_tensor_fidelity_run.py:153-175`). "noise" adds isotropic noise to the embeddings X only (`:211-216`). No synthetic split-half cosine is computed. The experiment therefore cannot separate label noise from other explanations: targets that are not smooth at the k=2,048 scale (vote fractions, catalogue systematics), worse real decoder fit (var. explained 0.966-0.985 vs 0.9999 synthetic), or real intrinsic dimension above 16. Note also that at the realistic noise 0.25 (var. expl. 0.971) the synthetic p25 of cos pf_tan is 0.48-0.69 (TF:111-116).

**I5. The fixture's in-sphere normal space is one-dimensional, so the "full contraction" direction does not depend on the probe.**
`InSphereGenerator.raw` puts the surface in the span of d+2 coordinates, [stereo(z) (d+1); a·bumps(z) (1); zeros] (`09_instrument_adjudication_run.py:267-270`). Inside that subspace's unit sphere the surface has codimension 1, so II_tan = n ⊗ A and <II_tan, w_N> = (n·w_N) A. Its direction is A for every probe, which is why cos pf_tan agrees to 1e-6 across lin, lam0.5 and lam2 (records). The test does validate the full d x d shape tensor, which goes beyond the trace and is real progress on concern 1. What it cannot validate is how w_N selects among the up to d(d+1)/2 normal directions II spans in real embeddings, and that is the mechanism the reviewer's "full contraction with the probe" points at. relerr pf_tan (0.31-0.49 at paper scale) is the only probe-dependent check.

**I6. The column validated synthetically is not the column in the tables.**
`10_tensor_fidelity_run.py:241-249` scores mismatch = hess_y - <w_N, II> (`hess_mismatch_dec`). Every real-data partial (paper tables, RR, SC, QM) uses `hess_mismatch_emp` = |hess_y - Hess(p)|, where Hess(p) is the local quadratic fit of the probe's predictions (`09_physics_probe_facing_split_run.py:192`; RR:3). That column contains no second fundamental form at all. It is the quadratic part of the residual y - Xw in the decoder's tangent chart, fitted on the same neighbours as the outcome. Paper main.tex:304 defines Δ with <w_N, II>. On galaxies emp and dec give partials within about 0.1 (vit_base mag -0.39 vs -0.30), so the numbers barely move. But R:19's "mismatch ranks correlate with the true mismatch" validates a different object than the one tabulated.

**I7. The magnitude-bias statement is unsourced, and the "rank-based, so unaffected" argument is wrong in general.**
relerr is unsigned (`10_tensor_fidelity_run.py:82-90`). Combining TF's median cosine and relerr gives an implied scale of about 1.28x (lam0), 1.3-1.5x (lam0.5-lam2), 1.8x (lin) and 2.0x (nonlin). "Overstates" is directly implied only for nonlin, where relerr > 1 with a positive cosine. "About a factor of two" describes the two toy labels, not the four realistic ones. Inflating hess_y but not K is not a monotone transform of |hess_y - K|, so rank-based partials are not automatically safe. The real evidence that ranks survive is the measured rho mismatch of 0.88-0.99. RR:228 repeats the same argument.

**I8. "Over-controls if anything" is contradicted by the data.**
R:31 (and RR:40) assert that roughness over-controls. Over-control would shrink the partial. Instead it grows in 13 of 20 pairs at alpha* (RR:71-90 vs RR:17-36), most sharply for stellar mass, where the published-control partial is near zero. RR's own caution about suppression (RR:40, 67) did not make it into R.

**I9. The surrogate captures a minority of what it stands in for.**
At alpha=100 the surrogate's own median Delta R2(t=1) is 0.013-0.030, against 0.050-0.127 for the data-side readout it replaces. So it recovers about 0.18-0.30 of the probe's in-sphere normal-component gain, and S_model < S at 95-100% of anchors (`cf_check2.py` on the published npz). The median per-anchor gap (0.035-0.093) is 2-5 times the surrogate's own effect. "Tracks the probe reasonably at the published ridge strength" (R:45) is generous. Ten-encoder ranges: rho 0.36-0.91, gap 0.030-0.093 (alpha=100); rho 0.15-0.73, gap 0.093-0.339 (alpha*).

**I10. Alignment is dropped from the breadth evidence.**
main.tex:309 calls mismatch *and* alignment the primary hypotheses. R:51 and R:53 report only mismatch and the counterfactual. SC:15-18 shows alignment mag positive-significant in 8/10 (DINOv3 L and H+ fail) and photo_z in 7/10, and that is with permutation p. Bootstrap gives 4/10 and 5/10 at alpha=100. QM:101-104 shows alignment 1/8 for gap, 2/8 for alpha, 3/8 for cv. Leaving this out reads as selective.

**I11. The robustness battery is not cross-fitted.**
`11_review_robustness_run.py:108-111` fits the label and probe Hessians on the full 2,048-neighbour panel, and `:99-100` computes local R2 on the same panel. The paper's own answer to shared-observation coupling (cross-fitting, Appendix B) was not repeated for alpha*, the extended controls, the bootstrap or the held-out test. At alpha=100 the paper shows cross-fitting changes little, but the "stronger at alpha*" claim was never checked under cross-fitting.

**I12. Counts across encoders are not independent replications.**
All galaxy encoders share the same 86,471 objects, the same anchor indices (`11_review_robustness_run.py:410-412`) and the same labels. If the signal is target curvature (C2), it lives in the labels and would be shared across encoders. "10 of 10 encoders" should be read as robustness to the embedding, not as 10 confirmations.

**I13. ChemFM 3B dipole moment is a significant reversal.**
QM:139 lists chemfm_3b mu mismatch at +0.11 with no "(ns)", and the record gives +0.111, p 0.015 (permutation). R:53 calls it a "fail". It is a significant effect of the wrong sign. At d=16 it is +0.001 (ns), so the sign is unstable.

**I14. A fixed alpha = 100 across D = 384 to 3072 confounds the QM9 comparison.**
ChemFM OOF R2 at alpha=100 is 0.27-0.63, against 0.54-0.90 tuned. The "size depends on the encoder" reading (R:53) is confounded with how heavily each encoder's probe is shrunk.

### Minor

- **M1.** R:31 range should read -0.31 to -0.74 (vit_large smooth_fraction -0.3117, RR:89).
- **M2.** R:33 "clears zero in only 7". It is 9 strictly positive, two of them at +0.0001 and +0.0003. Fine to keep 7 as "clearly above zero" if worded that way.
- **M3.** R:39 "only 4 of 10" is the alpha* figure. At alpha=100 morphology and stellar mass exclude zero in 0 of 10 (RR:106-177).
- **M4.** R:33 "raises out-of-sample R2 ... in all 20 pairs" refers to medians. The per-split fraction > 0 is 0.75-1.00 (RR:71-90).
- **M5.** The "5th percentile across 20 splits" is a quantile of split-to-split variability on one dataset, close to the minimum of 20 overlapping splits. It is not a confidence bound. Adjacent blocks share boundary points (RR:94).
- **M6.** "Every run reproduces its own reference to the last digit" (R:49) is a determinism self-check of the same code on the same geometry, not an independent replication. The spec's 1e-6 guard (`docs/superpowers/specs/2026-09-30-review-robustness-design.md:120`) was relaxed to 0.02 ("refit") for four encoders (RR:5-9). This is moot because the actual diffs are 0 / 1e-15, but say "exact".
- **M7.** `src/effdim/geometry.py:348-394` `tle_dimensionality` is the same formula as MLE (a library bug worth an issue, not a rebuttal item). two_nn is broken for ChemBERTa by duplicates (0.85-1.04, QM:10-14). So d is effectively round(mean(mle, mind_mlk)). MLE's +1e-10 epsilon (`geometry.py:66`) does not protect against duplicates and biases ChemBERTa MLE slightly low. The d=16 replicate (QM:71-76) covers the conclusion, so "intrinsic-dimension estimates" in R:49 is acceptable but generous.
- **M8.** The paper-scale tensor-fidelity cells are single-seed (`full_noise0.jsonl`, `full_noise025.jsonl`). The small fixture uses 64 anchors.
- **M9.** The pre-registered pf_full line is nearly vacuous (TF:5). R:17 concedes this, which is good.
- **M10.** QM9 pre-stated claim (c) is "help > random help **and** help > 0.5" (`docs/superpowers/specs/2026-10-01-c2-qm9-design.md:171`). R:53 leads with the passing half (32/32). Under the pre-stated criterion (c) passes 15/32 and (d) 6/32. R:53 does disclose both halves.
- **M11.** Multiplicity. Even with permutation p, a Bonferroni correction over 20 drops vit_base stellar (p 0.005) and dinov3 stellar (p 0.023) at alpha*, giving 18/20. This is small next to C1.
- **M12.** DINOv3 ladder (SC:171-204). photo_z mismatch weakens with size (-0.63 to -0.33/-0.37; descriptive Spearman +0.89), and mag alignment falls to about 0 for L and H+. "Stays negative at every size" (R:51) is true but leaves out the weakening.

## 4. Things the rebuttal should add or soften

Suggested prose avoids colons and em dashes.

1. **R:27, after the counts.** "These p-values come from anchor-level permutation, which treats overlapping neighbourhoods as independent. Under the block bootstrap of Section 4 the tuned-probe mismatch interval excludes zero in 14 of 20 pairs, all 10 for magnitude and redshift, and the alignment interval excludes zero in 10 of 20."
2. **R:27, replace "One detail stood out ... wanted more."** "The median best bend scale falls from 1.5 to 3.4 at alpha = 100 to about 1 at alpha*. Ridge shrinkage predicts exactly this for any part of the fitted weight, so we do not read it as evidence about curvature."
3. **R:27, counterfactual.** "Reversal hurting is close to automatic in this design, because it holds whenever the best scale is above minus one half. A random direction is also hurt by reversal at 64 to 95 percent of anchors, so the comparison that carries information is help against random help."
4. **R:27, tuned probe.** "For 5 of the 20 pairs the selected ridge strength is the smallest value on our grid, so the tuned probe is close to unregularised there, and the tuned weight also changes the probe-facing tensors. The two columns are therefore not a like-for-like comparison of the same geometry."
5. **R:31, report the baseline the reviewer asked for.** "On its own the label-Hessian norm gives partials within a few hundredths of the mismatch partial in most encoder-label pairs, so most of the marginal association is target curvature. What the probe-facing geometry adds beyond it is the held-out gain below, which at the published ridge strength is a few hundredths of R² for most labels and clears zero in 9 of 20 pairs." Delete "so it over-controls if anything", or replace it with "Adding these controls made the partial stronger in 13 of 20 pairs, which can be a sign of suppression, so we read these partials with care."
6. **R:31 or R:33, neighbourhood residuals.** "We did not run a baseline built from the neighbours' own probe residuals. Local accuracy is measured on those same neighbours, so such a baseline would predict it almost exactly. We therefore present the mismatch as an account of where local error comes from, not as a better predictor of it."
7. **R:21, magnitude.** "At this scale the relative error of the label Hessian is 0.29 to 1.18, consistent with its size being inflated by roughly 1.3 to 2 times depending on the label." Replace "Our partials are rank-based, so this does not change them" with "The mismatch ranks still agree with the truth at Spearman 0.88 to 0.99, which is our evidence that the partials survive this bias."
8. **R:21, label noise.** Replace the last sentence with "Our synthetic labels carry no label noise, so this test cannot separate label noise from other causes of the low split-half cosines on real labels, such as targets that are not smooth at the neighbourhood scale. We will add a synthetic run with label noise before making that claim."
9. **R:17 or R:19, fixture limitation.** "In this fixture the surface bends in a single direction inside the sphere. The test therefore checks the full d by d shape tensor, but not how the probe chooses among many normal directions, which real embeddings have."
10. **R:19, which column.** "The synthetic test scores the mismatch built from the decoder's probe-facing tensor. Our tables use the mismatch built from the empirical Hessian of the probe's predictions. On galaxies the two give partials within 0.1 of each other, and we will report both."
11. **R:39.** "...in only 4 of 10 with the tuned probe, and in none at alpha = 100."
12. **R:45, fidelity.** "Even at the published ridge strength the surrogate recovers only about a fifth to a third of the local gain that the probe's own in-sphere normal component gives on the data."
13. **R:51, breadth.** Use bootstrap counts. "Under the block bootstrap the mismatch interval excludes zero for magnitude and redshift in 19 of 20 encoder-label pairs at alpha = 100 and 20 of 20 with the tuned probe, and for morphology and stellar mass in 7 of 20 and 14 of 20." Add alignment. "Alignment is weaker. Its interval excludes zero for magnitude in 4 of 10 encoders and for redshift in 5 of 10 at alpha = 100."
14. **R:53, QM9.** "On molecules the per-anchor agreement between the surrogate and the probe is close to zero, so the molecular counterfactual says little about the actual probe. With a tuned probe the molecular counterfactual weakens, and the mismatch partial loses significance for polarisability and heat capacity on both ChemFM models. On ChemFM 3B the dipole-moment partial is significant with the opposite sign." Replace "We read this as the direction of the effect generalising ... depends on the encoder" with "We read the mismatch result as transferring to molecules for three of four properties. The counterfactual does not transfer, and we think the decoder's fit is the reason."
15. **R:51.** "These encoders share the same galaxies, anchors and labels, so the counts show robustness to the choice of embedding rather than independent replications."
16. **R:27, line 1.** Soften "With the tuned probe the results get stronger, not weaker" to "With the tuned probe the galaxy results hold and most partials get larger, though on molecules they do not." Without this, C3 would be a fair rebuttal-of-the-rebuttal.

## 5. What holds up well

- **Arithmetic fidelity.** About 45 rebuttal numbers recounted from the reports. One rounding slip (M1) and two conservative undercounts (M2, M3) were found. No inflated range.
- **Report fidelity.** Every spot-checked report cell matches the raw records across all four reports, including independent recomputation of the published counterfactual from `09_physics_normal_scaling_*_d16.npz` and of all 32 QM9 cells from npz and jsonl.
- **Reproduction guards are genuine.** max |diff| 0 for split cells and at most 2.2e-15 for counterfactual values on the five encoders (RR records, guard rows). Exact 0 on all 10 galaxy and 8 QM9 robust jobs. The OOF R2 identity holds to 1.1e-16 (SC:134-138).
- **The core magnitude/redshift association is robust to dependence.** The 32-block bootstrap excludes zero in 19/20 (alpha=100) and 20/20 (alpha*) magnitude/redshift pairs across ten galaxy encoders, stable at 16 and 64 blocks. QM9 gap, alpha and cv: 8/8 at alpha=100 under bootstrap.
- **The tuned-probe OOF R2 is leak-free.** Alpha is chosen inside each outer fold (`11_review_robustness_run.py:65-80`). Only the geometry-setting w uses the all-rows alpha*, and RR:13 discloses this.
- **Help/hurt definitions are identical across galaxies and QM9** (`sweep/extract.py:57-73`), so the cross-domain comparison is like for like in definition.
- **The pre-stated d for QM9 is real.** The d file sha256 is in every split, cf and robust environment row (QM:188-197), and the d=16 replicate gives identical counts.
- **No encoder selection.** The ten-encoder set is fixed in the C1 spec before the runs. The only extra record (LLaVA) is an environment-only memory probe.
- **Honest concessions already in the draft.** Radial-term dominance and the post-hoc pf_tan line (R:17-19), worse surrogate fidelity at alpha* reported next to the better number (R:45), morphology and stellar mass narrowed to inconclusive (R:39), and QM9 non-transfer of help > 0.5 and the sign test (R:53).
