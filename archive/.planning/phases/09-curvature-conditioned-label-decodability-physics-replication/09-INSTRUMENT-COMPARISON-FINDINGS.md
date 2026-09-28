# Two curvature instruments, one pipeline: what Phase 9 measured

Status: findings for external review. Written 2026-09-05 on branch `fixture-validity-audit`.
Records: `09-WAVE-A-RESULTS.md`, `09-WAVE-A-RESULTS-AMENDMENT-01.md`,
`09-SUPPLEMENT-01-COLLEAGUE-ESTIMATOR.md`, `09-SUPPLEMENT-02-INSTRUMENT-ADJUDICATION.md`,
`09-PREREGISTRATION.md`, `09-PREREGISTRATION-AMENDMENT-01.md`. Every number below comes from
those records or the JSONL files they cite. Nothing here reopens a sealed verdict.

The colleague's branch is `origin/curvature-experiments` at commit `97efb2eb`. His frozen result:
controlled Spearman between local curvature and the local out-of-fold R² of a ridge probe for
r-band magnitude equals −0.240 at chart rank 16, k = 2048, 512 anchors, ViT-B Physics
embeddings. We ran his design with our curvature estimator, then his estimator inside our
pipeline, then both estimators against a known answer. The three runs disagree in a way that
identifies which estimator to trust.

## 1. The two approaches

| | His approach (`curvature-experiments`) | Ours (Phase 7 to 9) |
|---|---|---|
| Data | ViT-B Physics embeddings, row-L2-normalised, hash-selected 16,384-row subset | Same embeddings, all 86,471 test rows, row-L2-normalised |
| Neighbourhood scale | k = largest preset with k ≤ n/8, so k = 2048; one inner-product k-NN table over the subset, anchor excluded | k = 2048 fixed to match his; Euclidean k-NN over all rows (identical ordering on the unit sphere); n/k = 42 where his was 8 |
| Anchors | 512 | 512, drawn from the 20 % of rows the autoencoder never trained on |
| What "curvature" means | Quadratic fit to the 2048 neighbours after sphere-centring and radial removal, nested PCA chart of rank d, two-stage ridge on quadratic features, second fundamental form projected sphere-normal, mean-curvature vector `H = (1/d) Σ B_ii`. Reported statistic `K_H^cross = ⟨H_A, H_B⟩` from three random split-halves of the neighbourhood. Curvature at the scale of the neighbourhood, k-dependent by construction. | Exact `H = tr_g(II)` of a trained decoder map `F: R^d → R^768` at `z = encode(x)`, by autodiff. `PlainAutoEncoder(768 → d, hidden (250, 250, 250), SiLU)`, 600 epochs, no early stopping. Since Amendment 01 the decoder image is projected to the unit sphere before differentiation, so the radial component equals −d and `‖H_tan‖` is the mean curvature within the sphere. Pointwise, no neighbourhood, scale set by the decoder. |
| Chart rank / latent d | 12, 16, 20 (positions fixed from a geometry-only variance rule) | 16, 20, 25, 32 |
| Outcome | Global 5-fold OOF ridge (α = 100) to `mag_r_desi`, scored as R² inside each anchor's 2048-neighbourhood | Identical construction, same α, same neighbourhood, same label column (`mag_r_desi`, ratified in `09-DATA-MANIFEST.md`) |
| Controls and inference | Rank-partial Spearman on log kNN radius, local label variance, evaluation count; Freedman-Lane permutation with FWER across d; paired-anchor bootstrap | Identical, plus a density-stratified permutation null |
| Reliability or validation | Split-half `R_H` (his run: median 0.514 at d = 16). No known-answer test. | Fixture fidelity 0.84 to 0.99 at d = 16 on flat analytic surfaces (Phase 7 sweep). Before this work, no known-answer test on a sphere-constrained, noisy, k = 2048 regime. |
| Row alignment | Assumed | Proved in-phase: shift-0 R² 0.516 against a best misaligned pairing of −0.0001, margin 0.10 |

Two design points matter for reading the results. His curvature and his outcome share one k-NN
table, so both are properties of the same 2048-point patch. Ours ties the outcome to that patch
and the curvature to a global model. And the two estimators couple to local density in opposite
directions, which section 2 quantifies.

## 2. Why we ran the comparison: the instruments disagree

Phase 9 pre-registered his design around our instrument. The frozen verdict rule asks whether the
controlled partial is negative and clears its Freedman-Lane null at any d. Wave A gave, for
`mag_r` on the gating field `H_tan`, controlled partials of +0.347, +0.030, +0.042 and −0.003 at
d = 16, 20, 25, 32, with the d = 16 cell significant at p < 1e-4. Same magnitude as his, opposite
sign. Verdict: `DOES NOT REPLICATE`.

We then ran his estimator, imported unchanged from his branch, inside our pipeline on the same
512 anchors, the same probe predictions, the same local R² and the same controls. His sign came
back: −0.149 at d = 16 (p = 0.0005) and −0.235 at d = 20 against his −0.233. At d = 12 he
reported +0.143; we measured −0.104.

On the shared anchors the two curvature fields anticorrelate: Spearman −0.463 at d = 16, −0.406
at d = 20. Most of that is density. His `K_H^cross` rises with kNN radius (Spearman +0.70 at
d = 16, +0.77 at d = 20; his own reanalysis found +0.765 on his data). Our `H_tan` falls with
radius (−0.60 at d = 16). Partial out radius and the two fields are close to unrelated (−0.07).
Local R² falls with radius (−0.23). A quadratic fitted over a wider patch bends more; a decoder
trained by mean squared error bends where it has data.

Density stratification does not reconcile them. Within radius deciles the mean Spearman with
local R² is +0.36 for our field and −0.19 for his at d = 16, every decile carrying the sign of
its instrument. So the sign of "curvature versus local decodability" on this data depends on
which curvature estimator you use, and neither the controls nor stratification remove that
dependence.

Two candidate explanations needed testing. First, our decoder is not constrained to the unit
sphere, and a synthetic test showed that radial wobble of the image can scramble the tangential
field's ordering while leaving image norms within 0.1 % of 1. Second, one or both estimators may
simply not measure the curvature of the data manifold in this regime. Amendment 01 addressed the
first; the adjudication fixture addressed the second.

## 3. Setup

### 3.1 Amendment 01: sphere-projected decoder image

Change: differentiate `F/‖F‖` instead of `F`. One new frozen constant
(`DECODER_IMAGE_PROJECTION = "sphere"`), a fresh freeze commit `e31b3010`, the SHA rewired into
both runners and both test suites, a fresh-clone ancestry proof, and a full re-run of the sweep,
both gates, verdict and seeds into a separate output root. Every other constant stayed
byte-identical. Validation before the run: on explicit analytic in-sphere maps at d = 16,
D = 768 the sealed formula reproduces `‖H‖`, `H_rad = −d`, `H_tan` and the direction of `H` to
1e-15; on off-sphere maps the projection restores that exactness to 1e-13.

### 3.2 His estimator in our pipeline

Runner `notebooks/diagnostics/09_colleague_estimator_run.py`. His code is imported from a
read-only checkout at `97efb2eb`; a small shim supplies ten names from a `topology` package his
branch imports but does not contain, none executed on the estimator path. Per anchor:
`nested_pca_frame` on the anchor's 2048 self-excluded neighbours, then `_fit_rank` at
d ∈ {12, 16, 20} with three splits and seed 0, averaged by his own `_rows_from_fits`. Anchors,
probe, local R², controls and nulls are the sealed Phase 9 objects. CPU, 16 threads,
1.92 s per anchor.

### 3.3 Known-answer adjudication

Runner `notebooks/diagnostics/09_instrument_adjudication_run.py`, commit `068490d`.

Fixture: `G(z) = normalize([φ(z); 0.8·h(z); 0…]) Qᵀ` with `φ` the inverse stereographic map
`R^16 → S^16`, `h` a sum of four Gaussian bumps in z, `Q` a fixed random rotation into `R^768`.
The image lies in the unit sphere with curvature that varies across points. Latents: 86,471 draws
from `N(0, s²I)` with `s ∈ {0.4, 0.6, 0.9}` at probabilities {0.2, 0.5, 0.3}, so the k-NN radius
varies about twofold across anchors (p05/p95 0.54/0.97). Two conditions: no noise, and isotropic
Gaussian noise in `R^768` re-normalised to the sphere, scaled so the median displacement equals
25 % of the median 2048-neighbourhood radius (σ = 0.0063 per coordinate, displacement 0.175).

Truth: the sealed autodiff of the explicit generator at each anchor's own latent. The generator is
a closed-form smooth map with an in-sphere image, so this is exact; the run asserts
`max|H_rad + 16| < 1e-8` and observed 2.1e-14. Truth `‖H_tan‖` has median 0.82 and a p95/p05
spread of 46.5, wider than the real field's 1.8 at d = 16, which makes the rank test permissive.

Both estimators saw the same points, the same 512 holdout anchors drawn by the frozen rule, and
the same k = 2048 neighbourhoods. Ours: the frozen fit protocol at d = 16 with the sphere
projection. His: the path of section 3.2 at d = 16. Scores: rank Spearman against the truth for
both; for ours also direction cosine, magnitude ratio and calibration of the `H_tan` vector
against the truth vector; for his, `K_H^cross` against `‖H_tan‖²/d²` (his averaged convention,
squared; rank is invariant to that monotone map). Decision rule, fixed before the numbers: an
instrument is validated in regime if rank ρ ≥ 0.7 and, for ours, direction cosine ≥ 0.8.

Low-d anchor: the Swiss roll (3,000 points, d = 2, k = 256, 256 anchors) against the sealed
analytic curvature.

All host runs used a fresh clone of the pushed branch, a strict-ancestor check against the freeze
commit, 16 threads on a 128-core CPU host, and digest-verified transfer of every record.

## 4. Results

### 4.1 Amendment 01 changed the field, not the answer

| d | `H_rad` (expected −d) | ρ(`H_tan` original, `H_tan` amended) | `mag_r` partial, original | amended | FWER p |
|---|---|---|---|---|---|
| 16 | −16.000000 | 0.997 | +0.347 | +0.328 | < 1e-4 |
| 20 | −20.000000 | 0.992 | +0.030 | +0.016 | 0.72 |
| 25 | −25.000000 | 0.992 | +0.042 | +0.031 | 0.50 |
| 32 | −32.000000 | 0.984 | −0.003 | −0.015 | 0.74 |

Fits reproduced to four decimals in variance explained (0.952 to 0.965). Secondary labels moved
by at most 0.01. Verdict unchanged. The off-sphere image was not the cause of the positive sign.

### 4.2 His estimator inside our pipeline

| d | label | raw ρ | controlled | FWER p | his frozen reference |
|---|---|---|---|---|---|
| 12 | mag_r | −0.232 | −0.104 | 0.019 | +0.143 |
| 16 | mag_r | −0.298 | −0.149 | 0.0005 | −0.240 (raw −0.412) |
| 20 | mag_r | −0.326 | −0.235 | < 1e-4 | −0.233 |
| 16 | photo_z | +0.056 | −0.139 | 0.0016 | |
| 16 | smooth_fraction | +0.030 | −0.158 | 0.0003 | |
| 16 | stellar_mass | −0.064 | +0.032 | 0.46 | |

His `R_H` medians on our data: 0.43, 0.45, 0.45 at d = 12, 16, 20. Under our instrument the same
three labels at d = 16 read +0.328, +0.358 and +0.341.

### 4.3 Adjudication

| noise | instrument | rank ρ vs truth | direction cosine | magnitude ratio | fit or reliability | ρ vs log radius | verdict |
|---|---|---|---|---|---|---|---|
| none | AE `H_tan` | 0.938 | 0.9992 | 0.9993 | var. explained 0.9999 | +0.32 | PASS |
| none | his `K_H^cross` | 0.622 | | | `R_H` 0.80 | +0.35 | FAIL |
| 25 % of patch | AE `H_tan` | 0.754 | 0.945 | 1.093 | var. explained 0.9707 | +0.17 | PASS |
| 25 % of patch | his `K_H^cross` | −0.297 | | | `R_H` −0.61 | −0.16 | FAIL |
| | truth | | | | | +0.42 | |

Swiss roll: ours rank ρ 0.553, direction 0.9998, magnitude ratio 1.05, PASS. His estimator
returns `K_H^cross` of order 1e-25 and `R_H` = 0 on the roll: in `R^3` with d = 2 his sphere
projection leaves no normal direction for the quadratic term, so the roll cannot test him.

On a clean known answer in the production regime the autoencoder field tracks the true curvature
in rank, direction and magnitude. His estimator tracks it weakly. With noise at a quarter of the
patch radius, ours keeps rank 0.75 and direction 0.95; his becomes anticorrelated with the truth
and his split-half reliability turns negative. On the real data his `R_H` reads 0.45, a value
this fixture shows says nothing about accuracy.

### 4.4 What this settles and what it does not

Settled. The Phase 9 verdict rests on an estimator that passes a known-answer test in the regime
where the two estimators disagree, and it survives the sphere-projection amendment. The
estimator that reproduces his sign inside our pipeline fails the same test and fails hardest
under noise. We do not read −0.240 as the reference value.

Not settled. The fixture is one generator family, one noise level, and a 46× dynamic range where
the real field spans 1.8×. A noise sweep and a narrow-range fixture would sharpen the rank test.
The fixture does not show that the real Physics manifold resembles it. The positive-control gate
in the frozen pipeline is structurally unable to pass on this data (documented in
`09-WAVE-A-RESULTS.md` §5) and is open for amendment. Scale matching, meaning the autoencoder
field averaged over the same 2048-neighbourhood, has not been run; it needs the field at all
rows.

### 4.5 Reproduction

Records under `notebooks/.cache/` (gitignored, digests in the supplements): `09_physics_curvature.jsonl`
and `09-amend01/09_physics_curvature.jsonl` with their anchor tables, `09_colleague_estimator.jsonl`
with `09_colleague_anchor_table_d{12,16,20}.npz`, and `09_instrument_adjudication.jsonl`. Runners:
`notebooks/diagnostics/09_physics_curvature_run.py`, `09_colleague_estimator_run.py`,
`09_instrument_adjudication_run.py`. Freeze commits: `5f7fbe27` (original), `e31b3010`
(Amendment 01). His code: `origin/curvature-experiments` at `97efb2eb`, imported unchanged.
