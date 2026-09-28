# Phases 7, 07.1, 8 — Concept Inventory

**Date:** 2026-09-01 · **Purpose:** one row per concept used or produced across the three phases,
each with what it is and the result it generated, for verification and cognitive ownership.

Every number below I re-read from the frozen records in `notebooks/.cache/`, not transcribed from
a SUMMARY. The verification ledger in §5 says exactly which claims I checked and which I could
not.

---

## 1. Phase 7 — Curvature-Conditioned Crossmodal Alignment

**Module:** `notebooks/pu_manifold/crossmodal_curvature.py` ·
**Record:** `notebooks/.cache/07_crossmodal_curvature.jsonl` (8 rows) ·
**Freeze** `f032745f6450068c63763993d39fa112fd36bb8c` → **run** `a4537369be204b784d026ac36c6bfc7b14ea483d`,
with 10 commits strictly between (verified).

| # | Concept | What it is | Result it produced |
|---|---|---|---|
| 1 | Plain-AE decoder curvature | `‖H‖` where `H = tr_g(II)`, obtained by differentiating `model.decode` alone through `decoder_curvature.plain_decoder_curvature`, never the encoder-composed round trip | The `h_norm_{20,25,32}` fields. Medians **37.19 / 41.41 / 47.03** |
| 2 | `d`-sweep truncation probe | `D_SWEEP = (20, 25, 32)`, three candidate AE bottleneck widths, run because PU's reconstruction never plateaus and no single `d` is defensible | `var_explained` **0.98194 / 0.98432 / 0.98647**. See §4 — this figure is uncentered |
| 3 | MKNN crossmodal alignment | Mutual-k-nearest-neighbour overlap between the HSC and LegacySurvey embeddings at `HEADLINE_K = 20`, the source paper's own headline probe | Per-point score `j/k`; the alignment side of every rho in the phase |
| 4 | Headline statistic | `spearman(‖H‖, MKNN)` — does higher curvature go with worse crossmodal alignment | **-0.112181 / -0.127891 / -0.023726**, negative at every `d`, matching the research hypothesis |
| 5 | Two-tailed permutation null | N=1000 label permutations, `PERMUTATION_SEED = 20260825`, `NULL_QUANTILE_PER_TAIL = 0.975`, Bonferroni-equivalent to one two-sided 0.05 test | Negative-tail thresholds **0.020624 / 0.019683 / 0.018728**. All three `d` clear the negative tail; none clears the positive |
| 6 | `apply_verdict` frozen rule | The pre-registered mapping from per-`d` clearance plus positive-control evidence to a verdict string, committed before any number existed | **`ASSOCIATION DETECTED`**. Signature takes two parameters, neither naming density, so "diagnostics gate nothing" is structural rather than promised |
| 7 | Rank-plant positive control | Plants a known rho onto PU's own realized `‖H‖` dynamic range by 40-iteration bisection, then runs the identical permutation machinery, to prove the test could detect an effect if one existed | Grid `(0.02, 0.05, 0.10, 0.20)`; `smallest_cleared_target = 0.05`. The **0.02 row missed its own threshold by 0.00047**, so the true detection floor sits somewhere in **0.021–0.05**, unresolved by this grid |
| 8 | k-NN local density | Radius to the 30th neighbour in ambient 768-D LegacySurvey space, converted by `1.0 / w` so the quantity rises with density, matching Phase 4's sign convention | `spearman(density, ‖H‖)` = **+0.4281 / +0.3150 / +0.0118**; `spearman(density, MKNN)` = **-0.2121**; `density_ratio_p95_p05` = **5.98e7** |
| 9 | Density partial correlation | `spearman(‖H‖, MKNN)` after residualizing both against density, to ask how much of the association is curvature rather than crowding | **-0.024189 / -0.065835 / -0.021719**. Density accounts for roughly **78% / 49% / 8%** of the raw association |
| 10 | Tie-handling rule | The per-point MKNN statistic is `j/k` for integer `j`, so it takes at most `k+1 = 21` distinct values across 10,000 points | Measured **15** distinct values at `k=20`, which is why significance runs through permutation rather than `spearmanr`'s asymptotic p |
| 11 | `MKNN_K_GRID` sensitivity | Point estimates at `k ∈ (5, 10, 20, 50)`, non-gating, carrying no null of their own | Sign and rough magnitude hold at every `k` (d=20: -0.0938 → -0.1292; d=25: -0.0846 → -0.1530). Not a `k` artifact |
| 12 | Hubness and chance floor | Skewness of the k-occurrence distribution per modality, plus the `k/n` floor a random pair would score | Hubness **1.0486** (HSC) / **1.1880** (LegacySurvey); chance floor **0.002** at n=10,000, k=20 |
| 13 | Analytic-fixture instrument validation | Two closed-form surfaces (`cubic`, `ridge`) at `d=20`, `D ∈ {28, 768}`, where the true curvature is known, scoring the estimator against ground truth | `INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99)`. **cubic@768 rho = 0.5253 at 99.70% reconstruction; ridge@768 rho = 0.9745 at 99.88%.** Reconstruction does not predict fidelity — `ii_cv` 0.104 vs 0.483 does, and PU's is unknown |
| 14 | Freeze plus strict-ancestor proof | `git merge-base --is-ancestor` **and** `git rev-list --count ≥ 1`, since a commit is its own ancestor and the first check alone would pass a number produced inside the freeze commit | Both checks pass with 10 commits between freeze and run |
| 15 | Swiss-roll gate by declaration | CLAUDE.md requires a Swiss roll notebook per new manifold model; Phase 7 introduces none, reusing `cae.PlainAutoEncoder` and `plain_decoder_curvature` unedited | Satisfied by the pre-existing `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb`. A declaration of an already-met obligation, not a waiver |

---

## 2. Phase 07.1 — Density-Stratified Null and Seed Stability

**Module:** `notebooks/pu_manifold/density_stratified_null.py` ·
**Record:** `notebooks/.cache/07.1_density_stratified_null.jsonl` (34 rows) ·
**Freeze** `6768666`.

| # | Concept | What it is | Result it produced |
|---|---|---|---|
| 16 | Stratified null for the partial itself | Quantile density strata, with `h` and `m` permuted independently inside each stratum, giving the density-controlled partial its own calibrated band instead of borrowing the raw statistic's | Own-side band edges at S=20: **-0.027944 / -0.021881 / -0.030537**. Wider than Phase 7's borrowed thresholds at every `d`, by roughly **35% / 11% / 63%** |
| 17 | `apply_partial_verdict` | The frozen rule mapping per-`d` clearance of the partial's own band to a verdict | **`SURVIVES AT SUBSET OF d`**, `{20: False, 25: True, 32: False}` |
| 18 | Exact signed margins | Margins reported to full precision whichever way they land, never rounded to a clean line | d=20 falls short by **+0.0037554** (20.19% of the band's own-side half-width); d=32 by **+0.0088186** (46.65%); d=25 clears by **-0.0439539** (234.29%) |
| 19 | Stratum-count grid | `S ∈ {10, 20, 50}` — a grid of **thresholds**, not point estimates, since the observed partial is invariant to `S` | No `d`'s clearance status flips anywhere on the grid. `null_std` flat at roughly 0.0097–0.0102 throughout |
| 20 | Null-mean-vs-`S` bias diagnostic | Checks whether finer strata bias the null mean, since a positive-biased null is liberal on the negative tail every residual here sits in | **Contradicted both priors.** `null_mean` is negative at every `S` and every `d`, non-monotonic, not decaying from a positive value. D-02's stated rationale for `S = 20` ("finer strata narrow the band") does not hold on this data; the width barely moves and the mean does. The **choice** survives, the **reason** does not |
| 21 | Direction-matched positive control | The same plant run in both the positive and negative direction against a 9-rung grid, on real PU data at d=20 | `smallest_cleared_target` = **0.02 in both directions**, with no `bracket_exhausted` cell. Resolves the floor exactly in the 0.021–0.05 interval Phase 7's coarser grid could not |
| 22 | Seed stability at d=25 | Three independently initialized decoders (`TORCH_INIT_SEEDS = 0, 1, 2`) with `SPLIT_SEED`, `HOLDOUT_FRACTION`, `PERMUTATION_SEED` and `N_PERMUTATIONS` all held fixed, so any difference traces to the curvature field alone | **`SEED STABLE AT d=25`**, 3-of-3 unanimous. Partials **-0.065835 / -0.131488 / -0.138050**. `split_checksum` identical across all three rows |
| 23 | Pairwise field agreement | Spearman between the three seeds' `h_norm` fields, to prove the three fits are genuinely different rather than one measurement counted thrice | **0.7627 / 0.7231 / 0.8464**, no pair bit-identical. Real spread, so the unanimity means something |
| 24 | Seed-0 reproduction check | Refits Phase 7's own realized seed under identical settings and compares against the frozen `.npz` field | `reproduction_spearman_vs_frozen_h_norm_25 = **1.0**`, `reproduction_partial_diff = **0.0**`. The whole field reproduces, not just the downstream scalar |
| 25 | Review-debt closure with no-op proofs | WR-01 (finite/constant guards), WR-02 (deduplicated distinct-count), WR-04 (runner test coverage), each closed with a demonstrated no-op rather than an assertion | `--mode smoke` output byte-identical before and after. WR-01 protects 07.1's own call sites only — Phase 7's `run_dsweep` never calls `density_diagnostics`, so it validates nothing retroactively about the frozen jsonl |

**Carry this one:** seed 0, the seed Phase 7 actually realized, is the **weakest of the three by
roughly 2.5x** (margin fraction 234% against 584% and 598%). All three clear, so the verdict
stands, but the realized seed is the marginal one.

---

## 3. Phase 8 — Curvature-Conditioned CKA Alignment

**Module:** `notebooks/pu_manifold/cka.py` (45 frozen constants) ·
**Record:** `notebooks/.cache/08_cka_alignment.jsonl`, **211 rows verified** (66 positive-control,
66 negative-control, 79 sweep), every row carrying freeze
`f023c8fa7ee1dc2a021e998c99a65e65f6bc7eea`.

| # | Concept | What it is | Result it produced |
|---|---|---|---|
| 26 | Unbiased HSIC and CKA | The second alignment metric, using the unbiased HSIC estimator with a zeroed Gram diagonal, centered before computing anything | Supersedes Phase 7's `ALIGNMENT_METRIC = "mknn"` through an explicit `SUPERSEDES` constant rather than in prose alone |
| 27 | Linear versus RBF kernel | Two Gram constructions, with RBF bandwidths frozen from the median heuristic before any verdict | `SIGMA_HSC = 0.6420152563705613`, `SIGMA_LEGACYSURVEY = 0.5696337821442163`. RBF is non-gating by `RBF_IS_NON_GATING` |
| 28 | σ-multiplier ladder | Bandwidth robustness at `(0.5, 1.0, 2.0)× σ`, reported but barred from voting | The 6 rows carrying `cleared: null` in the record — recorded and structurally unable to change a verdict |
| 29 | Within-stratum tertile split | Splits points into `‖H‖` tertiles **inside** each density stratum, then measures the CKA gap between the top and bottom tertile, so the contrast is curvature at matched density | The gating statistic. `realized_h_contrast ≈ **1.164**`, confirming PU's curvature spread is thin |
| 30 | Stratified label-permutation null | N=500 permutations of the tertile labels within strata, `NULL_QUANTILE_PER_TAIL = 0.975` | Null bands of roughly **±0.017** at every cell |
| 31 | `per_d_verdict` — clearance at every `S` | Requires the gap to clear at **all** of `S ∈ {10, 20, 50}`, so no stratum count can be retuned after the fact | d=20 **CLEARS AT EVERY S** (gaps -0.0494 / -0.0416 / -0.0369); d=25 **CLEARS AT EVERY S** (-0.0820 / -0.0798 / -0.0791); d=32 **DOES NOT CLEAR** (-0.0092 / -0.0065 / **+0.0019**) |
| 32 | `combine_seed_verdicts`, no pooling | Per-seed verdicts combined by a frozen unanimity rule; `--mode pool` raises, carrying Phase 5's one-way "do not pool" ratification | **`CLEARS IN ALL THREE SEEDS`** at d=25 |
| 33 | Negative control | 10 repeats × 6 (S, kernel) cells with labels carrying no real structure, measuring the realized false-positive rate against nominal 0.05 | **1 clearance in 60 cells = 0.017.** Five cells at 0.00, one (S=10, rbf) at 0.10. The test runs conservative, not broken |
| 34 | Positive control and detection floor | `PLANTED_EFFECT_GRID = (0.0, 0.05, 0.10, 0.20, 0.50)`, meant to establish the smallest injected effect the CKA test can detect | ❌ **INVALID.** The `magnitude = 0.0` anchor injects nothing, so it re-measures the live d=25 signal: its gap is **-0.082029**, bit-for-bit the real d=25 S=10 sweep gap. It clears at all six cells, `detection_floor = 0.0` is an artifact, and **Phase 8 has no power estimate** |
| 35 | Import-purity regression test | Imports the modules in four subprocess-isolated orders to prove no cross-module constant leaks between phases | D8-23 satisfied; no leakage under any order |
| 36 | Cost-aware pre-registration amendment | A measured re-freeze after the original constants priced out at ~276h, recorded as its own document with the developer's verbatim decision and a no-number-exists integrity check | **~276h → ~28.25h.** `N_PERMUTATIONS` 1000→500, `N_REPEATS` 30→10, grid 7→5 rungs, plus a value-preserving `np.trace(Kt @ Lt)` → `np.sum(Kt * Lt.T)` fix (5.9e-6 relative diff, 2.376× faster). `S_GRID` and the other 42 constants untouched |

### Post-freeze diagnostics (plan 08-07, every row `gates_nothing: true`)

| # | Concept | What it is | Result it produced |
|---|---|---|---|
| 37 | Graph-geodesic density control | Tests whether ambient density is itself curvature-contaminated, since a curved manifold's chord runs shorter than its geodesic. Compares ambient radius against a Dijkstra geodesic radius on the symmetric k-NN graph at Phase 2's frozen `k* = 15` | **Hypothesis rejected.** `rho(geo/amb ratio, ‖H‖)` = **+0.0234 / +0.0243 / +0.0125**, indistinguishable from zero. The geodesic partial is *larger* in magnitude (-0.0448 / -0.0798 / -0.0202), the direction that would flatter the result. Graph connected, `dropped_fraction = 0.000000`. Keep ambient |
| 38 | Rank-invariance of the density exponent | Shows `DENSITY_FIELD_D = 20` and the gamma-function ball volume cannot affect any rank statistic, since density is a strictly decreasing function of radius at fixed `d` | `spearman(density_ambient, -r_ambient) = **0.9999999999999999**`. The volume constant cancels |
| 39 | Exact permutation p-values | Recomputes p by permutation because the record stores thresholds, not null draws, and `spearmanr`'s asymptotic p is invalid under 15-way ties. Plain null at N=100,000 for raw rho; within-stratum null at N=20,000 for the partial | Raw: **< 1e-5 / < 1e-5 / 0.0090**. Partial by stratum count: d=20 **0.057 / 0.070 / 0.069**, d=25 **< 5e-5** at every `S`, d=32 **0.099 / 0.171 / 0.168**. Independently reproduces 07.1's `SURVIVES AT SUBSET OF d` by a different route |
| 40 | Self-validation before emitting | Recomputes a known quantity first and refuses to emit new numbers unless it matches | Plain arm matched the frozen `observed_rho` to 9 decimals at all three `d`; stratified arm landed within 3.1% and 9.4% of the sealed band, inside the 25% tolerance |
| 41 | The d=32 density coupling | Asks whether there is any confound at d=32 to control for | `rho(density, ‖H‖) = +0.011798`, **p = 0.121, not significant**. At d=32 the partial barely moves because there is no coupling, so d=32's null is about the effect, not about the control |

---

## 4. One correction the inventory turned up

`var_explained` (concept 2) is computed as `1 - mse_total / mean(‖x‖²)`
(`07_pu_latent_recon_sweep_run.py:86`, `07_crossmodal_curvature_run.py:180`). The denominator is
the mean squared **norm**, not the variance about the mean, and `subsample.l2_normalize` puts
every row on the unit sphere, so it equals exactly 1.0.

Measured on `subsample_20260729_a79b3460b838fd0a.npz`: **`‖x̄‖² = 0.8130493057493673`**. An
autoencoder emitting one fixed vector for all 10,000 rows scores 81.3% under this formula. Against
a centered denominator the reported 98.217% at d=20 becomes **90.465%**.

Full write-up, including the centered table and why the unit sphere must not be re-normalized:
`.planning/phases/08-curvature-conditioned-cka-alignment/08-NOTE-uncentered-variance-explained.md`.

---

## 5. Verification ledger

**Checked against the frozen records this session:**

- Phase 7's `observed_rho`, both tail thresholds, `var_explained`, the density partials and all
  four positive-control rows, to full float precision.
- Phase 7's ancestry: `git merge-base --is-ancestor f032745 a453736` exits 0, and
  `git rev-list --count f032745..a453736` returns **10**.
- 07.1's nine `null_grid` rows, both verdict rows, all three seed rows and the
  positive-control summary.
- Phase 8's 211 rows, the freeze SHA on every row, all four `per_d_verdict` rows and the
  `seed_combined_verdict`.
- Concept 34's invalidity, by matching the `magnitude = 0.0` gap against the real sweep gap.
- Concept 13's fidelity range, against the four rows in `07_plain_decoder_sweep.jsonl`.
- Concept 4's uncentered denominator, by computing `‖x̄‖²` on the frozen subsample.

**Not checked:** the notebooks' rendered outputs, the 761-test suite (not re-run), and anything in
`src/effdim/`, which the milestone forbids touching.

---

## 6. Open holes

1. **08-DIAGNOSTICS §2 (radial curvature decomposition) and §3 (per-`d` instrument fidelity) hold
   placeholders.** Both runs are live as of 2026-09-01 15:06 — `08_radial_curvature_decomposition_run.py`
   and `07_instrument_fixture_sweep_run.py --d 25` started at 14:03, with `--d 32` queued behind.
   Neither has emitted a number; `08_radial_curvature_decomposition.jsonl` does not exist and
   `07_plain_decoder_sweep.jsonl` still holds only the four sealed d=20 rows.
2. **§2's arithmetic is the largest standing threat to Phases 7 and 8.** `l2_normalize` puts every
   row on the unit sphere, so `H = tr_g(II)` carries a radial term of magnitude exactly `d` that
   says nothing about the manifold's own shape. Strip it in quadrature and the `‖H‖` medians
   37.19 / 41.41 / 47.03 become 31.36 / 33.02 / 34.46, a spread of 10% where the raw field's is
   26%. That would explain both PU's 1.5 p95/p05 spread and Phase 8's 1.164 contrast.
   `spearman(‖H‖, ‖H_tan‖)` is the decision-relevant number and remains unmeasured. It is the same
   geometry as §4's mean-dominance.
3. **No human has ratified anything from 07.1 onward.** 07.1's three blocking checkpoints and
   08-07's five decisions (a) through (e) were all advanced under a standing "keep working"
   instruction. 07.1-UAT tests 1 and 2 sit `pending`. Plan 08-06 never ran, so no `08-FINDINGS.md`
   exists.
4. **Phase 8 has no power estimate** (concept 34), which leaves d=32's `DOES NOT CLEAR`
   uninterpretable — indistinguishable from an underpowered test.
5. **Instrument fidelity is d=20-only.** Every d=25 and d=32 number leans on a range measured at a
   dimension those fits do not use. Concept 13's own finding, that reconstruction quality does not
   predict curvature fidelity, is what makes the gap matter.

---

## 7. The one-sentence read

**d=25 is the only cell that survives every control across all three phases** — the raw rho, its
own stratified null, an exact permutation p below 5e-5, three independent seeds, and CKA clearance
at every stratum count. d=20 clears CKA but dies under density control at p ≈ 0.06. d=32 clears
nothing that matters.

---

*Compiled 2026-09-01 from the frozen records. Ratified by nobody.*
