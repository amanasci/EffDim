# Phase 4 Findings — Region Partitioning & Regional Alignment (MKNN)

**Date:** 2026-08-24. **Milestone:** v1.1 PU Manifold Curvature. **Phase:**
04-region-partitioning-regional-alignment-mknn.

**One-line outcome.** `VERDICT_RULE` HOLDS at every `k` in `{5, 10, 20, 50}`, including the
pre-registered headline `k=20`, applied mechanically exactly as pre-registered — and every number
behind that verdict rests on a field accepted on reliability alone, a partition validated only at
codimension 1 against a codimension-748 problem, and a density confound that is reported, not
controlled. A fourth, mechanical qualification — the raw-score gap the rule fires on is mostly a
region-size artifact — is stated in full in Section 5. Phase 4 produces its result with no
known-answer anchor at any point in the chain: not the estimator, not the field, not the partition.

---

## 1. What this phase claims and what it deliberately does not

This section comes first for the same reason `03-FINDINGS.md` Section 1 does: a reader must not
reach the result before the conditions on it.

**Claim.** Under a verdict rule pre-registered before any regional MKNN number existed
(`04-PREREGISTRATION.md`, committed at `0305c77`, before commit `647d01d` computed the first
regional cell), region 1 of the PU curvature field's diametrical sign split scores higher than
region 0 on crossmodal HSC-vs-Legacy-Survey MKNN at every `k` in the pre-registered grid, the two
regions' 95% bootstrap CIs are disjoint at every `k`, and region 1 clears its own region-scoped
99th-percentile permutation null at every `k`. `VERDICT_RULE` therefore reads HOLDS, headline
included, and it was applied mechanically with no constant amended after any regional number
existed.

**What this phase deliberately does not establish, in its own words:**

**Gap 1 — the field is accepted on split-half reliability alone, and split-half reliability cannot
detect a bias both halves share.** The curvature field this phase partitions
(`centroid_mean_curvature`, density-corrected, `k=K_FROZEN=500`) is validated against closed-form
synthetic fixtures where the answer is known — planes, spheres, the Swiss roll — and, on PU itself,
checked only by comparing two disjoint halves of the same cloud against each other. Both halves use
the same estimator, the same `k`, and the same architecture, so a systematic bias shared by both
halves is perfectly reliable by this statistic and completely invisible to it. This is not a
hypothetical concern: it was measured directly on the Swiss roll, where the true answer is known —
`R_H = 0.990` (near-perfect two-half agreement) alongside `rho = 0.469` against the true curvature
(mediocre accuracy). Reliability and correctness came apart there by construction, on the one
fixture where both could be checked at once. There is no ground truth for PU's curvature field, so
this gap cannot be closed by more of the same measurement — running the split-half sweep further,
or at more seeds, only tightens agreement between two halves that could both be wrong together in
exactly the same way. The mitigation that would convert this blind spot into a measured number —
running the CAE chart decoder's `H` alongside `centroid_mean_curvature` and reporting their rank
agreement, one cell of compute — was recommended at the Phase 3 close and declined twice (D4-03,
reaffirmed by D4-08). It remains available and was not run in this phase.

**Gap 2 — the direction partition rests on codimension-1 evidence applied to a codimension-748
problem, and no fixture validation was run.** The partition (Section 3) splits PU's curvature field
by the sign of each point's projection onto the leading eigenvector of the unit-`H` covariance
(D4-09), a decision resting on spike 003's measurement that curvature DIRECTION survives at `d=20`
(cosine 0.77–1.000) while magnitude is roughly 50x attenuated and its rank ordering saturates
around `rho=0.5–0.65`. The limit on that evidence, stated plainly: every fixture spike 003 measured
this on — `saddle`, `bowl`, `aniso`, `cubic`, `sine`, `ridge` — is a codimension-1 graph, where
`H = H_scalar * n_hat`, so the curvature "direction" being measured there *is* the surface normal.
A cosine near 1.000 on those fixtures demonstrates recovery of normal ORIENTATION, a tangent-space
problem known to converge well — it does not demonstrate resolution of `H`'s direction *within* a
high-dimensional normal space. PU sits at `d~20` inside `D=768`, a codimension of roughly 748, which
is exactly the regime the codimension-1 fixtures cannot speak to. No known-answer fixture validation
— neither `make_ridge_graph_control` nor `make_multinormal_ridge_control` — was run before the PU
split was frozen (D4-10), which explicitly overrides D4-01's own body text naming the ridge check a
Phase 4 precondition. The developer's stated reason: `make_multinormal_ridge_control` tops out
around `m=8` normal dimensions, so running it would narrow codimension 1 to codimension 8 against
PU's ~748 — closer to the true regime, but nowhere near it — and risks being read as closing the gap
rather than merely narrowing it. That gap is unmeasured on PU and unclosed by anything in this
milestone; the sign split partitions whatever direction structure the field happens to carry,
without itself validating that structure at PU's codimension.

**Gap 3 — the density confound is reported, not controlled, and a regional MKNN difference cannot
be separated from a regional density difference by anything in this phase.** Under D4-14 the
density-confound battery run in this phase is the REGN-02 correlation and the region-level
comparison that follows it — no partial regression, no density-matched null, no centroid-distance
control, no density-matched stratification. The measured correlations are stark:
`spearman(density, signed_projection onto v) = +0.8208` (n=9500, p≈0) against
`spearman(density, ||H||) = -0.0273` (n=9500, p=0.0078) — essentially nil. The pre-registered split
axis is very nearly a density axis, and the confound is specific to the DIRECTION the partition
uses, not to curvature magnitude. Region density medians differ by roughly 5,735x
(region 0: `3.7642e10`, region 1: `6.5641e6`; Mann-Whitney U=18844954.0, p=0). MKNN is itself a
k-NN statistic and is therefore directly density-sensitive by construction. Without a
density-matched null, a regional MKNN difference cannot be separated from a regional density
difference by anything in this phase — the two correlations above are the only evidence bearing on
it, and at `rho=+0.82` the density confound is the dominant explanation available on the table, not
a caveat to bear in mind.

**Taken together, these three gaps mean Phase 4 produces its result with no known-answer anchor at
any point in the chain — estimator, field, or partition.** That is a deliberate and consistently
made developer choice, not an oversight: at every decision point (D4-03/D4-08 on the field, D4-10
on the partition, D4-14 on the density battery) a cheaper validating check was identified and
explicitly declined, for reasons recorded at the time rather than discovered after the fact. This
document presents it as a choice, made plainly, rather than a gap for a future reader to discover.

**A fourth qualification, mechanical rather than a validation gap, is stated in full in Section 5:
the region-size imbalance between region 0 (n=6256) and region 1 (n=3244) inflates region 1's raw
MKNN score relative to region 0's by mechanical construction (MKNN's chance floor is `k/n_region`),
and this accounts for nearly the entire raw-score gap `VERDICT_RULE` fires on.** This is not grounds
to amend the verdict — the rule was pre-registered on raw scores and applied mechanically, and it
HOLDS — but the record would be dishonest if it let a reader believe the raw-score gap were
evidence of the confound's magnitude rather than substantially an artifact of region size.

**This document does not reopen, soften, recompute or reinterpret any sealed verdict from Phases 2,
02.x, 3 or 03.1.** Where the decoder-versus-cloud head-to-head is referenced below, its own caveat
travels with it (Section 7).

---

## 2. The frozen configuration

**`K_FROZEN = 500`.** D4-07's freeze rule — "freeze at the smallest `k` where median `R_H` gains
less than +0.03 over the previous sweep point AND median `R_H >= 0.5`" — **did not fire anywhere in
the six-point sweep grid**. `k_frozen = 500` is the pre-registered fallback ("the largest `k`
actually run"), not a detected plateau, and this document never describes it as converged,
plateaued, or settled.

| `k` | `median_R_H` | delta vs prev | `median r_dir` | `r_knn` | `R_cloud` | `r/R` |
|---|---|---|---|---|---|---|
| 30 | 0.0279 | — | 0.0337 | 0.264 | 0.3774 | 0.699 |
| 60 | 0.0927 | +0.0648 | 0.1053 | 0.279 | 0.3774 | 0.740 |
| 120 | 0.1573 | +0.0646 | 0.1746 | 0.297 | 0.3774 | 0.788 |
| 231 | 0.2337 | +0.0763 | 0.2598 | 0.317 | 0.3774 | 0.839 |
| 350 | 0.2853 | +0.0516 | 0.3141 | 0.331 | 0.3774 | 0.878 |
| 500 | **0.3436** | +0.0583 | 0.3741 | 0.345 | 0.3774 | **0.915** |

`median_R_H` reached only `0.3436` against the rule's `0.5` floor, and the per-step increment never
collapsed toward the `0.03` ceiling — it **rose** at the last step (`0.0516 → 0.0583`) rather than
settling. Neither threshold was moved to make the rule fire, and neither was moved to make it fire
"more honestly" — both remain exactly as declared before the sweep ran. `r/R` is reported for
context and is never used as a gate (Spearman rank correlation is scale-free with respect to
locality regime, per spike 003).

Configuration: `d=20` (`FIELD_D`, explicit call-site value, never re-derived per D-07), `k_density=30`
(`K_DENSITY`), `density_correct=True`, `SEED=20260822`, resolved subsample file
`subsample_20260729_a79b3460b838fd0a.npz`.

**The four earlier uncorrected `R_H` rows (`k=30/60/120/231`, `median_R_H=0.0779/0.2474/0.428/0.5894`,
in `03.2_pu_curvature_rankability.jsonl`) are superseded, not extended** — they remain on disk
byte-for-byte unmodified, and the corrected sweep re-ran from `k=30` upward rather than continuing
past `231`. The density correction is adopted on its measured ~8–10% median-relative-error
reduction on a genuinely curved, strongly-skewed fixture (`02.5-02-SUMMARY.md`'s amended finding) —
**not** on the retracted flat-fixture inertness claim (that correction is provably inert on a flat
fixture, which says nothing about a curved one).

---

## 3. The partition

**Exclusion.** `MIN_NORM_PERCENTILE=5.0` (5th percentile of the field's own `||H||`, inclusive
boundary) excluded `500` points (5.0% of 10,000).

**Region counts.** `region_0 = 6256` (62.6%), `region_1 = 3244` (32.4%), `excluded = 500` (5.0%),
`n_zero_projection = 0` — sums exactly to 10,000. Both regions clear `MIN_REGION_N=500`; no cell is
undefined on size grounds for this split.

**`mean_unit_norm = 0.294748`.** The mean-centered and uncentered second-moment covariance forms
coincide only when `||mean(unit)||` is near zero; at this magnitude they do **not** coincide, so
`COVARIANCE_FORM="mean_centered"` (04-03's flagged question) is a live choice that materially
affected the split axis `v`, not a settled formality.

**Unit-`H` covariance eigenvalue spectrum (top 5):** `0.031611, 0.020180, 0.018141, 0.016507,
0.015396`. Top/second ratio `1.566` — `v` is reported as the chosen split axis and is explicitly
**not** presented as a well-separated principal axis.

**PU's measured `||H||` dynamic range.** At the frozen, density-corrected field (`k=500`), the
spread is `p95/p05 = 3.94x`. The original uncorrected sweep (D4-05) measured a flat `~4.8x` across
`k=30..231` (`5.54/4.83/4.79/4.86`). The runner's own calibration puts the unrankable
`quadratic_bowl` reference at `1.4x` (`rho +0.03`) and the rankable `cubic`/`ridge` references at
`28.2x`/`34.3x` (`rho +0.61`/`+0.41`). **PU sits far nearer the unrankable end on this axis at every
point in the sweep.** This gates nothing: direction is a unit vector and does not consume the
magnitude spread, and dynamic range does not predict rankability in any case — spike 003 measured
`rho = +0.48` at a spread of `1.1x` and `rho = +0.36` at `36x` on the same fixture family, so a
narrow or wide spread predicts nothing about rankability by itself.

---

## 4. Density

`spearman(density, ||H||) = -0.0273` (n=9500, p=0.0078) — essentially nil.
`spearman(density, signed_projection onto v) = +0.8208` (n=9500, p≈0) — very close to a density
axis.

Region-level: region 0 median density `3.7642e10` (IQR `4.7311e11`, n=6256) against region 1 median
density `6.5641e6` (IQR `8.6342e7`, n=3244) — a two-sided Mann-Whitney U test gives
`statistic=18844954.0, p=0`. The medians differ by roughly **5,735x**.

**Reported, not controlled** (D4-14): no partial regression, no density-matched null, no
centroid-distance control, no density-matched stratification were run. This is the entire evidence
base bearing on the confound anywhere in this phase.

---

## 5. MKNN

**Global reproduction, `n=10,000`.**

| `k` | raw score | `k/n` chance floor | ratio over chance | 95% CI | hub (hsc / ls) |
|---|---|---|---|---|---|
| 5 | 4.882% | 0.05% | 97.6x | [4.688%, 5.086%] | 1.239 / 1.494 |
| 10 | 6.594% | 0.10% | 65.9x | [6.417%, 6.781%] | 1.118 / 1.319 |
| 20 | 8.9805% | 0.20% | 44.9x | [8.802%, 9.165%] | 1.049 / 1.188 |
| 50 | 13.229% | 0.50% | 26.5x | [13.033%, 13.427%] | 0.966 / 1.013 |

All four raw scores fall **outside** the origin paper's published Legacy-vs-HSC range
(0.34%–2.25%, arXiv:2509.19453 Table 2) — but that paper's number is measured at `n=101,725`
against this phase's `n=10,000` (D4-19), an order-of-magnitude difference in `n` that the raw
percentage comparison hides. The `k/n` chance floor scales inversely with `n`; at the smaller `n`
used here the floor is roughly 10x higher, so a higher raw percentage is exactly what a smaller `n`
predicts even under identical underlying alignment. The comparison that carries meaning is
ratio-over-chance, not the raw band: every `k` here clears its own chance floor by 26x–98x, a
strong and monotonically narrowing signal as `k` grows, reported explicitly alongside the `n`
mismatch rather than papered over as either agreement or disagreement with the published range.

**The eight-cell regional grid** (from `notebooks/.cache/04_region_partition_mknn.jsonl`, verified
directly against `chance_floor == k_mknn / n_region` and `ratio_over_chance == score / chance_floor`
to `1e-9`):

```
  k reg     n     score      95% CI            null_thr        p     ratio/chance
  5   0  6256   0.04162   [0.03942, 0.04393]   0.00137   0.000999      52.1
  5   1  3244   0.09889   [0.09383, 0.10382]   0.00271   0.000999      64.2
 10   0  6256   0.05850   [0.05655, 0.06052]   0.00229   0.000999      36.6
 10   1  3244   0.13212   [0.12762, 0.13647]   0.00447   0.000999      42.9
 20   0  6256   0.08148   [0.07946, 0.08346]   0.00425   0.000999      25.5
 20   1  3244   0.17408   [0.16977, 0.17842]   0.00832   0.000999      28.2
 50   0  6256   0.12489   [0.12282, 0.12697]   0.01011   0.000999      15.6
 50   1  3244   0.24258   [0.23817, 0.24729]   0.01976   0.000999      15.7
```

**`VERDICT_RULE`, verbatim** (frozen in `region_partition.py` before any regional number existed):

> The high-vs-low regional MKNN result HOLDS at a given `k` if and only if BOTH: (a) the two
> regions' 95% percentile bootstrap CIs at that `k` are disjoint, AND (b) the higher-scoring
> region's observed MKNN strictly exceeds the 99th percentile of its OWN region-scoped permutation
> null. The headline call is made at `HEADLINE_K=20` alone; `k` in `{5, 10, 50}` are sensitivity
> only and take no separate multiplicity correction. "NO DETECTABLE DIFFERENCE" at the headline `k`
> is a complete, valid outcome, never a phase failure and never escalated by majority vote.

**HEADLINE VERDICT (k=20): HOLDS.** Region 1 scores higher than region 0 at every `k` in the grid,
the two regions' 95% CIs are disjoint at every `k`, and region 1's observed score strictly exceeds
its own 99th-percentile permutation-null threshold at every `k` (`p_value=0.000999` — the tightest
resolvable value at 1000 permutations — at every one of the eight cells). The rule was applied
mechanically; every value in `VERDICT_RULE` was read from `region_partition.py` as frozen at commit
`0305c77`, before any regional number existed, and no constant was amended in light of the result.

**The region-size artifact — a mandatory qualification independently verified from the JSONL.**
MKNN's chance floor is `k / n_region`. Region 1 (`n=3244`) is roughly half the size of region 0
(`n=6256`), so its chance floor is roughly twice as high at every `k`, and its raw MKNN score is
mechanically inflated relative to region 0's for that reason alone — before any question of
curvature or density enters. The consequence is visible directly by comparing the raw-score column
to the ratio-over-chance column:

| `k` | raw score gap (region 1 vs 0) | ratio-over-chance gap |
|---|---|---|
| 5 | 0.09889 vs 0.04162 (+137.6%) | 64.2 vs 52.1 (+23.2%) |
| 10 | 0.13212 vs 0.05850 (+125.9%) | 42.9 vs 36.6 (+17.2%) |
| 20 | 0.17408 vs 0.08148 (+113.6%) | 28.2 vs 25.5 (+10.6%) |
| 50 | 0.24258 vs 0.12489 (+94.3%) | 15.7 vs 15.6 (+0.6%) |

At the headline `k=20`, raw scores differ by ~114% but ratios over chance differ by only ~11%. At
`k=50` the raw gap is ~94% while the ratio-over-chance gap is effectively zero (15.7 vs 15.6). **So
almost the entire raw-score gap that `VERDICT_RULE` fires on is accounted for by the region-size
imbalance rather than by any difference in crossmodal alignment measurable by this phase.**

**This is not grounds to amend the verdict.** `VERDICT_RULE` was pre-registered on raw scores
before any regional number existed, and it was applied mechanically exactly as written — HOLDS is
the correct application of the rule that was frozen. Re-specifying the rule now, in ratio-over-chance
terms, after seeing that the raw-score gap and the ratio-over-chance gap tell different stories, is
exactly what the pre-registration freeze exists to forbid: it is post-hoc rule selection dressed as
a correction. The verdict stands as reported, mechanically, alongside this qualification — not
restated in ratio-over-chance terms as though that had been the rule all along, and not softened
into "no real difference" either, since a genuine (if much smaller) ratio-over-chance gap remains at
every `k` except `k=50`.

**Recommendation for any future phase re-running this comparison:** pre-register a size-matched or
chance-floor-normalized statistic (e.g. score minus chance floor, or score divided by chance floor,
as the primary quantity `VERDICT_RULE` operates on), or subsample the larger region to the smaller
region's `n` before scoring, so the statistic is not confounded with region size by construction.
Neither was pre-registered for this phase, and amending `VERDICT_RULE` retroactively to use one
would be exactly the after-the-fact rule change the pre-registration discipline forbids.

**Hubness (MKNN-08).** k-occurrence skewness ranges `0.966` to `1.494` across both embedding sides
and every `k` measured in this phase (global and regional cells combined) — printed beside every
reported score, never asserted as prose alone.

---

## 6. Requirement outcomes

| Requirement | Outcome | Evidence |
|---|---|---|
| REGN-01 | Complete — ambient 768-d local density shown for all 10,000 points (`rho_p05=6.07e4, rho_p50=2.29e9, rho_p95=3.63e12`, spread `p95/p05=5.98e7`) before the split is trusted | Section 4; `04_density_diagnostics.json`; notebook Section 4 |
| REGN-02 | Complete, reported-not-controlled — both Spearman correlations measured and reported; **the density confound is disclosed, not resolved** (D4-14) — see Gap 3, Section 1 | Section 4; `04_density_diagnostics.json` |
| REGN-03 | Complete — data-derived direction criterion (diametrical sign split on `Cov(H_i/‖H_i‖)`'s leading eigenvector), never a fixed absolute threshold; **validated only against a codimension-1 known answer, not at PU's codimension-748** (Gap 2, Section 1) | Section 3; `region_partition.py`; `04-03-SUMMARY.md`'s ARI=1.0 known-answer test |
| REGN-04 | Complete — the partition rule and every free parameter frozen in `04-PREREGISTRATION.md` (committed `0305c77`) before any regional MKNN number existed, ordering visible in notebook Section 6 | `04-PREREGISTRATION.md`; notebook Section 6 |
| REGN-05 | Complete — both region counts, excluded count and zero-projection count shown, sum verified to 10,000 | Section 3; notebook Section 5 |
| REGN-06 | Complete — `v`, `labels`, `keep_idx`, `excluded_idx`, `h_norm`, `signed_projection`, `eigval_spectrum` frozen via `cache.npz_cache` at stem `04_region_partition` before any MKNN number was computed | `notebooks/.cache/04_region_partition.npz`; `04-04-SUMMARY.md` |
| MKNN-01 | Complete — `mknn_score` matches the origin paper's formula, known-answer checks pass | Notebook Section 1; `mknn.py` |
| MKNN-02 | Complete, with an explicit `n` mismatch — global reproduction reported against the paper's range with the `k/n` chance floor and the 10,000-vs-101,725 mismatch stated explicitly, never papered over | Section 5; notebook Section 2 |
| MKNN-03 | Complete — per-region MKNN score shown for both regions at every `k` | Section 5; `04_region_partition_mknn.jsonl` |
| MKNN-04 | Complete — each region's permutation null computed within that region's own index set (`null_scope="region"`, `null_n=n_region` on every row), no global null reused | Section 5; `04_region_partition_mknn.jsonl` |
| MKNN-05 | Complete — 95% bootstrap CI shown on every regional score | Section 5 |
| MKNN-06 | Complete — the eight-cell grid across `k={5,10,20,50}` × 2 regions rendered as one table with headline `k` marked | Section 5; notebook Section 7 |
| MKNN-07 | Complete — explicit verdict rule pre-registered and applied mechanically; result is HOLDS at every `k`, **qualified by the region-size artifact (a mandatory finding of this closure plan, not evidence of a rule failure) and by the unresolved density confound (Gap 3)** | Section 5; notebook Section 8; `04-PREREGISTRATION.md`'s `VERDICT_RULE` |
| MKNN-08 | Complete — hubness caveat substantiated by printed k-occurrence skewness (0.966–1.494) beside every result, both global and regional | Section 5; notebook Sections 2, 7 |

No cell in this phase's eight-cell regional grid is `status: "undefined"` — both regions cleared
`MIN_REGION_N=500` at every `k`, and D4-07's freeze rule not firing did not leave any downstream
quantity undefined (`K_FROZEN` resolved to its pre-registered fallback value).

---

## 7. What a follow-on phase would need

Four mitigations were identified and deliberately declined in this phase, quoted from
`04-CONTEXT.md`'s Deferred Ideas rather than re-argued here:

1. **Cross-estimator agreement on PU** (D4-02 Amendment 02's cheap mitigation, declined again as
   D4-08). Run the CAE chart decoder's `H` field alongside `centroid_mean_curvature` at the frozen
   `k` and report their rank agreement. **Buys:** converts Gap 1 (D4-03's accepted blind spot) from
   an accepted risk into a measured number. **Costs:** one notebook cell; the decoder arm is known
   to be undertrained relative to Phase 3's sealed fits (see caveat below), so a low agreement
   number would need that caveat carried with it.
2. **Codimension-gap narrowing** via `make_multinormal_ridge_control` at `m=4, 8` (declined as
   D4-10). **Buys:** narrows the direction-partition validation from codimension 1 to codimension 8
   against PU's ~748. **Costs:** does not close the gap — `m=8` is still two orders of magnitude
   short of PU's actual codimension — and risks being read as having closed it if not stated
   carefully.
3. **Density-matched null / partial regression / centroid-distance checks** (declined as D4-14).
   **Buys:** these are the controls that would let a regional MKNN difference be attributed to
   curvature rather than to density — the single largest open question this phase leaves. **Costs:**
   nontrivial design work (what "density-matched" means for a permutation null needs its own
   pre-registration) plus additional compute.
4. **`tests/test_mknn.py`** with exact known answers — identical embeddings → 1.0; independent
   embeddings → `k/n`; a hand-computed `n=6, k=2` case; seed reproducibility (declined as D4-18).
   **Buys:** restores the package's one-test-file-per-module convention; `mknn.py` is currently the
   only module without one. **Costs:** near-zero; this is the cheapest of the four declined items
   and the only one with no scientific-judgment cost attached, only implementation time.

A fifth item, raised by this closure plan itself (Section 5): **a size-matched or
chance-floor-normalized regional MKNN statistic**, or subsampling the larger region to the smaller
region's `n` before scoring, so that a future regional comparison is not confounded with region size
by construction. This was not on `04-CONTEXT.md`'s Deferred list because the region-size artifact
was not identified until this closure plan's mandatory verification.

**The decoder-versus-cloud head-to-head, its own caveat carried alongside it.** Where referenced
above (item 1), recall the caveat from `03-NOTE-phase-4-decisions.md` Amendment 02: the decoder arm
was undertrained relative to Phase 3's sealed fits (`mse_per_dim` roughly 15x worse than the sealed
`d=20` cell), run at reduced `D`, `n` and epochs for affordability. **This is not a clean
disqualification of a well-trained decoder**, and the two arms' numbers were not measured on the
same fixture as the sealed saddle control — they must not be quoted as a comparison on that fixture.

---

*Phase: 04-region-partitioning-regional-alignment-mknn*
*Completed: 2026-08-24*
