# Phase 5 Pre-Registration — every free parameter frozen before a PU probe number exists

**Date:** 2026-08-24
**Git HEAD at time of writing:** `32dabe3` (plan `05-04` Task 2's freeze commit —
`notebooks/pu_manifold/linear_probe.py`'s 31-constant block filled, the freeze tripwire on
`test_curvature_convention_matches_sealed_modules` removed)
**Committed by:** plan `05-04`, Task 3, under D5-09, after Task 2's freeze commit and before
any PU probe number exists. No `notebooks/.cache/05_curvature_probe_decodability.jsonl` exists
anywhere in this repository as of this document's commit.

## Checkpoint ratification notes

Two blocking checkpoints gate this document. Both are recorded here plainly.

### `05-03` Task 1 — the one-way seed-handling decision

`05-CONTEXT.md` D5-04 called for pooling the three cached CAE seeds into one averaged `||H||`
field and naming that field the verdict field. That question was put to the developer at the
`05-03` Task 1 blocking checkpoint, with `05-02`'s measured inter-seed numbers already on the
table. **The ratified outcome was the opposite of D5-04: do not pool.** Run the probe once per
seed and report three per-seed verdicts and their spread. No pooled field, no pooled bucket
edges. Rejected on the record, alongside the raw average: per-seed median-divide then average
(`05-RESEARCH.md`'s own recommendation), per-seed percentile-rank then average, and halting the
phase. **`05-03-DECISION.md` is the authority wherever it and `05-CONTEXT.md` D5-04 conflict.**
D5-04 is SUPERSEDED.

Evidence the decision was made on, measured at `05-02` over all 10,000 PU points via
`chart_curvature.chart_curvature_field(model, x64, mode='reverse')` on the three sealed CAE
checkpoints `03_converged_cae_pu_nc4_seed2026081{3,4,5}`:

Pairwise Spearman on `H_norm` — sign-inconsistent, two of three negative:

| pair | rho | p |
|------|-----|---|
| 20260813 vs 20260814 | **-0.1402** | 4.8e-45 |
| 20260813 vs 20260815 | **+0.2019** | 1.8e-92 |
| 20260814 vs 20260815 | **-0.2725** | 8.9e-170 |

Direction axis, reported beside every rank statistic per the spike-findings requirement:

| pair | median cosine | fraction anti-aligned |
|------|---------------|------------------------|
| 20260813 vs 20260814 | 0.0039 | 46.1% |
| 20260813 vs 20260815 | 0.0014 | 48.1% |
| 20260814 vs 20260815 | 0.0007 | 46.4% |

Per-seed structure: seed 20260813 has 2 charts used, median `log10 det g` -68.5, continuous
field; seeds 20260814 and 20260815 have 4 and 3 charts used respectively, median `log10 det g`
around -165.6 / -165.7 — a metric determinant roughly 100 orders of magnitude from seed
20260813's. Any pooled field would not be a consensus: it would be seed 20260813's structure
plus two step-like functions that disagree with it and with each other.

**Correction carried here, not silently smoothed over.** `05-02-SUMMARY.md` states that seeds
20260814 and 20260815 have 5,301 and 9,852 exact distinct `H_norm` values and are therefore
"not literally piecewise-constant". **That claim is wrong** — those counts are float noise in
the last ULPs, an artifact of `np.round(H_norm, 6)`'s absolute rounding at magnitude ~5e4
(relative precision ~2e-11). Measured directly from the cached fields at RELATIVE precision,
stable from rel `1e-9` through rel `1e-3`, seed 20260814 has **4 effective levels** and seed
20260815 has **3 effective levels**. `05-RESEARCH.md` Pitfall 2 and `03-09-SUMMARY.md`'s
original "3-4 distinct values" measurement were both correct; `05-02-SUMMARY.md`'s 5,301 /
9,852 figures were last-ULP float noise, not effective structure, and must not travel further.

### `05-04` Task 1 — the full pre-registration ratification

**Selected: `ratify-recommended` — the planner's full proposal, as written in `05-04-PLAN.md`,
with no amendments.** Every value below is exactly the value the plan proposed before this
checkpoint existed; none was invented, relaxed, or tuned after the fact. The alternatives
`ratify-with-amendments` and `defer-verdict-rule` were both explicitly declined — the latter
because deferring the verdict rule, or only the seed-combination half of it, is precisely the
D5-09 violation that halted this phase at wave 3. The human was shown, and accepted, the
accepted-limitations list reproduced in full below.

## What this document forecloses

Once this document is committed, the following may not change for the remainder of Phase 5:

- Re-choosing `N_BUCKETS` (or `BUCKET_EDGES_PER_SEED`) after seeing the residuals.
- Switching `RESIDUAL_METRIC` from squared L2 to a cosine or normalized residual because the
  first pairing gave a null.
- Pooling the seeds after all, because the per-seed verdicts split.
- Re-defining `SPLIT ACROSS SEEDS` as partial support for the hypothesis once the split outcome
  is known.
- Upgrading a `NO DETECTABLE RELATIONSHIP` verdict — per seed or phase-level — by majority vote
  across seeds, by the continuous Spearman statistic, or by trying a different `N_BUCKETS`.
- Re-cutting any seed's `BUCKET_EDGES_PER_SEED` entry on the test split to balance its buckets.
- Dropping the one seed whose per-seed verdict disagrees with the other two.

Changing any pre-registered constant, `VERDICT_RULE`, or `SEED_VERDICT_COMBINATION_RULE` after
a PU probe number exists invalidates the phase's result and costs a fresh pre-registration plus
a complete re-run, per the `02.2` CAE precedent.

## Pre-registered constants

Every value below is the verbatim value of the identically-named constant in
`notebooks/pu_manifold/linear_probe.py`, as committed at plan `05-04` Task 2's freeze commit
`32dabe3`.

| Constant | Value | Source / rationale |
|---|---|---|
| `TRAIN_FRACTION` | `0.7` | RESEARCH A4's recommended split fraction, ratified |
| `SPLIT_SEED` | `20260824` | One shared split seed across all three seeds' bucketings |
| `SPLIT_RULE` | One permutation of `np.arange(10000)` under `np.random.default_rng(SPLIT_SEED)`; first 7,000 (of the permutation) train, last 3,000 test; both index arrays returned sorted ascending; NOT stratified by bucket | Deliberately not stratified — stratifying would manufacture the equal per-bucket test counts D5-08 exists to check for rather than assume |
| `RIDGE_ALPHA_GRID` | `(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)` | RESEARCH A2's ridge grid |
| `RIDGE_SELECTION_RULE` | scikit-learn `RidgeCV`'s generalized leave-one-out cross-validation on the training split alone, selecting one alpha from `RIDGE_ALPHA_GRID` | The GRID and the RULE are frozen; the selected alpha is an OUTPUT reported at `05-05`, never pre-specified — degrades gracefully to OLS if the design matrix is well-conditioned |
| `ALPHA_PER_TARGET` | `False` | One shared alpha across all 768 outputs |
| `FIT_INTERCEPT` | `True` | Standard ridge convention |
| `EMBEDDING_PREPROCESSING` | `"raw_as_cached..."` (full text in module) | Both modalities are already L2-normalized upstream (every row norm equals 1.0 to float64 rounding); re-normalizing would be a no-op dressed as a decision |
| `RESIDUAL_METRIC` | `"squared_l2_per_point"` | Paired with `R2_MULTIOUTPUT="variance_weighted"`, the one pairing satisfying CONTEXT's "R² and per-point residual derivable from one underlying quantity" constraint exactly |
| `R2_MULTIOUTPUT` | `"variance_weighted"` | See above; `05-01` Task 2 pins the identity numerically |
| `N_BUCKETS` | `3` | Tertiles — at a 70/30 split, three buckets leave roughly 1,000 test points each, supporting a percentile bootstrap without the CI collapsing; quartiles would leave roughly 750 and buy nothing |
| `BUCKET_RULE` | Equal-frequency rank partition of ONE seed's `\|\|H\|\|` field over all 10,000 points by stable argsort and `np.array_split`, applied independently per seed; a value equal to an edge lands in the HIGHER bucket | `assign_buckets`' documented tie rule |
| `BUCKET_EDGES_PER_SEED` | `((1225.4263017421292, 1538.3597929379368), (49062.2351870738, 66977.54374981482), (51694.86079512253, 75252.52609688243))` | Read programmatically, not retyped, from `notebooks/.cache/05_curvature_buckets_seed2026081{3,4,5}.npz`'s `bucket_edges` arrays, in `SEED_STEMS` order — see Provenance below |
| `SEED_HANDLING_RULE` | `"no_pooling_per_seed_verdicts"` | The ratified outcome of the `05-03` Task 1 checkpoint, checked by string equality in `assert_preregistered` |
| `SEED_VERDICT_COMBINATION_RULE` | (verbatim, quoted in full below) | `05-04` Task 1's new free parameter, created by the per-seed redesign |
| `PHASE_VERDICT_VALUES` | `("HOLDS IN ALL THREE SEEDS", "SPLIT ACROSS SEEDS", "NO DETECTABLE RELATIONSHIP IN ANY SEED")` | The three terminal phase-level outcomes |
| `SEED_STEMS` | `(20260813, 20260814, 20260815)` | The three sealed CAE checkpoint seeds |
| `N_CHARTS` | `4` | The CAE's chart count for these sealed fits |
| `CURVATURE_MODE` | `"reverse"` | `chart_curvature_field`'s autodiff mode |
| `CURVATURE_CONVENTION` | `"trace"` | Equals `chart_curvature.CURVATURE_CONVENTION` and `curvature_probe.CURVATURE_CONVENTION` — asserted equal by a passing test (D5-06), no longer an xfail |
| `CURVATURE_SOURCE_FUNCTION` | `"chart_curvature.chart_curvature_field"` | D5-03's corrected citation, frozen as an auditable constant |
| `SIZE_MATCH_RULE` | Per seed, subsample every bucket to the smallest REALIZED TEST-SPLIT bucket count FOR THAT SEED (never the full-field count, never a count borrowed from another seed), re-run the highest-vs-lowest comparison | The exact artifact that undercut Phase 4's HOLDS verdict, built into the protocol from the start |
| `SIZE_MATCH_N_REPEATS` | `200` | Repeat count for the size-matched re-check |
| `SIZE_MATCH_SEED` | `20260824` | Seed for the size-matched re-check's resampling |
| `N_BOOTSTRAP` | `1000` | Percentile bootstrap resample count, matching Phase 4's own value |
| `BOOTSTRAP_SEED` | `20260824` | Bootstrap RNG seed |
| `CONFIDENCE_LEVEL` | `0.95` | Bootstrap CI level, matching Phase 4's own value so interval widths are comparable |
| `K_DENSITY` | `30` | D5-13's density-confound estimator, matching Phase 4's own `K_DENSITY` |
| `FIELD_D` | `20` | Explicit call-site value, never re-derived, per D-07 |
| `VERDICT_RULE` | (verbatim, quoted in full below) | The per-seed HOLDS / NO DETECTABLE RELATIONSHIP rule |
| `PREREGISTRATION_PATH` | `".planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md"` | Points at this document |

**`POOLING_METHOD` and `BUCKET_EDGES` were RETIRED at `05-03`** and are deliberately absent
from the table above — structurally removed from `linear_probe.py`'s constants block, not
merely left unset, so a reader comparing against `05-CONTEXT.md` D5-04 sees a removal rather
than an omission.

## Provenance for the derived values

`BUCKET_EDGES_PER_SEED` is not a chosen number but the mechanical output of the frozen
`BUCKET_RULE` applied independently to each of the three per-seed `||H||` fields at `05-03`.
The three source artifacts, named explicitly, each carrying a cfg manifest naming the seed,
source field stem, bucket rule, subsample file, curvature convention, and
`no_pooling_per_seed_verdicts`:

- `notebooks/.cache/05_curvature_buckets_seed20260813.npz` → `(1225.4263017421292,
  1538.3597929379368)`
- `notebooks/.cache/05_curvature_buckets_seed20260814.npz` → `(49062.2351870738,
  66977.54374981482)`
- `notebooks/.cache/05_curvature_buckets_seed20260815.npz` → `(51694.86079512253,
  75252.52609688243)`

`05-04` Task 2's own `<verify>` block asserted these three tuples equal the artifacts'
`bucket_edges` arrays elementwise, as float64, with no tolerance.

`SEED_HANDLING_RULE = "no_pooling_per_seed_verdicts"` is the ratified outcome of the `05-03`
Task 1 checkpoint rather than a value anyone chose at this plan — this document's checkpoint
ratification note above records how it was decided.

## `VERDICT_RULE` (verbatim, from `linear_probe.py` at the freeze commit)

```
D5-09 per-seed VERDICT_RULE -- ratified at plan 05-04's Task 1 blocking
checkpoint, before any PU probe number existed.

Per seed, the headline comparison is that seed's highest-||H|| bucket (of N_BUCKETS = 3
tertiles) against its lowest, on mean per-point squared L2 residual over the ONE shared 70/30
test split (TRAIN_FRACTION, SPLIT_SEED), under that seed's own frozen BUCKET_EDGES_PER_SEED
entry.

That seed's verdict is HOLDS if and only if ALL three of:
  (a) the highest and lowest bucket's CONFIDENCE_LEVEL (0.95) percentile bootstrap CIs on
      mean per-point squared L2 residual are disjoint;
  (b) the highest bucket's mean residual strictly exceeds the lowest bucket's; AND
  (c) the sign survives that seed's SIZE_MATCH_RULE re-check (subsampled to that seed's
      realized test-split bucket counts) with CIs disjoint in at least half of
      SIZE_MATCH_N_REPEATS = 200 repeats.

NO DETECTABLE RELATIONSHIP is that seed's verdict whenever any one of (a)/(b)/(c) fails. It is
a complete, valid, TERMINAL per-seed outcome -- never a phase failure, never escalated by the
continuous statistic, and never re-decided by trying a different N_BUCKETS.

The three per-seed verdicts (HOLDS / NO DETECTABLE RELATIONSHIP) then combine under
SEED_VERDICT_COMBINATION_RULE into exactly one of PHASE_VERDICT_VALUES, including the
terminal outcome SPLIT ACROSS SEEDS -- see that rule's own text for the full mapping and for
why a split is not partial support.

The continuous Spearman between that seed's curvature magnitude and per-point residual on the
test split is reported per seed alongside the verdict as SENSITIVITY ONLY; it can neither
establish nor overturn any verdict at either the per-seed or the phase level.

D5-11 CAVEAT, carried in this rule's own text rather than only alongside it: the field this
rule buckets on has no demonstrated relationship to true curvature. The sealed d=20 decoder
row is rank_spearman_rho = -0.015106571347065712 against the only analytic-curvature control
that tests it, essentially zero, with 52 to 75 percent of points anti-aligned in direction. A
Swiss roll / low-d anchor was offered and declined for this phase. No verdict produced under
this rule can be attributed to curvature by anything in this phase. The mitigating context --
the sealed saddle control sets a constant analytic Hessian, so its ||H|| varies only through
the pullback metric, which may make that fixture structurally unable to show ordering at all
-- is reported and is explicitly NOT used to upgrade any result produced under this rule; the
question is open and it is not for autonomous action.

D5-12 CAVEAT, carried in this rule's own text: the CAE supplying every decoder this rule reads
curvature from failed its own validity gate (CAE_VERDICT = FAIL, Phase 02.2); Phase 3 ran on a
deliberate override of that gate; Phase 03.1 found the pullback metric repaired by the scale
prior while the curvature ordering only partially and non-seed-consistently moved. Every
verdict this rule produces inherits that chain.

D5-13 NOTE: the per-seed density Spearman (spearman(density, ||H||)) is reported alongside
every verdict as a disclosure only; it is not a gate under this rule.
```

## `SEED_VERDICT_COMBINATION_RULE` (verbatim, from `linear_probe.py` at the freeze commit)

```
D5-09 SEED_VERDICT_COMBINATION_RULE -- ratified at plan
05-04's Task 1 blocking checkpoint, before any PU probe number existed. Supersedes
05-CONTEXT.md D5-04's pooled-field design per 05-03-DECISION.md.

The probe is scored once per seed under the IDENTICAL protocol (the identical TRAIN_FRACTION
70/30 split, shared across all three seeds' bucketings via the one SPLIT_SEED) and the
IDENTICAL VERDICT_RULE, producing exactly one per-seed terminal verdict per seed: HOLDS or
NO DETECTABLE RELATIONSHIP.

The three per-seed verdicts combine into exactly one PHASE_VERDICT_VALUES member by counting
the HOLDS outcomes:
  * three of three HOLDS  -> "HOLDS IN ALL THREE SEEDS"
  * zero of three HOLDS   -> "NO DETECTABLE RELATIONSHIP IN ANY SEED"
  * one or two of three   -> "SPLIT ACROSS SEEDS"

SPLIT ACROSS SEEDS is a COMPLETE TERMINAL OUTCOME and is NOT partial support for the
hypothesis. The three seed fields were measured at 05-02 to be mutually anti-correlated on
rank (pairwise Spearman on H_norm -0.1402, +0.2019, -0.2725 -- sign-inconsistent, two of
three negative) and directionally orthogonal (median cosine of unit H_vec 0.0007 to 0.0039,
with 46 to 48 percent of points anti-aligned between any pair), so a relationship that appears
in one or two of three seeds' fields and not the third is a property of that individual
decoder fit, not of the manifold, and does not license the claim that decodability degrades
with curvature.

A split is NEVER upgraded to HOLDS IN ALL THREE SEEDS by majority vote, by the continuous
Spearman statistic, by a non-headline bucket, or by trying a different N_BUCKETS; and it is
NEVER downgraded to NO DETECTABLE RELATIONSHIP IN ANY SEED either -- it is reported exactly as
SPLIT ACROSS SEEDS, with all three per-seed verdicts and their supporting numbers beside it.

Because one split is shared across all three seeds' bucketings, the three per-seed verdicts
are NOT statistically independent -- they score the same held-out residuals under three
different bucketings -- which isolates the field as the only thing that differs between them,
but must be stated in 05-FINDINGS.md rather than left implicit.
```

## Resolving `05-CONTEXT.md`'s `### Claude's Discretion` items

All four items `05-CONTEXT.md` left open, resolved here by name with a concrete value:

1. **Train/test split fraction and cross-validation scheme.** `TRAIN_FRACTION = 0.7`, one
   permutation of `np.arange(10000)` under `SPLIT_SEED = 20260824` (`SPLIT_RULE`), no
   cross-validation on the split itself — a single 70/30 realization. **This one split is
   shared across all three seeds' bucketings**, so the three per-seed verdicts differ only in
   how the same held-out residuals are bucketed and never in which points were held out; they
   are therefore not statistically independent, which isolates the field as the only thing
   that differs between the three per-seed scorings, but must be reported as a limitation in
   `05-FINDINGS.md` rather than left implicit.
2. **Residual metric details.** `RESIDUAL_METRIC = "squared_l2_per_point"` paired with
   `R2_MULTIOUTPUT = "variance_weighted"` — the one pairing where the aggregate R² and the
   per-point residual share one underlying numerator, `sum_i r_i`, satisfying CONTEXT's
   constraint exactly rather than by approximation.
3. **Number and placement of `||H||` buckets.** `N_BUCKETS = 3` (tertiles), not quartiles —
   at a 70/30 split, tertiles leave roughly 1,000 test points per bucket, supporting a
   percentile bootstrap without the CI collapsing; quartiles would leave roughly 750 and buy
   no meaningful resolution gain.
4. **Raw vs re-normalized embeddings.** `EMBEDDING_PREPROCESSING = "raw_as_cached"` — both
   modalities are already L2-normalized upstream (every row norm equals 1.0 to float64
   rounding in the resolved npz), so re-normalizing would be a no-op dressed as a decision.

## Accepted gaps this pre-registration sits on top of

Stated here at full strength, in this phase's own words, not only by cross-reference:

**D5-11, no known-answer anchor.** The sealed `d=20` decoder row is
`rank_spearman_rho = -0.015106571347065712` — essentially zero rank correlation between
decoder-side curvature and analytic curvature on the only control that tests it, with 52 to 75
percent of points anti-aligned in direction. A Swiss roll or low-`d` anchor was offered and
declined for this phase. Any relationship this phase measures rests on a field with no
demonstrated relationship to true curvature, so a detected effect cannot be attributed to
curvature by anything in this phase. The unresolved mitigating context: the sealed saddle
control sets a constant analytic Hessian, so its curvature magnitude varies only through the
pullback metric and it may be structurally unable to show ordering at all — which would make
the sealed value a fact about the fixture rather than about the decoder. This is reported and
is **explicitly NOT used to upgrade the result**; the question is **open** and it is **not for
autonomous action**.

**D5-12, the CAE gate.** `CAE_VERDICT = FAIL` at Phase `02.2`. Phase 3 runs on a deliberate
override of that gate. Phase `03.1` found the pullback metric repaired by the `scale` prior,
but the curvature ordering only partially and non-seed-consistently moved. Every Phase 5
number inherits that chain.

**D5-05, in two parts.** Part one: the measured pairwise inter-seed Spearman
(`-0.1402`, `+0.2019`, `-0.2725`) and direction values (median cosine `0.0007` to `0.0039`, 46
to 48 percent anti-aligned) — the plain statement that the three seeds do not agree on either
axis, which is why there is no pooled field. Part two, the DISPOSITION of D5-05's second half:
D5-05 asks for the Spearman between each seed and the pooled field; no pooled field exists
because seed pooling was ratified NOT DONE at the `05-03` Task 1 checkpoint (superseding
D5-04); the statistic therefore has no referent and was not computed against a substitute.
This disposition is recorded, verbatim, as `pooled_field_disposition` in
`notebooks/.cache/05_density_diagnostics.json`. It is recorded here as **DISPOSITIONED, not
satisfied and not dropped**.

**D5-13, the density confound.** Measured per seed with Phase 4's own `local_density_weights`
estimator at `K_DENSITY = 30`, `FIELD_D = 20`: `spearman(density, ||H||)` is `+0.4875` for seed
20260813, `-0.1986` for seed 20260814, `+0.0556` for seed 20260815 — the three seeds disagree
with each other on the confound's sign, exactly as they disagree on curvature rank and
direction. This is a **disclosure, not a gate**, per D5-13. Note that Phase 4's `-0.0273` was
measured on the point-cloud field and its `+0.8208` attached to curvature direction, and
neither number transfers to this phase's decoder-side field.

**The CLAUDE.md Swiss roll determination.** The Swiss-roll sanity-check rule does not trigger
for a linear probe: no bottleneck, no latent space, no decoder, no manifold-recovery claim.
The per-seed amendment introduced no new model and so does not change that determination. The
curvature estimator the fields are read from is already covered by
`notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`. Recorded here so the
determination is visible rather than silently skipped.

**The `r/R` non-application.** `chart_curvature_field` is an autodiff path with no
neighbourhood and no `k`, so the spike-findings `r/R` disclosure has no value to report here —
recorded as a visible non-application, not an omission. The density Spearman's direction axis
is likewise inapplicable: both operands (`density`, `||H||`) are scalars, so no vector
direction axis exists and the sign of `rho` is the direction.

---
*Phase: 05-curvature-conditioned-linear-decodability*
*Plan: 05-04, Task 3*
