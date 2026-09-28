# Phase 5 Findings — Curvature-Conditioned Linear Decodability

**Date:** 2026-08-24. **Milestone:** v1.1 PU Manifold Curvature. **Phase:**
05-curvature-conditioned-linear-decodability.

**One-line outcome.** One global ridge map from `hsc` to `legacysurvey` (`n_train=7000`,
`n_test=3000`, `selected_alpha=0.1`, `r2_overall=0.643931`) was scored once per seed against
three independently-bucketed, per-seed decoder-side curvature-magnitude fields, under a
`VERDICT_RULE` and `SEED_VERDICT_COMBINATION_RULE` frozen in committed source before any of the
three numbers existed. **Phase verdict: `SPLIT ACROSS SEEDS`** (`n_holds=2` of 3 —
20260813 HOLDS, 20260814 NO DETECTABLE RELATIONSHIP, 20260815 HOLDS). This is a complete,
terminal, non-supportive outcome under the frozen rule — explicitly not partial support for the
hypothesis, and explicitly not upgradable by majority vote, by the continuous Spearman
statistic, or by trying a different `N_BUCKETS`. RESEARCH A2's stated reason for using ridge
regression was measured against the training split's own singular spectrum and was **not
confirmed**.

---

## 1. What this phase claims and what it deliberately does not

This section comes first for the same reason `04-FINDINGS.md` Section 1 does: a reader must not
reach the result before the conditions on it.

**Claim.** Under rules frozen before any of the three numbers below existed
(`05-PREREGISTRATION.md`, committed at `b45ae1b`, after the freeze commit `32dabe3`), held-out
per-point residual from a single global ridge map between the two PU modalities either does or
does not differ between the highest and lowest tertile of a decoder-side curvature-magnitude
field — asked independently of three CAE seeds, answered three times, and combined into one
phase read-out by a rule frozen alongside the per-seed rule. The answer is
`SPLIT ACROSS SEEDS`: two of the three seeds' decoder fits (20260813, 20260815) show the
relationship; the third (20260814) does not.

**What this phase deliberately does not claim, in its own words, at full strength:**

- **That the field measures true curvature.** See Section 6, D5-11: the sealed `d=20` decoder
  row is `rank_spearman_rho = -0.015106571347065712` against the only analytic-curvature control
  that tests it — essentially zero rank correlation, with 52 to 75 percent of points
  anti-aligned in direction. Any relationship this phase measures rests on a field with no
  demonstrated relationship to true curvature, so a detected effect cannot be attributed to
  curvature by anything in this phase.
- **That a difference, if found, is caused by curvature rather than by some other property of
  that seed's decoder fit.** The three per-seed fields were measured (Section 3) to be mutually
  anti-correlated on rank and directionally orthogonal — whatever `||H||` is picking up, it is
  not the same thing across the three seeds' CAE fits.
- **That the CAE the decoder comes from is valid.** See Section 6, D5-12: `CAE_VERDICT = FAIL`
  at Phase 02.2. Phase 3 runs on a deliberate override of that gate.
- **That a per-seed difference, where the verdicts split, generalizes beyond that seed's
  decoder fit.** `SPLIT ACROSS SEEDS` is reported here exactly as the frozen combination rule
  defines it: a complete, terminal, non-supportive outcome, not a 2-of-3 majority in favor of the
  hypothesis.
- **That the three per-seed verdicts are independent evidence.** Section 5 states this
  explicitly: one shared 70/30 split serves all three seeds' bucketings, so the three verdicts
  differ only in how the same 3,000 held-out residuals are bucketed, never in which points were
  held out or how the map was fit.

**This document does not reopen, soften, recompute, or reinterpret any sealed verdict from
Phases 2, 02.x, 3, 03.1, or `05-01` through `05-05`.** Where the CAE gate or the decoder curvature
chain is referenced below, its own caveat travels with it (Section 6).

---

## 2. The frozen configuration

Every value below is the verbatim value of the identically-named constant in
`notebooks/pu_manifold/linear_probe.py` as committed at the D5-09 freeze commit **`32dabe3`**
(plan `05-04` Task 2). The human-readable record is `05-PREREGISTRATION.md`, committed at
**`b45ae1b`** (plan `05-04` Task 3), after the freeze and before any PU probe number existed.

| Constant | Value |
|---|---|
| `TRAIN_FRACTION` | `0.7` |
| `SPLIT_SEED` | `20260824` |
| `SPLIT_RULE` | One permutation of `np.arange(10000)` under `SPLIT_SEED`; first 7,000 of the permutation train, last 3,000 test; both sorted ascending; NOT stratified by bucket |
| `RIDGE_ALPHA_GRID` | `(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)` |
| `RIDGE_SELECTION_RULE` | scikit-learn `RidgeCV`'s generalized leave-one-out CV on the training split alone, selecting one alpha from the grid |
| `ALPHA_PER_TARGET` | `False` |
| `FIT_INTERCEPT` | `True` |
| `EMBEDDING_PREPROCESSING` | `"raw_as_cached"` — both modalities already L2-normalized upstream |
| `RESIDUAL_METRIC` | `"squared_l2_per_point"` |
| `R2_MULTIOUTPUT` | `"variance_weighted"` |
| `N_BUCKETS` | `3` (tertiles) |
| `BUCKET_RULE` | Equal-frequency rank partition of ONE seed's `\|\|H\|\|` field, applied independently per seed; a value equal to an edge lands in the HIGHER bucket |
| `BUCKET_EDGES_PER_SEED` | `((1225.4263017421292, 1538.3597929379368), (49062.2351870738, 66977.54374981482), (51694.86079512253, 75252.52609688243))` |
| `SEED_HANDLING_RULE` | `"no_pooling_per_seed_verdicts"` |
| `SEED_VERDICT_COMBINATION_RULE` | (verbatim, quoted in full in Section 5) |
| `PHASE_VERDICT_VALUES` | `("HOLDS IN ALL THREE SEEDS", "SPLIT ACROSS SEEDS", "NO DETECTABLE RELATIONSHIP IN ANY SEED")` |
| `SEED_STEMS` | `(20260813, 20260814, 20260815)` |
| `N_CHARTS` | `4` |
| `CURVATURE_MODE` | `"reverse"` |
| `CURVATURE_CONVENTION` | `"trace"` |
| `CURVATURE_SOURCE_FUNCTION` | `"chart_curvature.chart_curvature_field"` |
| `SIZE_MATCH_RULE` | Per seed, subsample every bucket to that seed's smallest REALIZED test-split bucket count, re-run the highest-vs-lowest comparison |
| `SIZE_MATCH_N_REPEATS` | `200` |
| `SIZE_MATCH_SEED` | `20260824` |
| `N_BOOTSTRAP` | `1000` |
| `BOOTSTRAP_SEED` | `20260824` |
| `CONFIDENCE_LEVEL` | `0.95` |
| `K_DENSITY` | `30` |
| `FIELD_D` | `20` |
| `VERDICT_RULE` | (verbatim, quoted in full in Section 5) |
| `PREREGISTRATION_PATH` | `".planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md"` |

**`POOLING_METHOD` and `BUCKET_EDGES` were RETIRED at `05-03`**, before the freeze — structurally
removed from `linear_probe.py`'s constants block rather than merely left unset, so a reader
comparing against `05-CONTEXT.md` D5-04 sees a removal, not an omission. They were retired
because the pooled-field design they served was rejected at the `05-03` Task 1 blocking
checkpoint (Section 3).

---

## 3. The seed decision and the field

**`05-CONTEXT.md` D5-04 called for pooling** the three cached CAE seeds into one averaged
`||H||` field and naming it the verdict field. That question was put to the developer at the
`05-03` Task 1 blocking checkpoint with `05-02`'s measured inter-seed numbers already on the
table. **The ratified, one-way outcome was the opposite: do not pool.** Run the probe once per
seed and report three per-seed verdicts and their spread. This decision is recorded in
`05-03-DECISION.md`, which **SUPERSEDES** `05-CONTEXT.md` D5-04. Rejected alongside the raw
average: per-seed median-divide then average (`05-RESEARCH.md`'s own recommendation), per-seed
percentile-rank then average, and halting the phase.

**Evidence the decision was made on**, measured at `05-02` over all 10,000 PU points via
`chart_curvature.chart_curvature_field(model, x64, mode='reverse')` on the three sealed CAE
checkpoints `03_converged_cae_pu_nc4_seed2026081{3,4,5}`:

Pairwise Spearman on `H_norm` — sign-inconsistent, two of three negative:

| pair | rho | p |
|---|---|---|
| 20260813 vs 20260814 | **-0.1402** | 4.8e-45 |
| 20260813 vs 20260815 | **+0.2019** | 1.8e-92 |
| 20260814 vs 20260815 | **-0.2725** | 8.9e-170 |

Direction axis, reported beside every rank statistic per the spike-findings requirement:

| pair | median cosine | fraction anti-aligned |
|---|---|---|
| 20260813 vs 20260814 | 0.0039 | 46.1% |
| 20260813 vs 20260815 | 0.0014 | 48.1% |
| 20260814 vs 20260815 | 0.0007 | 46.4% |

Per-seed structure: seed 20260813 has 2 charts used, median `log10 det g` -68.5, and a
continuous field; seeds 20260814 and 20260815 have 4 and 3 charts used respectively, median
`log10 det g` around -165.6 / -165.7 — a metric determinant roughly 100 orders of magnitude
from seed 20260813's. Any pooled field would not have been a consensus: it would have been seed
20260813's structure plus two step-like functions that disagree with it and with each other.

**Per-seed field summary**, read directly from the cached field artifacts:

| seed | median `\|\|H\|\|` | min | max | charts used | effective distinct levels (rel 1e-9 to 1e-3) |
|---|---|---|---|---|---|
| 20260813 | 1,363.14 | 681.33 | 4,283.93 | 2 | continuous (10,000 / 9,904 / 1,173) |
| 20260814 | 51,437.9 | 29,699.4 | 66,977.5 | 4 | **4** |
| 20260815 | 70,794.1 | 51,694.9 | 75,252.5 | 3 | **3** |

**Correction, carried plainly rather than repeated.** `05-02-SUMMARY.md` reported seeds
20260814 and 20260815 as "not literally piecewise-constant — 5,301 / 9,852 exact distinct
`H_norm` values (not 3-4)". **That claim is wrong** — those counts are last-ULP float noise
from `np.round(H_norm, 6)`'s absolute rounding at a magnitude (~5e4) where six decimal places is
a relative precision of only ~2e-11. Measured directly from the cached fields at RELATIVE
precision, stable from rel `1e-9` through rel `1e-3`, seed 20260814 has **4 effective levels**
and seed 20260815 has **3 effective levels**. `05-RESEARCH.md` Pitfall 2 and
`03-09-SUMMARY.md`'s original "3-4 distinct values" measurement were both correct;
`05-02-SUMMARY.md`'s 5,301 / 9,852 figures were retracted at `05-03-DECISION.md` and must not
travel further than this retraction.

**The pooled-artifact question, answered from the decision record.** The original plan would
have built a pooled field and then asked whether it was an artifact of averaging fields that
share no signal. No pooled field was ever built, so that question is answered here from
`05-03-DECISION.md` rather than from a diagnostic run on an object that does not exist: the
three fields are mutually anti-correlated and directionally orthogonal (tables above), and two
of the three are 3-to-4-level step functions at a metric-determinant regime roughly 100 orders
of magnitude from the third's continuous field, so a pooled field would not have been a
consensus — it would have been seed 20260813's structure plus two step functions disagreeing
with it and with each other. This is a reasoned answer from measurement, not a measurement of
the pooled object itself.

**D5-05, recorded in two parts, neither dropped.** Part one — the measured pairwise inter-seed
Spearman and direction agreement above — is **met**: it was computed and is reported in full.
Part two — the Spearman between each seed and the pooled field — has **no referent**, because no
pooled field exists (seed pooling was ratified NOT DONE at `05-03`, superseding D5-04), and was
**NOT computed against a substitute**. This disposition is recorded verbatim as
`pooled_field_disposition` in `notebooks/.cache/05_density_diagnostics.json`:

> "05-CONTEXT.md D5-05 asks for the Spearman between each seed and the pooled field. No pooled
> field exists: seed pooling was put to the developer at the 05-03 Task 1 blocking checkpoint and
> ratified as NOT DONE in 05-03-DECISION.md, superseding D5-04. The statistic therefore has no
> referent and was NOT computed against a substitute. D5-05's first half — the pairwise
> inter-seed Spearman with its direction axis — was measured at 05-02 and is recorded in
> notebooks/.cache/05_inter_seed_diagnostics.json."

D5-05's second half is **DISPOSITIONED, not satisfied and not dropped.**

---

## 4. Density

The three re-measured per-seed Spearman values between local density (Phase 4's
`local_density_weights` estimator, `K_DENSITY=30`, `FIELD_D=20`) and decoder-side curvature
magnitude, on the decoder-side field this phase reads:

| seed | spearman(density, `\|\|H\|\|`) | p |
|---|---|---|
| 20260813 | **+0.4875** | ~0 |
| 20260814 | **-0.1986** | 1.7e-89 |
| 20260815 | **+0.0556** | 2.6e-08 |

Phase 4's own point-cloud reference: **`-0.0273`** (n=9500, p=0.0078), measured on
`centroid_mean_curvature`, the point-cloud field — a different curvature estimator than this
phase's decoder-side `chart_curvature_field` values. Phase 4's direction reference: **`+0.8208`**
(n=9500, p≈0), attached to curvature DIRECTION (the sign of the projection onto Phase 4's frozen
split axis `v`) — an axis this phase does not split on (Phase 5 buckets by `||H||` magnitude
only). **Neither Phase 4 number transfers** to this phase's decoder-side field.

The three seeds do **not** agree with each other on the confound's sign either — 20260813 and
20260815 measure positive, 20260814 measures negative — exactly as they disagree on curvature
rank and direction (Section 3). This is a **disclosure, not a gate** (D5-13): it is reported
alongside every verdict and moves none of them.

---

## 5. The result

**One global ridge fit, shared by all three seeds' comparisons** — because there is one
`TRAIN_FRACTION`/`SPLIT_SEED` split and one fit, the three per-seed verdicts differ only in
which bucket edges cut the same held-out residuals, never in which points were held out or how
the map was fit:

- `n_train = 7000`, `n_test = 3000`
- `selected_alpha = 0.1` (from the frozen grid `1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4`)
- `r2_overall = 0.643931`
- `mean_residual_overall = 0.066429`

**This means the three per-seed verdicts are NOT statistically independent** — they score the
same 3,000 held-out residuals under three different bucketings, which isolates the per-seed
curvature field as the only thing that differs between them. This must be read as a spread
measuring field disagreement, not sampling variability, and it is stated here rather than left
implicit.

**Per seed, bucket edges, realized test-split counts and full-field counts** (bucket 0 = lowest
curvature, bucket 2 = highest):

| Seed | Bucket edges | Realized test counts | Full-field counts | Verdict |
|---|---|---|---|---|
| 20260813 | (1225.426, 1538.360) | (1024, 987, 989) | (3334, 3333, 3333) | **HOLDS** |
| 20260814 | (49062.235, 66977.544) | (992, 873, 1135) | (3334, 2956, 3710) | **NO DETECTABLE RELATIONSHIP** |
| 20260815 | (51694.861, 75252.526) | (986, 1019, 995) | (3334, 3333, 3333) | **HOLDS** |

**Naming Phase 4's region-size artifact as the reason this check exists (D5-08).** Phase 4's own
closing plan found that its raw regional MKNN gap was mostly an artifact of unequal region size
(`04-FINDINGS.md` §5). Seed 20260814's realized test-split counts (992, 873, 1135) diverge from
an even 1,000-per-bucket split by up to 13.5%, against the other two seeds' near-exact splits
(within ~2.4%) — this traces to a 2,102-point exact-duplicate block at seed 20260814's field
maximum (`05-03-SUMMARY.md`), which the tie rule (D5-07) correctly routes entirely into the top
bucket. This is exactly why `SIZE_MATCH_RULE` subsamples to each seed's OWN realized minimum
before re-checking sign — never the full-field count, never another seed's count.

**Per-bucket detail:**

| Seed | Bucket | n | full-field n | mean residual | R² | 95% CI |
|---|---|---|---|---|---|---|
| 20260813 | 0 | 1024 | 3334 | 0.048285 | 0.6029 | [0.046401, 0.050535] |
| 20260813 | 1 | 987 | 3333 | 0.062024 | 0.6454 | [0.059345, 0.064487] |
| 20260813 | 2 | 989 | 3333 | 0.089612 | 0.6141 | [0.085701, 0.093362] |
| 20260814 | 0 | 992 | 3334 | 0.062839 | 0.6564 | [0.060013, 0.065761] |
| 20260814 | 1 | 873 | 2956 | 0.093686 | 0.5758 | [0.090008, 0.097461] |
| 20260814 | 2 | 1135 | 3710 | 0.048602 | 0.4433 | [0.046525, 0.050471] |
| 20260815 | 0 | 986 | 3334 | 0.056126 | 0.5398 | [0.053666, 0.058710] |
| 20260815 | 1 | 1019 | 3333 | 0.077089 | 0.6867 | [0.073572, 0.080588] |
| 20260815 | 2 | 995 | 3333 | 0.065722 | 0.4011 | [0.062984, 0.068434] |

**Size-matched re-check (D5-08)**, subsampled to each seed's own smallest realized test-split
bucket count, 200 repeats: all three seeds' signs were stable across every repeat
(`sign_stable=True`, `ci_disjoint_fraction=1.0` for all three), so the size match never
overturned a headline sign. Seed 20260814's `NO DETECTABLE RELATIONSHIP` verdict fails on
criterion (b) — its highest bucket's mean residual (0.048602) is the LOWEST of its three
buckets, not the highest — not on a size-imbalance artifact.

**Continuous spearman(`||H||`, per-point residual) on the test split, per seed — sensitivity
only, non-gating**: 20260813 `rho=+0.4239` (p=3.6e-131), 20260814 `rho=-0.1169` (p=1.3e-10),
20260815 `rho=+0.1384` (p=2.6e-14). `r/R` — the spike-findings neighbourhood-radius disclosure —
is not defined for this estimator: `chart_curvature_field` is an autodiff path through the
decoder with no neighbourhood and no `k`, so there is no `r/R` to report; this is a visible
non-application, recorded here rather than silently omitted. The direction axis for the
`spearman(density, ||H||)` and `spearman(||H||, residual)` statistics is likewise not
applicable: both operands in each case are scalar fields, so there is no pair of vectors to take
a cosine between, and the sign of `rho` is the direction.

**RESEARCH A2's ridge justification — measured, not confirmed (`05-05-SUMMARY.md`).** A2 claimed
the 768-d design matrix is effectively rank-deficient at the manifold's established 18-to-25
intrinsic dimension. Checked against the training split's own measured singular spectrum:
`condition_number = 99806.5`; `effective_rank_1pct = 531` of 768 possible (at every threshold
tested, several hundred singular values remain above even the most permissive 1% cutoff — a
truly ~20-25-dimensional design matrix would put `effective_rank_1pct` near 20-25, not 531);
`cumvar_first_20 = 0.810`, `cumvar_first_25 = 0.835`; `selected_alpha = 0.1` — the
second-smallest value in the grid, **not at a grid boundary** (`alpha_at_grid_boundary=False`).
The design matrix is better conditioned than RESEARCH A2 expected — closer to a mild, broad
shrinkage regime than to the severe near-20-dimensional rank deficiency A2 described. This
invalidates nothing pre-registered — the frozen `RIDGE_SELECTION_RULE` selected the alpha, not
the planner — but the stated REASON for using ridge is not borne out by the measured spectrum,
and this is recorded here plainly rather than silently accepted.

**Phase verdict.** Both per-seed verdicts and the phase verdict are quoted byte-for-byte from
their `probe_seed` and `probe_overall` record rows:

- seed 20260813: `HOLDS`
- seed 20260814: `NO DETECTABLE RELATIONSHIP`
- seed 20260815: `HOLDS`
- **Phase verdict: `SPLIT ACROSS SEEDS`** (`n_holds=2` of 3)

The per-seed verdicts came only from `linear_probe.apply_verdict_rule`; the phase verdict came
only from `linear_probe.combine_seed_verdicts` — both applied mechanically from rules committed
before any PU probe number existed.

`VERDICT_RULE`, quoted verbatim from `linear_probe.py` at the freeze commit:

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
```

`SEED_VERDICT_COMBINATION_RULE`, quoted verbatim:

```
D5-09 SEED_VERDICT_COMBINATION_RULE -- ratified at plan
05-04's Task 1 blocking checkpoint, before any PU probe number existed. Supersedes
05-CONTEXT.md D5-04's pooled-field design per 05-03-DECISION.md.

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
```

**`SPLIT ACROSS SEEDS` is reported here exactly as the frozen rule defines it — a complete
result, not a near-miss.** Two of three seeds' decoder fits (20260813, 20260815) show the
relationship and one (20260814) does not, but the frozen rule's own reason applies without
addition: the three seeds' curvature fields were measured (Section 3) to share no signal, so an
effect present in one or two of three is a property of those individual decoder fits, not of the
manifold, and does not license the claim that decodability degrades with curvature. The
agreeing seeds are not presented as the headline with the third set aside — all three verdicts
are reported together, exactly as the combination rule requires.

---

## 6. Accepted gaps, stated up front

Each gap below is stated in this phase's own words, in full, never only by cross-reference —
matching the standard `04-FINDINGS.md` set.

**D5-11. No known-answer anchor, by explicit choice.** The sealed `d=20` decoder row is
`rank_spearman_rho = -0.015106571347065712` — essentially zero rank correlation with analytic
curvature on the only control that tests it, with 52 to 75 percent of points anti-aligned in
direction. A Swiss roll or low-`d` anchor stage was offered and declined for this phase. Any
relationship this phase measures rests on a field with no demonstrated relationship to true
curvature, so a detected effect — including the `SPLIT ACROSS SEEDS` result above — cannot be
attributed to curvature by anything in this phase.

The unresolved mitigating context, reported and explicitly **NOT** used to upgrade the result:
the sealed saddle control sets a constant analytic Hessian, so its curvature magnitude varies
only through the pullback metric, and it may be structurally unable to show ordering at all —
which would make the sealed `-0.015106571347065712` value a fact about the fixture rather than
about the decoder. **That question is open and it is not for autonomous action.**

**D5-12. The CAE gate, in phase order.** Phase 02.2: `CAE_VERDICT = FAIL` — the Chart
Auto-Encoder underlying every decoder-side field this phase reads failed its own pre-registered
validity gate (two of three gates failed: T1 geodesic distortion, T3 held-out reconstruction
margin). Phase 3: ran on a deliberate override of that FAIL — the user chose to iterate rather
than adopt the alternative representation or stop, and Phase 3's own plan records the override
in its own artifacts. Phase 03.1: found the pullback metric fully repaired by a `scale` prior
(`log10_det_g` from -83.9 to +0.037) while curvature ordering only partially and
non-seed-consistently moved (rank `rho` from -0.122 to at most +0.116). Phase 5 (this phase):
every number above inherits that chain, and this phase's own measured inter-seed disagreement
(Section 3) is a direct continuation of Phase 03.1's non-seed-consistent ordering finding — not
a new, unrelated instability.

The FAIL at Phase 02.2 and the deliberate override at Phase 3 are two facts that stand
together; **neither cancels, supersedes, nor excuses the other.**

**The `r/R` non-application.** `chart_curvature_field` is an autodiff path through the decoder
with no neighbourhood and no `k`, so the spike-findings `r/R` disclosure has no value to report
here — a visible non-application, not an omission. The density and residual Spearmans' direction
axis is likewise inapplicable: both operands in each case are scalar fields, so there is no pair
of vectors to take a cosine between, and the sign of `rho` is the direction.

**The Swiss roll determination, restated.** CLAUDE.md's standing rule requires a Swiss roll
sanity-check notebook for every model that maps data to a lower-dimensional representation and
back, or that claims to recover manifold structure. That rule does not trigger for a linear
ridge probe: no bottleneck, no latent space, no decoder, no manifold-recovery claim. The
per-seed amendment introduced no new model and does not change the determination. The curvature
*estimator* the fields are read from is already covered by
`notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`.

**The spec-less probe fallback record, with its retraction.** An earlier draft of `05-06-PLAN.md`
recorded "spec-less probe fallback skipped — the phase has no requirement IDs to probe." That is
**wrong** and is **retracted here**: the phase has thirteen requirement IDs, D5-01 through
D5-13, and the deterministic edge probe surfaced 29 applicable items across them — 23 authored
into `must_haves.truths` across the four amended plans (`05-03` through `05-06`), 6 unclassified
rows surfaced as flagged assumptions. `23 + 6 = 29` — no silent drops.

**The shared-split dependence.** One `TRAIN_FRACTION`/`SPLIT_SEED` train/test split serves all
three seeds' bucketings, so the three per-seed verdicts differ only in how the same held-out
residuals are bucketed, never in which points were held out. This is deliberate — it isolates
the curvature field as the only thing that differs between the three per-seed scorings — but it
means the three per-seed verdicts are **not statistically independent replicates**, and the
spread between them measures field disagreement, not sampling variability. Stated here in full,
not left implicit, per this document's own instruction.

---

## 7. Requirement outcomes

| Requirement | Outcome | Evidence |
|---|---|---|
| D5-01 | Met — probe `hsc -> legacysurvey`, both 768-d, from the resolved subsample npz | Section 5; `load_pu_pair` |
| D5-02 | Met — one `W` fit globally on the training split, held-out residuals bucketed three times, never refit per bucket or per seed | Section 5; `_fit_and_evaluate`'s single call site (`05-05-SUMMARY.md`) |
| D5-03 | Met — decoder-side `||H||` via `chart_curvature.chart_curvature_field`, the corrected citation, never `decoder_curvature.py` | `linear_probe.py` docstring (a); Section 5 |
| D5-04 | **SUPERSEDED** — `05-CONTEXT.md`'s pooled-field design was rejected one-way at the `05-03` Task 1 blocking checkpoint; `05-03-DECISION.md` is the authority | Section 3 |
| D5-05 | Met (part one, inter-seed Spearman/direction) / **DISPOSITIONED** (part two, pooled-vs-seed Spearman — no referent) | Section 3; `pooled_field_disposition` in `05_density_diagnostics.json` |
| D5-06 | Met — `CURVATURE_CONVENTION = "trace"`, asserted equal across `linear_probe`/`chart_curvature`/`curvature_probe` by a passing test | `linear_probe.py`; `05-04-SUMMARY.md` |
| D5-07 | Met — split on `||H||` magnitude per seed; continuous Spearman reported alongside every bucketed verdict as sensitivity only | Section 5 |
| D5-08 | Met — realized test-split bucket counts reported beside full-field counts for all three seeds; size-matched check subsamples to each seed's own realized minimum; Phase 4's region-size artifact named as the reason | Section 5 |
| D5-09 | Met — full pre-registration freeze, git-ancestry-provable | `05-VERIFICATION.md` §1-4 |
| D5-10 | Met — `run_bucketed_mode` refuses (via `assert_preregistered()`) unless the pre-registration and all three per-seed bucket artifacts exist | `05-VERIFICATION.md`, Behavioral Spot-Checks |
| D5-11 | Met (accepted gap, stated at full strength) — no known-answer anchor, sealed rank `-0.015106571347065712`, saddle-fixture question open and not for autonomous action | Section 6 |
| D5-12 | Met (accepted gap, stated at full strength) — `CAE_VERDICT = FAIL` inheritance chain in phase order | Section 6 |
| D5-13 | Met (disclosure, not a gate) — three per-seed density Spearman values re-measured on the decoder-side field, Phase 4's point-cloud reference `-0.0273` / direction `+0.8208` beside them, neither transferring | Section 4 |

All 13 requirement IDs are accounted for. D5-04 reads `SUPERSEDED`; D5-05 is split into a met
half and a dispositioned half; every other requirement reads `Met`, with its accepted gaps (if
any) named at full strength rather than hidden inside the word "Met."

---

## 8. What a follow-on phase would need

Four items, named on `05-CONTEXT.md`'s own Deferred list — none smuggled into this phase:

1. **The declined low-`d` probe-methodology anchor** (D5-11). A Swiss roll or low-`d` synthetic
   stage that runs this exact ridge-probe-and-bucket protocol against a manifold with a known
   answer, so a future `SPLIT ACROSS SEEDS` or `HOLDS` result can be told apart from a
   methodology artifact.
2. **The saddle-fixture resolution** (spike-findings-effdim's open question). Whether
   `rank_spearman_rho = -0.015106571347065712` reflects the decoder's failure or the sealed
   saddle control's own inability to show ordering (its constant analytic Hessian). Its own
   phase; blocks nothing here.
3. **An external astrophysical label as the probe target.** Requires sourcing labels and a
   row-alignment proof against `row_indices` first.
4. **Per-region independent probe fits at matched `n`, as sensitivity-only.** Rejected as
   headline at D5-02; addable later without disturbing this phase's pre-registration.

**New from this phase:** whatever would be required to obtain three CAE fits whose curvature
fields agree with each other. Without that, a per-seed spread — like the `SPLIT ACROSS SEEDS`
result reported here — is the ceiling on what any curvature-conditioned claim about this
manifold can be, regardless of how the probe or the bucketing is designed. The three sealed CAE
seeds available to this milestone were measured (Section 3) to be mutually anti-correlated and
directionally orthogonal; no protocol change in Phase 5 could have produced agreement between
fields that do not agree with each other.

---

*Phase: 05-curvature-conditioned-linear-decodability*
*Completed: 2026-08-24*
