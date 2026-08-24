# 05-03 Task 1 — ratified one-way decision: seed pooling

**Date:** 2026-08-24
**Decided by:** developer, at the 05-03 Task 1 blocking checkpoint
**Status:** RATIFIED. D5-04 rates this one-way. Not revisable after the 05-04 freeze.

## Decision

**Do not pool the three seed curvature fields.** Run the probe once per seed and report
three per-seed verdicts and their spread. No pooled field, no pooled bucket edges.

Rejected: per-seed median-divide then average (05-RESEARCH.md's recommendation);
per-seed percentile-rank then average; halting the phase.

## Evidence the decision was made on

Measured at 05-02 over all 10,000 PU points, three sealed CAE checkpoints
`03_converged_cae_pu_nc4_seed2026081{3,4,5}`, via
`chart_curvature.chart_curvature_field(model, x64, mode='reverse')`.
Source artifact: `notebooks/.cache/05_inter_seed_diagnostics.json`.

Pairwise Spearman on `H_norm` — sign-inconsistent, two of three negative:

| pair | rho | p |
|------|-----|---|
| 20260813 vs 20260814 | −0.1402 | 4.8e−45 |
| 20260813 vs 20260815 | +0.2019 | 1.8e−92 |
| 20260814 vs 20260815 | −0.2725 | 8.9e−170 |

Direction axis, reported beside every rank statistic per the spike-findings requirement:

| pair | median cosine | fraction anti-aligned |
|------|---------------|----------------------|
| 20260813 vs 20260814 | 0.0039 | 46.1% |
| 20260813 vs 20260815 | 0.0014 | 48.1% |
| 20260814 vs 20260815 | 0.0007 | 46.4% |

Per-seed structure:

| seed | median ‖H‖ | charts used | median log10 det g | effective distinct levels |
|------|-----------|-------------|--------------------|--------------------------|
| 20260813 | 1,363 | 2 | −68.5 | 10,000 (continuous) |
| 20260814 | 51,438 | 4 | −165.6 | **4** |
| 20260815 | 70,794 | 3 | −165.7 | **3** |

`r/R` is `null` and the reason is recorded: `chart_curvature_field` is an autodiff path
through the decoder with no neighbourhood and no k, so no neighbourhood-radius-over-cloud-radius
ratio exists to disclose.

## Correction to 05-02-SUMMARY.md

05-02-SUMMARY.md states seeds 20260814/15 are "not literally piecewise-constant — 5,301 / 9,852
exact distinct `H_norm` values (not 3-4)". **That claim is wrong** and must not be carried into
05-PREREGISTRATION.md or 05-FINDINGS.md.

Those exact-distinct counts are float noise in the last ULPs. Measured directly from the npz at
relative precision, seed 20260814 has **4** distinct levels and seed 20260815 has **3**, stable
from rel 1e−9 through rel 1e−3. Seed 20260813 has 10,000 / 9,951 / 1,499 over the same range.
The `n_distinct_h_norm` field in the diagnostics artifact uses `np.round(H_norm, 6)` — absolute
rounding, which at magnitude 5e4 is a relative precision of ~2e−11 — and even it reports 16 and 59.

05-RESEARCH.md Pitfall 2's "3-4 distinct values" and 03-09-SUMMARY.md's original measurement
were both correct.

## Rationale

The three fields are mutually anti-correlated and directionally orthogonal, and two of the three
are 3–4 level step functions sitting at a metric-determinant regime ~100 orders of magnitude away
from the third. Any pooled field would not be a consensus: it would be seed 20260813's structure
plus two step functions that disagree with it and with each other.

- Under percentile-rank, seeds 20260814/15 enter as 3–4 giant tied blocks, making an
  equal-frequency tertile cut fragile.
- Under median-divide, they divide to near-constants that flatten the pooled variation, leaving
  a pooled field dominated by seed 20260813 — which 05-03's own must_have would then have to
  record as domination.

Pooling would assert a seed agreement the measurement did not find. Three per-seed verdicts with
their spread reports what was actually measured, and preserves the phase's ability to answer the
decodability question per seed rather than not at all.

## Consequence — waves 3-6 require amendment before the freeze

The pooled field is load-bearing in 05-03, 05-04, 05-05 and 05-06. This decision invalidates:

- **05-03**: the `notebooks/.cache/05_curvature_field.npz` pooled artifact must_have; the
  pooled-vs-seed Spearman must_have; the D5-13 density Spearman is re-measured per seed instead.
- **05-04**: `POOLING_METHOD` has no value to hold; `BUCKET_EDGES` becomes three per-seed edge
  sets rather than one; `VERDICT_RULE` must define how three per-seed verdicts combine into the
  phase read-out, including what a split outcome means.
- **05-05**: three probe runs bucketed by three fields, not one; the continuous Spearman becomes
  three values; the emitted verdict row schema changes.
- **05-06**: the notebook and findings report three verdicts and their spread; the "is the pooled
  field an artifact of averaging fields that share no signal" question is answered by this
  decision record rather than by a pooled diagnostic.

Amendment must land **before** any pre-registered constant is set, so that the freeze commit
remains an ancestor of the first commit carrying a PU probe number (D5-09).
