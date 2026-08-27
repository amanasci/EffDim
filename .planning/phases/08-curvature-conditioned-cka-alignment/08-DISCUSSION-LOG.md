# Phase 8: Curvature-Conditioned CKA Alignment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-27
**Phase:** 8-curvature-conditioned-cka-alignment
**Areas discussed:** CKA estimator variant, Split + density matching, Null and verdict rule, Validation ladder

---

## Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| CKA estimator variant | Linear vs RBF; biased vs unbiased HSIC; interaction with the equal-n mandate | ✓ |
| Split + density matching | Subset count and density-matching mechanism, given +0.43/+0.32 density coupling | ✓ |
| Null and verdict rule | What gets permuted, tails, how d=32 feeds the headline | ✓ |
| Validation ladder | Low-d anchor, planted-effect positive control, Swiss roll applicability | ✓ |

**User's choice:** all four.

---

## CKA estimator variant

### Q1 — Which CKA kernel does Phase 8 compute?

| Option | Description | Selected |
|--------|-------------|----------|
| Linear CKA only (recommended) | Kornblith 2019 default; closed-form Frobenius ratio on L2-normalized 768-d; no bandwidth to tune post-hoc; O(n·D²) not n×n Gram | |
| RBF CKA only | Closer to MKNN's nonlinear neighbourhood structure, but bandwidth becomes an unvalidated pre-registered constant | |
| Both, linear headline + RBF robustness | Linear carries the verdict, RBF gates nothing (D7-03 pattern); doubles the validation ladder | ✓ |

**User's choice:** Both, linear headline + RBF robustness.
**Notes:** Recorded as D8-01. The doubled validation burden was stated in the option text and accepted.

### Q2 — Biased or unbiased HSIC inside CKA?

| Option | Description | Selected |
|--------|-------------|----------|
| Unbiased HSIC (recommended) | Song 2012 / Nguyen 2021; removes O(1/n) upward bias so subset-size drift cannot masquerade as a curvature gap | ✓ |
| Biased HSIC, equal-n enforced | Kornblith 2019 form; bias cancels in the difference only under exactly equal n, making equal-n load-bearing for correctness | |
| Both, unbiased headline + biased reported | Comparability with published CKA numbers; the gap between them is a small-sample diagnostic | |

**User's choice:** Unbiased HSIC. Recorded as D8-02.

### Q3 — RBF bandwidth (first pass, rejected for clarification)

The user paused the question to ask what RBF is, then asked for a recommendation.

**Explanation given:** RBF/Gaussian kernel `exp(-||x-y||²/2σ²)`; RBF CKA feeds pairwise similarities
into an n×n Gram instead of linear CKA's inner products, catching nonlinear structure; σ is free, and
the usual median-distance recipe computed per subset gives the denser high-`||H||` subset a smaller σ
for density reasons alone.

**Recommendation given:** σ = median pairwise distance over all 10,000 points, computed once per
modality, frozen; plus a 0.5x/2x sensitivity pair reported as a diagnostic. L2-normalized embeddings
bound pairwise distance in [0, 2], so σ is well-conditioned.

**User's choice (free text):** "Lets do global median heuristic." Recorded as D8-03.

### Q4 — Bandwidth sensitivity ladder?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — 0.5σ / σ / 2σ, reported, gating nothing (recommended) | Gram matrices already built; a sign flip across the 4x range means the RBF read is worthless; all three multipliers in the freeze commit | ✓ |
| No — frozen σ only | Smallest pre-registration surface; leaves σ-sensitivity unmeasured so RBF-vs-linear disagreement has no diagnosis | |

**User's choice:** Yes. Recorded as D8-04.

---

## Split + density matching

### Q1 — How many `||H||` subsets?

| Option | Description | Selected |
|--------|-------------|----------|
| Two extremes, top vs bottom tertile (recommended) | Max `||H||` separation per point spent; single unambiguous difference to null; discards the middle third | |
| Three tertiles, Phases 5/6 pattern | Comparable to Phase 5/6 rows without translation; invites the monotonicity criterion Phase 6 failed on | ✓ |
| Two halves at the median | Every point used, best small-sample CKA behaviour; weakest contrast on a ~1.5x spread | |

**User's choice:** Three tertiles.
**Notes:** Recorded as D8-05. The monotonicity risk it re-imports was raised here and defused later at
the statistic question (D8-10 uses tertile 3 − tertile 1 with the middle non-gating).

### Q2 — How is density matched across the tertiles?

| Option | Description | Selected |
|--------|-------------|----------|
| Within-density-stratum tertile split (recommended) | Identical density marginals by construction; equal-n free; reuses 07.1 machinery; tertiles then rank density-residualized curvature | ✓ |
| Caliper matching on density | Keeps raw `||H||` meaning; unknown discard rate, n-drift, another arbitrary constant | |
| Density reweighting, no discard | Uses every point; weighted CKA has no published bias characterization | |

**User's choice:** Within-density-stratum tertile split. Recorded as D8-06.

### Q3 — Where do the density strata come from?

| Option | Description | Selected |
|--------|-------------|----------|
| Inherit 07.1 unchanged: S=20 headline, grid (10, 20, 50) | Row-for-row comparability; stratification already exercised | |
| Inherit the estimator, re-derive S for CKA | S=20 was tuned for a partial-correlation setting, never for a Gram-matrix statistic | ✓ |
| Inherit S, hold density at a single d | D7-03's rule that density is a property of the ambient cloud | |

**User's choice:** Inherit the estimator, re-derive S for CKA.
**Notes:** Estimator inheritance recorded as D8-07 (`1.0 / w` relative density). The re-derivation
prompted the follow-up below. Clarification supplied before that follow-up: S does not change pooled
subset size (~3,333 per tertile at any S) — it trades density-match tightness against realized
`||H||` contrast, since within-stratum tertiles are computed on n/S points.

### Q4 — What rule picks S, frozen before any PU CKA number?

| Option | Description | Selected |
|--------|-------------|----------|
| Split diagnostics only, pre-outcome (recommended) | Largest S keeping realized `||H||` contrast above a floor and density-tertile correlation below a ceiling; both computable without any CKA number | |
| Tune S on the planted-effect positive control | Optimizes detection power directly, never touches PU; transfer from the synthetic fixture is an assumption | |
| No single S — 07.1's threshold-grid pattern | Declare a grid, gate one headline, report the rest to expose the S artifact | ✓ |

**User's choice:** 07.1's threshold-grid pattern. Recorded as D8-08.

### Q5 — How does the S grid gate the verdict?

Raised because the grid still needed a gating rule — 07.1's grid had `N_STRATA_HEADLINE = 20` behind it.

| Option | Description | Selected |
|--------|-------------|----------|
| Clearance required at every S in the grid (recommended) | No headline S; an S-dependent gap self-reports as an artifact; strictest side to err on for a phase whose risk is manufactured gaps | ✓ |
| S=20 headline, 10 and 50 reported | Exact 07.1 inheritance; re-introduces the arbitrary choice the grid was meant to avoid | |
| Headline S picked by pre-outcome split diagnostics | Principled selection, still one gating cell, second pre-registered rule to defend | |

**User's choice:** Clearance at every S. Recorded as D8-09.

---

## Null and verdict rule

### Q1 — What gets permuted?

| Option | Description | Selected |
|--------|-------------|----------|
| Permute `||H||` tertile labels within density strata (recommended) | Preserves density and subset sizes, breaks only the curvature link; nulls the statistic the verdict reads; 07.1 analogue | ✓ |
| Permute the crossmodal row pairing | `mknn.permutation_null`'s "pairings"; nulls alignment itself — a question Phase 7 already settled | |
| Bootstrap CI on the tertile difference | Cheaper, gives an effect-size interval; CKA bootstrap bias uncharacterized, no precedent in this record | |

**User's choice:** Permute tertile labels within density strata. Recorded as D8-11.

### Q2 — What is the test statistic over the three tertiles?

| Option | Description | Selected |
|--------|-------------|----------|
| Tertile 3 minus tertile 1, middle reported but gating nothing (recommended) | Sidesteps Phase 6's exact failure mode; non-monotone panel visible without voiding a real extremes gap | ✓ |
| Monotone trend across all three | Stronger claim when it fires; re-imports the criterion Phase 6 failed on, near-powerless at n=3 buckets | |
| Compound criterion, Phase 6 pattern | Most conservative, directly comparable to Phase 6's verdict construction | |

**User's choice:** Tertile 3 minus tertile 1. Recorded as D8-10.

### Q3 — How does d=32 gate the headline?

Framed with the tension stated on both sides: d=32 has nil density-curvature coupling (+0.0118,
p=0.238) and no surviving 07.1 association, but the phase's premise is that CKA may see what MKNN
did not.

| Option | Description | Selected |
|--------|-------------|----------|
| Hard invalidator — a gap at d=32 voids all three d (recommended) | Cheapest insurance against a Phase 4 Gap-3 recurrence; a control that gates nothing is not a control | |
| Reported diagnostic, gating nothing | D7-03 pattern; preserves the possibility CKA genuinely sees what MKNN missed; puts the artifact judgement on the reader | ✓ |
| Invalidator only on same sign and comparable magnitude | Splits the difference; costs another frozen constant and a more intricate rule | |

**User's choice:** Reported diagnostic, gating nothing.
**Notes:** Recommendation not taken. Recorded as D8-12 with the tension and the consequence written
into CONTEXT.md so no downstream agent rediscovers it. Not re-litigated.

### Q4 — How do d=20 and d=25 combine?

| Option | Description | Selected |
|--------|-------------|----------|
| Reported independently, 07.1's D-14 pattern (recommended) | No pooled headline invented; row-for-row beside 07.1's per_d_results | ✓ |
| Headline requires both to clear | One phase-level answer; likely forces a null the evidence does not support | |
| d=25 carries the headline, d=20 reported | Concentrates the claim where the record is strongest; privileges a cell chosen by a prior phase's outcome | |

**User's choice:** Reported independently. Recorded as D8-13.

### Q5 — Which curvature fields?

Asked in a second round after the user chose "More questions" on the seed axis.

| Option | Description | Selected |
|--------|-------------|----------|
| Frozen Phase 7 fields only, one per d (recommended) | Smallest surface; Phase 8's split is literally 07.1's split | |
| Frozen fields plus 07.1's three d=25 seed fields | Adds the seed axis exactly where the record supports it (SEED STABLE, 3-of-3); decoders already exist | ✓ |
| Retrain decoders for Phase 8 | Named only to rule out — breaks comparability, 1457s at d=20 scaling as D·d² | |

**User's choice:** Frozen fields plus 07.1's three d=25 seed fields. Recorded as D8-14.

### Q6 — How do the three d=25 seed verdicts combine?

Prefaced with the one-way `05-03-DECISION.md` ratification: each seed gets its own split and verdict,
no averaged `||H||` field.

| Option | Description | Selected |
|--------|-------------|----------|
| Inherit 07.1's rule — unanimous 3-of-3 or nothing (recommended) | Directly comparable; rule already exercised on this exact seed set | ✓ |
| Phase 5's SEED_VERDICT_COMBINATION_RULE verbatim | Same effect, terminal-outcome language already written and defended | |
| Report all three per-seed, no combination rule | Fullest D-14 reading; leaves "is it seed-stable" unanswered | |

**User's choice:** Inherit 07.1's rule. Recorded as D8-15.

---

## Validation ladder

### Q1 — What is the low-d anchor?

| Option | Description | Selected |
|--------|-------------|----------|
| Invariance-property ladder on synthetic pairs (recommended) | Rotation/isotropic scaling give exactly 1.0 for linear CKA; independent columns ≈0; noise ladder must decay monotonically; catches bad centering, transposed Gram, bad HSIC correction | ✓ |
| Swiss roll, per the CLAUDE.md standing rule | Named so the gate is decided on purpose; CKA is not a manifold model so the rule's shape does not obviously fit | |
| Both — invariance ladder plus a Swiss roll pass | Satisfies the standing rule literally and keeps one-test-across-every-model comparability | |

**User's choice:** Invariance-property ladder only.
**Notes:** Recorded as D8-16, and the Swiss roll gate declared **not applicable by decision** (D8-17)
rather than by omission — the option was presented and not chosen.

### Q2 — What shape does the planted-effect positive control take?

| Option | Description | Selected |
|--------|-------------|----------|
| Effect-size ladder on real PU geometry (recommended) | Real `||H||`, real strata, real subset sizes; graded degradation swept to give a detection floor, not a pass/fail; answers Phase 7's D7-02 lesson directly | ✓ |
| Single planted effect at one pre-chosen size | Phase 7's D7-02 shape; cheapest; planted size becomes an arbitrary constant | |
| Fully synthetic fixture with planted coupling | Cleanest ground truth; transfer to PU is an assumption, and the spike record documents a fixture whose structure alone moved rho from +0.593 to +0.150 | |

**User's choice:** Effect-size ladder on real PU geometry. Recorded as D8-18.

### Q3 — A negative control on the split machinery itself?

Prefaced with: d=32 gates nothing now, so the phase has no machinery-artifact control that bites.

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — shuffled-`||H||` end-to-end calibration run (recommended) | Marginal preserved, point correspondence destroyed; whole pipeline repeated to read a false-positive rate; measures directly what d=32 was proposed to catch | ✓ |
| Yes — random field with matched marginal | Same failure tested, decoupled from PU's actual values; also changes spatial structure, so proves less | |
| No — d=32 and the permutation null suffice | Cheapest; no measured false-positive rate for the pipeline as a whole | |

**User's choice:** Shuffled-`||H||` end-to-end calibration run. Recorded as D8-19.

### Q4 — How strictly does the ladder gate the PU run?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard gate — no PU number until all three rungs pass (recommended) | Matches D7-06 and the spike rule that a FAIL with no anchor cannot be told apart from broken wiring | |
| Anchor hard-gates; control and calibration run alongside | Faster wall-clock; relies on discipline not to read the PU number early | |
| All three run, none gate — reported beside the verdict | Simplest execution; makes the ladder documentation rather than a gate | ✓ |

**User's choice:** All three run, none gate.
**Notes:** Recommendation not taken. Second declined gate. Concern stated once and not re-litigated:
with D8-12 and D8-20 both non-gating, Phase 8 has no structural mechanism preventing an artifact from
being written up as a result, so every safeguard becomes a reporting obligation. Recorded as D8-20.

### Q5 — What must FINDINGS state unconditionally, given nothing gates?

Raised as the constructive consequence of Q4 — the reporting rule is now load-bearing.

| Option | Description | Selected |
|--------|-------------|----------|
| Frozen unconditional reporting block (recommended) | Pre-register the exact numbers FINDINGS must print regardless of outcome, beside the headline not in an appendix; 07.1's D-15 precedent | |
| Verdict text must name the controls inline | Structurally impossible to quote a headline without its caveat — how Phase 4's number escaped its confound | |
| Both — frozen block plus caveat-bearing verdict text | Maximum protection available without an actual gate; most pre-registration text to write up front | ✓ |

**User's choice:** Both. Recorded as D8-21.

---

## Claude's Discretion

No area was answered "you decide". The following were never put to the user and are left to the
researcher and planner by design (CONTEXT.md → Claude's Discretion):

- The S grid's exact values.
- Permutation count and null RNG seed; whether they inherit 07.1's constants.
- One- vs two-tailed threshold (Phase 7's two-tailed wrapper is the default unless the record argues otherwise).
- Invariance-ladder tolerances, shuffled-`||H||` repeat count, planted-effect magnitude steps.
- Whether the RBF σ ladder runs at every rung of the validation ladder or only at σ.
- Module naming/layout, runtime budget, wave decomposition.
- How the S-grid axis and the d=25 seed axis interact.

---

## Deferred Ideas

- **Per-point local CKA** — presented as a design alternative and not chosen (unsealed instrument,
  small-sample CKA bias, full validation burden). Would make 07.1's per-point machinery reusable.
- **Intramodal CKA across a model-size ladder** — already deferred at project level; needs a second
  model size.
- **Promoting Phase 8 code into `src/effdim/`** — needs its own test suite and milestone.
- **Human ratification of Phase 7 / 07.1 verdicts** — outstanding UAT item, not a Phase 8 task.

No scope creep was raised during the discussion; every question stayed inside the phase boundary.
