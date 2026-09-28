# Phase 4 Pre-Registration — every free parameter frozen before a PU region label exists

**Date:** 2026-08-24
**Git HEAD at time of writing:** `e1106b4` (Task 1's commit — `notebooks/pu_manifold/region_partition.py`'s helper and known-answer test)
**Committed by:** plan `04-03`, Task 3, per the ROADMAP's Ordering constraint (the
garden-of-forking-paths guard). This document's own commit must precede every commit
carrying a regional MKNN number. No PU region label has been computed anywhere in the repo
as of this document's commit.

## Checkpoint ratification note

Plan `04-03`'s Task 2 is a `checkpoint:decision` gated `blocking`, requiring a human decision
before this pre-registration could be written. **The user was asleep and had explicitly
authorized this phase to run to completion without further human gates.** The orchestrator
that spawned this execution answered the checkpoint on the user's standing authorization,
ratifying the plan's recommended option (`ratify-recommended`) with **no amendments** — every
value below is exactly the value the plan proposed before this checkpoint existed, none was
invented, relaxed, or tuned. The alternative `majority-across-k` option was explicitly
rejected; the single-headline-`k` verdict shape stands. This is recorded here plainly as a
ratification made under standing authorization while the user was unavailable — **not** a
silent default, and **not** a claim that the user personally reviewed each value line by
line.

## What this document forecloses

Once this document is committed, the following may not change for the remainder of Phase 4:
the near-zero exclusion percentile (`MIN_NORM_PERCENTILE`), the region-size floor and its
undefined-cell behaviour (`MIN_REGION_N`), the frozen curvature-field `k` (`K_FROZEN`), the
MKNN `k` grid and its headline member (`MKNN_K_GRID`, `HEADLINE_K`), the null quantile and
confidence level (`NULL_QUANTILE`, `CONFIDENCE_LEVEL`), the permutation/bootstrap counts
(`N_PERMUTATIONS`, `N_BOOTSTRAP`), and the full text of `VERDICT_RULE`. Changing any of these
after a regional MKNN number exists invalidates the phase's result — undoing this
pre-registration costs a fresh pre-registration and a complete re-run of every regional cell,
per the 02.2 CAE precedent.

## Pre-registered constants

Every value below is the verbatim value of the identically-named constant in
`notebooks/pu_manifold/region_partition.py`, verified equal by this plan's own `<verify>`
step.

| Constant | Value | Source / rationale |
|---|---|---|
| `MIN_NORM_PERCENTILE` | `5.0` | Within-config percentile of the field's own `‖H‖`; ~500/10,000 points excluded; never an absolute threshold |
| `MIN_REGION_N` | `500` | `= 10 * k_max` at `k_max = 50` (RESEARCH A4's number, ratified — a reasoned default with no literature precedent) |
| `MKNN_K_GRID` | `(5, 10, 20, 50)` | D4-17/MKNN-06's grid |
| `HEADLINE_K` | `20` | RESEARCH A5's shape: one headline `k`, the rest sensitivity, ratified |
| `NULL_QUANTILE` | `0.99` | Permutation-null threshold |
| `CONFIDENCE_LEVEL` | `0.95` | Bootstrap CI level |
| `N_PERMUTATIONS` | `1000` | D4-17 |
| `N_BOOTSTRAP` | `1000` | D4-17 |
| `FIELD_D` | `20` | D-07: explicit call-site value, never re-derived |
| `K_DENSITY` | `30` | D4-15 |
| `SEED` | `20260822` | Existing runner's seed, kept for continuity, passed explicitly to every stochastic call |
| `K_FROZEN` | `500` | Copied verbatim from `notebooks/.cache/04_k_freeze.json` (plan `04-02`) — **fallback, not a detected plateau** (see below) |
| `COVARIANCE_FORM` | `"mean_centered"` | `np.cov`'s own form; `mean_unit_norm` is reported beside every `region_partition` result so a reader can see whether the mean-centered and uncentered forms coincide on PU |

## `K_FROZEN` provenance (inherited from plan `04-02`, restated here so this document is
self-contained)

`K_FROZEN = 500` is the **pre-registered fallback** — "the largest `k` actually run" — **not**
a detected reliability plateau. D4-07's freeze rule never fired anywhere in the six-point
sweep grid:

| `k` | `median_R_H` | delta vs prev |
|---|---|---|
| 30 | 0.0279 | — |
| 60 | 0.0927 | +0.0648 |
| 120 | 0.1573 | +0.0646 |
| 231 | 0.2337 | +0.0763 |
| 350 | 0.2853 | +0.0516 |
| 500 | 0.3436 | +0.0583 |

`rule_fired = false`. `median_R_H` reached only 0.3436 against the rule's 0.5 floor, and the
per-step increment never collapsed toward the 0.03 ceiling — it **rose** at the last step
(0.0516 → 0.0583) rather than settling. `K_FREEZE_RULE`, copied verbatim into
`region_partition.py` from `04_k_freeze.json`:

> D4-07: freeze the curvature-field k at the smallest k in the ordered sweep grid whose
> median_R_H gain over the immediately preceding sweep point is strictly less than 0.03 AND
> whose median_R_H is greater than or equal to 0.5. The rule is evaluated from the SECOND
> sweep point onward, because the gain at the first point is undefined. If no k in the grid
> satisfies both conditions, the frozen k is the largest k actually run and the outcome is
> recorded as not-fired -- never adjusted post hoc.

This sweep is never described as converged, plateaued, or settled anywhere in this phase's
record.

## `VERDICT_RULE` (verbatim, from `region_partition.py`)

```
MKNN-07 verdict rule -- ratified at this plan's Task 2 blocking checkpoint,
before any regional MKNN number existed.

The high-vs-low regional MKNN result HOLDS at a given k if and only if BOTH:
  (a) the two regions' CONFIDENCE_LEVEL (0.95) percentile bootstrap CIs at that k are
      disjoint, AND
  (b) the higher-scoring region's observed MKNN strictly exceeds the NULL_QUANTILE (0.99)
      percentile of its OWN region-scoped permutation null.

The headline call is made at HEADLINE_K = 20 alone. The remaining grid values, k in
MKNN_K_GRID = (5, 10, 50), are reported as sensitivity only: they cannot overturn or escalate
the headline verdict, and take no separate multiplicity correction. No multiplicity correction
is applied across the 2x4 grid, because the four k values are a nested sensitivity sweep on
the same two regions and the same embeddings, not independent trials.

"NO DETECTABLE DIFFERENCE" at the headline k is a complete, valid outcome. It is never treated
as a phase failure and it is never escalated by a majority vote across the sensitivity k --
that alternative verdict shape was considered and rejected at the Task 2 checkpoint.

D4-14 CAVEAT, carried in this rule's own text rather than only alongside it: the
density-confound battery run in this phase is the REGN-02 correlation only -- no
density-matched null. MKNN is itself a k-NN statistic and therefore directly
density-sensitive. A detected regional MKNN difference under this rule CANNOT be attributed
to curvature rather than to regional density by anything in this phase.
```

## Resolving `04-CONTEXT.md`'s `### Claude's Discretion` items

Every item CONTEXT.md left to the planner, resolved here with a concrete value — none
deferred:

1. **MKNN-07's verdict rule, headline `k`, multiplicity.** Resolved above: one headline
   `k = 20` with CI-disjointness AND own-null-exceedance both required, `k` in `{5, 10, 50}`
   as sensitivity only, no multiplicity correction across the 2x4 grid because the four `k`
   are a nested sensitivity sweep, not independent trials.
2. **Near-zero `‖H‖` exclusion.** 5th percentile (`MIN_NORM_PERCENTILE = 5.0`) of the field's
   own `‖H‖` distribution, inclusive at the boundary (`>=`, never `>`), excluded count
   reported via `region_partition`'s `excluded_idx`/`region_counts`.
3. **Unbalanced regions.** `MIN_REGION_N = 500`. Below it, that cell is recorded `undefined`
   with `reason: "n_region < MIN_REGION_N"` (the module's `MIN_REGION_N_UNDEFINED_REASON`
   constant), nothing is computed for it, and the headline verdict reads "NO DETECTABLE
   DIFFERENCE — comparison undefined at this split balance." Independently, any cell where
   `k + 1 > n_region` is `undefined` regardless of the floor.
4. **Whether `v` is computed on all 10k.** No: on all points surviving the near-zero
   exclusion (~9,500 of 10,000), with no further subsampling.
5. **Field computation scope.** The full 10,000 rows; `n_anchor = 1000` per the existing
   runner's precedent; `SEED = 20260822` passed explicitly to every stochastic call;
   `d = 20` (`FIELD_D`) kept as an explicit call-site value and never re-derived, per D-07.
6. **REGN-02 correlation statistic.** Both: plain Spearman of density against `‖H‖` and
   against the signed projection `<H_i/‖H_i‖, v>`. No partial correlation and no
   density-matched control, per D4-14.
7. **Region-level density comparison after the split.** Yes: median and interquartile range
   of the density per region, plus a two-sided Mann-Whitney U test.
8. **Exact `k` grid past 231.** `350` first, D4-07's rule applied immediately, `500` only
   because the rule had not fired at 350 — this is already recorded in
   `notebooks/.cache/04_k_freeze.json` and copied into `K_FROZEN`/`K_FREEZE_RULE` above.
9. **MKNN-08 hubness.** Substantiated, not stated only: k-occurrence skewness from the same
   membership matrix, per region, per embedding side, per `k`.
10. **Shipped artifact shape.** Runner (`region_partition_mknn_run.py`) plus JSONL cache
    (`04_region_partition_mknn.jsonl`) plus notebook
    (`notebooks/04_region_partition_mknn.ipynb`), following every prior phase's pattern.

## Accepted gaps this pre-registration sits on top of

This document pre-registers a verdict rule; it does not, and cannot, retroactively validate
the machinery underneath that rule. Three gaps are accepted, not closed, and are stated here
in this phase's own words rather than by reference (`04-06` carries the full write-up; this
is the short form so the pre-registration itself is honest about what it is pre-registering
on top of):

**The curvature field itself is unvalidated on real data (D4-03's inherited gap).**
`centroid_mean_curvature` is exercised against closed-form synthetic fixtures (planes,
spheres, the Swiss roll) where the answer is known in advance, and split-half reliability on
PU shows the two disjoint halves of the cloud increasingly agree with each other as `k` grows
(`median_R_H` climbing from 0.028 to 0.344 across the sweep, `fraction_negative` falling from
0.364 to 0.004). Agreement between two halves of the same cloud is not evidence of
correctness — a systematic estimator bias or a density artifact shared by both halves would
be perfectly reliable by this statistic and completely invisible to it. There is no ground
truth for PU's curvature field, and nothing in this milestone upgrades reliability to
correctness.

**The direction-partition's codimension gap is unclosed (D4-01/D4-10).** Every fixture the
direction-partition decision (D4-01) rests on is a codimension-1 graph, where
`H = H_scalar * n_hat` — so a measured cosine near 1.000 on those fixtures demonstrates
recovery of the surface's normal orientation, a tangent-space problem known to converge well,
not resolution of `H`'s direction within a high-dimensional normal space. PU's codimension is
roughly 748 (`d ~ 20` inside `D = 768`), and D4-10 explicitly declines to run either
`make_ridge_graph_control` or `make_multinormal_ridge_control` before freezing the PU split,
overriding D4-01's own body text that had called that check a Phase 4 precondition. That gap
is unmeasured on PU and is not closed by anything in this milestone; the sign split partitions
whatever direction structure the field happens to carry, without itself validating that
structure at PU's codimension.

**The density confound is reported, not controlled (D4-14).** The only density-confound check
this phase runs is REGN-02's plain Spearman correlation between local density and curvature,
before and after the split. No partial regression, no density-matched null, no
centroid-distance check, and no density-matched stratification are run. MKNN is itself a
k-NN statistic and is therefore directly density-sensitive by construction — so a detected
regional MKNN difference cannot be separated from a regional density difference by anything
in this phase. This consequence is written into `VERDICT_RULE`'s own text above, not stated
only alongside it.

**Phase 4 produces its result with no known-answer anchor at any point in the chain:
estimator, field, or partition.**

---
*Phase: 04-region-partitioning-regional-alignment-mknn*
*Plan: 04-03, Task 3*
