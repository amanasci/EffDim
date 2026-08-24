---
phase: 04-region-partitioning-regional-alignment-mknn
plan: 05
subsystem: manifold-curvature
tags: [mknn, region-partitioning, pre-registration, density-confound, permutation-test, bootstrap-ci, hubness]

# Dependency graph
requires:
  - phase: 04-region-partitioning-regional-alignment-mknn
    provides: "plan 04-03's region_partition.py (frozen VERDICT_RULE, HEADLINE_K=20,
      MKNN_K_GRID, NULL_QUANTILE, CONFIDENCE_LEVEL, N_PERMUTATIONS, N_BOOTSTRAP,
      MIN_REGION_N, assert_preregistered()) and 04-PREREGISTRATION.md, committed before any
      regional MKNN number existed; plan 04-04's frozen split artifact
      (notebooks/.cache/04_region_partition.npz: region_0=6256, region_1=3244,
      excluded=500) and its measured density confound
      (spearman(density, signed_projection)=+0.8208, n=9500)"
provides:
  - "notebooks/diagnostics/region_partition_mknn_run.py: run_regional_cell (per-region
    MKNN score, region-scoped permutation null, region-scoped bootstrap CI, hubness for
    both embedding sides, undersized/k+1>n_region skip conditions recorded as status:
    undefined) and apply_verdict (VERDICT_RULE applied mechanically) -- the --mode
    regional branch is now fully implemented"
  - "notebooks/.cache/04_region_partition_mknn.jsonl (gitignored): 8 mknn_regional rows,
    each carrying null_scope: region and null_n equal to its own n_region"
  - "notebooks/04_region_partition_mknn.ipynb sections 7-8 executed: the eight-cell
    regional table and per-region CI/floor plot, then the verbatim VERDICT_RULE and its
    mechanical application"
  - "The phase's actual result: VERDICT_RULE HOLDS at every k in {5,10,20,50} including
    HEADLINE_K=20 -- region 1 scores higher, CIs disjoint, exceeds its own 99th-percentile
    null at every k -- reported alongside the D4-14 caveat that this cannot be attributed
    to curvature rather than density by anything in this phase"
affects: [04-06]

tech-stack:
  added: []
  patterns:
    - "run_regional_cell records status:'undefined' with an explicit reason (never a
      silently-dropped or silently-zeroed cell) for either pre-registered skip condition,
      checked before any compute: n_region < MIN_REGION_N and k_mknn + 1 > n_region"
    - "apply_verdict is called directly from the notebook (rpm.apply_verdict) rather than
      reimplemented there -- the verdict logic that ran against the real jsonl on disk is
      the exact code a reader sees applied, not a notebook-local re-derivation"
    - "the regional grid is always region_partition.MKNN_K_GRID regardless of --mknn-k
      (which only affects --mode global) -- the pre-registered grid is not a CLI-tunable
      parameter for the regional path"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/region_partition_mknn_run.py
    - notebooks/04_region_partition_mknn.ipynb

key-decisions:
  - "No deviations from the pre-registered VERDICT_RULE, HEADLINE_K, NULL_QUANTILE,
    CONFIDENCE_LEVEL, N_PERMUTATIONS, N_BOOTSTRAP, or MIN_REGION_N -- every value was read
    from committed source (region_partition.py) and applied mechanically, exactly as
    04-PREREGISTRATION.md requires. No checkpoint was reached in this plan (autonomous:
    true, no checkpoint tasks) -- the standing orchestrator authorization in this session's
    instructions was not exercised because nothing required it."
  - "membership matrices are rebuilt once per function call (mknn_score, permutation_null,
    bootstrap_ci, hubness_skewness each independently call _membership_matrix) rather than
    built once and threaded through all four -- this mirrors run_global_cell's own
    already-committed pattern from plan 04-01 exactly, and 'no k-NN query inside any
    resampling loop' (the actual budget constraint D4-17 names) is satisfied because each
    function builds its matrices ONCE before its own resampling loop starts, never inside
    it. mknn.py is not modified by this plan."

requirements-completed: [MKNN-03, MKNN-04, MKNN-05, MKNN-06, MKNN-07, MKNN-08]

coverage:
  - id: D1
    description: "MKNN-03/04/05: eight regional cells (2 regions x k in {5,10,20,50}),
      each with its own MKNN score, region-scoped permutation null (1000 permutations),
      95% percentile bootstrap CI (1000 resamples), computed entirely within that region's
      own index set -- null_scope='region' and null_n=n_region on every row, no global
      null reused anywhere in the regional branch"
    requirement: MKNN-04
    verification:
      - kind: integration
        ref: "notebooks/diagnostics/region_partition_mknn_run.py --mode regional, plus
          the plan's own <verify> assertion script (8 distinct (region,k) cells; null_scope
          == 'region' on every row; null_n == n_region on every ok row; chance_floor ==
          k/n_region to 1e-12; clears_null == score > null_threshold exactly) run directly
          against notebooks/.cache/04_region_partition_mknn.jsonl"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -q (376 passed, 1 skipped -- unaffected, no
          sealed module touched)"
        status: pass
    human_judgment: false
  - id: D2
    description: "MKNN-06/MKNN-08: the eight-cell grid rendered as one table in notebook
      Section 7 (headline k marked, other three labelled sensitivity), a plot of both
      regions' scores vs k with 95% CI error bars and each region's OWN chance-floor
      reference line, and the hubness caveat substantiated by printed k-occurrence
      skewness for both embedding sides at every cell"
    requirement: MKNN-06
    verification:
      - kind: integration
        ref: "jupyter nbconvert --to notebook --execute --inplace, plus the plan's own
          post-execution assertion script (every code cell has execution_count; the
          pre-registration cell index (28) is strictly less than the first mknn_regional
          cell index (32); execution counts strictly increasing) run directly"
        status: pass
    human_judgment: false
  - id: D3
    description: "MKNN-07: the pre-registered VERDICT_RULE applied mechanically --
      HOLDS at every k including HEADLINE_K=20 (region 1 scores higher, CIs disjoint,
      exceeds its own 99th-percentile null threshold at every k), printed verbatim
      alongside the applied result in both the runner's stdout and notebook Section 8,
      with the D4-14 density caveat carried alongside the verdict rather than in a
      footnote"
    requirement: MKNN-07
    verification:
      - kind: other
        ref: "runner stdout and notebook Section 8 cell output, inspected directly:
          VERDICT_RULE text present verbatim, 'HEADLINE VERDICT (k=20): HOLDS...' line
          present, D4-14 caveat line present immediately after it"
        status: pass
    human_judgment: true
    rationale: "04-VALIDATION.md's Manual-Only table: the verdict rule is pre-registered
      prose applied to CIs; confirming the rule was applied AS WRITTEN rather than
      reinterpreted is a human judgment about scientific conduct, not a computable
      predicate (04-05-PLAN.md flagged_assumptions EDGE/MKNN-07)."

duration: ~25min (dominated by the ~4.5min background compute for the eight-cell grid:
  1000 permutations x 1000 bootstrap resamples x 8 cells, region_0 n=6256 and region_1
  n=3244)
completed: 2026-08-24
status: complete
---

# Phase 4 Plan 5: The eight-cell regional grid and the pre-registered verdict Summary

**VERDICT_RULE HOLDS at every k in {5, 10, 20, 50} including the pre-registered headline
k=20 -- region 1 (n=3244) scores higher than region 0 (n=6256) at every k, the two
regions' 95% bootstrap CIs are disjoint at every k, and region 1's observed score
strictly exceeds its own 99th-percentile permutation-null threshold at every k -- read
alongside the D4-14 caveat that this cannot be attributed to curvature rather than to
regional density by anything in this phase, given region 1's measured median density
(6.5641e6) is ~5,735x lower than region 0's (3.7642e10).**

## Performance

- **Duration:** ~25 min, dominated by the eight-cell grid's background compute
  (~4.5 min wall-clock: region 0 at n=6256 took roughly 3.5x region 1's per-cell cost at
  n=3244, consistent with the O(n^2) membership-matrix construction the plan flagged)
- **Tasks:** 2/2 complete
- **Files modified:** 2

## Accomplishments

- `run_regional_cell` added to `region_partition_mknn_run.py`: computes, per region and
  per `k_mknn` in `region_partition.MKNN_K_GRID`, the MKNN score, a region-scoped
  permutation null (`N_PERMUTATIONS=1000`), a region-scoped 95% bootstrap CI
  (`N_BOOTSTRAP=1000`), and the k-occurrence hubness skewness for both embedding sides --
  every neighbour set, score, null and CI computed entirely within that region's own
  index set (D4-16), no global null reused anywhere. Both pre-registered skip conditions
  (`n_region < MIN_REGION_N`, `k_mknn + 1 > n_region`) are checked before any compute and
  recorded as `status: "undefined"` with an explicit `reason` -- neither fired for this
  split (both regions clear `MIN_REGION_N=500` at every `k` in the grid).
- `apply_verdict` added: reads `HEADLINE_K`, `NULL_QUANTILE`, `CONFIDENCE_LEVEL` from
  `region_partition.py` and applies the frozen `VERDICT_RULE` mechanically -- a `k` HOLDS
  iff the two regions' CIs at that `k` are disjoint AND the higher-scoring region's score
  strictly exceeds its own null threshold. Prints the verbatim rule text alongside the
  applied per-k outcomes and the headline verdict.
- The `--mode regional` branch is now fully implemented: loads the frozen partition
  artifact, subsets both embeddings per region via `keep_idx[labels == region]`, runs all
  eight cells, writes JSONL rows, applies the verdict, and prints the three closing
  statements (D4-14 density caveat, D4-16 not-comparable-to-global, MKNN-08 hubness
  substantiated by printed skewness).
- **The measured result:** every one of the eight cells clears its own region-scoped null
  (`p_value=0.0010` at every cell, the tightest resolvable value at 1000 permutations).
  Region 1's score is roughly double region 0's at every `k` (e.g. `k=20`: region 0
  `8.148%` vs region 1 `17.41%`), and the 95% CIs never overlap. `VERDICT_RULE` therefore
  reads **HOLDS** at every `k`, including the pre-registered `HEADLINE_K=20` alone --
  applied exactly as frozen, with no amendment to any constant now that regional numbers
  exist.
- **The D4-14 caveat travels with the verdict, not in a footnote**, exactly as the
  pre-registered rule's own text requires: region 1 (the higher-scoring region) has a
  measured median density of `6.5641e6` against region 0's `3.7642e10` -- roughly
  5,735x lower, per plan 04-04's Mann-Whitney U test (`statistic=18844954.0, p=0.0`).
  Because MKNN is itself a k-NN statistic and therefore directly density-sensitive by
  construction, and this phase ran no density-matched null, no partial regression, no
  centroid-distance control and no density-matched stratification, **the HOLDS verdict
  above cannot be attributed to curvature rather than to regional density by anything in
  this phase.** This is stated plainly, not softened and not inflated in either
  direction.
- Notebook sections 7-8 appended and executed end to end: Section 7 renders the eight-cell
  table (headline `k` marked, the other three labelled sensitivity) and a plot with both
  regions' CIs as error bars and each region's own chance-floor reference line (a shared
  floor would misrepresent D4-16's own point, since the two regions have different `n`).
  Section 8 reprints the verbatim `VERDICT_RULE` immediately above the applied result
  (via `rpm.apply_verdict`, not reimplemented in the notebook) followed by three markdown
  paragraphs: what the verdict says, the D4-14 caveat with the actual Spearman and
  Mann-Whitney numbers quoted rather than referred to, and provenance (the rule is
  byte-identical to the commit printed in Section 6, fixed before any regional number
  existed). Cells 0-29 (sections 0-6) are source-byte-identical to before this plan; the
  pre-registration cell (index 28) precedes the first `mknn_regional` cell (index 32) in
  both notebook order and execution order.

## Task Commits

Each task was committed atomically:

1. **Task 1: The eight-cell regional grid with region-scoped nulls and CIs** -
   `647d01d` (feat)
2. **Task 2: Notebook sections 7-8 -- the regional grid and the verdict** -
   `bec5cc0` (feat)

## Files Created/Modified

- `notebooks/diagnostics/region_partition_mknn_run.py` -- `run_regional_cell`,
  `_regional_row_print`, `apply_verdict`, `_print_regional_closing_statements`, and the
  completed `--mode regional` branch in `main()`
- `notebooks/04_region_partition_mknn.ipynb` -- sections 7-8 appended and executed
- `notebooks/.cache/04_region_partition_mknn.jsonl` (gitignored) -- 8 new `mknn_regional`
  rows appended (6 pre-existing `mknn_global` rows from plan 04-01 untouched)

## Decisions Made

- **No amendment to any pre-registered constant.** `VERDICT_RULE`, `HEADLINE_K=20`,
  `NULL_QUANTILE=0.99`, `CONFIDENCE_LEVEL=0.95`, `N_PERMUTATIONS=1000`, `N_BOOTSTRAP=1000`,
  and `MIN_REGION_N=500` were all read from committed `region_partition.py` and applied
  exactly. No checkpoint existed in this plan to ratify under standing authorization
  (`autonomous: true`, no `checkpoint:*` tasks) -- the session's standing authorization for
  unattended checkpoints was available but not exercised, because no checkpoint was
  reached.
- **`run_regional_cell` mirrors `run_global_cell`'s existing membership-matrix pattern
  exactly** (each of `mknn_score`, `permutation_null`, `bootstrap_ci`,
  `hubness_skewness` independently calls `_membership_matrix`, rather than building the
  matrix once and threading it through all four) -- this is the already-committed pattern
  from plan 04-01, and `mknn.py` is not modified by this plan. D4-17's actual constraint
  ("no k-NN query inside any resampling loop") is satisfied: each function's own
  membership matrix is built once, before that function's own resampling loop, never
  inside it.
- **The regional grid always uses `region_partition.MKNN_K_GRID`**, never the CLI's
  `--mknn-k` flag (which only affects `--mode global`) -- the pre-registered grid is not a
  tunable parameter for the regional path, by construction rather than by convention.

## Deviations from Plan

None -- plan executed exactly as written. No Rule 1/2/3 auto-fixes were needed; the
implementation matched the plan's `<action>` specification directly and both automated
`<verify>` scripts passed on the first run.

## Issues Encountered

None. Exactly one background process was launched for the regional grid (confirmed via
`pgrep` before and after launch, per this phase's standing duplicate-launch caution), ran
to completion in ~4.5 minutes, and no stray process was ever running.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

- **Plan 04-06 inherits a sealed regional verdict: HOLDS at every k, headline included,**
  with the density confound as its permanent, undissociable caveat. 04-06 is this phase's
  record/closure plan -- it should carry the HOLDS verdict and the D4-14 caveat forward
  together, never one without the other, and should not attempt retroactive control for
  density (D4-14 explicitly declines every control this phase could have run).
- **All eight regional cells are `status: "ok"`** -- no cell was undefined on size or
  `k+1 > n_region` grounds for this split, so 04-06 does not need to special-case an
  undefined cell in its phase record.
- **The three accepted gaps this pre-registration sits on top of are unchanged and
  unclosed**, restated once more for 04-06's inheritance: the curvature field itself is
  unvalidated on real data (no ground truth for PU); the direction-partition's
  codimension gap is unmeasured at PU's ~748-wide normal space; and the density confound
  is reported, never controlled. None of these three gaps is affected by this plan's
  HOLDS result -- a HOLDS verdict under the pre-registered rule is not evidence against
  any of them.
- No blockers for 04-06.

---
*Phase: 04-region-partitioning-regional-alignment-mknn*
*Completed: 2026-08-24*
