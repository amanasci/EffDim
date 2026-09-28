---
phase: 04-region-partitioning-regional-alignment-mknn
plan: 01
subsystem: manifold-alignment
tags: [mknn, crossmodal, hubness, permutation-test, bootstrap-ci, jupyter, k-nn]

# Dependency graph
requires:
  - phase: 01-data-and-embeddings
    provides: subsample_*.npz carrying row-aligned hsc/legacysurvey embedding columns
provides:
  - "mknn.py: mknn_score, permutation_null, bootstrap_ci, hubness_skewness, chance_floor,
    all on a shared k-NN membership-matrix architecture built once per (side, k) cell"
  - "region_partition_mknn_run.py: load_pu_pair, run_global_cell, selfcheck, --mode global
    across the full k grid {5, 10, 20, 50}, summarize() with paper-comparison framing"
  - "notebooks/.cache/04_region_partition_mknn.jsonl: 6 global rows (2 duplicate k=10 rows
    from Task 1's tracer, 4 distinct k in {5,10,20,50} from Task 2's grid run)"
  - "notebooks/04_region_partition_mknn.ipynb: executed 3-section read-out notebook"
affects: [04-03, 04-04, 04-05, 04-06]

tech-stack:
  added: []
  patterns:
    - "k-NN membership matrix built once per (side, k), reused by score/permutation-null/
      bootstrap-CI/hubness-skewness -- no k-NN query inside any resampling loop"
    - "permutation_type=\"pairings\" passed explicitly to scipy.stats.permutation_test"
    - "method=\"percentile\" passed explicitly to scipy.stats.bootstrap, resampling points
      (never the pairing) from a precomputed per-point overlap array"
    - "runner + JSONL cache + notebook: all compute lives in the runner, the notebook reads
      the cache and computes nothing heavy"
    - "diagnostics/ imported into a notebook as an implicit namespace package (no __init__.py
      needed) once notebooks/ itself is on sys.path"

key-files:
  created:
    - notebooks/04_region_partition_mknn.ipynb
  modified:
    - notebooks/pu_manifold/mknn.py
    - notebooks/diagnostics/region_partition_mknn_run.py

key-decisions:
  - "Ratio-over-chance carries the paper comparison, not the raw percentage (D4-19): all four
    raw MKNN scores at n=10,000 fall OUTSIDE the paper's 0.34%-2.25% band at n=101,725, but
    every k clears its chance floor by 26x-98x -- the finding is reported as an explicit n
    mismatch plus a strong, monotonically-narrowing ratio, not papered over as agreement"
  - "MKNN-08's hubness caveat is substantiated by the printed k-occurrence skewness (Radovanovic
    et al. JMLR 2010) at every reported result, never asserted as prose alone -- both the runner's
    summarize() and the notebook's Section 2 print it beside every score"
  - "The hubness-range text is computed from the data (min/max across all k and both sides) in
    both the runner and the notebook, not a hardcoded '~1.1-1.3' guess -- caught and fixed after
    the k=50 run measured hub_hsc=0.966, outside that initial estimate"
  - "D4-12's Swiss-roll-rule non-applicability is stated explicitly in the notebook: MKNN has no
    latent space, no training, no reconstruction; the sign-split partition in 04-03 reads an
    already-computed field rather than recovering manifold structure; the estimator underneath
    is separately covered by notebooks/02.5_swiss_roll_curvature_probe_check.ipynb"

requirements-completed: [MKNN-01, MKNN-02, MKNN-05, MKNN-08]

coverage:
  - id: D1
    description: "mknn_score/permutation_null/bootstrap_ci/hubness_skewness/chance_floor
      implemented on a shared membership-matrix architecture, matching the origin paper's
      formula, with known-answer sanity checks passing"
    requirement: MKNN-01
    verification:
      - kind: unit
        ref: "notebooks/diagnostics/region_partition_mknn_run.py --selfcheck"
        status: pass
      - kind: e2e
        ref: "notebooks/04_region_partition_mknn.ipynb Section 1 (inline known-answer checks)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Global crossmodal HSC-vs-Legacy-Survey MKNN reproduced across k in
      {5, 10, 20, 50} and reported against the paper's 0.34%-2.25% range with the k/n chance
      floor and the n mismatch (10,000 vs 101,725) stated explicitly"
    requirement: MKNN-02
    verification:
      - kind: e2e
        ref: "region_partition_mknn_run.py --mode global (stdout contains 0.34, 2.25, 101725)"
        status: pass
      - kind: e2e
        ref: "notebooks/04_region_partition_mknn.ipynb Section 2"
        status: pass
    human_judgment: true
    rationale: "Whether the ratio-over-chance framing (vs. the raw band comparison) is the
      right way to read this result against the paper is a judgment call about scientific
      interpretation, not a mechanical pass/fail -- flagged in the plan's must_haves as an
      explicit backstop verification."
  - id: D3
    description: "Bootstrap 95% CI machinery (percentile method, resampling points from a
      fixed per-point overlap array) exercised on every global cell"
    requirement: MKNN-05
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -q (369 passed, 1 skipped)"
        status: pass
      - kind: e2e
        ref: "notebooks/.cache/04_region_partition_mknn.jsonl (ci_low/ci_high on every row)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Hubness caveat stated and substantiated by k-occurrence skewness beside
      every MKNN result, both in the runner's stdout and the notebook"
    requirement: MKNN-08
    verification:
      - kind: e2e
        ref: "region_partition_mknn_run.py --mode global stdout (hub_hsc/hub_ls columns)"
        status: pass
      - kind: e2e
        ref: "notebooks/04_region_partition_mknn.ipynb Section 2 hubness cell"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-08-24
status: complete
---

# Phase 4 Plan 1: Global crossmodal MKNN — mknn.py, the runner, the full k grid, and the read-out notebook Summary

**Implemented `mknn.py`'s shared k-NN-membership-matrix architecture and reproduced the global
HSC-vs-Legacy-Survey MKNN across k in {5, 10, 20, 50}: raw scores fall outside the origin
paper's 0.34%-2.25% band at n=10,000 (vs. the paper's n=101,725), but each clears its k/n
chance floor by 26x-98x, with the ratio-over-chance framing (D4-19) and the MKNN-08 hubness
caveat (skewness 0.966-1.494) carrying every reported number.**

## Performance

- **Duration:** ~15 min this session (Task 2 only; Task 1 ran in a prior session, see its own
  commit `bc60f38` and the earlier checkpoint). Total plan wall-clock spans two sessions
  (2026-08-23 15:00 -> 2026-08-24 03:16 local, including the human-verify pause between them).
- **Started (this session):** 2026-08-24T07:16Z (resume)
- **Completed:** 2026-08-24T07:16Z region of commit `c8dce0c`
- **Tasks:** 2/2 complete
- **Files modified:** 3 (2 code files touched across both tasks, 1 notebook created)

## Accomplishments
- `mknn.py`'s three stubs (`mknn_score`, `permutation_null`, `bootstrap_ci`) filled, plus
  `_membership_matrix`, `hubness_skewness`, `chance_floor` — all built on one shared
  `(n, n)` boolean k-NN membership matrix per (side, k), computed once and reused everywhere.
- `region_partition_mknn_run.py` created and extended: `load_pu_pair`, `run_global_cell`,
  `selfcheck`, `--mode global` (looping the full `{5, 10, 20, 50}` grid), and a `summarize()`
  read-out with the paper-comparison block.
- Global crossmodal MKNN reproduced across all four k values: 4.882% / 6.594% / 8.980% /
  13.23% at k=5/10/20/50, each with a permutation p-value of 0.000999, a 95% bootstrap CI,
  and both sides' hubness skewness.
- `notebooks/04_region_partition_mknn.ipynb` created and executed end to end (5 code cells,
  all non-null `execution_count`, no errors): provenance, MKNN-01 known-answer checks plus
  the Swiss-roll-rule non-applicability statement (D4-12), and MKNN-02's global reproduction
  with a plot and a three-part read-out.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end global crossmodal MKNN — one k, one path, real number** - `bc60f38` (feat)
2. **Task 2: Full global k grid, chance-floor framing, and the notebook read-out** - `c8dce0c` (feat)

_Task 1 was a `type="tracer"` task; its `<verify>` steps all passed and it was confirmed at a
blocking human-verify checkpoint (human replied "verified") before this continuation resumed
at Task 2. No concerns were raised about the Task 1 numbers._

## Files Created/Modified
- `notebooks/pu_manifold/mknn.py` - MKNN score, permutation null, bootstrap CI, hubness
  skewness, chance floor, all on the shared membership-matrix architecture (Task 1)
- `notebooks/diagnostics/region_partition_mknn_run.py` - two-column PU loader, global MKNN
  pass across the full k grid, self-check, JSONL cache append, paper-comparison summary (Task 1
  created it; Task 2 extended `summarize()`)
- `notebooks/04_region_partition_mknn.ipynb` - new, executed read-out notebook, sections 0-2
  (Task 2)

## Decisions Made
- **Ratio-over-chance carries the paper comparison (D4-19), not the raw number.** All four raw
  scores land outside the paper's 0.34%-2.25% band, but the `n` mismatch (10,000 vs 101,725)
  makes a raw-number comparison meaningless; the read-out states both the raw values and their
  distance from chance, explicitly, rather than resolving "inside/outside the band" as if it
  settled anything.
- **Hubness caveat is substantiated, not asserted.** Every printed MKNN result (runner stdout
  and notebook) carries the k-occurrence skewness for both embedding sides beside it, per the
  plan's explicit prohibition against reporting MKNN without that substantiation.
- **Hubness-range text computed from data, not hardcoded.** Caught during execution: an early
  draft of the caveat text guessed "~1.1-1.3" from the k=10 tracer row alone; the full grid's
  k=50 cell measured `hub_hsc=0.966`, outside that guess. Both the runner and the notebook now
  compute `min`/`max` over all k and both sides at print time.
- **D4-12's Swiss-roll-rule non-applicability is stated in the phase's own words**, in a
  notebook markdown cell, naming `notebooks/02.5_swiss_roll_curvature_probe_check.ipynb` as the
  estimator's existing coverage, per the plan's explicit requirement that this reasoning not be
  silently omitted.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Hardcoded hubness-skewness range corrected to a data-derived range**
- **Found during:** Task 2, after the full k-grid run completed
- **Issue:** The runner's `summarize()` caveat text was drafted with a fixed "~1.1-1.3"
  hubness-skewness estimate (matching only the k=10 tracer row from Task 1). The full grid run
  measured `hub_hsc=0.966` at k=50 and `hub_ls=1.494` at k=5 — both outside that hardcoded
  range, which would have understated the caveat.
- **Fix:** Replaced the hardcoded text in both `region_partition_mknn_run.py`'s `summarize()`
  and the notebook's Section 2 read-out cell with a `min`/`max` computed over all k and both
  embedding sides at print time.
- **Files modified:** `notebooks/diagnostics/region_partition_mknn_run.py`,
  `notebooks/04_region_partition_mknn.ipynb`
- **Verification:** Re-ran the runner and re-executed the notebook; both now print
  "0.966 to 1.494", matching the JSONL cache's recorded values exactly.
- **Committed in:** `c8dce0c` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix keeps the caveat's substantiating number honest against the data
actually measured; no scope creep, no architecture change.

## Issues Encountered
None beyond the one auto-fixed deviation above. The full k-grid permutation/bootstrap run took
~7.5 minutes wall-clock at n=10,000 (four k values, 1000 permutations + 1000 resamples each),
consistent with the plan's ~140s-per-k estimate; run in the background and polled rather than
shrinking the pre-registered grid or resample counts.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The membership-matrix architecture (`mknn.py`) and the runner's `run_global_cell` shape are
  proven at the worst-case memory footprint (n=10,000, dense `(n,n)` boolean matrices per side)
  and are ready for plan `04-03` onward to reuse for regional cells.
- `notebooks/04_region_partition_mknn.ipynb` sections 0-2 are complete; plans `04-04`/`04-05`/
  `04-06` extend it additively with regional sections, per the phase's artifact map.
- No blockers. The pre-registered partition freeze (plan `04-03`) is unaffected — this plan
  computed no regional number and read no region label, per its own ordering note.

---
*Phase: 04-region-partitioning-regional-alignment-mknn*
*Completed: 2026-08-24*

## Self-Check: PASSED
- FOUND: notebooks/pu_manifold/mknn.py
- FOUND: notebooks/diagnostics/region_partition_mknn_run.py
- FOUND: notebooks/04_region_partition_mknn.ipynb
- FOUND: commit bc60f38
- FOUND: commit c8dce0c
