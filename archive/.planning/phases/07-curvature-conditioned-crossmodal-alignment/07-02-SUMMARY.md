---
phase: 07-curvature-conditioned-crossmodal-alignment
plan: 02
subsystem: research-instrumentation
tags: [mknn, curvature, spearman, permutation-test, tdd, pytest, tracer]

# Dependency graph
requires:
  - phase: 07-curvature-conditioned-crossmodal-alignment
    provides: "crossmodal_curvature.py's frozen constants block, VERDICT_RULE, VERDICT_VALUES, assert_preregistered(), the freeze commit f032745f6450068c63763993d39fa112fd36bb8c (plan 07-01)"
provides:
  - "crossmodal_curvature.split_indices, per_point_mknn, two_tailed_permutation_null, apply_verdict -- the four compute functions the rest of Phase 7 is built on"
  - "notebooks/diagnostics/07_crossmodal_curvature_run.py -- the single runner proving data -> fit -> curvature -> per-point MKNN -> both permutation tails -> verdict end to end on real PU rows"
  - "22 new tests pinning per_point_mknn against mknn.mknn_score's mean, row alignment under a shared permutation, all four degenerate-input guards, the relative-precision distinct-value bound, two_tailed_permutation_null's direction/clears_either/observed_rho contract, the explicit single-one-sided-call regression documenting the closed defect, apply_verdict's four terminal outcomes plus both malformed-key ValueError paths, and split_indices' shape/disjointness/coverage/determinism -- 140 tests total in test_crossmodal_curvature.py (118 -> 140)"
affects: [07-03-plan, 07-04-plan, 07-05-plan]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Compute functions added strictly below the freeze block, in a new import group (from typing import Tuple; import numpy; from . import curvature_probe, mknn) never merged into the frozen module's original 'from typing import Any, Dict' line -- keeps the freeze commit's own diff untouched by later additions"
    - "Thread-cap-before-torch-import: OMP_NUM_THREADS/MKL_NUM_THREADS/NUMEXPR_NUM_THREADS set from raw sys.argv parsing above every numpy/torch/pu_manifold import, plus torch.set_num_threads after import -- measured fix for a ~10x contention slowdown from concurrent unthrottled torch jobs on this 20-core machine"
    - "Relative-precision distinct-value counting (divide by max abs value, round to 12 decimals, then np.unique) rather than raw float equality -- the 05-02-SUMMARY.md retraction (5,301/9,852 reported vs. 4/3 true) is the cautionary precedent, restated in code comments at every count site"

key-files:
  created:
    - notebooks/diagnostics/07_crossmodal_curvature_run.py
  modified:
    - notebooks/pu_manifold/crossmodal_curvature.py
    - notebooks/pu_manifold/tests/test_crossmodal_curvature.py

key-decisions:
  - "Task 1 tracer feedback gate: human reviewed the smoke run's printed output (--selfcheck 6/6, smoke exit 0 in 139.9s, var_explained=0.7567, 16 distinct per-point MKNN values, observed_rho=-0.3282, negative tail cleared / positive tail did not) and confirmed with 'Confirm -- proceed to Task 2'. The printed 'ASSOCIATION DETECTED' verdict was explicitly understood by all parties as a stubbed smoke-path artifact (all three D_SWEEP keys mapped from the one measured d=20 boolean, positive_control_cleared_at stubbed), not a Phase 7 finding."
  - "Task 2's anti-correlated test fixture is built exactly as the plan specifies: h from a uniform draw, m = floor(k * (1 - (rankdata(h) - 0.5) / n)) -- a monotone, discretized function of h's rank -- so the pair mimics the real per-point MKNN array's tie-dense discretization rather than a smooth synthetic correlation."
  - "The one-sided-defect regression test calls curvature_probe.permutation_null directly (not through crossmodal_curvature) on the same anti-correlated fixture used for the two-tailed test, asserting clears_null is False -- this is the test that would have caught the original one-sided defect had it existed at freeze time."

requirements-completed: [D7-01, D7-04, D7-05, D7-06]

coverage:
  - id: D1
    description: "End-to-end tracer: one command carries a real 800-row PU subsample through data load, decoder fit, curvature field, per-point MKNN, both permutation tails, and verdict application, printing a terminal VERDICT_VALUES member and writing nothing to the frozen record"
    requirement: "D7-01, D7-05, D7-06"
    verification:
      - kind: manual_procedural
        ref: ".venv/bin/python notebooks/diagnostics/07_crossmodal_curvature_run.py --mode smoke --smoke-rows 800 --smoke-permutations 50 -- verified at the Task 1 tracer feedback checkpoint and independently re-run this session with identical output"
        status: pass
    human_judgment: true
    rationale: "A tracer proving a real end-to-end research pipeline against live PU data is a human-verified gate by design (the plan's own type=\"tracer\" feedback loop), not a unit-testable claim."
  - id: D2
    description: "per_point_mknn, two_tailed_permutation_null, apply_verdict, split_indices pinned against the sealed functions they compose or mirror -- mean-agreement, row alignment, all degenerate-input guards, relative-precision distinct-value bound, direction/clears_either correctness, the closed one-sided defect, all four verdict outcomes plus malformed-key errors, and split determinism"
    requirement: "D7-04"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_crossmodal_curvature.py -- 22 new tests (140 total), all pass in 2.89s"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -- full suite, 556 passed / 1 skipped (534 -> 556, delta matches the 22 new tests, no regressions), 164s"
        status: pass
    human_judgment: false
  - id: D3
    description: "No sealed module or frozen constant altered since the freeze commit; crossmodal_curvature.py's compute-function additions land strictly below the frozen constants block"
    requirement: "D7-05, D7-06"
    verification:
      - kind: other
        ref: "git diff --stat f032745..HEAD -- mknn.py linear_probe.py pointcloud_probe.py cae.py decoder_curvature.py curvature_probe.py cross_split_curvature.py src/effdim/ -- empty output"
        status: pass
    human_judgment: false

duration: 27min
completed: 2026-08-26
status: complete
---

# Phase 7 Plan 2: Wire and Prove the Curvature-Crossmodal Pipeline End to End Summary

**One runner (`07_crossmodal_curvature_run.py`) proves the whole Phase 7 pipeline -- data load, decoder fit, curvature field, per-point MKNN, two-tailed permutation significance, verdict -- on real PU rows in under three minutes, and the D7-04 per-point MKNN gap-fill is pinned against `mknn.mknn_score`'s mean by 22 regression tests, including the explicit test that would have caught the original one-sided-permutation defect.**

## Performance

- **Duration:** 27 min (this continuation session, Task 2 only; Task 1 was executed and verified in a prior session)
- **Started:** 2026-08-26T12:20:00Z (approx.)
- **Completed:** 2026-08-26T12:47:00Z (approx.)
- **Tasks:** 2 (Task 1 completed and verified at its tracer feedback gate in a prior session; Task 2 executed this session)
- **Files modified:** 1 (test file only, this session); 2 total across the plan

## Accomplishments

- **Task 1 (prior session, verified this session's gate):** `notebooks/pu_manifold/crossmodal_curvature.py` gained `split_indices`, `per_point_mknn`, `two_tailed_permutation_null`, `apply_verdict` -- pure compute functions added strictly below the frozen constants block, zero frozen-constant edits. `notebooks/diagnostics/07_crossmodal_curvature_run.py` was created: `--selfcheck` (6 assertions, no PU data, no torch training), `--mode smoke` (the tracer path -- 800 rows, d=20, max_epochs=2, 50 permutations), and stub `--mode dsweep` / `--mode positive-control` raising `NotImplementedError` naming the plans that fill them (07-04, 07-03).
- **Tracer re-verified this session** (independent re-run, not a re-execution of Task 1's work): `--mode smoke --smoke-rows 800 --smoke-permutations 50` exits 0 in 138.1s and reproduces the checkpoint's numbers exactly -- `var_explained=0.7567`, `cond(g)` median `2.6605e+01`, `per_point_mknn` 16 distinct values (<= `HEADLINE_K + 1 = 21`), `observed_rho=-0.3282`, negative tail clears (`threshold=0.0459`), positive tail does not (`threshold=0.0464`), printed `VERDICT (smoke -- all three d stubbed ...): ASSOCIATION DETECTED`. `ls notebooks/.cache/07_crossmodal_curvature.jsonl` confirms smoke wrote nothing.
- **Task 2 (this session): 22 new tests** in `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` (118 -> 140 total), covering:
  - `per_point_mknn(z1, z2, k).mean() == mknn.mknn_score(z1, z2, k)` to `pytest.approx` on a `(400, 16)` Gaussian fixture -- the D7-04 mean-agreement pin.
  - `per_point_mknn(z, z, k)` all-ones; independent-cloud mean within a factor of 3 of `mknn.chance_floor`.
  - Row alignment: applying one shared permutation to both `z1` and `z2` permutes the output by that same permutation.
  - All four degenerate-input guards: mismatched row counts, a non-finite entry, `n < 2`, `k + 1 > n` -- each raising `ValueError` with a message naming what was wrong, inherited by composition from `mknn._membership_matrix`.
  - Distinct-value count bounded by `k + 1`, counted at relative precision (divide by max abs value, round to 12 decimals) rather than raw float equality.
  - `two_tailed_permutation_null`: `direction == "negative"` and `clears_either is True` on a strongly anti-correlated, tie-discretized fixture; `"positive"` on the mirrored correlated fixture; `clears_either is False` on an independent pair; `observed_rho` equals `scipy.stats.spearmanr(h, m).statistic` and the negative tail's `observed_rho` is its exact negation.
  - **The regression that documents the closed defect:** a single, un-mirrored `curvature_probe.permutation_null(h, m, ...)` call on the same anti-correlated fixture asserts `clears_null is False` -- proving the one-sided test as written cannot detect a negative association, which is exactly why `two_tailed_permutation_null` exists.
  - `apply_verdict`: all four `VERDICT_VALUES` reached from the corresponding input (`ASSOCIATION DETECTED`, `NO DETECTABLE RELATIONSHIP`, `UNDERPOWERED -- NO CLAIM`, `SPLIT ACROSS d`), plus `ValueError` on a partial-sweep key set and on an extra key outside `D_SWEEP`.
  - `split_indices(10000, SPLIT_SEED, HOLDOUT_FRACTION)`: 8000 train / 2000 holdout, disjoint, covering `range(10000)`, deterministic across two calls.
- **Full regression confirmed:** `notebooks/pu_manifold/tests/` -- 556 passed, 1 skipped (up from the pre-session baseline of 534 passed, 1 skipped; the +22 delta matches exactly the 22 new tests), 164s, zero regressions.
- **No sealed module touched:** `git diff --stat f032745..HEAD` across `mknn.py`, `linear_probe.py`, `pointcloud_probe.py`, `cae.py`, `decoder_curvature.py`, `curvature_probe.py`, `cross_split_curvature.py`, `src/effdim/` is empty for the whole plan.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end "one d, one verdict" tracer (D7-01, D7-04, D7-05)** -- `031cc4a` (feat) -- prior session; stopped at the tracer feedback checkpoint
2. **Task 2: Pin the D7-04 gap-fill against the sealed function it re-composes** -- `129de14` (test) -- this session

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `notebooks/pu_manifold/crossmodal_curvature.py` -- Task 1 added `split_indices`, `per_point_mknn`, `two_tailed_permutation_null`, `apply_verdict` below the frozen constants block (unchanged this session).
- `notebooks/diagnostics/07_crossmodal_curvature_run.py` -- Task 1's new runner: `load_pu_pair`, `fit_and_field`, `append_record_row`, `--selfcheck`, `--mode smoke`, thread-cap bootstrap (unchanged this session).
- `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` -- this session's 256-line addition: 22 tests pinning the four Task 1 compute functions.

## Decisions Made

See `key-decisions` in frontmatter above. In short: the tracer feedback gate was resolved `Confirm -- proceed to Task 2` with the smoke verdict explicitly understood as a stubbed artifact, not a finding; the anti-correlated test fixture follows the plan's exact discretized-rank construction; the one-sided-defect regression calls `curvature_probe.permutation_null` directly rather than through a wrapper, so it fails loudly if the defect it documents is ever reintroduced.

## Deviations from Plan

None -- plan executed exactly as written. Task 1's compute functions and runner (prior session) matched the plan's specification; Task 2's tests (this session) implement one test per listed `<behavior>` entry, using the exact fixture constructions and precision conventions the plan specifies (relative-precision distinct-value counting, `n_resamples=199` in tests rather than the frozen `N_PERMUTATIONS`, `quantile_per_tail=cc.NULL_QUANTILE_PER_TAIL` passed explicitly).

## Issues Encountered

None. The full `notebooks/pu_manifold/tests/` suite takes ~2m45s (556 tests, mostly torch-backed); this was run once as a final regression check after Task 2's commit and is not part of Task 2's own `<verify>` step (which targets only `test_crossmodal_curvature.py`, completing in 2.89s).

## Known Stubs

None introduced by this plan's code. The smoke path's `ASSOCIATION DETECTED` printout is a documented, intentional stub (all three `D_SWEEP` keys mapped from the single measured d=20 boolean, `positive_control_cleared_at` stubbed) -- explicitly not a Phase 7 finding, and the runner's own printed line states this ("SMOKE MODE: writes nothing"). `--mode dsweep` and `--mode positive-control` raise `NotImplementedError` naming plans 07-04 and 07-03, which is the plan's stated scope boundary, not an unplanned gap.

## Threat Flags

None. `T-07-01` (record-path traversal), `T-07-02` (frozen-constant tampering), `T-07-03` (record repudiation), `T-07-04` (uncapped torch thread pools) were all mitigated in Task 1 (prior session) and are unchanged by this session's test-only addition.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

Plan 07-03 (positive control) and plan 07-04 (the real d-sweep) may now begin: the four compute functions they call (`split_indices`, `per_point_mknn`, `two_tailed_permutation_null`, `apply_verdict`) are proven end to end on real PU data and pinned against the sealed function each re-composes. The runner's `--mode dsweep` and `--mode positive-control` stubs are in place, raising `NotImplementedError` until those plans fill them. No PU number has been written to the frozen record anywhere in the tree -- `notebooks/.cache/07_crossmodal_curvature.jsonl` still does not exist.

---
*Phase: 07-curvature-conditioned-crossmodal-alignment*
*Completed: 2026-08-26*

## Self-Check: PASSED

- FOUND: `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-02-SUMMARY.md`
- FOUND: `notebooks/pu_manifold/tests/test_crossmodal_curvature.py`
- FOUND: `notebooks/pu_manifold/crossmodal_curvature.py`
- FOUND: `notebooks/diagnostics/07_crossmodal_curvature_run.py`
- FOUND commit `031cc4a` in `git log --oneline --all`
- FOUND commit `129de14` in `git log --oneline --all`
