---
phase: 08-curvature-conditioned-cka-alignment
plan: 05
subsystem: testing
tags: [numpy, cka, hsic, permutation-null, runtime-budget, blocked]

# Dependency graph
requires:
  - phase: 08-04
    provides: "The D8-22 freeze commit (816863cae2209261470d1d041dcc4484a3056947), all 45
      pre-registered constants filled, FREEZE_COMMIT_SHA wired into the runner's strict-ancestor
      gate"
provides:
  - "notebooks/diagnostics/08_cka_alignment_run.py's compute_density, plant_alignment_degradation,
    run_cell, run_positive_control, shuffle_h_field, run_negative_control, run_sweep -- all three
    production modes IMPLEMENTED and verified correct at small scale and against real PU data at a
    single-cell scope, but NOT executed to completion at the frozen production scale"
  - "A measured, reproducible runtime-cost finding: one full label-permutation null
    (N_PERMUTATIONS=1000) at PU's real ~3,333-point pooled tertile subset size costs ~2.14 hours
    of wall-clock on this machine (8 threads) -- about 500-750x RESEARCH.md's un-piloted 'tens of
    milliseconds' estimate -- making the plan's 129 required full-null computations
    (21 + 90 + 18) cost an estimated ~276 hours (~11.5 days) of continuous compute, not the '1-2
    hours' the plan budgeted for D8-18/19 alone"
affects: [08-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "run_cell(h, density, K_full, L_full, s_strata, n_permutations, seed) as the one shared
      unit all three production modes call -- density_strata -> tertile_split_within_strata ->
      tertile_gap_panel (all kernels) -> stratified_tertile_label_null (NULL_KERNELS only) ->
      null_threshold, returning a flat per-kernel dict of plain Python scalars"
    - "Modality A's Gram matrices built directly via cka.linear_gram/cka.rbf_gram (not through
      build_gram_matrices, which always builds both modalities) so run_positive_control never
      pays for a modality-B Gram build it would immediately discard"
    - "In-process monkeypatch of frozen constants (N_PERMUTATIONS, S_GRID, PLANTED_EFFECT_GRID)
      for CORRECTNESS VERIFICATION ONLY, against real PU data, writing to a throwaway scratch
      path, never the production record path, never claimed as a deliverable -- cka.py itself was
      never edited on disk"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/08_cka_alignment_run.py

key-decisions:
  - "Did NOT run any of the three production CLI modes (--mode positive-control/negative-control/
    sweep) to completion against real PU data. Direct empirical measurement (see Performance)
    established that full completion requires ~276 hours of continuous compute -- infeasible
    within any interactive execution session and far beyond the plan's own '1-2 hour' budget for
    D8-18/19. Per the plan's own explicit instruction ('If a run genuinely cannot complete, stop
    and report rather than trimming it'), this plan halts here rather than (a) starting a
    multi-day unattended computation with no ability to supervise or safely interrupt it without
    violating 'do NOT kill a run early', or (b) reducing N_PERMUTATIONS/N_REPEATS/S_GRID/
    PLANTED_EFFECT_GRID to fit a session, which D8-22 names a pre-registration BREACH."
  - "Modality A = hsc (fixed throughout), modality B = legacysurvey (the one D8-18 degrades) --
    a discretion choice; the plan does not name which modality is 'A' vs 'B', only that the
    injection touches one modality only. Documented here since a future reader implementing 08-06
    or a fresh pre-registration needs this fixed."
  - "run_sweep's pooled_field_guard call is implemented as a self-proving assertion (call it with
    all three seed field names, expect RuntimeError, re-raise AssertionError if it does NOT raise)
    rather than a live gate in the per-seed loop -- the guard's own contract is 'raises for more
    than one field', so calling it with the three seed names IS the demonstration that a pooled
    path would be refused, not a step the per-seed loop needs to pass through."
  - "One suspected implementation inefficiency in cka.py is reported here, per the plan's own
    'stop and report a genuine bug, do not fix it' instruction (cka.py is off-limits to this
    plan): unbiased_hsic's term1 = np.trace(Kt @ Lt) computes a full O(n^3) dense matrix product
    to extract a trace value that is mathematically identical to np.sum(Kt * Lt.T), an O(n^2)
    computation. At the ~3,333-point pooled tertile size D8-08 establishes as S-independent, this
    is very likely the dominant driver of the measured per-resample cost. This is NOT a
    correctness bug (both forms are provably equal) and was NOT fixed -- cka.py was not touched."

patterns-established: []

requirements-completed: []
# NONE of D8-01/04/06/07/09/10/11/12/13/14/15/18/19/22 are marked complete here: the code that
# WOULD satisfy them exists, but no production run executed to completion, so no Phase 8 number
# exists to verify any of D8-09/12/13/15/18/19 against. D8-01/04/06/07/10/11/14/22 are structural
# (kernel choice, HSIC form, stratification, density convention, statistic, null construction,
# field provenance, freeze discipline) and are satisfied by the CODE as written, but this SUMMARY
# does not claim the requirement "complete" while the plan's own <verify> blocks never ran.

coverage: []
# Coverage block intentionally empty (not omitted) -- see requirements-completed note above. Every
# deliverable in this plan requires a completed production run to verify against its acceptance
# criteria (JSONL row counts, detection floors, false-positive rates, per-d verdicts), and no
# production run completed. Routing this to a human via the fallback prose path is correct: a
# human (the developer) must decide how to proceed given the runtime finding below.

# Metrics
duration: ~50min
completed: 2026-08-28
status: blocked
---

# Phase 08 Plan 05: CKA Production Modes -- Implemented, Blocked on a ~276-Hour Runtime Discovery

**All three production modes (`--mode positive-control`, `--mode negative-control`, `--mode
sweep`) are implemented and verified correct on synthetic data and against real PU data at reduced
scope, but were NOT run to completion: direct empirical measurement shows the frozen
pre-registration constants require an estimated ~276 hours (~11.5 days) of continuous compute --
roughly 500-750x the plan's own un-piloted "1-2 hour" cost estimate -- so no Phase 8 number exists
in `notebooks/.cache/08_cka_alignment.jsonl` (the file does not exist) and this plan halts for a
developer decision rather than trimming a frozen constant or starting a multi-day unattended run.**

## Performance

- **Duration:** ~50 min (implementation + verification + this report; no production run attempted
  to completion)
- **Started:** 2026-08-28T03:30:00Z (approx, following 08-04's close)
- **Completed:** 2026-08-28T04:15:00Z (approx)
- **Tasks:** 0/3 verified complete against their own `<verify>` acceptance criteria (all 3 tasks'
  code is written; see Deviations / Issues below for why none was run to completion)
- **Files modified:** 1 (`notebooks/diagnostics/08_cka_alignment_run.py`)

## The runtime-cost finding, in full precision

This is the substantive result of this session and the reason the plan halts here. Two
independent, direct measurements converge on the same number:

**Measurement 1 -- isolated null timing, production-representative synthetic data.**
`cka.stratified_tertile_label_null` called directly on `(10000, 10000)` float32 Gram matrices for
`NULL_KERNELS = ("linear", "rbf_sigma")`, `S=20` density strata (pooled tertile size ~3,333,
matching D8-08's own S-independent measured fact), under the runner's default
`OMP_NUM_THREADS=8`/`MKL_NUM_THREADS=8`:

- 5 resamples: **38.35674264805857 s** wallclock -> **7.6713485296117145 s/resample**.
- At the frozen `N_PERMUTATIONS = 1000`: **7671.35 s = 127.86 min = 2.131 hours** for ONE
  (field/magnitude/repeat, S) full null.

**Measurement 2 -- one real positive-control cell, run against the actual 10,000-point PU
data** (`subsample_20260729_a79b3460b838fd0a.npz`), with `N_PERMUTATIONS` temporarily reduced to
5 for THIS DIAGNOSTIC ONLY (never written to the production record, never claimed as a Phase 8
result -- see Deviations):

- `S=10 magnitude=0.0` (the no-injection anchor): **137.70 s** wallclock total.
- Isolating the null's own share at 5 permutations (`5 x 7.6713s = 38.36s`) leaves **~99.34 s** of
  fixed per-cell overhead (rebuilding modality B's 4 Gram matrices + the 4-kernel tertile panel).
- Extrapolating the SAME cell to the frozen `N_PERMUTATIONS = 1000`:
  `1000 x 7.6713s + 99.34s = 7770.6 s = 129.5 min = 2.158 hours` -- within 1.3% of Measurement 1.

**Total compute required by this plan's three tasks, at ~2.14 h/cell average:**

| Task | Cells (full null computations) | Estimated wall-clock |
|---|---|---|
| Task 1 -- positive control (D8-18) | `S_GRID (3) x PLANTED_EFFECT_GRID (7)` = 21 | ~44.9 h |
| Task 2 -- negative control (D8-19) | `S_GRID (3) x N_REPEATS (30)` = 90 | ~192.6 h |
| Task 3 -- sweep (D8-09/13/15) | `S_GRID (3) x 6 fields` = 18 | ~38.5 h |
| **Total** | **129** | **~276 h (~11.5 days)** |

**Why this was not anticipated.** `08-RESEARCH.md`'s own Runtime/Cost Model section states its
per-cell estimate ("tens of milliseconds each with BLAS", "~1000 x (tens of ms) ~ tens of seconds
per (d/seed, S) cell") explicitly as MEDIUM confidence: *"no pilot run was executed in this
session to confirm wall-clock numbers."* That pilot never happened before the D8-22 freeze
(`816863c`) sealed `N_PERMUTATIONS = 1000`, `N_REPEATS = 30`, `S_GRID = (10, 20, 50)` and the
7-rung `PLANTED_EFFECT_GRID`. The measured reality is **~500-750x** the assumed per-resample cost.

**A suspected (unfixed) technical cause, reported per this plan's own instruction to report a
genuine `cka.py` finding without touching the file:** `cka.unbiased_hsic`'s
`term1 = np.trace(Kt @ Lt)` computes a full `O(n^3)` dense matrix product to extract a trace value
that is mathematically identical to `np.sum(Kt * Lt.T)`, an `O(n^2)` computation. At the
~3,333-point pooled tertile subset size (D8-08's own measured, `S`-independent fact), this is very
likely the dominant driver of the measured 7.67s/resample. **This is not a correctness bug** (both
forms are provably equal to machine precision) and it was **not fixed** -- `cka.py` is outside this
plan's `files_modified` and is frozen under D8-22; only the developer can decide whether a
performance-only, value-preserving change to it warrants a fresh pre-registration.

## Why this plan halts instead of running anyway

The plan's own `<environment>` section anticipates exactly this shape of outcome: *"Give each
production run a generous timeout -- do NOT kill a run early and do NOT reduce N_REPEATS, the
grid, or the permutation count to make it finish faster... If a run genuinely cannot complete,
stop and report rather than trimming it."* Given a measured ~276-hour total requirement:

1. **Starting any one mode commits to days, not hours.** Task 1 alone (~45 h) already exceeds any
   session bound available here. Once started, the "do NOT kill a run early" instruction forbids
   interrupting it -- so starting a run I cannot responsibly supervise to completion, and cannot
   honestly promise to let run un-killed for days, is worse than not starting it.
2. **Reducing any frozen constant to fit a session is explicitly a pre-registration breach**
   (D8-22) -- not an option available to this executor regardless of time pressure.
3. **The plan's own contingency ("stop and report") is exactly this situation.** Nothing about the
   discovery implicates a bug in this plan's logic (see Verification below) -- it is a resource
   finding that requires a decision only the developer can make (run it unattended over multiple
   real days outside this session, or open a fresh, cost-aware pre-registration).

## What WAS verified before halting

Given the magnitude of the finding, correctness was checked thoroughly before concluding the
block is a resource problem and not a code problem:

1. **Synthetic-scale correctness** (`n=900`, tiny, fast): `run_cell` returns the expected
   `per_kernel`/`realized_h_contrast`/`n_t1/t2/t3` structure for all four kernels, with
   `null_lo`/`null_hi`/`cleared` populated only for `NULL_KERNELS`. `plant_alignment_degradation`
   at `magnitude=0.0` returns a byte-identical copy; at `magnitude=0.5` it changes the expected
   ~50% of the high-tertile's rows.
2. **Freeze-gate integration**: `--mode positive-control|negative-control|sweep` all still exit 1
   with the correct D8-22 error message on a missing or wrong `--freeze-commit`, failing BEFORE
   any expensive computation -- confirmed for all three modes directly.
3. **Real-data integration, one cell, reduced `N_PERMUTATIONS` for diagnosis only**:
   `run_positive_control` was invoked against the actual `subsample_20260729_a79b3460b838fd0a.npz`
   pair with `cka.N_PERMUTATIONS` monkeypatched to 5 and `cka.PLANTED_EFFECT_GRID` shortened to
   `(0.0, 0.5)` **in-process only** (never written to `cka.py` on disk), writing to
   `notebooks/.cache/08_scratch_realdata_check.jsonl` (deleted immediately after inspection, never
   committed, never treated as a Phase 8 record). Confirmed: data loads with the correct shapes
   and column names, `compute_density` prints sane p05/p50/p95, all four Gram matrices build, the
   tertile panel and null execute without error, every JSONL row is a plain-JSON-serializable
   dict, and `preregistration_commit` is stamped with the exact 40-character freeze SHA
   (`816863cae2209261470d1d041dcc4484a3056947`). The `cleared=True` reading this diagnostic
   produced at `magnitude=0.0` is an EXPECTED artifact of estimating a 97.5th/2.5th-percentile
   threshold from only 5 permutation draws -- far too few to be informative -- and is explicitly
   NOT a finding about the real, frozen `N_PERMUTATIONS=1000` pipeline's positive-control validity.
4. **Full suite regression check**: `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q`
   -> **761 passed, 1 skipped** both before and after this plan's edits (unchanged from 08-04's
   close) -- expected, since this plan touches only the runner, never `cka.py` or any test file.
5. **Purity checks**: `grep -n "torch"`, `grep -n "per_point_mknn\|mknn_score\|partial_spearman"`,
   and the `n-repeats`/`N_REPEATS` and `mean(...seed`/`average` prohibition greps against
   `08_cka_alignment_run.py` all print nothing (two prose-only false-positive hits were found and
   reworded during this session -- see Deviations). `git diff 816863c..HEAD --
   notebooks/pu_manifold/cka.py` is empty; `cka.py` is byte-identical to the freeze commit.
   `git diff --name-only c34ba15..HEAD -- src/effdim/ notebooks/pu_manifold/` lists only
   `cka.py`, `tests/test_cka.py` and `tests/test_cka_import_purity.py` (all from prior plans) --
   no new file under `notebooks/pu_manifold/` from this plan.

None of this rules out a bug that would only surface at the real 1000-permutation scale (larger
`n_resamples` cannot introduce a NEW code path that a 5-permutation run does not exercise -- the
loop body is identical), so this is treated as sufficient correctness evidence for a halt-and-report
rather than a silent partial-completion claim.

## Task Commits

No task's `<verify>` block was run to completion (all three require the real, frozen-scale
production CLI invocation, estimated at 45-193 hours each). Per the sequential-execution
requirement to write and commit the SUMMARY before narrating, the implementation is committed as
a single `feat` commit (not three atomic per-task commits, since none can honestly be marked
"verified" against its own acceptance criteria yet):

1. **Tasks 1-3 implementation (unexecuted at production scale): `compute_density`,
   `plant_alignment_degradation`, `run_cell`, `run_positive_control`, `shuffle_h_field`,
   `run_negative_control`, `run_sweep`** - commit hash recorded after this SUMMARY commits (see
   final commit list in the executor's completion report)

**Plan metadata:** this SUMMARY, committed separately.

## Files Created/Modified

- `notebooks/diagnostics/08_cka_alignment_run.py` - Added `compute_density`,
  `_sigma_multiplier_for_kernel_name`, `run_cell`, `plant_alignment_degradation`,
  `run_positive_control` (Task 1); `shuffle_h_field`, `run_negative_control` (Task 2); `run_sweep`
  (Task 3); wired all three into `main()`'s dispatch, replacing the `NOT_YET_IMPLEMENTED_MODES`
  branch for these three modes with real calls. No change to any gate, constant, or existing
  Task 1-3 (08-01/08-03) function.

## Decisions Made

See `key-decisions` in frontmatter for the full list. Summary: modality A/B assignment
(hsc=fixed, legacysurvey=degraded) is a discretion choice documented for future reference;
`pooled_field_guard` is called as a self-proving assertion, not a live per-seed gate; a suspected
`cka.py` performance inefficiency (`O(n^3)` trace vs. an equivalent `O(n^2)` form) is reported, not
fixed. The central decision -- **halt rather than run or trim** -- is justified at length above.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Two prohibition-check false positives in this plan's own docstring prose**
- **Found during:** post-implementation purity grep sweep (the plan's own
  `must_haves.prohibitions` verification commands)
- **Issue:** `run_sweep`'s print statement said "never averaged (D8-15)", which contains the
  literal substring `average` and would fail the prohibition's own
  `grep -n 'mean(.*seed\|average' ... prints nothing` check despite being reinforcing prose, not a
  violation (the same class of issue `08-01-SUMMARY.md` recorded for a `notebooks/.cache` prose
  false positive). Separately, `run_negative_control`'s docstring said "there is no `--n-repeats`
  ... CLI flag", matching the `n-repeats` pattern in the `grep -n "n-repeats\|n_repeats=" ... |
  grep -v N_REPEATS` prohibition check without also containing `N_REPEATS` to be excluded.
- **Fix:** Reworded both lines to state the same fact without the flagged substrings ("never
  pooled into one (D8-15)"; "there is no CLI-flag override of N_REPEATS"). No behavior change.
- **Files modified:** `notebooks/diagnostics/08_cka_alignment_run.py`
- **Verification:** Both grep commands, run exactly as the plan's `must_haves.prohibitions`
  specify, now print nothing.
- **Committed in:** the single implementation commit (see Task Commits)

**2. [Rule 1 - Bug] `run_positive_control`'s modality-A Gram build wasted a full modality-B build**
- **Found during:** implementation review, before the real-data correctness check
- **Issue:** The plan's action text says this mode "builds the eight Gram matrices once for
  modality A" -- ambiguous wording (modality A alone has only 4 kernel variants, not 8), but the
  first-draft implementation called the shared `build_gram_matrices(X_hsc, X_ls, ...)` helper,
  which unconditionally builds BOTH modalities' Gram matrices, then discarded the modality-B half
  immediately (`_grams_b_unused`). This wastes a full ~90-130s Gram build with no purpose.
- **Fix:** Modality A's four Gram matrices are now built directly via `cka.linear_gram`/
  `cka.rbf_gram` at the frozen `SIGMA_HSC` scale, never calling the two-modality
  `build_gram_matrices` helper in this function.
- **Files modified:** `notebooks/diagnostics/08_cka_alignment_run.py`
- **Verification:** Confirmed via the real-data correctness check (Measurement 2 above) that
  `run_positive_control` still produces correct, complete rows after the fix.
- **Committed in:** the single implementation commit (see Task Commits)

---

**Total deviations:** 2 auto-fixed (both Rule 1 -- a plan-prose grep false positive and a wasted
computation, neither changing any frozen constant or any measured result).
**Impact on plan:** Neither affects the central runtime-cost finding above (the wasted build was
~90-130s against a ~276-hour total, i.e. immaterial to the halt decision).

## Known Stubs

None in the traditional sense -- there is no hardcoded empty value or placeholder UI. The
substantive gap is that **`notebooks/.cache/08_cka_alignment.jsonl` does not exist**: zero Phase 8
production rows have been written by any mode. This is the plan's entire deliverable and is
recorded here, and in `.planning/WINDOWS.md`, as an `unrun-verify` entry for all three tasks.

## Threat Flags

None new. The threat register's mitigations (freeze-ancestry gate, `assert_preregistered`,
raw-numpy JSONL guard, `pooled_field_guard`) were all exercised and held during the correctness
checks above; none was bypassed.

## Issues Encountered

The core issue is the runtime-cost discovery itself, detailed above. Practically, this also meant
two background diagnostic subprocesses were started and later killed once they had served their
verification purpose (a synthetic-scale timing probe and the real-data single-cell check) --
neither was a governed production run under the plan's "do NOT kill a run early" instruction
(that instruction applies to `--mode positive-control/negative-control/sweep` invocations against
the real frozen record, none of which were started), and neither left any file behind (both
scratch JSONL paths were deleted after inspection).

## User Setup Required

None from a service-configuration standpoint. What IS required is a developer decision on how to
proceed, presented as the two options below.

## Next Phase Readiness -- BLOCKED, developer decision required

**This plan does not advance the phase.** `08-06` (whatever it produces -- likely
`08-FINDINGS.md`) depends on this plan's JSONL record existing, and it does not. Two paths, put to
the developer plainly:

1. **Run the three production modes unattended, outside this interactive session, over multiple
   real days.** The exact commands are already correct and ready:
   ```
   .venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode positive-control --freeze-commit 816863cae2209261470d1d041dcc4484a3056947
   .venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode negative-control --freeze-commit 816863cae2209261470d1d041dcc4484a3056947
   .venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode sweep --freeze-commit 816863cae2209261470d1d041dcc4484a3056947
   ```
   Run in the plan's own stated order (controls before the sweep). Each appends incrementally to
   `notebooks/.cache/08_cka_alignment.jsonl`, so a `nohup`'d, unattended, multi-day run is safe to
   let sit; if interrupted, the completed `(S, ...)` cells' rows remain valid and the plan's own
   text for Task 2 explicitly sanctions reporting a partial `S` set as not-run rather than
   extrapolating.
2. **Issue a fresh, cost-aware pre-registration.** Per D8-22, changing `N_PERMUTATIONS`,
   `N_REPEATS`, `S_GRID` or `PLANTED_EFFECT_GRID` now -- after a Phase 8 cost model exists, even
   though no Phase 8 RESULT number exists yet -- requires a new freeze commit and a documented
   reason (this measured runtime discovery). This executor takes no position on which constants a
   fresh pre-registration should choose; that is the developer's call to make with the numbers
   above in hand.

No blocker exists in the CODE. The blocker is entirely a compute-budget one, now measured
precisely rather than assumed.

---
*Phase: 08-curvature-conditioned-cka-alignment*
*Completed: 2026-08-28 (blocked, not advanced)*

## Self-Check: PASSED

- `notebooks/diagnostics/08_cka_alignment_run.py` confirmed present and modified on disk.
- `.venv/bin/python -c "import ast; ast.parse(open('notebooks/diagnostics/08_cka_alignment_run.py').read())"` exits 0 (valid syntax).
- `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` re-confirmed: 761 passed, 1 skipped.
- `git diff 816863cae2209261470d1d041dcc4484a3056947..HEAD -- notebooks/pu_manifold/cka.py` confirmed empty.
- `notebooks/.cache/08_cka_alignment.jsonl` confirmed absent (no production row exists).
