---
phase: 05-curvature-conditioned-linear-decodability
plan: 04
subsystem: geometry
tags: [pre-registration, linear-probe, ridge, bucketing, seed-verdict-combination, freeze]

# Dependency graph
requires:
  - phase: 05-curvature-conditioned-linear-decodability (05-03)
    provides: linear_probe.py restructured for three per-seed verdicts (all 31 constants
      still unset), the three per-seed bucket artifacts (05_curvature_buckets_seed*.npz),
      combine_seed_verdicts
  - phase: 05-curvature-conditioned-linear-decodability (05-03 Task 1 checkpoint)
    provides: 05-03-DECISION.md -- the ratified, one-way refusal to pool the three seeds
provides:
  - notebooks/pu_manifold/linear_probe.py with all 31 pre-registered constants filled and
    assert_preregistered() passing -- the D5-09 freeze commit, closed to further edits
  - VERDICT_RULE and SEED_VERDICT_COMBINATION_RULE frozen as full multi-line strings carrying
    the D5-11/D5-12/D5-13 caveats and the SPLIT ACROSS SEEDS non-support framing in their own
    committed text
  - .planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md -- the
    committed, human-readable pre-registration record
  - test_curvature_convention_matches_sealed_modules passing (freeze tripwire xfail removed)
affects: [05-05, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Freeze commit as the sole remaining edit to a pre-registration module: this plan's
      Task 2 is a VALUE-only edit to linear_probe.py's constants block (verified by
      'git diff HEAD~1 HEAD --stat' showing no function def lines touched), leaving the file
      at exactly three commits (creation, per-seed restructure, freeze) -- 05-06 proves this
      is the final commit on the file via git ancestry"
    - "Multi-line rule strings that carry their own caveats: VERDICT_RULE and
      SEED_VERDICT_COMBINATION_RULE are frozen as full prose, not just structural constants,
      so the D5-11 no-anchor caveat, the D5-12 CAE_VERDICT=FAIL inheritance, and the SPLIT
      ACROSS SEEDS non-support framing travel with every number the rule ever produces,
      readable from the rule's own text without executing it"

key-files:
  created:
    - .planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md
  modified:
    - notebooks/pu_manifold/linear_probe.py
    - notebooks/pu_manifold/tests/test_linear_probe.py

key-decisions:
  - "Ratified ratify-recommended with no amendments at the Task 1 checkpoint (human response:
    'ratify-recommended') -- every value frozen is exactly the planner's proposal, none
    invented, relaxed, or tuned after the fact"
  - "test_assert_preregistered_raises_when_absent's first assertion was inverted (was: module
    as shipped raises; now: module as shipped -- frozen -- does not raise) because the freeze
    itself makes the original pre-freeze assertion false; this is the freeze's own intended
    consequence, not a bug, and the test's remaining two assertions (monkeypatch to a broken
    rule -> raises; monkeypatch back to well-formed -> does not raise) are unchanged"
  - "BUCKET_EDGES_PER_SEED's three inner tuples were read programmatically from the three
    05-03 npz artifacts and diffed elementwise against the module's written values rather than
    retyped, per the plan's own anti-transcription-error instruction"

requirements-completed: [D5-02, D5-04, D5-06, D5-07, D5-08, D5-09, D5-10, D5-11, D5-12]

coverage:
  - id: D1
    description: "notebooks/pu_manifold/linear_probe.py's 31 pre-registered constants are all
      filled with the values ratified at the Task 1 checkpoint (ratify-recommended, no
      amendments); assert_preregistered() passes; SEED_HANDLING_RULE equals
      no_pooling_per_seed_verdicts by exact-equality check; BUCKET_EDGES_PER_SEED's three
      inner tuples equal the three 05-03 npz artifacts' bucket_edges arrays elementwise as
      float64 with no tolerance; VERDICT_RULE and SEED_VERDICT_COMBINATION_RULE carry every
      required literal including the sealed -0.015106571347065712 rank value and
      CAE_VERDICT = FAIL; the freeze tripwire xfail marker on
      test_curvature_convention_matches_sealed_modules is removed and the test passes; the
      file carries exactly three commits (creation, 05-03 restructure, this freeze)"
    requirement: D5-09
    verification:
      - kind: integration
        ref: "python -c assert script over linear_probe module attributes, the three npz artifacts, VERDICT_RULE/SEED_VERDICT_COMBINATION_RULE literals, and combine_seed_verdicts -- see plan 05-04 Task 2 <verify>"
        status: pass
      - kind: unit
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q -rxX (390 passed, 1 skipped, 0 xfailed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "05-PREREGISTRATION.md exists, restates all 31 constants verbatim from the
      frozen module, quotes VERDICT_RULE and SEED_VERDICT_COMBINATION_RULE verbatim, records
      ratification notes for both the 05-03 Task 1 and 05-04 Task 1 blocking checkpoints,
      records the D5-04 supersession by 05-03-DECISION.md and the D5-05 pooled-half
      disposition, carries the corrected 4/3 effective level counts with the retracted
      5,301/9,852 figures appearing only inside a retracting sentence, resolves all four of
      05-CONTEXT.md's Claude's Discretion items by name, and states the D5-11/D5-12/D5-13
      accepted gaps at full strength"
    requirement: D5-09
    verification:
      - kind: integration
        ref: "python -c assert script over 05-PREREGISTRATION.md's text -- all 31 names, required literals, retraction-window check, verbatim-quote check -- see plan 05-04 Task 3 <verify>"
        status: pass
    human_judgment: false
  - id: D3
    description: "No PU probe number exists anywhere in the repository at the freeze commit or
      at the pre-registration commit"
    requirement: D5-09
    verification:
      - kind: integration
        ref: "pathlib check that notebooks/.cache/05_curvature_probe_decodability.jsonl does not exist -- asserted in both Task 2 and Task 3 <verify> blocks"
        status: pass
    human_judgment: false

duration: ~7 min (commit-to-commit span for this continuation agent; excludes plan/context reading and the checkpoint round-trip)
completed: 2026-08-24
status: complete
---

# Phase 5 Plan 4: Pre-Registration Freeze Summary

**All 31 free parameters of Phase 5 -- including the three-into-one seed verdict combination rule and what a split means -- are now frozen in committed source (`linear_probe.py`, commit `32dabe3`) and in a committed human-readable record (`05-PREREGISTRATION.md`, commit `b45ae1b`), with no PU probe number anywhere in the repository.**

## Performance

- **Duration:** ~7 min (commit span 23:07:59Z -- 23:10:15Z UTC / 19:07:59 -- 19:10:15 local); this continuation agent picked up at Task 1 with the checkpoint already resolved (`ratify-recommended`, no amendments) and proceeded directly to Tasks 2 and 3
- **Started:** 2026-08-24T23:07:59Z (Task 2 commit)
- **Completed:** 2026-08-24T23:10:15Z (Task 3 commit)
- **Tasks:** 3 (Task 1 is the blocking checkpoint, resolved before this agent was spawned; Tasks 2 and 3 executed here)
- **Files modified:** 3 (2 code/test files, 1 new planning document)

## Accomplishments

- **Task 1 (checkpoint, resolved prior to this agent):** the human selected `ratify-recommended`
  -- the planner's full proposal in `05-04-PLAN.md`, verbatim, with no amendments. Every
  constant frozen at Task 2 is exactly what the plan proposed before the checkpoint existed.
- **Task 2, the freeze commit (`32dabe3`):** all 31 constants in `linear_probe.py` filled --
  `TRAIN_FRACTION=0.7`/`SPLIT_SEED=20260824` (one shared 70/30 split across all three seeds'
  bucketings), `RIDGE_ALPHA_GRID=(1e-2..1e4)` with RidgeCV LOOCV selection,
  `RESIDUAL_METRIC="squared_l2_per_point"` paired with `R2_MULTIOUTPUT="variance_weighted"`,
  `EMBEDDING_PREPROCESSING="raw_as_cached"`, `N_BUCKETS=3` with `BUCKET_EDGES_PER_SEED` read
  programmatically (not retyped) from the three `05-03` npz artifacts, `SEED_HANDLING_RULE=
  "no_pooling_per_seed_verdicts"`, a 200-repeat size-matched re-check, and a 1000-resample
  0.95-confidence bootstrap.
- `VERDICT_RULE` and `SEED_VERDICT_COMBINATION_RULE` written as full multi-line strings, each
  carrying its own caveats in its own committed text: the per-seed HOLDS conjunction (disjoint
  CIs, higher mean residual, size-matched sign survival), the three-way HOLDS/SPLIT/NONE
  phase-combination mapping with `SPLIT ACROSS SEEDS` explicitly framed as a complete
  non-supportive terminal outcome, D5-11's sealed `-0.015106571347065712` no-anchor caveat, and
  D5-12's `CAE_VERDICT = FAIL` inheritance.
- The freeze tripwire cleared: `test_curvature_convention_matches_sealed_modules`'s strict
  `xfail` marker removed; the test now passes for real (`CURVATURE_CONVENTION` equals `"trace"`
  across `linear_probe`, `chart_curvature`, and `curvature_probe`).
- Full suite green: `390 passed, 1 skipped, 0 xfailed` (up from `389 passed, 1 skipped,
  1 xfailed` before this plan -- the freed xfail is the one net gain).
- `notebooks/pu_manifold/linear_probe.py` now carries exactly three commits in its history
  (`5888d0d` creation at `05-01`, `94735b7` per-seed restructure at `05-03`, `32dabe3` this
  freeze) -- verified by `git log --oneline -- notebooks/pu_manifold/linear_probe.py`.
- **Task 3, the pre-registration record (`b45ae1b`):** `05-PREREGISTRATION.md` created,
  restating all 31 constants verbatim from the frozen module, quoting both rule strings in
  full, recording both checkpoint ratification notes (including the `05-03-DECISION.md`
  supersession of `05-CONTEXT.md` D5-04 and the three rejected alternatives), the D5-05
  pooled-half disposition, the corrected 4/3 effective-level counts (with the retracted
  5,301/9,852 figures appearing only inside the sentence that retracts them), all four
  resolved `Claude's Discretion` items, and the D5-11/D5-12/D5-13 accepted gaps stated at full
  strength.
- `notebooks/.cache/05_curvature_probe_decodability.jsonl` does not exist at either commit --
  verified directly in both tasks' `<verify>` blocks.

## Task Commits

Each task was committed atomically:

1. **Task 1: Ratify the full pre-registration** - checkpoint, resolved by the human
   (`ratify-recommended`) before this continuation agent was spawned; no commit of its own.
2. **Task 2: Fill the 31 constants -- the freeze commit** - `32dabe3` (feat)
3. **Task 3: The committed pre-registration record** - `b45ae1b` (docs)

**Plan metadata:** this commit (SUMMARY + STATE + ROADMAP + REQUIREMENTS, once created)

## Files Created/Modified

- `notebooks/pu_manifold/linear_probe.py` - All 31 constants filled with the ratified values;
  header comment above the constants block rewritten from "WRITTEN UNSET" to "FROZEN"; module
  docstring paragraph (a)'s `CURVATURE_SOURCE_FUNCTION` note updated from "unset here" to
  "ratified and filled at the 05-04 freeze, this commit"
- `notebooks/pu_manifold/tests/test_linear_probe.py` - Strict `xfail` marker removed from
  `test_curvature_convention_matches_sealed_modules`; its docstring updated to describe the
  post-freeze state; `test_assert_preregistered_raises_when_absent`'s first assertion inverted
  (module as shipped now passes `assert_preregistered()` rather than raising) with its
  docstring updated accordingly, since the module ships frozen rather than unset as of this
  plan
- `.planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md` - New.
  The full committed pre-registration record

## Decisions Made

- **Ratified `ratify-recommended` with no amendments** at the Task 1 checkpoint, per the
  human's explicit response. Every value frozen at Task 2 is exactly the planner's proposal.
- **`test_assert_preregistered_raises_when_absent`'s first assertion was inverted, not
  deleted**, to match the module's new frozen (not unset) shipped state -- this is a direct,
  intended consequence of the freeze this plan performs, not a defect being patched around.
  The test's remaining two assertions (a broken monkeypatched rule still raises; a well-formed
  monkeypatched rule does not) are unchanged and continue to prove the guard is live in both
  directions, now on top of a frozen module rather than an absent one.
- **`BUCKET_EDGES_PER_SEED` read programmatically from the three `05-03` npz artifacts**, then
  diffed elementwise (float64, no tolerance) against the written module values, rather than
  retyped from prose, per the plan's own anti-transcription-error instruction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `test_assert_preregistered_raises_when_absent` failed after the freeze because its first assertion tested pre-freeze (unset) behavior**
- **Found during:** Task 2, running the full test suite after filling the constants.
- **Issue:** The test's first block asserted `assert_preregistered()` raises "on the module as
  shipped" -- true only while every constant was unset. Once Task 2 froze real values into the
  module, that assertion became false by design: the freeze's entire point is that the shipped
  module now passes the guard. `1 failed, 389 passed` on first run.
- **Fix:** Inverted the first assertion (`lp.assert_preregistered()` must NOT raise on the
  frozen shipped module) and updated the docstring to describe the post-freeze contract. Left
  the remaining monkeypatch-to-broken / monkeypatch-back-to-well-formed assertions unchanged --
  they still prove the guard fires correctly in both directions.
- **Files modified:** `notebooks/pu_manifold/tests/test_linear_probe.py`
- **Verification:** Full suite re-run: `390 passed, 1 skipped, 0 xfailed` (0 failed).
- **Committed in:** `32dabe3` (Task 2 commit, alongside the constants that caused the change)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug fix -- a test whose pre-freeze assumption was
invalidated by this plan's own intended freeze behavior).
**Impact on plan:** No scope creep. The fix is the freeze's own designed consequence surfacing
in a test that had not yet been updated for it; nothing outside `linear_probe.py`'s and its
test file's constants-and-guard behavior was touched.

## Issues Encountered

None beyond the auto-fixed test assertion documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `notebooks/pu_manifold/linear_probe.py` is closed: three commits total, the freeze most
  recent, `assert_preregistered()` passing. `05-05` and `05-06` must never edit it again; any
  future change is a recorded pre-registration BREACH, not a silent fix.
- `--mode bucketed` is live for the first time (`assert_preregistered()` no longer raises), so
  `05-05` is unblocked to run the three per-seed probes.
- `05-PREREGISTRATION.md` is committed and is the authoritative human-readable record `05-05`
  and `05-06` should cite rather than re-deriving values from the module or from this SUMMARY.
- `05-06` must prove, via `git merge-base --is-ancestor`, that the freeze commit `32dabe3` is
  an ancestor of the first commit carrying a PU probe number, and that
  `git diff 32dabe3 HEAD -- notebooks/pu_manifold/linear_probe.py` remains empty.
- No blockers. CLAUDE.md's Swiss-roll sanity-check rule does not trigger for this plan: no new
  manifold-learning or representation-learning model was introduced -- this plan assigns
  constants and writes a document, exactly as recorded in `05-PREREGISTRATION.md`'s own
  accepted-gaps section.

---
*Phase: 05-curvature-conditioned-linear-decodability*
*Completed: 2026-08-24*
