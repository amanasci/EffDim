---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 09
subsystem: research-instrumentation
tags: [physics-replication, wave-b, seed-stability, unanimity-rule, not-triggered]

# Dependency graph
requires:
  - phase: 09-08
    provides: "The Wave A sweep result (DOES NOT REPLICATE at every d) and the recorded Wave B trigger determination (WAVE_B_NOT_TRIGGERED)"
provides:
  - "--mode seeds: reads its d scope from Wave A's own record row (never a CLI flag), records WAVE_B_NOT_TRIGGERED as a complete terminal outcome when the scope is empty, and otherwise refits at TORCH_INIT_SEEDS_WAVE_B and combines via the frozen unanimity rule"
  - "combine_seed_verdicts: pinned exact-equality-guarded, raises on any entry count other than three, never averages or upgrades a split"
  - "09-WAVE-B-RESULTS.md: the frozen rules quoted, the empty scope stated explicitly with the reason no seed agreement is claimed, and the unchanged per-d cell verdicts carried into 09-10"
affects: [09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Empty-scope-is-a-recorded-outcome: WAVE_B_NOT_TRIGGERED is its own record row and document statement, never an absence a reader has to infer"
    - "Host-run-confirms-rather-than-widens: running --mode seeds on the host after the precondition already permitted skipping it produces independent confirmation, not a scope change"

key-files:
  created:
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-WAVE-B-RESULTS.md
  modified: []

key-decisions:
  - "The developer's standing 2026-09-04 UTC instruction directed the orchestrator to run experiments on the SSH host; under it, Task 2's host steps ran even though the precondition already permitted skipping them (Wave A had already recorded WAVE_B_NOT_TRIGGERED). The host run is recorded as independent confirmation via the runner's own live read of the Wave A record, never as a widening of scope or a second measurement of anything Wave A did not already determine."
  - "The archive's SHA-256 was recomputed locally and compared before extraction, per T-09-67 -- matched (793c7e55...)."
  - "09-WAVE-B-RESULTS.md's per-d seed table and field-disagreement diagnostic sections are marked 'not applicable' rather than omitted, so a reader cannot mistake their absence for an unmeasured gap."
  - "A plan-text defect in Task 1's acceptance criteria is recorded, not fixed in code (see Deviations): the literal snippet indexes PER_D_VERDICT_VALUES[2], which does not exist (the tuple is frozen at exactly two entries). The prior executor verified the evident intent (a 2-of-3 split combines to the literal string 'SPLIT ACROSS SEEDS') instead of the broken literal."

patterns-established: []

requirements-completed: [D9-10, D9-12, D9-17]

coverage:
  - id: D1
    description: "--mode seeds implemented: reads scope from Wave A's record, records WAVE_B_NOT_TRIGGERED on empty scope, refits per seed and combines via the frozen unanimity rule when non-empty; combine_seed_verdicts raises on any count other than three"
    requirement: "D9-17"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py -x -q (101 passed); full suite notebooks/pu_manifold/tests/ -q also run in this continuation"
        status: pass
    human_judgment: false
  - id: D2
    description: "Wave B ran for real on the execution host (--mode seeds), confirmed WAVE_B_NOT_TRIGGERED independently via the runner's own live read of the Wave A record; archive returned with matching SHA-256"
    requirement: "D9-12"
    verification:
      - kind: other
        ref: "sha256sum comparison (host-reported vs locally recomputed, both 793c7e55...) plus the plan's exact automated verify command (prints 'not_triggered 1')"
        status: pass
    human_judgment: false
  - id: D3
    description: "09-WAVE-B-RESULTS.md: frozen rules quoted beside the freeze SHA, empty scope stated with an explicit no-seed-agreement-claimed sentence, split/unanimity vocabulary stated for completeness, unchanged per-d cell verdicts in D_SWEEP order, no phase verdict written"
    requirement: "D9-10"
    verification:
      - kind: other
        ref: "Task 3's exact automated verify command -- prints 'wave B doc ok'"
        status: pass
    human_judgment: false

# Metrics
duration: "~15 minutes (this continuation: Task 2 header/extraction/verify plus Task 3's document, following Task 1's implementation from a prior session; the host round-trip itself ran 11 s, 2026-09-04T18:42:20Z-18:42:31Z)"
completed: 2026-09-04
status: complete
---

# Phase 9 Plan 9: Wave B Seed Sweep — WAVE_B_NOT_TRIGGERED Summary

**`--mode seeds` ran for real on the execution host and independently confirmed what Wave A's
record already determined: zero of the four `D_SWEEP` cells fired, so Wave B's three-seed sweep
never fits an autoencoder — `WAVE_B_NOT_TRIGGERED` is recorded as a complete, terminal outcome,
not an absence, and the per-`d` cell verdicts 09-10 will read are unchanged from Wave A.**

## Performance

- **Duration:** This continuation ran ~15 minutes (digest verification, extraction, document
  authoring); the host round-trip for `--mode seeds` itself was 11 seconds wall-clock
  (2026-09-04T18:42:20Z start to 2026-09-04T18:42:31Z finish) — no autoencoder fit ran, since the
  triggered `d` list read from Wave A's record was empty.
- **Tasks:** 3/3 complete (Task 1 in a prior session; Task 2 and Task 3 in this continuation)
- **Files modified:** 3 total across the plan (`09_physics_curvature_run.py`,
  `test_physics_curvature_probe.py` in Task 1; `09-WAVE-B-RESULTS.md` created across Task 2/3)

## Accomplishments

- Implemented and pinned `run_seeds(args)`, `_triggered_d_values(record_path)`, and three new
  tests (`test_seeds_mode_refuses_untriggered_d`, `test_seeds_mode_records_wave_b_not_triggered`,
  `test_seed_cell_verdict_never_upgrades_a_split`) in a prior session (Task 1, commit `2d61e36`).
- Ran `--mode seeds --freeze-commit 5f7fbe27… --threads 16` on the execution host (`pod128`),
  under the developer's standing instruction, in the same clone pulled to run commit `2d61e368…`.
  The mode read the triggered `d` list from Wave A's own `verdict` record row, found it empty, and
  recorded `WAVE_B_NOT_TRIGGERED` — printed verbatim and reproduced in full in
  `09-WAVE-B-RESULTS.md`'s Run record section.
- Recomputed the returned archive's SHA-256 locally (`793c7e55…`) and matched it against the
  host-reported digest before reading anything from it (T-09-67). Extracted under
  `notebooks/.cache/`: the 16 anchor tables are byte-identical to Wave A's own, and
  `09_physics_curvature.jsonl` grew from 299 to 301 rows — one new `environment` row and one
  `seed_cell_verdict` row carrying `d: null`, `wave_b: "WAVE_B_NOT_TRIGGERED"`,
  `cell_verdict: "WAVE_B_NOT_TRIGGERED"`, `seeds: [0, 1, 2]`.
- Wrote `09-WAVE-B-RESULTS.md` in full: the four frozen rules quoted verbatim beside the freeze
  SHA and `05-03-DECISION.md`'s citation; the empty scope stated explicitly with the reason no
  seed agreement is claimed; the per-`d` seed table and field-disagreement diagnostic sections
  marked "not applicable" rather than silently omitted; what a split vs. unanimous cell means
  stated for completeness though neither occurred; and the closing per-`d` cell table (unchanged
  from Wave A, `DOES NOT CLEAR` at every `d`) that 09-10 will read, with no phase verdict written.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement the seed wave and pin the combination rule** — `2d61e36` (feat, prior session)
2. **Task 2: Run the seed wave on the execution host and return the records** — `775707a` (docs — checkpoint header: digest match, extraction, verify, verbatim printed output)
3. **Task 3: Write the Wave B results document** — `3742159` (docs — full analysis)

**Plan metadata:** this commit (docs: complete plan)

## Files Created/Modified

- `notebooks/diagnostics/09_physics_curvature_run.py` — `run_seeds(args)` and `_triggered_d_values` (Task 1, prior session)
- `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` — three new tests (Task 1, prior session)
- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-WAVE-B-RESULTS.md` — the full Wave B record (Task 2 header + Task 3 analysis)

## Decisions Made

- Task 2's host run proceeded under the developer's standing 2026-09-04 UTC instruction even
  though the plan's own `<precondition>` already permitted skipping it (`09-WAVE-A-RESULTS.md`
  had already recorded `WAVE_B_NOT_TRIGGERED`). This is recorded throughout as independent
  confirmation from the runner's own live read of the Wave A record, never as a scope widening or
  a fresh measurement of anything not already determined.
- The archive's SHA-256 was recomputed and compared before any extraction or value was read;
  both digests (`793c7e55…`) matched.
- `09-WAVE-B-RESULTS.md`'s empty-scope sections (§3 per-`d` seed table, §4 field disagreement)
  state "not applicable" explicitly rather than being left out, so a reader cannot mistake their
  absence for an unmeasured gap.
- No phase verdict was written in this document, per this plan's own `<discretion_decisions>` —
  that act belongs to 09-10, and the per-`d` cells it will read are unchanged from Wave A.

## Deviations from Plan

None (Rules 1-3) on code — Task 1's implementation and Task 2/3's document-authoring were executed
as written; no code, constant, or record was changed by this continuation. One plan-text defect is
flagged below since it affects how a future reader should interpret Task 1's acceptance criteria,
not a deviation in what was built:

### Flagged plan-text defect (carried forward from the prior session, no code changed)

**Task 1's acceptance criteria contain a snippet that raises `IndexError` as literally written.**
The plan's fourth acceptance-criteria bullet is:

```
.venv/bin/python -c "...assert p.combine_seed_verdicts([p.PER_D_VERDICT_VALUES[0], p.PER_D_VERDICT_VALUES[0], p.PER_D_VERDICT_VALUES[1]]) == p.PER_D_VERDICT_VALUES[2]"
```

`PER_D_VERDICT_VALUES = ("NEGATIVE AND CLEARS FWER NULL", "DOES NOT CLEAR")` is frozen at exactly
two entries (`09-PREREGISTRATION.md`; see also `09-05-SUMMARY.md`'s note on the dropped third
entry). `PER_D_VERDICT_VALUES[2]` does not exist and the literal snippet cannot run. The prior
executor verified the evident intent instead — that a two-cleared/one-not-cleared input combines
to the literal string `"SPLIT ACROSS SEEDS"` (`combine_seed_verdicts`'s own documented return
value for any non-unanimous split) — and confirmed that behavior directly against the frozen
function rather than the broken literal. No code, constant, or plan text was edited to fix this;
it is recorded here so 09-10 or a later audit does not re-attempt the literal snippet and conclude
a regression exists where none does.

---

**Total deviations:** 0 auto-fixed. One plan-text defect flagged, changing no code and no frozen
constant.
**Impact on plan:** None on scope — Task 1's actual verification (both automated commands in its
`<verify>` block, the full test suite, and the corrected acceptance check) all passed; only the
one literal acceptance-criteria snippet as transcribed in the plan file cannot run as written.

## Issues Encountered

None. The host run, digest verification, extraction, and both plan verify commands (Task 2's
record-completeness check, Task 3's document-content check) all passed on the first attempt. The
plan-level verification block (full `notebooks/pu_manifold/tests/` suite, `--mode seeds` without
`--freeze-commit` exiting 1, `combine_seed_verdicts` raising on two and on four entries, `--mode
smoke` still exiting 0, `git diff --name-only 5c68a3e..HEAD -- notebooks/pu_manifold/` listing
only the test file) was re-run in this continuation and passed in full.

## Open Questions for the Developer

**Carried forward, unresolved by this plan** — `09-08-SUMMARY.md`'s open question about whether
to amend the positive-control gate's plant direction and/or target grid remains open. This plan
does not touch it (Wave B never ran a seed, so no gate re-run was in scope), and 09-10 should
surface it again if the developer has not yet responded.

## User Setup Required

None beyond the already-completed `checkpoint:human-action` (Task 2) — the execution-host run
this plan's Task 2 asked for has already happened (independently confirming `WAVE_B_NOT_TRIGGERED`)
and its records are ingested.

## Next Phase Readiness

- 09-10 can write the phase verdict directly from Wave A's per-`d` cells (`DOES NOT CLEAR` at
  every `d` in `D_SWEEP`), since Wave B changed nothing — `phase_verdict = "DOES NOT REPLICATE"`
  per `VERDICT_RULE`, unchanged from `09-08-SUMMARY.md`'s own "Next Phase Readiness" note.
- 09-10 should also carry forward the open positive-control gate question above, still
  undecided by the developer as of this plan.
- No blocker to 09-10 proceeding.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-04*

## Self-Check: PASSED

All four claimed files found on disk (`notebooks/diagnostics/09_physics_curvature_run.py`,
`notebooks/pu_manifold/tests/test_physics_curvature_probe.py`, `09-WAVE-B-RESULTS.md`,
`09-09-SUMMARY.md`); all three claimed task commits (`2d61e36`, `775707a`, `3742159`) found in
`git log --oneline --all`. Full test suite re-run in this continuation: 919 passed, 1 skipped
(391.38s).
