---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 08
subsystem: research-instrumentation
tags: [physics-replication, curvature, freedman-lane, positive-control, autoencoder, spearman, out-of-fold-ridge]

# Dependency graph
requires:
  - phase: 09-07
    provides: The proved row-alignment (shift 0 PASS, gap 0.516 vs margin 0.10) that makes a Physics number meaningful at all.
provides:
  - "The four production runner modes (dsweep, positive-control, shuffled-label, verdict), gated on the freeze proof and both pre-registration guards"
  - "The real Wave A sweep at D_SWEEP = (16, 20, 25, 32), 86,471 rows, 512 holdout anchors, run once on the execution host"
  - "Both gates' calibration: positive control (no target cleared at any d) and shuffled-label false-positive rate (5/80 = 0.0625, modestly above nominal 0.05)"
  - "09-WAVE-A-RESULTS.md: full per-d, per-label record, the H_tan-vs-H_norm diagnostic, the colleague's numbers beside ours, and the Wave B trigger determination (WAVE_B_NOT_TRIGGERED)"
affects: [09-09, 09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Verdict-print-recomputes-nothing: --mode verdict reads the JSONL record and anchor tables only, never re-fits"
    - "Gate-refuses-to-regenerate-field: positive-control/shuffled-label require --field-npz naming a Wave A anchor table"

key-files:
  created:
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-WAVE-A-RESULTS.md
  modified: []

key-decisions:
  - "The Wave A host run was executed by the orchestrator over SSH under the developer's standing 2026-09-04 UTC instruction, not typed interactively; recorded as evidence throughout, never as authorization for any plan change."
  - "Positive-control gate structural finding recorded as evidence beside the frozen §8-literal reading, not as a revision: the achievable statistic at d=20/25/32 is bounded by the real, near-zero h_real-y partial, and the gate plants negative targets while the real d=16 relation is strongly positive, collapsing that bisection to slope ~0."
  - "Whether to amend the positive-control gate's plant direction/grid is left open for the developer (would require a sealed-module edit, a fresh freeze, and a gate-only re-run) — not decided in this plan."
  - "Applied WAVE_B_TRIGGER_RULE literally to four DOES NOT CLEAR per-d verdicts: Wave B is not triggered at any d; 09-09's three-seed sweep does not run."

patterns-established: []

requirements-completed: [D9-01, D9-02, D9-03, D9-04, D9-09, D9-10, D9-11, D9-12, D9-13, D9-14, D9-15, D9-16]

coverage:
  - id: D1
    description: "Four production runner modes (dsweep, positive-control, shuffled-label, verdict) implemented, gated on the freeze proof and both pre-registration guards, pinned on synthetic paths"
    requirement: "D9-09"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py -x -q (three new tests + full suite: 916 passed, 1 skipped)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Wave A sweep and both gates run for real on the execution host at D_SWEEP=(16,20,25,32), 86,471 rows, 512 holdout anchors; archive returned with matching SHA-256"
    requirement: "D9-12"
    verification:
      - kind: other
        ref: "sha256sum comparison (host-reported vs locally recomputed, both c43a886c...) plus the automated record-completeness check in 09-WAVE-A-RESULTS.md"
        status: pass
    human_judgment: false
  - id: D3
    description: "09-WAVE-A-RESULTS.md: per-d/per-label table, fit quality, H_tan-vs-H_norm diagnostic, both gates' calibration, colleague comparison, secondary labels, Wave B trigger"
    requirement: "D9-10"
    verification:
      - kind: other
        ref: "Task 3's exact automated verify command (VERDICT_RULE/WAVE_B_TRIGGER_RULE present, -0.2405 and 0.765 quoted, H_tan/H_norm both present, no 'confirms', WAVE_B_NOT_TRIGGERED present) — exits with 'wave A doc ok'"
        status: pass
    human_judgment: false
  - id: D4
    description: "Positive-control gate's structural failure mechanism (bounded-achievable-statistic and negative-target-vs-positive-real-relation) recorded as evidence beside the frozen verdict, with the gate-amendment question left open for the developer"
    verification: []
    human_judgment: true
    rationale: "Whether to amend a sealed gate's plant direction/grid is a developer decision requiring a fresh freeze and re-run; no automated check can determine the right amendment, only that the finding is recorded and the frozen verdict is left unchanged."

# Metrics
duration: ~5h35m (Task 1 authored in a prior session; this continuation covered Task 2's extraction/header and Task 3's full analysis; the host round-trip itself ran 2026-09-04T13:07:42Z-18:14:13Z, ~5h07m)
completed: 2026-09-04
status: complete
---

# Phase 9 Plan 8: Wave A Sweep, Both Gates, and the Wave A Results Document Summary

**Four-`d` autoencoder sweep (86,471 rows, 512 holdout anchors) run for real on the execution
host: `DOES NOT REPLICATE` at every `d` — the one statistically decisive cell (`d=16`,
`p_fwer < 9.999e-05`) has the wrong sign, the positive-control gate cleared no target at any `d`,
and Wave B is not triggered.**

## Performance

- **Duration:** Host round-trip ~5h07m (2026-09-04T13:07:42Z start of `dsweep` to
  2026-09-04T18:14:13Z script finish); this continuation's own extraction, verification and
  document-authoring work followed immediately after the archive returned.
- **Tasks:** 3/3 complete (Task 1 in a prior session; Task 2 and Task 3 in this continuation)
- **Files modified:** 3 total across the plan (`09_physics_curvature_run.py`,
  `test_physics_curvature_probe.py` in Task 1; `09-WAVE-A-RESULTS.md` created across Task 2/3)

## Accomplishments

- Implemented and pinned `--mode dsweep`, `--mode positive-control`, `--mode shuffled-label` and
  `--mode verdict` on synthetic paths (Task 1, prior session).
- Ran the real Wave A sweep and both gates on the execution host (`pod128`, 128 cores, 16 threads),
  recomputed and matched the returned archive's SHA-256 before reading anything from it, and
  extracted 289 record rows plus 16 anchor tables (512 rows each) locally.
- Wrote `09-WAVE-A-RESULTS.md` in full: the verdict block verbatim, the per-`d` table for `mag_r`
  on both `H_tan_norm` and `H_norm`, fit quality (the `H_rad`-vs-`-d` backstop truth misses 10% at
  every `d`, 15-27%, despite `var_explained` staying above 95%), the `H_tan`-vs-`H_norm` sign/
  magnitude comparison against `08-DIAGNOSTICS.md`'s own `d=25` collapse and `d=32` sign flip
  (not reproduced identically here — disagreement lands at `d=20`/`d=32` instead, reversed
  direction at `d=32`), both gates' calibration, the colleague's numbers beside ours (opposite
  sign at every overlapping `d`; `rho(H_tan, log_knn_radius)` opposite sign at every `d` of this
  phase's own sweep versus his `+0.765`), the three secondary labels' own tables, and the Wave B
  trigger applied literally: `WAVE_B_NOT_TRIGGERED`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement the four production modes and pin them on synthetic paths** - `39089f7` (feat)
2. **Task 2: Run Wave A and both gates on the execution host and return the records** - `87baaab` (docs — checkpoint header: digest match, extraction, verify, verdict block verbatim)
3. **Task 3: Write the Wave A results document and determine the Wave B trigger** - `1e898a8` (docs — full analysis)

**Plan metadata:** this commit (docs: complete plan)

## Files Created/Modified

- `notebooks/diagnostics/09_physics_curvature_run.py` - the four production modes (Task 1)
- `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` - three new tests pinning verdict-gating and record ordering (Task 1)
- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-WAVE-A-RESULTS.md` - the full Wave A record (Task 2 header + Task 3 analysis)

## Decisions Made

- The Wave A host run was executed by the orchestrator over SSH under the developer's standing
  2026-09-04 UTC instruction ("begin with running experiments on ssh server... adhere strictly to
  the user-guide"), not typed interactively by the developer. Recorded throughout as evidence,
  never as authorization for any change to this plan's structure or the frozen constants.
- The archive's SHA-256 was recomputed and compared before any extraction or value was read, per
  T-09-55 — both digests (`c43a886c...`) matched.
- Applied `WAVE_B_TRIGGER_RULE` literally: zero of the four per-`d` cells fired
  (`PER_D_VERDICT_VALUES[0]`), so Wave B does not run at any `d`. 09-09 has nothing to execute.
- No phase verdict was written in this document — Wave B not firing does not by itself finalize
  the phase verdict; per this plan's own `<discretion_decisions>`, that act belongs to 09-10.

## Deviations from Plan

None (Rules 1-3) — the plan's tasks were executed as written. One finding is recorded as evidence
without changing any code, constant, or the frozen verdict — flagged separately below since it is
not a deviation from what this plan asked for, but a fact the plan's own instructions required
recording:

### Recorded finding (not an auto-fix — no code, constant, or record was changed)

**The positive-control gate's failure to clear any target at any `d` has a structural mechanism,
not only an instrument-sensitivity reading.** `plant_curvature_positive_control` bisects a
spread-matched, binomially-noised rank-copy of the real `H_tan_norm` field against
`controlled_partial(planted, y, Z)`. Two consequences, both verified directly against the record:
(a) at `d=20/25/32` the bisection ran to the bracket ceiling (slope 2.0) and the achieved value
tracks the unplanted real partial closely (0.0534/0.0309/0.0011 vs. the real 0.0303/0.0421/-0.0035
at the same `d`) — no target in `POSITIVE_CONTROL_TARGET_RHOS = (0.05...0.25)` is reachable unless
the real relationship already carried a comparable effect, which it does not; (b) the runner
plants negative targets (`target_rho` = -0.05...-0.25, confirmed from the record) while the real
`d=16` relation is strongly positive (`+0.346967`), so the direction test at `d=16` collapsed the
bisection to slope `~0` and reported the pure-noise-floor partial. This is stated plainly in
`09-WAVE-A-RESULTS.md` §5 beside the frozen `09-EXECUTION-HOST.md` §8-literal reading ("positive-
control gate FAILED; per the runbook, dsweep numbers are not to be read as meaningful"). The
frozen verdict (`DOES NOT REPLICATE`) is not softened by this finding and no new statistic was
added to the record — Rule 4 applies (this is an architectural question about the gate's own
design, not a bug in this plan's code), so it is put to the developer below rather than decided.

---

**Total deviations:** 0 auto-fixed. One evidence-only finding recorded per the continuation's
explicit instruction, changing no code and no frozen constant.
**Impact on plan:** None on scope — the plan's tasks, files, and acceptance criteria are all
satisfied exactly as written; the finding is additional context for the open question below.

## Issues Encountered

None beyond the recorded finding above. The archive transfer, digest verification, extraction,
and both plan verify commands (Task 2's record-completeness check, Task 3's document-content
check) all passed on the first attempt.

## Open Questions for the Developer

**Whether to amend the positive-control gate's plant direction and/or target grid.** As recorded
above and in `09-WAVE-A-RESULTS.md` §5, the gate as currently wired cannot detect an effect of the
size `POSITIVE_CONTROL_TARGET_RHOS` asks for at `d=20/25/32` (the achievable statistic is bounded
by the real, near-zero relationship at those `d`), and at `d=16` it plants in the wrong direction
relative to the real, strongly positive relationship, collapsing the search to near-zero slope.
Amending either would require: (1) a sealed-module edit to
`plant_curvature_positive_control` and/or `POSITIVE_CONTROL_TARGET_RHOS`/`POSITIVE_CONTROL_RULE`
in `notebooks/pu_manifold/physics_curvature_probe.py`, (2) a fresh freeze commit and a numbered
pre-registration amendment document superseding `09-PREREGISTRATION.md` in full, and (3) a
re-run of `--mode positive-control` at all four `d` values on the execution host (the sweep and
`--mode shuffled-label` results would not need to be re-run, since neither depends on the gate's
plant mechanism). This is **not** decided in this plan — it is an architectural change (Rule 4)
to a sealed, frozen module after a Physics number already exists, which `09-PREREGISTRATION.md`'s
own closing rule makes a pre-registration BREACH remediable only through that three-step process.
The frozen Wave A verdict recorded here (`DOES NOT REPLICATE`) stands regardless of how this
question is resolved; amending the gate could only change confidence in *why* it does not
replicate, not the recorded per-`d` sign-and-FWER outcome itself.

## User Setup Required

None beyond the already-completed `checkpoint:human-action` (Task 2) — the execution-host run
this plan required has already happened and its records are ingested.

## Next Phase Readiness

- `WAVE_B_NOT_TRIGGERED`: 09-09's three-seed sweep has nothing to run — no `d` value fired at
  Wave A. 09-09 should read this document, confirm the trigger is empty, and record that its own
  `--mode seeds` step is skipped per `09-EXECUTION-HOST.md` §4 step 6.
- 09-10 can now write the phase verdict: Wave A's own per-`d` verdicts (`DOES NOT CLEAR` at every
  `d`) already determine `phase_verdict = DOES NOT REPLICATE` per `VERDICT_RULE`, and Wave B does
  not change any cell since it never runs. 09-10 should also surface this plan's open question
  (the positive-control gate amendment) to the developer if it has not already been raised.
- No blocker to 09-09/09-10 proceeding; the recorded evidence and the open question above are
  informational, not gating.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-04*

## Self-Check: PASSED

All four claimed files found on disk (`notebooks/diagnostics/09_physics_curvature_run.py`,
`notebooks/pu_manifold/tests/test_physics_curvature_probe.py`, `09-WAVE-A-RESULTS.md`,
`09-08-SUMMARY.md`); all three claimed task commits (`39089f7`, `87baaab`, `1e898a8`) found in
`git log --oneline --all`.
