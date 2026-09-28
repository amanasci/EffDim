---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 02
subsystem: research-stats
tags: [instrument-validation, analytic-fixtures, curvature, torch, known-answer-check]

# Dependency graph
requires:
  - phase: 07-curvature-conditioned-crossmodal-alignment
    provides: "07_instrument_fixture_sweep_run.py runner, INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99) at d=20"
  - phase: 08-cka-alignment-and-instrument-fidelity
    provides: "plan 08-07's --d/--out flags on the fixture sweep runner, d=25 fidelity measurement (0.17, 0.97), HANDOFF-v1.1.md §5.3"
provides:
  - "09-FIXTURE-FIDELITY-D16.md: the plain-autoencoder decoder curvature instrument's known-answer fidelity at d=16, INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882), quoted beside the d=20 and d=25 ranges, with the d=32 gap stated as unmeasurable and its cause"
  - "COVERAGE.md: reasoned no-external-API declaration for the seal-time api-coverage.verify-pre gate"
  - "notebooks/.cache/09_fixture_fidelity_d16.jsonl: the four measured cells (gitignored)"
affects: [09-05, 09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Analytic-fixture known-answer measurement invoked read-only through an existing sealed runner's --d/--out flags, no code change"
    - "Reasoned single-line API-coverage declaration (copied Phase 7's COVERAGE.md form) in place of a capability matrix when the phase has no request/response API surface"

key-files:
  created:
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-FIXTURE-FIDELITY-D16.md
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/COVERAGE.md
  modified: []

key-decisions:
  - "Fixture fidelity at d=16 measured before the Physics run (09-CONTEXT.md discretion): it is the one d matching the colleague's chart rank directly and the phase's most-read comparison"
  - "d=32 fixture fidelity explicitly not attempted: rotate_and_pad requires D>=33 at d=32 while the small-ambient arm is hard-capped at D=28; recorded as an unmeasurable limitation, not patched or estimated"
  - "This plan runs locally rather than on the execution host: the runner carries a hard-coded developer-machine path and touches no Physics data, so the execution-host rule (which constrains where real Physics numbers are produced) does not apply"

requirements-completed: [D9-12, D9-18]

coverage:
  - id: D1
    description: "Instrument fidelity at d=16 measured on four analytic-fixture cells (cubic/ridge x D=28/D=768), recorded as INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882) for 09-05 to freeze and 09-10 to quote"
    requirement: "D9-12"
    verification:
      - kind: other
        ref: ".venv/bin/python notebooks/diagnostics/07_instrument_fixture_sweep_run.py --d 16 --out notebooks/.cache/09_fixture_fidelity_d16.jsonl (exit 0, final line DONE, 4 cells all d==16)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Phase API-coverage declaration (COVERAGE.md) accepted in place of a capability matrix"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "grep -q '^No external API integration:' COVERAGE.md; grep -cE '^\\|' COVERAGE.md == 0; grep -c OPT-OUT == 0"
        status: pass
    human_judgment: false

# Metrics
duration: 38min
completed: 2026-09-02
status: complete
---

# Phase 9 Plan 2: Instrument fidelity at d=16 and phase API-coverage declaration Summary

**Measured the plain-autoencoder decoder curvature instrument's known-answer fidelity at d=16 on four analytic fixtures (`INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882)`), and wrote the phase's reasoned no-external-API declaration.**

## Performance

- **Duration:** 38 min
- **Started:** 2026-09-02T19:57:55Z
- **Completed:** 2026-09-02T20:35:36Z
- **Tasks:** 2
- **Files modified:** 2 (both new, plus one gitignored cache record)

## Accomplishments

- Ran `notebooks/diagnostics/07_instrument_fixture_sweep_run.py --d 16 --out notebooks/.cache/09_fixture_fidelity_d16.jsonl` unmodified; it exited 0 with a final `DONE` line and produced exactly four cells, all `d==16`, covering `{cubic, ridge}` x `{D=28, D=768}`.
- Recorded `INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882)` in `09-FIXTURE-FIDELITY-D16.md` — the value plan 09-05 freezes into `physics_curvature_probe.INSTRUMENT_FIDELITY_RANGE_D16` (currently the empty placeholder from 09-01) and the value plan 09-10 quotes beside the `d=16` verdict.
- Stated the `d=32` fixture-fidelity gap as unmeasured and unmeasurable with its cause (`rotate_and_pad` requires `D >= 33` at `d=32`, the small-ambient fixture arm is hard-capped at `D=28`), quoting `HANDOFF-v1.1.md` §5.3's prior ratification of the same limitation.
- Quoted the `d=20` range `(0.53, 0.99)` and `d=25` range `(0.17, 0.97)` beside the new `d=16` numbers, and explained in the document's own words why split-half reliability (`06-FINDINGS.md`'s `R_H = 0.990` beside `rho = 0.469` on the Swiss roll; the colleague's own `d=16` `R_H` median `0.514` with 42% of anchors below 0.5) cannot substitute for a known-answer fixture check.
- Wrote `COVERAGE.md` as a single reasoned declaration (no capability table, no `OPT-OUT` rows), copying Phase 7's form for the same no-external-API situation, naming the `v2.0` HuggingFace revision pin as a data-provenance constant rather than an API version.
- No file under `notebooks/` or `src/effdim/` was created, modified, or deleted — verified via `git status --porcelain notebooks/ src/effdim/` printing nothing both before and after.

## Task Commits

Each task was committed atomically:

1. **Task 2: Write the phase's API-coverage declaration** - `032a6dd` (docs) — committed first since it had no dependency on the (longer-running) fixture sweep
2. **Task 1: Measure instrument fidelity at d=16 on the analytic fixtures and record it** - `1efacbf` (docs)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md + REQUIREMENTS.md)

## Files Created/Modified

- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/COVERAGE.md` - reasoned no-external-API declaration for the seal-time gate
- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-FIXTURE-FIDELITY-D16.md` - d=16 known-answer fidelity measurement, the d=32 gap statement, and the reliability-is-not-fidelity argument
- `notebooks/.cache/09_fixture_fidelity_d16.jsonl` - the four measured cells (gitignored, not a tracked artifact)

## Measured cells (decoder arm, full precision from the JSONL record)

| fixture | D | var_explained | cond(g) median | ii_cv | rho | median_cosine | median_ratio |
|---|---|---|---|---|---|---|---|
| cubic | 28 | 0.998946 | 2.7936 | 0.11441 | 0.942277 | 0.993279 | 0.961040 |
| cubic | 768 | 0.998599 | 3.9320 | 0.11441 | 0.837613 | 0.971920 | 0.997200 |
| ridge | 28 | 0.999441 | 1.9976 | 0.48851 | 0.987231 | 0.999566 | 0.992306 |
| ridge | 768 | 0.999371 | 2.2478 | 0.48851 | 0.988174 | 0.999474 | 0.985317 |

**`INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882)`** — floor at `cubic`/`D=768`, ceiling at
`ridge`/`D=768`. Decoder beats the point-cloud baseline on rank in 4 of 4 cells, unlike `d=20`/
`d=25` where the ranking is fixture-dependent.

**Measured wallclock:** total 2093.8 s (~34.9 min) across all four cells. The two `D=768` cells
(the ones relevant to a 5,000-row, ambient-768 cost estimate) took 126.4 s / 124.2 s for AE
training and 578.8 s / 577.9 s for curvature-field derivation — roughly 705-707 s per cell,
dominated by the `torch.func` Jacobian/Hessian curvature-field pass rather than training. Run
single-process on a 12th Gen Intel Core i7-1280P (20 logical threads, `nproc=20`), no explicit
thread cap set.

## Decisions Made

- **Fixture fidelity at `d=16` IS measured**, per `09-CONTEXT.md`'s discretion clause — it is the one `d` matching the colleague's chart rank directly and the phase's most-read comparison, and no `d` in this milestone's sweep had a known-answer measurement at `d=16` before this plan.
- **Ran locally, not on the execution host.** `07_instrument_fixture_sweep_run.py` carries a hard-coded absolute path from Phase 7 and works only on the developer's machine; it touches no Physics data, so the execution-host rule (which constrains where real Physics numbers are produced) does not apply to this diagnostic. This runner is deliberately excluded from the 09-06 hand-off bundle for the same reason.
- **Task 2 (COVERAGE.md) committed before Task 1** despite plan ordering, since it had no dependency on the multi-minute fixture sweep and could be verified and committed immediately; Task 1's document was written and committed once the sweep's background run completed.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' verification commands and acceptance criteria passed on the first attempt; no auto-fixes were needed.

## Issues Encountered

None. The fixture sweep runner ran to completion without incident; wallclock (~35 min total) was within the plan's "tens of minutes of CPU" budget.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

`INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882)` is available for plan 09-05 to freeze into
`physics_curvature_probe.py` (currently the empty placeholder `()`) and for plan 09-10 to quote
beside the `d=16` verdict sentence. `COVERAGE.md` is in place for the seal-time
`api-coverage.verify-pre` gate. No blockers for downstream plans in this wave or later waves; the
`d=32` fixture-fidelity gap remains open by design, unaffected by this plan.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-02*

## Self-Check: PASSED

- FOUND: `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-FIXTURE-FIDELITY-D16.md`
- FOUND: `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/COVERAGE.md`
- FOUND commit: `032a6dd` (Task 2)
- FOUND commit: `1efacbf` (Task 1)
