---
phase: 03-decoder-curvature-field
plan: 03
subsystem: testing
tags: [torch, torch.func, finite-differences, derivative-bridge, pytest]

# Dependency graph
requires:
  - phase: 02.6-decoder-substrate-screening
    provides: derivative_bridge.py itself (D-16/D-17/D-18), and 02.6-REVIEW.md's WR-01/WR-02/WR-03 findings with prescribed fixes
provides:
  - "derivative_bridge.py's three finite-difference entry points raise a friendly ValueError naming model.double() for a float32 model, instead of a bare torch RuntimeError (WR-01)"
  - "calibrate_fd_step and derivative_agreement share one chunked autodiff-Hessian helper (_chunked_vmap_hessian), correct above VMAP_CHUNK rather than by BRIDGE_N_POINTS==VMAP_CHUNK coincidence (WR-03)"
  - "_agreement_stats reports near_zero_reference_fraction alongside the relative-error columns, with a thin-denominator docstring caveat (WR-02)"
affects: [03-09 (bridge run at PU scale), any future derivative_bridge caller]

# Tech tracking
tech-stack:
  added: []
  patterns: ["translate a bare torch RuntimeError from a dtype-mismatched matmul into a friendly ValueError at the point decode_batch is actually invoked, rather than spending a separate probe call that would inflate a bounded-cost invocation-count contract"]

key-files:
  created: []
  modified:
    - notebooks/pu_manifold/derivative_bridge.py
    - notebooks/pu_manifold/tests/test_derivative_bridge.py

key-decisions:
  - "WR-01's model-dtype check rides the real first decode_batch call (via _chunked_eval and calibrate_fd_step's autodiff call) instead of a dedicated probe call, because a separate probe would add one extra decode_batch invocation and break test_finite_difference_hessian_invocation_count_matches_chunk_arithmetic's exact ceil(batch*n_offsets/MAX_FD_ROWS) count"
  - "02.6-REVIEW.md's WR-01 fix sketch (_assert_decode_batch_float64 probing decode_batch(z_chart[:1]).dtype directly) does not work as literally written: a float32-parameter decoder fed a float64 probe row raises RuntimeError from its own internal matmul before ever returning a dtype to check, so the probe itself needed a try/except translating that RuntimeError -- verified directly against torch.func.hessian and a plain matmul before choosing the final implementation"
  - "DEFAULT_REL_FLOOR = 1e-12 chosen relative to float64 scale (>>1e-16 machine epsilon), documented as a reading-aid threshold only, never an acceptance bar (D-18)"

requirements-completed: [CURV-05]

coverage:
  - id: D1
    description: "WR-01 closed: finite_difference_jacobian, finite_difference_hessian and calibrate_fd_step raise a friendly ValueError naming model.double() for a float32 model, not a bare RuntimeError"
    requirement: "CURV-05"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_derivative_bridge.py#test_derivative_bridge_float32_model_raises_friendly_value_error"
        status: pass
    human_judgment: false
  - id: D2
    description: "WR-03 closed: calibrate_fd_step and derivative_agreement share one chunked Hessian helper, correct above VMAP_CHUNK"
    requirement: "CURV-05"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_derivative_bridge.py#test_calibrate_fd_step_chunks_above_vmap_chunk"
        status: pass
    human_judgment: false
  - id: D3
    description: "WR-02 closed: near_zero_reference_fraction reported alongside the relative-error columns, no threshold introduced"
    requirement: "CURV-05"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_derivative_bridge.py#test_derivative_bridge_reports_near_zero_relative_reference_fraction"
        status: pass
    human_judgment: false
  - id: D4
    description: "derivative_agreement's report-never-gate contract (separate full_hessian_agreement / reduced_mean_curvature_agreement keys, no combined score, no boolean) is unchanged"
    requirement: "CURV-05"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_derivative_bridge.py#test_derivative_agreement_returns_separate_keys_no_combined_no_boolean"
        status: pass
    human_judgment: false

duration: ~30min
completed: 2026-08-14
status: complete
---

# Phase 03 Plan 03: Derivative-Bridge WR-01/02/03 Closure Summary

**Closed all three `02.6-REVIEW.md` warnings in `derivative_bridge.py` (probe-based float64 guard, shared chunked comparison Hessian, near-zero-reference diagnostic) with three regression tests, before the bridge runs at PU scale in plan 03-09.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-08-14T13:50:00Z (approx.)
- **Completed:** 2026-08-14T14:19:20Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- **WR-01** — `finite_difference_jacobian`, `finite_difference_hessian` and `calibrate_fd_step` now raise a friendly `ValueError` naming `model.double()` for a float32-parameter model, matching the message `derivative_agreement` already produced. `_assert_float64` is now called from exactly one place (`derivative_agreement`, passing the model object), confirmed by `grep -n '_assert_float64'` showing only the import and that one call.
- **WR-03** — `calibrate_fd_step` and `derivative_agreement` now compute their comparison autodiff Hessian through one shared helper, `_chunked_vmap_hessian`, chunked at `chart_curvature.VMAP_CHUNK` with the same padding discipline `derivative_agreement`'s block already used. Correctness above `VMAP_CHUNK` no longer depends on `BRIDGE_N_POINTS == VMAP_CHUNK` being numerically equal.
- **WR-02** — `_agreement_stats` gained a `rel_floor` keyword (`DEFAULT_REL_FLOOR = 1e-12`) and returns an additional `near_zero_reference_fraction` key alongside the existing absolute/relative columns, with a thin-denominator docstring caveat in `persistence_probe.max_persistence`'s style, naming the recorded `full_hess_max_abs_rel = 1.1351e+00` instance. No threshold or boolean introduced anywhere.
- Full `pu_manifold` test suite: 272 passed (269 baseline + 3 new regression tests), including every pre-existing `test_derivative_bridge.py` test unchanged.

## Task Commits

Each task was committed atomically:

1. **Task 1: WR-01 — probe-based float64 guard on the finite-difference side** - `3382665` (fix)
2. **Task 2: WR-03 chunked calibration Hessian, WR-02 near-zero-reference diagnostic** - `76f26f2` (fix)

**Plan metadata:** (this commit, docs)

## Files Created/Modified

- `notebooks/pu_manifold/derivative_bridge.py` — Added `_assert_decode_batch_float64` (z-only check), `_friendly_model_dtype_error` (shared message builder), `_chunked_vmap_hessian` (WR-03's shared chunked comparison Hessian), and `DEFAULT_REL_FLOOR` + `near_zero_reference_fraction` on `_agreement_stats` (WR-02). `_chunked_eval` and `calibrate_fd_step`'s autodiff call both translate a float32-model `RuntimeError` into the friendly `ValueError` at the point of the real invocation, rather than a separate probe call.
- `notebooks/pu_manifold/tests/test_derivative_bridge.py` — Three new regression tests: `test_derivative_bridge_float32_model_raises_friendly_value_error`, `test_calibrate_fd_step_chunks_above_vmap_chunk`, `test_derivative_bridge_reports_near_zero_relative_reference_fraction`.

## Decisions Made

- **The `02.6-REVIEW.md` WR-01 fix sketch (a direct probe call `decode_batch(z_chart[:1]).dtype`) does not work as literally written.** Verified directly: a float32-parameter decoder fed a float64 probe row raises a bare `RuntimeError` from its own internal matmul (`expected m1 and m2 to have the same dtype, but got: double != float`) before ever returning a dtype to inspect — the same failure WR-01 reports, reproduced inside the guard itself. A dedicated probe call would also add exactly one extra `decode_batch` invocation, breaking `test_finite_difference_hessian_invocation_count_matches_chunk_arithmetic`'s exact `ceil(batch * n_offsets / MAX_FD_ROWS)` count (confirmed by running it: `expected 1, got 2` at `batch=50`). Resolved by splitting the check: `_assert_decode_batch_float64` validates only `z_chart`'s dtype up front (no extra call), and the model-dtype half is enforced where `decode_batch` is genuinely invoked for real work — `_chunked_eval` (used by both `finite_difference_jacobian` and `finite_difference_hessian`) and `calibrate_fd_step`'s own autodiff Hessian call — both catching `RuntimeError` and translating it (plus a dtype check for the theoretical non-raising case) into the same friendly message. This satisfies the plan's `<behavior>` and `<acceptance_criteria>` exactly (friendly `ValueError` naming `model.double()`, zero change to pre-existing tests) while adding zero extra `decode_batch` calls anywhere.
- **`DEFAULT_REL_FLOOR = 1e-12`**, chosen relative to float64's own scale (`~2.2e-16` machine epsilon) rather than any acceptance bar, per the plan's explicit instruction and D-18's report-never-gate discipline.
- **`_chunked_vmap_hessian` computes only the Hessian, not the Jacobian.** `derivative_agreement` still computes its Jacobian in its own `VMAP_CHUNK`-chunked loop (needed for the pullback-metric solve, out of WR-03's scope) and now calls the shared helper separately for the comparison Hessian — two loops instead of one combined loop, trading a small amount of redundant chunk-padding work for a genuinely shared instrument between `calibrate_fd_step` and `derivative_agreement`, per the plan's explicit "do not invent a new chunk loop" instruction.
- **Renamed the WR-02 test to `test_derivative_bridge_reports_near_zero_relative_reference_fraction`** (plan named it `test_derivative_bridge_reports_near_zero_reference_fraction`) so it collects under the acceptance criterion's `-k "chunk or relative"` filter (which needs at least 3 tests; without "relative" in the name only 2 collected). The behavior tested is unchanged from the plan's specification.

## Deviations from Plan

None requiring Rule 4 — the two items above (WR-01's probe-call redesign, the test rename) are both within Rule 1 (fixing a fix sketch that does not actually work as written) and Rule 3 (unblocking an acceptance criterion), documented above rather than silently applied.

## Issues Encountered

- The plan's WR-01 pseudocode, taken verbatim, would have (a) raised the same undiagnosable `RuntimeError` it was meant to fix, since the probe call itself triggers the dtype-mismatched matmul, and (b) broken the pre-existing invocation-count regression test by adding an extra `decode_batch` call. Both were caught by actually running the test suite after a literal-transcription attempt, before committing; resolved as described in Decisions Made.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three `02.6-REVIEW.md` warnings (WR-01, WR-02, WR-03) are closed with regression tests; `derivative_bridge.py` is ready to be run at PU scale in plan 03-09 without carrying forward known correctness or reporting-clarity gaps.
- Full `pu_manifold` test suite (272 tests) is green; no existing tolerance was relaxed and no recorded Phase 02.6 number changes (verified by `test_derivative_agreement_end_to_end_sphere_known_answer` and `test_derivative_agreement_returns_separate_keys_no_combined_no_boolean` still passing unchanged).
- No blockers for the next plan in Phase 3's wave order.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-14*

## Self-Check: PASSED

- FOUND: notebooks/pu_manifold/derivative_bridge.py
- FOUND: notebooks/pu_manifold/tests/test_derivative_bridge.py
- FOUND: .planning/phases/03-decoder-curvature-field/03-03-SUMMARY.md
- FOUND: 3382665 (Task 1 commit)
- FOUND: 76f26f2 (Task 2 commit)
- FOUND: a55a19a (SUMMARY commit)
