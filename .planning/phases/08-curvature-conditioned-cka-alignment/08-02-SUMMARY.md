---
phase: 08-curvature-conditioned-cka-alignment
plan: 02
subsystem: testing
tags: [numpy, scipy, cka, hsic, kernel-methods, pytest, permutation-null]

# Dependency graph
requires:
  - phase: 08-01
    provides: "cka.py's proven unbiased-HSIC/CKA estimator (unbiased_hsic, cka, linear_gram,
      rbf_gram, median_pairwise_distance, cka_on_subset) and the 14-constant freeze-guard
      shell (_REQUIRED_CONSTANTS, assert_preregistered)"
provides:
  - "notebooks/pu_manifold/cka.py — the within-density-stratum tertile split
    (tertile_split_within_strata, realized_h_contrast), the tertile-difference panel and its
    label-permutation null (tertile_gap_panel, stratified_tertile_label_null, null_threshold),
    and the verdict layer (per_d_verdict, combine_seed_verdicts, pooled_field_guard) — 37
    gating constants total, all still UNSET, assert_preregistered() still raises"
  - "notebooks/pu_manifold/tests/test_cka.py — 47 new tests (71 total in this file) pinning
    the split's within-stratum property, its measured (not literally-as-planned) equal-n
    bound, the null's size-preservation/seed-reproducibility/no-Gram-rebuild properties, and
    every verdict/seed-guard branch"
affects: [08-03, 08-04, 08-05, 08-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Within-stratum contiguous rank-block split (n_s // 3, remainder to the last block),
      mirroring density_strata's own remainder-to-last-stratum convention at one level down"
    - "Precompute-all-permutations-before-the-loop optimization applied at the label-
      permutation level (stratified_partial_null's own optimization, one level down)"
    - "assert_preregistered() extended beyond its 08-01 generic UNSET sweep with two
      exact-content checks (SEED_HANDLING_RULE by string equality, VERDICT_RULE naming
      S_GRID), mirroring linear_probe.py's own guard shape"

key-files:
  created: []
  modified:
    - notebooks/pu_manifold/cka.py
    - notebooks/pu_manifold/tests/test_cka.py

key-decisions:
  - "test_tertile_split_equal_n_at_every_s's bound corrected from the plan's literal 'at most
    the number of strata' to the measured 'at most 2 * number of strata': dividing a stratum
    of size n_s into 3 blocks leaves a remainder of 0, 1 OR 2 (n_s % 3), and at n=10,000 the
    grid values S=20 and S=50 both have n_s % 3 == 2 (n_s=500, 200), not 1. The pooled
    max-min difference is exactly S * (n_s % 3), verified directly against the algorithm the
    plan itself specifies (remainder-to-last-block, matching density_strata's own
    convention) -- the plan's arithmetic assumed remainder <= 1, which is only true for
    S=10 (n_s=1000, remainder=1) among the three grid values."
  - "combine_seed_verdicts' four clearance counts (0/1/2/3) map to exactly THREE terminal
    outcome strings, not four: counts 1 and 2 both give the single terminal 'SPLIT ACROSS
    SEEDS', matching D8-15's own must_haves truth ('returns a terminal split outcome for one
    or two clearances without upgrading by majority vote') and linear_probe.py's own
    three-outcome precedent. The plan's acceptance-criteria phrasing ('four distinct outcome
    strings') is corrected to the ratified vocabulary; upgrading a 2-of-3 split to a fourth,
    more favorable string would itself be the majority-vote upgrade D8-15 forbids."

patterns-established:
  - "Verdict functions (per_d_verdict) take only the pre-reduced gap/threshold values they
    need, never the full tertile-gap panel, so the middle tertile's CKA is structurally
    unreadable rather than merely unread by convention."

requirements-completed: [D8-01, D8-05, D8-06, D8-08, D8-09, D8-10, D8-11, D8-12, D8-13, D8-15, D8-20, D8-22, D8-23]

coverage:
  - id: D1
    description: "Within-density-stratum tertile split: disjoint, covers every index once,
      density-marginal-matched up to each stratum's own remainder, tertile 3 holds the
      highest-h third WITHIN every stratum (not globally)"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_tertile_within_stratum_split"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_tertile_split_density_marginals_match"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_tertile_split_equal_n_at_every_s"
        status: pass
    human_judgment: false
  - id: D2
    description: "realized_h_contrast and the split's ValueError guards (length mismatch,
      non-finite h, stratum below 3 points)"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_realized_h_contrast_exceeds_one_for_nonconstant_h"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_tertile_split_raises_on_length_mismatch"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_tertile_split_raises_on_small_stratum"
        status: pass
    human_judgment: false
  - id: D3
    description: "Tertile-difference panel (per-kernel cka_t1/t2/t3/gap) and its key-mismatch
      guard"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_tertile_gap_panel_returns_gap_per_kernel"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_tertile_gap_panel_raises_on_key_mismatch"
        status: pass
    human_judgment: false
  - id: D4
    description: "Within-stratum label-permutation null: preserves observed tertile sizes on
      every resample, one array per kernel, seed-reproducible, never rebuilds a Gram matrix"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_stratified_tertile_null_preserves_sizes"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_null_panel_has_one_array_per_kernel"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_null_is_seed_reproducible"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_null_does_not_rebuild_grams"
        status: pass
      - kind: other
        ref: "grep -n permutation_type notebooks/pu_manifold/cka.py (empty)"
        status: pass
    human_judgment: false
  - id: D5
    description: "null_threshold's two-tailed empirical quantile pair"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_null_threshold_is_two_tailed"
        status: pass
    human_judgment: false
  - id: D6
    description: "per_d_verdict: clearance only when every S clears, per-S detail retained,
      middle tertile structurally unread, independent per-d/seed cell"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_verdict_requires_clearance_at_every_s"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_middle_tertile_does_not_gate"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_per_d_verdicts_are_independent"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_per_d_verdict_requires_rule"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_per_d_verdict_raises_on_key_mismatch"
        status: pass
    human_judgment: false
  - id: D7
    description: "combine_seed_verdicts: exactly-three enforcement, unanimous-or-nothing
      3-outcome mapping (never upgraded by majority vote), invalid-value guard"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_combine_seed_verdicts_requires_exactly_three"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_combine_seed_verdicts_requires_rule"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_combine_seed_verdicts_rejects_invalid_verdict_value"
        status: pass
    human_judgment: false
  - id: D8
    description: "pooled_field_guard raises naming 05-03-DECISION.md for more than one seed
      field"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_seed_pooling_raises"
        status: pass
    human_judgment: false
  - id: D9
    description: "assert_preregistered's two new exact-content guards (SEED_HANDLING_RULE
      equality, VERDICT_RULE names S_GRID) beyond the generic UNSET sweep; 37 constants
      total, all still UNSET"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_assert_preregistered_rejects_wrong_seed_handling_rule"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_assert_preregistered_rejects_verdict_rule_missing_s_grid"
        status: pass
      - kind: other
        ref: ".venv/bin/python -c \"...print(len(cka._REQUIRED_CONSTANTS))\" -> 37"
        status: pass
      - kind: other
        ref: ".venv/bin/python -c \"...cka.assert_preregistered()\" -> RuntimeError (KERNELS UNSET)"
        status: pass
    human_judgment: false
  - id: D10
    description: "src/effdim/ untouched; no PU data opened; no torch import; only cka.py and
      tests/test_cka.py modified since the plan's base commit"
    verification:
      - kind: other
        ref: "git diff --name-only c34ba15..HEAD -- notebooks/ src/ -> 08_cka_alignment_run.py (08-01, unmodified), cka.py, tests/test_cka.py"
        status: pass
      - kind: other
        ref: "grep -n 'notebooks/.cache\\|np.load' notebooks/pu_manifold/cka.py (empty)"
        status: pass
    human_judgment: false

duration: ~40min
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 02: CKA Tertile Split, Label-Permutation Null, and Verdict Rules Summary

**Within-density-stratum `||H||` tertile split, its within-stratum label-permutation null, and
the full verdict layer (clearance-at-every-`S`, per-`d` independence, seed unanimity, four
non-gating declarations) added to `cka.py` — all 37 Phase 8 gating constants remain UNSET, 71
tests pass in `test_cka.py`, full suite 742 passed / 1 skipped.**

## Performance

- **Duration:** ~40 min (three tasks, each committed atomically)
- **Started:** 2026-08-28T01:13:25Z (per STATE.md's prior session marker)
- **Completed:** 2026-08-28T01:44:02Z
- **Tasks:** 3/3
- **Files modified:** 2 (`notebooks/pu_manifold/cka.py`, `notebooks/pu_manifold/tests/test_cka.py`)

## Accomplishments

- `tertile_split_within_strata(h, strata)` splits every density stratum into three contiguous
  rank blocks (`n_s // 3` each, remainder to the last/highest-`h` block), pools across strata,
  and returns three disjoint, exhaustive index arrays. Measured: tertile 3 draws exactly one
  third of EVERY stratum, not only the globally highest points — confirmed on a synthetic field
  designed so a global top-third split would look entirely different.
- `realized_h_contrast(h, tertiles)` — D8-21's mandatory reporting number — returns the
  tertile-3/tertile-1 median ratio.
- `tertile_gap_panel(K_full, L_full, tertiles)` computes `cka_t1`/`cka_t2`/`cka_t3`/`gap` per
  kernel from already-built Gram matrices; the middle tertile's value is present but
  structurally never read by any verdict function (verified: sabotaging `cka_t2` to `NaN` in a
  live panel leaves `per_d_verdict`'s output byte-identical).
- `stratified_tertile_label_null(...)` — D8-11's null — permutes `||H||` labels within density
  strata, precomputes all resample permutations before the recomputation loop, and never calls
  `linear_gram`/`rbf_gram` inside it (verified two ways: `inspect.getsource` string absence, and
  a monkeypatch test that would raise `AssertionError` if either builder were called). **Measured
  wallclock for one 25-resample, 2-kernel null at n=600: 0.2746s (≈11.0ms/resample)** — the
  number 08-05 can extrapolate the real `d`×seed×`S` grid's null cost from.
- `null_threshold`, `per_d_verdict`, `combine_seed_verdicts`, `pooled_field_guard` complete the
  verdict layer: clearance fires only when every `S` in the grid clears (`n_s_cleared` reported
  alongside so a reader sees exactly which `S` failed); seeds combine via the exact
  `linear_probe.py` three-outcome shape (`CLEARS IN ALL THREE SEEDS` / `NO CLEARANCE IN ANY
  SEED` / `SPLIT ACROSS SEEDS`), never upgraded by majority vote; `pooled_field_guard` raises
  naming `05-03-DECISION.md` for any attempt to combine more than one seed field.
- `assert_preregistered()` gained two exact-content guards beyond its 08-01 generic UNSET sweep:
  `SEED_HANDLING_RULE` by EXACT STRING EQUALITY to `"no_pooling_per_seed_verdicts"` (not
  truthiness), and `VERDICT_RULE` must name `S_GRID` — both mirroring `linear_probe.py`'s own
  guard shape, both verified with dedicated wrong-value tests.
- 37 Phase 8 gating constants total (`_REQUIRED_CONSTANTS`), all still UNSET;
  `assert_preregistered()` raises on `KERNELS` (first in declaration order), exactly as at
  08-01's close — no number has been computed, no constant has been filled.
- `test_cka.py`: 71 tests pass (47 net new). Full `notebooks/pu_manifold/tests/` suite: **742
  passed, 1 skipped** (baseline before this plan: 695 passed, 1 skipped).
- `08_cka_alignment_run.py --mode selfcheck` still exits 0 (the D8-16 invariance ladder,
  unregressed) — this plan did not touch that file.
- `src/effdim/` untouched; only `cka.py` and `tests/test_cka.py` modified since the plan's base
  commit (confirmed by `git diff --name-only c34ba15..HEAD -- notebooks/ src/`); no
  `notebooks/.cache/` read or `np.load` call anywhere in `cka.py`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Within-density-stratum tertile split and the realized-contrast diagnostic** -
   `8758f40` (feat)
2. **Task 2: Tertile-difference panel and the within-stratum label-permutation null** -
   `1e26178` (feat)
3. **Task 3: Verdict rules — clearance at every S, per-d independence, seed unanimity, and the
   four non-gating declarations** - `bd0eb31` (feat)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md)

## Files Created/Modified

- `notebooks/pu_manifold/cka.py` - +8 constants (Task 1), +7 constants (Task 2), +8 constants
  (Task 3, 37 total); `tertile_split_within_strata`, `realized_h_contrast`,
  `tertile_gap_panel`, `stratified_tertile_label_null`, `null_threshold`, `per_d_verdict`,
  `combine_seed_verdicts`, `pooled_field_guard`; two new exact-content checks in
  `assert_preregistered`
- `notebooks/pu_manifold/tests/test_cka.py` - 47 new tests across the three tasks; `dsn` import
  added for synthetic strata generation via the real `density_strata` binning logic;
  `_PLAUSIBLE_FILLED_VALUES` extended with all 23 new constants' plausible filled values

## Decisions Made

- **`density_stratified_null.density_strata` is imported in `test_cka.py`, not in `cka.py`
  itself.** `cka.py`'s functions all take an already-computed `strata` array as a parameter
  (never calling `density_strata` themselves), so the "imported as a pure function, never for a
  gating value" boundary the plan's threat model describes is satisfied at the test-file call
  site, which builds synthetic strata via the real production binning logic rather than
  hand-rolling an approximation of it. `cka.py` gains no new import.
- Both corrected acceptance criteria documented above under `key-decisions` (the equal-n bound
  and the seed-combination outcome count) were measured directly against the plan's own
  specified algorithm before deciding they were plan bugs, not implementation bugs — same
  verification discipline `08-01-SUMMARY.md` applied to its own test-description correction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan's test acceptance bound] Corrected `test_tertile_split_equal_n_at_every_s`'s max-min bound from "at most S" to "at most 2*S"**
- **Found during:** Task 1, before writing the test (derived by hand from the plan's own
  specified split algorithm, then confirmed numerically)
- **Issue:** The plan's acceptance criterion states the pooled max-minus-min tertile size is
  "at most the number of strata." Dividing a stratum of size `n_s` into 3 contiguous blocks
  (`n_s // 3` each, remainder to the last block — the exact algorithm the plan itself
  specifies, mirroring `density_strata`'s own convention) leaves a remainder `n_s % 3 ∈ {0, 1,
  2}`. At `n=10,000`, `S ∈ {10, 20, 50}` gives `n_s ∈ {1000, 500, 200}` and `n_s % 3 ∈ {1, 2,
  2}` respectively — only `S=10` matches the plan's literal "at most S" bound; `S=20` and
  `S=50` both produce a measured max-min difference of `2*S` (40 and 100 respectively), not
  `S`. This is unsatisfiable by a correct implementation of the algorithm the plan itself
  specifies — the plan's arithmetic assumed a remainder of at most 1.
- **Fix:** Implemented `tertile_split_within_strata` exactly as specified (remainder-to-last-
  block). Wrote `test_tertile_split_equal_n_at_every_s` to assert the TRUE, measured invariant
  (`max - min == S * (n_s % 3)`, bounded above by `2*S`), documented in the test's own
  docstring and here.
- **Files modified:** `notebooks/pu_manifold/tests/test_cka.py`
- **Verification:** Test passes at all three grid values; the exact `S * (n_s % 3)` prediction
  is asserted, not only the loose bound, so the fix is pinned precisely rather than merely
  loosened.
- **Committed in:** `8758f40` (Task 1 commit)

**2. [Rule 1 - Bug in plan's test acceptance description] Corrected `test_combine_seed_verdicts_requires_exactly_three`'s "four distinct outcome strings" to the ratified three-outcome mapping**
- **Found during:** Task 3, before writing the test (cross-checked against the plan's own
  `must_haves.truths` entry for this function and `linear_probe.py`'s precedent)
- **Issue:** The plan's acceptance criterion asks for "clearance counts 0, 1, 2 and 3 returning
  the four distinct outcome strings." The SAME plan's `must_haves.truths` states
  `combine_seed_verdicts` "returns a terminal split outcome for one or two clearances without
  upgrading by majority vote" — i.e. counts 1 and 2 must map to the SAME terminal string, not
  two different ones. Four counts mapping to four distinct strings is inconsistent with the
  ratified D8-15 vocabulary (`linear_probe.py`'s own `combine_seed_verdicts` has exactly three
  outcomes: three-of-three, zero, one-or-two) and would itself constitute the majority-vote
  upgrade D8-15 explicitly forbids.
- **Fix:** Implemented the three-outcome mapping (`CLEARS IN ALL THREE SEEDS` /
  `NO CLEARANCE IN ANY SEED` / `SPLIT ACROSS SEEDS`), matching `linear_probe.py`'s shape
  exactly. Test asserts all four clearance counts against this true mapping (counts 1 and 2
  both yield `SPLIT ACROSS SEEDS`), documented in the test's own docstring and here.
- **Files modified:** `notebooks/pu_manifold/tests/test_cka.py`
- **Verification:** Test passes; cross-checked against the plan's own `must_haves.truths` text
  and `linear_probe.combine_seed_verdicts`'s source (read directly during planning-context
  gathering) before deciding this was a plan-description bug rather than an implementation gap.
- **Committed in:** `bd0eb31` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — plan test-specification bugs, same class as
`08-01-SUMMARY.md`'s `test_double_centering_changes_the_answer` correction, corrected before
committing)
**Impact on plan:** Both corrections strengthen rather than weaken the intended protections
(equal-n subsets up to a small bounded remainder; unanimous-or-nothing seed combination that
never upgrades a split by majority vote). No scope creep; no change to any Task 1/2/3 action
item beyond the two test assertions named above.

## Issues Encountered

None. All three tasks' acceptance criteria were run and confirmed directly (not assumed) before
each commit: per-task pytest subsets, the growing `_REQUIRED_CONSTANTS` count (22 → 29 → 37),
`assert_preregistered()`'s non-zero exit at every stage, the `permutation_type`/Gram-builder
absence greps, the full `notebooks/pu_manifold/tests/` suite (742 passed, 1 skipped), the
`08_cka_alignment_run.py --mode selfcheck` re-run, and the `git diff --name-only` purity check
against `src/effdim/` and every non-`cka.py`/`test_cka.py` file under `notebooks/`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `cka.py` now carries the complete split/null/verdict machinery 08-03 (the runner) and 08-04
  (the freeze) will drive: `tertile_split_within_strata`, `realized_h_contrast`,
  `tertile_gap_panel`, `stratified_tertile_label_null`, `null_threshold`, `per_d_verdict`,
  `combine_seed_verdicts`, `pooled_field_guard`, all covered by unit tests on synthetic fields.
- All 37 gating constants remain UNSET, as required — 08-04 is still the single commit that may
  fill them. `assert_preregistered()`'s two new exact-content checks (`SEED_HANDLING_RULE`
  equality, `VERDICT_RULE` naming `S_GRID`) are already in place for that freeze to satisfy.
- Measured null wallclock (0.2746s for 25 resamples × 2 kernels at n=600) is available for
  08-05's real-grid runtime extrapolation.
- 08-03 also touches `notebooks/diagnostics/08_cka_alignment_run.py`, which this plan did not
  modify — no collision.
- No blockers. `src/effdim/` confirmed untouched; only `cka.py` and `tests/test_cka.py` changed.

---
*Phase: 08-curvature-conditioned-cka-alignment*
*Completed: 2026-08-28*

## Self-Check: PASSED
