---
phase: 07-curvature-conditioned-crossmodal-alignment
plan: 01
subsystem: research-instrumentation
tags: [pre-registration, freeze, mknn, curvature, spearman, permutation-test, pytest]

# Dependency graph
requires:
  - phase: 04-region-partitioning-regional-alignment-mknn
    provides: "HEADLINE_K=20, MKNN_K_GRID, N_PERMUTATIONS, DENSITY_K, DENSITY_FIELD_D constants re-declared as literals; the region_partition.py structural precedent"
  - phase: 05-decoder-curvature-conditioned-linear-decodability
    provides: "SPLIT ACROSS SEEDS verdict-vocabulary precedent (mirrored by SPLIT ACROSS d); measured seed-instability finding inherited as SEED_HANDLING_RULE's accepted limitation"
  - phase: 06-point-cloud-curvature-conditioned-linear-decodability
    provides: "pointcloud_probe.py structural template (constants block, VERDICT_RULE literal, _REQUIRED_CONSTANTS, assert_preregistered, verdict_is_terminal, describe_inheritance)"
provides:
  - "crossmodal_curvature.py: Phase 7's full pre-registration constants block (D_SWEEP, fit protocol, ALIGNMENT_METRIC, significance/tie-handling rules, density/hubness constants, positive-control mechanism, VERDICT_RULE, VERDICT_VALUES)"
  - "assert_preregistered() gate, verdict_is_terminal(), describe_inheritance() audit surface"
  - "A frozen freeze commit (f032745) that is the strict git ancestor every later Phase 7 PU number must be proven against"
  - "118-test guard suite pinning every constant's malformed-value boundary plus the strict-ancestor proof shape"
affects: [07-02-plan, 07-03-plan, 07-04-plan, 07-05-plan]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Freeze-first plan structure: a checkpoint:decision ratifies open questions BLIND (before any number exists), then a constants-only commit is the freeze, then a separate commit pins the guard with tests -- three commits, three roles, never merged"
    - "Strict-ancestor proof shape: git merge-base --is-ancestor alone is insufficient (a commit is its own ancestor); git rev-list --count <freeze>..HEAD >= 1 closes that gap"
    - "Re-declare, never import, for cross-phase constant inheritance -- makes a same-named constant collision structurally impossible and the inheritance diffable by eye (describe_inheritance())"

key-files:
  created:
    - notebooks/pu_manifold/crossmodal_curvature.py
    - notebooks/pu_manifold/tests/test_crossmodal_curvature.py
  modified: []

key-decisions:
  - "Task 1 checkpoint ratified ratify-all: all six open pre-registration decisions accepted exactly as the planner stated them (HEADLINE_K=20, two-tailed permutation route, four-value VERDICT_VALUES, notebooks/.cache/ record location, single seed across the d-sweep, positive control frozen in the same commit)."
  - "D7-03 flagged assumption ACCEPTED: density statistic reported on 1.0/w (not raw w), matching Phase 4's REGN-01 sign convention."
  - "D7-07 flagged assumption ACCEPTED: ALIGNMENT_METRIC='mknn' frozen as a checkable constant carried on every record row, proving CKA's exclusion positively rather than only in prose."
  - "SEED_HANDLING_RULE='single_seed_across_d_sweep' is recorded as an ACCEPTED LIMITATION inherited from Phase 5's measured seed-instability finding, written into both the module docstring and VERDICT_RULE's own text -- never presented as a silent stability assumption."
  - "The freeze commit (Task 2) contains crossmodal_curvature.py alone -- no numpy/scipy import, no compute function -- so the ancestry proof in Task 3 is unambiguous: everything after this commit is number-producing code, everything at or before it is not."

requirements-completed: [D7-01, D7-02, D7-03, D7-04, D7-05, D7-06, D7-07]

coverage:
  - id: D1
    description: "Six open pre-registration decisions ratified blind at a checkpoint:decision, before any PU number existed"
    verification:
      - kind: manual_procedural
        ref: "Task 1 checkpoint:decision, resolved ratify-all by human decision"
        status: pass
    human_judgment: true
    rationale: "A pre-registration ratification is inherently a human decision, not a code-verifiable deliverable."
  - id: D2
    description: "crossmodal_curvature.py freeze commit: constants block, VERDICT_RULE, VERDICT_VALUES, assert_preregistered(), verdict_is_terminal(), describe_inheritance() -- no compute functions"
    requirement: "D7-01, D7-02, D7-03, D7-04, D7-05, D7-06, D7-07"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_crossmodal_curvature.py -- 118 tests, all pass"
        status: pass
      - kind: other
        ref: "git show --stat f032745 -- exactly one file changed"
        status: pass
    human_judgment: false
  - id: D3
    description: "Freeze-guard test suite: malformed-constant boundary sweep over every _REQUIRED_CONSTANTS entry, D7-01/D7-02 boundary checks, and the strict-ancestor proof (git rev-list --count >= 1, not merely --is-ancestor)"
    requirement: "D7-06"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_freeze_commit_is_a_strict_ancestor_of_head"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -- full suite, 533 passed / 2 skipped, no regressions"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-08-26
status: complete
---

# Phase 7 Plan 1: Freeze Phase 7 Pre-Registration Summary

**Froze Phase 7's constants (D-sweep, two-tailed permutation significance, four-value verdict vocabulary, positive-control mechanism) into `crossmodal_curvature.py` before any PU number exists, at commit `f032745`, and pinned the guard with a 118-test suite proving strict git ancestry.**

## Performance

- **Duration:** 11 min (this continuation session; Task 1's checkpoint was resolved in a prior session)
- **Started:** 2026-08-26T08:07:00-04:00 (approx., this session)
- **Completed:** 2026-08-26T08:18:21-04:00
- **Tasks:** 3 (Task 1 resolved at checkpoint in a prior session; Tasks 2-3 executed this session)
- **Files modified:** 2 created

## Accomplishments

- **The freeze commit (`f032745f6450068c63763993d39fa112fd36bb8c`).** `notebooks/pu_manifold/crossmodal_curvature.py` committed alone, containing only the constants block, `VERDICT_RULE`, `VERDICT_VALUES`, `_REQUIRED_CONSTANTS`, `assert_preregistered()`, `verdict_is_terminal()`, and `describe_inheritance()`. No numpy, scipy, torch, or `pu_manifold` imports — only `typing`. No compute functions. `git show --stat HEAD` confirms exactly one file changed. **Every later Phase 7 PU number must be proven a strict git descendant of this commit.**
- **All six Task 1 decisions ratified `ratify-all`** and encoded as frozen constants: `HEADLINE_K = 20` inherited from Phase 4; the two-tailed permutation significance route (`SIGNIFICANCE_TAIL_RULE`, calling `curvature_probe.permutation_null` on both `(H, MKNN)` and `(-H, MKNN)` at `NULL_QUANTILE_PER_TAIL = 0.975`); `VERDICT_VALUES` as the four-tuple `("ASSOCIATION DETECTED", "NO DETECTABLE RELATIONSHIP", "SPLIT ACROSS d", "UNDERPOWERED -- NO CLAIM")`; the record path `notebooks/.cache/07_crossmodal_curvature.jsonl` via `cache.cache_path`; `SEED_HANDLING_RULE = "single_seed_across_d_sweep"`; and the positive control (`POSITIVE_CONTROL_TARGET_RHOS`, `POSITIVE_CONTROL_SEED`, `POSITIVE_CONTROL_RULE`) frozen in the same commit as everything else.
- **Both flagged planner assumptions accepted**: D7-03's density statistic is reported on `1.0 / w` (matching Phase 4's REGN-01 sign convention, `DENSITY_SIGN_CONVENTION`); D7-07's `ALIGNMENT_METRIC = "mknn"` is frozen as a checkable constant carried on every future record row, proving CKA's exclusion positively.
- **118-test freeze-guard suite** (`test_crossmodal_curvature.py`): parameterized None/absent/blank-string/empty-tuple sweep over every one of the 39 `_REQUIRED_CONSTANTS` entries; the D7-01 `D_SWEEP` non-positive/non-int boundary; the D7-02 `POSITIVE_CONTROL_TARGET_RHOS` strict-increasing boundary; `test_required_constants_covers_every_frozen_constant` (bidirectional set equality between `_REQUIRED_CONSTANTS` and every module-level `UPPER_CASE` name); `VERDICT_RULE` caveat-coverage check; and `test_freeze_commit_is_a_strict_ancestor_of_head`, which asserts BOTH `git merge-base --is-ancestor f032745 HEAD` exits 0 AND `git rev-list --count f032745..HEAD >= 1` — the precision D7-06 requires, since a commit is its own ancestor and `--is-ancestor` alone would pass even if a number were produced in the freeze commit itself. Confirmed passing (not skipped) once the test file's own commit landed after the freeze.
- **No regression:** full `notebooks/pu_manifold/tests/` suite (533 passed, 2 skipped) is green after both commits.
- **No sealed module touched:** `git diff --stat` across the whole plan against `mknn.py`, `linear_probe.py`, `pointcloud_probe.py`, `cae.py`, `decoder_curvature.py`, `curvature_probe.py`, `cross_split_curvature.py`, and `src/effdim/` produces empty output.

## Task Commits

Each task was committed atomically:

1. **Task 1: Ratify the six open pre-registration decisions, blind** — no commit (pure decision gate; resolved in a prior session's `checkpoint:decision`, `ratify-all`)
2. **Task 2: Commit the pre-registration constants block — THE FREEZE COMMIT** — `f032745` (feat)
3. **Task 3: Pin the freeze guard with tests, including the strict-ancestor boundary** — `9f16332` (test)

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `notebooks/pu_manifold/crossmodal_curvature.py` — Phase 7's frozen pre-registration constants: field/instrument, fit protocol, alignment statistic, significance, density/hubness, positive control, provenance/record, `VERDICT_RULE`, `VERDICT_VALUES`, `assert_preregistered()`, `verdict_is_terminal()`, `describe_inheritance()`.
- `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` — 118-test guard suite: malformed-constant sweep, D7-01/D7-02 boundary checks, caveat coverage, and the strict-ancestor freeze proof.

## Decisions Made

See `key-decisions` in frontmatter above. In short: `ratify-all` on all six Task 1 recommendations; both flagged planner assumptions (D7-03 density sign convention, D7-07 `ALIGNMENT_METRIC` as a checkable constant) accepted as proposed; single seed across the `d`-sweep recorded explicitly as an accepted limitation, never a silent assumption.

## Deviations from Plan

None — plan executed exactly as written, with one clarification: the plan's Task 2 `<acceptance_criteria>` prose describes the required-constants equality check without excluding underscore-prefixed names (`{n for n in vars(cc) if n.isupper()}`), while Task 3's `<action>` explicitly specifies excluding them (`n.isupper() and not n.startswith("_")`). Task 3's more precise instruction was followed, since a literal reading of Task 2's prose would require `_REQUIRED_CONSTANTS` to list itself, which no version of this pattern (including the `pointcloud_probe.py` template it mirrors) does. This is a wording imprecision in the plan text, not a deviation in the code: `set(cc._REQUIRED_CONSTANTS)` equals `{n for n in vars(cc) if n.isupper() and not n.startswith("_")}` exactly, as `test_required_constants_covers_every_frozen_constant` proves.

Also fixed one cosmetic issue during authoring, before any commit: a `VERDICT_RULE` prose line originally began with the word "from" at the start of a line, which made `grep -E '^(from|import)'` (an acceptance-criteria check for import-only lines) return a false positive from inside the docstring. Reworded the sentence so no docstring line begins with `from`/`import`; the grep now returns exactly one line (`from typing import Any, Dict`).

## Issues Encountered

None. The full `notebooks/pu_manifold/tests/` suite takes ~2m25s to run (533 tests, mostly torch-backed); this was run once as a final regression check after Task 3's commit and is not part of Task 3's own `<verify>` step (which targets only `test_crossmodal_curvature.py`, completing in under 1 second).

## Known Stubs

None. This module deliberately contains no compute functions by design (D7-06's freeze-before-compute ordering constraint) — plan 07-02 adds them. This is not a stub; it is the plan's stated scope boundary, restated in the module's own docstring and enforced by the acceptance criteria's "no numpy/scipy import" check.

## Threat Flags

None. `T-07-02` (the pre-registered-constants tampering threat) is the one `high`-severity item in this plan's threat model, and it is fully mitigated by `assert_preregistered()` plus the strict-ancestor proof — both delivered in this plan, not deferred.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Plan 07-02 may now begin: `crossmodal_curvature.py` is frozen, its guard is tested, and the freeze commit SHA (`f032745f6450068c63763993d39fa112fd36bb8c`) is recorded here for 07-02 and 07-04 to read and prove ancestry against. No PU number exists anywhere in the tree. Plan 07-02 must call `assert_preregistered()` first in every number-producing code path and must not edit any constant frozen here without recording the edit as a pre-registration BREACH.

---
*Phase: 07-curvature-conditioned-crossmodal-alignment*
*Completed: 2026-08-26*

## Self-Check: PASSED

- FOUND: `notebooks/pu_manifold/crossmodal_curvature.py`
- FOUND: `notebooks/pu_manifold/tests/test_crossmodal_curvature.py`
- FOUND: `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-01-SUMMARY.md`
- FOUND commit `f032745` in `git log --oneline --all`
- FOUND commit `9f16332` in `git log --oneline --all`
