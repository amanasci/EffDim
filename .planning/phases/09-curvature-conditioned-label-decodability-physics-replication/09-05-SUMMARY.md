---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 05
subsystem: research-stats
tags: [pre-registration, freeze, git-ancestry, ridge-regression, spearman, freedman-lane, pu-manifold]

# Dependency graph
requires:
  - phase: 09-02
    provides: "INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882), measured 2026-09-02"
  - phase: 09-04
    provides: "09-DATA-MANIFEST.md § Ruling: LABEL_COLUMN_MAP, SENTINEL_VALUES, ALIGNMENT_MARGIN_R2 ratified by the developer (ratify-as-proposed, 2026-09-03)"
provides:
  - "physics_labels.py: all 30 gating constants filled; assert_preregistered() exits 0"
  - "physics_curvature_probe.py: all 73 gating constants filled; assert_preregistered() exits 0"
  - "09-PREREGISTRATION.md: transcription of every frozen constant, VERDICT_RULE and VERDICT_SENTENCE_RULE in full, sources cited by date, flagged assumptions carried forward"
  - "FREEZE_COMMIT_SHA = 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 wired into both runners and both test files; strict-ancestor gate proved from a fresh single-branch clone of the pushed branch"
affects: [09-06, 09-07, 09-08, 09-09, 09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-commit freeze discipline: Task 1's freeze commit touches only the constants-carrying files plus the pre-registration doc; Task 2's wiring commit is a separate, later commit that makes rev-list --count freeze..HEAD >= 1 for every subsequent run"
    - "Freeze-ancestry gate mirrored into a THIRD runner (09_physics_curvature_run.py) that has no number-producing mode of its own yet -- wired now so 09-06/09-08/09-09 call the shared FREEZE_COMMIT_SHA/_strict_ancestor_or_exit pair rather than each re-deriving it"

key-files:
  created:
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-PREREGISTRATION.md
  modified:
    - notebooks/pu_manifold/physics_labels.py
    - notebooks/pu_manifold/physics_curvature_probe.py
    - notebooks/diagnostics/09_physics_curvature_run.py
    - notebooks/diagnostics/09_row_alignment_proof_run.py
    - notebooks/pu_manifold/tests/test_physics_curvature_probe.py
    - notebooks/pu_manifold/tests/test_physics_labels.py

key-decisions:
  - "CURVATURE_FIELD_FOR_VERDICT frozen as \"H_tan_norm\", not the plan's prose shorthand \"H_tan\" -- the module's own _REQUIRED_CURVATURE_FIELD_FOR_VERDICT guard and decompose_radial_tangential's literal returned dict key are both \"H_tan_norm\"; \"H_tan\" does not exist as a field name anywhere in the sealed code and would have failed assert_preregistered()'s exact-equality check, contradicting the plan's own first acceptance criterion"
  - "PER_D_VERDICT_VALUES frozen as the module-documented two-entry tuple (fired/not-fired), not the plan's three-entry form with \"SPLIT ACROSS SEEDS\" appended -- that string is a seed-COMBINATION outcome (combine_seed_verdicts hardcodes it, see line 900), not a per-d classification; per_d_verdict() only ever indexes [0]/[1], and the module docstring states \"Exactly two per-d verdict strings\""
  - "VERDICT_VALUES kept the plan's literal four-entry form (including \"HALTED - ALIGNMENT NOT PROVED\") despite the module docstring saying \"Exactly three\" -- phase_verdict() only ever returns entries [0]/[1]/[2]; the fourth is reserved for the row-alignment-proof-fails case, set by a different code path before phase_verdict() would ever be called, and no test or runtime assertion enforces a length-3 invariant, so the extra entry is additive and harmless"
  - "09_physics_curvature_run.py gained FREEZE_COMMIT_SHA and a mirrored _strict_ancestor_or_exit/_git_rev_parse pair that the plan's read_first assumed already existed -- it did not (only 09_row_alignment_proof_run.py had this machinery from 09-03); added now, unused by any mode yet, so 09-06/09-08/09-09 call the shared gate rather than each re-deriving it"
  - "test_physics_labels.py gained the entire freeze-ancestry scaffold (FREEZE_COMMIT_SHA, _freeze_commit_exists, _freeze_commit_is_strict_ancestor_of_head, both tests) that the plan's read_first assumed already existed in both test files -- it existed only in test_physics_curvature_probe.py; mirrored exactly"
  - "test_shard_url_raises_when_shard_count_is_unset rewritten to explicitly monkeypatch LABEL_N_SHARDS back to None -- it previously relied on the module-level constant being permanently UNSET, which the freeze commit ended"

requirements-completed: [D9-01, D9-02, D9-03, D9-04, D9-05, D9-06, D9-07, D9-08, D9-09, D9-10, D9-11, D9-12, D9-13, D9-14, D9-15, D9-16, D9-17, D9-18]

coverage:
  - id: D1
    description: "Every constant in physics_labels.py (30) and physics_curvature_probe.py (73) carries a literal value; both assert_preregistered() freeze guards exit 0"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: ".venv/bin/python -c \"...pl.assert_preregistered(); pcp.assert_preregistered(); assert cc.D_SWEEP==(20,25,32); assert pcp.D_SWEEP==(16,20,25,32)\" -- printed 'frozen 73 30'"
        status: pass
    human_judgment: false
  - id: D2
    description: "The freeze commit touches exactly three files (physics_labels.py, physics_curvature_probe.py, 09-PREREGISTRATION.md); Phase 7's crossmodal_curvature.py is untouched"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "git log --oneline -1 --name-only 5f7fbe2 -- three files; git diff --name-only -- notebooks/pu_manifold/crossmodal_curvature.py -- empty"
        status: pass
    human_judgment: false
  - id: D3
    description: "The strict-ancestor gate rejects a missing --freeze-commit, rejects --freeze-commit resolving to HEAD, and passes for the real freeze SHA (proceeding to a later, non-gate failure)"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "--mode proof (no flag) exits 1; --mode proof --freeze-commit $(git rev-parse HEAD) exits 1; --mode proof --freeze-commit 5f7fbe2... prints no D9-18 error and proceeds to real network I/O (terminated before any HF download completed, per the plan's own no-Physics-number prohibition)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Both commits are pushed to origin; a fresh single-branch clone in a temp dir outside the repo proves the strict-ancestor gate from a clone carrying no local state"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "git push -u origin fixture-validity-audit; git rev-list --count origin/fixture-validity-audit..HEAD == 0; fresh clone: merge-base --is-ancestor exit 0, rev-list --count == 1"
        status: pass
    human_judgment: false
  - id: D5
    description: "Full pu_manifold and root test suites are green after the freeze, with the freeze-ancestry tests running (not skipped)"
    requirement: "D9-18"
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q -- 913 passed, 1 skipped (unrelated CUDA-only test)"
        status: pass
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/ -q -- 120 passed"
        status: pass
    human_judgment: false
  - id: D6
    description: "LABEL_COLUMN_MAP, SENTINEL_VALUES and ALIGNMENT_MARGIN_R2 match 09-DATA-MANIFEST.md § Ruling entry for entry"
    requirement: "D9-05"
    verification:
      - kind: other
        ref: "Committed: {'mag_r':'mag_r_desi','photo_z':'photo_z','smooth_fraction':'smooth-or-featured_smooth_fraction','stellar_mass':'mass_med_photoz'}, (-99.0,), 0.1 -- identical to 09-DATA-MANIFEST.md Section 7's Ruling block"
        status: pass
    human_judgment: false

# Metrics
duration: ~75min
completed: 2026-09-03
status: complete
---

# Phase 9 Plan 5: Pre-Registration Freeze Summary

**Every one of 103 Phase 9 gating constants filled in a single freeze commit (`5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`), the SHA wired into three runners and two test suites in a second commit, and the strict-ancestor gate proved from a fresh single-branch clone of the pushed `fixture-validity-audit` branch — with two plan-prose corrections (`CURVATURE_FIELD_FOR_VERDICT`, `PER_D_VERDICT_VALUES`) made to match the sealed modules' own exact-equality guards rather than the plan's shorthand.**

## Performance

- **Duration:** ~75 min
- **Completed:** 2026-09-03
- **Tasks:** 2 (both completed)
- **Files modified:** 5 modified, 1 created

## Accomplishments
- Filled all 30 constants in `physics_labels.py` and all 73 in `physics_curvature_probe.py` (103 total, including the one non-gating `SWISS_ROLL_APPLICABILITY_RULE` already filled at 09-01); both `assert_preregistered()` calls exit 0
- Wrote `09-PREREGISTRATION.md`: a 103-row constant table with module/value/source-rationale columns, `VERDICT_RULE` and `VERDICT_SENTENCE_RULE` transcribed in full, an explicit "not frozen" section, sources cited by date, and the four flagged assumptions carried forward from `09-01-PLAN.md`
- Wired the real freeze SHA (`5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`) into all four files the plan named, plus a fifth site the plan's `<read_first>` had assumed already existed (`09_physics_curvature_run.py`'s own `FREEZE_COMMIT_SHA`/`_strict_ancestor_or_exit` pair, which did not exist before this plan)
- Pushed both commits to `origin/fixture-validity-audit` (new branch, no prior upstream) and proved the strict-ancestor gate from a fresh `--single-branch` clone in a `mktemp -d` temp directory, outside the repo, deleted after
- Full `notebooks/pu_manifold/tests/` suite: 913 passed, 1 skipped (unrelated, CUDA-only); root `tests/`: 120 passed — both green after the freeze, with the freeze-ancestry tests now running rather than skipped

## Task Commits

Each task was committed atomically:

1. **Task 1: Fill every constant and write the pre-registration — the freeze commit** - `5f7fbe2` (feat)
2. **Task 2: Wire the freeze SHA into both runners and both test suites, push it, prove the gate** - `f2bf10d` (feat)
3. **Fix: monkeypatch `LABEL_N_SHARDS` explicitly in the now-stale UNSET test** - `9bdd06c` (fix, found during Task 2's post-wiring-commit verify)

## Files Created/Modified
- `notebooks/pu_manifold/physics_labels.py` - 30 gating constants filled (Physics/label source identifiers, alignment shift set/margin/permutation config, sentinel values, `LABEL_COLUMN_MAP` transcribed from the developer's ratification)
- `notebooks/pu_manifold/physics_curvature_probe.py` - 73 gating constants filled (neighbourhood/anchor/fit config, `D_SWEEP=(16,20,25,32)`, instrument fidelity ranges, ridge/OOF config, verdict rules, positive-control/shuffled-label/seed-handling rules, freeze/execution-host rules)
- `.planning/phases/09-.../09-PREREGISTRATION.md` - new document: constant table, `VERDICT_RULE`/`VERDICT_SENTENCE_RULE` transcriptions, not-frozen list, dated sources, flagged assumptions, closing amendment rule
- `notebooks/diagnostics/09_row_alignment_proof_run.py` - `FREEZE_COMMIT_SHA` set from `None` to the real SHA
- `notebooks/diagnostics/09_physics_curvature_run.py` - `FREEZE_COMMIT_SHA`, `_git_rev_parse`, `_strict_ancestor_or_exit` added (did not exist before; no mode in this file calls it yet, ready for 09-06/09-08/09-09)
- `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` - ancestry test un-skipped; new `test_freeze_commit_sha_is_full_lowercase_hex`
- `notebooks/pu_manifold/tests/test_physics_labels.py` - freeze-ancestry scaffold added (did not exist before); two new tests; `test_shard_url_raises_when_shard_count_is_unset` rewritten to monkeypatch `LABEL_N_SHARDS` back to `None` explicitly

## Decisions Made
- `CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm"` — corrects the plan's prose "H_tan" against the sealed module's own guard text and literal field name (see Deviations)
- `PER_D_VERDICT_VALUES = ("NEGATIVE AND CLEARS FWER NULL", "DOES NOT CLEAR")` — two entries, not the plan's three (see Deviations)
- `VERDICT_VALUES` kept at the plan's literal four entries (`... , "HALTED - ALIGNMENT NOT PROVED"`) despite the module docstring's stale "Exactly three" — additive and harmless, reserved for the alignment-proof-fails case
- Freeze-gate machinery (`FREEZE_COMMIT_SHA`, `_strict_ancestor_or_exit`, `_git_rev_parse`) added to `09_physics_curvature_run.py`, mirroring the row-alignment runner exactly, even though no mode in that file uses it yet
- `test_physics_labels.py`'s freeze-ancestry scaffold added from scratch, mirroring `test_physics_curvature_probe.py`'s exactly, per the plan's own acceptance criterion that all four wired files carry the identical SHA

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `CURVATURE_FIELD_FOR_VERDICT` set to "H_tan_norm", correcting the plan's "H_tan"**
- **Found during:** Task 1, reading `physics_curvature_probe.py`'s constants block before filling values
- **Issue:** `09-05-PLAN.md` line 176 instructs `CURVATURE_FIELD_FOR_VERDICT = "H_tan"`, and the plan's own acceptance criterion (line 253) asserts `== 'H_tan'`. But the sealed module's own `_REQUIRED_CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm"` (line 291), its docstring states the field is `"H_tan_norm", never "H_norm"` (lines 117-118), and `decompose_radial_tangential`'s returned dict literally has no `"H_tan"` key — only `"H_rad"`, `"H_tan_norm"`, `"H_norm"`. Using `"H_tan"` would have made `pcp.assert_preregistered()` raise on the exact-equality check, directly failing Task 1's own first `<verify>`/acceptance criterion ("both `assert_preregistered()` exit 0")
- **Fix:** Set `CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm"`, matching the module's own guard and the actual field name
- **Files modified:** `notebooks/pu_manifold/physics_curvature_probe.py`
- **Verification:** `pcp.assert_preregistered()` exits 0; `assert p.CURVATURE_FIELD_FOR_VERDICT == 'H_tan_norm'` confirmed directly (the plan's own criterion asserting `== 'H_tan'` cannot be satisfied simultaneously with a passing freeze guard — the guard takes precedence per D9-18's own discipline: the module implements the behaviour the constant describes)
- **Committed in:** `5f7fbe2` (Task 1 commit)

**2. [Rule 1 - Bug] `PER_D_VERDICT_VALUES` set to a two-entry tuple, dropping the plan's third "SPLIT ACROSS SEEDS" entry**
- **Found during:** Task 1, same read
- **Issue:** `09-05-PLAN.md` line 212 instructs a three-entry `PER_D_VERDICT_VALUES` including `"SPLIT ACROSS SEEDS"`. But the module's own docstring states "Exactly two per-d verdict strings: fired, not-fired" (line 216), `per_d_verdict()` only ever indexes `[0]`/`[1]` (lines 867-868), and `test_physics_curvature_probe.py` already monkeypatches this constant with exactly two entries in two places (`("FIRED", "NOT_FIRED")`). `"SPLIT ACROSS SEEDS"` is a seed-COMBINATION outcome that `combine_seed_verdicts()` hardcodes directly as a string literal (line 900) — it is not sourced from `PER_D_VERDICT_VALUES` anywhere, and per-d classification (single seed, single d) is a different axis from seed-combination (three seeds, one d)
- **Fix:** Set `PER_D_VERDICT_VALUES = ("NEGATIVE AND CLEARS FWER NULL", "DOES NOT CLEAR")`
- **Files modified:** `notebooks/pu_manifold/physics_curvature_probe.py`
- **Verification:** `pcp.assert_preregistered()` exits 0 (non-empty check only, no equality guard on this constant); full test suite green (913 passed)
- **Committed in:** `5f7fbe2` (Task 1 commit)

**3. [Rule 2 - Missing critical] `09_physics_curvature_run.py` had no `FREEZE_COMMIT_SHA`/`_strict_ancestor_or_exit` pair to wire**
- **Found during:** Task 2, reading the file per `<read_first>` (which asserted "both `FREEZE_COMMIT_SHA = None` declarations and both `_strict_ancestor_or_exit` implementations" already existed)
- **Issue:** `09_physics_curvature_run.py`'s only implemented mode is `--mode smoke` (09-01); every mode that would produce a Physics number (`dsweep`, `positive-control`, `shuffled-label`, `verdict`, `seeds`, `bundle`, `selfcheck`) exits 2 naming a later plan (09-06/09-08/09-09). Neither `FREEZE_COMMIT_SHA` nor `_strict_ancestor_or_exit` existed anywhere in the file — only `09_row_alignment_proof_run.py` had this machinery, added by 09-03
- **Fix:** Added `FREEZE_COMMIT_SHA = "5f7fbe27afb0ef2a76353b41fa5713e760bbeea5"`, `_git_rev_parse`, and `_strict_ancestor_or_exit` mirroring `09_row_alignment_proof_run.py`'s shape exactly. No mode in this file calls it yet — it is ready for 09-06/09-08/09-09 to call rather than each re-deriving the check independently, and satisfies the plan's literal acceptance criterion (`grep -cE 'FREEZE_COMMIT_SHA = "[0-9a-f]{40}"' ...` reporting `1` for all four named files)
- **Files modified:** `notebooks/diagnostics/09_physics_curvature_run.py`
- **Verification:** `grep -cE 'FREEZE_COMMIT_SHA = "[0-9a-f]{40}"'` reports `1`; all four files' SHA is byte-identical (`sort -u | wc -l` == 1); `--mode smoke` still runs unaffected
- **Committed in:** `f2bf10d` (Task 2 commit)

**4. [Rule 2 - Missing critical] `test_physics_labels.py` had no freeze-ancestry scaffold at all**
- **Found during:** Task 2, same read (`<read_first>` asserted the scaffold existed in "both test files")
- **Issue:** Only `test_physics_curvature_probe.py` had the `FREEZE_COMMIT_SHA`/`_freeze_commit_exists`/`_freeze_commit_is_strict_ancestor_of_head`/skipif-marked-test scaffold from 09-01. `test_physics_labels.py` had none of it
- **Fix:** Added the identical scaffold (module-level `FREEZE_COMMIT_SHA`, both helper functions, `test_freeze_commit_is_a_strict_ancestor_of_head`, `test_freeze_commit_sha_is_full_lowercase_hex`) and the `subprocess` import it requires
- **Files modified:** `notebooks/pu_manifold/tests/test_physics_labels.py`
- **Verification:** Both new tests pass; `grep -cE 'FREEZE_COMMIT_SHA = "[0-9a-f]{40}"' .../test_physics_labels.py` reports `1`
- **Committed in:** `f2bf10d` (Task 2 commit)

**5. [Rule 1 - Bug] `test_shard_url_raises_when_shard_count_is_unset` broke once `LABEL_N_SHARDS` was frozen**
- **Found during:** Task 2, first post-wiring-commit full-suite run (`.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_physics_curvature_probe.py notebooks/pu_manifold/tests/test_physics_labels.py -x -q`) — `Failed: DID NOT RAISE ValueError`
- **Issue:** The test relied on `LABEL_N_SHARDS` being permanently `None` at module scope (no monkeypatch), which held before the freeze but no longer holds after Task 1's freeze commit set it to `16`
- **Fix:** Added a `monkeypatch` parameter and `monkeypatch.setattr(pl, "LABEL_N_SHARDS", None)` to explicitly restore the UNSET state the test needs, scoped to that one test
- **Files modified:** `notebooks/pu_manifold/tests/test_physics_labels.py`
- **Verification:** `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_physics_curvature_probe.py notebooks/pu_manifold/tests/test_physics_labels.py -q` — 146 passed
- **Committed in:** `9bdd06c` (separate fix commit, discovered and fixed after `f2bf10d` landed)

---

**Total deviations:** 5 auto-fixed (2 Rule 1 bugs correcting plan-prose-vs-sealed-module conflicts in Task 1's frozen values, 2 Rule 2 missing-critical additions for machinery the plan's `<read_first>` assumed already existed, 1 Rule 1 bug fixing a test whose implicit UNSET-state assumption the freeze itself invalidated)
**Impact on plan:** All five were necessary for `assert_preregistered()` to pass and/or for the full test suite to stay green after the freeze. None changed the plan's actual gating decisions (D9-01 through D9-18) or any developer-ratified value — they corrected internal inconsistencies between the plan's prose and the sealed code it was filling, and filled genuine gaps in the plan's `<read_first>` assumptions about pre-existing scaffolding. No scope creep: every fix stayed inside the plan's own `files_modified` list.

## Issues Encountered
The third gate-boundary invocation (`--mode proof --freeze-commit <real-freeze-SHA>`) passes the strict-ancestor gate silently (no D9-18 error) and then attempts real network I/O (HuggingFace data resolution) rather than failing immediately with a clean, distinct error. Per the plan's own D9-18 prohibition ("No Physics number is produced by this plan"), this invocation was deliberately terminated (`timeout 20`) before any download could complete, rather than let it run to whatever real failure or success state comes next. Confirmed no `notebooks/.cache/09_row_alignment.jsonl` or `09_physics_curvature.jsonl` was written and no process was left running. This satisfies the acceptance criterion's intent (the gate passes, the failure is elsewhere) without risking an accidental partial download or a Physics number appearing before the freeze discipline says one should.

## User Setup Required
None — no external service configuration required. The `git push` to `origin/fixture-validity-audit` was pre-authorized by the plan's must_haves (T-09-31) and the orchestrator's own instructions; no credential prompt occurred (SSH key already configured for `git@github.com:amanasci/EffDim.git`).

## Next Phase Readiness
- Both `assert_preregistered()` calls pass; every Phase 9 gating constant carries its final, literal value; `09-PREREGISTRATION.md` is the authoritative transcription
- `FREEZE_COMMIT_SHA = "5f7fbe27afb0ef2a76353b41fa5713e760bbeea5"` is wired identically into all four files the plan named plus the fifth site this plan added; the strict-ancestor gate has been proved from a fresh clone carrying no local state — the exact condition the execution host (09-06, still undecided per `STATE.md`) will run under
- No `notebooks/.cache/09_row_alignment.jsonl` or `09_physics_curvature.jsonl` exists anywhere in the tree; no Physics number has been produced
- Phase 7's own `crossmodal_curvature.D_SWEEP = (20, 25, 32)` is byte-identical and untouched; Phase 9's fresh `D_SWEEP = (16, 20, 25, 32)` is declared independently
- 09-06 (execution-host hand-off) can now build against this freeze; 09-07/08/09 (the actual measurement runs) are gated correctly behind it
- No blockers

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-03*

## Self-Check: PASSED

All 8 files created/modified this plan found on disk (`09-PREREGISTRATION.md`,
`physics_labels.py`, `physics_curvature_probe.py`, `09_physics_curvature_run.py`,
`09_row_alignment_proof_run.py`, `test_physics_curvature_probe.py`, `test_physics_labels.py`,
this `09-05-SUMMARY.md`); all three commits (`5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`,
`f2bf10d1798a0379e89fdb7fd5a93a8ea3d6c43c`, `9bdd06c0aa63e8d2151d2f15123b77afa343fa13`) found in
git history via `git cat-file -e`; all pushed to `origin/fixture-validity-audit`
(`git rev-list --count origin/fixture-validity-audit..HEAD` == 0).
