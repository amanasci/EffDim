---
phase: 05-curvature-conditioned-linear-decodability
plan: 06
subsystem: geometry
tags: [phase-close, findings, verification, notebook, split-across-seeds]

# Dependency graph
requires:
  - phase: 05-curvature-conditioned-linear-decodability (05-05)
    provides: notebooks/.cache/05_curvature_probe_decodability.jsonl -- the headline probe
      record (1 probe_overall, 3 probe_seed, 9 probe_bucket, 1 probe_conditioning rows),
      SPLIT ACROSS SEEDS phase verdict
provides:
  - notebooks/05_curvature_conditioned_linear_decodability.ipynb -- the executed, committed
    reader notebook: both frozen rules printed before any probe number, three seeds plotted
    side by side with no pooled panel, per-bucket table with realized-vs-full-field counts,
    closing markdown quoting all three per-seed verdicts and the phase verdict verbatim
  - .planning/phases/05-curvature-conditioned-linear-decodability/05-FINDINGS.md -- the phase
    record: claims and non-claims, frozen configuration, the seed decision and pooled-artifact
    answer from 05-03-DECISION.md, density disclosure, the result (SPLIT ACROSS SEEDS), D5-11
    and D5-12 accepted gaps at full strength, requirement outcomes for D5-01..D5-13, follow-on
    needs
  - .planning/phases/05-curvature-conditioned-linear-decodability/05-VERIFICATION.md -- the
    mechanical ordering-guarantee proof from git ancestry (merge-base, restricted diff, the
    frozen file's exactly-three-commit history explained commit by commit)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Phase-closing notebook reads cached artifacts only -- no fit_probe, no
      chart_curvature_field, no pool_seed_fields, no bucket_by_field call anywhere in its
      source, enforced by a negative grep in the task's own automated verify"
    - "Ordering guarantee proved by re-running the git commands themselves inside
      05-VERIFICATION.md rather than narrating a description of them, and independently
      re-run again by the orchestrator at the Task 3 checkpoint before approval"

key-files:
  created:
    - notebooks/05_curvature_conditioned_linear_decodability.ipynb
    - .planning/phases/05-curvature-conditioned-linear-decodability/05-FINDINGS.md
    - .planning/phases/05-curvature-conditioned-linear-decodability/05-VERIFICATION.md
  modified: []

key-decisions:
  - "Notebook step 10's continuous per-point scatter was substituted with the already-recorded
    per-seed Spearman rho/p/n, because 05-05 never persisted raw per-point residual/||H||
    arrays on the test split (only bucket- and seed-level aggregates), and refitting inside
    the notebook to reconstruct that data is forbidden by the plan's own action text and its
    automated verify. Documented in-notebook and disclosed to the human at the Task 3
    checkpoint."
  - "A second stale pre-freeze assertion, this time in the runner's own selfcheck(), was found
    and inverted before Task 1 (df392bd) -- the same class of fix already applied to
    test_linear_probe.py at 05-04. linear_probe.py itself was never touched by either fix."
  - "The ancestry proof in 05-VERIFICATION.md runs against 05-05 Task 1's commit (2c8b062),
    not a dedicated 05-05 Task 2 commit, because Task 2's real run produced no commit of its
    own (its JSONL output is gitignored). This gap is disclosed at full strength in
    05-VERIFICATION.md rather than glossed over, and was explicitly shown to and accepted by
    the human before Task 3's approval."

requirements-completed: [D5-04, D5-05, D5-07, D5-08, D5-09, D5-11, D5-12, D5-13]

coverage:
  - id: T1
    description: "Executed, committed notebook shows both frozen rules before any probe
      number, plots three seeds with no pooled panel, and closes on the phase verdict and
      three per-seed verdicts quoted verbatim from their record rows"
    requirement: D5-09
    verification:
      - kind: integration
        ref: "python -c assert script over notebook JSON (15 cells, all executed, all
          required literal tokens present, no recompute call sites, all verdict strings
          present) -- see plan 05-06 Task 1 <verify>"
        status: pass
    human_judgment: true
  - id: T2
    description: "05-FINDINGS.md and 05-VERIFICATION.md written to Phase 4's honesty
      standard; D5-04 recorded SUPERSEDED, D5-05 split into met/dispositioned, D5-11/D5-12
      accepted gaps stated at full strength, ordering guarantee proved mechanically with the
      frozen file's three-commit history explained"
    requirement: D5-09
    verification:
      - kind: integration
        ref: "python -c assert script over both documents' required literals, the
          three-commit git log, the empty restricted diff, and byte-identical verdict
          strings against the JSONL rows -- see plan 05-06 Task 2 <verify>"
        status: pass
    human_judgment: true
  - id: T3
    description: "Human verification of the four judgements no automated check can make:
      whether the frozen rules really admit their stated terminal outcomes, whether the
      accepted-gaps section states each gap at full strength without deferring to a
      cross-reference, whether the ordering-guarantee evidence says what the verification
      document claims, and whether the SPLIT ACROSS SEEDS write-up honours the frozen
      combination rule's own non-support framing"
    requirement: D5-09
    verification:
      - kind: manual
        ref: "Orchestrator independently re-ran the mechanical half (git merge-base
          --is-ancestor, restricted diff, three-commit log, --mode pool refusal,
          --selfcheck, full test suite, notebook cell/output audit, verdict-string
          cross-check) before presenting the checkpoint; human responded 'approved' after
          being shown the ancestry-proof gap, the substituted notebook artifact, and the
          second stale-assertion inversion"
        status: pass
    human_judgment: true

duration: ~35 min (Task 1 commit 7e6bba8 through this plan's metadata commit; excludes the
  pause at the Task 3 checkpoint awaiting human response)
completed: 2026-08-24
status: complete
---

# Phase 5 Plan 6: Close Phase 5 -- Notebook, Findings, Verification, Human Sign-Off Summary

**Phase 5 closes with a `SPLIT ACROSS SEEDS` phase verdict (2 of 3 seeds HOLDS), reported exactly
as its frozen combination rule defines it -- a complete, non-supportive terminal outcome, not
partial support -- in an executed notebook and a findings/verification pair the human
independently verified and approved.**

## Performance

- **Duration:** ~35 min task-commit span (excludes the pause awaiting the human's checkpoint
  response)
- **Tasks:** 3 (2 auto, 1 blocking human-verify checkpoint)
- **Files created:** 3 (notebook, FINDINGS, VERIFICATION)

## Accomplishments

- **Pre-task fix (`df392bd`):** Found and inverted `selfcheck()`'s stale pre-freeze
  `assert_preregistered` check -- it asserted the guard raises while constants are unset, true
  only before the `05-04` freeze. Since the module now ships frozen, `--selfcheck` exited 1 for
  real. Mirrors the identical fix already applied to `test_linear_probe.py` at `05-04`.
  `linear_probe.py` itself was never touched.
- **Task 1 (`7e6bba8`):** `notebooks/05_curvature_conditioned_linear_decodability.ipynb` created,
  executed end to end (15 cells, all code cells with sequential execution counts 1-10 and
  outputs), and committed. Imports `linear_probe` unchanged, reads every number from cached
  artifacts, prints both frozen rules and `PHASE_VERDICT_VALUES` before any probe number, plots
  all three seeds' curvature-magnitude distributions with no pooled panel, tables realized
  test-split counts beside full-field counts for every seed, and closes on all three per-seed
  verdicts and the phase verdict quoted verbatim.
- **Task 2 (`c46220d`):** `05-FINDINGS.md` and `05-VERIFICATION.md` written to `04-FINDINGS.md`'s
  honesty standard. The ordering guarantee is proved mechanically: `git merge-base
  --is-ancestor 32dabe3 2c8b062` exits 0, `git diff 32dabe3 HEAD -- notebooks/pu_manifold/
  linear_probe.py` is empty, and `git log --oneline -- notebooks/pu_manifold/linear_probe.py`
  lists exactly three commits (`5888d0d` creation at `05-01`, `94735b7` the per-seed structural
  repair at `05-03` -- explained as preceding the freeze and assigning no constant, `32dabe3`
  the freeze at `05-04`, confirmed most recent). D5-04 recorded `SUPERSEDED` by
  `05-03-DECISION.md`; D5-05 split into a met inter-seed-agreement half and a dispositioned
  pooled-vs-seed half with no referent. D5-11's sealed `rank_spearman_rho =
  -0.015106571347065712` and D5-12's `CAE_VERDICT = FAIL` -> Phase 3 override -> Phase 03.1
  partial repair chain are both stated in the phase's own words, in phase order, each gap at
  full strength.
- **Task 3 (checkpoint, this continuation):** Human verification requested and **approved**. The
  orchestrator independently re-ran the mechanical half of the verification procedure before
  presenting the checkpoint (see Checkpoint Resolution below) and the human was explicitly shown
  and accepted three disclosed gaps before approving. No file changes accompany Task 3 itself --
  it is a read-and-judge gate, not a code task.

## Checkpoint Resolution

**Type:** human-verify (blocking)
**Resume signal received:** `approved`

Before presenting the checkpoint, the orchestrator independently confirmed, not merely accepted
on narration:

- `git merge-base --is-ancestor 32dabe3 2c8b062` -> exit 0
- `git diff 32dabe3 HEAD -- notebooks/pu_manifold/linear_probe.py` -> empty
- `git log --oneline -- notebooks/pu_manifold/linear_probe.py` -> exactly 3 commits, `32dabe3` newest
- `--mode pool` -> exit 1, refusal message names `05-03-DECISION.md`
- `--selfcheck` -> 8/8 PASS, exit 0
- Full suite (`tests` + `notebooks/pu_manifold/tests`) -> 510 passed, 1 skipped
- Notebook -> 15 cells, 10 code cells, all with outputs, execution counts 1-10 sequential
- Verdicts read directly from the JSONL: 20260813 HOLDS, 20260814 NO DETECTABLE RELATIONSHIP,
  20260815 HOLDS, `phase_verdict = SPLIT ACROSS SEEDS`, `n_holds = 2`,
  `r2_overall = 0.6439307736500615`, `selected_alpha = 0.1`, `n_train = 7000`, `n_test = 3000`

Three things were explicitly disclosed to and accepted by the human before approval:

1. The ancestry-proof gap: `05-05` Task 2 left no commit of its own (its JSONL output is
   gitignored), so the mechanical proof runs against `2c8b062` (Task 1's commit) rather than a
   dedicated Task-2 commit. Disclosed in `05-VERIFICATION.md` §1, lines 32-49.
2. The substituted notebook artifact: step 10's per-point continuous scatter was replaced with
   the recorded per-seed Spearman rho/p/n, because no cached per-point data exists and refitting
   inside the notebook is forbidden by the plan's own verify block.
3. The second stale-assertion inversion in the runner's `selfcheck()` (`df392bd`).

No re-verification from scratch was performed for this continuation; the checkpoint was treated
as satisfied per the resume instructions.

## Phase Verdict

**`SPLIT ACROSS SEEDS`** (`n_holds = 2` of 3): seed 20260813 HOLDS, seed 20260814 NO DETECTABLE
RELATIONSHIP, seed 20260815 HOLDS. Reported exactly as the frozen `SEED_VERDICT_COMBINATION_RULE`
defines it -- a complete, terminal, non-supportive outcome. The rule's own reason: the three
seeds' curvature fields were measured (`05-02`, ratified at the `05-03` blocking checkpoint) to be
mutually anti-correlated and directionally orthogonal, so an effect present in one or two of
three decoder fits is a property of those individual fits, not of the manifold, and does not
license a claim that decodability degrades with curvature. This is not partial support, not a
majority result, and the agreeing seeds are not presented as the headline with the third set
aside -- all three verdicts stand together, exactly as the plan's prohibitions require.

## Task Commits

1. **Pre-task fix: invert `selfcheck()`'s stale pre-freeze `assert_preregistered` check** - `df392bd` (fix)
2. **Task 1: The executed notebook -- three seeds side by side** - `7e6bba8` (feat)
3. **Task 2: The phase record and the mechanical ordering proof** - `c46220d` (docs)
4. **Task 3: Close Phase 5** - checkpoint, human-verify, no source commit (`approved`)

**Plan metadata:** this commit (SUMMARY + STATE + ROADMAP + REQUIREMENTS)

## Files Created/Modified

- `notebooks/05_curvature_conditioned_linear_decodability.ipynb` - new, executed, committed with
  outputs (`7e6bba8`)
- `.planning/phases/05-curvature-conditioned-linear-decodability/05-FINDINGS.md` - new (`c46220d`)
- `.planning/phases/05-curvature-conditioned-linear-decodability/05-VERIFICATION.md` - new (`c46220d`)
- `notebooks/diagnostics/curvature_probe_decodability_run.py` - `selfcheck()`'s stale assertion
  inverted (`df392bd`); `linear_probe.py` untouched by this fix

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `selfcheck()`'s stale pre-freeze `assert_preregistered` check**
- **Found during:** Before Task 1.
- **Issue:** `selfcheck()` asserted `assert_preregistered()` raises while constants are unset --
  true only before the `05-04` freeze. Since the module now ships frozen, `--selfcheck` exited 1
  for real.
- **Fix:** Inverted to assert the guard now passes, mirroring the identical fix already applied
  to `test_linear_probe.py::test_assert_preregistered_raises_when_absent` at `05-04`.
  `linear_probe.py` itself was never touched.
- **Files modified:** `notebooks/diagnostics/curvature_probe_decodability_run.py`
- **Commit:** `df392bd`

**2. [Rule 3 - Blocking] Notebook step 10's continuous per-point scatter has no cached raw
per-point data to read**
- **Found during:** Task 1.
- **Issue:** `05-05` never persisted raw per-point residual/`||H||` arrays on the test split --
  only bucket- and seed-level aggregates were written to the JSONL. Reconstructing a per-point
  scatter would require refitting the ridge map inside the notebook, which the plan's own action
  text and its automated verify both forbid.
- **Fix:** Substituted the already-recorded per-seed Spearman rho/p/n, documented in-notebook and
  disclosed at the Task 3 checkpoint.
- **Files modified:** `notebooks/05_curvature_conditioned_linear_decodability.ipynb`
- **Commit:** `7e6bba8`

Neither deviation touches `notebooks/pu_manifold/linear_probe.py`, changes any verdict, or
recomputes any number the frozen pipeline produced.

**Total deviations:** 2 auto-fixed (1 Rule 1 bug fix, 1 Rule 3 blocking-issue substitution), both
disclosed in full to the human before the Task 3 checkpoint's approval.

## Issues Encountered

None beyond the two auto-fixed deviations above and the pre-existing, honestly-recorded ancestry
gap (no dedicated `05-05` Task 2 commit, since its JSONL output is gitignored) -- all three were
surfaced to the human, not discovered independently at review.

## User Setup Required

None -- no external service configuration required.

## Known Stubs

None. Every number in the notebook, `05-FINDINGS.md`, and `05-VERIFICATION.md` is quoted from a
cached record row or a git command's literal output; nothing is a hardcoded placeholder.

## Threat Flags

None. No new network surface, auth path, file access pattern, or schema change at a trust
boundary was introduced by this plan -- it is a read-only reporting layer over artifacts already
produced and validated in prior plans.

## Next Phase Readiness

- **Phase 5 is CLOSED.** `SPLIT ACROSS SEEDS` is the terminal phase verdict, reported at full
  strength with every acknowledged weakness named: no known-answer anchor (D5-11, sealed
  `rank_spearman_rho = -0.015106571347065712`), an overridden CAE validity gate (D5-12,
  `CAE_VERDICT = FAIL` at Phase 02.2), three curvature fields measured to share no signal
  (`05-03-DECISION.md`), and a re-measured, non-transferring density confound (D5-13).
- All 13 requirement IDs (D5-01 through D5-13) are accounted for in `05-FINDINGS.md` §7. D5-04
  reads `SUPERSEDED`; D5-05 is split into a met half and a dispositioned half; every other
  requirement reads Met, with its accepted gaps named at full strength rather than hidden inside
  the word "Met."
- **What a follow-on phase would need**, per `05-FINDINGS.md` §8: the declined low-`d`
  probe-methodology anchor, the saddle-fixture resolution (open, not for autonomous action), an
  external astrophysical label as the probe target, per-region independent fits at matched `n` as
  sensitivity-only, and -- new from this phase -- whatever would be required to obtain three CAE
  fits whose curvature fields agree with each other, since without that a per-seed spread is the
  ceiling on what any curvature-conditioned claim about this manifold can be.
- `notebooks/pu_manifold/linear_probe.py` remains frozen at exactly three commits, the freeze
  (`32dabe3`) most recent, unchanged by this plan.
- No blockers.

---
*Phase: 05-curvature-conditioned-linear-decodability*
*Completed: 2026-08-24*

## Self-Check: PASSED

`notebooks/05_curvature_conditioned_linear_decodability.ipynb`,
`.planning/phases/05-curvature-conditioned-linear-decodability/05-FINDINGS.md`, and
`.planning/phases/05-curvature-conditioned-linear-decodability/05-VERIFICATION.md` all found on
disk. Commit hashes `df392bd`, `7e6bba8`, `c46220d` all found in `git log --oneline -8`.
`notebooks/pu_manifold/linear_probe.py` confirmed at exactly three commits via
`git log --oneline -- notebooks/pu_manifold/linear_probe.py`, freeze commit `32dabe3` most
recent, matching `05-04-SUMMARY.md`'s recorded `freeze_commit`.
