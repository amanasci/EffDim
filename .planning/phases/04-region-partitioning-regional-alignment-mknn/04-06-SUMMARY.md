---
phase: 04-region-partitioning-regional-alignment-mknn
plan: 06
subsystem: manifold-curvature
tags: [phase-closure, findings, pre-registration, mknn, density-confound, codimension, region-size-artifact]

# Dependency graph
requires:
  - phase: 04-region-partitioning-regional-alignment-mknn
    provides: "plans 04-01..04-05's complete pipeline: the global and regional MKNN
      machinery, the density-corrected k-freeze, the frozen diametrical sign-split
      partition, the density confound measurement, and the sealed HOLDS verdict
      (commits 647d01d, bec5cc0, 6aa43ad, 7feb1f4)"
provides:
  - "notebooks/04_region_partition_mknn.ipynb Section 9: the inherited-field gap,
    the codimension gap, the density confound, the no-known-answer-anchor statement,
    the Swiss-roll-rule non-applicability reasoning, and citations with confidence
    markers -- all printed beside the numbers they qualify"
  - "04-FINDINGS.md: the phase closure record -- the three accepted gaps at full
    length, the frozen configuration, the partition, density, and MKNN results, a
    mandatory region-size-artifact qualification independently verified from the
    JSONL, requirement outcomes for all 14 Phase 4 requirements, and the four
    declined follow-on mitigations"
  - "REQUIREMENTS.md Phase 4 coverage rows (REGN-02, REGN-03, MKNN-02, MKNN-07)
    annotated with honest, non-blanket outcomes"
affects: []

tech-stack:
  added: []
  patterns:
    - "Phase-closure record structure: gaps stated at full length before results
      (03-FINDINGS.md Section 1 precedent), every quoted number traceable to a
      cache artifact or a printed notebook cell, no sealed verdict from prior
      phases reopened"
    - "A mechanical qualification (region-size artifact) documented beside a
      pre-registered verdict without amending the verdict -- the pre-registration
      discipline forbids re-specifying a rule after seeing the numbers it produced,
      even when the new framing is more informative"

key-files:
  created:
    - .planning/phases/04-region-partitioning-regional-alignment-mknn/04-FINDINGS.md
  modified:
    - notebooks/04_region_partition_mknn.ipynb
    - .planning/REQUIREMENTS.md

key-decisions:
  - "The region-size artifact (MKNN's chance floor scales as k/n_region, so region
    1 at n=3244 versus region 0 at n=6256 mechanically inflates the raw-score gap)
    is reported as a qualification alongside the HOLDS verdict, not as grounds to
    amend VERDICT_RULE. The rule was pre-registered on raw scores before any
    regional number existed and applied mechanically; re-specifying it in
    ratio-over-chance terms now, after seeing that framing tells a different
    story, is exactly the post-hoc rule selection the pre-registration freeze
    exists to forbid. Verified independently from the JSONL
    (chance_floor == k_mknn/n_region, ratio_over_chance == score/chance_floor,
    to 1e-9) before being written into the record, per the orchestrator's explicit
    instruction not to take the finding on its word."
  - "REQUIREMENTS.md's Phase 4 rows were already 'Complete' (not 'Pending') going
    into this plan, set by prior plans' state_updates step. This plan overrides
    the blanket Complete text for REGN-02, REGN-03, MKNN-02 and MKNN-07 -- the
    four rows whose evidence carries an accepted gap or a stated qualification --
    with the CURV-04/06/07 precedent's honest-outcome format, rather than leaving
    a plain Complete that would hide the qualification from a reader scanning the
    coverage table alone."
  - "COVERAGE.md required no edit: plan 04-01 already wrote the exact no-external-
    API declaration this plan's task list calls for, and it was verified to
    already satisfy every acceptance criterion in this plan's <verify> block."

requirements-completed: [REGN-02, MKNN-07, MKNN-08]

coverage:
  - id: D1
    description: "Notebook Section 9 states the three accepted gaps (inherited
      field, codimension, density confound), the no-known-answer-anchor sentence,
      the Swiss-roll-rule non-applicability reasoning, and citations with
      confidence markers, each beside the numbers it qualifies; sections 0-8 left
      byte-identical"
    requirement: MKNN-08
    verification:
      - kind: e2e
        ref: "jupyter nbconvert --to notebook --execute --inplace, plus the
          plan's own post-execution assertion script (literal-presence checks,
          pre-registration-cell-precedes-regional-cell ordering check, execution
          counts strictly increasing) run directly"
        status: pass
      - kind: other
        ref: "git diff confirms cells 0-38 (sections 0-8) source-byte-identical
          to the pre-plan notebook"
        status: pass
    human_judgment: false
  - id: D2
    description: "04-FINDINGS.md written: section 1 states all three gaps at full
      length ending with the no-known-answer-anchor sentence; sections 2-5 report
      the frozen configuration, partition, density and MKNN results with every
      number traceable to a cache artifact; section 5 includes the mandatory
      region-size-artifact qualification, independently verified from the JSONL;
      section 6 gives one outcome row per Phase-4 requirement; section 7 lists
      the four declined mitigations"
    requirement: REGN-02
    verification:
      - kind: other
        ref: "the plan's own <verify> assertion script (literal-presence checks,
          all 14 requirement IDs present) run directly against 04-FINDINGS.md"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -q (376 passed, 1 skipped)"
        status: pass
    human_judgment: true
    rationale: "Whether the caveats are framed honestly rather than merely
      present as strings is a human judgment the plan's own flagged_assumptions
      reserve explicitly for 04-VALIDATION.md's Manual-Only table, not something
      a mechanical grep can settle."
  - id: D3
    description: "REQUIREMENTS.md's Phase 4 coverage rows updated from a blanket
      Complete to honest outcomes for the four rows carrying a stated
      qualification (REGN-02, REGN-03, MKNN-02, MKNN-07); edit scoped to those
      rows, verified by git diff to touch nothing else in the file"
    requirement: MKNN-07
    verification:
      - kind: other
        ref: "git diff .planning/REQUIREMENTS.md, inspected directly -- confined
          to the four annotated rows"
        status: pass
    human_judgment: false

duration: ~45min
completed: 2026-08-24
status: complete
---

# Phase 4 Plan 6: Phase closure -- three accepted gaps, the region-size artifact, and honest requirement outcomes Summary

**The phase record states, at the same standard `03-FINDINGS.md` Section 1 was held to, that
Phase 4's HOLDS verdict (region 1 beats region 0 at every k including the headline k=20, applied
mechanically from a pre-registered rule) rests on a field accepted on split-half reliability
alone, a direction partition validated only at codimension 1 against PU's codimension-~748, and a
density confound reported but never controlled -- and adds a fourth, independently-verified
qualification: the raw-score gap the verdict rule fires on is mostly a region-size artifact
(MKNN's chance floor scales as k/n_region), shrinking from 94-138% raw to 0.6-23.2% once read as
ratio-over-chance, without that artifact licensing any amendment to the pre-registered rule.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-08-24T13:35Z (approx.)
- **Completed:** 2026-08-24T14:20Z (approx.)
- **Tasks:** 2/2 complete
- **Files modified:** 3 (1 notebook extended, 1 new findings document, 1 requirements table
  edited)

## Accomplishments

- **Notebook Section 9 appended and executed end to end** (11 new cells, sections 0-8 left
  byte-identical): 9.1 the inherited field (Swiss roll `R_H=0.990` / `rho=0.469` beside this
  phase's own corrected `median_R_H=0.3436` at `K_FROZEN=500`), 9.2 the codimension gap (PU's
  codimension `748` printed beside the eigenvalue spectrum from Section 5), 9.3 the density
  confound (both Spearman correlations and the Mann-Whitney statistic printed as numbers, not
  references), 9.4 the taken-together no-known-answer-anchor statement, 9.5 why no Swiss roll
  notebook is delivered for this phase's models (naming
  `02.5_swiss_roll_curvature_probe_check.ipynb` and `test_region_partition.py`'s antipodal-cone
  fixture as existing coverage), 9.6 citations with explicit confidence markers
  (arXiv:2509.19453 read directly; Radovanovic and Dhillon read in secondary summary only).
- **`04-FINDINGS.md` written**: Section 1 leads with all three accepted gaps at full length in
  the phase's own words (not by reference), closing with the no-known-answer-anchor sentence.
  Sections 2-4 report the frozen configuration (the not-fired D4-07 rule, the six-point sweep
  table), the partition (region counts, `mean_unit_norm=0.294748`, the weak eigenvalue
  separation, PU's `||H||` spread against the bowl/cubic/ridge calibration), and the density
  confound (both correlations, the region-level Mann-Whitney comparison). Section 5 reports the
  global and regional MKNN results, the verbatim `VERDICT_RULE`, the headline HOLDS verdict, and
  the mandatory region-size-artifact qualification (independently re-derived from the JSONL
  before being written, per the orchestrator's explicit instruction). Section 6 gives one
  outcome row per each of the 14 Phase-4 requirements. Section 7 lists the four declined
  follow-on mitigations quoted from `04-CONTEXT.md`'s Deferred Ideas, plus a fifth item this
  closure plan itself surfaced (a size-matched or chance-floor-normalized regional statistic for
  any future re-run).
- **The region-size artifact, independently verified.** Recomputed `chance_floor` (`==
  k_mknn/n_region`) and `ratio_over_chance` (`== score/chance_floor`) directly from
  `04_region_partition_mknn.jsonl`'s eight regional rows before writing anything about them: at
  `k=20` raw scores differ by ~114% (`0.17408` vs `0.08148`) while ratios over chance differ by
  only ~11% (`28.2` vs `25.5`); at `k=50` the raw gap is ~94% while the ratio-over-chance gap is
  effectively zero (`15.7` vs `15.6`). Recorded as a qualification beside the HOLDS verdict, not
  as grounds to amend `VERDICT_RULE` — the rule was pre-registered on raw scores and applied
  mechanically, and re-specifying it now in ratio-over-chance terms would be exactly the
  post-hoc rule change the pre-registration freeze exists to forbid.
- **`COVERAGE.md` verified, not edited.** Plan `04-01` already wrote the exact no-external-API
  declaration this plan's deliverable list calls for; it was checked against every acceptance
  criterion in this plan's `<verify>` block and needed no change.
- **`REQUIREMENTS.md`'s Phase 4 coverage rows annotated with honest outcomes** for the four rows
  carrying a stated qualification — REGN-02 (density confound reported, not controlled), REGN-03
  (partition validated only at codimension 1, not at PU's ~748), MKNN-02 (raw scores fall outside
  the origin paper's range at an order-of-magnitude smaller `n`), and MKNN-07 (HOLDS, qualified
  by the region-size artifact and the unresolved density confound) — following the
  CURV-04/06/07 precedent already established in this same table. `git diff` confirms the edit
  touched nothing else in the file.

## Task Commits

Each task was committed atomically:

1. **Task 1: Notebook section 9 — the accepted gaps, the Swiss roll reasoning, and the
   citations** - `18cb8be` (feat)
2. **Task 2: 04-FINDINGS.md, COVERAGE.md, and honest requirement outcomes** - `3d50575` (docs)

**Plan metadata:** committed as part of this SUMMARY's own docs commit.

## Files Created/Modified

- `notebooks/04_region_partition_mknn.ipynb` — 11 new cells appended as Section 9; sections 0-8
  untouched
- `.planning/phases/04-region-partitioning-regional-alignment-mknn/04-FINDINGS.md` — new, the
  phase closure record
- `.planning/REQUIREMENTS.md` — Phase 4 coverage table rows REGN-02, REGN-03, MKNN-02, MKNN-07
  annotated with honest outcomes; no other section changed

## Decisions Made

- **The region-size artifact does not amend `VERDICT_RULE`.** Stated explicitly in
  `04-FINDINGS.md` §5: the rule was pre-registered on raw scores before any regional number
  existed and applied mechanically to produce HOLDS; re-specifying the comparison in
  ratio-over-chance terms after seeing that the two framings diverge would be the exact
  post-hoc rule selection the phase's own pre-registration discipline exists to prevent. The
  artifact is reported beside the verdict, not folded into a restated verdict.
- **Four rows, not all fourteen, received honest-outcome annotations.** REGN-01, REGN-04,
  REGN-05, REGN-06, MKNN-01, MKNN-03, MKNN-04, MKNN-05, MKNN-06 and MKNN-08 have no accepted
  gap or qualification specific to them beyond the phase-wide no-known-answer-anchor statement
  already carried in Section 1 of the findings — annotating every row identically would dilute
  the four rows that actually need a reader's attention, mirroring how CURV-01/02/03/05/08
  stayed plain `Complete` in the Phase 3 precedent while only CURV-04/06/07 carried elaboration.
- **`COVERAGE.md` left unmodified.** Re-writing an already-correct, already-committed declaration
  to satisfy a task checklist item would be edit-for-its-own-sake; the plan's actual requirement
  (the declaration exists and is accepted by the seal-time gate) was already met.

## Deviations from Plan

### Auto-fixed Issues

None — no bugs, missing critical functionality, or blocking issues were found during execution
that required Rule 1-3 auto-fixes.

**One documented plan-interpretation note, not a rule-triggered deviation:** the plan's task
list names `COVERAGE.md` as a file this task modifies, but it was found already correct and
already committed (from plan `04-01`). No edit was made; this is noted here rather than silently
skipped, following the same disclosure precedent set by 04-02's and 04-03's own documented
plan-ambiguity resolutions.

---

**Total deviations:** 0 rule-triggered auto-fixes; 1 documented plan-interpretation note (above).
**Impact on plan:** None on scope or correctness — `COVERAGE.md`'s acceptance criteria were
verified directly against the pre-existing file.

## Issues Encountered

None. Both automated `<verify>` scripts passed on the first run; `notebooks/pu_manifold/tests/
-q` ran to completion (376 passed, 1 skipped) with no code outside this plan's own files
touched.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Phase 4 is closed.** `04-FINDINGS.md` is the phase's canonical record: the HOLDS verdict,
  the three accepted gaps (unvalidated field, unclosed codimension gap, uncontrolled density
  confound), and the region-size artifact all travel together, with every quoted number
  traceable to a cache artifact or a printed notebook cell.
- **The five follow-on items in `04-FINDINGS.md` §7** (cross-estimator agreement, codimension
  narrowing, density-matched controls, `tests/test_mknn.py`, and a size-matched or
  chance-floor-normalized regional statistic) are the concrete starting points for any future
  phase that wants to close rather than merely narrow these gaps.
- No blockers. No sealed verdict from Phases 2, 02.x, 3 or 03.1 was reopened, softened,
  recomputed or reinterpreted by this plan.

---
*Phase: 04-region-partitioning-regional-alignment-mknn*
*Completed: 2026-08-24*

## Self-Check: PASSED
- FOUND: notebooks/04_region_partition_mknn.ipynb
- FOUND: .planning/phases/04-region-partitioning-regional-alignment-mknn/04-FINDINGS.md
- FOUND: .planning/phases/04-region-partitioning-regional-alignment-mknn/04-06-SUMMARY.md
- FOUND: commit 18cb8be
- FOUND: commit 3d50575
