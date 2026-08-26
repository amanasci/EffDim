---
phase: 07-curvature-conditioned-crossmodal-alignment
plan: 05
subsystem: research-instrumentation
tags: [curvature, mknn, spearman, permutation-test, positive-control, density-confound, findings-report]

# Dependency graph
requires:
  - phase: 07-curvature-conditioned-crossmodal-alignment
    provides: "The frozen record (07-04, notebooks/.cache/07_crossmodal_curvature.jsonl, 8 rows) and fields npz -- the phase's sole numeric source for this plan's notebook and findings"
provides:
  - "notebooks/07_crossmodal_curvature_check.ipynb -- the reporting notebook, executed end to end and committed with outputs, reading only the frozen record (no decoder fit, no MKNN recomputation except a proof-of-no-drift re-derivation of the per-point array, asserted byte-identical to the record's own observed_rho before plotting)"
  - "07-FINDINGS.md -- the phase's answer (ASSOCIATION DETECTED), the D7-02 positive-control power evidence with threshold margins, the instrument's honest fidelity RANGE (+0.53 to +0.99), accepted limitations, explicit non-claims, and freeze-to-run provenance"
  - "The human-approved amendment sharpening the positive-control detection-floor claim: 0.05 is the smallest grid point that cleared, not the floor; the actual null-band edge is ~0.0205; the true floor is unresolved within 0.021-0.05; the density-controlled residuals at d=20/32 are shown sitting at that same marginal-detectability magnitude"
affects: []

tech-stack:
  added: []
  patterns:
    - "Reporting notebook re-derives one quantity (the per-point MKNN array) the frozen record didn't persist, but proves no drift by asserting the re-derivation reproduces the record's own observed_rho to floating-point precision before using it for anything -- a re-computation that is provably not a second, drifted convention."
    - "Findings amendment applied surgically to three named sections (positive-control table, detection-floor prose, density-comparison table) rather than a full rewrite, keeping the untouched two-thirds of the document (verdict, instrument range, limitations, non-claims, provenance) byte-stable across the human checkpoint round-trip."

key-files:
  created:
    - notebooks/07_crossmodal_curvature_check.ipynb
    - .planning/phases/07-curvature-conditioned-crossmodal-alignment/07-FINDINGS.md
  modified: []

key-decisions:
  - "Verdict stated verbatim from the frozen record's verdict row: ASSOCIATION DETECTED. Not softened, not requalified, across both the original write and the human-requested amendment."
  - "The positive control's smallest_cleared_target=0.05 is kept as the recorded pre-registered quantity but is no longer described as 'the floor' -- the amendment states the actual null-band edge (~0.0205, read off the 0.02 row's own positive-tail threshold) and names the interval 0.021-0.05 as unresolved by this four-point grid."
  - "The density-controlled residuals at d=20 (-0.02419) and d=32 (-0.02172) are now compared directly against that same ~0.0205 null-band edge (18% and 6% above it respectively), with an explicit caveat that a partial correlation borrows the raw statistic's threshold and is orientation only, not a significance test -- attached to the pre-existing 17%/16% figures as well as the new comparison."
  - "CLAUDE.md's Swiss roll gate is declared satisfied by 02.6_swiss_roll_plainae_curvature_check.ipynb rather than waived: Phase 7 introduces no new manifold-learning model, reusing cae.PlainAutoEncoder and decoder_curvature.plain_decoder_curvature verbatim."

requirements-completed: [D7-01, D7-02, D7-03, D7-05, D7-07]

coverage:
  - id: D1
    description: "The reporting notebook, executed end to end with outputs, reading only the frozen record"
    requirement: "D7-01"
    verification:
      - kind: other
        ref: "notebooks/07_crossmodal_curvature_check.ipynb -- 14 cells, all 7 code cells carry non-empty outputs, grep confirms no train_plain_ae/plain_decoder_curvature/permutation_null call"
        status: pass
    human_judgment: false
  - id: D2
    description: "07-FINDINGS.md states the answer, the D7-02 power evidence, the instrument fidelity range, accepted limitations, and explicit non-claims"
    requirement: "D7-02"
    verification:
      - kind: other
        ref: ".planning/phases/07-curvature-conditioned-crossmodal-alignment/07-FINDINGS.md -- contains verdict string verbatim, 0.53/0.99 fidelity range, no bare point-estimate fidelity figure"
        status: pass
    human_judgment: false
  - id: D3
    description: "Human read and approved both artifacts, with one documentation amendment to the positive-control detection-floor framing"
    requirement: "D7-05"
    verification: []
    human_judgment: true
    rationale: "Checkpoint required a human to read the notebook and findings and judge whether the verdict's register and the non-claims list were honestly stated -- not automatable."
  - id: D4
    description: "Amendment applied: threshold margins added to the positive-control table, the 0.05-as-floor claim corrected, and the density-controlled residuals compared against the real null-band edge with a partial-correlation caveat"
    requirement: "D7-03"
    verification:
      - kind: other
        ref: "git diff df8502f^..df8502f -- .planning/phases/07-curvature-conditioned-crossmodal-alignment/07-FINDINGS.md -- 45 insertions, 13 deletions, confined to Sec 1.3, Sec 2, Sec 1.4; every added number independently recomputed from notebooks/.cache/07_crossmodal_curvature.jsonl and matched to 5 decimal places"
        status: pass
    human_judgment: false

duration: ~5h07m total across this plan (Task 1+2 authored and committed in a prior session ending ~12:19 local; Task 3's blocking checkpoint awaited human review; this continuation session applied the one-amendment approval, re-ran the full test suite, and closed the plan, ending ~17:24 local)
completed: 2026-08-26
status: complete
---

# Phase 7 Plan 5: The Reporting Notebook and Findings, Amended and Approved Summary

**The phase's research question closes ASSOCIATION DETECTED, reported in a notebook that reads only the frozen record and a findings document sharpened by one human-requested amendment: the positive control's `smallest_cleared_target=0.05` is not "the detection floor" — the real null-band edge is ~0.0205, and the true floor is unresolved somewhere in 0.021-0.05, a distinction that matters because the density-controlled residuals at d=20 and d=32 sit at exactly that marginal magnitude.**

## Performance

- **Duration:** ~5h07m total across this plan — see frontmatter `duration` for the session breakdown
- **Completed:** 2026-08-26
- **Tasks:** 3 (Task 1: `9c6ed0e`; Task 2: `48a3917`; Task 3: human-approved checkpoint with amendment, `df8502f`)
- **Files modified:** 2 (`notebooks/07_crossmodal_curvature_check.ipynb` created; `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-FINDINGS.md` created, then amended in place)

## Accomplishments

- **The reporting notebook (`notebooks/07_crossmodal_curvature_check.ipynb`, 14 cells) reads the frozen record and fields npz and recomputes nothing production-relevant.** The one exception — re-deriving the per-point MKNN array, which the fields npz doesn't persist — is proven not to drift: the cell asserts the re-derived array reproduces the record's own `observed_rho` at d=20 to floating-point precision before it is used for the banded scatter plot.
- **`07-FINDINGS.md` states the phase's answer verbatim from the record's verdict row (`ASSOCIATION DETECTED`)**, alongside the D7-02 positive control's power evidence, the instrument's honest fidelity RANGE (`+0.53` to `+0.99`, never a point estimate), the accepted limitations (single seed, all-10,000-rows evaluation, no reconstruction plateau through d=48, non-gating density/hubness diagnostics, the 21-distinct-value MKNN tie structure), the explicit non-claims (no ground truth for PU curvature, no CKA, no extrapolation to n=101,725, no reopening of Phases 2-6, Phase 4's `HOLDS` not cited as curvature-alignment evidence), and provenance (freeze commit `f032745`, run commit `a453736`, strict-ancestor proof, 10 commits between them).
- **The human checkpoint (Task 3) approved with one amendment**, prompted by a question about why `target_rho=0.02` failed detection when `achieved_rho=0.02004` matched the plant almost exactly. The answer exposed that the document was quoting the detection floor imprecisely.
- **The amendment (three edits, one commit `df8502f`, 45 insertions / 13 deletions confined to Sec 1.3, Sec 2, Sec 1.4):**
  1. **Sec 1.3's positive-control table gained threshold-margin columns.** The 0.02 row's margin is now explicit: `achieved 0.020037` vs. `positive-tail threshold 0.020506` = `-0.00047`, roughly 2% of its own threshold — the plant succeeded (bisection converged) but landed a hair inside the noise band, a distinction the table now states in prose (`achieved_rho` measures the plant; `clears_either` measures the test).
  2. **Sec 2's "recovers a planted effect as small as `rho=0.05`" claim was corrected.** `smallest_cleared_target=0.05` is kept as the recorded, pre-registered quantity, but is no longer described as "the floor": the grid has no point between 0.02 and 0.05, the actual null-band edge sits at approximately 0.0205 (read off the 0.02 row's own threshold), so the true detection floor lies somewhere in the unresolved interval 0.021-0.05.
  3. **Sec 1.4 gained a direct comparison table** — density-controlled residual vs. the ~0.0205 null-band edge — showing d=20 at ~18% above, d=25 at ~3.2x above, d=32 at ~6% above, with the conclusion stated plainly: the curvature-specific signal at d=20 and d=32 sits at the magnitude where this phase's own positive control demonstrated the test cannot reliably separate signal from noise; only d=25 survives density-control with room to spare. The pre-existing 17%/16% figures were given the same caveat: a partial correlation borrows the raw statistic's threshold and is orientation only, not a significance test.
  - Every added number was independently verified against the frozen record (`notebooks/.cache/07_crossmodal_curvature.jsonl`) before the edit, not copied from the amendment instructions unchecked.
- **No causal language was found or introduced.** A scan for phrasing like "curvature reduces alignment" or "curvature impacts MKNN" against the full document found none — the existing text consistently uses "association," "correlated," and reserves "explain" for the research question's own framing as a question.
- **No sealed module, the frozen record, the fields npz, or the notebook was touched by the amendment.** `git diff --stat` for `df8502f` shows exactly one file changed: `07-FINDINGS.md`. The verdict string is unchanged.
- **Full regression: `notebooks/pu_manifold/tests/ -q` — 573 passed, 1 skipped**, matching the environment's stated baseline exactly, run after the amendment.

## Task Commits

1. **Task 1: The reporting notebook, executed end to end with outputs** — `9c6ed0e` (feat) [prior session]
2. **Task 2: `07-FINDINGS.md`** — `48a3917` (docs) [prior session]
3. **Task 3: Read and approve** — human-approved with one documentation amendment; amendment applied and committed as `df8502f` (docs) in this continuation session

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `notebooks/07_crossmodal_curvature_check.ipynb` — the reporting notebook: research-question framing, record/npz load with freeze and run commit printout, the three-`d` table, the banded `||H||`-vs-MKNN scatter (proof-of-no-drift re-derivation), the `||H||` distributions (order of magnitude only), the D7-02 positive control, the non-gating D7-03 diagnostics, and a final verdict/fidelity read-out.
- `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-FINDINGS.md` — the phase's answer, power evidence, instrument fidelity range, Swiss roll gate declaration, accepted limitations, non-claims, provenance, and the two named planner-assumption resolutions (D7-03 density sign, D7-07 `ALIGNMENT_METRIC` scope proof). Amended in this session per the human checkpoint.

## Decisions Made

See `key-decisions` in frontmatter. In short: the verdict stays `ASSOCIATION DETECTED` throughout; `smallest_cleared_target=0.05` stays the recorded pre-registered quantity but is no longer called "the floor"; the density-controlled residuals at d=20/32 are now shown sitting at the same marginal-detectability magnitude the positive control itself measured; and the Swiss roll gate is declared satisfied by name (`02.6_swiss_roll_plainae_curvature_check.ipynb`) rather than skipped, because Phase 7 introduces no new model.

## Deviations from Plan

None beyond the plan's own Task 3 checkpoint mechanism — the human-requested documentation amendment is the checkpoint's designed `ratify-with-amendments`-style outcome, not an unplanned deviation. No Rule 1-4 auto-fix was needed: this was a documentation-precision correction requested by the human reviewer, not a bug, missing functionality, blocker, or architectural change discovered during execution.

## Issues Encountered

None. The amendment's three edits applied cleanly, every added number was independently verified against the frozen record before writing, the causal-language scan found nothing to fix, and the full test suite reproduced the stated baseline exactly (573 passed, 1 skipped).

## Known Stubs

None. Both artifacts report real measured numbers from the frozen record; no placeholder or synthetic surrogate appears anywhere.

## Threat Flags

None. `T-07-06` (notebook recomputing rather than reading) is confirmed mitigated: the notebook's one re-derivation is proven non-drifting by an in-notebook assertion against the record's own `observed_rho`. `T-07-07` and `T-07-03` (repudiation of findings claims and stamped SHAs) are confirmed mitigated: every number in the amended findings was independently checked against `notebooks/.cache/07_crossmodal_curvature.jsonl` before this session's edit, and the provenance section's SHAs are untouched by the amendment.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Phase 7 (`07-curvature-conditioned-crossmodal-alignment`) is complete: all 5 plans executed, the frozen record is sealed, the reporting notebook and findings are committed and human-approved. The milestone's headline result — `ASSOCIATION DETECTED`, carried mostly by d=20 and d=25, mostly density at d=20, and marginal at d=20/32 by the phase's own positive-control-measured detection threshold — is the terminal, non-clean-but-recorded answer to this milestone's research question. No further plan is scheduled under this phase.

---
*Phase: 07-curvature-conditioned-crossmodal-alignment*
*Completed: 2026-08-26*

## Self-Check: PASSED

- FOUND: `notebooks/07_crossmodal_curvature_check.ipynb`
- FOUND: `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-FINDINGS.md`
- FOUND commit `9c6ed0e` in `git log --oneline --all`
- FOUND commit `48a3917` in `git log --oneline --all`
- FOUND commit `df8502f` in `git log --oneline --all`
