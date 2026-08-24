---
phase: 04-region-partitioning-regional-alignment-mknn
plan: 02
subsystem: manifold-curvature
tags: [curvature, density-correction, split-half-reliability, pre-registration, k-freeze, locality]

# Dependency graph
requires:
  - phase: 03-decoder-curvature-field
    provides: "curvature_probe.centroid_mean_curvature (D-05 gating estimator, density
      correction per D-06), cross_split_curvature.reliability_summary (split-half R_H),
      the uncorrected k=30/60/120/231 rows this plan supersedes"
provides:
  - "K_FREEZE_RULE, measure_r_over_R, freeze_k added to pu_curvature_rankability_run.py --
    the D4-07 pre-registration, committed BEFORE any density-corrected R_H was measured"
  - "notebooks/.cache/04_pu_curvature_rankability_corrected.jsonl: 6 rows, k in
    {30,60,120,231,350,500}, density_correct=True, k_density=30, d=20"
  - "notebooks/.cache/04_k_freeze.json: k_frozen=500, rule_fired=False (fallback provenance,
    not a detected plateau) -- the k plan 04-03 must inherit"
affects: [04-03, 04-04, 04-06]

tech-stack:
  added: []
  patterns:
    - "Pre-registration-before-measurement, enforced by commit ordering: Task 1 committed
      K_FREEZE_RULE and freeze_k with zero corrected R_H numbers on disk; Task 2's sweep ran
      only after that commit existed"
    - "freeze_k reads accumulated records from the record_path file itself (not just the
      current invocation's in-memory list), so a second CLI pass adding one k value still
      freezes against the full grid"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/pu_curvature_rankability_run.py

key-decisions:
  - "--k-density has no CLI default (None, not the pre-registered value 30) so that
    --density-correct without --k-density propagates straight into
    centroid_mean_curvature's own ValueError naming k_density, per the acceptance criterion
    -- the pre-registered value 30 is always passed explicitly at the call site instead"
  - "--freeze-out re-reads the full record_path file (filtered to matching
    density_correct/k_density/d, deduplicated by k) rather than only the current run's
    records, so the k=500 second pass correctly froze against the full 6-point grid rather
    than a single new row"
  - "k=500 second pass run in full per D4-06's floor ('run at least k=350, 500'), not
    skipped despite the rule being very unlikely to fire there -- the plan's own criterion
    for running it is 'rule_fired is false at 350', not a forecast of the outcome"

requirements-completed: [REGN-04]

coverage:
  - id: D1
    description: "K_FREEZE_RULE declared as a committed constant, freeze_k implementing it
      mechanically, and measure_r_over_R reproducing spike 003's locality statistic -- all
      committed before any density-corrected number existed"
    requirement: REGN-04
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -q (369 passed, 1 skipped, unaffected by this
          plan's changes -- no sealed module touched)"
        status: pass
      - kind: other
        ref: "freeze_k boundary-case assertions (fires at k=350 on a synthetic 5-point
          series, does not fire on a 2-point prefix, does not fire on a 1-point series) --
          Task 1 <verify> block, run directly"
        status: pass
    human_judgment: false
  - id: D2
    description: "Density-corrected split-half reliability sweep run at
      density_correct=True, k_density=30, d=20 across k in {30,60,120,231,350,500}; k frozen
      by D4-07's rule, which did not fire anywhere in the grid -- k_frozen=500 is the
      fallback largest-k-run outcome, not a detected plateau"
    requirement: REGN-04
    verification:
      - kind: other
        ref: "notebooks/.cache/04_pu_curvature_rankability_corrected.jsonl (6 rows) and
          notebooks/.cache/04_k_freeze.json (k_frozen=500, rule_fired=false, rule_text
          byte-identical to the committed K_FREEZE_RULE constant) -- both gitignored cache
          artifacts, inspected directly"
        status: pass
    human_judgment: true
    rationale: "Whether a not-fired k-freeze outcome is an acceptable basis for plan 04-03's
      downstream partition (rather than a signal to revisit the estimator or the sweep
      range) is a scientific-validity judgment the plan explicitly reserves for the phase
      record (04-FINDINGS.md), not something this plan's automated checks can settle."

duration: ~2h40min (dominated by compute: k=350 alone took ~59min single-threaded-equivalent
  wall time within the BLAS-parallelized run, k=500 similar)
completed: 2026-08-24
status: complete
---

# Phase 4 Plan 2: Density-corrected curvature k-freeze — D4-07 rule declared, then a not-fired outcome measured Summary

**D4-07's spacing-free absolute-increment freeze rule was committed to source before any
density-corrected number existed; the corrected split-half-reliability sweep then ran
`k = 30, 60, 120, 231, 350, 500` at `density_correct=True, k_density=30, d=20` and the rule
never fired — `median_R_H` climbed from 0.028 to 0.344 across the grid, still short of the
0.5 floor and with increments (0.065, 0.065, 0.076, 0.052, 0.058) never collapsing toward the
0.03 ceiling, so `k_frozen = 500` is recorded as the fallback "largest k actually run," not a
detected plateau.**

## Performance

- **Duration:** ~2h40min wall clock (two background sweep passes; k=350 alone ran ~59min,
  k=500 ~52min, dominated entirely by `centroid_mean_curvature`'s per-point Python loop at
  `n=10,000`, not by any code written in this plan)
- **Started:** 2026-08-24T03:24Z (first commit)
- **Completed:** 2026-08-24T06:05Z (second sweep pass finished)
- **Tasks:** 2/2 complete
- **Files modified:** 1

## Accomplishments
- `K_FREEZE_RULE` (D4-07's rule, verbatim), `measure_r_over_R` (spike 003's locality
  statistic), and `freeze_k` (the mechanical application of the rule to a per-k record list)
  added to `pu_curvature_rankability_run.py` and committed in a standalone commit **before**
  any density-corrected `R_H` value was measured — the ordering constraint this plan exists
  to enforce.
- `run_cell` extended to thread `density_correct`/`k_density` into every
  `centroid_mean_curvature` call site, and to record `r_knn`, `R_cloud`, `r_over_R`,
  `density_correct`, `k_density`, and `subsample_file` on every row. The runner's pre-plan
  default behaviour (no new flags) is unchanged — verified by `--smoke` producing the same
  row shape as before, plus the new `r/R` column.
- The corrected split-half sweep ran end to end: `notebooks/.cache/
  04_pu_curvature_rankability_corrected.jsonl` now holds 6 rows (k = 30, 60, 120, 231, 350,
  500), each with `density_correct=True`, `k_density=30`, `d=20` explicit, and the full
  reliability/locality/direction table described below.
- `notebooks/.cache/04_k_freeze.json` records the freeze outcome: **`k_frozen = 500`,
  `rule_fired = false`**, `rule_text` byte-identical to the committed `K_FREEZE_RULE`
  constant, the full per-k `median_R_H` and delta tables, and the reason string.

## The measured sweep

| k | median_R_H | delta vs prev | fraction_negative | median r_dir | h_spread | r_knn | R_cloud | r/R | wallclock (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.0279 | — | 0.364 | +0.034 | 4.85x | 0.264 | 0.377 | 0.699 | 59 |
| 60 | 0.0927 | +0.0648 | 0.164 | +0.105 | 4.03x | 0.279 | 0.377 | 0.740 | 275 |
| 120 | 0.1573 | +0.0646 | 0.062 | +0.175 | 4.04x | 0.297 | 0.377 | 0.788 | 1,169 |
| 231 | 0.2337 | +0.0763 | 0.023 | +0.260 | 4.03x | 0.317 | 0.377 | 0.839 | 1,699 |
| 350 | 0.2853 | +0.0516 | 0.009 | +0.314 | 4.00x | 0.331 | 0.377 | 0.878 | 3,566 |
| 500 | 0.3436 | +0.0583 | 0.004 | +0.374 | 3.94x | 0.345 | 0.377 | **0.915** | 3,105 |

## THE k-FREEZE OUTCOME — stated plainly for 04-FINDINGS.md to lift verbatim

**D4-07's rule did not fire anywhere in this 6-point grid.** At no `k` was the increment over
the previous point both (a) strictly less than 0.03 AND (b) at a level where `median_R_H` was
already `>= 0.5`. `median_R_H` ends the grid at 0.3436 — still well short of the 0.5 floor —
and the per-step gain is **not collapsing toward the 0.03 ceiling**: it went 0.0648, 0.0646,
0.0763, 0.0516, 0.0583 across k=60..500, rising again at the last point rather than settling.

Per the rule's own pre-registered fallback (declared in `K_FREEZE_RULE`, committed in Task 1,
before this number existed): **`k_frozen = 500` is the largest k actually run, and the outcome
is recorded as `rule_fired = false`.** This is a compute-budget ceiling, not a plateau the rule
detected. **Downstream plans (04-03 onward) inherit `k=500` because it is where the
pre-registered sweep grid ended, not because the field's reliability leveled off there.** At
the observed rate, `median_R_H` would need several more sweep points beyond 500 to reach 0.5,
and nothing in the recorded deltas suggests the increment is approaching 0.03 either. Neither
threshold (0.03, 0.5) was moved to make the rule fire, and neither was moved to make it fire
"more honestly" either — both remain exactly as declared in the Task 1 commit.

This is a valid, reportable result precisely because the rule was declared first: had the
thresholds been chosen after seeing this sweep, a rule that never fires would invite the
suspicion that it was tuned to fire. Declaring it blind and then watching it not fire is what
makes the `k=500` fallback trustworthy as a *stated limitation* rather than a post-hoc
rationalization.

## Locality (`r/R`), reported and not gated on

`r/R` rose monotonically from 0.699 at k=30 to **0.9151 at k=500** — still below 1, so on
**this** PU cloud the k=500 neighbourhood has **not** grown past the cloud's own radius from
its centroid. This differs from spike 003's own fixture, where `r/R` measured 1.0992 at
k=500 — past 1, i.e. no longer local there. The two numbers are not directly comparable (they
come from different clouds), but the direction of the finding is worth stating plainly: PU's
`r/R` trajectory, extrapolated at its current per-point growth, would likely cross 1 somewhere
past k=500 too, though this plan does not run that extrapolation as a measurement. Per the
plan's explicit instruction, `r/R` is reported here for context and is **never used as a
gate** — spike 003's own sweep found Spearman rank correlation is scale-free with respect to
locality regime, so a smaller or larger `r/R` predicts nothing about rankability by itself.

## Dynamic range and the D4-05 spread comparison

The corrected field's spread stayed near-flat across the whole grid (4.85x at k=30, settling
to ~4.0x by k=120 and 3.94x at k=500) — consistent in shape with D4-05's uncorrected
measurement (~4.8x-4.86x at k=30-231), and, per that same finding, **far nearer the
unrankable `quadratic_bowl` reference (1.4x, `rho +0.03`) than the rankable `cubic`/`ridge`
references (28.2x/34.3x, `rho +0.61`/`+0.41`)**. As D4-05 already established and this plan
does not re-litigate: direction is a unit vector and does not consume the magnitude spread,
and spike 003 measured `rho = +0.48` at a spread of 1.1x and `rho = +0.36` at 36x on the same
fixture family — Spearman rank correlation is scale-free, so dynamic range predicts nothing
about rankability either way. The spread number is reported for context, not gated on.

## Reading `R_H` correctly — reliability is not correctness

`median_R_H` rising and `fraction_negative` falling monotonically across the grid (0.364 down
to 0.004) shows the two disjoint-data halves increasingly agree with each other as `k` grows.
**This is not evidence the field is correct.** As the runner's own read-out states: split-half
reliability certifies that a measurement *reproduces*, never that it *is right* — a bias both
halves of the cloud share (e.g. a systematic estimator bias, or a density artifact common to
both halves) is perfectly reliable by this statistic and completely invisible to it. There is
no ground truth on real PU data, so this sweep can never be upgraded from a reproducibility
claim to a correctness claim by running more of the same measurement. This framing is
unchanged from D4-03's accepted gap, restated here because it directly bears on how the
k=500 fallback should be read: a highly reliable field at a fallback k is still only a
reliable field, not a validated one.

## Corrected vs. uncorrected — superseded, not extended

Per D4-15, the density-corrected sweep is the headline field going forward. The four
pre-existing uncorrected rows in `notebooks/.cache/03.2_pu_curvature_rankability.jsonl`
(`k = 30, 60, 120, 231`, `median_R_H = 0.0779, 0.2474, 0.428, 0.5894`) are **superseded, not
extended** — they remain on disk, byte-for-byte unmodified (verified: still 4 lines, still
those exact k values and `median_R_H` numbers), and were never touched by this plan's runs,
which wrote exclusively to the new `04_pu_curvature_rankability_corrected.jsonl` path. The
correction's justification, stated per D4-15's own recorded caveat, is the ~8-10% median
relative error reduction measured on a genuinely curved, strongly-skewed fixture in
`02.5-02-SUMMARY.md` — not the retracted flat-fixture inertness claim.

Note for anyone comparing the two tables directly: the corrected `median_R_H` values are
systematically **lower** than the uncorrected ones at the same k (e.g. 0.0279 vs 0.0779 at
k=30; 0.2337 vs 0.5894 at k=231) — an expected consequence of weighting neighbours by inverse
local density rather than a regression, since the two estimators are measuring genuinely
different (weighted vs. unweighted) quantities, not the same quantity with different noise.

## Task Commits

1. **Task 1: Declare the D4-07 freeze rule and add the corrected / r-over-R sweep machinery**
   - `5a05541` (feat) — `K_FREEZE_RULE`, `measure_r_over_R`, `freeze_k`, CLI flags, record
     schema extension. Committed with **zero** density-corrected `R_H` numbers on disk.
2. **Task 2: Run the density-corrected sweep and freeze k mechanically** — no code commit
   (this task's deliverable is the gitignored `notebooks/.cache/` sweep and freeze
   artifacts, reproducible from Task 1's committed runner; nothing under version control
   changed). Verified via the artifact-shape assertions from the plan's `<verify>` block,
   run directly against the produced files (see Coverage D2).

**Plan metadata:** committed as part of this SUMMARY's own docs commit.

## Files Created/Modified
- `notebooks/diagnostics/pu_curvature_rankability_run.py` - `K_FREEZE_RULE`,
  `measure_r_over_R`, `freeze_k`, `density_correct`/`k_density` threading through `run_cell`,
  `--density-correct`/`--k-density`/`--freeze-out` CLI flags, `r/R` in every printed row and
  in `summarize()`'s read-out

## Decisions Made
- **`--k-density` has no CLI default**, so that `--density-correct` passed without
  `--k-density` propagates directly into `centroid_mean_curvature`'s existing
  `k_density`-naming `ValueError`, satisfying the plan's acceptance criterion by reusing the
  sealed estimator's own guard rather than adding a second, parallel validation path. The
  pre-registered value (30) is always supplied explicitly at every call site that uses it
  (both smoke invocations and the real sweep), so this never silently defaults.
- **`--freeze-out` reads the full record file, not just the current invocation's in-memory
  rows**, filtered to the current run's `(density_correct, k_density, d)` and deduplicated
  by `k` (last-written row per `k` wins). This was necessary for the plan's own two-pass
  protocol to work correctly: the k=500 second pass appended exactly one new row, and without
  reading the accumulated file, `freeze_k` would have evaluated a rule against a single-point
  series (never fires by construction) rather than the full 6-point grid the rule needs to
  see.
- **k=500 was run in full**, per D4-06's explicit floor ("run at least k=350, 500") and
  Task 2's own branching instruction, even though the observed trend made it very unlikely to
  change `rule_fired`. The plan's criterion for running the second pass is objective
  (`rule_fired` false at 350), not a forecast of whether it would help — running it anyway is
  what "characterise the trend honestly" means in practice.

## Deviations from Plan

### Auto-fixed Issues
None — no bugs, missing critical functionality, or blocking issues were found in the plan's
own action text that required Rule 1-3 auto-fixes.

**One plan-ambiguity resolution, not a rule-triggered deviation:** the plan's action text for
Task 1 describes `--k-density` as "(int, default 30, ...)" while its own acceptance criteria
require that omitting `--k-density` alongside `--density-correct` raise a `ValueError`. A
literal argparse default of `30` would make that omission silently succeed instead. Resolved
in favor of the acceptance criterion (the machine-checked contract) over the action prose: the
argparse default is `None`, and the pre-registered value `30` is supplied explicitly at both
call sites that use it. Documented here rather than silently picked, since it is a genuine
tension in the plan's own text.

---

**Total deviations:** 0 rule-triggered auto-fixes; 1 documented plan-ambiguity resolution
(above).
**Impact on plan:** None on scope or correctness — the resolution only affects how the
`ValueError` path is reached (CLI-level default vs. propagation into the existing sealed
guard), and the acceptance-criteria-mandated behaviour was verified directly (see Task 1
verification below).

## Issues Encountered
- **The first k=500 launch died silently** at ~57 minutes into the run, with a 0-byte log
  and no row written to `04_pu_curvature_rankability_corrected.jsonl` — a stdout-buffering
  artifact of the `nohup ... &` + `disown` launch pattern used for this plan's long-running
  background sweeps (Python's stdout buffers fully when not attached to a TTY, and the
  process was killed/lost before a flush). The coordinator relaunched it detached with
  `setsid` and `python -u` (unbuffered), which completed cleanly and produced the 6th row.
  No code in this plan needed to change for this — it is an execution-environment artifact
  of how the sweep was invoked, not a defect in `pu_curvature_rankability_run.py` itself.
  Not re-run a third time, per the coordinator's explicit instruction.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- **Plan 04-03 (the partition rule) inherits `k_frozen = 500` as a fallback compute-ceiling
  value, not a detected reliability plateau.** This must be stated with the same clarity
  there and in `04-FINDINGS.md` — no plan downstream of this one should describe the k
  choice as "the reliability sweep converged" or similar language implying the rule fired.
- `04_pu_curvature_rankability_corrected.jsonl` and `04_k_freeze.json` are both on disk,
  reproducible from the committed runner (`pu_curvature_rankability_run.py`) via the exact
  commands recorded above, should either need regenerating.
- No blockers for 04-03. The density-corrected field at k=500 (`density_correct=True,
  k_density=30, d=20`) is the field 04-03's partition operates on; its accepted gaps
  (reliability-not-correctness, the not-fired k-freeze, and the direction-axis-only
  partition scope from D4-09) are all separate, independently-tracked limitations that
  04-06's phase record must state together, not conflated into one caveat.

---
*Phase: 04-region-partitioning-regional-alignment-mknn*
*Completed: 2026-08-24*

## Self-Check: PASSED
- FOUND: notebooks/diagnostics/pu_curvature_rankability_run.py
- FOUND: commit 5a05541
- FOUND: notebooks/.cache/04_pu_curvature_rankability_corrected.jsonl
- FOUND: notebooks/.cache/04_k_freeze.json
