---
phase: 05-curvature-conditioned-linear-decodability
plan: 02
subsystem: geometry
tags: [curvature, chart-autoencoder, decoder-side, spearman, seed-spread, npz-cache, spike-findings]

# Dependency graph
requires:
  - phase: 03-decoder-curvature-field
    provides: the three sealed converged CAE checkpoints
      (03_converged_cae_pu_nc4_seed2026081{3,4,5}.pt) and chart_curvature.chart_curvature_field,
      plus 03-09-SUMMARY.md's single-seed field measurement this plan re-measures across all
      three seeds
  - phase: 05-curvature-conditioned-linear-decodability (05-01)
    provides: extract_seed_field, run_field_mode, load_pu_pair, load_converged_model,
      _spearman_report and the cache.npz_cache/json_cache manifest discipline this plan extends
provides:
  - notebooks/.cache/05_curvature_field_seed2026081{3,4,5}.npz -- three independently cached
    decoder-side curvature fields over all 10,000 PU points, one per sealed seed, each carrying
    H_norm, H_vec, chart_assignment, metric_condition_number, lambda_min, lambda_max, det_g,
    log10_det_g, n_charts_used
  - notebooks/.cache/05_inter_seed_diagnostics.json -- D5-05's pairwise inter-seed Spearman with
    the direction-cosine axis beside every entry, per-seed distinct-value/median/chart-fraction
    statistics, and the r/R null non-application disclosure
  - _direction_axis_report and run_inter_seed_diagnostics in
    notebooks/diagnostics/curvature_probe_decodability_run.py
affects: [05-03, 05-04, 05-05, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Diagnostics-after-extraction: run_inter_seed_diagnostics runs at the end of every --mode
      field invocation, but only fires once all three CANONICAL_SEED_STEMS npz artifacts exist
      on disk -- a partial invocation (one seed per command, this plan's own execution shape)
      prints a one-line skip notice and writes nothing rather than erroring"
    - "A rank statistic is never printed alone: every _spearman_report call in
      run_inter_seed_diagnostics is immediately followed by the paired _direction_axis_report
      call for the same seed pair, satisfying the spike-findings-effdim disclosure rule
      mechanically rather than by convention"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/curvature_probe_decodability_run.py

key-decisions:
  - "Distinct-H_norm-value counts are reported at three precisions (exact float64, rounded to 6
    decimals, rounded to 12 decimals), not one -- 03-09-SUMMARY.md's own table measured
    'filled hist bins' over a 20-bin histogram (4/20, 3/20 for seeds 14/15), which this plan's
    remeasurement reproduces exactly, but RESEARCH.md's restatement of that as '3-4 distinct
    values, one per surviving chart' is not the same measurement and does not hold at any
    rounding precision this plan checked (seed 14: 5,301 exact / 16 at 6dp / 5,234 at 12dp
    distinct values, not 4; seed 15: 9,852 exact / 59 at 6dp / 9,827 at 12dp, not 3) -- see
    Findings below"
  - "The direction axis is computed once per pair, immediately beside its Spearman value, inside
    the same function rather than as a separate pass -- makes the spike-findings 'never report a
    rank statistic without the direction axis beside it' rule structurally true of the code, not
    just true by writer discipline"

requirements-completed: [D5-03, D5-05, D5-06]

coverage:
  - id: D1
    description: "Three independently cached decoder-side curvature fields exist over all
      10,000 PU points (one per sealed CAE seed), produced by
      chart_curvature.chart_curvature_field(model, x64, mode='reverse') and cached through
      cache.npz_cache with a full cfg manifest"
    requirement: D5-03
    verification:
      - kind: integration
        ref: "python -c assert script over the three 05_curvature_field_seed*.npz artifacts (shape, finiteness, sidecar meta.json) -- see plan 05-02 Task 1 <verify>"
        status: pass
      - kind: integration
        ref: "re-running --mode field --seeds 20260813 completes in 3.6s wall (cache hit, not recompute)"
        status: pass
    human_judgment: false
  - id: D2
    description: "D5-05's inter-seed agreement measured and recorded in
      notebooks/.cache/05_inter_seed_diagnostics.json: three pairwise Spearman entries with the
      direction-cosine axis beside every one, three per-seed summaries, and the r_over_R=null
      non-application disclosure"
    requirement: D5-05
    verification:
      - kind: integration
        ref: "python -c assert script over 05_inter_seed_diagnostics.json's schema -- see plan 05-02 Task 2 <verify>"
        status: pass
      - kind: unit
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q (385 passed, 1 skipped, 1 xfailed)"
        status: pass
    human_judgment: false

duration: ~3h20m (session span; ~2h29m of that is the three seeds' CPU-bound field extraction --
  see Deviations for the idle gap inside that span)
completed: 2026-08-24
status: complete
---

# Phase 5 Plan 2: Three-Seed Curvature Field Extraction and Inter-Seed Diagnostics Summary

**Decoder-side `||H||` fields extracted for all three sealed CAE seeds over the full 10,000-point PU cloud (52-minute-class CPU wall-clock each), plus D5-05's inter-seed diagnostics -- and the three seeds' fields disagree on both rank order (Spearman -0.14 / +0.20 / -0.27, sign-inconsistent) and direction (median cosine ~0.001-0.004, ~46-48% anti-aligned) -- the exact numbers 05-03's one-way pooling checkpoint needs.**

## Performance

- **Duration:** ~3h20m session span (started ~12:16 local, completed ~15:35 local)
- **Started:** 2026-08-24T16:16:00Z (approx, first smoke-test invocation)
- **Completed:** 2026-08-24T19:35:00Z (approx)
- **Tasks:** 2
- **Files modified:** 1 (`notebooks/diagnostics/curvature_probe_decodability_run.py`)

## Accomplishments

- All three sealed CAE seeds' decoder-side curvature fields extracted over the full 10,000-row
  `legacysurvey` subsample via `chart_curvature.chart_curvature_field(model, x64,
  mode="reverse")`, each independently cached at `notebooks/.cache/05_curvature_field_seed{seed}.npz`
  with a sidecar `.meta.json` manifest (seed, n_charts, mode, batch_size, n_rows, subsample_file,
  curvature_convention, source_function)
- Resolved subsample file: `subsample_20260729_a79b3460b838fd0a.npz` -- the identical file
  `04-FINDINGS.md` names, confirmed by direct read of `load_pu_pair`'s resolution output
- `_direction_axis_report` added: row-normalizes two seeds' `H_vec` fields (with the codebase's
  own `np.maximum(norm, 1e-12)` divide-by-zero floor), reports median/q25/q75 cosine and the
  anti-alignment fraction -- run immediately beside every `_spearman_report` call, never
  separately
- `run_inter_seed_diagnostics` added: writes `notebooks/.cache/05_inter_seed_diagnostics.json`
  through `cache.json_cache` once all three canonical seeds are cached -- three pairwise
  Spearman entries, three pairwise direction entries, three per-seed summaries, and the
  `r_over_R = null` non-application disclosure with its reason string
- No pooled field, no bucket edges, no pre-registered constant set, no PU probe number anywhere
  in the repository at the end of this plan
- `notebooks/pu_manifold/linear_probe.py` reached this commit unchanged from 05-01 (`git status
  --porcelain` empty)
- Full test suite green: `385 passed, 1 skipped, 1 xfailed`

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract all three seeds' decoder-side curvature fields** - no source diff (see
   Deviations -- this task's entire deliverable is three gitignored `notebooks/.cache/` npz
   artifacts, per `CLAUDE.md`'s "Milestone artifacts live in the gitignored notebooks/.cache/")
2. **Task 2: Inter-seed agreement, with the direction axis beside every rank statistic** -
   `ed8f3f7` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `notebooks/diagnostics/curvature_probe_decodability_run.py` - Added `_direction_axis_report`,
  `_load_cached_seed_field`, `run_inter_seed_diagnostics`, and the `CANONICAL_SEED_STEMS`
  constant; wired `run_inter_seed_diagnostics` to run at the end of `run_field_mode`
- `notebooks/.cache/05_curvature_field_seed20260813.npz` (+`.meta.json`) - gitignored, not
  committed
- `notebooks/.cache/05_curvature_field_seed20260814.npz` (+`.meta.json`) - gitignored, not
  committed
- `notebooks/.cache/05_curvature_field_seed20260815.npz` (+`.meta.json`) - gitignored, not
  committed
- `notebooks/.cache/05_inter_seed_diagnostics.json` - gitignored, not committed

## Findings

### The measured field, per seed

| seed | wallclock | median `\|\|H\|\|` | min | max | n_charts_used | median `log10 det(g)` | distinct (exact) | distinct (round 6dp) | distinct (round 12dp) | 20-bin hist filled |
|---|---|---|---|---|---|---|---|---|---|---|
| 20260813 | 2964.6s | 1,363.14 | 681.33 | 4,283.93 | 2 | -68.454 | 10,000 | 10,000 | 10,000 | 20/20 |
| 20260814 | 2970.3s | 51,437.9 | 29,699.4 | 66,977.5 | 4 | -165.569 | 5,301 | 16 | 5,234 | 4/20 |
| 20260815 | 3031.4s | 70,794.1 | 51,694.9 | 75,252.5 | 3 | -165.650 | 9,852 | 59 | 9,827 | 3/20 |

Total field-extraction wallclock: 8,966.3s (~2h29m) across the three seeds, close to the
~2.6h budgeted estimate. Medians match `03-09-SUMMARY.md`'s single-seed-of-three measurement to
within ~0.3% (1,363.1 here vs 1,359.0 there for seed 20260813 -- the small difference is `03-09`'s
own reported "all-points" 1,363.1 vs "unflagged" 1,359.0 distinction, not a new discrepancy) and
reproduce the 52x median range exactly (1,359-1,363 / 51,438 / 70,794).

**A discrepancy this plan's own measurement surfaces, reported plainly rather than reconciled.**
`03-09-SUMMARY.md`'s table column is literally "filled hist bins" over a 20-bin histogram -- and
this plan's independent 20-bin histogram reproduces its 4/20 and 3/20 counts for seeds 14/15
exactly. But `05-RESEARCH.md`'s restatement of that finding as "two of the three fields
piecewise-constant (only 3-4 distinct values, one per surviving chart)" is a different, stronger
claim -- a literal count of distinct raw `H_norm` values -- and it does not hold at any precision
this plan checked. Seed 20260814 has **5,301** exact distinct `H_norm` values among 10,000 points
(16 at 6-decimal rounding, 5,234 at 12-decimal rounding), not 4. Seed 20260815 has **9,852** exact
distinct values (59 at 6dp, 9,827 at 12dp), not 3. The fields are **not literally
piecewise-constant** -- they carry substantial continuous variation, heavily concentrated into a
narrow dynamic range that clusters into ~`n_charts_used` dense bands when viewed through a
coarse (20-bin) histogram. The "collapsed metric spectrum" (`log10 det(g) ~ -165.6`, i.e.
`det(g) ~ 10^-166`) that `03-09` and `05-RESEARCH.md` both cite is real and confirmed here
(`-165.569` and `-165.650` median), but it does not make the field a step function -- it makes
the field's *dynamic range* small relative to its own scale while still varying point to point.
This changes what "piecewise-constant" means for the pooling discussion at `05-03`: seeds 14/15
are dominated by a **narrow-range, chart-clustered but continuous** field, not by a true 3-4-value
step function.

### D5-05: the three seeds do not agree, on either axis

| pair | rank axis (Spearman `H_norm`) | direction axis (median cosine of unit `H_vec`) | fraction anti-aligned |
|---|---|---|---|
| 20260813 vs 20260814 | rho = -0.1402 (p = 4.8e-45) | +0.0039 | 46.07% |
| 20260813 vs 20260815 | rho = +0.2019 (p = 1.8e-92) | +0.0014 | 48.09% |
| 20260814 vs 20260815 | rho = -0.2725 (p = 8.9e-170) | +0.0007 | 46.39% |

**In the phase's own words: the three seeds' curvature fields do not agree, on the rank axis or
the direction axis.** The three pairwise Spearman values are weak in magnitude and inconsistent
in sign (-0.14, +0.20, -0.27) -- there is no shared ordering across all three seeds; whichever
pair is compared, the "agreement" is either near-zero or actively negative. The direction axis is
even more stark: median cosine between any two seeds' unit `H_vec` fields sits at essentially
zero (0.0007 to 0.0039, against a maximum possible magnitude of 1.0), and 46-48% of points carry
an anti-aligned curvature vector between any pair of seeds -- statistically indistinguishable from
two independent random directions in a high-codimension space. This is consistent with Phase
03.1's finding that the curvature ordering is not seed-consistent, and it independently confirms
`spike-findings-effdim`'s measured `d=20` anti-alignment fraction (52-75% against the analytic
saddle control) in a different comparison (inter-seed here, vs seed-vs-truth there) landing in
the same qualitative place: near-chance direction agreement.

**This measured spread -- not a recommendation -- is the evidence `05-03`'s one-way pooling
checkpoint is decided on**, per this plan's own must-have. It is reported here as measured
numbers for the developer to weigh at that checkpoint, not interpreted into a pooling
recommendation by this plan.

### The r/R disclosure

`r_over_R` is recorded as `null` in `05_inter_seed_diagnostics.json` with reason `"not defined
for an autodiff decoder-side estimator: chart_curvature_field has no neighbourhood and no k"`.
`chart_curvature_field` is an autodiff path through the decoder with no k-nearest-neighbour
step, so the spike-findings `r/R` disclosure requirement is answered as a visible non-application,
not silently omitted.

## Decisions Made

- **Distinct-value counts reported at three precisions**, not one, because 03-09's own "filled
  hist bins" measurement and RESEARCH's "3-4 distinct values" restatement are two different
  quantities that turned out to disagree once measured directly -- reporting only one precision
  would have silently picked a side in that discrepancy rather than surfacing it.
- **The direction axis is computed inside the same loop iteration as the Spearman value**, one
  function call immediately after the other, so the spike-findings disclosure rule is
  structurally true of the code (adjacent print lines, same JSON entry pair) rather than merely
  a convention a future editor could accidentally break.

## Deviations from Plan

### Auto-fixed Issues

None -- no bugs, missing functionality, or blocking issues required a code fix during execution.

### Process deviation (not a Rule 1-4 auto-fix; recorded for the record)

**1. An idle gap occurred inside the three-seed extraction sequence, and Task 1 produced no
git-committable diff.**
- **What happened:** Seed 20260814's extraction finished at 13:56 local. Seed 20260815's
  extraction was not launched until ~14:39 local (per the orchestrator's mid-task correction
  message), an idle gap of roughly 38 minutes with no compute running, before this session
  resumed and launched the third seed. All three seeds' artifacts and the diagnostics JSON are
  now present, correct, and verified; the gap cost wall-clock time only, not correctness.
- **Also recorded here:** Task 1 (the three-seed extraction) has no committable source diff --
  its entire deliverable is three `notebooks/.cache/05_curvature_field_seed*.npz` artifacts,
  which are gitignored per `CLAUDE.md`'s "Milestone artifacts live in the gitignored
  notebooks/.cache/" convention. Per the task-commit protocol's step 1 ("check modified files"),
  there is nothing to stage for Task 1 alone; the plan's one committable diff
  (`_direction_axis_report` / `run_inter_seed_diagnostics`) is Task 2's, committed at `ed8f3f7`.
- **Impact:** None on the plan's deliverables or verification. Documented for the record per the
  coordinator's own instruction to address it explicitly rather than silently smoothing over it.

---

**Total deviations:** 0 auto-fixed; 1 process note (idle gap + Task 1's artifact-only, no-diff
nature), both recorded above.
**Impact on plan:** None on correctness or scope. All acceptance criteria met.

## Issues Encountered

None beyond the process deviation documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Three independently cached per-seed curvature fields and the inter-seed diagnostics JSON are
  ready for `05-03`'s blocking pooling-method checkpoint. The measured spread above (Spearman
  sign-inconsistent across pairs, direction axis near-chance) is the evidence that checkpoint
  needs in front of it.
- `notebooks/pu_manifold/linear_probe.py` remains unchanged from 05-01 -- still unblocked for the
  05-04 freeze once 05-03/05-05 land.
- The distinct-value-count discrepancy above (measured 5,301/9,852 vs `03-09`'s cited "3-4") is
  worth carrying into `05-03`'s pooling discussion: seeds 14/15 are narrow-range and
  chart-clustered, not literal step functions, which may affect which pooling normalization
  (`per_seed_median_divide` vs `per_seed_rank_uniform`) behaves best on them.
- No blockers.

---
*Phase: 05-curvature-conditioned-linear-decodability*
*Completed: 2026-08-24*

## Self-Check: PASSED

All three per-seed npz artifacts and the inter-seed diagnostics JSON found on disk; both commit
hashes (`ed8f3f7`, `2229357`) found in `git log`.
