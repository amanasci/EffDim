---
phase: 05-curvature-conditioned-linear-decodability
plan: 03
subsystem: geometry
tags: [curvature, linear-probe, seed-pooling, bucketing, spearman, density-confound, npz-cache]

# Dependency graph
requires:
  - phase: 05-curvature-conditioned-linear-decodability (05-01)
    provides: linear_probe.py's pre-registration block/assert_preregistered idiom,
      apply_verdict_rule, bucket_edges_from_field/assign_buckets/bucket_by_field, the runner's
      --mode field/pool/bucketed skeleton
  - phase: 05-curvature-conditioned-linear-decodability (05-02)
    provides: three sealed per-seed curvature field artifacts
      (05_curvature_field_seed2026081{3,4,5}.npz), the measured inter-seed disagreement
      (05_inter_seed_diagnostics.json) that grounds the ratified no-pooling decision
  - phase: 05-curvature-conditioned-linear-decodability (05-03 Task 1 checkpoint)
    provides: 05-03-DECISION.md -- the ratified, one-way refusal to pool the three seeds
provides:
  - notebooks/pu_manifold/linear_probe.py restructured for a three-verdict, per-seed design --
    POOLING_METHOD/BUCKET_EDGES removed; SEED_HANDLING_RULE, BUCKET_EDGES_PER_SEED,
    SEED_VERDICT_COMBINATION_RULE, PHASE_VERDICT_VALUES added (all still unset);
    combine_seed_verdicts mapping three per-seed verdicts to one phase outcome
  - notebooks/.cache/05_curvature_buckets_seed2026081{3,4,5}.npz -- three independent per-seed
    bucket artifacts, each carrying that seed's own bucket_labels, bucket_edges and
    effective_distinct_levels measured at relative precision
  - notebooks/.cache/05_density_diagnostics.json -- D5-13's per-seed density confound and
    D5-05's pooled-half disposition
  - --mode perseed (new), --mode pool (refuses by name), --mode bucketed's guard now checking
    three per-seed artifacts, in notebooks/diagnostics/curvature_probe_decodability_run.py
affects: [05-04, 05-05, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Structural removal over unused retention for a rejected design: POOLING_METHOD and
      BUCKET_EDGES are deleted from linear_probe.py's constants block (not merely left unset
      forever), so a later executor reading only 05-CONTEXT.md D5-04 finds no constant to
      assign -- the ratified refusal to pool is enforced by the module's shape, not only by a
      runtime check"
    - "assert_preregistered self-verifies its own PHASE_VERDICT_VALUES invariant by calling
      combine_seed_verdicts on three canonical seed-verdict trios and checking every returned
      phase_verdict is a member of PHASE_VERDICT_VALUES -- catches a future edit that changes
      combine_seed_verdicts' hardcoded strings without updating the frozen tuple"
    - "Relative-precision level counting (_effective_distinct_levels) replaces absolute
      rounding for measuring a field's effective distinct values -- the walk-sorted-once,
      open-a-new-level-when-gap-exceeds-tolerance-times-representative algorithm is immune to
      last-ULP float noise that an np.round(..., 6) count is not"

key-files:
  created: []
  modified:
    - notebooks/pu_manifold/linear_probe.py
    - notebooks/pu_manifold/tests/test_linear_probe.py
    - notebooks/diagnostics/curvature_probe_decodability_run.py

key-decisions:
  - "Both new assert_preregistered checks on VERDICT_RULE and SEED_VERDICT_COMBINATION_RULE
    require the literal substring \"SPLIT ACROSS SEEDS\", not merely a non-empty string --
    guards against a frozen rule text that omits the split branch, matching D5-09's ratified
    'SPLIT ACROSS SEEDS is a complete terminal non-supportive outcome' framing"
  - "SEED_STEMS is validated in assert_preregistered BEFORE BUCKET_EDGES_PER_SEED (reordered
    from the original POOLING_METHOD/BUCKET_EDGES/SEED_STEMS sequence), because the new
    per-seed shape check needs len(SEED_STEMS) to compute its expected tuple length --
    necessary reordering, every individual check's body is otherwise unchanged"
  - "run_pool_mode and the --pooling-method tripwire share one message constant
    (POOLING_REFUSAL_MESSAGE) naming 05-03-DECISION.md, so --mode pool and
    --pooling-method <anything> under any mode raise byte-identical text -- one source of
    truth for the refusal rather than two independently-worded RuntimeErrors"

requirements-completed: [D5-03, D5-04, D5-05, D5-07, D5-09, D5-10, D5-13]

coverage:
  - id: D1
    description: "linear_probe.py's pre-registration block restructured for three per-seed
      verdicts: POOLING_METHOD and BUCKET_EDGES structurally removed; SEED_HANDLING_RULE,
      BUCKET_EDGES_PER_SEED, SEED_VERDICT_COMBINATION_RULE, PHASE_VERDICT_VALUES added, all
      still unset; combine_seed_verdicts added; assert_preregistered still raises with every
      constant unset"
    requirement: D5-09
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_linear_probe.py -q (389 passed, 1 skipped, 1 xfailed)"
        status: pass
      - kind: integration
        ref: "python -c assert script over linear_probe module attributes and combine_seed_verdicts' eight combinations -- see plan 05-03 Task 1 <verify>"
        status: pass
    human_judgment: false
  - id: D2
    description: "Three independent per-seed bucket artifacts exist, each cut over that seed's
      own 10,000-point field via linear_probe.bucket_by_field(H_norm, n_buckets=3); no pooled
      artifact exists anywhere; effective distinct levels at relative precision measure 4 for
      seed 20260814 and 3 for seed 20260815 at rel 1e-9/1e-6/1e-3, correcting
      05-02-SUMMARY.md's 5,301/9,852 exact-float claim"
    requirement: D5-07
    verification:
      - kind: integration
        ref: "python -c assert script over the three 05_curvature_buckets_seed*.npz artifacts (shape, label set, edge ordering, seed_stem, effective_distinct_levels) -- see plan 05-03 Task 2 <verify>"
        status: pass
    human_judgment: false
  - id: D3
    description: "--mode pool refuses by name (RuntimeError naming 05-03-DECISION.md) instead
      of stubbing NotImplementedError; --pooling-method is a tripwire under any mode;
      --mode bucketed's guard checks all three per-seed bucket artifacts and still runs
      assert_preregistered first"
    requirement: D5-10
    verification:
      - kind: integration
        ref: "subprocess invocations of --mode pool, --mode perseed --pooling-method ..., --mode bucketed -- see plan 05-03 Task 2 <verify> and Task 2 acceptance criteria"
        status: pass
    human_judgment: false
  - id: D4
    description: "D5-13's density confound re-measured per seed with Phase 4's own
      local_density_weights estimator at K_DENSITY=30, FIELD_D=20, recorded in
      05_density_diagnostics.json with Phase 4's -0.0273/+0.8208 references beside it; D5-05's
      pooled-versus-seed half dispositioned (has no referent, not computed against a
      substitute) rather than dropped"
    requirement: D5-13
    verification:
      - kind: integration
        ref: "python -c assert script over 05_density_diagnostics.json's schema -- see plan 05-03 Task 3 <verify>"
        status: pass
      - kind: unit
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q (389 passed, 1 skipped, 1 xfailed)"
        status: pass
    human_judgment: false

duration: ~15 min (commit-to-commit span; excludes plan/context reading time)
completed: 2026-08-24
status: complete
---

# Phase 5 Plan 3: Per-Seed Bucketing, Verdict Combination, and Density Confound Summary

**The ratified one-way "do not pool" decision is now structural code, not only a decision document: `POOLING_METHOD`/`BUCKET_EDGES` are removed from `linear_probe.py` (not merely left unset), three independent per-seed bucket artifacts exist with corrected relative-precision level counts (4 and 3, not 05-02's 5,301/9,852), `--mode pool` refuses by name, and D5-13's density confound is re-measured per seed -- seed 20260813's confound (rho=-0.4875) is far stronger than Phase 4's -0.0273 point-cloud reference.**

## Performance

- **Duration:** ~15 min (commit span 17:47:29 -- 17:59:49 local); reading the plan, decision record and five source files preceded the first commit
- **Started:** 2026-08-24T21:47:29Z (first task commit)
- **Completed:** 2026-08-24T21:59:49Z (third task commit)
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `linear_probe.py`'s pre-registration block restructured: `POOLING_METHOD` and `BUCKET_EDGES`
  are structurally removed (not merely unused); `SEED_HANDLING_RULE`, `BUCKET_EDGES_PER_SEED`,
  `SEED_VERDICT_COMBINATION_RULE`, `PHASE_VERDICT_VALUES` added, all still unset. The block
  carries exactly 31 names.
- `combine_seed_verdicts` added: maps three per-seed terminal verdicts to one of
  `HOLDS IN ALL THREE SEEDS` / `SPLIT ACROSS SEEDS` / `NO DETECTABLE RELATIONSHIP IN ANY SEED`;
  raises `RuntimeError` on an empty rule (pre-freeze guard) and `ValueError` on a non-three-seed
  input or an unrecognized per-seed verdict string. All eight three-seed combinations verified
  to land on exactly one of the three terminal strings.
- `assert_preregistered` amended: `SEED_HANDLING_RULE` checked by exact equality (not
  non-empty), `BUCKET_EDGES_PER_SEED` checked as `len(SEED_STEMS)` per-seed tuples (not one
  flat tuple), `SEED_VERDICT_COMBINATION_RULE` and `PHASE_VERDICT_VALUES` added as new checks,
  `VERDICT_RULE` now also requires the literal `SPLIT ACROSS SEEDS`. Still raises `RuntimeError`
  with every constant unset.
- `--mode perseed` added to the runner: three independent bucketings (`n_buckets=3`), each cut
  over that seed's own 10,000-point field, written to
  `notebooks/.cache/05_curvature_buckets_seed{seed}.npz` with a cfg manifest naming the seed,
  source field stem, bucket rule, subsample file, curvature convention and
  `no_pooling_per_seed_verdicts`. `linear_probe.pool_seed_fields` is called nowhere.
- `_effective_distinct_levels` added: relative-precision level counting, measured at rel
  `1e-9`/`1e-6`/`1e-3`. Confirms seed 20260814 = **4** levels and seed 20260815 = **3** levels at
  every tolerance -- matching `05-03-DECISION.md`'s ratified correction exactly.
- `--mode pool` no longer stubs `NotImplementedError` -- it raises `RuntimeError` naming
  `05-03-DECISION.md` and directing to `--mode perseed`. `--pooling-method` is a tripwire: any
  mode raises the identical error if it is supplied.
- `--mode bucketed`'s artifact guard now checks all three per-seed bucket paths (was one pooled
  path); `assert_preregistered()` still runs first (D5-10 guard order unchanged).
- `run_density_diagnostics` added, called at the end of `--mode perseed`: re-measures
  `spearman(density, ||H||)` per seed with Phase 4's own `local_density_weights` estimator at
  Phase 4's own `K_DENSITY=30`/`FIELD_D=20`, writes `notebooks/.cache/05_density_diagnostics.json`
  with Phase 4's `-0.0273`/`+0.8208` references quoted beside it, the null `direction_axis` with
  its scalar-operand reason, and `pooled_field_disposition` discharging D5-05's pooled-half.
- No pooled field, no pooled bucket edges, no pre-registered constant set, no PU probe number
  anywhere in the repository at the end of this plan.
- The three sealed `05_curvature_field_seed*.npz` artifacts and their `.meta.json` sidecars are
  byte-for-byte untouched (verified by unchanged mtimes before and after all three tasks).
- Full test suite green throughout: `389 passed, 1 skipped, 1 xfailed` (4 new tests over 05-02's
  385).

## Task Commits

Each task was committed atomically:

1. **Task 1: Restructure the pre-registration block for three verdicts** - `94735b7` (feat)
2. **Task 2: Three per-seed bucketings, corrected level counts, refusal to pool** - `525a137` (feat)
3. **Task 3: Per-seed density confound diagnostics; dispose of D5-05's pooled half** - `75ea691` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `notebooks/pu_manifold/linear_probe.py` - Constants block restructured (31 names, all
  unset); `assert_preregistered` amended with the new/reordered checks;
  `combine_seed_verdicts` added; docstring paragraph (b) rewritten with the ratified decision
  and measured evidence, plus the 5,301/9,852 -> 4/3 correction
- `notebooks/pu_manifold/tests/test_linear_probe.py` - `test_assert_preregistered_raises_when_absent`
  updated to the renamed constants; four new tests added
  (`test_combine_seed_verdicts_known_answer`, `test_combine_seed_verdicts_requires_three_seeds`,
  `test_combine_seed_verdicts_raises_on_empty_rule`,
  `test_assert_preregistered_rejects_flat_bucket_edges`)
- `notebooks/diagnostics/curvature_probe_decodability_run.py` - `run_pool_mode` refuses by
  name; `_effective_distinct_levels`, `run_perseed_mode`, `run_density_diagnostics` added;
  `run_bucketed_mode`'s guard checks three per-seed paths; `--mode perseed` and `--bucket-stem`
  added to the arg parser; `--pooling-method` tripwire wired into `main()`
- `notebooks/.cache/05_curvature_buckets_seed20260813.npz` (+`.meta.json`) - gitignored
- `notebooks/.cache/05_curvature_buckets_seed20260814.npz` (+`.meta.json`) - gitignored
- `notebooks/.cache/05_curvature_buckets_seed20260815.npz` (+`.meta.json`) - gitignored
- `notebooks/.cache/05_density_diagnostics.json` - gitignored

## Findings

### Corrected effective level counts (relative precision, replacing 05-02's exact-float claim)

| seed | rel 1e-9 | rel 1e-6 | rel 1e-3 | bucket edges (full float64 repr) |
|---|---|---|---|---|
| 20260813 | 10,000 | 9,904 | 1,173 | (1225.4263017421292, 1538.3597929379368) |
| 20260814 | **4** | **4** | **4** | (49062.2351870738, 66977.54374981482) |
| 20260815 | **3** | **3** | **3** | (51694.86079512253, 75252.52609688243) |

Seeds 20260814 and 20260815 match `05-03-DECISION.md`'s ratified correction exactly, at every
one of the three tolerances. `05-02-SUMMARY.md`'s 5,301 / 9,852 "exact distinct" counts were
last-ULP float noise, not effective structure; `05-RESEARCH.md` Pitfall 2 and
`03-09-SUMMARY.md`'s original "3-4 distinct values" measurement were correct.

### An unplanned discovery: seed 20260814 carries one exact-duplicate block of 2,102 points

Investigating a bucket-count imbalance surfaced a fact worth recording plainly. Seed 20260814's
field has 5,301 exact-float64-distinct values among 10,000 points (05-02's own count, confirmed
here) -- but they are not evenly spread: **2,102 of the 10,000 points carry the field's maximum
value, `66977.54374981482`, to the exact bit.** `bucket_edges_from_field`'s equal-frequency
`array_split` puts that value at the second bucket edge (assumed near-3334/3333/3333), but
`assign_buckets`' documented tie rule (D5-07: a value equal to an edge lands in the HIGHER
bucket, and two points with exactly-equal `H_norm` always share a label) sends the ENTIRE
2,102-point block to the top bucket. Realized full-field counts for seed 20260814 are
**[3334, 2956, 3710]** -- a spread of 754, not the "near-exactly 3334/3333/3333 by construction"
this plan's own action text assumed. Seeds 20260813 and 20260815 both land at
`[3334, 3333, 3333]` (spread of 1) as expected; only 20260814 carries this large exact-tie
block. Verified: the tie rule is applied correctly and consistently -- every point sharing that
exact value carries the same bucket label, and the three labels sum exactly to 10,000. This is
not a code bug; it is not fixable without either violating the already-locked D5-07 tie rule or
recomputing the sealed field artifact (both prohibited). It is additional, independent evidence
of how degenerate seed 20260814's field is at raw float64 precision -- consistent with, and
strengthening, `05-03-DECISION.md`'s "collapsed metric" characterization. `05-04`/`05-05` should
be aware that this seed's realized test-split bucket counts (after the 70/30 split) may be
similarly unbalanced; `size_matched_check`'s realized-count logic (D5-08) already handles
unequal realized bucket sizes correctly by design, so no code change is implied.

### D5-13: the density confound, measured per seed, disagrees in both magnitude and sign

| seed | spearman(inverse_density_weight, ‖H‖) | spearman(relative_density, ‖H‖) | n |
|---|---|---|---|
| 20260813 | +0.4875 (p≈0) | -0.4875 | 10,000 |
| 20260814 | -0.1986 (p=1.7e-89) | +0.1986 | 10,000 |
| 20260815 | +0.0556 (p=2.6e-8) | -0.0556 | 10,000 |

Phase 4's point-cloud reference is `spearman(density, centroid_mean_curvature) = -0.0273`
(near-nil). **D5-13's expectation -- that the decoder-side confound would be weaker on
magnitude than Phase 4's `+0.8208` direction confound -- holds for all three seeds** (all three
magnitudes are well under 0.82), but seed 20260813's confound (`|rho| = 0.4875`) is **far
stronger than Phase 4's own point-cloud reference of `-0.0273`**, not weaker. The three seeds
disagree with each other on the confound's sign, exactly as they disagree on curvature rank
(`05-02`) and direction (`05-02`). Per D5-13, this is a disclosure requirement, not a gate: it
gates nothing here, but `05-04`/`05-06` should carry seed 20260813's stronger-than-Phase-4
confound forward as a caveat on any decodability result from that seed specifically.

### D5-05's pooled-versus-seed half: dispositioned, not dropped

`pooled_field_disposition` in `05_density_diagnostics.json` states, verbatim in the artifact:
D5-05 asks for the Spearman between each seed and the pooled field; no pooled field exists
because seed pooling was ratified NOT DONE at the `05-03` Task 1 checkpoint
(`05-03-DECISION.md`, superseding D5-04); the statistic therefore has no referent and was not
computed against a substitute; D5-05's first half (pairwise inter-seed Spearman with its
direction axis) was measured at `05-02` and is recorded in `05_inter_seed_diagnostics.json`.

## Decisions Made

- **`SEED_HANDLING_RULE` checked by exact equality**, not a non-empty-string check, so a future
  edit assigning it a pooling-method name fails the guard rather than passing it -- the guard
  proves the refusal, it does not merely prove *a* string was set.
- **`SEED_STEMS` validated before `BUCKET_EDGES_PER_SEED`** in `assert_preregistered` (reordered
  from the original constant sequence), because the new per-seed shape check needs
  `len(SEED_STEMS)` to compute its expected tuple length. Every individual check's body is
  otherwise byte-identical to the plan's specification.
- **`assert_preregistered` self-verifies `PHASE_VERDICT_VALUES`** by calling
  `combine_seed_verdicts` on three canonical seed-verdict trios (all-HOLDS, all-NO, one-HOLDS)
  under the frozen `SEED_VERDICT_COMBINATION_RULE` and asserting every returned `phase_verdict`
  is a member of `PHASE_VERDICT_VALUES` -- a structural check, not merely a presence check, per
  this module's established idiom of self-consistency guards.
- **One shared refusal-message constant** (`POOLING_REFUSAL_MESSAGE`) backs both `run_pool_mode`
  and the `--pooling-method` tripwire, so the two refusal paths cannot drift to different
  wording over time.

## Deviations from Plan

### Auto-fixed Issues

None -- no bugs, missing functionality, or blocking issues required a code fix during execution.

### Measured findings recorded, not smoothed over (not Rule 1-4 auto-fixes)

**1. Seed 20260814's full-field bucket counts do not satisfy the "near-exactly balanced" assumption in this plan's own action text.**
- **Found during:** Task 2, verifying the automated `<verify>` block's
  `counts.max() - counts.min() <= 1` assertion.
- **What was found:** A genuine 2,102-point exact-duplicate block at the field's maximum value
  (see Findings above) causes realized counts of `[3334, 2956, 3710]` (spread 754) for seed
  20260814, against `[3334, 3333, 3333]` (spread 1) for the other two seeds.
- **Why not auto-fixed:** The behavior is exactly what D5-07's already-locked tie rule requires
  (all points sharing an exact `H_norm` value share a bucket label, and a tied value lands in
  the higher bucket) -- correct code, real data. "Fixing" it would mean either violating the
  tie rule (a Rule 4 architectural change to a decision locked since `05-01`, out of scope for
  this plan) or recomputing the sealed `05_curvature_field_seed20260814.npz` artifact
  (prohibited by this plan's own must-haves and CLAUDE.md's additive-only rule). Verified
  instead: tie-consistency holds exactly (every exact-duplicate value shares one label), the
  three labels take exactly `{0, 1, 2}`, and the three counts sum exactly to 10,000 -- the
  properties that actually matter for `05-05`'s downstream `size_matched_check`, which already
  handles unequal realized bucket counts by design (D5-08).
- **Files affected:** None (no code change) -- `notebooks/.cache/05_curvature_buckets_seed20260814.npz` records the realized counts as measured.
- **Verification:** Direct inspection confirmed tie-consistency across all 5,301 distinct exact
  values in the field; `counts.sum() == 10000` and label set `== {0, 1, 2}` both hold.

---

**Total deviations:** 0 auto-fixed; 1 measured finding recorded plainly (not a code fix), documented above.
**Impact on plan:** None on correctness, scope, or the ratified no-pooling decision. All acceptance criteria requiring code behavior are met; the one criterion describing an unverified assumption about the data ("near-exactly balanced by construction") is corrected against direct measurement, in keeping with this phase's established practice of reporting what is measured rather than what was assumed.

## Issues Encountered

None beyond the measured finding documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `linear_probe.py`'s pre-registration block is restructured and ready for `05-04`'s freeze:
  every one of its 31 constants is still unset, `assert_preregistered()` still raises, and the
  file carries exactly two commits in its history (`05-01`'s creation, `94735b7`'s restructure)
  -- `05-04`'s freeze will be the third and final commit.
- Three independent per-seed bucket artifacts and the per-seed density diagnostics are ready
  for `05-05`'s three probe runs. `05-05` should read seed 20260814's realized (unbalanced)
  test-split bucket counts directly from `size_matched_check` rather than assuming
  near-equality.
- `combine_seed_verdicts` is implemented, tested against all eight combinations, and dead until
  the freeze (raises `RuntimeError` on an empty rule) -- ready for `05-05`/`05-06` to call once
  three per-seed verdicts exist.
- No blockers. CLAUDE.md's Swiss-roll sanity-check rule does not trigger for this plan: no new
  manifold-learning or representation-learning model was introduced -- this plan renames
  constants, adds a verdict-combining function, and buckets three already-extracted fields. The
  curvature estimator these fields come from is already covered by
  `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`.

---
*Phase: 05-curvature-conditioned-linear-decodability*
*Completed: 2026-08-24*

## Self-Check: PASSED

All modified source files, all four new/modified `.cache/` artifacts, and this SUMMARY.md
found on disk; all three task commit hashes (`94735b7`, `525a137`, `75ea691`) found in
`git log`.
