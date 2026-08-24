---
phase: 05-curvature-conditioned-linear-decodability
plan: 05
subsystem: geometry
tags: [linear-probe, ridge, ridgecv, bucketing, seed-verdict, pre-registration-application, conditioning]

# Dependency graph
requires:
  - phase: 05-curvature-conditioned-linear-decodability (05-04)
    provides: notebooks/pu_manifold/linear_probe.py frozen at all 31 constants (commit 32dabe3),
      05-PREREGISTRATION.md, VERDICT_RULE and SEED_VERDICT_COMBINATION_RULE committed in full
  - phase: 05-curvature-conditioned-linear-decodability (05-03)
    provides: three per-seed bucket artifacts (05_curvature_buckets_seed2026081{3,4,5}.npz),
      combine_seed_verdicts
provides:
  - notebooks/.cache/05_curvature_probe_decodability.jsonl -- the phase's headline result. One
    global ridge fit, three per-seed bucketed comparisons, three per-seed verdicts
    (HOLDS/HOLDS/NO DETECTABLE RELATIONSHIP), phase verdict SPLIT ACROSS SEEDS (n_holds=2 of 3),
    plus one probe_conditioning row checking RESEARCH A2's ridge justification
  - notebooks/diagnostics/curvature_probe_decodability_run.py's run_bucketed_mode implemented
    end to end: _fit_and_evaluate (the file's one fit_probe call site, shared with selfcheck()),
    _score_one_seed, _load_bucket_artifact, _conditioning_diagnostics
affects: [05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One shared fit-and-score helper (_fit_and_evaluate) used by both the synthetic
      selfcheck() fixture and the real run_bucketed_mode path, so a file-wide grep for the
      literal fit_probe call site returns exactly one line even though two callers exercise it
      -- the acceptance-criteria-driven consolidation is documented as a deviation below"
    - "Conditioning diagnostic computed from the training split already in scope inside the
      real-run branch (no second fit, no new mode/flag), appended to the same JSONL record
      after the headline rows -- since the pipeline is fully deterministic under its frozen
      seeds, re-running it after adding the diagnostic reproduced byte-identical headline
      numbers, so no row was duplicated and none was silently rewritten"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/curvature_probe_decodability_run.py

key-decisions:
  - "Consolidated the file's fit_probe call site through a new _fit_and_evaluate helper,
    shared with selfcheck() (which already called linear_probe.fit_probe directly before this
    plan). Task 1's own acceptance criteria require a file-wide grep for the literal string
    'linear_probe.fit_probe' to return exactly 1; selfcheck()'s pre-existing call meant a
    second, independent call added in run_bucketed_mode would have made that grep return 2.
    selfcheck()'s behavior is byte-identical, only its plumbing moved into the shared helper."
  - "The probe_conditioning row is computed inline inside run_bucketed_mode's existing
    real-run branch, not behind a new CLI mode or flag -- the plan's own artifacts section
    pins the runner's flag surface to the four existing --mode values with no addition, and
    the row needs the training split (X_train) already in scope from the one fit, which a
    separate standalone invocation would either have to re-derive (fine, since
    train_test_split_indices is deterministic and cheap) or duplicate. Folding it into the
    same branch avoids a second driver path for what is definitionally one training split's
    diagnostic."

requirements-completed: [D5-01, D5-02, D5-05, D5-07, D5-08, D5-09, D5-10, D5-13]

coverage:
  - id: D1
    description: "run_bucketed_mode fits ONE global ridge map via the sole fit_probe call
      site in the file (_fit_and_evaluate), reads every frozen constant off linear_probe with
      no CLI override, and buckets the held-out residuals three times via _score_one_seed --
      one per seed's frozen BUCKET_EDGES_PER_SEED entry, never refit per bucket or per seed"
    requirement: D5-02
    verification:
      - kind: integration
        ref: "grep -c 'linear_probe.fit_probe' returns 1; --mode bucketed --smoke exits 0,
          writes nothing, prints three per-seed verdicts and one phase verdict -- see plan
          05-05 Task 1 <verify>"
        status: pass
      - kind: unit
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q (390 passed, 1 skipped)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The full real run produced the phase's headline numbers: n_train=7000,
      n_test=3000, selected_alpha=0.1, r2_overall=0.643931, three per-seed verdicts
      (20260813=HOLDS, 20260814=NO DETECTABLE RELATIONSHIP, 20260815=HOLDS), phase verdict
      SPLIT ACROSS SEEDS (n_holds=2 of 3) -- all produced only by
      linear_probe.apply_verdict_rule / combine_seed_verdicts, never a runner-side comparison"
    requirement: D5-09
    verification:
      - kind: integration
        ref: "python -c assert script over 05_curvature_probe_decodability.jsonl's row counts,
          shared selected_alpha/r2_overall across all three seed rows, realized-count sums,
          size-match bounds, and bucket_edges equality against BUCKET_EDGES_PER_SEED -- see
          plan 05-05 Task 2 <verify>"
        status: pass
    human_judgment: false
  - id: D3
    description: "Per-seed realized TEST-SPLIT bucket counts and size-matched checks use each
      seed's own realized counts, never a full-field count and never another seed's count
      (RESEARCH Pitfall 4 / the artifact that undercut Phase 4's verdict); seed 20260814's
      realized counts (992/873/1135) are visibly unbalanced against the other two seeds'
      near-equal splits, reported plainly rather than smoothed over"
    requirement: D5-08
    verification:
      - kind: integration
        ref: "per-seed size_match_n <= min(realized_bucket_counts) asserted for all three
          seeds; realized counts sum to n_test=3000 for all three seeds -- see plan 05-05
          Task 2 <verify>"
        status: pass
    human_judgment: false
  - id: D4
    description: "RESEARCH A2's ridge justification (severe rank deficiency at ~18-25
      effective dimensions) checked against the training split's own measured singular
      spectrum via a new probe_conditioning row: condition_number~9.98e4,
      effective_rank_1pct=531 of 768, cumvar_first_20=0.810, cumvar_first_25=0.835,
      selected_alpha=0.1, not at the RIDGE_ALPHA_GRID boundary. The claim is NOT confirmed --
      recorded plainly, not silently accepted"
    requirement: D5-09
    verification:
      - kind: integration
        ref: "python -c assert script over the probe_conditioning row's schema and the
          unchanged prior row counts (1 overall, 3 seed, 9 bucket) -- see plan 05-05 Task 3
          <verify>"
        status: pass
      - kind: unit
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q (390 passed, 1 skipped)"
        status: pass
    human_judgment: false

duration: ~7 min (task-commit span, 19:27:15 -- 19:33:44 local; excludes plan/context reading
  and the ~2.5 min pytest full-suite runs between commits)
completed: 2026-08-24
status: complete
---

# Phase 5 Plan 5: The Bucketed Probe Run -- Headline Numbers and Verdicts Summary

**One global ridge map (`R^2=0.6439`, `alpha=0.1`), three per-seed bucketed comparisons under the frozen `VERDICT_RULE`, and a phase verdict of `SPLIT ACROSS SEEDS` (2 of 3 seeds HOLDS) -- applied mechanically, with RESEARCH A2's ridge justification measured and found NOT confirmed.**

## Performance

- **Duration:** ~7 min (task-commit span 19:27:15 -- 19:33:44 local, 2026-08-24); two full test-suite runs (~2.5 min each) ran between commits and are excluded from the span
- **Started:** 2026-08-24T23:27:15Z (Task 1 commit)
- **Completed:** 2026-08-24T23:33:44Z (Task 3 commit)
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- **Task 1:** `run_bucketed_mode` implemented end to end. Reads all 17 constants it needs off
  the frozen `linear_probe` module with no CLI override, fits ONE global ridge map via a new
  `_fit_and_evaluate` helper (this file's sole `fit_probe` call site, shared with `selfcheck()`
  so a whole-file grep for the literal string still returns exactly one line), and scores the
  held-out residuals three times via `_score_one_seed`, once per seed's frozen
  `BUCKET_EDGES_PER_SEED` entry. Per-seed verdicts come only from `apply_verdict_rule`, the
  phase verdict only from `combine_seed_verdicts`. `--mode bucketed --smoke` proves the
  complete path end to end on a 64-row slice, writes nothing, and `--pooling-method` still
  refuses naming `05-03-DECISION.md`.
- **Task 2:** The real run, once, on the full 10,000 points. `n_train=7000`, `n_test=3000`,
  `selected_alpha=0.1` (from `RIDGE_ALPHA_GRID = (1e-2..1e4)`), `r2_overall=0.643931`, shared
  identically by all three per-seed rows. Wrote
  `notebooks/.cache/05_curvature_probe_decodability.jsonl` with 9 `probe_bucket` rows, 3
  `probe_seed` rows, 1 `probe_overall` row.
- **Task 3:** `_conditioning_diagnostics` added, called once from the same real-run branch
  (no second fit) on the already-in-scope `X_train`. Measured the training split's own
  singular spectrum against RESEARCH A2's stated reason for ridge and found it **not
  confirmed** -- see Findings.
- Full suite green throughout: `390 passed, 1 skipped` (unchanged from 05-04's baseline; this
  plan added no new test file, per its own scope).

## The headline result

| Seed | Bucket edges | Realized test counts | Full-field counts | Verdict |
|---|---|---|---|---|
| 20260813 | (1225.426, 1538.360) | (1024, 987, 989) | (3334, 3333, 3333) | **HOLDS** |
| 20260814 | (49062.235, 66977.544) | (992, 873, 1135) | (3334, 2956, 3710) | **NO DETECTABLE RELATIONSHIP** |
| 20260815 | (51694.861, 75252.526) | (986, 1019, 995) | (3334, 3333, 3333) | **HOLDS** |

**Phase verdict: `SPLIT ACROSS SEEDS`** (`n_holds=2` of 3). Under `SEED_VERDICT_COMBINATION_RULE`,
this is a complete, terminal, non-supportive outcome -- reported exactly as `SPLIT ACROSS SEEDS`,
never upgraded to `HOLDS IN ALL THREE SEEDS` and never downgraded to
`NO DETECTABLE RELATIONSHIP IN ANY SEED`. Because all three seeds share one `TRAIN_FRACTION`/
`SPLIT_SEED` split, the three per-seed verdicts are NOT statistically independent -- they score
the same 3,000 held-out residuals under three different bucketings -- which isolates the
per-seed curvature field as the only thing that differs between them, and this must carry into
`05-FINDINGS.md` rather than be read as three independent trials.

**Per-seed detail (bucket 0 = lowest curvature, bucket 2 = highest):**

| Seed | Bucket | n | full-field n | mean residual | R² | 95% CI |
|---|---|---|---|---|---|---|
| 20260813 | 0 | 1024 | 3334 | 0.048285 | 0.6029 | [0.046401, 0.050535] |
| 20260813 | 1 | 987 | 3333 | 0.062024 | 0.6454 | [0.059345, 0.064487] |
| 20260813 | 2 | 989 | 3333 | 0.089612 | 0.6141 | [0.085701, 0.093362] |
| 20260814 | 0 | 992 | 3334 | 0.062839 | 0.6564 | [0.060013, 0.065761] |
| 20260814 | 1 | 873 | 2956 | 0.093686 | 0.5758 | [0.090008, 0.097461] |
| 20260814 | 2 | 1135 | 3710 | 0.048602 | 0.4433 | [0.046525, 0.050471] |
| 20260815 | 0 | 986 | 3334 | 0.056126 | 0.5398 | [0.053666, 0.058710] |
| 20260815 | 1 | 1019 | 3333 | 0.077089 | 0.6867 | [0.073572, 0.080588] |
| 20260815 | 2 | 995 | 3333 | 0.065722 | 0.4011 | [0.062984, 0.068434] |

**Continuous spearman(‖H‖, per-point residual) on the test split, per seed** (sensitivity only,
non-gating, direction axis recorded as an explicit null -- both operands are scalars):
20260813 `rho=+0.4239` (p=3.6e-131), 20260814 `rho=-0.1169` (p=1.3e-10), 20260815 `rho=+0.1384`
(p=2.6e-14). The sign of the continuous statistic agrees with the headline highest-vs-lowest
comparison for all three seeds (20260813 and 20260815 positive/HOLDS, 20260814
negative/NO DETECTABLE RELATIONSHIP), so the bucketed and continuous views are, as expected,
two views of the same underlying quantity here -- not a disagreement to explain away.

**Size-matched re-check** (D5-08, subsampled to each seed's own smallest realized test-split
bucket count): all three seeds' signs were stable across all 200 repeats
(`sign_stable=True`, `ci_disjoint_fraction=1.0` for every seed), so the size match never
overturned a headline sign -- 20260814's failure is on criterion (b) (`residual_higher_at_high_curvature=False`:
its highest bucket's mean residual, 0.048602, is actually the LOWEST of its three buckets, not
the highest), not on a size-imbalance artifact.

**Seed 20260814's realized bucket counts are the imbalanced ones flagged at `05-03`.** Its
realized test-split counts (992, 873, 1135) diverge from the even `n_test/N_BUCKETS = 1000`
split by up to 13.5% (bucket 2), against the other two seeds' near-exact splits (within ~2.4%).
This traces to the 2,102-point exact-duplicate block at this seed's field maximum
(`05-03-SUMMARY.md`'s finding), which the tie rule (D5-07) correctly routes entirely into the
top bucket. The three seeds' realized imbalances differ from each other -- seeds 20260813 and
20260815 are both near-even, only 20260814 is skewed -- consistent with it being a property of
that specific seed's collapsed metric, not of the bucketing protocol.

## RESEARCH A2's ridge justification: measured, not confirmed

RESEARCH A2 claimed the 768-d design matrix is effectively rank-deficient at the manifold's
established 18-to-25 intrinsic dimension. The training split's own measured spectrum does
**not** confirm this:

- `condition_number = 99806.5` -- large, but not by itself evidence of a ~20-dimensional
  effective rank.
- `effective_rank_1pct = 531` (of 768 possible), `effective_rank_0.1pct = 767`,
  `effective_rank_0.01pct = 767` -- at every threshold tested, several hundred singular values
  remain above even the most permissive (1%) cutoff. If the design matrix were truly
  rank ~20-25, `effective_rank_1pct` would be near 20-25, not 531.
- `cumvar_first_20 = 0.810`, `cumvar_first_25 = 0.835` -- the first 20-25 components capture
  roughly 81-83% of variance, a substantial majority but far from the near-total variance a
  genuinely ~20-dimensional design matrix would show.
- `selected_alpha = 0.1` -- the second-smallest value in `RIDGE_ALPHA_GRID = (1e-2..1e4)`, NOT
  at a grid boundary. A near-OLS-degenerate fit (as A2's own fallback describes) would select
  the smallest grid value; `0.1` is small but not at the floor.

**Read-out:** the design matrix is better conditioned than RESEARCH A2 expected -- closer to a
mild, broad-spectrum shrinkage regime than to the severe near-20-dimensional rank deficiency A2
described. The selected alpha (0.1) sitting away from both grid ends means `RIDGE_ALPHA_GRID`
was not too narrow for this fit; the frozen rule's own graceful-degradation clause (RidgeCV
selecting the smallest alpha under a well-conditioned matrix) also did not trigger. This
invalidates nothing pre-registered -- the frozen `RIDGE_SELECTION_RULE` selected the alpha, not
the planner -- but the stated REASON in `05-PREREGISTRATION.md` for using ridge is not borne out
by the measured spectrum, and `05-FINDINGS.md` must say so per this task's own instruction.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement the bucketed branch -- one fit, three bucketings, every constant read** - `2c8b062` (feat)
2. **Task 2: The real run -- the phase's headline numbers, three verdicts and their spread** - no source diff (the run itself, recorded in this SUMMARY; `notebooks/.cache/05_curvature_probe_decodability.jsonl` is gitignored per `.gitignore` line 17)
3. **Task 3: Check the ridge justification against the training split's own spectrum** - `690a220` (feat)

**Plan metadata:** this commit (SUMMARY + STATE + ROADMAP + REQUIREMENTS, once created)

## Files Created/Modified

- `notebooks/diagnostics/curvature_probe_decodability_run.py` - `run_bucketed_mode` implemented
  in full; new helpers `_fit_and_evaluate`, `_score_one_seed`, `_load_bucket_artifact`,
  `_conditioning_diagnostics`; `selfcheck()`'s fit block routed through `_fit_and_evaluate`
  (behavior unchanged) so the file carries exactly one `fit_probe` call site
- `notebooks/.cache/05_curvature_probe_decodability.jsonl` - gitignored. 14 rows: 9
  `probe_bucket`, 3 `probe_seed`, 1 `probe_overall`, 1 `probe_conditioning`

## Decisions Made

- **Consolidated the file's `fit_probe` call site through `_fit_and_evaluate`**, shared with
  `selfcheck()`. See key-decisions above and Deviations below -- this is documented as a
  deviation because it required editing `selfcheck()`, which Task 1's action text did not
  explicitly mention touching.
- **The conditioning diagnostic runs inline inside `run_bucketed_mode`'s existing real-run
  branch**, not behind a new mode/flag, since the plan's own artifacts section pins the
  runner's `--mode` surface to the four pre-existing values and the diagnostic needs the
  training split already in scope from the one fit.
- **Ran the full pipeline twice** (once for Task 2's commit, once again after Task 3's code
  addition) rather than hand-editing the JSONL file. Both runs are numerically identical
  (fully deterministic under the frozen seeds) -- verified directly, not assumed -- so this is
  not "re-running after seeing a number in hope of a different one" (the pre-registration's
  own prohibition); it is the same deterministic protocol executed twice, the second time with
  one additional read-only diagnostic appended. Task 2's reported headline numbers are exactly
  what both runs produced.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `selfcheck()`'s pre-existing `fit_probe` call conflicted with Task 1's "exactly one call site" acceptance criterion**
- **Found during:** Task 1, before writing `run_bucketed_mode`'s body.
- **Issue:** Task 1's `<verify>` block greps the whole file (comment lines excluded, but
  docstrings and code lines both counted) for the literal string `linear_probe.fit_probe` and
  asserts the count is exactly 1. `selfcheck()` (established at plan `05-01`, unmodified since)
  already called `linear_probe.fit_probe` directly -- so simply adding the planned call inside
  `run_bucketed_mode` would have made the count 2, failing the acceptance criterion the plan
  itself specifies.
- **Fix:** Extracted a shared `_fit_and_evaluate(X_train, Y_train, X_test, Y_test, alpha_grid,
  alpha_per_target, fit_intercept, r2_multioutput)` helper that wraps `linear_probe.fit_probe`
  / `predict_probe` / `per_point_residuals` / `aggregate_r2` in the same order both callers
  already used. Rewired `selfcheck()` to call it (identical arguments, identical resulting
  values) and used it as `run_bucketed_mode`'s sole fit path. This is the only way to satisfy
  "exactly one fit_probe call site" as a textual property of the file once a second real caller
  exists, while keeping the actual semantic property the plan cares about ("Exactly one probe
  fit happens per invocation" -- unaffected either way, since `selfcheck()` and
  `run_bucketed_mode()` were always separate invocations).
- **Files modified:** `notebooks/diagnostics/curvature_probe_decodability_run.py`
- **Verification:** `grep -v '^#' ... | grep -c 'linear_probe.fit_probe'` returns 1;
  `selfcheck()`'s own assertions (recovered R² > 0.99, Frobenius identity) still pass, verified
  by direct inspection of the refactored call producing the same four return values in the
  same order the original inline code computed.
- **Committed in:** `2c8b062` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 3 blocking-issue fix -- a pre-existing call site that
would have broken this task's own acceptance criterion once a second legitimate caller was
added).
**Impact on plan:** No scope creep and no behavior change to `selfcheck()`. The fix is
structural (one shared helper instead of two independent call sites) and was necessary to
satisfy the plan's own stated grep-based acceptance check.

## Issues Encountered

None beyond the auto-fixed call-site consolidation documented above.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None. Every row in the JSONL record is a real, measured value from the full 10,000-point run;
no field is a hardcoded placeholder.

## Next Phase Readiness

- `notebooks/pu_manifold/linear_probe.py` remains untouched (`git diff --quiet` verified after
  every task) and the `05-04` freeze commit (`32dabe3`) remains an ancestor of `HEAD`
  (`git merge-base --is-ancestor` verified).
- `notebooks/.cache/05_curvature_probe_decodability.jsonl` exists with the phase's complete
  headline result: `SPLIT ACROSS SEEDS`, per-seed verdicts and their supporting statistics, and
  the conditioning diagnostic -- everything `05-06` (the phase notebook) needs to read and
  present.
- **`05-FINDINGS.md` (not yet written, presumably `05-06`'s job or a later task) must carry
  forward, verbatim, three things measured here:** (1) the three per-seed verdicts are not
  statistically independent (one shared split), (2) RESEARCH A2's ridge justification is not
  confirmed by the measured spectrum, and (3) seed 20260814's realized test-split imbalance
  traces to its known exact-duplicate block, not to a bucketing-protocol defect.
- CLAUDE.md's Swiss-roll sanity-check rule does not trigger for this plan: no new
  manifold-learning or representation-learning model was introduced -- this plan fits one ridge
  map and buckets residuals against curvature fields already extracted and validated in prior
  plans, per this plan's own `<verification>` block's explicit determination.
- No blockers.

---
*Phase: 05-curvature-conditioned-linear-decodability*
*Completed: 2026-08-24*

## Self-Check: PASSED

`notebooks/diagnostics/curvature_probe_decodability_run.py` found on disk with both task
commits' changes present; commit hashes `2c8b062` and `690a220` found in `git log`;
`notebooks/.cache/05_curvature_probe_decodability.jsonl` found on disk with 14 rows (9
`probe_bucket`, 3 `probe_seed`, 1 `probe_overall`, 1 `probe_conditioning`), all values matching
those reported above by direct re-inspection.
