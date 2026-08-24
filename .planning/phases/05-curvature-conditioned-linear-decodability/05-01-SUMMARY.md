---
phase: 05-curvature-conditioned-linear-decodability
plan: 01
subsystem: geometry
tags: [ridge-regression, sklearn, spearman, curvature, pre-registration, pytest]

# Dependency graph
requires:
  - phase: 03-decoder-side-curvature
    provides: sealed CAE checkpoints (03_converged_cae_pu_nc4_seed2026081{3,4,5}.pt) and
      chart_curvature.chart_curvature_field
  - phase: 04-region-partitioning-regional-alignment-mknn
    provides: the pre-registration constants-block + assert_preregistered() + JSONL-runner
      pattern this plan copies (region_partition.py / region_partition_mknn_run.py)
provides:
  - notebooks/pu_manifold/linear_probe.py -- pure probe/pool/bucket/verdict functions, the
    UNSET pre-registration constants block, and assert_preregistered()
  - notebooks/pu_manifold/tests/test_linear_probe.py -- the Wave 0 test gaps
  - notebooks/diagnostics/curvature_probe_decodability_run.py -- --selfcheck (working),
    --mode field (working, real CAE checkpoint), --mode pool (NotImplementedError, 05-03),
    --mode bucketed (D5-10 guard live, body NotImplementedError until 05-05)
  - one verdict row in notebooks/.cache/05_probe_selfcheck.jsonl from planted synthetic data
affects: [05-02, 05-03, 05-04, 05-05, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pre-registration constants block + assert_preregistered() guard, copied from
      region_partition.py (Phase 4): unset today, filled once at a blocking checkpoint,
      raises RuntimeError on the first malformed constant"
    - "--selfcheck known-answer self-check on synthetic, dimensionally-matched data before any
      real number is computed, copied from region_partition_mknn_run.py's selfcheck() shape"
    - "Per-seed cache.npz_cache keyed by a full cfg dict (seed, n_charts, mode, batch_size,
      n_rows, subsample_file, curvature_convention, source_function) so a re-run under a
      different configuration raises rather than silently reusing a stale artifact"

key-files:
  created:
    - notebooks/pu_manifold/linear_probe.py
    - notebooks/pu_manifold/tests/test_linear_probe.py
    - notebooks/diagnostics/curvature_probe_decodability_run.py
  modified: []

key-decisions:
  - "POOLING_METHOD's unset sentinel is None (not the empty string the general 'string
    constants are \"\"' convention would suggest) -- this plan's own must_haves/acceptance
    criteria name it explicitly alongside BUCKET_EDGES and N_BUCKETS as the three constants
    that gate whether the bucketed path can run at all"
  - "test_pool_seeds_no_single_seed_dominates uses 8 piecewise levels, not the 4 named in the
    plan's <behavior> text -- with exactly 4 levels over 500 points the tie-corrected Spearman
    ceiling between ANY continuous comparator and the tied field itself is 0.9682, provably
    below the plan's own >0.99 threshold regardless of estimator quality; 8 levels raises that
    ceiling above 0.99 while remaining a small, collapsed-metric level count"
  - "The fixture's independent seed (field1) is drawn from a positive uniform distribution,
    not the same underlying continuous field the two piecewise seeds are quantized from --
    reusing the same underlying field made the two piecewise seeds' per-seed-median
    normalization exactly cancel their scale difference, making pool_seed_fields and a plain
    np.mean produce IDENTICAL rank orders (no failure mode to demonstrate); an independent
    seed is required for normalization's equal-footing effect to actually diverge from
    magnitude-weighted averaging"

requirements-completed: [D5-01, D5-03, D5-06, D5-08, D5-10]

coverage:
  - id: D1
    description: "notebooks/pu_manifold/linear_probe.py: pre-registration constants block
      (written UNSET), assert_preregistered() guard, and 13 pure probe/pool/bucket/verdict
      functions"
    requirement: D5-01
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_linear_probe.py -q (9 passed, 1 xfailed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "D5-03's citation corrected: the decoder-side curvature call site is pinned
      to chart_curvature.chart_curvature_field(model, x, mode='reverse'), never
      decoder_curvature -- verified against a genuine sealed CAE checkpoint on real PU rows"
    requirement: D5-03
    verification:
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --mode field --smoke --smoke-n 64"
        status: pass
    human_judgment: false
  - id: D3
    description: "The D5-10 guard: --mode bucketed refuses to compute anything, raising
      RuntimeError from assert_preregistered() before touching any data, with every
      pre-registration constant still unset"
    requirement: D5-10
    verification:
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --mode bucketed (exits non-zero, RuntimeError from assert_preregistered)"
        status: pass
    human_judgment: false
  - id: D4
    description: "--selfcheck runs the complete probe-to-verdict path on planted synthetic
      data (n=900, dimensionally PU-shaped) and returns the planted answer; writes exactly
      one JSONL row tagged data_source=synthetic_planted; no PU probe number exists anywhere"
    verification:
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck (exit 0, no [FAIL] lines)"
        status: pass
    human_judgment: false

duration: 19min
completed: 2026-08-24
status: complete
---

# Phase 5 Plan 1: Whole-Machine Tracer on Planted Data Summary

**Pre-registration-guarded ridge probe + decoder-side curvature pipeline (linear_probe.py, its
test suite, and curvature_probe_decodability_run.py), proven end-to-end on synthetic data and
on 64 real PU rows through a genuine sealed CAE checkpoint, with the bucketed path provably
dead behind assert_preregistered() until the 05-04 freeze.**

## Performance

- **Duration:** 19 min (first task commit to last; excludes context-loading)
- **Started:** 2026-08-24T11:48:03-04:00
- **Completed:** 2026-08-24T12:06:55-04:00
- **Tasks:** 3
- **Files modified:** 3 (all new)

## Accomplishments

- `notebooks/pu_manifold/linear_probe.py`: 13 pure functions (`train_test_split_indices`,
  `fit_probe`, `predict_probe`, `per_point_residuals`, `aggregate_r2`, `pool_seed_fields`,
  `bucket_edges_from_field`, `assign_buckets`, `bucket_by_field`, `bucket_counts`,
  `bucket_residual_ci`, `size_matched_check`, `apply_verdict_rule`) plus
  `assert_preregistered()` and the full pre-registration constants block, every constant
  written UNSET (`VERDICT_RULE = ""`, `BUCKET_EDGES = None`, `POOLING_METHOD = None`,
  `N_BUCKETS = None`, ...)
- D5-03's own citation corrected in the module's own words: the decoder-side curvature call
  site is `chart_curvature.chart_curvature_field(model, x, mode="reverse")`, never
  `decoder_curvature` (which is `chart_curvature.py` with the chart-routed two-hop composition
  removed, built for Phase 02.6's no-chart-index substrates) -- verified against a genuine
  sealed checkpoint (`03_converged_cae_pu_nc4_seed20260813.pt`) on 64 real `legacysurvey` rows
- 10 Wave-0 tests in `test_linear_probe.py`, including the RESEARCH A3 R²/residual identity
  verified numerically (not trusted from citation), the realized-test-split size-match
  distinction (D5-08/Pitfall 4) that undercut Phase 4's verdict, and the D5-10 guard proven
  live in both directions via `monkeypatch`
- `curvature_probe_decodability_run.py`'s `--selfcheck` runs the complete probe-to-verdict
  path on synthetic, dimensionally PU-shaped data (n=900, d=768) with a planted linear map and
  a planted curvature-to-residual ordering, recovers R²>0.99, and returns verdict `HOLDS`
- `--mode bucketed`'s D5-10 guard is live: calls `linear_probe.assert_preregistered()` before
  touching any data, which raises today because every constant is unset
- No PU probe number exists anywhere in the repository at the end of this plan
  (`notebooks/.cache/05_curvature_probe_decodability.jsonl` does not exist)

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end verdict path on planted data** - `5888d0d` (feat)
2. **Task 2: The Wave 0 test file** - `f26b9c4` (test)
3. **Task 3: The real curvature call site and the D5-10 guard** - `694cda9` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `notebooks/pu_manifold/linear_probe.py` - Probe fit/score, seed pooling, bucketing, verdict
  functions; the UNSET pre-registration constants block; `assert_preregistered()`
- `notebooks/pu_manifold/tests/test_linear_probe.py` - 10 known-answer/boundary/guard tests
  for `linear_probe.py`, run explicitly (excluded from `pyproject.toml`'s `testpaths`)
- `notebooks/diagnostics/curvature_probe_decodability_run.py` - `--selfcheck`, `--mode field`
  (working), `--mode pool` (NotImplementedError until 05-03), `--mode bucketed` (D5-10 guard
  live, body NotImplementedError until 05-05), `build_cae`/`load_converged_model`/
  `extract_seed_field`

## Decisions Made

- **POOLING_METHOD's unset value is `None`, not `""`.** The plan's own must_haves and
  acceptance criteria name `POOLING_METHOD is None` explicitly (grouped with `BUCKET_EDGES`
  and `N_BUCKETS` as the three constants gating the bucketed path), overriding the general
  "string constants are empty-string" convention stated in the same plan for descriptive-text
  constants like `VERDICT_RULE`.
- **`test_pool_seeds_no_single_seed_dominates` uses 8 piecewise levels, not 4.** Measured
  directly: with exactly 4 tied levels over 500 points, the tie-corrected Spearman ceiling
  between any fully-resolved comparator and the tied field itself is `0.9682477730493367`,
  strictly below the plan's own `>0.99` raw-average threshold -- a mathematical impossibility,
  not an estimator failure. 8 levels raises the ceiling above `0.99` (measured `0.9950`) while
  remaining a small, "collapsed-metric" level count consistent with the real fields' shape.
- **The fixture's third (lowest-magnitude) seed is independent noise, not a scaled copy of the
  same underlying continuous field the two piecewise seeds are quantized from.** Reusing the
  same underlying field for all three seeds made the two piecewise seeds' per-seed-median
  normalization exactly cancel their scale difference (dividing a scalar multiple of a pattern
  by its own median always recovers the same normalized pattern), so `pool_seed_fields` and a
  plain `np.mean` produced IDENTICAL rank orders — there was no failure mode left to
  demonstrate. An independent third seed is required for magnitude-weighted averaging and
  equal-footing normalization to actually diverge.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_pool_seeds_no_single_seed_dominates fixture redesigned for
mathematical feasibility**
- **Found during:** Task 2 (Wave 0 test file)
- **Issue:** The plan's `<behavior>` text specified "two piecewise-constant with only four
  distinct values each" and asserted the raw np.mean's Spearman against the largest-magnitude
  seed must exceed 0.99. Measured directly: with 4 tied levels over 500 points, no comparator
  can exceed a Spearman of `0.9682` against that tied field, regardless of how well-ordered it
  is -- the assertion as specified could never pass with any implementation.
- **Fix:** Raised the piecewise level count to 8 (ceiling `0.9950` > `0.99`) and made the
  fixture's continuous/independent seed genuinely uncorrelated with the piecewise pattern
  (rather than a scaled copy of it), which also fixed a second latent issue: identical
  underlying patterns made per_seed_median_divide normalization mathematically equivalent to
  plain averaging, giving no failure mode to demonstrate at any level count.
- **Files modified:** notebooks/pu_manifold/tests/test_linear_probe.py
- **Verification:** `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q` -- 9 passed, 1 xfailed; verified robust across 20 rng seeds before finalizing
- **Committed in:** f26b9c4 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary for the test to be satisfiable at all; the qualitative claim the
test demonstrates (magnitude-weighted averaging lets the largest seed dominate; per-seed
normalization removes that) is unchanged from the plan's intent.

## Issues Encountered

None beyond the deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `linear_probe.py`'s functions, `assert_preregistered()` guard, and the corrected
  `chart_curvature_field` call site are ready for 05-02 (per-seed field extraction over the
  full 10,000-row subsample and inter-seed diagnostics) and 05-03 (`--mode pool`
  implementation).
- The pre-registration constants remain fully unset; no PU probe number exists anywhere.
  05-04's freeze is unblocked to proceed once 05-02/05-03 land.
- No blockers.

---
*Phase: 05-curvature-conditioned-linear-decodability*
*Completed: 2026-08-24*

## Self-Check: PASSED

All created files found on disk; all three task commit hashes (`5888d0d`, `f26b9c4`,
`694cda9`) found in `git log`.
