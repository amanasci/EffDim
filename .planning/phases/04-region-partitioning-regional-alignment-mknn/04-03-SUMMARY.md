---
phase: 04-region-partitioning-regional-alignment-mknn
plan: 03
subsystem: manifold-curvature
tags: [curvature, region-partitioning, diametrical-clustering, pre-registration, mknn]

# Dependency graph
requires:
  - phase: 04-region-partitioning-regional-alignment-mknn
    provides: "plan 04-01's mknn.py and region_partition_mknn_run.py (--mode global);
      plan 04-02's density-corrected k-freeze (K_FROZEN=500, rule_fired=false, fallback
      provenance, in notebooks/.cache/04_k_freeze.json)"
provides:
  - "notebooks/pu_manifold/region_partition.py: canonical_eigvec_sign, region_partition
    (D4-09 diametrical sign split), region_counts, and every Phase-4 pre-registered
    constant (MIN_NORM_PERCENTILE, MIN_REGION_N, MKNN_K_GRID, HEADLINE_K, NULL_QUANTILE,
    CONFIDENCE_LEVEL, N_PERMUTATIONS, N_BOOTSTRAP, FIELD_D, K_DENSITY, SEED, K_FROZEN,
    K_FREEZE_RULE, COVARIANCE_FORM, VERDICT_RULE) plus assert_preregistered()"
  - "notebooks/diagnostics/region_partition_mknn_run.py: --mode regional guarded behind
    assert_preregistered() and a frozen-partition-artifact existence check, failing
    loudly rather than computing anything"
  - "04-PREREGISTRATION.md: the committed, timestamped pre-registration record --
    every constant, the verbatim verdict rule, K_FROZEN's fallback provenance, all ten
    CONTEXT.md discretion items resolved, the checkpoint's ratification note, and the
    three accepted gaps this pre-registration sits on top of"
affects: [04-04, 04-05, 04-06]

tech-stack:
  added: []
  patterns:
    - "Pre-registration-before-measurement, enforced by commit ordering: Task 1 committed
      the sign-split helper with zero PU region labels on disk; Task 3 committed the
      frozen constants and 04-PREREGISTRATION.md, still with zero regional MKNN numbers
      on disk; the runner's --mode regional branch is now structurally incapable of
      computing a regional cell without both existing"
    - "Fail-loudly guard pattern (assert_preregistered + artifact existence check) mirrors
      D4-07's freeze_k pattern from plan 04-02: never silently defaults, never adjusts a
      threshold after the fact"

key-files:
  created:
    - notebooks/pu_manifold/region_partition.py
    - notebooks/pu_manifold/tests/test_region_partition.py
    - .planning/phases/04-region-partitioning-regional-alignment-mknn/04-PREREGISTRATION.md
  modified:
    - notebooks/diagnostics/region_partition_mknn_run.py

key-decisions:
  - "Checkpoint (Task 2, blocking) ratified under the user's standing authorization while
    asleep, via the orchestrator's pre-authorization: ratify-recommended selected with no
    amendments -- every constant is exactly the plan's proposed value, and the
    majority-across-k alternative was explicitly rejected. Recorded in
    04-PREREGISTRATION.md as a ratification under standing authorization, not a silent
    default and not a claim of line-by-line personal review."
  - "region_counts(labels, n_excluded, n_zero_projection=0): the plan's text lists
    region_counts's call signature as (labels, n_excluded) but also requires its return
    dict to carry n_zero_projection, which is not derivable from labels+n_excluded alone.
    Resolved by adding n_zero_projection as an optional third positional/keyword argument
    (default 0) -- callers pass region_partition's own n_zero_projection value through;
    the two-argument call shape the plan's function-signature line names still works
    unchanged."
  - "Test 2's inclusive-boundary fixture uses 21 points (norms 1..21), not the plan's
    illustrative 1..20: with n=20 and NumPy's default linear-interpolation percentile
    method, the 25th percentile of 1..20 falls at 5.75 -- strictly between two data
    points, so no point's norm equals it exactly and the >=-boundary behaviour the test
    exists to prove could not be exercised. n=21 places the 25th percentile exactly on a
    data point (6.0) with no interpolation, which is what the acceptance criterion
    actually requires."

requirements-completed: [REGN-03, REGN-04, MKNN-07]

coverage:
  - id: D1
    description: "Diametrical sign-split partition helper (canonical_eigvec_sign,
      region_partition, region_counts) recovers a known two-antipodal-cone split exactly
      (ARI=1.0, |dot(v,w)|>0.99), keeps the exact-percentile boundary point (inclusive
      >=), is reproducible with a canonical eigenvector sign, and correctly sums
      region/exclusion counts including zero-projection points assigned to region 0"
    requirement: REGN-03
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_region_partition.py -q (7 passed)"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -q (376 passed, 1 skipped, full suite unaffected)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every Phase-4 free parameter (near-zero exclusion percentile, region
      floor and its undefined-cell behaviour, MKNN k grid and headline k, null quantile,
      confidence level, permutation/bootstrap counts, d, k_density, seed, frozen field k,
      verdict rule text with the D4-14 density caveat) frozen as named constants in
      committed source and in a committed 04-PREREGISTRATION.md, before any PU region
      label or regional MKNN number exists anywhere in the repo"
    requirement: REGN-04
    verification:
      - kind: unit
        ref: "region_partition.assert_preregistered() plus the constants/VERDICT_RULE
          assertions from the plan's Task 3 <verify> block, run directly"
        status: pass
      - kind: other
        ref: "04-PREREGISTRATION.md's constants table cross-checked against the module's
          live values (all 13 match); K_FREEZE_RULE compared byte-for-byte against
          notebooks/.cache/04_k_freeze.json's rule_text (identical)"
        status: pass
    human_judgment: true
    rationale: "Whether the ratified values themselves (MIN_REGION_N=500, the
      single-headline-k verdict shape, no multiplicity correction) are the scientifically
      correct choices, versus merely internally consistent and committed in the right
      order, was resolved at Task 2's blocking checkpoint under the user's standing
      authorization while unavailable -- not something an automated check can validate on
      its own, and the record states plainly that it was a ratification under standing
      authorization rather than a personally-reviewed decision."
  - id: D3
    description: "The runner's --mode regional refuses to compute anything: it raises
      unless region_partition.VERDICT_RULE is non-empty (assert_preregistered) and the
      frozen partition artifact notebooks/.cache/04_region_partition.npz exists, and it
      raises loudly (FileNotFoundError naming the missing artifact) rather than silently
      no-op-ing or defaulting"
    requirement: MKNN-07
    verification:
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/region_partition_mknn_run.py --mode
          regional (exit code 1, FileNotFoundError naming
          notebooks/.cache/04_region_partition.npz)"
        status: pass
      - kind: other
        ref: "notebooks/.cache/04_region_partition_mknn.jsonl contains 6 rows, all
          region=='global', 0 non-global rows -- confirmed by direct grep after this
          plan's work"
        status: pass
    human_judgment: false

duration: ~25min (no heavy compute; two full-suite pytest runs at ~2m22s each dominate
  wall time)
completed: 2026-08-24
status: complete
---

# Phase 4 Plan 3: Diametrical sign-split partition, frozen before any PU region label exists Summary

**`region_partition.py`'s diametrical sign-split helper recovers a known two-antipodal-cone
answer exactly (ARI=1.0), and every remaining free parameter of Phase 4 — the exclusion
percentile, region floor, MKNN k grid and headline k, null quantile, confidence level,
permutation/bootstrap counts, the frozen field k, and the full MKNN-07 verdict rule text —
is now frozen as named constants in committed source and in a committed
`04-PREREGISTRATION.md`, with the runner's `--mode regional` structurally unable to compute
a regional cell until both exist.**

## Performance

- **Duration:** ~25 min (no heavy compute; two full `pytest notebooks/pu_manifold/tests/`
  runs at ~2m22s each dominate wall time)
- **Started:** 2026-08-24T11:15Z (approx.)
- **Completed:** 2026-08-24T11:41Z
- **Tasks:** 3/3 complete (Task 2 is a checkpoint, resolved without a separate commit)
- **Files modified:** 4 (2 created, 1 test file created, 1 extended)

## Accomplishments
- `canonical_eigvec_sign`, `region_partition` (D4-09's diametrical sign split — within-config
  percentile exclusion, inclusive `>=` boundary, mean-centered covariance with
  `mean_unit_norm` reported beside it), and `region_counts` implemented and proven against a
  known two-antipodal-cone answer: adjusted Rand index exactly 1.0, `|dot(v, w)| > 0.99`.
- The module docstring states, in this phase's own words, the ~748-wide codimension gap
  D4-01/D4-10 leave unclosed, and names diametrical clustering (Dhillon, Marcotte & Roshan,
  *Bioinformatics* 19(13), 2003) with an explicit secondary-source citation caveat.
- Task 2's blocking checkpoint ratified — under the user's standing authorization while
  asleep, delivered via the orchestrator's pre-authorization — with **no amendments**: every
  one of the five decision items (near-zero exclusion, region floor + undefined behaviour,
  MKNN-07 verdict rule, headline k, multiplicity posture) is exactly the plan's proposed
  value.
- Every Phase-4 pre-registered constant is now a named module-level constant in
  `region_partition.py`, verified byte-for-byte against `04_k_freeze.json` for `K_FROZEN`/
  `K_FREEZE_RULE`, and `assert_preregistered()` enforces the pre-registration is intact
  before any regional computation runs.
- `04-PREREGISTRATION.md` committed: every constant's value, the verbatim `VERDICT_RULE`
  text (with the D4-14 density caveat written into the rule's own text, not stated only
  alongside it), `K_FROZEN`'s fallback-not-plateau provenance restated with its full per-k
  table, all ten `04-CONTEXT.md` discretion items resolved with concrete values, and the
  checkpoint's ratification note.
- `region_partition_mknn_run.py`'s `--mode regional` branch now calls
  `assert_preregistered()` and requires `notebooks/.cache/04_region_partition.npz` to
  exist, raising `FileNotFoundError` naming the missing artifact rather than computing
  anything — verified by direct invocation (exit code 1).
- `notebooks/.cache/04_region_partition_mknn.jsonl` still holds exactly the 6 `global` rows
  from plan 04-01 and no `regional` rows — confirmed directly after this plan's work.

## Task Commits

Each task was committed atomically:

1. **Task 1: The diametrical sign-split helper, against a known answer** - `e1106b4` (feat)
2. **Task 2: Ratify the pre-registration before any PU region label exists** — checkpoint,
   resolved under the orchestrator's pre-authorization (see Decisions Made); no separate
   commit, folded into Task 3's ratification record
3. **Task 3: Freeze the pre-registration into committed source and into the phase record** -
   `0305c77` (docs)

**Plan metadata:** committed as part of this SUMMARY's own docs commit.

_Note: Task 1 is `tdd="true"` — tests were written first, confirmed failing
(`ImportError: cannot import name 'region_partition'`), then made to pass by the
implementation, all within the single Task 1 commit (this codebase's established
single-commit-per-task convention; see 04-01/04-02's own commits for precedent)._

## Files Created/Modified
- `notebooks/pu_manifold/region_partition.py` — `canonical_eigvec_sign`,
  `region_partition`, `region_counts`, all Phase-4 pre-registered constants,
  `assert_preregistered()`
- `notebooks/pu_manifold/tests/test_region_partition.py` — known-answer, inclusive-boundary,
  reproducibility, and count-closure/zero-projection tests, plus guard tests
- `notebooks/diagnostics/region_partition_mknn_run.py` — `--mode regional` guarded behind
  `assert_preregistered()` and the frozen-artifact existence check
- `.planning/phases/04-region-partitioning-regional-alignment-mknn/04-PREREGISTRATION.md` —
  the committed pre-registration record

## Decisions Made
- **Checkpoint ratified under standing authorization, no amendments.** Task 2's blocking
  decision was answered by the orchestrator on the user's explicit standing authorization
  (asleep, phase pre-authorized to run to completion). `ratify-recommended` was selected
  verbatim; `majority-across-k` was explicitly rejected. `04-PREREGISTRATION.md` states this
  plainly as a ratification made under standing authorization while the user was
  unavailable — not a silent default, and not a claim that the user personally reviewed
  each value line by line.
- **`region_counts` gained an optional third argument, `n_zero_projection=0`.** The plan's
  text names the call shape `region_counts(labels, n_excluded)` while also requiring the
  returned dict to carry `n_zero_projection` — a value that cannot be derived from `labels`
  and `n_excluded` alone (a region-0 label doesn't distinguish "proj > 0" from
  "proj == 0"). Resolved by accepting it as an optional pass-through argument (default 0):
  the plan's named two-argument call shape still works unchanged, and callers that want the
  field populated pass `region_partition`'s own `n_zero_projection` value through.
- **Inclusive-boundary test uses 21 points, not the plan's illustrative 20.** With `n=20`
  integer norms `1..20` and NumPy's default linear-interpolation percentile, the 25th
  percentile of that sequence is `5.75` — strictly between two data points, so no point's
  norm equals it exactly and the `>=`-boundary behaviour under test could not be exercised
  as literally described. Using `n=21` (norms `1..21`) places the 25th percentile exactly
  on the data point `6.0` with zero interpolation, which is what the acceptance criterion
  ("the point at the exact percentile value is in `keep_idx`") actually requires to be
  testable at all.

## Deviations from Plan

### Auto-fixed Issues
None — no bugs, missing critical functionality, or blocking issues were found in the plan's
own action text that required Rule 1-3 auto-fixes.

**Two plan-ambiguity resolutions, not rule-triggered deviations** (documented above under
Decisions Made): `region_counts`'s third argument, and the 21- vs 20-point boundary fixture.
Both are resolutions of a genuine tension between the plan's illustrative prose and its own
machine-checked acceptance criteria, resolved in favor of the acceptance criteria, same
precedent 04-02-SUMMARY.md set for its `--k-density` default resolution.

---

**Total deviations:** 0 rule-triggered auto-fixes; 2 documented plan-ambiguity resolutions
(above).
**Impact on plan:** None on scope or correctness — both resolutions only affect how a test
fixture is constructed or how one optional argument is threaded through; the
machine-checked acceptance criteria were verified directly in both cases.

## Issues Encountered
None. Both full-suite `pytest notebooks/pu_manifold/tests/ -q` runs (before and after Task
3's edits) passed at 376 passed, 1 skipped, with no code outside this plan's own files
touched.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- **Plan 04-04 onward inherits a fully frozen pre-registration.** Every constant
  `region_partition.py` exposes, and `04-PREREGISTRATION.md`'s full record, must be treated
  as immutable for the rest of Phase 4 — amending any of them after a regional MKNN number
  exists invalidates the phase's result (per `04-PREREGISTRATION.md`'s own "what this
  document forecloses" section).
- **`--mode partition` is still `NotImplementedError`** — plan 04-04 (or whichever plan
  implements the partition-computation path) must write the actual `region_partition` call
  against the real PU field, freeze `v`/`labels`/`excluded_idx` via
  `cache.npz_cache` at stem `04_region_partition` (REGN-06), and only then does
  `--mode regional`'s existence check begin to pass.
- **No PU region label exists anywhere in the repo as of this plan's commits.** Both
  `region_partition.py` and `region_partition_mknn_run.py`'s guard were verified against
  synthetic fixtures only; the guard's `FileNotFoundError` against the real artifact path
  is itself evidence of this.
- No blockers for 04-04. The three accepted gaps (unvalidated field, unclosed codimension
  gap, reported-not-controlled density confound) `04-PREREGISTRATION.md` restates are
  unchanged from plan 04-02's handoff and remain independently tracked, to be written up in
  full by plan 04-06.

---
*Phase: 04-region-partitioning-regional-alignment-mknn*
*Completed: 2026-08-24*
