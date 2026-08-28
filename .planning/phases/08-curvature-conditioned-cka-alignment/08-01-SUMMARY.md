---
phase: 08-curvature-conditioned-cka-alignment
plan: 01
subsystem: testing
tags: [numpy, scipy, cka, hsic, kernel-methods, pytest]

# Dependency graph
requires:
  - phase: 07-crossmodal-curvature-alignment
    provides: "crossmodal_curvature.py's ALIGNMENT_METRIC='mknn' (D7-07), superseded by phase
      decision, not edited"
provides:
  - "notebooks/pu_manifold/cka.py — pure-numpy Song et al. (2012) unbiased-HSIC estimator,
    linear/RBF Gram builders, cka_on_subset (Gram-matrix-once/index-many), all 14 Phase 8
    gating constants declared UNSET with assert_preregistered() guard"
  - "notebooks/diagnostics/08_cka_alignment_run.py — --mode selfcheck drives the D8-16
    invariance ladder for both kernels on synthetic pairs; other modes exit 2 naming the
    plan that implements them"
  - "notebooks/pu_manifold/tests/test_cka.py — 24 passing tests pinning the estimator's
    closed-form behavior, the double-centering trap, the Gram-matrix-once identity, dtype
    agreement, and every freeze-guard branch"
affects: [08-02, 08-03, 08-04, 08-05, 08-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Freeze-provenance module docstring + constants-block (value + prose-rule string pairs,
      all UNSET) + assert_preregistered(), copied from density_stratified_null.py/linear_probe.py"
    - "Gram-matrix-once, submatrix-index-many (cka_on_subset) as an exact, not approximate,
      operation"
    - "sigma as a required positional argument with no default, closing the per-subset-bandwidth
      confound at the interface level"
    - "Behavioral pin against an estimator-substitution trap (test asserts against the CLASSICAL
      BIASED formula built independently in the test) rather than a source grep"

key-files:
  created:
    - notebooks/pu_manifold/cka.py
    - notebooks/diagnostics/08_cka_alignment_run.py
    - notebooks/pu_manifold/tests/test_cka.py
  modified: []

key-decisions:
  - "Task 2's double-centering test description was inverted from what a mathematically correct
    unbiased-HSIC estimator actually does: unbiased_hsic is provably INVARIANT to
    double-centering its input (verified by hand-derivation and numerically to ~1e-16 relative on
    multiple seeds/sizes), because the U-statistic correction terms are exactly what makes
    explicit centering unnecessary. test_double_centering_changes_the_answer was rewritten to pin
    the real Pitfall-1 trap instead: silently substituting the CLASSICAL BIASED HSIC formula
    (tr(K_c L_c)/(n-1)^2 on explicitly double-centered K_c/L_c) for the unbiased correction-term
    formula. The test builds the biased quantity independently in the test body (never via
    cka.unbiased_hsic) and asserts it differs materially from cka.unbiased_hsic's own output on
    the same raw K/L, plus a positive confirmation that unbiased_hsic on an explicitly
    pre-centered pair matches its own raw-input value to near machine precision."
  - "float32 vs float64 Gram storage agree on linear CKA to 1.71e-11 absolute at n=3000
    (well inside the 1e-5 acceptance threshold), confirming GRAM_DTYPE='float32' (to be frozen
    at 08-04) costs nothing measurable."

patterns-established:
  - "Test-local tolerance constants (ATOL_CLOSED_FORM, ATOL_INDEPENDENCE, RTOL_REFERENCE,
    ATOL_DTYPE) live in the test file, not as pre-registered cka.py constants, because D8-20
    makes the invariance ladder non-gating."

requirements-completed: [D8-01, D8-02, D8-04, D8-16, D8-17, D8-22, D8-23, D8-24]

coverage:
  - id: D1
    description: "Song et al. (2012) unbiased-HSIC estimator, correct against a reference value
      computed independently in the test as three explicit terms"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_unbiased_hsic_matches_reference"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_unbiased_hsic_raises_below_n4"
        status: pass
    human_judgment: false
  - id: D2
    description: "Double-centering / classical-biased-HSIC substitution trap pinned behaviorally"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_double_centering_changes_the_answer"
        status: pass
    human_judgment: false
  - id: D3
    description: "Linear and RBF CKA invariance ladder (rotation, scaling, independence,
      noise-ladder monotonicity) for both kernels"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_linear_cka_invariances"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_rbf_cka_invariances"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_noise_ladder_monotone"
        status: pass
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode selfcheck --record-path notebooks/.cache/08_scratch_tracer.jsonl"
        status: pass
    human_judgment: false
  - id: D4
    description: "cka_on_subset's Gram-matrix-once/index-many identity, exact to 1e-10"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_cka_on_subset_matches_direct"
        status: pass
    human_judgment: false
  - id: D5
    description: "float32/float64 Gram dtype agreement within 1e-5 absolute at n=3000"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_gram_dtype_agreement"
        status: pass
    human_judgment: false
  - id: D6
    description: "rbf_gram's sigma has no default (D8-03 per-subset-bandwidth confound closed
      at the interface)"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_sigma_is_required_argument"
        status: pass
    human_judgment: false
  - id: D7
    description: "Every one of the 14 Phase 8 gating constants declared UNSET;
      assert_preregistered() raises on each individually and passes when all 14 are filled"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_assert_preregistered_rejects_unset_constant"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka.py#test_assert_preregistered_passes_when_all_constants_set"
        status: pass
    human_judgment: false
  - id: D8
    description: "src/effdim/ and all nine sealed pu_manifold modules remain byte-identical;
      no PU data opened; no torch import"
    verification:
      - kind: other
        ref: "git diff --name-only c34ba15..HEAD -- src/effdim/ (empty); git status --porcelain notebooks/pu_manifold/ (only tests/test_cka.py, untracked/new)"
        status: pass
    human_judgment: false

duration: ~1h10m (across two sessions; Task 1 commit 20:11:52 to Task 2 commit 21:10:47 local)
completed: 2026-08-27
status: complete
---

# Phase 08 Plan 01: CKA Estimator Tracer + Test Suite Summary

**Song et al. (2012) unbiased-HSIC/CKA estimator with linear and RBF kernels, a
Gram-matrix-once/index-many subset path, and a 14-constant pre-registration freeze guard — proven
end to end on a synthetic invariance ladder and pinned by 24 passing unit tests, with every gating
constant still UNSET.**

## Performance

- **Duration:** ~1h (Task 1 to Task 2, across a session interruption/resume)
- **Completed:** 2026-08-27T21:10:47-04:00
- **Tasks:** 2/2
- **Files modified:** 3 (all new)

## Accomplishments

- `unbiased_hsic`/`cka`/`linear_gram`/`rbf_gram`/`median_pairwise_distance`/`cka_on_subset`
  implemented as pure numpy functions with no file I/O and no module-level defaults.
- D8-16's invariance ladder runs end to end for both kernels from one command
  (`08_cka_alignment_run.py --mode selfcheck`): orthogonal rotation and isotropic scaling both
  land on 1.0 for linear CKA (deviation 0.0 and 2.22e-16 respectively); RBF is correctly NOT
  scale-invariant at fixed sigma (measured 0.851708, 0.148 away from 1.0 — the ladder's negative
  check firing as intended); independent columns read -0.000916 (linear) / -0.001021 (rbf), both
  well under the 0.05 threshold; the six-point additive-noise ladder is strictly decreasing for
  both kernels (linear: 1.0 → 0.058210, rbf: 1.0 → 0.030374). Selfcheck wallclock: 15.03s.
- float32 vs float64 Gram storage agree on CKA to 1.71e-11 absolute at n=3000 (measured directly;
  well inside the 1e-5 acceptance bound).
- 24/24 unit tests pass in `test_cka.py` (10 named behavioral tests + 1 additional
  all-constants-filled positive check + the 14-way parametrized `_REQUIRED_CONSTANTS` rejection
  sweep). Full `notebooks/pu_manifold/tests/` suite stays green: 695 passed, 1 pre-existing skip
  (unrelated to this plan).
- All 14 Phase 8 gating constants remain UNSET; `assert_preregistered()` raises `RuntimeError`
  naming `KERNELS` (first in declaration order) as required at this stage of the phase.
- `src/effdim/` untouched; all nine sealed `notebooks/pu_manifold/*.py` modules byte-identical;
  no `torch` import in either new file; no PU `.npz`/subsample file opened.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end CKA estimator — synthetic invariance ladder, one path, both kernels** -
   `91bd70a` (feat) — committed in a prior session
2. **Task 2: Unit tests for the estimator, the invariance ladder and the freeze guard** -
   `37f0e25` (test) — committed this session after independently re-verifying every acceptance
   criterion (did not trust the prior interrupted agent's "24 tests pass" claim; re-ran the full
   suite, the parametrized-case count, the `git diff`/`git status` purity checks, and the
   selfcheck run from scratch)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md)

## Files Created/Modified

- `notebooks/pu_manifold/cka.py` - Unbiased-HSIC/CKA estimator, Gram builders, freeze guard
  (14 UNSET constants)
- `notebooks/diagnostics/08_cka_alignment_run.py` - `--mode selfcheck` runner driving the D8-16
  invariance ladder; other modes exit 2 naming the implementing plan
- `notebooks/pu_manifold/tests/test_cka.py` - 24 tests: reference-value pin, n<=3 ValueError,
  double-centering-trap pin, both kernels' invariances, noise-ladder monotonicity, subset
  identity, dtype agreement, sigma-required-argument, freeze-guard rejection sweep + positive
  check

## Decisions Made

- **Double-centering test rewritten from the plan's literal description** (see `key-decisions`
  in frontmatter): the plan asked for a test that feeds a double-centered Gram to
  `unbiased_hsic` and asserts the result differs from the raw zero-diagonal result. Measured
  directly, a mathematically correct unbiased-HSIC implementation is provably INVARIANT to
  double-centering (the U-statistic correction terms are exactly what makes explicit centering
  unnecessary), so the plan's literal assertion direction can never be satisfied by correct code.
  `test_double_centering_changes_the_answer` was rewritten to pin the real Pitfall-1 trap instead
  — silent substitution of the classical biased HSIC formula — built independently in the test
  and compared against `cka.unbiased_hsic`'s own output on the same raw inputs. This is a
  correctness fix to the plan's test description (Rule 1: the plan's literal construction cannot
  be satisfied by correct code; the underlying protection the plan wanted — pinning the
  double-centering trap — is preserved and, if anything, made stricter), not a weakening of
  coverage. Documented in the test's own docstring as well, so a future reader does not mistake
  the construction for an oversight.
- Tolerance literals (`ATOL_CLOSED_FORM`, `ATOL_INDEPENDENCE`, `RTOL_REFERENCE`, `ATOL_DTYPE`)
  kept as test-local constants per the plan's `<discretion_decisions>` — not pre-registered,
  since D8-20 makes the invariance ladder non-gating.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan's test specification] Corrected `test_double_centering_changes_the_answer`'s assertion direction**
- **Found during:** Task 2 verification (re-derived by hand and numerically before trusting the
  prior interrupted agent's test)
- **Issue:** The plan's literal instruction — assert `unbiased_hsic` on a double-centered Gram
  differs from the raw zero-diagonal result by more than 1e-9 relative — is false for a correct
  implementation. `unbiased_hsic(H K H, H L H) == unbiased_hsic(K, L)` to machine precision by
  construction of the U-statistic correction terms.
- **Fix:** Rewrote the test to build the CLASSICAL BIASED HSIC formula independently (never via
  `cka.unbiased_hsic`) and assert it differs materially from the unbiased estimator's own output
  on the same raw inputs, plus a positive confirmation of the invariance property itself.
- **Files modified:** `notebooks/pu_manifold/tests/test_cka.py`
- **Verification:** Test passes; hand-derivation and numerical check (multiple seeds/sizes,
  agreement to ~1e-16 relative) confirm the invariance property is correct, not a masked bug.
- **Committed in:** `37f0e25` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — plan test-specification bug, corrected before
committing)
**Impact on plan:** The correction strengthens rather than weakens the Pitfall-1 protection the
plan intended. No scope creep; no change to `cka.py` itself.

## Issues Encountered

A previous executor session was interrupted mid-Task-2, after writing `test_cka.py` to disk and
claiming "24 tests pass" without having verified the plan's acceptance criteria or committed. This
session did not trust that claim: re-ran the full test file, the parametrized rejection-sweep
count (confirmed 14, matching `len(cka._REQUIRED_CONSTANTS)`), the full `notebooks/pu_manifold/tests/`
suite (695 passed, 1 pre-existing unrelated skip), the selfcheck runner from a clean scratch
JSONL, the `assert_preregistered()` non-zero-exit check, and the `git diff`/`git status` purity
checks against `src/effdim/` and the sealed `pu_manifold` modules — all independently confirmed
before committing. One test (`test_double_centering_changes_the_answer`) needed correction; see
Deviations above.

The plan's acceptance criterion `grep -n "notebooks/.cache\|subsample_\|np.load" notebooks/pu_manifold/tests/test_cka.py prints nothing` technically fails on a literal match: the test file's header docstring states in prose that the suite "reads nothing from `notebooks/.cache/`" — the same convention already present verbatim in the sealed `test_density_stratified_null.py` (line 8), which would fail the identical literal grep. This is a docstring-prose false positive, not actual cache/npz usage (no `np.load`, no `subsample_*.npz` read, no cache path constructed), and is accepted as consistent with established precedent.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `cka.py`'s estimator surface (`unbiased_hsic`, `cka`, `linear_gram`, `rbf_gram`,
  `median_pairwise_distance`, `cka_on_subset`) is complete, tested, and ready for 08-02's
  splitting logic (`tertile_split_within_strata`, `realized_h_contrast`, etc.) to land in the
  same file without touching the estimator.
- All 14 gating constants remain UNSET, as required — 08-04 is still the single commit that may
  fill them.
- No blockers. `src/effdim/` and all sealed modules confirmed untouched.

---
*Phase: 08-curvature-conditioned-cka-alignment*
*Completed: 2026-08-27*

## Self-Check: PASSED

All created files confirmed present on disk; both task commits (`91bd70a`, `37f0e25`) confirmed
in git log.
</content>
