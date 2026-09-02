---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 01
subsystem: research-stats
tags: [numpy, scipy, scikit-learn, torch, ridge-regression, spearman, freedman-lane, pu-manifold]

# Dependency graph
requires:
  - phase: 07-crossmodal-curvature
    provides: "crossmodal_curvature.split_indices, plant_positive_control mechanism, assert_preregistered idiom"
  - phase: 07.1-density-stratified-null-and-seed-stability
    provides: "density_stratified_null.density_strata, fresh-redeclaration-across-freeze-boundary discipline"
  - phase: 08-cka-alignment-and-instrument-fidelity
    provides: "test_cka_import_purity.py's subprocess-per-import-order snapshot mechanism"
provides:
  - "physics_labels.py: row-alignment loader/proof module (alignment_r2_curve, alignment_verdict, mask_sentinels, canonical_label, shifted_pairing), all gating constants UNSET"
  - "physics_curvature_probe.py: OOF ridge wrapper, anchor draw, radial/tangential decomposition, 3-control partial (delegates to cross_split_curvature.partial_spearman), Freedman-Lane null, verdict rules, all gating constants UNSET except the non-gating SWISS_ROLL_APPLICABILITY_RULE"
  - "09_physics_curvature_run.py --mode smoke: proves the whole Phase 9 statistical path end to end on synthetic arrays"
  - "94 tests across three new test files, full pu_manifold suite green (854 passed, 2 skipped)"
affects: [09-02, 09-03, 09-04, 09-05, 09-06, 09-07, 09-08, 09-09, 09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fresh-redeclaration-across-freeze-boundary (density_stratified_null.py's own discipline), applied to two new modules simultaneously"
    - "Freeze guard (assert_preregistered) with per-constant fail-fast RuntimeError, plus exact-string-equality guards on rule constants owned by the module that implements the behaviour they describe"
    - "Two acyclic modules joined by dependency injection (the OOF estimator passed as a callable) rather than a direct import"
    - "Gaussian-copula golden-array construction for a tight-tolerance statistical parity pin, solved offline and replayed as a fixed literal at test time"

key-files:
  created:
    - notebooks/pu_manifold/physics_labels.py
    - notebooks/pu_manifold/physics_curvature_probe.py
    - notebooks/diagnostics/09_physics_curvature_run.py
    - notebooks/pu_manifold/tests/test_physics_curvature_probe.py
    - notebooks/pu_manifold/tests/test_physics_labels.py
    - notebooks/pu_manifold/tests/test_physics_import_purity.py
  modified: []

key-decisions:
  - "oof_ridge_predictions passes alpha_grid=(float(alpha), float(alpha)) -- a two-entry duplicate-valued tuple -- to the sealed linear_probe.fit_probe, rather than the plan's literal one-element (float(alpha),), to route around a measured sklearn==1.9.0 RidgeCV in-place-mutation TypeError on single-candidate tuples. Bit-identical to a true single-alpha Ridge fit; linear_probe.py itself is not edited."
  - "plant_curvature_positive_control's bisection direction is measured empirically (achieved statistic at slope 0.0 vs 2.0) rather than assumed fixed-increasing, because the retargeted statistic (controlled_partial(planted, y, Z), not spearmanr(h_real, planted)) can legitimately decrease with slope when h_real and y are negatively associated -- exactly this phase's own D9-09 hypothesis. Internal discretization reduced from an initial 1000 to 10 so the achieved statistic stays smooth across the whole bisection bracket instead of saturating in its first few percent."
  - "SWISS_ROLL_APPLICABILITY_RULE is filled now (not left UNSET) -- a non-gating declarative fact about phase methodology per the plan's own standing_rule_declarations, distinct from the ~72 gating constants that stay UNSET until 09-05."
  - "The parity test used the Gaussian-copula construction, not the transcribed-array fallback: a correlation matrix was solved offline (scipy.optimize, against cross_split_curvature.partial_spearman's own formula) to reproduce the colleague's raw -0.4124 and controlled -0.2405 to within ~2e-6 (well inside the 1e-3 tolerance); the solved Cholesky factor is committed as a fixed literal, so the test itself performs no optimization at run time."

requirements-completed: [D9-04, D9-06, D9-09, D9-11, D9-13, D9-14, D9-15, D9-17, D9-18]

coverage:
  - id: D1
    description: "Whole Phase 9 statistical path (aligned synthetic pair, shifted-alignment curve, 5-fold OOF ridge, one AE fit, anchor curvature, radial decomposition, 3-control partial, Freedman-Lane FWER null, JSONL record) runs end to end from one command on synthetic arrays"
    requirement: "D9-18"
    verification:
      - kind: automated_ui
        ref: ".venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py --mode smoke --record-path notebooks/.cache/09_scratch_tracer.jsonl"
        status: pass
    human_judgment: false
  - id: D2
    description: "Both assert_preregistered() freeze guards raise RuntimeError naming an unset constant; every Phase 9 gating constant stays UNSET in this plan"
    requirement: "D9-18"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py::test_assert_preregistered_rejects_unset_constant"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_labels.py::test_assert_preregistered_rejects_unset_constant"
        status: pass
    human_judgment: false
  - id: D3
    description: "controlled_partial reproduces the colleague's published raw (-0.4124) and 3-control-controlled (-0.2405) numbers to within 1e-3"
    requirement: "D9-09"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py::test_controlled_partial_reproduces_colleague_numbers"
        status: pass
    human_judgment: false
  - id: D4
    description: "decompose_radial_tangential satisfies the Pythagorean identity to 1e-10 relative and recovers H_rad within 10% of -d on an analytic d-sphere"
    requirement: "D9-11"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py::test_radial_decomposition_identity"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py::test_radial_decomposition_on_analytic_sphere"
        status: pass
    human_judgment: false
  - id: D5
    description: "Anchor draw is disjoint from AE training rows, sorted, duplicate-free, and stable across d/seed reuse"
    requirement: "D9-04"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py::test_anchor_indices_disjoint_and_sorted"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_curvature_probe.py::test_anchor_indices_stable_across_d"
        status: pass
    human_judgment: false
  - id: D6
    description: "Importing either new module in any of four distinct orders leaves all ten sealed pu_manifold modules byte-identical"
    requirement: "D9-18"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_import_purity.py::test_import_purity_holds_under_every_order"
        status: pass
    human_judgment: false

duration: ~55min
completed: 2026-09-02
status: complete
---

# Phase 9 Plan 1: Tracer Slice for Curvature-Conditioned Label Decodability Summary

**Two new pure modules (`physics_labels.py`, `physics_curvature_probe.py`) and a `--mode smoke` runner wire the entire Phase 9 statistical pipeline — shifted-alignment R2 curve, 5-fold OOF ridge, autoencoder fit, radial/tangential curvature decomposition, 3-control partial Spearman, Freedman-Lane null — end to end on synthetic arrays, with every one of 73+30 gating constants still declared UNSET.**

## Performance

- **Duration:** ~55 min
- **Completed:** 2026-09-02
- **Tasks:** 3 (all completed)
- **Files modified:** 6 created, 0 modified

## Accomplishments
- `notebooks/diagnostics/09_physics_curvature_run.py --mode smoke` runs the whole Phase 9 path in ~5.8s wallclock, exits 0, last line `SMOKE PASS`, and writes a 9-row JSONL record containing no `verdict`/`phase_verdict` key
- Both freeze guards (`physics_labels.assert_preregistered`, `physics_curvature_probe.assert_preregistered`) raise `RuntimeError` naming the first UNSET constant — no Physics number can be computed against this commit
- 94 new tests (test_physics_curvature_probe.py: 93 passed + 1 skipped ancestry placeholder; test_physics_labels.py: 39 passed; test_physics_import_purity.py: 6 passed), full `notebooks/pu_manifold/tests/` suite green at 854 passed / 2 skipped / 0 failures
- `controlled_partial` reproduces the colleague's published `-0.4124` raw / `-0.2405` controlled numbers (measured: `-0.412402` / `-0.240499`, both within 2e-6 of target) via a Gaussian-copula construction solved offline and replayed as a fixed literal
- `decompose_radial_tangential` recovers `H_rad` median exactly `-5.0` on an analytic d=5 sphere fixture (target `-d = -5`)

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end Phase 9 pipeline on synthetic data** - `98c6379` (feat)
2. **Task 2: Unit tests for physics_curvature_probe.py** - `a8ad929` (test, includes a Rule-1 bug fix to `plant_curvature_positive_control` found while writing its tests)
3. **Task 3: Unit tests for physics_labels.py and import-purity regression** - `cd7cca8` (test)

## Files Created/Modified
- `notebooks/pu_manifold/physics_labels.py` - Row-alignment loader/proof module: `mask_sentinels`, `canonical_label`, `shifted_pairing`, `alignment_r2_curve`, `alignment_verdict`, `assert_preregistered`; 30 UNSET constants
- `notebooks/pu_manifold/physics_curvature_probe.py` - Statistics module: `oof_ridge_predictions`, `anchor_indices`, `knn_panel`, `local_r2_panel`, `decompose_radial_tangential`, `controlled_partial`, `freedman_lane_y`, `p_value_from_null`, `permutation_fwer`, `stratified_partial_null_3control`, `paired_anchor_bootstrap`, `plant_curvature_positive_control`, `shuffled_label_repeat`, `per_d_verdict`, `phase_verdict`, `combine_seed_verdicts`, `verdict_sentence`, `assert_preregistered`; 73 required constants (72 UNSET, 1 pre-filled non-gating declaration)
- `notebooks/diagnostics/09_physics_curvature_run.py` - Runner with `--mode smoke` implemented; every other `--mode` value exits 2 naming the plan that implements it
- `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` - 21 named test functions (94 collected with the freeze-guard parametrization)
- `notebooks/pu_manifold/tests/test_physics_labels.py` - 10 named test functions (39 collected with the freeze-guard parametrization)
- `notebooks/pu_manifold/tests/test_physics_import_purity.py` - Phase-9-scoped sibling of `test_cka_import_purity.py`, reusing its subprocess-per-import-order mechanism

## Decisions Made
- The `alpha_grid` passed to the sealed `linear_probe.fit_probe` is `(float(alpha), float(alpha))` rather than a one-element tuple, working around a measured `sklearn==1.9.0` defect (see Deviations)
- `plant_curvature_positive_control`'s bisection direction is measured empirically rather than assumed, because the retargeted statistic can legitimately move in either direction depending on the empirical sign of the `h_real`-`y` relationship
- `SWISS_ROLL_APPLICABILITY_RULE` is filled immediately (non-gating declarative fact), unlike every other Phase 9 constant which stays UNSET until the 09-05 freeze
- The colleague-numbers parity test used the Gaussian-copula construction (not the transcribed-array fallback) — the offline-solved correlation matrix reproduces both target statistics to ~2e-6, well inside the 1e-3 tolerance

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Worked around a sklearn==1.9.0 RidgeCV defect on single-candidate alpha grids**
- **Found during:** Task 1, first `--mode smoke` run
- **Issue:** `sklearn.linear_model.RidgeCV.fit` mutates `self.alphas[0]` in place on its single-candidate fast path, which raises `TypeError: 'tuple' object does not support item assignment` when `alphas` is an immutable one-element tuple — exactly what the sealed `linear_probe.fit_probe` always constructs (`alpha_grid = tuple(float(a) for a in alpha_grid)`), regardless of what the caller passes in
- **Fix:** `oof_ridge_predictions` passes `alpha_grid=(float(alpha), float(alpha))` — a two-entry tuple whose entries are bit-identical, verified to produce bit-identical `coef_`/predictions to a genuine single-alpha `Ridge` fit (a candidate grid with one DISTINCT value cannot select anything but that value). `linear_probe.py` itself is not edited (sealed, D9-18)
- **Files modified:** `notebooks/pu_manifold/physics_curvature_probe.py`
- **Verification:** `--mode smoke` runs to completion; `test_oof_predictions_are_out_of_fold` and the colleague-parity test both pass
- **Committed in:** `98c6379` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed plant_curvature_positive_control's bisection direction assumption**
- **Found during:** Task 2, while writing `test_positive_control_hits_target_grid`
- **Issue:** The bisection copied from the sealed `plant_positive_control` assumes the achieved statistic increases with slope — true there because it always targets `spearmanr(h_real, planted)` directly. This phase's retargeted `controlled_partial(planted, y, Z)` decreases with slope whenever `h_real` and `y` are negatively associated (this phase's own D9-09 hypothesis), causing the bisection to converge to the wrong bracket endpoint for every target
- **Fix:** Direction is now measured empirically (achieved statistic at slope 0.0 vs 2.0) before bisecting; internal discretization reduced from 1000 to 10 so the achieved statistic stays a smooth, near-monotonic function of slope across the whole `[0.0, 2.0]` bracket instead of saturating within its first few percent
- **Files modified:** `notebooks/pu_manifold/physics_curvature_probe.py`
- **Verification:** `test_positive_control_hits_target_grid` — five targets each within 0.02 of the achieved controlled partial
- **Committed in:** `a8ad929` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bugs, both in `physics_curvature_probe.py`, both discovered by actually running the code rather than assumed correct from the plan's excerpted mechanism)
**Impact on plan:** Both fixes were necessary for correctness; neither changes any pre-registered constant, any sealed module, or any function's public contract. No scope creep.

## Issues Encountered
None beyond the two auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both new modules and the runner are ready for 09-02 (instrument fidelity at d=16) and 09-03 (real data-acquisition layer) to build on
- `assert_preregistered()` in both modules still raises — no Physics number exists anywhere in the tree, satisfying the D9-18 ordering this whole nine-plan phase depends on
- No blockers

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-02*

## Self-Check: PASSED

All 6 created files found on disk; all 3 task commits (`98c6379`, `a8ad929`, `cd7cca8`) found in
git history.
