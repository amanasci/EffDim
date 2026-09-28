---
phase: 07-curvature-conditioned-crossmodal-alignment
plan: 03
subsystem: research-instrumentation
tags: [positive-control, permutation-test, density, hubness, partial-correlation, pytest, tdd]

# Dependency graph
requires:
  - phase: 07-curvature-conditioned-crossmodal-alignment
    provides: "crossmodal_curvature.py's frozen constants block (POSITIVE_CONTROL_TARGET_RHOS, POSITIVE_CONTROL_SEED, POSITIVE_CONTROL_RULE, DENSITY_SIGN_CONVENTION), the freeze commit f032745f6450068c63763993d39fa112fd36bb8c (plan 07-01); split_indices, per_point_mknn, two_tailed_permutation_null, apply_verdict and the 07_crossmodal_curvature_run.py runner scaffold (plan 07-02)"
provides:
  - "crossmodal_curvature.plant_positive_control, crossmodal_curvature.smallest_cleared_target -- D7-02's positive control, licensing the phase to report a null at all"
  - "crossmodal_curvature.density_diagnostics -- D7-03's density and hubness diagnostics, reported and gating nothing"
  - "07_crossmodal_curvature_run.py --mode positive-control and --field-npz -- runnable, refuses to invent a field it does not have"
  - "17 new tests (140 -> 157) pinning both arms; --selfcheck extended to 9 checks"
affects: [07-04-plan, 07-05-plan]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Rank-invariant positive control: plant_positive_control rank-transforms h_real (scipy.stats.rankdata) before ever touching its raw values, so the planting mechanism is by construction invariant to h_real's magnitude/spread -- verified empirically in this plan's own wide-vs-narrow separation test, not merely asserted."
    - "Module-scoped pytest fixture sharing a single expensive plant_positive_control call (~9s at n=2000, N_PERMUTATIONS=1000 x 2 tails x 4 targets) across every assertion that only reads the result, rather than recomputing it per test -- kept the 157-test file at 22.7s, well under the plan's 60s ceiling."
    - "Density-confounded test fixture built by making LOCAL DENSITY itself (via quadratic point spacing ordered by a shared latent's rank) drive both h and m, rather than merely correlating three independent variables -- the only construction that actually separates partial_rho_density_controlled from partial_rho_raw."

key-files:
  created: []
  modified:
    - notebooks/pu_manifold/crossmodal_curvature.py
    - notebooks/pu_manifold/tests/test_crossmodal_curvature.py
    - notebooks/diagnostics/07_crossmodal_curvature_run.py

key-decisions:
  - "plant_positive_control's bisection keeps the 'high' endpoint of each 40-iteration bisection as the final slope (the endpoint whose achieved Spearman is >= target under the mechanism's monotonic-in-slope assumption) -- verified empirically monotonic across all tested h_real distributions; achieved_rho is recorded beside target_rho in every case, never substituted for it."
  - "plant_positive_control's result dicts merge the FULL two_tailed_permutation_null result in via **null_result (positive_tail, negative_tail, observed_rho, clears_either, direction all present at the top level) alongside target_rho/achieved_rho/slope/n_distinct/planted -- this is what lets smallest_cleared_target read result['clears_either'] directly, per the plan's stated behavior."
  - "The wide-spread-vs-narrow-spread separation test's MEASURED outcome (recorded per the plan's explicit instruction to record whichever way it falls): the two do NOT separate. Both a PU-matched narrow fixture (ratio ~1.47) and a Phase-6-matched wide fixture (rng.random(n), ratio ~20) recovered smallest_cleared_target == 0.10 at n=500, POSITIVE_CONTROL_SEED. This is explained, not merely observed: plant_positive_control rank-transforms h_real via scipy.stats.rankdata before any use, so its mechanism is invariant to h_real's raw magnitude/spread by construction -- only rank order enters the binomial draw. D7-02's objection to Phase 6's rng.random(n) selfcheck is therefore about WHAT was planted there (a noise term scaled by the field's own raw, unranked magnitude, so a wider raw spread mechanically produced a stronger signal-to-noise ratio), not about detectability specifically at PU's narrow dynamic range. The test asserts the measured value (0.10 for both) rather than a value chosen in advance."
  - "Density-diagnostics' confounded-fixture test could not be built by simply correlating three independent normal variables (an initial attempt using a rank-1 linear structure on X's position produced no measurable density-controlled separation, since a linear position shift does not change LOCAL point density). The fixture that actually separates partial_rho_density_controlled from partial_rho_raw places points at quadratically-spaced positions ordered by a shared latent's own rank -- this makes local k-NN distance itself, not merely position, a function of the latent."

requirements-completed: [D7-02, D7-03, D7-05]

coverage:
  - id: D1
    description: "plant_positive_control plants a curvature-MKNN relationship at PU's own realized ||H|| dynamic range (2,000-point fixture matched to PU's measured p95/p05 spread of 1.495), deterministically, with achieved_rho recorded beside target_rho, guarding constant/non-finite/too-short input before any search"
    requirement: "D7-02"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_crossmodal_curvature.py -- 10 new tests (test_plant_positive_control_*, test_smallest_cleared_target_positive_control_*), all pass"
        status: pass
      - kind: manual_procedural
        ref: "notebooks/diagnostics/07_crossmodal_curvature_run.py --mode positive-control run end to end against a synthetic 2,000-point field via --field-npz, writing 4 record rows and printing smallest_cleared_target: 0.05 -- confirmed this session, record file cleaned up after"
        status: pass
    human_judgment: false
  - id: D2
    description: "density_diagnostics reports the density partial, the density-vs-curvature Spearman, the density distribution and both columns' hubness skewness on the 1/w convention, gating nothing -- apply_verdict's two-parameter signature is structurally unable to accept a density number"
    requirement: "D7-03"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_crossmodal_curvature.py -- 6 new tests (test_density_diagnostics_*), all pass"
        status: pass
      - kind: other
        ref: "inspect.signature(cc.apply_verdict).parameters has exactly two entries, neither naming density -- pinned by test_density_diagnostics_never_reaches_apply_verdict"
        status: pass
    human_judgment: false
  - id: D3
    description: "No sealed module or frozen constant altered since the freeze commit; all crossmodal_curvature.py additions land strictly below the frozen constants block (pure additions, zero deletions since the freeze)"
    requirement: "D7-05"
    verification:
      - kind: other
        ref: "git diff --stat f032745..HEAD -- curvature_probe.py pointcloud_probe.py linear_probe.py mknn.py cae.py decoder_curvature.py cross_split_curvature.py src/effdim/ -- empty output"
        status: pass
      - kind: other
        ref: "git diff f032745..HEAD -- crossmodal_curvature.py -- 365 insertions(+), 0 deletions(-)"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -- full suite, 573 passed / 1 skipped (556 -> 573, delta matches the 17 new tests, no regressions), 182s"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-08-26
status: complete
---

# Phase 7 Plan 3: Positive Control and Density/Hubness Diagnostics Summary

**`plant_positive_control` plants a curvature-MKNN relationship at PU's own realized `d=20` `||H||` dynamic range and recovers targets within 0.02 of their achieved Spearman deterministically; `density_diagnostics` reports the `1/w` density partial and hubness skewness alongside every verdict while remaining structurally unable to reach `apply_verdict`; the wide-vs-narrow-spread separation test measured NO separation, and explains why: the planting mechanism is rank-invariant by construction, unlike Phase 6's magnitude-driven selfcheck it replaces.**

## Performance

- **Duration:** 30 min
- **Started:** 2026-08-26T08:52:00-04:00 (approx.)
- **Completed:** 2026-08-26T09:14:00-04:00
- **Tasks:** 3
- **Files modified:** 3 (`crossmodal_curvature.py`, `test_crossmodal_curvature.py`, `07_crossmodal_curvature_run.py`)

## Accomplishments

- **Task 1 -- `plant_positive_control(h_real, k, target_rhos, seed)` and `smallest_cleared_target(results)`.** Implements `POSITIVE_CONTROL_RULE` exactly: rank-transforms `h_real` once (`u = (rankdata(h_real) - 0.5) / n`), bisects a candidate `slope` over 40 iterations on `[0.0, 2.0]` against the achieved Spearman of a binomial-draw planted array (`j ~ Binomial(k, p)`, re-seeded to `seed` at every trial via a `_planted_array` helper, so the whole search is deterministic), and runs every planted pair through `two_tailed_permutation_null` using the frozen `N_PERMUTATIONS`/`PERMUTATION_SEED`/`NULL_QUANTILE_PER_TAIL` passed explicitly at the call site. Guards first: raises `ValueError` naming `h_real` when constant, non-finite, or fewer than `k + 2` rows. On a synthetic 2,000-point fixture matched to PU's measured p95/p05 spread of 1.495 (lognormal, sigma=0.12, ratio=1.490), all four `POSITIVE_CONTROL_TARGET_RHOS` recovered `achieved_rho` within 0.003 of target (well inside the 0.02 tolerance), same sign, planted arrays exactly `j/20`-discretized with at most 16 distinct values (<= k+1=21). `smallest_cleared_target` returned `0.05` on that fixture.
- **Task 2 -- `density_diagnostics(X_ambient, h, m, z_a, z_b, k, density_k, density_field_d)`.** Computes `density = 1.0 / w` (`curvature_probe.local_density_weights` returns the INVERSE density), matching Phase 4's REGN-01 sign convention. Reports `spearman_density_vs_h`, `spearman_density_vs_mknn`, `partial_rho_raw` and `partial_rho_density_controlled` (both via `cross_split_curvature.partial_spearman`, reused unchanged -- no local residualize-and-correlate routine written), the density p05/p50/p95 and ratio, `hubness_skewness_a`/`_b` (`mknn.hubness_skewness`), and `chance_floor` (`mknn.chance_floor`). Every value coerced to a plain `float`. `partial_rho_raw` matches raw `scipy.stats.spearmanr` to `rel=1e-6` on a tie-free fixture; on a fixture where a shared latent drives both local point density (via quadratically-spaced positions ordered by the latent's rank) and `h`/`m`, `abs(partial_rho_density_controlled) < abs(partial_rho_raw)` (0.9467 < 0.9968); on an independent fixture the two agree to within 0.0004. `apply_verdict`'s signature is untouched -- exactly two parameters, neither naming density -- so the non-gating property stays structural.
- **Task 3 -- `--mode positive-control`, `--field-npz`, extended `--selfcheck`, and the D7-02-vs-Phase-6 separation measurement.** `run_positive_control` calls `assert_preregistered()` first, requires `--field-npz` (resolved through `cache._assert_inside_cache`, T-07-01), refuses to invent a field -- `--mode positive-control` with no `--field-npz` exits 1 naming plan 07-04 and the missing field. Confirmed end to end against a synthetic 2,000-point `.npz` field: 4 record rows written (`preregistration_commit`/`run_commit` per T-07-03), printed `smallest_cleared_target: 0.05`, scratch files cleaned up after. `--selfcheck` gained 3 assertions (j/k discretization, planter determinism, `partial_spearman` vs. raw Spearman agreement) -- 9/9 pass, tally sums to total. The wide-spread-vs-narrow-spread separation test (see Key Decisions) measured NO separation and records why.
- **17 new tests** (`test_crossmodal_curvature.py`, 140 -> 157): 10 for `plant_positive_control`/`smallest_cleared_target`, 6 for `density_diagnostics`, 1 for the separation measurement. Full file: 157 passed in 22.7s (well under the plan's 60s ceiling; the expensive full-permutation calls -- ~9s at n=2000, ~4-5s at n=300-500 -- are deliberately minimized and shared via a module-scoped pytest fixture for the 2,000-point behavior tests).
- **Full regression confirmed:** `notebooks/pu_manifold/tests/` -- 573 passed, 1 skipped (up from the pre-plan baseline of 556 passed, 1 skipped; the +17 delta matches exactly the 17 new tests), 182s, zero regressions.
- **No sealed module touched, zero deletions since the freeze:** `git diff --stat f032745..HEAD` across `curvature_probe.py`, `pointcloud_probe.py`, `linear_probe.py`, `mknn.py`, `cae.py`, `decoder_curvature.py`, `cross_split_curvature.py`, `src/effdim/` is empty for the whole plan; `git diff f032745..HEAD -- crossmodal_curvature.py` shows 365 insertions, 0 deletions.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build the D7-02 positive control at PU's own realized dynamic range** -- `65ba5d6` (test)
2. **Task 2: Wire the D7-03 density and hubness diagnostics, gating nothing** -- `082624d` (feat)
3. **Task 3: Pin both arms with tests and extend the runner's selfcheck** -- `7b691fc` (feat)

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `notebooks/pu_manifold/crossmodal_curvature.py` -- added `_relative_precision_distinct_count`, `_planted_array`, `plant_positive_control`, `smallest_cleared_target` (Task 1) and `density_diagnostics` (Task 2), plus the import additions (`List`, `scipy.stats.rankdata`/`spearmanr`, `cross_split_curvature`) each function needed, all below the freeze block. 733 lines total (505 -> 733).
- `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` -- 17 new tests across three clearly marked sections (`# Plan 07-03, Task 1/2/3`), plus a module-scoped `_pu_matched_positive_control` fixture shared across the Task 1 behavior tests. 748 lines total (471 -> 748).
- `notebooks/diagnostics/07_crossmodal_curvature_run.py` -- added `run_positive_control`, the `--field-npz` CLI flag, wired `--mode positive-control` to the real implementation (no longer raising `NotImplementedError`), extended `--selfcheck` with 3 assertions, updated the module docstring. 492 lines total (358 -> 492).

## Decisions Made

See `key-decisions` in frontmatter above. In short: the bisection keeps the "high" endpoint as the final slope; `plant_positive_control`'s result dicts merge the full `two_tailed_permutation_null` result in via `**null_result` so `smallest_cleared_target` can read `clears_either` directly at the top level; the wide-vs-narrow separation test's measured outcome is NO separation, explained by the mechanism's rank-invariance; and the density-confounded test fixture required making local point density itself (not merely position) a function of the shared latent to actually demonstrate separation.

## Deviations from Plan

**1. [Minor, administrative] Task 1's `<files>` tag listed only `crossmodal_curvature.py`, but its own `<verify>` step (`pytest -k positive_control`) requires matching tests to exist to pass (an empty `-k` selection exits pytest code 5, not 0).** Added the corresponding tests to `test_crossmodal_curvature.py` within Task 1's commit rather than deferring all test-writing to Task 3, mirroring how the plan's own Task 3 acceptance criteria implicitly require coverage to exist by the time each task's `<verify>` runs (Task 2's `<verify>` is similarly `-k density`). Task 3 then added only the tests specific to its own new behavior (the runner wiring and the separation measurement) rather than re-deriving coverage Tasks 1-2 had already written. No behavior, constant, or acceptance criterion changed as a result -- this is a task-commit-boundary detail, not a functional deviation.

**2. [Rule 3-adjacent, not a bug] The plan's action text for Task 1 describes the permutation parameters as "passed in explicitly by the caller" without listing them in `plant_positive_control`'s own 4-argument signature (`h_real, k, target_rhos, seed`), which the plan's own acceptance-criteria invocation (`plant_positive_control(h_real, 20, (0.02, 0.05, 0.10, 0.20), 20260825)`) confirms is exactly 4 arguments.** Resolved by reading "passed explicitly" as describing the function's OWN internal call site to `two_tailed_permutation_null` (which itself takes no defaults) using the frozen `N_PERMUTATIONS`/`PERMUTATION_SEED`/`NULL_QUANTILE_PER_TAIL` module constants, not as requiring extra parameters on `plant_positive_control` itself. This reading is the only one consistent with the plan's own acceptance-criteria call signature.

No other deviations -- both arms' behaviors, guards, docstrings and non-gating structure match the plan's `<behavior>` and `<action>` sections as written.

## Issues Encountered

None. The full `notebooks/pu_manifold/tests/` suite takes ~3 minutes (573 tests, mostly torch-backed); this was run once as a final regression check after Task 3's commit and is not part of any individual task's own `<verify>` step.

## Known Stubs

None. `--mode dsweep` still raises `NotImplementedError` naming plan 07-04, unchanged by this plan and matching its own stated scope boundary (this plan's scope is D7-02 and D7-03 only).

## Threat Flags

None beyond the plan's own pre-registered threat model. `T-07-01` (`--field-npz` path traversal) is mitigated via `cache._assert_inside_cache`, confirmed by the manual end-to-end run this session. `T-07-02` (frozen-constant tampering) and `T-07-05` (a degenerate positive control) are both mitigated as designed: `assert_preregistered()` runs first in `run_positive_control`, and `plant_positive_control`'s three guards (constant/non-finite/too-few-rows) raise before any search, pinned by 3 dedicated tests. `T-07-03` (record repudiation) is mitigated -- every appended row carries `preregistration_commit` and `run_commit`, confirmed in the manual end-to-end run.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

Plan 07-04 (the real `d`-sweep) may now begin: `plant_positive_control`, `smallest_cleared_target` and `density_diagnostics` are proven on synthetic fixtures matched to PU's own measured spread and pinned by 17 tests. `--mode positive-control` is runnable end to end once plan 07-04 writes a real `d=20` `.npz` field carrying an `h_norm` array to `notebooks/.cache/` and its path is passed via `--field-npz` -- this plan deliberately does not write that field itself, and `run_positive_control` refuses to regenerate one if it is missing. No PU number has been written to the frozen record anywhere in the tree by this plan; the manual end-to-end verification used a synthetic field and a scratch record path, both deleted after the run. `notebooks/.cache/07_crossmodal_curvature.jsonl` still does not exist.

---
*Phase: 07-curvature-conditioned-crossmodal-alignment*
*Completed: 2026-08-26*

## Self-Check: PASSED

- FOUND: `notebooks/pu_manifold/crossmodal_curvature.py`
- FOUND: `notebooks/pu_manifold/tests/test_crossmodal_curvature.py`
- FOUND: `notebooks/diagnostics/07_crossmodal_curvature_run.py`
- FOUND: `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-03-SUMMARY.md`
- FOUND commit `65ba5d6` in `git log --oneline --all`
- FOUND commit `082624d` in `git log --oneline --all`
- FOUND commit `7b691fc` in `git log --oneline --all`
