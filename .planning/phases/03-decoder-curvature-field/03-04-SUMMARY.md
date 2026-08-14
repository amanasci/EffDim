---
phase: 03-decoder-curvature-field
plan: 04
subsystem: testing
tags: [numpy, curvature, synthetic-fixtures, finite-differences, pytest]

# Dependency graph
requires:
  - phase: 02.5-local-curvature-feasibility-cae-re-gate
    provides: curvature_probe.py (graph_mean_curvature, make_flat_fixture, make_graph_of_function_fixture) -- sealed, imported unmodified
  - phase: 03-01
    provides: chart_curvature.py's CURVATURE_CONVENTION constant, ratified as the phase's trace convention
provides:
  - "synthetic_controls.py with three known-geometry fixtures (flat, sphere, saddle) constructing at PU scale (d=20, D=768)"
  - "make_flat_control -- thin delegate to curvature_probe.make_flat_fixture, analytic H_norm exactly zero"
  - "make_sphere_control -- analytic ||H|| = d/R derived from the d-sphere's own second fundamental form, pinned against d not 1 not (d+2)/d"
  - "make_saddle_control -- mixed-sign quadratic saddle, hand-computed grad/hess fed to curvature_probe.graph_mean_curvature unmodified, cross-checked against an independent finite-difference computation"
  - "rotate_and_pad -- shared zero-pad / one-fixed-rotation / centre-and-rescale embedding step reused by all three controls"
affects: [03-05 (Swiss roll chart-curvature-field notebook), 03-08/03-09 (PU curvature field runner and notebook), any plan consuming step 4's ground truth]

# Tech tracking
tech-stack:
  added: []
  patterns: ["own CURVATURE_CONVENTION constant declared and asserted equal to every sealed module's own constant at import time, so a future drift in either sealed module breaks this module's import instead of silently propagating a factor-of-d error (decoder_curvature.py's pattern, reused verbatim)"]

key-files:
  created:
    - notebooks/pu_manifold/synthetic_controls.py
    - notebooks/pu_manifold/tests/test_synthetic_controls.py
  modified: []

key-decisions:
  - "make_sphere_control derives H_local = -(d/R^2) * X_local analytically from the d-sphere's own second fundamental form rather than routing through graph_mean_curvature -- a sphere has no single graph-of-function parametrization over all d dimensions, so the closed-form derivation (matching test_decoder_curvature.py's _SphereDecoder precedent) is the correct approach, not a shortcut"
  - "make_saddle_control's finite-difference cross-check test is self-contained -- it draws its own signs and domain points with a fixed seed rather than reaching into make_saddle_control's return dict, so the test exercises the grad/hess formula directly rather than coupling to the wrapper's exact RNG stream"
  - "rotate_and_pad's Q is drawn from a fresh np.random.default_rng(seed) using the same seed value passed to the calling control's own point-sampling rng -- two independent rng objects seeded identically, not a shared stream; no numerical coupling, matches the plan's literal instruction"

requirements-completed: [CURV-06]

coverage:
  - id: D1
    description: "synthetic_controls.py declares its own CURVATURE_CONVENTION = 'trace' and raises ValueError at import time if it disagrees with chart_curvature.CURVATURE_CONVENTION or curvature_probe.CURVATURE_CONVENTION"
    requirement: "CURV-06"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_controls_convention_agrees_with_sealed_modules"
        status: pass
    human_judgment: false
  - id: D2
    description: "make_flat_control delegates to curvature_probe.make_flat_fixture; H_norm is exactly 0.0 at every point, no tolerance"
    requirement: "CURV-06"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_flat_control_is_exactly_zero"
        status: pass
    human_judgment: false
  - id: D3
    description: "make_sphere_control returns H_norm/global_std = d/R to 1e-12 at both (d=4, R=1.0) and (d=20, R=2.0), and the value is explicitly not 1.0 and not (d+2)/d"
    requirement: "CURV-06"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_sphere_control_matches_d_over_R"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_controls_convention_is_trace_not_averaged"
        status: pass
    human_judgment: false
  - id: D4
    description: "All three controls construct at d=20, D=768, n=200 in well under 10 seconds each (measured 0.07-0.12s), materializing no (D,D) array beyond the one fixed rotation matrix"
    requirement: "CURV-06"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_controls_construct_at_pu_scale"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_saddle_control_constructs_at_pu_scale"
        status: pass
    human_judgment: false
  - id: D5
    description: "make_saddle_control's hand-computed grad/hess arrays are cross-checked against an independent central-finite-difference computation of the same quadratic (rtol=1e-8), closing 03-RESEARCH.md Assumption A2"
    requirement: "CURV-06"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_saddle_control_matches_finite_difference"
        status: pass
    human_judgment: false
  - id: D6
    description: "The saddle's field genuinely varies -- coefficient of variation > 0.05 and H_norm.min() at least 10x below H_norm.max() -- so it exercises trace-cancellation, not merely near-flatness"
    requirement: "CURV-06"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_synthetic_controls.py#test_synthetic_saddle_control_field_genuinely_varies"
        status: pass
    human_judgment: false
  - id: D7
    description: "curvature_probe.py and cae.py are not edited by this plan; synthetic_controls.py writes no new curvature formula except the saddle's own grad/hess (cross-checked in D5)"
    requirement: "CURV-06"
    verification:
      - kind: other
        ref: "git diff --stat notebooks/pu_manifold/curvature_probe.py notebooks/pu_manifold/cae.py -- empty"
        status: pass
    human_judgment: false

# Metrics
duration: ~15min
completed: 2026-08-14
status: complete
---

# Phase 03 Plan 04: Synthetic Control Fixtures Summary

**Three known-geometry manifolds (flat, sphere, saddle) with analytic mean curvature at PU's actual scale (d=20, D=768), giving step 4 ground truth it can cross-check against rather than assume.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-08-14T14:20:00Z (approx, from prior plan's STATE.md timestamp)
- **Completed:** 2026-08-14T14:33:39Z
- **Tasks:** 2
- **Files modified:** 2 (both new)

## Accomplishments

- `notebooks/pu_manifold/synthetic_controls.py` created with an import-time `CURVATURE_CONVENTION = "trace"` guard that raises `ValueError` if it disagrees with `chart_curvature.CURVATURE_CONVENTION` or `curvature_probe.CURVATURE_CONVENTION`
- `rotate_and_pad` factored out as the shared embedding step (zero-pad, one fixed random orthogonal rotation, centre-and-rescale by one global scalar std) reused by all three controls
- `make_flat_control` delegates outright to the sealed `curvature_probe.make_flat_fixture` -- analytic `H_norm` exactly `0.0`, no tolerance
- `make_sphere_control` derives `||H|| = d/R` analytically from the `d`-sphere's own second fundamental form, pinned by regression tests at `d=4` and `d=20` that also explicitly reject `1.0` and the averaged `(d+2)/d`
- `make_saddle_control` builds a mixed-sign quadratic (`f(x) = 0.5 x^T diag(signs) x`) whose hand-computed `grad`/`hess` arrays are fed to the sealed `curvature_probe.graph_mean_curvature` unmodified -- no new curvature formula written -- and cross-checked against an independent central-finite-difference computation to `rtol=1e-8` before being trusted as ground truth (closing `03-RESEARCH.md` Assumption A2)
- 8 tests added, all passing; full `pu_manifold` suite green at 280/280 (was 272/272 before this plan)

## Task Commits

Each task was committed atomically:

1. **Task 1: synthetic_controls.py -- convention guard, rotate-and-pad, flat and sphere controls** - `9406bd4` (feat)
2. **Task 2: Saddle control and its finite-difference cross-check** - `0621f89` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `notebooks/pu_manifold/synthetic_controls.py` - three known-geometry control fixtures (flat, sphere, saddle) at matched `d`/`D`, with the import-time convention guard and shared `rotate_and_pad` embedding helper
- `notebooks/pu_manifold/tests/test_synthetic_controls.py` - 8 tests: convention agreement, flat exact-zero, sphere `d/R` at two scales, trace-vs-averaged regression guard, PU-scale construction (flat/sphere and saddle separately), the saddle finite-difference cross-check, and the saddle's genuine-variation check

## Decisions Made

- `make_sphere_control` derives its analytic `H` in closed form rather than routing through `graph_mean_curvature`, because a sphere has no single graph-of-function parametrization spanning all `d` dimensions -- matches the precedent `test_decoder_curvature.py`'s `_SphereDecoder` sets for known-geometry closed forms.
- The saddle's finite-difference cross-check test is self-contained (own seed, own signs, own domain points) rather than reaching into `make_saddle_control`'s return dict, so it validates the `grad`/`hess` formula itself rather than coupling to the wrapper's exact RNG behavior.
- `rotate_and_pad` instantiates its own `np.random.default_rng(seed)` for the rotation matrix `Q`, independent of whatever rng the calling control used for point sampling (even when both are seeded with the same integer) -- two separate deterministic streams, no numerical coupling, matching the plan's literal instruction.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' `<behavior>`, `<action>`, and `<verify>` blocks were implemented as specified; all acceptance criteria (import-time convention print, `CURVATURE_CONVENTION`/`make_flat_fixture`/`graph_mean_curvature` grep counts, empty `git diff --stat` on `curvature_probe.py` and `cae.py`, full-suite green) were checked directly and pass.

## Issues Encountered

None. All 8 new tests passed on first run; the finite-difference cross-check's implicit precision concern (division by `step**2 = 1e-8` amplifying float64 rounding in the off-diagonal Hessian terms, which are analytically exactly zero for this diagonal-only quadratic) was anticipated and handled with `atol=1e-6` alongside `rtol=1e-8` in the `assert_allclose` call -- no separate fix needed, this was accounted for during writing, not discovered as a failure.

## Next Phase Readiness

- Step 4's ground truth exists and is verified: flat/sphere/saddle at PU scale, all three cross-checked (delegation for flat, closed-form derivation with a rejecting regression test for sphere, independent finite-difference agreement for saddle).
- **Carried limitation, stated in the module docstring itself (not only here):** a synthetic manifold sampled cleanly and fitted under this protocol trains to a clean, unfragmented atlas. It has none of `02.5-09`'s chart-count-driven fragmentation or the `cond(g)` blow-up to `122.22` the phase's gate override is worried about. A control that passes on these fixtures establishes only that the decoder-curvature *pipeline* is correct on an easy-to-fit manifold -- it is necessary, not sufficient, and MUST NOT be presented as evidence the PU curvature field is free of parameterization damage. This travels with every consumer of `synthetic_controls.py`, not just this SUMMARY.
- `cond(g)` is not yet computed or reported anywhere in this module -- the fixtures return `H_vec`/`H_norm`/`global_std` only. Any downstream plan (03-08/03-09) that needs to distinguish a genuine near-zero `||H||` from a near-singular pullback metric on these fixtures will need to compute `cond(g)` itself, consistent with the standing prohibition carried from `02.5-09`'s `n_charts=5, seed=0` finding (`cond_max = 37770.88` alongside `rho_chart = 0.8469`).
- No blockers for `03-05` (the Swiss roll chart-curvature-field notebook) or later plans that consume these fixtures.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-14*
