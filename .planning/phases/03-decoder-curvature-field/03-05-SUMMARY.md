---
phase: 03-decoder-curvature-field
plan: 05
subsystem: curvature
tags: [torch.func, jacfwd, jacrev, autodiff, vmap, forward-mode, pytest]

# Dependency graph
requires:
  - phase: 02.5-local-curvature-feasibility-cae-re-gate
    provides: chart_curvature.py (chart_mean_curvature, chart_curvature_field), sealed and previously untouched since 02.5-09
  - phase: 03-02
    provides: the reproduced rho_chart=-0.0604 roll anchor at n_charts=8, seed=0, the regression target this plan proves the edit does not move
affects: [any future plan that calls chart_mean_curvature or chart_curvature_field with mode="forward" for a measured wall-clock win at PU scale]

# Tech tracking
tech-stack:
  added: []
  patterns: ["mode dispatch factored into a single module-private _jacobian_hessian(decode_one, chunk, mode) function -- the only thing that branches on mode; everything downstream of the (J, Hess) pair it returns stays untouched, so an equivalence proof only needs to compare that one function's two branches rather than re-verify the whole pipeline twice"]

key-files:
  created: []
  modified:
    - notebooks/pu_manifold/chart_curvature.py
    - notebooks/pu_manifold/tests/test_curvature_probe.py

key-decisions:
  - "The forward-Hessian composition is the PRIMARY jacfwd(jacfwd(f)), not the documented jacfwd(jacrev(f)) fallback -- Task 1's spike ran it against the real cae.ChartAutoEncoder chart-decoder architecture (chart_dim=20, out_dim=768, hidden=[250,250,250], silu, float64) and it completed without a RuntimeError, returning the expected (batch, out_dim, chart_dim, chart_dim) shape. The fallback was never needed."
  - "mode stays add-alongside, not promoted: chart_mean_curvature and chart_curvature_field both default to mode=\"reverse\", exactly as ratified in the plan's assumption_delta_decision. No call site in this codebase was changed to pass mode=\"forward\" -- the toggle exists to be selected explicitly by a future caller, not to change any existing behavior."
  - "The golden-value pin (test_chart_curvature_reverse_mode_is_bit_identical_to_sealed_baseline) was written and committed BEFORE chart_curvature.py was touched, against the unmodified module -- so a torch.equal comparison after the edit is a genuine before/after proof, not a value transcribed from the edited code."

requirements-completed: [CURV-01, CURV-02]

coverage:
  - id: D1
    description: "vmap(jacfwd(jacfwd(decode_one))) was spiked against the real cae.ChartAutoEncoder chart-decoder architecture (chart_dim=20, out_dim=768, hidden=[250,250,250], silu) and completed without raising, with the expected (4,768,20,20) Hessian shape and (4,768,20) Jacobian shape; a single-chunk (32-row) Hessian measured ~6.08s reverse vs ~0.26s forward, a ~23.6x wall-clock speedup"
    verification:
      - kind: other
        ref: "scratch spike script run against .venv, output transcribed below in Task 1 Spike Transcript"
        status: pass
    human_judgment: false
  - id: D2
    description: "chart_mean_curvature and chart_curvature_field accept mode with default \"reverse\"; an unknown mode string raises ValueError naming the offending value on both entry points"
    requirement: "CURV-01"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_curvature_probe.py#test_chart_curvature_rejects_unknown_mode"
        status: pass
    human_judgment: false
  - id: D3
    description: "The reverse path is bit-identical to its pre-edit output -- a torch.equal golden-array test written and committed before the edit, and the anchor n_charts=8 seed=0 roll cell reproduces rho_chart=-0.06041003026778113 byte-identically post-edit"
    requirement: "CURV-01"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_curvature_probe.py#test_chart_curvature_reverse_mode_is_bit_identical_to_sealed_baseline"
        status: pass
      - kind: other
        ref: ".venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --n-charts 8 --seeds 0 --max-combos 1 --record-path /tmp/03-05-reverse-check.jsonl -- rho_chart matched notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl's sealed value exactly"
        status: pass
    human_judgment: false
  - id: D4
    description: "Forward and reverse agree to float64 round-off (rtol=1e-9, atol=1e-12) on H_vec, H_norm and metric_condition_number at both the per-chart and the field level; jacobian_shape/hessian_shape are equal between modes; both modes agree on chart_assignment and n_charts_used at the field level"
    requirement: "CURV-02"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_curvature_probe.py#test_chart_curvature_forward_mode_matches_reverse_to_float64_round_off"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_curvature_probe.py#test_chart_curvature_field_forward_mode_matches_reverse_across_charts"
        status: pass
    human_judgment: false
  - id: D5
    description: "Both exact-tuple shape assertions (Jacobian and Hessian) fire identically on the forward path -- a wrong torch.func composition would still run and silently return a differently-shaped tensor"
    requirement: "CURV-02"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_curvature_probe.py#test_chart_curvature_forward_mode_keeps_shape_assertions"
        status: pass
    human_judgment: false
  - id: D6
    description: "The C2-activation guard is reached on the forward path exactly as on reverse -- a ReLU-family decoder raises rather than returning an identically-zero second fundamental form"
    requirement: "CURV-02"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_curvature_probe.py#test_chart_curvature_forward_mode_calls_c2_guard"
        status: pass
    human_judgment: false
  - id: D7
    description: "The trace-first-then-project block (g-trace, d-by-d solve, normal projection) is unchanged byte-for-byte; decoder_curvature.py and derivative_bridge.py are untouched; forward mode is NOT the default before or after equivalence passes"
    verification:
      - kind: other
        ref: "git diff shows no change between the g = torch.einsum(...) line and the H_parts.append(...) line; git diff --stat on decoder_curvature.py and derivative_bridge.py is empty; grep -c 'mode: str = \"reverse\"' returns 2"
        status: pass
    human_judgment: false

# Metrics
duration: ~15min
completed: 2026-08-14
status: complete
---

# Phase 03 Plan 05: Forward-Mode Curvature Differentiation Toggle Summary

**Opt-in `mode="forward"` on `chart_mean_curvature`/`chart_curvature_field`, proved equal to the existing `jacrev`-based reverse path to float64 round-off, with `jacfwd(jacfwd(f))` -- the primary composition, no fallback needed -- measuring a ~23.6x single-chunk wall-clock speedup on the real decoder architecture.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-08-14T14:36:00Z (approx, from prior plan's STATE.md timestamp)
- **Completed:** 2026-08-14T14:46:27Z
- **Tasks:** 3
- **Files modified:** 2 (both pre-existing, edited in place)

## Task 1 Spike Transcript

Both calls run against a real `cae.ChartAutoEncoder` at `in_dim=768, embed_dim=40, chart_dim=20,
n_charts=4, hidden=[250, 250, 250], activation="silu"`, cast `.double()`, `decode_one =
chart_decoder_map(model, 0)`, on a batch of 4 float64 rows:

```
=== Jacobian: vmap(jacfwd(decode_one)) ===
OK, shape: (4, 768, 20)

=== Hessian: vmap(jacfwd(jacfwd(decode_one))) ===
OK, shape: (4, 768, 20, 20)

=== Hessian fallback: vmap(jacfwd(jacrev(decode_one))) ===
OK, shape: (4, 768, 20, 20)
```

Neither call raised a `RuntimeError`. Both Hessian compositions were exercised for completeness;
since the primary `jacfwd(jacfwd(f))` succeeded, the documented `jacfwd(jacrev(f))` fallback was
not needed and is not used anywhere in `chart_curvature.py`.

**Timing** (single 32-row chunk Hessian, warmed up, mean of 5 repetitions, same architecture):

```
reverse (hessian=jacfwd(jacrev)) mean wall time over 5 reps: 6.0816s
forward (jacfwd(jacfwd)) mean wall time over 5 reps: 0.2580s
```

~23.6x wall-clock speedup -- a real, substantial, measured win, well short of the ~38x
operation-count ceiling (expected: PyTorch's forward-mode path is documented as less optimized
than reverse, and `vmap` over dual numbers carries its own constants). This is the first time
forward mode has been timed anywhere in this milestone.

**Decision:** the forward branch's Hessian composition is `jacfwd(jacfwd(f))`, the primary
composition, not the fallback.

## Accomplishments

- `CURVATURE_MODES = ("reverse", "forward")` added as a module constant on `chart_curvature.py`, with a docstring stating reverse is the default and stays the default
- `_jacobian_hessian(decode_one, chunk, mode)` factored out as the single dispatch point -- reverse moved verbatim (`vmap(jacrev(...))` / `vmap(hessian(...))`), forward added (`vmap(jacfwd(...))` / `vmap(jacfwd(jacfwd(...)))`), unknown mode raises `ValueError` naming the value
- `mode: str = "reverse"` threaded through `chart_mean_curvature` and `chart_curvature_field`; `"mode"` added to `chart_mean_curvature`'s returned provenance dict
- Both exact-tuple shape assertions kept unchanged and reachable on both branches; the `g`-trace-first, `d`-by-`d`-solve, normal-project block downstream of `(J, Hess)` is untouched byte-for-byte -- confirmed by `git diff` showing no change between the `g = torch.einsum(...)` line and the `H_parts.append(...)` line
- `test_chart_curvature_reverse_mode_is_bit_identical_to_sealed_baseline` written and committed against the unmodified module, then reconfirmed passing after the edit -- `torch.equal`, not a tolerance
- The `n_charts=8, seed=0` roll anchor re-run against a scratch record path reproduces `rho_chart=-0.06041003026778113` byte-identically against `notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl`'s sealed value
- 5 new equivalence/guard/rejection tests added (forward-vs-reverse per-chart, forward-vs-reverse field-level, shape assertions on forward, C2 guard on forward, unknown-mode rejection on both entry points); full `pu_manifold` suite green at 286/286 (was 280/280 before this plan)

## Task Commits

Each task was committed atomically:

1. **Task 1: Spike jacfwd-over-jacfwd, pin pre-edit golden values** - `823723b` (test)
2. **Task 2: Implement the mode toggle over Jacobian/Hessian construction only** - `fc40492` (feat)
3. **Task 3: D-09 equivalence, unknown-mode rejection, guards on the forward path** - `cdef570` (test)

**Plan metadata:** (this commit)

## Files Created/Modified

- `notebooks/pu_manifold/chart_curvature.py` - `CURVATURE_MODES` constant, `_jacobian_hessian` dispatch helper, `mode` keyword on both public entry points, `"mode"` provenance key
- `notebooks/pu_manifold/tests/test_curvature_probe.py` - 6 new tests: the pre-edit golden-value pin, forward/reverse per-chart equivalence, forward/reverse field-level equivalence, forward-path shape assertions, forward-path C2 guard, unknown-mode rejection

## Decisions Made

- Forward-Hessian composition is the primary `jacfwd(jacfwd(f))`, established empirically by Task 1's spike against the real architecture rather than assumed from research's caution -- the documented `jacfwd(jacrev(f))` fallback exists in `_jacobian_hessian`'s docstring as a documented alternative but is not used.
- `mode` stays `add-alongside` per the plan's `assumption_delta_decision`: no existing call site was changed to pass `mode="forward"`. Reverse is the default before and after this plan, matching the explicit user instruction recorded in `03-CONTEXT.md`.
- The golden-value test was written and committed in Task 1, strictly before any edit to `chart_curvature.py`, so its later re-pass in Task 2/3 is a genuine before/after proof rather than a value derived from the already-edited code.

## Deviations from Plan

None - plan executed exactly as written. One planning-artifact note, not a deviation: the plan's
Task 1 acceptance criterion `.venv/bin/python -m pytest ... -k bit_identical` "collects exactly 1
test" undercounted -- two pre-existing tests (`test_centroid_mean_curvature_both_densities_is_bit_identical`,
`test_measure_cell_precomputed_h_vec_is_bit_identical_and_shares_the_pass`) already match that
keyword, so the actual collection is 3, all passing. The new test's own identity
(`test_chart_curvature_reverse_mode_is_bit_identical_to_sealed_baseline`) is unambiguous and its
assertion is exactly as specified; only the keyword-search count in the plan's acceptance
criteria was imprecise.

## Issues Encountered

None. The spike (Task 1A) succeeded on the primary composition on the first attempt, so no
fallback investigation was needed. All new tests passed on first run.

## Next Phase Readiness

- `chart_curvature.py`'s forward-mode toggle exists, is proved equal to reverse at float64
  round-off, and is proved to leave every existing caller's behavior untouched. Any future PU-scale
  plan that wants the measured ~23.6x single-chunk speedup can opt in with `mode="forward"`
  without touching this plan's tests or the reverse path.
- The ~23.6x figure is a single-chunk, single-architecture measurement (`chart_dim=20,
  out_dim=768, hidden=[250,250,250]`) on a CPU. It has not been measured across a full field
  computation (many chunks, real PU point counts) or against `VMAP_CHUNK`'s bit-reproducibility
  guarantee under forward mode specifically -- the field-level equivalence test in this plan
  proves correctness at `batch_size=8`/`24` rows on a toy fixture, not at PU scale. A later plan
  adopting `mode="forward"` for a real run should re-measure wall-clock at that scale before
  relying on the ~23.6x figure.
- No blockers for `03-06` or later plans.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-14*

## Self-Check: PASSED

- FOUND: `notebooks/pu_manifold/chart_curvature.py`
- FOUND: `notebooks/pu_manifold/tests/test_curvature_probe.py`
- FOUND: `.planning/phases/03-decoder-curvature-field/03-05-SUMMARY.md`
- FOUND: commit `823723b` (Task 1)
- FOUND: commit `fc40492` (Task 2)
- FOUND: commit `cdef570` (Task 3)
