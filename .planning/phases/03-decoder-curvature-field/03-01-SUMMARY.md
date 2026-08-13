---
phase: 03-decoder-curvature-field
plan: 01
subsystem: research-notebook
tags: [swiss-roll, chart-autoencoder, curvature, torch.func, sweep-runner, d-02, d-05]

# Dependency graph
requires:
  - phase: 02.5-08
    provides: "chart_curvature.py -- chart_curvature_field, chart_mean_curvature, assert_c2_activation, curvature_fidelity_report, consumed unchanged"
  - phase: 02.5-05
    provides: "curvature_probe.py -- make_swiss_roll_fixture, centroid_mean_curvature, spearman_gate_statistic, median_relative_error, consumed unchanged"
  - phase: 02.2
    provides: "cae.ChartAutoEncoder / train_cae / PlainAutoEncoder / train_plain_ae / reconstruction_stats, consumed unchanged"
  - phase: 02.5-09
    provides: "the sealed single-seed reference measurement (rho_chart = -0.0604 at n_charts=8, seed=0) this plan's tracer reproduces"
provides:
  - "The D-02 Step-1 floor (median rho_chart > 0.65 over 5 seeds, best-of-swept-config, raw-point 0.6712 demoted to context) and the D-05 n_charts scope ruling (n_charts opened for Phase 3 only, swept set {2,3,5,8}) ratified at blocking checkpoints, before any Phase 3 rho_chart value existed"
  - "notebooks/diagnostics/swiss_roll_curvature_sweep_run.py -- resumable n_charts x seed Swiss roll curvature sweep runner (--smoke/--dry-run/--resume/--max-combos/--n-charts/--seeds), exercised on exactly one cell"
  - "One end-to-end reproduction: n_charts=8, seed=0 reproduces 02.5-09's rho_chart=-0.0604 bit-for-bit (-0.06041003026778113) through fit -> chart decoder -> torch.func curvature -> Spearman, confirming the whole measurement chain is wired correctly before any sweep expansion"
affects: ["03-02"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gate machinery is source-only (D-15): ROLL_FLOOR and RAW_BASELINE_CONTEXT are plain module constants with no PREREGISTRATION.md, ratification commit, or verdict JSON -- the file itself, committed before any measurement, is the whole mechanism"
    - "Raw-point context baseline computed ONCE per invocation (not per cell) and hard-asserted to four decimal places against RAW_BASELINE_CONTEXT, so any drift in the raw-point path is caught structurally rather than mistaken for a decoder effect"
    - "--smoke deliberately never writes to the record file -- a smoke key (n_charts=2, seed=0, reduced epochs/points) would otherwise collide with the real sweep's own (n_charts, seed) resumability index"

key-files:
  created:
    - notebooks/diagnostics/swiss_roll_curvature_sweep_run.py
  modified: []

decisions:
  - "D-02 ratified exactly as written (option ratify-d02): statistic = median rho_chart over 5 torch seeds with full spread reported; bar = absolute floor median rho_chart > 0.65; RAW_BASELINE_CONTEXT (0.6712) reported as context only, gates nothing; under the n_charts sweep the floor applies to the best swept config with the full sweep table printed and the multiple-comparisons caveat named; Swiss roll only, no PU equivalent; on no-clear, Phase 3 stops and reports (D-05a). Ratified before any file under notebooks/ contained a Phase 3 rho_chart value (verified: no 03_swiss_roll_curvature_sweep.jsonl existed at ratification time)."
  - "D-05 ratified exactly as written plus the planner's discretionary values (option ratify-d05): n_charts is in scope for Phase 3 and nothing else in the phase-2 stage is reopened -- Phases 02.3, 02.5, 02.6 and 02.7 remain on hold, no sealed verdict is reopened, softened, recomputed or reinterpreted. Roll sweep spans N_CHARTS_SWEEP = (2, 3, 5, 8) -- the measured monotone range {3,5,8} from 02.5-09 plus one untested lower value, 2. Seeds TORCH_SEEDS = (0,1,2,3,4); seed 0 is the exact 02.5-09 configuration and the reproduction anchor. D-06 stands: nothing measured on the roll ever selects a PU hyperparameter."
  - "The gate override (Phase 3's Depends-on line names a PASS that no method in this milestone has produced -- 02, 02.2, 02.4, 02.5 stage 1 are all FAIL) and its parameterization-damage consequence are restated in this plan's own artifacts (this file and the runner's module docstring), not inherited by reference alone: a curvature field decoded from an unvalidated CAE parameterization conflates real curvature with parameterization damage, and CURV-06/07's synthetic control provably cannot detect that, because a synthetic manifold that trains cleanly never reproduces the atlas-fragmentation pathology 02.5-09 measured."
  - "Task 3's reproduction was NOT adjusted to force a match -- the measured value (-0.06041003026778113) was compared honestly against the target (-0.0604) and found to reproduce bit-for-bit to more than four decimal places, which is itself evidence the fixture, split, training protocol and curvature chain are wired identically to 02.5-09's notebook."

requirements-completed: [CURV-01, CURV-03, DEC-05]

coverage:
  - id: D1
    description: "Swiss roll curvature sweep runner (swiss_roll_curvature_sweep_run.py) with --smoke/--dry-run/--resume/--max-combos/--n-charts/--seeds, ROLL_FLOOR=0.65 and N_CHARTS_SWEEP=(2,3,5,8) declared in source"
    requirement: "CURV-01"
    verification:
      - kind: other
        ref: ".venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --dry-run (exit 0, 20 cells printed, ROLL_FLOOR=0.65 printed, nothing written)"
        status: pass
      - kind: other
        ref: ".venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --smoke (exit 0, tally printed, nothing written to record)"
        status: pass
    human_judgment: false
  - id: D2
    description: "One reproduction cell (n_charts=8, seed=0) reproduces 02.5-09's rho_chart=-0.0604 through the full fit -> chart decoder -> torch.func curvature -> Spearman chain, torch-seed-reproducible per DEC-05"
    requirement: "DEC-05"
    verification:
      - kind: other
        ref: ".venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --n-charts 8 --seeds 0 --max-combos 1 -> rho_chart=-0.06041003026778113 (target -0.0604, within 2e-3, in fact within 1e-5)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Mean curvature vector field (H_vec) and its norm (H_norm) recorded per cell, labelled trace convention -- never Gaussian or principal curvature"
    requirement: "CURV-03"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_curvature_probe.py -x -q (52 passed)"
        status: pass
      - kind: other
        ref: "recorded JSONL field curvature_convention == 'trace' at n_charts=8, seed=0"
        status: pass
    human_judgment: false
  - id: D4
    description: "Both blocking checkpoints (D-02 floor/statistic ratification, D-05 n_charts scope + swept set ratification) answered by the developer and their dispositions recorded verbatim"
    verification: []
    human_judgment: true
    rationale: "Ratification of a one-way, pre-registration-style decision is a human judgment call by design -- these are the two blocking checkpoints Task 1 and Task 2 exist to force, and no automated check can substitute for the developer's own reply."

# Metrics
duration: "~25 min active (2 checkpoint round-trips for D-02/D-05 ratification, then continuous execution to completion)"
completed: 2026-08-13
status: complete
---

# Phase 3 Plan 01: Declare the Bar, Then Prove the Chain on One Swiss Roll Cell Summary

**Ratified Phase 3's only Step-1 bar (median rho_chart > 0.65 over 5 seeds, raw-point 0.6712 demoted to context-only) and the n_charts scope-opening before any number existed, then built `swiss_roll_curvature_sweep_run.py` and proved the whole fit -> chart decoder -> torch.func curvature -> Spearman chain reproduces 02.5-09's -0.0604 bit-for-bit at n_charts=8, seed=0.**

## Performance

- **Duration:** ~25 min active work (plus checkpoint round-trip time for the developer's D-02/D-05 replies)
- **Tasks:** 3 (2 blocking checkpoints + 1 tracer task)
- **Files modified:** 1 (`notebooks/diagnostics/swiss_roll_curvature_sweep_run.py`, new)

## Accomplishments

- **Task 1 (D-02 ratification):** the Step-1 statistic (median rho_chart over 5 seeds), the absolute floor (0.65), and the raw-point baseline's disposition (context only, gates nothing) were locked in before any Phase 3 `rho_chart` value existed anywhere under `notebooks/`.
- **Task 2 (D-05 ratification):** `n_charts` opened as an in-scope Phase 3 hyperparameter, swept set `{2, 3, 5, 8}`, seeds `{0, 1, 2, 3, 4}` -- with the scope explicitly bounded to this one knob; Phases 02.3, 02.5, 02.6, 02.7 remain on hold.
- **Task 3 (tracer):** built the sweep runner with the ratified floor and swept set as module constants, and ran exactly one cell (`n_charts=8, seed=0`) end to end. The measured `rho_chart = -0.06041003026778113` reproduces `02.5-09`'s `-0.0604` to well beyond the required `2e-3` tolerance, and every supporting number matched too: raw-point baseline `rho = 0.6712` (exact to four decimals), `cond_max = 63.19`, `mre_chart = 0.6644`, `curvature_convention = "trace"`, `activation = "silu"`.

## Task Commits

1. **Task 1: Ratify the Step-1 floor (D-02, one-way)** — checkpoint, no file changes; ratification recorded in this SUMMARY.
2. **Task 2: Ratify the n_charts scope ruling and swept set (D-05, one-way)** — checkpoint, no file changes; ratification recorded in this SUMMARY.
3. **Task 3: End-to-end Swiss roll curvature cell** - `4dc9b05` (feat)

**Plan metadata:** committed with this SUMMARY (see final commit below).

## Files Created/Modified

- `notebooks/diagnostics/swiss_roll_curvature_sweep_run.py` - Resumable n_charts x seed Swiss roll curvature sweep runner. Module constants `ROLL_FLOOR = 0.65`, `RAW_BASELINE_CONTEXT = 0.6712`, `N_CHARTS_SWEEP = (2, 3, 5, 8)`, `TORCH_SEEDS = (0, 1, 2, 3, 4)` declared in source before any measurement. `run_cell()` reproduces `02.5_swiss_roll_chart_curvature_check.ipynb`'s exact sequence (fixture -> analytic H derivation with a `< 1e-12` pin check -> 80/20 split seeded by the torch seed -> `cae.ChartAutoEncoder`/`cae.train_cae` -> matched `cae.PlainAutoEncoder`/`cae.train_plain_ae` -> `model.double()` -> `chart_curvature.chart_curvature_field` -> `curvature_probe.spearman_gate_statistic`/`median_relative_error` -> `chart_curvature.curvature_fidelity_report`). `raw_baseline_context()` computes the raw-point centroid Spearman once per invocation and hard-asserts it against `RAW_BASELINE_CONTEXT` to four decimal places. CLI: `--dry-run`, `--smoke`, `--resume`, `--record-path`, `--max-combos`, `--n-charts`, `--seeds`. Records append to `notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl` (gitignored) keyed by `(n_charts_configured, seed)` for resumability. Does NOT implement the median/floor/read-out summary layer -- that is plan 03-02's job.

## Decisions Made

See frontmatter `decisions:` for the verbatim ratification text. In one sentence each:

- **D-02:** absolute floor `median rho_chart > 0.65` over 5 seeds, raw-point `0.6712` is context only, best-of-swept-config applies with the multiple-comparisons caveat named, no PU equivalent gate exists.
- **D-05:** `n_charts` opened in scope for Phase 3 only (`{2,3,5,8}` swept, seeds `{0..4}`), every other phase-2-stage hold stays exactly where it was.
- **Gate override:** restated (not merely referenced) that a curvature field decoded from an unvalidated CAE parameterization conflates real curvature with parameterization damage, and the synthetic control cannot detect it.

## Deviations from Plan

None — plan executed exactly as written. The reproduction cell matched the target to well within tolerance on the first run; no constant was adjusted to force agreement, and none needed to be.

## Issues Encountered

None. The `.planning/STATE.md` "modified" state visible in `git status` at session start predates this plan's execution (set by the orchestrator's `init.execute-phase` step before this agent was spawned, marking `status: executing`) and was left untouched until this plan's own STATE.md update below.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- The measurement chain (fixture -> chart decoder -> `torch.func` curvature -> Spearman vs analytic `H`) is proven correct end to end at production quality, at the exact configuration `02.5-09` measured.
- `swiss_roll_curvature_sweep_run.py` is ready to run its full 20-cell grid (4 `n_charts` values x 5 seeds) in plan 03-02, which owns the median/floor/best-of-swept-config read-out layer this plan deliberately did not build.
- Both one-way ratifications (D-02, D-05) are now fixed in this plan's own artifacts and in the runner's module docstring; plan 03-02 inherits them without re-deciding anything.
- No blockers.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-13*
