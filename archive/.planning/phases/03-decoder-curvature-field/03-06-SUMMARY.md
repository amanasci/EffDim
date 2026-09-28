---
phase: 03-decoder-curvature-field
plan: 06
subsystem: testing
tags: [swiss-roll, cae, chart-curvature, torch.func, forward-mode, sanity-check, claude-md-compliance]

# Dependency graph
requires:
  - phase: 03-02
    provides: "the Step-1 sweep's winning n_charts=2 config at N_POINTS=3000 (03-02-SUMMARY.md), and the sweep runner that owns the phase's actual gate"
  - phase: 03-05
    provides: "the opt-in mode=\"forward\" path on chart_mean_curvature/chart_curvature_field, proved equal to reverse on a synthetic equivalence fixture"
provides:
  - "notebooks/03_swiss_roll_chart_curvature_field_check.ipynb -- the CLAUDE.md-mandated Swiss roll sanity check for Phase 3's chart-decoder curvature field, committed executed"
  - "First exercise of the plan 03-05 forward-mode toggle against a real trained decoder (not only the synthetic equivalence fixture), confirmed agreeing with reverse at rtol=1e-9, atol=1e-12"
  - "An independent cross-check that this notebook and notebooks/diagnostics/swiss_roll_curvature_sweep_run.py measure the same pipeline: both report rho_chart=0.7817 at n_charts=2, seed=0, n=3000"
affects: [any future plan adding a new manifold model that needs a Swiss roll notebook precedent for the chart-decoder curvature chain]

# Tech tracking
tech-stack:
  added: []
  patterns: ["single-seed CLAUDE.md sanity notebook explicitly separated from a phase's pre-registered gate: reports the same statistics the gate uses (rho_chart, fidelity axes) but labels every one 'reported context' and states in prose where the gate actually lives, rather than printing a second pass/fail line keyed on the gate's floor"]

key-files:
  created:
    - notebooks/03_swiss_roll_chart_curvature_field_check.ipynb
  modified: []

key-decisions:
  - "N_CHARTS = 2, the Step-1 sweep's winning config specifically at N_POINTS=3000 (03-02-SUMMARY.md's pre-registered-configuration table), not the n=12000 amended-gate winner -- this notebook runs at n=3000 per CLAUDE.md, so the n=3000-scale winner is the correct reference value, and n_charts=2 wins at both scales anyway."
  - "N_POINTS = 3000 per CLAUDE.md's mandated sanity-check scale and its two-minute bound, held exactly as the plan specified, despite the phase's own Step-1 gate having found n=3000 marginal for curvature specifically (03-02-AMENDMENT-01.md: MISS at n=3000, median rho_chart=0.4347 vs floor 0.65; CLEARS at n=12000, median 0.8302). This tension is deliberate, not an oversight: the notebook is a cheap readable sanity check that must stay under two minutes on CPU, not the gate, and the gate's own amended scale lives entirely in the sweep runner. N_POINTS was not raised to \"help\" this notebook's numbers."
  - "Read-out cell's MODES check compares full H_vec (not just H_norm) between mode=\"reverse\" and mode=\"forward\" via torch.allclose(rtol=1e-9, atol=1e-12) -- the same tolerance plan 03-05's equivalence test uses -- so this is the first agreement check run against a real fit rather than only the synthetic fixture."
  - "Fixed the plan's own naive verify-script false positive during authoring: the first markdown cell's prose originally used the literal substrings '.cache' and 'verdict' while explaining what the notebook does NOT do (does not touch notebooks/.cache/, has no verdict JSON). Reworded to 'the milestone's gitignored cache directory' and 'no scored-outcome artifact' so the prohibition check (which greps for literal substrings, not intent) passes without weakening what the sentence says."

patterns-established:
  - "Extract embedded notebook plot PNGs via nbformat/base64 decode for direct visual inspection during a checkpoint, rather than only trusting the printed pass/fail lines -- used here to give the human reviewer (and myself, independently) a same-session look at the x-z colour-ordering criterion before approval."

requirements-completed: [DEC-01, DEC-02, CURV-03]

coverage:
  - id: D1
    description: "notebooks/03_swiss_roll_chart_curvature_field_check.ipynb exists, is committed executed with its outputs, and runs end to end in under two minutes on CPU (12 cells, imports chart_curvature/cae/curvature_probe unchanged, chart_dim=2, trains from scratch, never touches notebooks/.cache/)"
    requirement: "DEC-01"
    verification:
      - kind: other
        ref: ".venv/bin/jupyter nbconvert --to notebook --execute --inplace -- real wall clock 40.6s, in-notebook measured wall clock 36.2s printed in the read-out cell's own output"
        status: pass
      - kind: other
        ref: "automated verify script from 03-06-PLAN.md (cell count <=15, no '.cache'/'PREREGISTRATION'/'merge-base'/'verdict' substring, CHART_DIM = 2, PlainAutoEncoder present, at least one executed code cell) -- self-check re-run below"
        status: pass
    human_judgment: false
  - id: D2
    description: "Original and CAE reconstruction plotted side by side as both a 3-D scatter and an x-z scatter, coloured by t, with the reconstruction's colour bands staying in order along the spiral (no crossing or folding)"
    requirement: "DEC-02"
    verification:
      - kind: manual_procedural
        ref: "blocking checkpoint:human-verify, Task 2 -- developer replied 'approved'"
        status: pass
    human_judgment: true
    rationale: "Visual read of colour-band ordering in a rendered plot is exactly the class of judgment CLAUDE.md's backstop truth requires a human to make -- the developer's own eyes on the committed x-z panel, not a numeric proxy."
  - id: D3
    description: "chart_curvature_field exercised at mode=\"reverse\" and mode=\"forward\" against a real trained decoder for the first time (plan 03-05 proved equivalence only on a synthetic fixture); H_vec agrees to float64 round-off"
    requirement: "CURV-03"
    verification:
      - kind: other
        ref: "notebook cell output: 'forward/reverse H_vec agree (rtol=1e-9, atol=1e-12): True'"
        status: pass
    human_judgment: false
  - id: D4
    description: "Notebook ends with exactly four printed pass/fail lines (RECON, BASELINE, CHARTS, MODES) plus a one-sentence read-out, with none keyed on the phase gate's 0.65 floor or 0.6712 baseline, and the notebook states plainly that the Step-1 gate lives in the sweep runner"
    requirement: "DEC-01"
    verification:
      - kind: other
        ref: "notebook read-out cell output, transcribed in this SUMMARY's Accomplishments section; grep for '0.65' and 'gate' in notebook source"
        status: pass
    human_judgment: false

# Metrics
duration: ~25min active (checkpoint hold between Task 1 commit and Task 2 approval)
completed: 2026-08-14
status: complete
---

# Phase 3 Plan 06: Swiss Roll Sanity Check for the Chart-Decoder Curvature Field Summary

**CLAUDE.md-mandated Swiss roll notebook for Phase 3's chart-decoder curvature chain: trains a fresh CAE + matched plain-AE, reconstructs to 2.2% error (27.38x the baseline), and exercises the plan-03-05 forward-mode curvature path against a real fit for the first time, agreeing with reverse to float64 round-off.**

## Performance

- **Duration:** ~25min active (Task 1 build/execute/commit, then a checkpoint hold for Task 2's human visual verification)
- **Started:** 2026-08-14 (session start)
- **Completed:** 2026-08-14
- **Tasks:** 2 (1 auto, 1 blocking checkpoint:human-verify)
- **Files modified:** 1 created (`notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`)

## Accomplishments

- Built and executed `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`, 12 cells, real wall clock 40.6s (nbconvert including kernel startup), in-notebook measured wall clock **36.2s** — well under CLAUDE.md's two-minute bound.
- Trained a fresh `cae.ChartAutoEncoder` (`chart_dim=2`, `n_charts=2` — the Step-1 sweep's winning config at `N_POINTS=3000`) and a matched `cae.PlainAutoEncoder`, both from scratch, no cached fit reused.
- Exercised `chart_curvature.chart_curvature_field` at both `mode="reverse"` (default) and `mode="forward"` (plan 03-05's opt-in path) against this notebook's own real fit for the first time — previously proved equal only on a synthetic equivalence fixture.
- All four read-out lines pass:
  - **`RECON`**: True — held-out mean relative reconstruction error **2.2%** (bound 10%)
  - **`BASELINE`**: True — CAE `mse_per_dim = 0.000287` vs matched plain-AE `mse_per_dim = 0.007844`, CAE is **27.38x** better
  - **`CHARTS`**: True — `n_charts_used = 2 / N_CHARTS = 2`, atlas did not collapse
  - **`MODES`**: True — forward and reverse `H_vec` agree via `torch.allclose(rtol=1e-9, atol=1e-12)`
  - One-sentence read-out (as printed): *"the chart auto-encoder reconstructs the Swiss roll to 2.2% relative error and beats its matched plain-AE baseline by 27.38x on a manifold it was designed for, using 2/2 charts, while curvature through its trained decoder scores rho_chart=0.7817 against the raw-point baseline's 0.6712 (reported context only — see the sweep runner for the phase's actual gate), and the plan 03-05 forward-mode path agrees with reverse to float64 round-off on this real fit."*
- No pass/fail line anywhere is keyed on the phase gate's `0.65` floor or `0.6712` raw-point baseline. `0.6712` appears exactly twice in the notebook, both times as reported context printed beside the chart-decoder number, never as a threshold. The notebook states in two separate cells that the Step-1 gate is decided by `notebooks/diagnostics/swiss_roll_curvature_sweep_run.py`, not here.
- Both reference notebooks (`notebooks/02.5_swiss_roll_chart_curvature_check.ipynb`, `notebooks/02.2_swiss_roll_cae_check.ipynb`) confirmed untouched (`git status --porcelain` clean on both) — the project's additive-only rule held.
- `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` still exits 0 at **286 passed, 1 skipped**.

## Cross-check found during execution (context, not a gate result)

The notebook's `rho_chart = 0.7817` at `n_charts=2, seed=0, n=3000` is **byte-identical in value** to `03-02-SUMMARY.md`'s recorded per-seed number for that exact `(n_charts, seed)` cell in the 20-cell pre-registered sweep. This is an independent confirmation, found rather than engineered, that this notebook's pipeline (fresh training, `chart_curvature_field` at default reverse mode, `spearman_gate_statistic` against the analytic fixture) and `swiss_roll_curvature_sweep_run.py`'s pipeline compute the same thing on the same configuration. It is recorded here as corroborating context, exactly as the plan requires — it changes nothing about which artifact owns the phase's gate.

## The N_POINTS=3000 tension, stated plainly

This notebook runs at `N_POINTS = 3000`, per CLAUDE.md's mandated sanity-check scale and its two-minute-on-CPU bound. The phase's own Step-1 gate (`03-02-SUMMARY.md`, `03-02-AMENDMENT-01.md`) separately found `N_POINTS = 3000` **marginal for curvature specifically**: the pre-registered gate MISSED at that scale (best config `n_charts=2`, median `rho_chart=0.4347` against the `0.65` floor) and only CLEARED after the sweep was amended to `N_POINTS = 12000` (median `0.8302`). Those are two different artifacts answering two different questions at two different scales, and that is deliberate, not an oversight: CLAUDE.md's "~3,000 points" protocol was written for reconstruction sanity (a zeroth-order property), this notebook honors that protocol exactly as instructed, and the phase's real gate — which needed the denser sample — lives entirely in the sweep runner at its own amended scale. Nothing here was tuned, and `N_POINTS` was not raised, to make this notebook's numbers look better; the measured `0.7817` at `n=3000` happens to be a healthy number, and it is reported as exactly what it is: a single-seed sanity read at CLAUDE.md's mandated scale, not evidence about the gate.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build and execute `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`** — `435fa6c` (feat)
2. **Task 2: Read the committed notebook and confirm the colour ordering survived** — blocking `checkpoint:human-verify`, no code changes; developer replied `approved` with the colour-ordering read transcribed below.

**Plan metadata:** (this commit — see final commit hash after this SUMMARY lands)

## Files Created/Modified

- `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb` — the CLAUDE.md-mandated Swiss roll sanity notebook for Phase 3's chart-decoder curvature field; 12 cells, committed executed with outputs.

## Decisions Made

- `N_CHARTS = 2`: the Step-1 sweep's winning config at `N_POINTS=3000` specifically (03-02-SUMMARY.md's pre-registered-scale table), matching the scale this notebook actually runs at. `n_charts=2` also wins (tied with `n_charts=8`) at the amended `n=12000` scale, so the choice is robust to which table is read, but the `n=3000` table is the one that actually applies here.
- `N_POINTS = 3000` held exactly as CLAUDE.md and the plan specify, not raised despite the phase's own finding that `n=3000` is marginal for the *gate*. See "The N_POINTS=3000 tension" above.
- `MODES` check compares full `H_vec` (not just the scalar `H_norm`) between the two differentiation modes, at the same `rtol=1e-9, atol=1e-12` tolerance plan 03-05's own equivalence test uses, so this is a like-for-like extension of that proof onto a real (not fixture) fit.
- Reworded one sentence in the intro markdown cell during authoring, before commit, to avoid the literal substrings `.cache` and `verdict` that the plan's own automated verify script and acceptance criteria grep for — the sentence's meaning (this notebook does not touch the cache and produces no scored-outcome artifact) is unchanged; only the literal substrings were avoided. See Deviations below.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reworded intro prose to avoid literal `.cache`/`verdict` substrings the plan's own acceptance check greps for**
- **Found during:** Task 1, immediately after first execution — running the plan's given automated verify script against the freshly executed notebook.
- **Issue:** The plan's `<verify><automated>` block and acceptance criteria assert `'.cache' not in src` and (per acceptance criteria) no occurrence of `verdict`. The first markdown cell's prose, written to explain what the notebook deliberately does *not* do, used the phrases "reads/writes `notebooks/.cache/`" and "no verdict JSON" — both literal substring matches, even though the notebook's actual code never touches the cache and produces no verdict artifact. The check is a naive substring grep, not an intent parser, so it fired as a false positive against explanatory prose.
- **Fix:** Reworded to "reads from / writes to the milestone's gitignored cache directory" and "no scored-outcome artifact" — same meaning, no literal substring match.
- **Files modified:** `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb` (markdown cell 1 only; no code cell touched, no re-execution needed since markdown carries no outputs).
- **Verification:** Re-ran the plan's exact automated verify script plus the acceptance-criteria substring checks for `.cache`, `PREREGISTRATION`, `merge-base`, `verdict` — all pass. Re-validated the notebook with `nbformat.validate` and confirmed execution counts/outputs were untouched by the edit.
- **Committed in:** `435fa6c` (part of Task 1's commit — the edit was made before the task's own commit, not as a follow-up).

---

**Total deviations:** 1 auto-fixed (1 blocking — a false-positive-prone acceptance check against explanatory prose, not a defect in the notebook's actual behavior).
**Impact on plan:** No scope creep; no change to any executed cell, plotted figure, or printed number. Purely a wording fix in markdown prose to satisfy the plan's own literal-substring acceptance check.

## Issues Encountered

None beyond the deviation above. Training, plotting, and both curvature-mode computations ran cleanly on the first execution.

## Colour Ordering — the checkpoint's verification record

**Developer response: `approved`.**

Colour ordering read, by the developer and independently by the orchestrator (which extracted the x-z panels from the committed notebook via `nbformat`/base64 decode and viewed them directly before approving): the reconstruction traces the same spiral as the original in the same colour sequence — purple at the inner terminus through blue, teal, green, to yellow at the outer edge — with no crossing, no fold, and no band out of sequence. The reconstruction is visibly noisier than the original, scattering slightly around the curve where the original is a sharp line, consistent with the measured 2.2% held-out relative error and read as a noise difference rather than a structural failure. **The surface stayed in order.**

The executor independently extracted and viewed the same two plot cells (the input roll and the reconstruction comparison) before the checkpoint was raised, and reached the same read: colour bands in order, no folding, in both the 3-D and x-z panels.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 3's CLAUDE.md obligation for the chart-decoder curvature field (both the reverse path inherited from Phase 02.5 and the new forward path from plan 03-05) is discharged: the mandatory Swiss roll notebook exists, is committed executed, and passed its checkpoint.
- No blockers for `03-07` onward. The forward-mode toggle is now proven equal to reverse both on the plan 03-05 synthetic fixture and on this notebook's own real fit — any future plan choosing `mode="forward"` for a wall-clock win at PU scale has two independent equivalence proofs to point to, not one.
- Test suite remains green at 286 passed, 1 skipped (the CUDA-gated device-parity test, correctly skipped on this CPU-only machine).

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-14*
