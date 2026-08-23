---
phase: 03-decoder-curvature-field
plan: 11
subsystem: phase-record
tags: [findings, requirements-remint, presentation-notebook, phase-close]
status: complete
executed: 2026-08-17..2026-08-18
recorded: 2026-08-23

# Dependency graph
requires:
  - phase: 03-decoder-curvature-field (03-02)
    provides: the Step-1 Swiss roll known-answer gate and its outcome
  - phase: 03-decoder-curvature-field (03-06)
    provides: the chart-decoder Swiss roll sanity notebook and the forward/reverse equality
  - phase: 03-decoder-curvature-field (03-09)
    provides: the deliverable PU curvature field and its three-seed spread
  - phase: 03-decoder-curvature-field (03-10)
    provides: the synthetic controls and the cond(g) -> artifact-curvature band table
provides:
  - 03-FINDINGS.md -- the phase record: override, n_charts scope ruling, dimension
    justification, four steps, caveats, re-mint mapping
  - .planning/REQUIREMENTS.md "Phase 3 Requirement Re-Mint" -- all 13 DEC/CURV IDs re-minted,
    none dropped, with an old-to-new mapping table
  - notebooks/03_pu_curvature_field.ipynb -- presentation of the PU field and the synthetic
    controls, read from recorded runner JSONL with no training inline
affects: [Phase 4 (blocked), spike 003 (annotates this record's section 6)]
---

# 03-11 — the phase record and the requirement re-mint

## What this summary is

**The work of 03-11 was executed on 2026-08-17/18. No `03-11-SUMMARY.md` was written at the
time**, which is why the phase has read `11 plans / 10 summaries` ever since and why routing
kept treating Phase 3 as unfinished. This document records what actually landed, verified
against the plan's `must_haves` rather than asserted, and closes the plan.

Nothing was re-executed to produce this summary. No artifact was created to satisfy a
checklist. Where a must-have is only partly met, it is named below rather than smoothed over.

## Must-have verification

| must-have | status | evidence |
|---|---|---|
| 13 stale IDs re-minted under the same namespace, none dropped, mapping recorded | **met** | `REQUIREMENTS.md:245` "Phase 3 Requirement Re-Mint (2026-08-13, executed 2026-08-17)", mapping table from line 261; text states "None was retired, dropped or re-pointed" |
| Gate override stated in Phase 3's own words | **met** | `03-FINDINGS.md` §1 "The override, in this phase's own words" |
| D-05 `n_charts` scope ruling in Phase 3's own words, opening that knob and nothing else | **met** | §2 "The D-05 `n_charts` scope ruling, in this phase's own words" |
| `chart_dim = 20` justification; `d_frozen = 5` rejection with reason | **met** | §3 "The working dimension" |
| Every statistic carries its seed spread; no single draw presented as a result | **met** | §5 "The three-seed spread — and why the field does not survive it"; 32 occurrences of `seed` |
| Multiple-comparisons, parameterization-damage and 0.6712-is-context caveats beside their numbers | **met** | §6 opens with the parameterization-damage caveat inline; §8 "Carried limitations" |
| Presentation notebook reads runner JSONL, no training or expensive compute inline | **met** | 16 cells, 8 with outputs; `train_cae` and `chart_curvature_field` appear **zero** times; reads `.cache/*.jsonl` |
| Step-1-only branch written as a complete outcome if the gate did not clear | **n/a** | the Step-1 gate cleared; the full four-step record was written instead |
| Synthetic control reported before Phase 4 with the blindness caveat beside it | **met** | §6, stated before the table rather than in a limitations section |
| No sealed verdict from 2, 02.1, 02.2, 02.4, 02.5, 02.6, 02.7 reopened | **met** | §1 and §6 both state results as conditioned on the override; no upstream verdict is recomputed |

## Phase 3 outcome, in one paragraph

The decoder curvature field was built, measured and reported, and **it is not validated**.
`CURV-07` is answered "neither established": seed 20260813's field sits 351x above the measured
`cond(g)` artifact floor, the instrument is proven correct at `d=4` (`rho = 0.989`), and yet the
field **does not reproduce across seeds** — the `‖H‖` median spans 52x over three converged
draws and two of the three are numerically degenerate. That last point needs no control and no
analytic truth; it is the phase's own declared reporting unit applied honestly, and it is why
Phase 4 stays blocked. The phase's most transferable result is the `cond(g)` ->
artifact-curvature band table, monotone across four decades.

## One inference in this record is superseded — see the supplement

`03-FINDINGS.md` §6 point 3 argues that PU's accuracy is untested because "no curved control
reached PU's conditioning, so nothing bounds `‖H‖` — neither magnitude nor, **after the
saddle's `rho`**, ordering."

The clause after the dash no longer follows. Spike 003 (2026-08-22) measured that the `d=20`
saddle control is **unrankable by construction** — its second fundamental form is constant, so
its `‖H‖` varies only through the metric — and that a control with a varying second fundamental
form ranks at `rho = +0.65` at the same `d`, `D`, `n` and `k`. The saddle's `rho = -0.0151` was
therefore uninformative about ordering rather than evidence against it.

**The conclusion of §6 point 3 is unchanged** — PU's accuracy remains untested, and the
structural gap it names (no CAE has fitted a *curved* 20-manifold to PU-comparable quality) is
still open, because spike 003 used the point-cloud estimator and not the decoder. Only the
supporting clause is withdrawn. Recorded in `03-FINDINGS-SUPPLEMENT-01.md` rather than by
editing the sealed section.

## Known gaps at close

1. **`autonomous: false` on this plan.** The original plan required a human checkpoint at
   execution. The 2026-08-17/18 execution predates this summary and no checkpoint record
   exists for it; closure here rests on the developer's 2026-08-23 instruction to close the
   phase, not on a recovered checkpoint.
2. **`03-FINDINGS.md` predates spikes 001, 002 and 003.** It is accurate as written and is
   annotated by the supplement rather than revised.
3. **Phase 4 remains blocked** and this plan does not unblock it. D-11 stands: no route out was
   proposed by Phase 3 itself. Spike 003 proposes candidate routes; none is ratified.
