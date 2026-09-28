---
phase: 03-decoder-curvature-field
plan: 08
subsystem: curvature-instrumentation
tags: [pytorch, chart-autoencoder, persistent-homology, pu-manifold, selection-rule]

# Dependency graph
requires:
  - phase: 03-decoder-curvature-field (03-07)
    provides: curvature_field_pu_run.py -- the resumable nine-cell instrument, its four
      D-07 diagnostics and the lexicographic selection rule declared before any PU number
      existed
provides:
  - a complete corrected nine-cell PU grid plus six control cells in
    notebooks/.cache/03_curvature_field_pu.jsonl, and one selected n_charts (4) chosen by
    the pre-declared rule, unchanged
affects: [03-09 (computes the curvature field on the selected config)]

tech-stack:
  added: []
  patterns:
    - "Grid re-run from scratch after a defect ledger, with the defective record file kept
       beside the corrected one under an explicit _DEFECTIVE_pre_fix stem rather than deleted"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/curvature_field_pu_run.py

key-decisions:
  - "The pre-declared lexicographic rule was applied unchanged and selected n_charts=4;
     the ranking function and its docstring were not edited during this plan"
  - "D-12's escalation checkpoint (Task 3) is answered by 03-NOTE-d12-retirement.md: the
     trigger fired on both legs and was then retired as the wrong instrument for a C2
     question. No d sweep."
  - "The Swiss roll's winning n_charts=2 played no part in the PU selection (D-06)"

requirements-completed: [DEC-03, DEC-04, DEC-05]

duration: ~4.4h measured training wall clock across two sessions
completed: 2026-08-16
status: complete
---

# Phase 3 Plan 08: PU Nine-Cell Grid and Selection Summary

**The corrected nine-cell PU grid ran to completion (4.34 h training wall clock), the rule declared in plan 03-07 was applied unchanged, and it selected `n_charts = 4` on metric conditioning -- an axis whose across-seed spread at that config covers five orders of magnitude and whose winning median is only reachable because one of the three seeds collapsed onto a single chart.**

## Status of this record

This SUMMARY was written **after** the grid ran, closing out a plan whose execution had
already landed (the grid, the defect ledger, the corrected re-run and the D-12 note were all
committed) but whose close-out was never written. Nothing here is a fresh measurement: every
number is read back from `notebooks/.cache/03_curvature_field_pu.jsonl` by
`--select-only` and by direct record inspection. The ordering is disclosed rather than
implied.

## Task 1 — the nine-cell grid

Nine `(n_charts, seed)` cells over `{4, 8, 16} x {20260813, 20260814, 20260815}`, all at
`chart_dim = 20` (D-11) and `activation = silu`, plus six control cells (three matched
`latent_dim = 20` D-12 controls, three separately-labelled `latent_dim = 40` capacity
references). `--select-only` reports **9 of 9 planned grid cells present**, **3 of 3** matched
controls and **3 of 3** capacity references, and exits 0.

The grid was run twice. The first run is preserved at
`notebooks/.cache/03_curvature_field_pu_DEFECTIVE_pre_fix.jsonl` and is invalidated by the
three defects recorded in `03-08-DEFECTS-01.md` (unmatched D-12 control, truncated training
protocol, PH `sqrt(d)` saturation). The corrected run — commits `34ccc54`, `260a614`, with
the reasoning in `03-08-SUPPLEMENT-02.md` — is the record every number below comes from. The
defective file was kept, not deleted.

### Wall clock against D-13

| Term | Measured |
|---|---|
| Nine CAE cells, training only | **15,602.8 s = 4.334 h** |
| Six control cells, training only | 199.7 s = 0.055 h |
| Total training | **15,802.5 s = 4.390 h** |
| 02.2's per-fit anchor (n_charts=16, d=20) | 1,941.2 s |
| This grid's per-fit mean | 1,733.6 s |

Training alone lands inside D-13's 3–5 hour envelope. 03-07's `--timing-probe` had projected
**~5.6–5.7 h** for training plus curvature and halted over the envelope by design; the
corrected grid's training term came in under that projection, and the per-fit mean sits
below 02.2's anchor rather than above it.

**Gap, stated plainly:** the plan's acceptance criterion asked that every fit record carry
both a `train_wall_s` and a `curv_wall_s` value. Records carry `train_wallclock_s` only —
`_run_cae_cell` never recorded a separate curvature timing. The grid total above is therefore
a **training** total, and the per-cell curvature cost of this grid was never measured. The
curvature wall clock at `d = 20, D = 768` is measured for the first time in the convergence
run recorded below, and again in plan 03-09 over the full cloud.

## Task 2 — the D-07 table and the pre-declared rule

### Per-seed values, all four diagnostics, nothing collapsed

| n_charts | seed | epochs_run | early_stopped | train s | occupancy | cond median | cond p90 | cond p99 | cond max | mse_per_dim | mean_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 20260813 | 30 | true | 646 | **2** | 9.758e+06 | 1.263e+07 | 3.106e+07 | **4.886e+07** | 1.2474e-04 | 0.29309 |
| 4 | 20260814 | 27 | true | 608 | **1** | 3.657e+02 | 3.657e+02 | 3.657e+02 | **3.657e+02** | 2.4382e-04 | 0.41121 |
| 4 | 20260815 | 27 | true | 574 | 3 | 1.357e+03 | 2.233e+04 | 2.233e+04 | **2.233e+04** | 1.7331e-04 | 0.34802 |
| 8 | 20260813 | 40 | false | 1569 | 7 | 8.407e+02 | 5.750e+04 | 8.525e+04 | 8.526e+04 | 1.4288e-04 | 0.31620 |
| 8 | 20260814 | 32 | true | 1280 | 2 | 4.805e+06 | 8.657e+06 | 1.511e+07 | 2.137e+07 | 1.2275e-04 | 0.28986 |
| 8 | 20260815 | 40 | false | 1573 | 6 | 4.051e+06 | 7.535e+06 | 1.598e+07 | 3.133e+07 | **8.8986e-05** | 0.24680 |
| 16 | 20260813 | 40 | false | 3098 | 15 | 1.409e+04 | 9.414e+04 | 1.495e+05 | 1.495e+05 | 1.2678e-04 | 0.29751 |
| 16 | 20260814 | 40 | false | 3124 | 6 | 7.682e+06 | 1.392e+07 | 3.126e+07 | 4.202e+07 | 1.2035e-04 | 0.28639 |
| 16 | 20260815 | 40 | false | 3127 | 10 | 7.928e+06 | 1.436e+07 | 2.500e+07 | 3.239e+07 | 1.0534e-04 | 0.26670 |

Per-output-dimension reconstruction, reported beside the aggregate as required:

| n_charts | seed | dim_mse_median | dim_mse_p95 | dim_mse_max |
|---|---|---|---|---|
| 4 | 20260813 | 1.1145e-04 | 2.5760e-04 | 4.9821e-04 |
| 4 | 20260814 | 2.0179e-04 | 5.2093e-04 | 1.1620e-03 |
| 4 | 20260815 | 1.5401e-04 | 3.3080e-04 | 8.3454e-04 |
| 8 | 20260813 | 1.2973e-04 | 2.6328e-04 | 5.1617e-04 |
| 8 | 20260814 | 1.1080e-04 | 2.4688e-04 | 5.3878e-04 |
| 8 | 20260815 | 8.2321e-05 | 1.5867e-04 | 2.9342e-04 |
| 16 | 20260813 | 1.1744e-04 | 2.2730e-04 | 4.1400e-04 |
| 16 | 20260814 | 1.0877e-04 | 2.2756e-04 | 4.7817e-04 |
| 16 | 20260815 | 9.6632e-05 | 1.9403e-04 | 4.5202e-04 |

Primary PH cells (`latent|ambient|*|bottleneck_norm`), the two the rule may read:

| n_charts | seed | H0 | H1 |
|---|---|---|---|
| 4 | 20260813 | 0.6217 | 0.8451 |
| 4 | 20260814 | 2.3620 | 0.5000 |
| 4 | 20260815 | 0.5000 | 1.0309 |

### Across-seed medians, verbatim from `--select-only`

```
9 of 9 planned grid cells present in notebooks/.cache/03_curvature_field_pu.jsonl
Selection table (per n_charts, across-seed medians):
  n_charts=  4 seeds_present=3 median_occupancy=2.00 median_max_cond_g=2.233e+04 median_mse_per_dim=0.000173309 median_H0_bottleneck_norm=0.6217 median_H1_bottleneck_norm=0.8451
  n_charts=  8 seeds_present=3 median_occupancy=6.00 median_max_cond_g=2.137e+07 median_mse_per_dim=0.000122746 median_H0_bottleneck_norm=0.323 median_H1_bottleneck_norm=0.8122
  n_charts= 16 seeds_present=3 median_occupancy=10.00 median_max_cond_g=3.239e+07 median_mse_per_dim=0.000120349 median_H0_bottleneck_norm=0.4484 median_H1_bottleneck_norm=0.6451
Lexicographic ranking (ascending -- first is selected):
  n_charts=4
  n_charts=16
  n_charts=8
No weighted composite formed; no single score printed above this line.
Selected n_charts: 4
```

### Selected: `n_charts = 4`. Deciding axis: metric conditioning.

Axis 1 (`median max cond(g)`, factor-of-2 tie band) decided it outright and no axis below it
was reached: `2.233e+04` against `2.137e+07` and `3.239e+07` — three orders of magnitude
outside the tie band, so `mse_per_dim` and the PH cells never entered the comparison. **No
axis was tied.**

The rule was confirmed unchanged before being applied. `git diff` over
`notebooks/diagnostics/curvature_field_pu_run.py` shows no edit to `apply_selection_rule` or
to its declaring docstring during this plan. Every commit touching the file since the rule was
declared in `52cbb01` is accounted for: `0166baa` (`--device` and a device-aware timing probe,
03-07-SUPPLEMENT-01), `34ccc54` (the three defect fixes), `136bd8f` (a stale `prescale=True`
docstring correction), and the additive `--converge` mode recorded below. `git diff 52cbb01
HEAD -- notebooks/diagnostics/curvature_field_pu_run.py` contains **no** added or removed line
inside `apply_selection_rule`, its comparator, or its tie band.

### What the rule bought, said plainly

The selected config is the **worst of the three on reconstruction** (median `mse_per_dim`
1.733e-04 against 1.227e-04 and 1.203e-04) and the **worst on chart occupancy** (median 2.00
against 6.00 and 10.00). It won on conditioning alone. Three observations the reader needs
in order to hold that result at the right weight:

1. **`n_charts = 4` survives the occupancy disqualifier by exactly zero margin.** The rule
   disqualifies a config whose median occupancy is `< 2`. Its median occupancy is `2.00`.
2. **The winning axis has a five-order-of-magnitude across-seed spread at that config**:
   `max cond(g)` of `4.886e+07`, `3.657e+02`, `2.233e+04` across the three seeds. The median
   is not a summary of a tight cluster; it is the middle of a set whose extremes differ by a
   factor of ~10^5.
3. **The seed that pulls the median down is a collapsed fit.** Seed 20260814 has argmax
   occupancy `1` — every held-out point routed to a single chart — and correspondingly a
   near-constant `cond(g)` (median, p90, p99 and max all `3.657e+02`). A degenerate
   parameterization is well-conditioned *because* it is degenerate. It also has the worst
   reconstruction in the entire nine-cell grid (`2.4382e-04`).

This is not a defect in the rule's application — the rule was fixed in advance and applied
unchanged, which is exactly the discipline that replaces a gate here. It is a statement about
what the rule's own answer rests on, recorded because 03-09 computes the phase's deliverable
on this config.

### The Swiss roll's winner played no part

The Step-1 Swiss roll sweep selected **`n_charts = 2`** at `N_POINTS = 3000`
(`03-02-SUMMARY.md`, carried into `03-06-SUMMARY.md`). The PU selection above reads only the
four model-side D-07 diagnostics computed on PU data; no roll quantity appears in
`apply_selection_rule`, and the roll's winner is not in the swept PU set at all
(`PU_N_CHARTS_SWEEP = (4, 8, 16)`). D-06's separation holds and is visible.

## Task 3 — the D-12 escalation checkpoint

**The trigger fired, on both legs.** Verbatim:

```
best d=20 CAE (n_charts=4) mse_per_dim=0.000173309 vs control mse_per_dim=3.58866e-05 -> loses_reconstruction=True
best H0/H1 bottleneck_norm=(0.6217,0.8451) vs control=(0.2144,0.8247) -> loses_ph_agreement=True
TRIGGER FIRES = True
```

**Outcome: no `d` sweep.** The decision is recorded in full in
`03-NOTE-d12-retirement.md` (commit `505e890`), which retires D-12's comparison outright
rather than choosing among this plan's three options. Its grounds, restated here in this
plan's own words rather than referenced:

- Reconstruction loss and PH bottleneck agreement are **C0** quantities — measures of where
  points land. Curvature is a **C2** quantity — a measure of how the decoder's second
  derivatives behave. Small C0 error does not bound C2 error. `chart_curvature.py`'s own
  worked example: a decoder learning `y = 0.7 a x^2` where truth is `y = a x^2` attenuates
  curvature by 30% with essentially no reconstruction signal.
- Therefore **the trigger is strong evidence about the representation and close to no
  evidence about the curvature field.** Losing both C0 legs is exactly as uninformative
  about curvature fidelity as winning both would have been. That limitation is the reason
  the escalation was declined, and it must appear alongside every Step-2 through Step-4
  number in `03-FINDINGS.md` — not only in a limitations section.
- The retirement is **not** a criterion dropped because it returned an unfavourable answer:
  the answer it returned is recorded above, unhedged, and the note's own section 2 states
  the distinction.

Two further findings the note carries that bind on 03-09:

- **The training objective constrains nothing about the decoder's derivatives.**
  `cae.train_cae` regularizes `model.chart_encoders`; `chart_curvature.chart_decoder_map`
  composes `model.chart_decoders[i]` with `model.embedding_decoder`. Disjoint parameter sets.
  Measured consequence: `cond(g)` reaches `4.886e+07` on this grid against the Swiss roll's
  `1.4`–`8.3` on identical machinery — five to seven orders apart, which destroys roughly
  seven digits of float64 precision in the `g^-1` contraction inside `H = sum_jk g^jk II_jk`.
- **The C0 replacement threshold is DEFERRED, not ratified**, on two open questions (the
  isometry prior would change what reconstruction the CAE achieves; and anchoring a bar to
  the CAE's own measured ceiling is circular). Nothing in this plan settles it.

## Developer-directed addition — `--converge`

Recorded here because it lands in this plan's file and reads this plan's selected config,
and because the ordering matters: it was requested **after** every number above was visible.

The directive: *train the CAE until it succeeds on PU, based off reconstruction loss, then
compute the curvature field.* Resolved at a blocking question to **stopping criterion only**,
with **no pass/fail bar** and **one seed first**:

- The pre-declared selection rule is **not** touched and still owns `n_charts`. `--converge`
  asserts its own `n_charts` against the live rule and raises on disagreement rather than
  silently overriding it. The alternative reading — re-rank the grid on reconstruction, which
  would have moved the answer from 4 to 16 — was declined precisely because it is threat
  **T-3-24** (changing the rule after the numbers are visible).
- The defect it targets is the one measured in `03-NOTE-d12-retirement.md` section 5:
  `train_cae` early-stops on **total** loss (reconstruction + cross-entropy + Lipschitz), so
  the `nc=4` PU fit halts at epoch 30 whether `MAX_EPOCHS` is 40 or 300 — bit-identically,
  the cap was never binding — while its reconstruction is still descending. **No cell in the
  table above is a converged fit**, and they fail to be for two different reasons: four of
  nine early-stopped on total loss at 27–32 epochs, and the other five ran out the 40-epoch
  cap with `early_stopped = false`, meaning the budget ended training rather than a plateau
  of any kind.
- The fix is the narrowest one that removes that mechanism and is applied in this runner's
  own cfg, never in `cae.py`: `early_stop_patience = max_epochs + 1` (structurally inert —
  `plateau_count` increments at most once per epoch, so it cannot be reached within the
  budget) and `wallclock_ceiling_s = inf`. Every model and optimizer constant is the grid's,
  unchanged, so the converged fit differs from a grid cell in how long it trains and in
  nothing else.

Results of that run are recorded in `03-08-SUPPLEMENT-03.md`, not here.

## Deviations from Plan

1. **This SUMMARY was written after the fact**, closing out an already-executed plan. Stated
   at the top rather than implied.
2. **`curv_wall_s` was never recorded per grid cell**, so the acceptance criterion asking for
   it is unmet and the reported grid total is training-only. Recorded rather than
   back-filled: back-filling would mean re-running the grid.
3. **`--select-only` prints three per-config summary rows, not the nine per-fit rows** the
   plan's acceptance criterion named. The nine per-seed rows are reproduced in this SUMMARY
   from the record file directly; the runner's printer was not changed to produce them,
   because editing the reporting path of a rule-applying runner after its numbers are visible
   buys nothing and costs the guarantee that the file was untouched.
4. **Task 3's checkpoint was answered outside this plan**, by `03-NOTE-d12-retirement.md`.
   Its three options were superseded rather than chosen among; the note states that
   explicitly and says 03-08-PLAN.md Task 3 is not rewritten.
5. **`--converge` was added to the runner under a developer directive** that postdates every
   number above.

## Verification

- Nine fit records, no duplicate key: **pass** (`--select-only`: 9 of 9).
- `--select-only` exits 0: **pass**.
- Every record carries `chart_dim = 20` and `activation = silu`: **pass**.
- Every record carries `train_wall_s` and `curv_wall_s`: **fail** — training timing only, see
  Deviation 2.
- Applied rule matches 03-07's pre-grid record: **pass**, and `git diff` shows the ranking
  function and its docstring unedited during this plan.
- No record references a sealed 02.2 artifact stem; no new `cae_seed_2026080*` cache entry:
  **pass**.
- D-12 checkpoint answered and outcome recorded: **pass**, via `03-NOTE-d12-retirement.md`.
- `.venv/bin/python -m pytest notebooks/pu_manifold/tests/`: see Supplement 03.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-16*
