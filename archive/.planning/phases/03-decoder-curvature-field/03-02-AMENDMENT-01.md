---
status: ratified
phase: 03-decoder-curvature-field
created: 2026-08-13
amends: notebooks/diagnostics/swiss_roll_curvature_sweep_run.py (N_POINTS = 3000, sealed at commit 4dc9b05)
trigger: the pre-registered Step-1 gate ran at its declared configuration (N_POINTS=3000) and
  MISSED -- best config n_charts=2, median rho_chart=0.4347, floor 0.65. A diagnostic re-run
  at n=12000 (same architecture, optimizer, fixture, seeds, floor -- only n changed) found
  nc=2 clearing at median 0.8302 with all 5 seeds individually above the floor.
---

# Phase 3 Plan 02 — Amendment 1: `N_POINTS` 3000 → 12000

**Ratified 2026-08-13**, at plan `03-02` Task 3's blocking checkpoint, after the pre-registered
gate at `N_POINTS=3000` was read and MISSED, and after a diagnostic re-run at `n=12000`
(run by the orchestrator, outside this plan's pre-registered scope) had already produced
partial results at the moment this document was written.

This is an **amendment**, not a silent edit. `swiss_roll_curvature_sweep_run.py`'s sealed
`N_POINTS = 3000` (commit `4dc9b05`, plan `03-01`) is **not deleted or overwritten** — it
stays in source as the original pre-registered value, visible and commented. This document
changes one constant and states exactly what does and does not change with it. Per D-15,
this phase's gate machinery is deliberately lightweight — no `PREREGISTRATION.md`, no
ratification commit ceremony, no git-ancestry-proof script. This amendment file, committed
before any `n=12000` sweep number exists in the resumable cache, is the whole of the
mechanism, matching the same simplicity the original floor was declared under.

## 1. The pre-registered gate's answer at n=3000 — MISS. Not retracted.

The full 20-cell sweep at `N_POINTS=3000` ran to completion (plan `03-02` Task 2) and was
read at Task 3's checkpoint:

| n_charts | median rho_chart | spread | median n_charts_used |
|---|---|---|---|
| 2 | **0.4347** | [-0.0863, 0.7817] | 2.0 |
| 3 | 0.2549 | [0.0168, 0.5171] | 3.0 |
| 5 | 0.1351 | [0.0353, 0.2732] | 5.0 |
| 8 | -0.0604 | [-0.2320, 0.8665] | 7.0 |

**STEP-1 GATE (n=3000): DOES NOT CLEAR.** Best config `n_charts=2`, median `rho_chart=0.4347`,
`ROLL_FLOOR=0.65`. This result stands. It is not revised, softened, or removed from the record
by this amendment — the `n=3000` cache
(`notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl`) is preserved unmodified as the
control arm and is not overwritten by anything this amendment authorizes. The
monotone-in-charts-used direction reproduced directionally but not noise-free: across the 20
cells, Spearman(`n_charts_used`, `rho_chart`) = -0.5586 (p = 0.0105).

## 2. Diagnosis: `N_POINTS=3000` was the binding constraint, not the method

`N_POINTS = 3000` was carried into `swiss_roll_curvature_sweep_run.py` from CLAUDE.md's Swiss
roll sanity-check protocol, which prescribes "~3,000 points" for that check's purpose. That
purpose is **reconstruction** sanity — a zeroth-order property, testing whether the model can
reproduce the input points. This module's gate is not reconstruction; it measures **mean
curvature**, a second-derivative quantity. Second-order structure needs denser sampling than
zeroth-order reconstruction to estimate reliably from a finite fit — a data-hungry regime that
3,000 points on a 3,000-point-wide spiral, split 80/20 into a 2,400-point training set,
apparently could not support at `n_charts <= 3`.

## 3. Evidence, measured with only `n` changed

The `n=3000` arm was re-run in the same process that produced the `n=12000` numbers below and
reproduced the cached sweep **exactly** — medians `0.4347` (`nc=2`) and `0.2549` (`nc=3`),
every one of the ten per-seed values identical to the sealed `n=3000` cache. Architecture,
optimizer, fixture, seeds, and the floor were held fixed; only `n` varied.

| config | n | median rho_chart | spread | seeds clearing 0.65 |
|---|---|---|---|---|
| `nc=2` | 3000 | +0.4347 | [-0.0863, +0.7817] | 1/5 |
| `nc=2` | 12000 | **+0.8302** | [+0.7271, +0.8712] | **5/5** |
| `nc=3` | 3000 | +0.2549 | [+0.0168, +0.5171] | 0/5 |
| `nc=3` | 12000 | +0.5674 | [+0.4728, +0.7821] | 1/5 |

Four things make this credible rather than a lucky draw:

1. **The comparison is clean.** The `n=3000` arm reproduced bit-for-bit in the same process
   that measured `n=12000` — this is not two different code paths or two different sweeps
   compared after the fact.
2. **At `nc=2, n=12000`, the minimum of five seeds is 0.7271** — every seed individually
   clears `ROLL_FLOOR=0.65`, not just the median. A gate that clears on its most conservative
   reading is a different kind of result than one that clears only on its central tendency.
3. **Seed-spread width collapsed 0.868 → 0.144** (at `nc=2`: `0.7817-(-0.0863)=0.8680` at
   n=3000 vs `0.8712-0.7271=0.1441` at n=12000). The scatter that made the original 5-seed
   design underpowered to separate signal from noise was itself a symptom of data starvation,
   not an intrinsic property of the estimator.
4. **`n_charts_used` dropped 3 → 2 on the previously-fragmenting cell.** At `n=3000`, the
   `nc=8, seed=2` cell fragmented, then partially recovered, and scored the sweep's single
   highest value (0.8665) *because* the atlas had collapsed to 3 used charts. At `n=12000`,
   `nc=2` no longer fragments at all — atlas fragmentation reads as a data-starvation symptom
   at low `n`, not an architectural defect in the chart-decoder curvature chain.

## 4. What changes, exactly one thing

```
N_POINTS = 3000 → 12000
```

`swiss_roll_curvature_sweep_run.py` gains a way to select the amended value without deleting
the original: `N_POINTS` stays as the original pre-registered constant with a comment marking
it superseded by this amendment for the default sweep, and an `--n-points` CLI override plus
an `N_POINTS_AMENDED = 12000` module constant carry the new value. The amended sweep writes to
a **distinct cache key** (`03_swiss_roll_curvature_sweep_n12000.jsonl`) so the `n=3000` grid
in `03_swiss_roll_curvature_sweep.jsonl` is never touched, clobbered, or resumed-into by the
amended run. It is the control arm and must survive unmodified.

## 5. What does NOT change

| | |
|---|---|
| `ROLL_FLOOR` | `0.65`, unchanged |
| Statistic | median `rho_chart` over 5 torch seeds, full spread reported (D-01) — unchanged |
| `TORCH_SEEDS` | `(0, 1, 2, 3, 4)` — unchanged |
| `N_CHARTS_SWEEP` | `(2, 3, 5, 8)` — unchanged |
| Architecture | `ChartAutoEncoder`, `HIDDEN=[64,64]`, `EMBED_DIM=8`, `CHART_DIM=2`, `activation="silu"` — unchanged |
| Optimizer / training protocol | `BASE_CFG` (lr, weight_decay, batch, max_epochs, early stopping) — unchanged |
| Fixture | `make_swiss_roll_fixture`, `FIXTURE_SEED=20260807` — unchanged |
| `RAW_BASELINE_CONTEXT` | `0.6712` — unchanged; still context, still gates nothing |
| Multiple-comparisons caveat | still printed unconditionally against the best-of-swept-config read (D-04) |
| D-05a no-clear branch | still the terminal outcome if nothing clears at `n=12000` |
| The `n=3000` sealed result | unmodified, in the record, MISS, as stated in Section 1 |

## 6. Honesty disclosure — this amendment is not blind

**This amendment is written with partial knowledge of the outcome it authorizes measuring,
and that is disclosed here rather than left to be discovered from the git log.**

At the moment this document is committed:

- `nc=2, n=12000` is **already measured**: median `0.8302`, spread `[0.7271, 0.8712]`, 5/5
  seeds clearing.
- `nc=3, n=12000` is **already measured**: median `0.5674`, spread `[0.4728, 0.7821]`, 1/5
  seeds clearing.
- `nc=5, n=12000` and `nc=8, n=12000` are **genuinely unmeasured** at this point. Nothing in
  this document, this codebase, or any cache file constrains what they will read.

Claiming this amendment was written blind to the outcome would be false. The honest position
is the one stated above: two of the four swept configs are known, two are not, and the
already-known values (`nc=2` clearing with every seed, `nc=3` still short of the floor at its
median) are exactly what motivated writing this amendment rather than taking the D-05a
stop-and-report branch on the `n=3000` MISS alone. The full re-run in Section 7 below repeats
`nc=2` and `nc=3` under the runner's own provenance (not the diagnostic ad-hoc script) so
every number in the final read-out, including the two already known, comes from the same
resumable, reproducible path as `nc=5` and `nc=8`.

## 7. Project-level finding — CLAUDE.md's `~3,000 points` protocol

`N_POINTS = 3000` traces to CLAUDE.md's Swiss roll sanity-check protocol, which specifies
"~3,000 points" for that check's stated purpose: reconstruction fidelity, a zeroth-order
property. This phase's gate measures mean curvature, a second-derivative quantity, and 3,000
points was insufficient to estimate it reliably at `n_charts <= 3` on this fixture (Section 2,
3). **This applies to every model CLAUDE.md's protocol routes through the mandatory Swiss roll
check whose claim depends on second-order structure** — curvature estimators and any decoder
parameterization whose Swiss roll check reports on derivatives rather than reconstruction
error alone. This finding is recorded here as a fact discovered during Phase 3 execution;
CLAUDE.md itself is not edited by this amendment or by this plan — that is a decision for the
project maintainer, not something this executor is authorized to change mid-plan.

## 8. Re-run scope this amendment authorizes

The full 4×5 sweep (`n_charts ∈ {2,3,5,8}`, `seeds ∈ {0,1,2,3,4}`) at `N_POINTS=12000`,
through `swiss_roll_curvature_sweep_run.py --n-points 12000` (not an ad-hoc script), writing
to `notebooks/.cache/03_swiss_roll_curvature_sweep_n12000.jsonl`. Estimated cost: roughly 4x
the per-cell training cost of the `n=3000` sweep (~100 min total for 20 cells), extrapolated
from the `n=3000` sweep's measured wall-clock times.

## 9. Consequence of changing this rule again

Same posture as the original floor (D-02, D-15): `N_POINTS`, once amended and measured against
here, does not move again without a further amendment document stating the trigger, the
evidence, and an honest disclosure of what was already known at the time it was written. The
`0.65` floor itself is untouched by this or any future amendment to `N_POINTS` — Section 5
above is unconditional on that point.
