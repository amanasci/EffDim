---
phase: 03-decoder-curvature-field
plan: 02
subsystem: research-notebook
tags: [swiss-roll, chart-autoencoder, curvature, sweep-runner, amendment, sample-size, d-01, d-02, d-04, d-05, d-05a]

# Dependency graph
requires:
  - phase: 03-01
    provides: "swiss_roll_curvature_sweep_run.py -- resumable n_charts x seed sweep runner, ROLL_FLOOR/RAW_BASELINE_CONTEXT/N_CHARTS_SWEEP/TORCH_SEEDS module constants, the n_charts=8 seed=0 tracer reproduction of 02.5-09's rho_chart=-0.0604, D-02/D-05 ratifications"
provides:
  - "The Step-1 gate's answer at the pre-registered configuration (N_POINTS=3000): DOES NOT CLEAR, best config n_charts=2, median rho_chart=0.4347 -- recorded, not retracted"
  - "03-02-AMENDMENT-01.md -- N_POINTS 3000 -> 12000, ratified with an honest partial-knowledge disclosure, control cache preserved"
  - "The Step-1 gate's answer at the amended configuration (N_POINTS=12000): CLEARS -- two statistically indistinguishable configs, n_charts=2 (median 0.8302) and n_charts=8 (median 0.8234), differ by 0.0068"
  - "D-05's monotone-in-charts-used premise -- 02.5-09's central empirical claim and the entire justification for opening n_charts as Phase 3's only measured lever -- is PARTLY FALSIFIED by Phase 3's own n=12000 data: Spearman(n_charts_used, rho) collapses from -0.5586 (p=0.0105, n=3000) to -0.2866 (p=0.221, n=12000, not significant), and per-config medians are no longer monotone (n_charts=8 scores second-highest, not lowest). Flagged for 03-11's requirement re-mint; NOT re-decided here."
  - "A conditioning blow-up (cond_max=37770.88, ~309x the n=3000 control's own maximum of 122.22) at n_charts=5, seed=0, n=12000, co-occurring with a high rho_chart=0.8469 -- the exact failure mode T-02.5-20/T-3-31 exist to catch, now observed for the first time in this codebase's own data rather than only as a design-time risk"
  - "--n-points CLI override on swiss_roll_curvature_sweep_run.py, routing to a distinct cache key, with a mixed-n_points resume guard"
affects: ["03-03", "03-09", "03-11"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Amendment discipline scaled to D-15's simplicity mandate: a single markdown file with an explicit ordering disclosure (what was already measured at the moment the amendment was written), committed before the authorized re-run, no PREREGISTRATION.md/ratification-commit/ancestry-proof-script ceremony -- deliberately lighter than 02.5-PREREGISTRATION-AMENDMENT-01.md's format while keeping its one load-bearing property: disclosing what was already known rather than implying blindness."
    - "Distinct cache key per amended constant (03_swiss_roll_curvature_sweep_n12000.jsonl vs the sealed 03_swiss_roll_curvature_sweep.jsonl), with a --resume guard that refuses to combine records written under two different n_points into one table -- extends 02.6-FINDINGS-02.md section 12's constant-consistency pattern to a new axis."
    - "A gate amendment does not retract the miss it responds to -- both grids stay in the record, printable independently via --record-path, so a later reader sees the control and the amended result side by side rather than only the second, more favorable number."

key-files:
  created:
    - .planning/phases/03-decoder-curvature-field/03-02-AMENDMENT-01.md
  modified:
    - notebooks/diagnostics/swiss_roll_curvature_sweep_run.py

decisions:
  - "D-01/D-04 applied exactly as ratified: median rho_chart over 5 torch seeds is the Step-1 statistic at both n=3000 and n=12000; the full spread is reported at every config; no mean, no best-seed row, at either grid."
  - "D-02's floor (median rho_chart > 0.65) was never moved, softened, or reinterpreted at any point across the MISS, the amendment, or the CLEAR. RAW_BASELINE_CONTEXT=0.6712 stayed context-only throughout and was never promoted into a gate."
  - "Amendment 1 (N_POINTS 3000 -> 12000), ratified at the checkpoint after the n=3000 MISS: CLAUDE.md's ~3,000-point Swiss roll protocol is written for reconstruction sanity (a zeroth-order property); this gate measures mean curvature (a second-derivative quantity), which needs denser sampling than reconstruction to estimate reliably. Disclosed as NOT a blind amendment -- n_charts=2 and n_charts=3 at n=12000 were already measured (0.8302, 0.5674) when the amendment was written; n_charts=5 and n_charts=8 were genuinely unmeasured."
  - "STEP-1 GATE (final): CLEARS. Two configs clear and are statistically indistinguishable -- n_charts=2 (median 0.8302, spread [0.7271, 0.8712], 5/5 seeds clear) and n_charts=8 (median 0.8234, spread [0.2877, 0.9743], 4/5 seeds clear), differing by 0.0068. Reported as a two-config clear with the D-04 multiple-comparisons caveat materially live (4 configs, 4 shots at the bar, top two tied), never as a clean single winner."
  - "D-05's monotone-in-charts-used premise is partly falsified by this plan's own n=12000 data, and that finding is recorded here rather than silently absorbed: at n=3000 the direction was significant (Spearman -0.5586, p=0.0105) and per-config medians were cleanly monotone-decreasing; at n=12000 the direction is no longer significant (Spearman -0.2866, p=0.221) and n_charts=8 -- the config 02.5-09 measured as worst -- scores second-highest of the four. Atlas fragmentation at n_charts=8 reads as a data-starvation symptom, not an architectural chart-count pathology. This is NOT re-decided here -- D-05 itself is not reopened -- it is flagged for plan 03-11's requirement re-mint, which owns reconciling stale requirement text against measured findings."
  - "The n_charts=5, seed=0, n=12000 cell recorded cond_max=37770.88 against the n=3000 control grid's own maximum of 122.22 (~309x), while still scoring rho_chart=0.8469. Reported explicitly, not averaged into the n_charts=5 per-config summary row, per the plan's own backstop requirement that cond(g) travel beside ||H|| rather than substitute for it. This is the design-time risk named T-02.5-20 (02.5-08's threat register) and T-3-31 (03-09-PLAN.md's threat register for the PU field stage) now observed concretely for the first time in this codebase, not only as an anticipated failure mode."

requirements-completed: [DEC-05, CURV-03, CURV-04]

coverage:
  - id: D1
    description: "Sweep summary layer (--summary mode): full 20-row table per grid, per-config median-of-5-seeds-and-spread rows, context baseline labelled gates-nothing, floor decision against the best swept config with the unconditional multiple-comparisons caveat, and the D-05a stop-and-report branch printed only on a non-clear"
    requirement: "CURV-03"
    verification:
      - kind: other
        ref: ".venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py --dry-run && --smoke && --summary (all exit 0)"
        status: pass
      - kind: unit
        ref: "grep-based acceptance criteria: multiple-comparisons (3 hits), median (27 hits), D-05a (3 hits), no statistics.mean/np.mean/.mean() inside per_config_summary, no new *verdict*.json/*THRESHOLD* files"
        status: pass
    human_judgment: false
  - id: D2
    description: "Full 4x5 sweep executed at the pre-registered N_POINTS=3000: 20 cells, no duplicate keys, n_charts=8 seed=0 anchor reproduces 03-01's -0.06041003026778113 byte-identical, curvature_convention=trace and activation=silu on every cell"
    requirement: "DEC-05"
    verification:
      - kind: other
        ref: "notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl -- 20 records, 20 unique keys, verified independently post-hoc"
        status: pass
    human_judgment: false
  - id: D3
    description: "Task 3 blocking checkpoint: Step-1 gate read at n=3000 (DOES NOT CLEAR), Amendment 1 ratified with honest partial-knowledge disclosure, D-05's monotone-in-charts premise flagged as partly falsified"
    verification: []
    human_judgment: true
    rationale: "The gate decision, the amendment's honesty disclosure, and the decision not to re-decide D-05 (only flag it) are all human/developer judgment calls this plan exists to force -- no automated check substitutes for the developer's own read of the sweep table or their approval of the amendment's scope."
  - id: D4
    description: "Full 4x5 sweep executed at the amended N_POINTS=12000, written to a distinct cache key, control cache (n=3000) verified unmodified: STEP-1 GATE CLEARS at n_charts=2 (median 0.8302) and n_charts=8 (median 0.8234, statistically indistinguishable), with the cond_max=37770.88 outlier at n_charts=5 seed=0 reported explicitly"
    requirement: "CURV-04"
    verification:
      - kind: other
        ref: "notebooks/.cache/03_swiss_roll_curvature_sweep_n12000.jsonl -- 20 records, 20 unique keys, all figures (medians, spreads, cond_max, Spearman) independently recomputed from the raw cache rather than taken from the reported values"
        status: pass
    human_judgment: false

# Metrics
duration: "~1h20min active (Task 1 build + verify, Task 2 n=3000 sweep execution, Task 3 checkpoint round-trip, amendment authoring, --n-points wiring) + ~13h wall-clock for the n=12000 sweep (chunked batches plus an orchestrator-side wait-loop stall that cost no data -- --resume picked up cleanly from the 4 cells already cached)"
completed: 2026-08-14
status: complete
---

# Phase 3 Plan 02: Swiss Roll Sweep, Step-1 Gate Miss, Amendment 1, Step-1 Gate Clear Summary

**The pre-registered Step-1 gate at N_POINTS=3000 MISSED (best median 0.4347 vs floor 0.65); Amendment 1 raised N_POINTS to 12000 and the gate CLEARS, but on two statistically indistinguishable configs (n_charts=2 and n_charts=8, differing by 0.0068) rather than the clean single winner D-05 predicted, and D-05's own monotone-in-charts-used premise is partly falsified by the n=12000 data that cleared the gate.**

## Performance

- **Duration:** ~1h20min active work; ~13h wall-clock for the amended sweep (chunked batches, one orchestrator-side wait-loop stall with no data loss -- `--resume` picked up from 4 already-cached cells)
- **Tasks:** 3 (2 auto tasks + 1 blocking checkpoint, expanded mid-plan by a developer-directed amendment)
- **Files modified:** 2 tracked (`swiss_roll_curvature_sweep_run.py`, `03-02-AMENDMENT-01.md`) + 2 gitignored cache files (`03_swiss_roll_curvature_sweep.jsonl` control, `03_swiss_roll_curvature_sweep_n12000.jsonl` amended)

## Accomplishments

- **Task 1:** Built the `--summary` read-out layer -- full sweep table, per-config median-and-spread rows (D-01), the context baseline labelled gates-nothing (D-02), the floor decision against the best swept config with an unconditional multiple-comparisons caveat (D-04), and the D-05a stop-and-report branch, printed only on a non-clear. Verified against every acceptance criterion; full `pu_manifold` suite green (269 passed).
- **Task 2:** Executed the full 4x5 sweep at the pre-registered `N_POINTS=3000`. 20 cells, no duplicates, anchor cell byte-identical to 03-01.
- **Task 3 (checkpoint):** Read the n=3000 table. **STEP-1 GATE: DOES NOT CLEAR** -- best config `n_charts=2`, median `rho_chart=0.4347`, spread `[-0.0863, 0.7817]`, floor `0.65`. The monotone-in-charts-used direction reproduced directionally (Spearman `-0.5586`, p=0.0105) but was not noise-free.
- **Scope addition, developer-directed (Rule 4, approved at checkpoint):** rather than take the D-05a stop-and-report branch, the developer ratified **Amendment 1** -- `N_POINTS` 3000 -> 12000 -- on the diagnosis that curvature (a second-derivative quantity) needs denser sampling than CLAUDE.md's reconstruction-sanity protocol was designed to provide. The amendment discloses, rather than hides, that it was written with two of the four configs already measured.
- **Amended sweep:** Full 4x5 sweep re-run at `N_POINTS=12000` through the runner's own `--n-points` mechanism (not an ad-hoc script), written to a distinct cache key. **STEP-1 GATE: CLEARS** -- but on two statistically indistinguishable configs, not the single winner the pre-amendment reasoning anticipated.

## Task Commits

1. **Task 1: Sweep summary, median-of-5-seeds statistic, floor application, D-05a branch** - `58b430d` (feat)
2. **Task 2: Execute the full 4x5 Swiss roll sweep at N_POINTS=3000** - no commit (only appends to gitignored `notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl`; no tracked file changed, per D-15's "the printed read-out plus the plan SUMMARY is the whole of the record")
3. **Task 3: Read the sweep table, record the Step-1 gate outcome** - checkpoint, no file changes; recorded in this SUMMARY
4. **Amendment 1 (developer-directed scope addition)** - `c42e052` (docs): `.planning/phases/03-decoder-curvature-field/03-02-AMENDMENT-01.md`, committed before any `n=12000` sweep number existed in the resumable cache
5. **`--n-points` mechanism** - `a014c77` (feat): `swiss_roll_curvature_sweep_run.py` gains the amended override, a distinct default cache key, and a mixed-`n_points` resume guard

**Plan metadata:** committed with this SUMMARY (see final commit below).

## Files Created/Modified

- `notebooks/diagnostics/swiss_roll_curvature_sweep_run.py` -- Task 1 adds `print_full_sweep_table`, `per_config_summary`, `print_per_config_summary`, `print_context_baseline`, `print_floor_decision`, `print_d05a_branch`, `summarize`, wired to a new `--summary` flag (standalone, and automatic as the trailing step of a full run). The amendment adds `N_POINTS_AMENDED = 12000` (documented, not the default), a `--n-points` CLI override passed through to `run_cell()`, a distinct default cache key when `n_points != N_POINTS`, and a `--resume` guard against silently mixing two `n_points` values in one record file. `N_POINTS = 3000` is unchanged in source, still the value `raw_baseline_context()` measures against.
- `.planning/phases/03-decoder-curvature-field/03-02-AMENDMENT-01.md` -- new. States the n=3000 MISS (not retracted), the diagnosis, the evidence table, what changes (`N_POINTS` only) and what does not (floor, statistic, seeds, swept set, architecture, optimizer, fixture, caveat, D-05a branch), the honesty disclosure of partial prior knowledge, the CLAUDE.md project-level finding, and the re-run scope it authorizes.
- `notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl` -- gitignored, the n=3000 control grid, **preserved unmodified**: still 20 records, still the same 20 unique `(n_charts, seed)` keys, anchor cell still `-0.06041003026778113`.
- `notebooks/.cache/03_swiss_roll_curvature_sweep_n12000.jsonl` -- gitignored, new, the amended n=12000 grid, 20 records, 20 unique keys.

## The Step-1 gate: two readings, both in the record

### Reading 1 — pre-registered configuration, N_POINTS=3000

| n_charts | median rho_chart | spread | seeds clearing 0.65 | median n_charts_used |
|---|---|---|---|---|
| 2 | +0.4347 | [-0.0863, +0.7817] | 1/5 | 2.0 |
| 3 | +0.2549 | [+0.0168, +0.5171] | 0/5 | 3.0 |
| 5 | +0.1351 | [+0.0353, +0.2732] | 0/5 | 5.0 |
| 8 | -0.0604 | [-0.2320, +0.8665] | 1/5 | 7.0 |

**STEP-1 GATE (n=3000): DOES NOT CLEAR.** Best config `n_charts=2`, median `0.4347` vs `ROLL_FLOOR=0.65`. Multiple-comparisons caveat: 4 configs swept, floor applied to the best of them. Context baseline `rho=0.6712`, labelled gates-nothing. This result is **not retracted or revised** by anything below -- the cache file backing it is byte-for-byte the same 20 records as when Task 3's checkpoint read it.

### Reading 2 — amended configuration, N_POINTS=12000 (Amendment 1)

| n_charts | median rho_chart | spread | seeds clearing 0.65 | median n_charts_used |
|---|---|---|---|---|
| **2** | **+0.8302** | [+0.7271, +0.8712] | **5/5** | 2.0 |
| 3 | +0.5674 | [+0.4728, +0.7821] | 1/5 | 3.0 |
| 5 | +0.5481 | [+0.1543, +0.9215] | 2/5 | 4.0 |
| **8** | **+0.8234** | [+0.2877, +0.9743] | **4/5** | 4.0 |

**STEP-1 GATE (n=12000): CLEARS.** Best config `n_charts=2`, median `0.8302`, all five seeds individually above the floor. **`n_charts=8` also clears**, median `0.8234` -- **0.0068 below `n_charts=2`**, well inside the seed-level noise both configs show (spreads of 0.144 and 0.686 respectively). Context baseline unchanged at `rho=0.6712`, still labelled gates-nothing. Multiple-comparisons caveat: 4 configs swept, floor applied to the best of them -- **now materially live**, not decorative, because the top two are tied. This is reported as a **two-config clear**, never as a clean single winner.

## Finding 1 — D-05's monotone-in-charts-used premise is partly falsified

`D-05` (03-CONTEXT.md) opened `n_charts` as Phase 3's only measured lever across the Phase 02.3 hold boundary, on the strength of `02.5-09`'s finding that `rho_chart` was monotone in charts actually used (3 -> 0.8665, 5 -> 0.4250, 8 -> -0.0604/-0.1444). That finding **does not survive at adequate sample size**:

| | n=3000 | n=12000 |
|---|---|---|
| Per-config medians (nc=2,3,5,8) | [0.4347, 0.2549, 0.1351, **-0.0604**] | [0.8302, 0.5674, 0.5481, **+0.8234**] |
| Monotone-decreasing in n_charts? | **True** | **False** -- nc=8 is second-highest, not lowest |
| Spearman(n_charts_used, rho_chart), all 20 cells | -0.5586 (p=0.0105, significant) | -0.2866 (p=0.221, **not significant**) |

`n_charts=8` -- the config `02.5-09` measured as worst, and the config that motivated D-05's entire rationale -- scored the *second-highest* median at `n=12000`, with 4 of 5 seeds clearing the floor. Atlas fragmentation at high `n_charts` reads as a **data-starvation symptom**, not an intrinsic chart-count pathology of the chart-decoder curvature chain. **This finding is recorded here, plainly, and is NOT used to reopen or re-decide D-05 in this plan** -- D-05 itself stays exactly as ratified in `03-01`. It is flagged for plan `03-11`, which owns reconciling stale requirement text against measured findings and re-minting requirements; `03-11` should read this section before re-minting any requirement whose text assumes 02.5-09's monotone direction as settled.

## Finding 2 — two statistically indistinguishable configs clear, not one

`n_charts=2` (0.8302) and `n_charts=8` (0.8234) differ by **0.0068** -- an order of magnitude smaller than either config's own seed spread (0.144 and 0.686 respectively). The D-04 multiple-comparisons caveat -- "N configs give N shots at a fixed bar" -- is not decorative here: with the top two configs tied, a different seed draw at either config could plausibly flip which one nominally "wins." Downstream consumers of this plan's result (in particular `03-11`'s requirement re-mint and any plan choosing a PU `n_charts` value) must **not** read this as "n_charts=2 is the validated winner" -- it is one of two configs the pipeline recovers a known answer at, and D-06 already forbids using anything measured on the roll to pick the PU `n_charts` value regardless.

## Finding 3 — a conditioning blow-up, reported explicitly per the plan's own backstop requirement

Cell `n_charts=5, seed=0, n=12000` recorded `cond_max=37770.88` against the n=3000 control grid's own maximum across all 20 cells of `122.22` -- a **~309x jump** -- while still scoring `rho_chart=0.8469`, one of the higher values in its own config's spread. This is precisely the threat named at design time: `T-02.5-20` (`02.5-08-PLAN.md`'s threat register, "a near-singular pullback metric producing spurious large curvature") and its Phase-3 restatement `T-3-31` (`03-09-PLAN.md`, "near-singular points averaged into the reported field"). It is the **first time this codebase has observed the failure mode concretely** rather than only anticipating it. Per the plan's own backstop truth ("`cond(g)` alongside `||H||`, never substituted for it"), this cell is named here explicitly and is **not averaged into the `n_charts=5` per-config summary row** -- that row's median (0.5481) is reported as computed, with this outlier called out separately rather than silently smoothed over. `03-09`'s `COND_FLAG_PERCENTILE=99.0` near-singular flagging mechanism is exactly the machinery designed to catch this class of point at PU scale; this cell is a concrete instance to check that mechanism against when `03-09` runs.

## Control cache integrity

`notebooks/.cache/03_swiss_roll_curvature_sweep.jsonl` (the n=3000 control) was independently re-verified after the amended sweep completed: still 20 records, still 20 unique `(n_charts, seed)` keys, `n_charts=8 seed=0` still `-0.06041003026778113`. `N_POINTS = 3000` remains visible in `swiss_roll_curvature_sweep_run.py`'s source as the original pre-registered value, commented as superseded-for-the-default-sweep rather than deleted, exactly as Amendment 1 Section 4 requires.

## Decisions Made

See frontmatter `decisions:` for the verbatim record. In one sentence each:

- **D-01/D-04 applied unchanged at both grids:** median over 5 seeds with full spread, no mean, floor applied to the best-of-swept-config with the caveat named.
- **D-02's floor never moved:** `0.65` throughout; `0.6712` stayed context-only throughout.
- **Amendment 1 ratified with disclosed partial knowledge:** `N_POINTS` 3000 -> 12000, honestly stating two of four configs were already measured when the amendment was written.
- **Final gate: CLEARS, on two tied configs, not one.**
- **D-05's premise partly falsified, flagged for 03-11, not re-decided here.**
- **Conditioning outlier reported explicitly per the backstop truth, not smoothed into a per-config number.**

## Deviations from Plan

### Auto-fixed Issues

None during Tasks 1-2 -- both executed as written, verified against every stated acceptance criterion.

### Architectural change (Rule 4, developer-approved at the Task 3 checkpoint)

**1. [Rule 4] N_POINTS amendment and full sweep re-run at n=12000**
- **Found during:** Task 3's checkpoint, after reading the n=3000 STEP-1 GATE: DOES NOT CLEAR result.
- **Issue:** The pre-registered `N_POINTS=3000` (inherited from CLAUDE.md's reconstruction-sanity Swiss roll protocol) appeared to be a sample-size floor rather than a genuine pipeline defect, based on diagnostic evidence gathered outside this plan's pre-registered scope.
- **Change:** Rather than take the plan's own D-05a stop-and-report branch on the n=3000 MISS, the developer ratified Amendment 1 (`N_POINTS` 3000 -> 12000) and authorized a full re-run under the runner's own provenance, with the n=3000 control preserved and an honest disclosure of what was already known when the amendment was written.
- **Files modified:** `.planning/phases/03-decoder-curvature-field/03-02-AMENDMENT-01.md` (new), `notebooks/diagnostics/swiss_roll_curvature_sweep_run.py` (`--n-points` mechanism).
- **Verification:** n=3000 control cache re-verified unmodified after the amended sweep (20 records, byte-identical anchor); n=12000 sweep independently re-derived from raw cache (medians, spreads, seed-clear counts, Spearman correlations, max cond) rather than taken on trust from the numbers reported at ratification.
- **Committed in:** `c42e052` (amendment), `a014c77` (`--n-points` mechanism).

---

**Total deviations:** 1 (Rule 4, developer-approved, not autonomous).
**Impact on plan:** Materially changes the plan's output -- from a D-05a stop-and-report MISS to a CLEAR, but a two-config CLEAR with D-05's own premise partly falsified, not a clean confirmation of the pre-amendment expectation. All follow-on scope (`03-11`'s re-mint) is directed to read Findings 1-2 before proceeding, per the coordinator's explicit instruction not to re-decide D-05 within this plan.

## Issues Encountered

An orchestrator-side wait-loop, external to this plan's own execution, stalled for approximately 13 hours while monitoring the backgrounded n=12000 sweep batches. No data was lost -- the resumable cache and `--resume`'s per-`(n_charts, seed)` key meant the sweep picked up cleanly from the 4 cells already cached at the point of the stall and completed the remaining 16 without re-measuring anything. Both grids were independently re-verified against the raw cache files after the stall resolved, rather than accepted from the completion report alone.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

- The Step-1 gate CLEARS at `n_charts` in `{2, 8}` at `N_POINTS=12000`. Plans `03-03` onward are unblocked to proceed (the D-05a stop-and-report branch is not taken).
- **`03-11` must read this SUMMARY's Findings 1-2 before re-minting any requirement whose text assumes `02.5-09`'s monotone-in-charts direction as settled** -- that direction does not survive at adequate sample size, and the D-05 premise built on it is partly falsified, though D-05 itself (the scope-opening decision) is not reopened here.
- **`03-09`'s near-singular flagging mechanism (`COND_FLAG_PERCENTILE=99.0`) has a concrete calibration case** from this plan's data: `n_charts=5, seed=0, n=12000`, `cond_max=37770.88`, `rho_chart=0.8469`.
- **D-06 still governs:** nothing measured on the roll -- including which of the two tied configs "wins" -- selects the PU `n_charts` value. The PU sweep gets its own independent selection per D-07's four model-side diagnostics.
- The `N_POINTS=3000 -> 12000` finding is a project-level one: any future model routed through CLAUDE.md's Swiss roll protocol whose sanity check depends on second-order structure (curvature estimators, decoder parameterizations reporting derivatives) should expect the same undersampling risk. CLAUDE.md itself is not edited by this plan -- that is a maintainer decision, recorded here as a finding rather than acted on unilaterally.
- No blockers.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-14*
