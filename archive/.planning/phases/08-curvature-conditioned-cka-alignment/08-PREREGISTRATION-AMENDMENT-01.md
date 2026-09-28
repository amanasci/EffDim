# Phase 8 Pre-Registration Amendment 01 — Cost-Aware Re-Freeze (D8-22 Amendment)

**Date:** 2026-08-28. **Status:** applied. **Raised by:** the executing agent (08-05), on
measuring the frozen constants' real production cost before running any mode.

**This amendment SUPERSEDES the `816863cae2209261470d1d041dcc4484a3056947` freeze in full.**
Every constant that freeze filled remains ratified except the four named below. The freeze commit
that carries this amendment is the commit immediately following this document's own commit in
git history — see that commit's message and `git show --stat` for its SHA, and `cka.py`'s module
docstring, which names it explicitly once it exists. No SHA is quoted here because this document
is, by the commit-ordering discipline it follows, written before that commit exists.

## Why this amendment exists, and why it is legitimate

`08-05-SUMMARY.md` implemented all three Phase 8 production modes (`positive-control`,
`negative-control`, `sweep`) and, before running any of them to completion, measured their real
cost directly against the frozen `816863c` constants:

- One full label-permutation null (`N_PERMUTATIONS = 1000`) at PU's real ~3,333-point pooled
  tertile size costs **~2.14 hours** wall-clock on this machine (8 threads) — two independent
  measurements agreed within 1.3% (**2.131 h** from isolated null timing on synthetic
  production-representative Gram matrices; **2.158 h** extrapolated from one real positive-control
  cell against actual PU data).
- The frozen pre-registration required **129 full-null computations**
  (`S_GRID(3) × PLANTED_EFFECT_GRID(7) = 21` for the positive control, `S_GRID(3) × N_REPEATS(30)
  = 90` for the negative control, `S_GRID(3) × 6 fields = 18` for the sweep), for an estimated
  **~276 hours (~11.5 days)** of continuous unattended compute — roughly 500–750x
  `08-RESEARCH.md`'s own un-piloted "tens of milliseconds per resample" estimate, which that
  document stated explicitly as MEDIUM confidence with no pilot run. That pilot never happened
  before `816863c` sealed `N_PERMUTATIONS = 1000`, `N_REPEATS = 30`, `S_GRID = (10, 20, 50)` and
  the 7-rung `PLANTED_EFFECT_GRID`.
- 08-05 correctly halted rather than starting a multi-day unattended run it could not responsibly
  supervise, and rather than trimming a frozen constant to fit a session — either would have been
  a pre-registration breach or an unsafe commitment. See `08-05-SUMMARY.md` for the full
  measurement and correctness verification.

**Critical integrity fact, verified directly by the orchestrator and re-verified here: NO Phase 8
number exists.**

```
$ ls notebooks/.cache/08_cka_alignment.jsonl
ls: cannot access 'notebooks/.cache/08_cka_alignment.jsonl': No such file or directory
```

`notebooks/.cache/08_cka_alignment.jsonl` was never created by any of the three production modes.
Zero Phase 8 production rows exist anywhere in the tree. **This is what makes the amendment
legitimate rather than a D8-22 breach**: nothing is being retro-fitted to a result anyone has
seen, no threshold is being adjusted because a headline number was inconvenient, and no verdict
rule is touched. The only things this amendment changes are (a) how much compute three
already-implemented, already-verified-correct production modes spend, and (b) one
value-preserving performance fix inside the estimator those modes call — measured and recorded
below, not assumed.

This follows the precedent of `06-PREREGISTRATION-AMENDMENT-01.md` and the `02.4`
amendment documents: an amendment is recorded in its own document, states exactly what changed and
what it could and could not have moved, and — per this amendment's own terms — a fresh freeze
commit and a fresh (not yet run) production record, since unlike the `06` amendment there is no
prior-run record to leave in place.

## The developer's decision, verbatim

The developer was shown the measured cost table above and a menu of budget shapes (run as-is
unattended over multiple days, optimize-then-run, issue a fresh cost-aware freeze, or halt the
phase). They selected **"Fresh cost-aware freeze"**, and within that path, selected the following
option via the orchestrator's checkpoint prompt on **2026-08-28**:

> **"Balanced ~28h (Recommended)"** — "N_PERMUTATIONS=500, N_REPEATS=10, PLANTED_EFFECT_GRID
> 5 rungs. 12.5 permutations per null tail keeps clearance verdicts meaningful; false-positive
> rate readable at 1/10 granularity. ~1.2 days unattended."

This is a real human response to a blocking gate, given directly, in response to the
orchestrator's presentation of the measured cost table and the budget-shape menu — not a standing
authorization and not an auto-approval under `AUTO_CFG`/`AUTO_CHAIN`. (Earlier in the same
session, the same developer had ratified the original `816863c` freeze at `08-04`'s checkpoint,
recorded in `08-04-DECISION.md`, and separately chose "Fresh cost-aware freeze" over running
as-is, optimizing-then-running, or halting the phase entirely.)

## Exactly these four changes, and nothing else

`S_GRID = (10, 20, 50)` **stays untouched** — D8-09's clearance-at-every-`S` requirement is the
anti-retuning discipline this phase is built on and is **not** being relaxed by this amendment.
Every other one of the 45 constants keeps its `816863c`-ratified value. Only the following four
change:

### 1. `N_PERMUTATIONS`: `1000` → `500`

Halves the null's resample count directly. 12.5 permutations land in each of the two tails at
`NULL_QUANTILE_PER_TAIL = 0.975` — the developer's own stated rationale: "keeps clearance verdicts
meaningful."

### 2. `N_REPEATS`: `30` → `10`

Reduces the negative-control's shuffled-`||H||` calibration repeat count. The developer's own
stated rationale: "false-positive rate readable at 1/10 granularity." Negative-control full-null
computations drop from `S_GRID(3) × 30 = 90` to `S_GRID(3) × 10 = 30`.

### 3. `PLANTED_EFFECT_GRID`: `(0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)` → `(0.0, 0.05, 0.10, 0.20, 0.50)`

Positive-control full-null computations drop from `S_GRID(3) × 7 = 21` to `S_GRID(3) × 5 = 15`.

**This specific rung choice was made by the orchestrator, not named by the developer, and is
flagged here explicitly so the developer can object before any run is launched.** The
orchestrator's stated rationale for the five kept/dropped rungs:

- Keeps the `0.0` rung — the null anchor with no injected effect.
- Keeps low/mid resolution (`0.05`, `0.10`, `0.20`) — this is where PU's realized ~1.5x dynamic
  range (measured, not Phase 6's ~20x planted spread) makes the detection floor actually live;
  losing resolution here would blunt the one number (D8-18's detection floor) this control exists
  to report.
- Keeps `0.50` as the strong anchor proving the test *can* detect an effect at all.
- Drops `0.02` — below any plausible detection floor given the measured ~1.5x dynamic range, so a
  clearance or non-clearance reading there was unlikely to be informative either way.
- Drops `0.35` — redundant between the kept `0.20` and `0.50` rungs.

### 4. `unbiased_hsic`'s `term1`: `np.trace(Kt @ Lt)` → `np.sum(Kt * Lt.T)`

08-05 reported this as a *suspected, unfixed* `cka.py` inefficiency without touching the sealed
file (`cka.py` was off-limits to that plan). This amendment is the developer-authorized fix.

**Mathematically identical, not bit-identical — measured, not assumed.** Both forms compute
`sum_ij Kt[i,j] * Lt[j,i]` (the trace of a matrix product equals the sum of the elementwise
product of one matrix with the other's transpose); `np.trace(Kt @ Lt)` computes a full `O(n^3)`
dense matrix product to extract that single scalar, `np.sum(Kt * Lt.T)` computes it directly in
`O(n^2)`.

**Measurement, this session, `n=3333` (PU's real pooled tertile size), `float32`, real
`cka.linear_gram` output on i.i.d. Gaussian data (768-dim, matching PU's ambient dimensionality),
20-repeat average:**

| Quantity | `np.trace(Kt @ Lt)` (old) | `np.sum(Kt * Lt.T)` (new) | Ratio |
|---|---|---|---|
| `term1` value | `-74533.9375` | `-74534.375` | — |
| `term1` wall-clock | 238.48 ms | 33.55 ms | **7.109x faster** |
| full `unbiased_hsic(K, L)` value | `-0.002565930357768186` | `-0.002565969776124436` | — |
| full `unbiased_hsic(K, L)` wall-clock | 367.15 ms | 154.53 ms | **2.376x faster** |

**Relative difference, at full precision:**

- `term1` alone: absolute `0.4375`, relative `5.8698093066665105e-06`.
- Whole `unbiased_hsic` call: absolute `3.94183562501052e-08`, relative
  `1.5362208148311083e-05`.

Both are well inside the `~1e-5` relative acceptance bound this amendment was required to check
before freezing (the whole-call figure sits just above the term1-only figure because `term2`/
`term3` carry their own independent float32 rounding, compounding rather than cancelling — still
materially smaller than any threshold that would change a clearance verdict). **This is NOT a
pure no-op**: the two forms diverge at the ~1e-5 relative level in float32 arithmetic, purely from
floating-point summation order, and this amendment records that honestly rather than claiming
bit-identity. The measured speedups (7.109x on `term1`, 2.376x on the whole call) closely match
the developer's own prior estimate ("7.2x faster on that term and 2.37x on the whole call")
quoted when this change was authorized.

At the ~3,333-point pooled tertile size (D8-08's own `S`-independent measured fact), `term1`'s
`O(n^3)` cost was very likely the dominant driver of the ~7.67s/resample 08-05 measured — this
speedup is the primary reason the recomputed budget below is not simply half of ~276h.

## Recomputed budget

New cell counts: `S_GRID(3) × PLANTED_EFFECT_GRID(5) = 15` (positive control) +
`S_GRID(3) × N_REPEATS(10) = 30` (negative control) + `S_GRID(3) × 6 fields = 18` (sweep) =
**63 full-null computations**, down from 129.

Applying the measured `N_PERMUTATIONS` scaling (`500/1000 = 0.5x` the resample count) and the
measured whole-call `term1` speedup (`2.376x`) to the original per-cell cost
(`2.131 h`, Measurement 1 in `08-05-SUMMARY.md`):

```
63 cells × 2.131 h/cell × (500/1000) / 2.376x  ≈  28.25 hours
```

This closely matches the developer's own back-of-envelope figure quoted at decision time
("~1.2 days unattended", i.e. ~28h) using the round numbers `2.14 h` and `2.37x`
(`63 × 2.14 × 0.5 / 2.37 ≈ 28.44 h`). Both figures agree to within 1%; **~28 hours (~1.2 days) is
the recorded expected budget for all three production modes to run to completion**, down from the
original ~276 hours (~11.5 days) — roughly a 9.8x reduction, driven jointly by the ~2x cell-count
reduction, the 2x `N_PERMUTATIONS` reduction, and the ~2.4x `term1` fix.

This is a resource estimate, not a guarantee; actual wall-clock will still be measured directly
when a production mode is next run (out of scope for this amendment — see `<objective>`: no
production mode is executed here).

## What this amendment does not change

- **No verdict rule, threshold, or gating logic is touched.** `S_GRID`, `NULL_QUANTILE_PER_TAIL`,
  `VERDICT_RULE`, `SEED_HANDLING_RULE`, `TERTILE_STATISTIC_RULE`, `NULL_CONSTRUCTION_RULE`, all
  `*_IS_NON_GATING` flags, `D32_IS_NON_GATING`, and `REPORTING_BLOCK_ROWS`/`REPORTING_BLOCK_RULE`/
  `VERDICT_SENTENCE_RULE` are unchanged.
- **No sigma, kernel, or estimator-form constant beyond `term1` is touched.** `KERNELS`,
  `SIGMA_MULTIPLIERS`, `SIGMA_HSC`, `SIGMA_LEGACYSURVEY`, `HSIC_ESTIMATOR_RULE`,
  `SIGMA_FREEZE_RULE` are unchanged.
- **No structural constant is touched.** `N_TERTILES`, `DENSITY_K`, `DENSITY_FIELD_D`,
  `DENSITY_INPUT`, `D_SWEEP`, `SEED_FIELD_D`, `TORCH_INIT_SEEDS`, `NEGATIVE_CONTROL_FIELD`,
  `PLANTED_EFFECT_SEED`, `PERMUTATION_SEED`, `RECORD_STEM` are unchanged.
- **No test file and no runner logic beyond the freeze-SHA literal is touched by this amendment's
  freeze commit** (`cka.py` only, per commit discipline; the runner and test-suite SHA wiring is a
  separate, third commit).
- **`assert_preregistered()` still passes and all 45 constants remain guarded** by
  `_REQUIRED_CONSTANTS` — the guard-coverage fix from `08-04-DECISION.md`'s ratification is
  unaffected.

## Why this is an amendment and not a post-hoc adjustment

The three changed numeric constants were **not chosen to fit a convenient outcome** — no Phase 8
number existed to be convenient or inconvenient about, verified above. They were chosen from a
directly measured cost table, by the developer, before any production mode ran. The `term1` fix is
**not a free choice** either: both forms compute the same closed-form quantity, and the only
question was whether the measured ~1e-5 relative floating-point difference was small enough to
freeze — checked here, before freezing, per this amendment's own verification requirement. No
threshold, no bucket rule, no verdict criterion, no kernel, and no `S_GRID` value was touched.
