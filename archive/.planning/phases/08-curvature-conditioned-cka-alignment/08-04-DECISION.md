# 08-04 Task 1 Decision: Ratify Every Phase 8 Pre-Registered Constant

**Date:** 2026-08-27
**Checkpoint:** `08-04-PLAN.md` Task 1 — "Ratify every Phase 8 pre-registered constant — the point
of no return" (`type="checkpoint:decision"`, `gate="blocking"`)

## A standing authorization is not a user response

This checkpoint is `gate="blocking"` and five of the decisions it ratifies (D8-03, D8-09, D8-15,
D8-21, D8-22) are marked one-way in `08-CONTEXT.md`. **A standing overnight authorization, an
`AUTO_CFG`/auto-mode auto-approval, or any prior "keep working" instruction does NOT close this
checkpoint.** This checkpoint is closed only by the developer's own words, given directly, in
response to the orchestrator's presentation of every constant's value and one-line reason. That is
what happened here, and this file records it as such — not as an inferred or assumed approval.

## The developer's decision, verbatim

The developer selected, via the orchestrator's checkpoint prompt on 2026-08-27:

> **Ratify with 37→45 fix** — "Freeze every value as presented, AND extend `_REQUIRED_CONSTANTS`
> to cover all 45 so `assert_preregistered()` guards the five currently-unguarded constants
> (`PLANTED_EFFECT_GRID`, `PLANTED_EFFECT_SEED`, `N_REPEATS`, `NEGATIVE_CONTROL_FIELD`,
> `RECORD_STEM`). Closes the guard-coverage hole in the same commit that creates the freeze."

This is a real human response to a blocking-human-shaped checkpoint (`gate="blocking"` on a
`checkpoint:decision` task carrying five one-way sub-decisions), given directly by the developer,
not a standing authorization and not an auto-approval under `AUTO_CFG`/`AUTO_CHAIN`.

## What this ratifies

**Every value presented in Task 1's `<context>` block is ratified exactly as presented.** This
includes, at minimum:

- **D8-01/D8-02** — `KERNELS = ("linear", "rbf")`, linear carrying the headline verdict, RBF
  gating nothing; the unbiased Song et al. (2012) HSIC form.
- **D8-03 (one-way)** — the two frozen global RBF bandwidths, quoted below at full precision.
- **D8-04** — `SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)`.
- **D8-08** — `S_GRID = (10, 20, 50)`.
- **D8-09 (one-way)** — no headline `S`; clearance required at every grid point.
- **D8-05/D8-10/D8-11** — three `||H||` tertiles, `CKA(tertile 3) - CKA(tertile 1)`, middle
  tertile non-gating, the within-stratum label-permutation null with `N_PERMUTATIONS = 1000`,
  `PERMUTATION_SEED = 20260827`, `NULL_QUANTILE_PER_TAIL = 0.975` (two-tailed),
  `NULL_KERNELS = ("linear", "rbf_sigma")`.
- **D8-13/D8-14** — `D_SWEEP = (20, 25, 32)`, `SEED_FIELD_D = 25`,
  `TORCH_INIT_SEEDS = (0, 1, 2)`, per-`d` verdicts reported independently, 18 cells (the S-grid
  axis crossed with the seed axis at all three `S` values — the resolution of `08-RESEARCH.md`
  Open Question 1, departing from that document's recommendation to restrict the seed axis to a
  single headline `S`, because D8-09 leaves no headline `S` to restrict to).
- **D8-15 (one-way)** — unanimous 3-of-3 or nothing, seeds never pooled; `SEED_HANDLING_RULE`
  carries `05-03-DECISION.md`'s ratified never-pool constraint verbatim.
- **D8-18/D8-19** — `PLANTED_EFFECT_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)`,
  `PLANTED_EFFECT_SEED = 20260827`, `N_REPEATS = 30`, `NEGATIVE_CONTROL_FIELD = "h_norm_25"`, run
  at all three `S`.
- **D8-21 (one-way)** — the frozen unconditional reporting block: `REPORTING_BLOCK_ROWS` naming
  exactly five rows (`d32_gap`, `shuffled_h_false_positive_rate`,
  `planted_effect_detection_floor`, `realized_h_contrast_per_s`, `sigma_rungs`); the verdict
  sentence cannot be written without the `d=32` gap and the shuffled-`||H||` false-positive rate
  in the same sentence.
- **D8-22 (one-way)** — the freeze itself: the next commit contains every value above, and after
  it, changing any of them costs a new pre-registration and a full re-run.

No amendment was made to any presented value. The developer's addition is entirely to the guard
mechanism (below), not to any ratified value.

## The 37→45 guard-coverage fix, precisely

The orchestrator confirmed at the time of this checkpoint that the live `cka.py` module declared
37 entries in `_REQUIRED_CONSTANTS`, while the plan's own `<artifacts_this_phase_produces>` block
and Task 2's acceptance criteria state the freeze must leave exactly 45 filled and guarded. The
gap is the eight constants named in `<artifacts_this_phase_produces>` as "born already-frozen"
(`N_REPEATS`, `NEGATIVE_CONTROL_FIELD`, `PLANTED_EFFECT_GRID`, `PLANTED_EFFECT_SEED`,
`REPORTING_BLOCK_ROWS`, `REPORTING_BLOCK_RULE`, `VERDICT_SENTENCE_RULE`, `RECORD_STEM`) — 37 + 8 =
45, reconciling cleanly with no discrepancy to report.

The developer's ratification explicitly authorizes Task 2 to:

1. Declare all eight of these constants (they do not exist in `cka.py` at all before this freeze
   commit — confirmed by `grep` returning no matches for any of the eight names in the pre-freeze
   file).
2. Fill all eight with their ratified values in the same single freeze commit (not left UNSET
   first, per the plan's own "born already-frozen" instruction — nothing before the freeze reads
   them).
3. Add each of the eight to `_REQUIRED_CONSTANTS`, bringing it from 37 to 45 entries.
4. Verify `assert_preregistered()` covers all 45 by construction of the generic UNSET-sweep loop
   (no additional guard branch needed beyond the two existing exact-content checks, since none of
   the eight new constants carries an exact-equality requirement beyond being non-UNSET).

This reconciles to exactly 45 — not a number other than 45 — so no halt is required under the
checkpoint resolution's contingency clause.

## Full-precision sigma values being frozen

Quoted verbatim from `08-03-SUMMARY.md` (measured by `--mode sigma` over all 10,000 points of the
resolved `subsample_20260729_a79b3460b838fd0a.npz` pair, before any subset existed):

- `SIGMA_HSC = 0.6420152563705613`
- `SIGMA_LEGACYSURVEY = 0.5696337821442163`

These are re-measured nowhere in this plan — read directly from `08-03-SUMMARY.md` and
`notebooks/.cache/08_scratch_sigma.jsonl`, both already on disk, and copied into `cka.py` as
literals at Task 2's freeze commit.

## Consequence

After the next commit (Task 2's freeze commit), every one of these 45 constants is a
pre-registration fact. Changing any of them once a Phase 8 number exists is a pre-registration
breach whose only remedy is a fresh freeze and a fresh run — the discipline `02.2`'s sealed FAIL
and `06-PREREGISTRATION-AMENDMENT-01` both establish, never a silent fix.
