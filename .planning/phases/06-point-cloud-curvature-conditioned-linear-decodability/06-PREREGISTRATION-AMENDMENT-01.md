# Phase 6 Pre-Registration Amendment 01 — `R2_MULTIOUTPUT` was dropped from the freeze

**Date:** 2026-08-24. **Status:** applied, with a re-run. **Raised by:** the executing agent,
after the first `--mode bucketed` run, on noticing that the reported `r2_overall` differed from
Phase 5's while the residuals did not.

## What was wrong

Phase 5 freezes `R2_MULTIOUTPUT = "variance_weighted"` (`linear_probe.py:121`). Phase 6's first
freeze (`c11218c`) **omitted it from the inherited constants block**, and the runner
(`37d1ba8`) passed the literal `"uniform_average"` to `linear_probe.aggregate_r2` instead.

The first `--mode bucketed` run therefore reported `r2_overall = 0.605806` where Phase 5, on the
identical rows, reported `0.643931`.

## What it did and did not affect

**Did not affect the verdict, and this is checkable rather than asserted.**

- `apply_verdict_rule` contains **zero** references to `r2` — grep returns 0. The verdict depends
  only on the per-bucket bootstrap CIs on mean per-point squared-L2 residual, the sign of the
  high-minus-low difference, and the size-matched re-check.
- The residuals themselves were unaffected: Phase 6's `mean_residual_overall` is
  **`0.06642936194948156`**, byte-identical to Phase 5's `0.06642936194948156`. The two phases
  score literally the same 3,000 held-out numbers, which is the condition the whole design rests
  on and which this defect never touched.

**Did affect** one reported diagnostic, `r2_overall`, and — more seriously — it falsified Phase
6's central claim that it inherits Phase 5's protocol unchanged and alters only the field. That
claim is the reason the two phases are comparable, so a silent divergence in it is a real defect
even where the number it moved gates nothing.

## The fix

1. `R2_MULTIOUTPUT = "variance_weighted"` added to `pointcloud_probe.py`'s inherited block, to
   `_REQUIRED_CONSTANTS` (so `assert_preregistered` covers it), and to `describe_inheritance`.
2. The runner now passes `pp.R2_MULTIOUTPUT` rather than a literal.
3. **`test_no_phase_5_scalar_constant_is_silently_dropped` added** — the check that would have
   caught this. Rather than hand-listing what Phase 6 inherits (which is exactly how one gets
   missed), it enumerates every scalar constant in `linear_probe` and requires each to be either
   inherited with an equal value or named in an explicit `deliberately_excluded` set. It passes,
   which establishes that `R2_MULTIOUTPUT` was the only omission.
4. `--mode bucketed` re-run. The superseded first record remains in
   `06_pointcloud_probe_decodability.jsonl` as the first row rather than being deleted; the
   second row is authoritative and is distinguishable by its `inheritance.R2_MULTIOUTPUT` field.

## Why this is an amendment and not a post-hoc adjustment

The corrected value was **not a free choice**. `R2_MULTIOUTPUT` is fixed by Phase 5's sealed
constant; there was exactly one value the fix could take, and it was determined before any Phase
6 number existed. No threshold, no bucket rule, no verdict criterion and no `k` was touched. The
verdict of the first run and the second run is the same string, for the same reasons, on the same
residuals.

This follows the precedent of `02.4-PREREGISTRATION-AMENDMENT-01.md` and `-02.md`: an amendment
is recorded in its own document, states what changed and what it could and could not have moved,
and leaves the superseded artifact in place.
