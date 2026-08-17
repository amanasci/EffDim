---
phase: 03-decoder-curvature-field
plan: 10
subsystem: curvature-instrumentation
tags: [pytorch, chart-autoencoder, curvature, synthetic-controls, conditioning]

# Dependency graph
requires:
  - phase: 03-decoder-curvature-field (03-04)
    provides: synthetic_controls.py -- flat, sphere and saddle fixtures with analytic H at PU's
      working scale, the saddle's arithmetic cross-checked against finite differences
  - phase: 03-decoder-curvature-field (03-09)
    provides: the PU curvature field these controls exist to calibrate, and the converged
      checkpoint the matched protocol is matched TO
provides:
  - notebooks/diagnostics/synthetic_control_run.py, executed -- matched-architecture control
    fits at d=20 and a low-d discriminator at d=4, four uncollapsed fidelity axes each
  - the cond(g) -> artifact-curvature band table: the first quantitative measurement in this
    milestone of how much spurious curvature a given conditioning manufactures
  - an end-to-end validation of chart_curvature.py against analytic H on a varying, mixed-sign
    field, and the localization of the d=20 failure to the CAE's parameterization
affects: [03-11 (the phase record), any resumption of the decoder-prior work]

tech-stack:
  added: []
  patterns:
    - "Conditional artifact analysis: restrict a zero-truth control's measured field to the
       cond(g) band of the target fit, rather than comparing unconditioned distributions"
    - "Low-d discriminator: hold every constant except the intrinsic dimension, to separate a
       broken instrument from an under-sampled regime"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/synthetic_control_run.py
    - notebooks/diagnostics/curvature_field_pu_run.py

key-decisions:
  - "Controls train in 25-epoch blocks: the matched continuous protocol diverges to non-finite
     weights at ~epoch 220. Documented deviation, not a tuning choice."
  - "The flat fixture gets no four-axis comparison -- against an exactly-zero truth, cosine,
     magnitude ratio and calibration slope are all undefined, and the report refuses them."
  - "The sphere gets no rank or calibration axis -- its truth is constant, so rho has no ranks
     and the regression has a zero-variance predictor."
  - "CURV-07 is answered NEGATIVELY and conditionally: the PU field is not validated."

requirements-completed: [CURV-06, CURV-07]

duration: ~24h wall clock across five fits plus three diagnostics
completed: 2026-08-17
status: partial
---

# Phase 3 Plan 10: Synthetic Controls Summary

**The controls did not validate the PU field -- they bounded it, and in doing so they localized the phase's failure precisely: `chart_curvature.py` recovers analytic mean curvature to 0.5-11% on every axis at `d=4`, and returns numbers uncorrelated with truth at `d=20`, with `cond(g)` median 4.10 against 1.9e+08 the only thing that changed.**

## Status: PARTIAL

Task 1 complete and verified. Task 2 executed with one gap: **the flat fixture at `d=20` was
never recorded through the runner.** Its production fit is what discovered the divergence
described in §1, and its numbers come instead from a diagnostic probe (§2). Every other cell
is a full record in `notebooks/.cache/03_synthetic_controls.jsonl`.

`--fixture flat --chart-dim 20` under the blocked protocol would close that gap for ~2.75 h
and has not been run.

## 1. The matched protocol does not survive its own budget

Found before any control number existed, by fitting the easiest possible manifold long enough
to hit it.

A single continuous 300-epoch `train_cae` call on the flat fixture at full scale diverges to
non-finite weights at roughly **epoch 220 (~27,500 optimizer steps)**, surfacing as
`linalg.svd: input matrix contained non-finite values` inside `cae.lipschitz_penalty`'s
spectral-norm product. The identical configuration -- same data, same seed, same
hyperparameters -- trained in 25-epoch blocks completed all 300 epochs cleanly, weights stable
throughout (enc 0.16 -> 0.48, dec 0.22 -> 0.51, emb 0.16 -> 0.64) and reconstruction descending
monotonically.

`cae.train_cae` builds a fresh optimizer per call, so blocking restarts Adam's moment estimates
each block. That is the only systematic difference between the two runs, which points at
accumulated optimizer state rather than weight magnitude. **A hypothesis consistent with the
evidence, not a confirmed cause** -- confirming it needs a continuous run with per-step
instrumentation, which has not been done.

Two consequences that must travel forward:

1. Every control below is trained in blocks, so the controls are **not** trained identically to
   the converged PU fit (one continuous 300-epoch call). Architecture, data scale, `lr`, batch,
   Lipschitz weight and epoch budget all still match; the optimizer-state schedule does not.
2. **The converged PU fit ran deep into the same regime and did not crash.** Its numbers are
   real. But "did not crash" is a weaker guarantee than anyone held when 03-09's field was
   computed, and `03-08-SUPPLEMENT-03.md` should be read with that known.

## 2. Flat at `d=20` -- the noise floor, and the band table

Measured from a diagnostic probe rather than a runner record: 2,000 rows through the
300-epoch blocked checkpoint `notebooks/.cache/03_flat_probe_ep300.pt`. Analytic `||H||` is
**exactly zero at every point**, so every value is artifact and its scale is directly
interpretable.

Unconditioned, the floor looks fatal:

| statistic | measured `||H||` |
|---|---|
| median | 82.56 |
| p95 | 27,482 |
| max | 2.15e+06 |
| mean | 10,378 |

PU's field median is 1359. Read this way the PU numbers sit inside the artifact distribution.

**That reading is wrong, and the reason is the whole point of the fixture.** This flat fit is
far worse conditioned than the PU fit -- `cond(g)` median 4.76e+08 against PU's 9.93e+06, and
**95.3% of its points have `cond(g)` above PU's entire range**. Restricting to the band PU
actually occupies:

| `cond(g)` band | n | `||H||` median | p95 | max |
|---|---|---|---|---|
| ≤ 3.82e+07 (PU's full range) | 94 | **3.87** | 11.24 | 24.50 |
| 3.82e+07 – 1e+09 | 1,183 | 35.02 | 294.6 | 2,929 |
| > 1e+09 | 723 | 1,410 | 146,180 | 2.15e+06 |

**Artifact curvature scales monotonically with `cond(g)` across three decades.** This is the
first quantitative measurement in this milestone of a mechanism the phase had only argued
verbally since `03-NOTE-d12-retirement.md` §4, and it is the most transferable result here: it
says how much conditioning a curvature field can tolerate before its values stop meaning
anything.

At PU's own conditioning the false-positive floor is **3.87**, against PU's median of 1359 --
a **351×** margin, with PU's *minimum* (681) still 28× the artifact's *maximum* (24.50). It
also retroactively justifies two of 03-09's design choices: the 99th-percentile flagging, and
reporting `||H||` jointly with `cond(g)` rather than alone.

Limits: 94 points, from a fit that was itself unconverged (recon still descending at epoch 300)
and trained under the blocked protocol.

## 3. The `d=20` controls -- both curved fixtures fail on every informative axis

| | sphere | saddle |
|---|---|---|
| analytic `||H||` | 4.364 (constant) | 0.0327 median (varying, mixed sign) |
| measured `||H||` median | 255.1 | 274.6 |
| direction median cosine | 0.008875 | −0.000478 |
| magnitude median ratio | 58.46 (CV 8.03) | 9955 (CV 10.15) |
| calibration slope / R² | *undefined* | 370.9 / **0.000002** |
| rank `rho` | *undefined* | **−0.015107** |
| `cond(g)` median | 1.53e+08 | 1.88e+08 |
| holdout `mse_per_dim` | 2.13e-02 | 1.62e-02 |
| charts used | 3 of 4 | 4 of 4 |

The sphere's rank and calibration axes are **undefined, not bad**: its truth is constant, so
Spearman has no ranks to correlate and the regression has a zero-variance predictor. The
record on disk predates the guard that now refuses them and still carries `rho = 0.005` and
`R² = 0.0`; **those two values must not be quoted.** Direction and magnitude remain well posed
against a constant truth, and both are bad.

The saddle carries no such excuse -- its curvature genuinely varies, so all four axes are
meaningful, and all four show **no relationship to truth**.

The saddle's cancellation test fails outright: on points with the lowest 10% true curvature
the pipeline reports median `||H||` **283.1** against an overall **274.6** -- indistinguishable.
The one thing this fixture existed to test, the pipeline is blind to.

Both fits also reconstruct **345-453× worse** than the PU fit (4.71e-05), so they test the
pipeline in a worse regime than PU's. They establish failure *there*. They provide no evidence
it succeeds at PU's regime either, and nothing else does.

## 4. The `d=4` discriminator -- the instrument is correct

Two explanations survived §3, with completely different consequences: a curved `d`-manifold at
`d=20` with `n=10,000` is drastically under-sampled and no method recovers its curvature; or
the curvature path itself is defective. `--chart-dim 4` separates them by holding everything
else fixed. **At `d=4` a sphere and a saddle are densely covered by 10,000 points.**

**These are diagnostics, NOT matched controls, and must never be quoted as controls for the PU
field.** The runner prints a `DIAGNOSTIC MODE` banner and keys the dimension into every
`config_id` for exactly that reason.

| axis | saddle `d=4` | saddle `d=20` | sphere `d=4` | sphere `d=20` |
|---|---|---|---|---|
| direction cosine | **0.992725** | −0.000478 | **0.999893** | 0.008875 |
| magnitude ratio | **0.934713** | 9955 | **0.999747** | 58.46 |
| calibration R² | **0.979850** | 0.000002 | undefined | undefined |
| rank `rho` | **0.988709** | −0.015107 | undefined | undefined |
| median rel. error | **0.1063** | 9954 | **0.005375** | 57.46 |
| `cond(g)` median | **4.098** | 1.88e+08 | **2.076** | 1.53e+08 |
| holdout `mse_per_dim` | **5.11e-07** | 1.62e-02 | **1.25e-06** | 2.13e-02 |
| charts used | 1 of 4 | 4 of 4 | 1 of 4 | 3 of 4 |

**`chart_curvature.py` is validated.** The sphere recovers a constant analytic curvature to
four significant figures (1.788392 against 1.788845, 0.54% relative error). The saddle -- a
*varying, mixed-sign* field, where rank and calibration are both meaningful -- recovers it at
`rho = 0.989`, `R² = 0.980`, calibration slope 0.927, median relative error 10.6%.

This matters beyond the controls. This phase was the first ever to edit `chart_curvature.py`,
and plan 03-05's forward-versus-reverse test compares two autodiff paths and structurally
cannot see a bug they share. This is an independent analytic check on a non-trivial field, and
it passes.

The saddle's cancellation test also passes at `d=4`: median `||H||` **0.0200** on the
lowest-10%-true-curvature points against **0.1235** overall, a 6.2× drop correctly tracking
genuine trace cancellation -- the same test the `d=20` fit was blind to.

One number not to over-read: the `d=4` saddle's magnitude **CV = 9.23** beside a median ratio
of 0.935. It is dominated by the cancellation points themselves, where true `||H|| -> 0` makes
the per-point ratio blow up by construction. The calibration slope (0.927, R² 0.980) and the
median relative error (10.6%) are the trustworthy magnitude reads.

**Both `d=4` fits used 1 of 4 charts** -- a single chart covered the whole manifold, with no
fragmentation and no seams. The `d=20` fits fragmented across 3-4 charts. That is a cheap
alternative hypothesis worth one arm before assuming a regularizer is the only lever.

## 5. CURV-07, answered

**Is the measured PU curvature a property of the data manifold or an artifact of the fitted
decoder?**

**Neither has been established. The PU field is not validated.** Stated conditionally, as the
plan requires: no PASS exists anywhere upstream in this milestone, so this answer rests on the
gate override and must never be read as if the parameterization had been independently
validated.

What the evidence does support, precisely:

1. **It is not conditioning artifact.** At PU's own `cond(g)`, the pipeline's false-positive
   floor is 3.87 against PU's 1359 -- a 351× margin (§2).
2. **The instrument is correct.** Given a well-conditioned parameterization, the pipeline
   recovers analytic curvature on a varying mixed-sign field at `rho = 0.989`, `R² = 0.980`
   (§4). Any remaining failure is upstream of the curvature computation.
3. **PU's own accuracy is untested.** Every fixture with known curvature that reached PU's
   dimension failed to *train* to PU-comparable quality (`cond(g)` 15-19× worse, reconstruction
   345-453× worse). **There is no curved control at PU's conditioning**, so nothing bounds the
   accuracy of PU's `||H|| = 1359` -- neither its magnitude nor, after the saddle's `rho`,
   its ordering.

That third point is the gap, and it is structural rather than an oversight: producing such a
control requires a CAE that fits a *curved* 20-manifold as well as it fits PU, and no fit in
this milestone has done that.

**And the caveat that outranks all of it, stated here rather than in a limitations section:**
these fixtures fit cleanly and therefore never reproduce the atlas fragmentation `02.5-09`
measured on real data. A control that passes establishes only that the pipeline is correct on
a manifold that is easy to fit. It cannot rule out parameterization damage on PU, because the
one failure mode the gate override worries about is precisely the one a cleanly-training
synthetic manifold cannot exhibit.

## 6. What this licenses next

The failure is localized to the parameterization the CAE produces at `d=20`: `cond(g) ~ 10⁸`
where the validated instrument needs `~10⁰`-`10⁷`. Two levers, and the evidence favours testing
them together rather than assuming the first:

- **The second-order Christoffel prior** (`decoder_priors.py` mode `"christoffel"`, added
  2026-08-17). Penalizes the tangential part of `D²F` -- pure parameterization content -- and
  provably not the normal part, which is the curvature. Built and unit-proven, **never run**:
  there is no evidence yet that it lowers `cond(g)`, and it will cost reconstruction.
- **Chart count.** Both `d=4` fits used one chart and reached `cond(g) ~ 2-4`; both `d=20` fits
  fragmented and reached `~10⁸`. Fewer charts may buy much of the conditioning for free.

The `d=20` sphere or saddle is now the right test bed for either, because it has known truth
*and* currently fails -- so cosine, magnitude ratio, `R²` and `rho` moving toward 1 as `cond(g)`
falls is a directly readable mechanism check rather than a guess.

## Deviations from Plan

1. **Flat at `d=20` is not a runner record** -- its production fit discovered the divergence.
   Numbers come from a 2,000-row diagnostic probe. Gap named in §Status.
2. **Controls train in 25-epoch blocks**, not one continuous call (§1).
3. **The flat fixture gets no four-axis comparison** -- undefined against an exactly-zero truth.
4. **The sphere gets no rank or calibration axis** -- undefined against a constant truth. Its
   stored record predates the guard and carries misleading values; §3 states they must not be
   quoted.
5. **The `d=4` diagnostics are outside the plan's scope**, added to separate a broken instrument
   from an under-sampled regime. Guarded against being quoted as controls.
6. **`--epoch-block`, `--chart-dim` and `--embed-dim` were added to the runner** for the above.

## Verification

- `--dry-run` exits 0, names all three fixtures, the matched architecture, the four axes, the
  damage caveat and the protocol deviation: **pass**.
- `--smoke` exits 0 on all three fixtures, flat reporting NOT APPLICABLE and sphere reporting
  `rho=UNDEFINED(const truth)`: **pass**.
- `CONTROL_FIXTURES`, `curvature_field_pu_run`, `assert_c2_decoder`,
  `curvature_fidelity_report` all present uncommented: **pass**.
- No arithmetic combines the four axes into one score: **pass**.
- `pytest notebooks/pu_manifold/tests/ -q`: **307 passed, 1 skipped**.
- `test_synthetic_controls.py` including the saddle finite-difference cross-check (Task 2's
  precondition): **8 passed**.
- Three matched controls at `d=20` recorded: **not met** -- flat is a diagnostic probe, see
  §Status.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-17*
