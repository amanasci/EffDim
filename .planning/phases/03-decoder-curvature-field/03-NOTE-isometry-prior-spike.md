# 03-NOTE Isometry Prior Spike — halted on budget, then halted again by developer decision

**Date:** 2026-08-15
**Status:** spike — gates nothing, selects nothing, adopts nothing. **Two distinct halts, not
one.** (1) The runner's own pre-declared `--probe` budget gate stopped the ladder before any
cell trained (`BUDGET NOT MET` — §3). (2) At Task 5's checkpoint, the developer separately
decided to HALT here rather than raise the budget or shrink the ladder, pending additional
information not yet provided (§8). Neither halt produced a measurement. No `rho_chart`,
`cond(g)`, `median_magnitude_ratio` or `calibration_slope` number exists from this spike, and
none will until the developer's second halt is lifted.
**Raised by:** the developer, from a measured structural fact. Reconstruction loss is a C0
quantity and curvature is a C2 quantity; small C0 error does not bound C2 error, so nothing in
`cae.train_cae`'s objective constrains the decoder's derivatives unless something is added that
does.

---

## 1. Motivation

`cae.train_cae` regularizes `model.chart_encoders` through `cae.lipschitz_penalty` (`cae.py`
line 483). `chart_curvature.chart_decoder_map` (`chart_curvature.py` line 183) composes
`model.chart_decoders[i]` with `model.embedding_decoder`. The regularized set and the
differentiated set share no parameter — the disjoint-regularizer finding. `cae.chart_loss`
constrains the decoder only through reconstruction, a C0 quantity, and small C0 error does not
bound C2 error (`chart_curvature.py`'s own worked example: a decoder learning `y = 0.7 a x^2`
where the truth is `y = a x^2` has tiny reconstruction error near `x = 0` while its second
derivative is `1.4a` instead of `2a` — 30% curvature attenuation with no reconstruction signal
at all).

Measured consequence: on the corrected PU grid (`notebooks/.cache/03_curvature_field_pu.jsonl`,
15-record grid, 2026-08-15), `cond(g)` reaches `4.886e7` at `n_charts=4`, against the Swiss
roll's `1.4`–`8.3` on the identical machinery — five to seven orders of magnitude apart. This
destroys roughly seven digits of float64 precision in the `g^-1` contraction inside
`H = sum_jk g^jk II_jk`.

## 2. What was built

`notebooks/pu_manifold/decoder_priors.py` — `metric_deviation` (isometry and conformal
per-point deviations, pure-tensor, known-answer tested), `chart_decoder_jacobian`
(differentiable Jacobian of the chart decoder, via `vmap(jacfwd(decode_one))` — verified to
carry a live autograd graph back to both `chart_decoders` and `embedding_decoder` parameters),
`isometry_penalty` (the training-time scalar), and `decoder_prior_active` — a
`@contextmanager` that rebinds `cae.chart_loss` to a wrapper adding the penalty to `total` only,
restoring the original binding in a `finally` block (including on exit by exception). At
`weight == 0.0` it installs nothing at all, so the zero-weight arm is structurally, not merely
numerically, the untouched `cae.train_cae` code path.

**`cae.py` was not edited.** `git diff --quiet notebooks/pu_manifold/cae.py` passes. The
permanent form of this seam — a four-line optional `extra_loss=None` callable parameter on
`cae.train_cae`, replacing the scoped shim — was proposed to the developer at Task 5's
checkpoint. **Resolved: confirmed NOT applied.** The scoped `cae.chart_loss` shim
(`decoder_prior_active`) stays as the tested mechanism; `cae.py` remains unedited.

`notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py` — the weight-ladder runner,
copying the fixture, protocol and metric call sequence from
`swiss_roll_curvature_sweep_run.py` (not edited) rather than restating them.

**The anchor reproduced exactly, `==`, not a tolerance:**

```
--anchor-check: weight=0.0, n_charts=8, seed=0, n_points=3000, sweep.BASE_CFG verbatim
measured rho_chart = -0.06041003026778113
ANCHOR OK: rho_chart == -0.06041003026778113 exactly.
```

This proves the runner's fixture, protocol and metric call sequence are a faithful copy, not a
re-derivation, before the ladder itself is considered.

## 3. The budget probe — `BUDGET NOT MET`

Per Task 2's design, the runner projects the wall-clock cost of the ladder (8 cells: 2 seeds ×
4 weights, at `n_charts=8`, `n_points=12000`, early stopping disabled) from two `max_epochs=10`
probe cells — one at `weight=0.0`, one at `weight=LADDER_WEIGHTS[-1]=0.1` — before committing
to any epoch count:

```
--probe: two max_epochs=10 cells at n_charts=8 n_points=12000 seed=0
measured: t0=3.685s/epoch (weight=0.0)  t1=10.699s/epoch (weight=0.1)  c=7.213s/cell (curvature)
  E=150   projected_total=10792.4s
  E=100   projected_total=7214.2s
  E=75    projected_total=5425.1s
  E=50    projected_total=3635.9s
BUDGET NOT MET -- even the smallest candidate exceeds LADDER_BUDGET_S (3000s):
  E=150   projected_total=10792.4s  (budget=3000s)
  E=100   projected_total=7214.2s  (budget=3000s)
  E=75    projected_total=5425.1s  (budget=3000s)
  E=50    projected_total=3635.9s  (budget=3000s)
```

The isometry penalty triples the per-epoch training cost (`10.699s` vs `3.685s`) — every
training step at `weight > 0` computes a full differentiable chart-decoder Jacobian
(`vmap(jacfwd(...))`) over the batch, on top of the ordinary forward/backward pass. Even the
smallest pre-declared candidate (`E=50`, projecting `3635.9s`) exceeds the `3000s` budget.

**Per Task 2's own design, this is a real terminal branch, not an error to work around:**
picking an epoch count outside the pre-declared `LADDER_MAX_EPOCHS_CANDIDATES = (150, 100, 75,
50)` to force a fit under budget would be exactly the "silently degrading the design further"
the runner's own `--probe` docstring forbids. **No ladder cell was run.** `--summary` was
invoked and, correctly, reported no cells on record:

```
ISOMETRY PRIOR SWEEP SUMMARY -- 0 cell(s) on record at notebooks/.cache/quick_isometry_prior_sweep.jsonl
No cells recorded yet -- nothing to summarize.
```

Consequently: **no `rho_chart`, `cond(g)`, `median_magnitude_ratio` or `calibration_slope`
number exists from this spike.** No `MECHANISM DEMONSTRATED`/`MECHANISM NOT DEMONSTRATED`
verdict was computed — the mechanism check requires trained cells and none were trained. No
`FLATTENING SUSPECTED` / `CLEAN IMPROVEMENT` / `NO EFFECT` bias verdict was computed for the
same reason. This is reported as plainly as a positive result would have been (hard constraint
5): the spike did not fail to demonstrate the mechanism — it never reached the point where the
mechanism could be tested.

## 4. The conformal branch

**Not taken.** Task 3's pre-declared branch ("if the isometry arm prints `MECHANISM NOT
DEMONSTRATED`, run the conformal arm") never triggers, because the isometry arm never printed
any mechanism verdict at all — it halted at the budget gate before training a single ladder
cell. The conformal arm therefore also carries no measurement from this spike. **Confirmed at
Task 5's checkpoint: moot** — no mechanism verdict exists because no ladder cell trained, so
there is nothing for the conformal branch to respond to yet.

## 5. Limitations, unhedged

- Two seeds (`LADDER_SEEDS = (0, 3)`) would not have separated a small effect from seed noise
  even had the ladder run.
- The roll's `cond(g)` is already benign (`1.4`–`8.3`); the roll could only ever have
  demonstrated the mechanism and the absence of bias, never a fix at the PU pathology of
  `4.9e7`.
- The `weight=0` arm, had it run, would have been a fresh fixed-length measurement at
  `n_points=12000`, `early_stop_patience` disabled, and is not comparable cell-for-cell to the
  sealed `n=12000` grid in `notebooks/.cache/03_swiss_roll_curvature_sweep_n12000.jsonl`
  (measured fact 4), which used the un-disabled `early_stop_patience=25`.
- **The budget probe itself carries one seed's worth of timing information** (`seed=0` only,
  per the `--probe` design) — a different seed's per-epoch cost could differ, though not by
  enough to close a ~3.6x-to-1x budget gap.

## 6. The notebook obligation

This spike is a script, not the CLAUDE.md Swiss roll notebook, because CLAUDE.md's Swiss roll
mandate is triggered by a new manifold-learning or representation-learning **model**, and this
is not a new model: it is one additive, default-off loss term on the existing CAE, whose Swiss
roll notebook already exists (`notebooks/02.2_swiss_roll_cae_check.ipynb`) and whose
chart-decoder curvature notebook also already exists
(`notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`). What the spike additionally
needs — a weight ladder, a threshold table, a JSONL record and a resumable runner — is precisely
what CLAUDE.md forbids inside a sanity-check notebook.

**If the prior is ever adopted, `notebooks/03_swiss_roll_isometry_prior_check.ipynb` must exist
and pass before any PU fit runs with a non-zero prior weight.** That obligation stands
regardless of this spike's outcome, and is not discharged by anything in this note — there is,
as of this note, nothing to discharge it against, since no adoption evidence exists yet.
**Confirmed at Task 5's checkpoint:** this is CLAUDE.md's standing rule for any new
manifold-learning model variant, not a new decision made here — it was restated for
confirmation, not decided.

## 7. What this note does NOT claim

- No adoption recommendation — none is possible without a measurement.
- No PU number.
- No `n_charts` selection.
- Nothing about whether the prior helps or hurts at `d = 20`, `D = 768`, or at `n_charts = 8`
  on the roll itself — the ladder that would have measured this did not run.
- It does not claim the isometry prior is broken, expensive beyond use, or a bad idea — only
  that the pre-declared budget (`LADDER_BUDGET_S = 3000`) does not cover the pre-declared
  ladder (`LADDER_MAX_EPOCHS_CANDIDATES`, `LADDER_WEIGHTS`, `LADDER_SEEDS`) at the measured
  per-epoch cost. A larger budget, a smaller ladder, or a cheaper Jacobian computation are all
  live options a developer could choose — none is chosen here.

## 8. Checkpoint resolution (2026-08-15) — the developer's second, separate halt

Task 5's checkpoint took four questions to the developer, on top of §3's budget halt. Resolved:

**(a) The isometry-prior spike: HALT, deliberately, distinct from §3's budget halt.** The
developer is stopping here and has additional information to provide before the spike
proceeds — this is not the runner refusing on its own pre-declared terms (that already
happened in §3); it is a separate, deliberate developer decision **not** to raise
`LADDER_BUDGET_S`, **not** to shrink the ladder, and **not** to run any ladder cell for now.
Both halts are real and are recorded distinctly: §3 is the runner's own terminal branch,
firing automatically against a declared budget before any human judgement was invoked; this
one is a human choosing to stop even though §3's proposed relief valves (raise the budget ~21%,
or shrink the ladder) were available and were not taken. **The spike remains exactly where §3
left it: no ladder cell trained, no `rho_chart`/`cond(g)`/`median_magnitude_ratio`/
`calibration_slope` number, no mechanism or bias verdict.** `decoder_priors.py` and the runner
both exist, are tested, and are re-runnable at any revised scope without further code changes,
whenever the developer's additional information resolves the halt.

**(b) The conformal branch: not run, confirmed moot** — see §4.

**(c) The `extra_loss` seam: confirmed NOT applied** — see §2. `cae.py` stays unedited; the
scoped `cae.chart_loss` shim is the tested mechanism.

**(d) The notebook obligation: confirmed** — see §6. Standing CLAUDE.md rule, not decided here.

**What remains open:** the "additional information" the developer will provide before deciding
whether to raise the budget, shrink the ladder, or abandon the spike. This note will be updated
again if and when that happens. Nothing in this section changes the fact stated throughout this
note: **this spike delivered tested, working infrastructure and zero measurement.**
