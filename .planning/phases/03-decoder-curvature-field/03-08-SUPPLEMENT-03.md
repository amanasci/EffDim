# 03-08 Supplement 03 — the converged PU fit: reconstruction moved 62%, `cond(g)` did not move at all

**Date:** 2026-08-16
**Status:** developer-directed work, executed. No sealed verdict is reopened, softened, or
recomputed by anything here.
**Directive:** *"train CAE until it succeeds on PU, base off reconstruction loss; then compute
the deliverable curvature field."*

---

## 1. What was asked, and what it was narrowed to

The directive was resolved at a blocking question before any code ran, because two readings of
it lead to materially different work and the wrong one is expensive. The resolutions:

| Question | Resolution |
|---|---|
| What does "base off reconstruction loss" govern? | **Stopping criterion only.** The pre-declared lexicographic selection rule is untouched and still owns `n_charts`. |
| What counts as "succeeds"? | **Convergence only, no bar.** Report where the curve lands; compute the field either way. |
| Where does it run? | **Local CPU, one seed first.** Seed 20260813 as a probe, then decide about the other two. |

The declined reading is worth recording. "Base off reconstruction loss" could have meant
*re-rank the grid on reconstruction*, which would have moved the selected config from
`n_charts = 4` to `n_charts = 16`. That was declined because it is exactly threat **T-3-24** in
03-08's own threat model — changing the selection rule after its numbers are visible. The rule
stands; only the stopping rule changed.

## 2. The defect removed

`cae.train_cae` early-stops on **total** loss — reconstruction + cross-entropy + the Lipschitz
penalty. `03-NOTE-d12-retirement.md` §5 measured the consequence at PU scale: the `nc=4` fit
halts at epoch 30 whether `MAX_EPOCHS` is 40 or 300, **bit-identically**, because the cap was
never the binding constraint. A plateau in terms unrelated to reconstruction ended training
while reconstruction was still descending.

No cell in the corrected nine-cell grid is a converged fit, and they fail to be for two
different reasons: four of nine early-stopped at 27–32 epochs, and the other five ran out the
40-epoch cap with `early_stopped = false`.

**The fix, in this runner's own cfg and never in `cae.py`:** `early_stop_patience = max_epochs + 1`,
which is structurally inert — `plateau_count` increments at most once per epoch, so it cannot be
reached within the budget — plus `wallclock_ceiling_s = inf`. Every model and optimizer constant
is the grid's, unchanged: `lr = 3e-4`, `batch = 64`, `lip_weight = 1e-3`,
`fps_pretrain_epochs = 5`. The converged fit differs from a grid cell in **how long it trains and
in nothing else**.

It worked: `epochs_run = 300` of a 300 budget, `early_stopped = false`,
`wallclock_truncated = false`. The stopping rule that fired at epoch 30 no longer fires.

## 3. Result

`n_charts = 4`, seed `20260813`, `chart_dim = 20`, 300 epochs, 6,767.9 s (1.88 h) on CPU.

### Reconstruction — substantially better, and not overfitting

| Quantity | 40-epoch grid cell | Converged (300 epochs) | Change |
|---|---|---|---|
| holdout `mse_per_dim` | 1.247445e-04 | **4.710866e-05** | **-62.2%** |
| holdout `mean_norm` | 0.29309 | **0.177879** | -39.3% |
| `dim_mse_median` | 1.1145e-04 | 4.5186e-05 | -59.5% |
| `dim_mse_p95` | 2.5760e-04 | 7.8047e-05 | -69.7% |
| `dim_mse_max` | 4.9821e-04 | 1.1681e-04 | -76.6% |

Same seed, same split, same architecture, same optimizer settings. The only difference is that
training was allowed to continue.

**Not overfitting.** The held-out figure fell alongside the training curve rather than diverging
from it, and the per-dimension tail (`p95`, `max`) improved *more* than the median — the
opposite of the signature overfitting leaves.

Against the independent anchor in `03-NOTE-d12-retirement.md` §5 — a plain autoencoder at
`latent_dim = 20` trained 300 epochs, `mse_per_dim = 2.2646e-05` — the CAE's deficit narrows
from roughly **5.5x to 2.08x**. Narrower, not closed. That anchor is context here and nothing
more: D-12's comparison is retired, and §5's own threshold remains DEFERRED.

### The curve did NOT plateau

```
epoch    1: 2.937505e-01
epoch   25: 7.319718e-02
epoch   50: 6.968164e-02
epoch  100: 4.893212e-02
epoch  200: 3.702551e-02
epoch  275: 3.054569e-02
epoch  300: 2.893576e-02   <- best epoch is the LAST epoch
trailing 25-epoch relative improvement = 5.271e-02, against a 1.0e-03 tolerance
status = still_descending
```

**The budget ended training, not convergence.** Stated plainly because the directive was "train
until it succeeds": on the criterion chosen — a reconstruction plateau — this fit has **not**
succeeded, and a longer budget would very likely keep improving it. Proceeding to the curvature
field on it was an explicit developer decision ("proceed regardless"), not an inference that the
fit was converged.

Extending is a fresh run, not a resume: `train_cae` constructs its optimizer per call, so there
is no way to continue these 300 epochs from where they stopped.

### `cond(g)` — unchanged, and this is the finding

| Quantity | 40-epoch grid cell | Converged (300 epochs) |
|---|---|---|
| `cond(g)` median | 9.758e+06 | **1.0033e+07** |
| `cond(g)` p90 | 1.263e+07 | 1.6048e+07 |
| `cond(g)` p99 | 3.106e+07 | 2.3254e+07 |
| `cond(g)` max | 4.886e+07 | 3.6751e+07 |

A 62% improvement in reconstruction bought **nothing** on metric conditioning. Same order of
magnitude, still ~10⁷, against the Swiss roll's 1.4–8.3 on identical machinery.

This was predicted before the run, and the mechanism is on record in
`03-NOTE-d12-retirement.md` §4: `cae.train_cae` regularizes `model.chart_encoders`, while
`chart_curvature.chart_decoder_map` composes `model.chart_decoders[i]` with
`model.embedding_decoder`. **The two sets share no parameter.** Nothing in the training
objective constrains the decoder's derivatives at any order, so training longer on a
reconstruction objective cannot be expected to improve decoder conditioning — and did not.

The consequence carries into every curvature number: `cond(g) ~ 10⁷` destroys roughly seven
digits of float64 precision in the `g⁻¹` contraction inside `H = Σ_jk g^jk II_jk`. That is why
the field's near-singular flagging exists, and why `cond(g)` is reported beside `||H||` rather
than instead of it.

### Chart occupancy — still degenerate

`argmax` occupancy is **2 of 4** distinct charts (chart 1: 1,312 points; chart 3: 688). Two of
the four charts receive no held-out point at all. Unchanged from the grid cell's 2 of 4.

The configuration selected by the pre-declared rule clears its occupancy disqualifier
(median `< 2`) by exactly zero margin, and converging it did not add a live chart.

### Other diagnostics

| Quantity | Value |
|---|---|
| PH `latent\|ambient\|H0\|bottleneck_norm` | 0.3432 (grid cell: 0.6217) |
| PH `latent\|ambient\|H1\|bottleneck_norm` | 0.6838 (grid cell: 0.8451) |
| curvature wall clock, 2,000 holdout rows | 592.1 s |
| checkpoint | `notebooks/.cache/03_converged_cae_pu_nc4_seed20260813.pt` (float64, eval mode) |

## 4. One seed

This is **one seed**, and one seed is a probe. The reported unit in this milestone is the
three-seed spread; `--field` prints a probe notice and no spread table for a single draw.

Converging seeds `20260814` and `20260815` costs roughly 3.8 h more CPU and has not been
authorised. Until it is, every number above describes one draw and no dispersion is claimed.

A defect found and fixed before it could bite: the converge checkpoint stem was originally
fixed, so a second seed would have silently overwritten the first. It is now keyed by
`(n_charts, seed)` (commit `db6fbec`). Caught before the second seed existed, so no measurement
is affected.

## 5. What this does and does not license

**Does:** the removal of total-loss early stopping is a real, measured improvement to how this
milestone trains a CAE on PU — 62% on held-out reconstruction, from a one-line change to a
stopping parameter, with no architecture or optimizer change of any kind. Every grid number in
`03-08-SUMMARY.md` was measured under the truncating protocol and should be read as such.

**Does not:** this says nothing about curvature fidelity. Reconstruction is a **C0** quantity;
curvature is **C2**; small C0 error does not bound C2 error, and `chart_curvature.py`'s own
worked example has a decoder attenuating curvature 30% with essentially no reconstruction
signal. A better-reconstructing CAE with an identically ill-conditioned pullback metric is
exactly what §3 measured, and it is not evidence that the curvature field improved.

The gate override stands unchanged: Phase 3 runs on a deliberate override of its own
precondition (`02-NOTE-phase-2-stage-on-hold.md` §3), every sealed verdict in this milestone is
FAIL, and a curvature field decoded from this parameterization conflates real curvature with
parameterization damage. Nothing in this supplement changes that, and the caveat travels with
the field's numbers rather than being filed away from them.

## 6. Cross-references

- `03-NOTE-d12-retirement.md` §4 (disjoint regularizer), §5 (the total-loss stopping measurement
  and the plain-AE anchor).
- `03-08-SUMMARY.md` — the nine-cell grid this fit's baseline cell comes from, and the selection
  rule that chose `n_charts = 4`.
- `02-NOTE-phase-2-stage-on-hold.md` §3 — the gate override.

---
*Phase: 03-decoder-curvature-field*
*Recorded: 2026-08-16*
