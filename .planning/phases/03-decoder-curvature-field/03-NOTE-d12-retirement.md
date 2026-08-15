# 03-NOTE D-12 Retirement — the CAE-vs-plain-AE comparison retired, replaced by a direct C0/C2 bar

**Date:** 2026-08-15
**Status:** decision recorded. No sealed verdict is reopened, softened, or reinterpreted by
this note.
**Raised by:** the developer.

---

## 1. The decision

Stop comparing the CAE against a plain autoencoder. The plain AE was only ever an instrument
for detecting a broken or undertrained CAE; if the CAE succeeds at reconstruction at both the C0
and the C2 level, a relative comparison against a model there is no intention of shipping adds
nothing. A direct absolute bar is strictly better than a proxy.

## 2. D-12 FIRED BEFORE IT WAS RETIRED

**This is not a criterion dropped because it returned an unfavourable answer.** The answer it
returned is recorded here in full, and the reason for retirement is that the comparison was the
wrong instrument for the question — not that the instrument said something unwelcome. Naming
that distinction explicitly is why this section sits this high in the note, and this is the same
phase that already carries `02.6-FINDINGS.md` §4's record of a criterion changed after an
unfavourable result:

> `02.6-SCREENING-RULE.md` was ratified blind and committed **before** any 02.6 measurement
> existed... The decision to change the ranking axis was made **after** the plain-AE result was
> in hand and unfavourable. ... this is written down as what it is: a criterion changed after
> seeing a result.

D-12's own text: **"Escalate to a `d` sweep only if the best `d=20` config loses to a matched
plain-AE control on held-out reconstruction and PH H0/H1 agreement."** On the corrected grid
(defects 1 and 2 fixed per `03-08-DEFECTS-01.md`, defect 3's normalizer replaced), against the
**matched** `latent_dim=20` control, `n_charts=4` selected. Verbatim from
`.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --select-only`:

```
Selected n_charts: 4
best d=20 CAE (n_charts=4) mse_per_dim=0.000173309 vs control mse_per_dim=3.58866e-05 -> loses_reconstruction=True
best H0/H1 bottleneck_norm=(0.6217,0.8451) vs control=(0.2144,0.8247) -> loses_ph_agreement=True
TRIGGER FIRES = True
```

**D-12 fired on both legs.** Unhedged: this retirement does not soften, dispute, or explain
away that result. The trigger fired exactly as designed, on the corrected grid, and the decision
recorded here is that the comparison itself — not the CAE's showing against it — is the wrong
instrument for the question this phase needs answered.

## 3. The C0/C2 argument

Reconstruction loss is a C0 quantity; curvature is a C2 quantity; small C0 error does not bound
C2 error. `chart_curvature.py`'s own module docstring states this from the other side, in its
own worked example: a decoder learning `y = 0.7 a x^2` where the truth is `y = a x^2` has tiny
reconstruction error wherever the sampled `x` sit near zero, while its second derivative is
`1.4a` instead of `2a` — **30% curvature attenuation with no reconstruction signal at all**.

A comparison conducted entirely in C0 — held-out `mse_per_dim`, PH bottleneck agreement, both
zeroth-order measures of where points land, not of how the decoder's derivatives behave —
therefore cannot rank two models on a C2 question, whichever way it comes out. Losing on both
C0 legs is exactly as uninformative about curvature fidelity as winning on both would have been.

## 4. The disjoint-regularizer finding

`cae.train_cae` regularizes `model.chart_encoders` through `cae.lipschitz_penalty` (`cae.py`
line 483). `chart_curvature.chart_decoder_map` (`chart_curvature.py` line 183) composes
`model.chart_decoders[i]` with `model.embedding_decoder`. The two sets share no parameter, so
nothing in the training objective constrains the decoder's derivatives at any order.

Measured consequence: `cond(g)` reaches `4.886e7` on the corrected PU grid
(`notebooks/.cache/03_curvature_field_pu.jsonl`, 15-record grid) against the Swiss roll's
`1.4`–`8.3` on the identical machinery — five to seven orders of magnitude apart. That destroys
roughly seven digits of float64 precision in the `g^-1` contraction inside
`H = sum_jk g^jk II_jk`.

`03-NOTE-isometry-prior-spike.md` is the first attempt at a fix — a first-order prior on the
decoder's Jacobian, opt-in and default off, added without editing `cae.py`. Its outcome: the
spike **halted at its own pre-declared budget gate before any ladder cell trained** (the
isometry penalty roughly triples per-epoch training cost, and even the cheapest pre-declared
epoch count exceeds the pre-declared compute budget). No mechanism or bias verdict exists from
that spike. It is not evidence for or against the prior working; it is evidence that the current
compute budget does not cover measuring it at this scope.

## 5. The replacement criterion, direct and two-part

Both legs must clear. Neither leg references a control model.

**C0 leg — held-out reconstruction below an absolute threshold on `mse_per_dim`, no control
model involved.** The measured distribution, ground for the proposed number: the nine
corrected-grid CAE cells span `mse_per_dim` **8.899e-05 … 2.438e-04**, with mean residual L2
norm (`mean_norm`) **0.2468 … 0.4112**. The training-budget confound behind D-12's reconstruction
leg is now measured directly and independently confirms the C0-comparison problem this section
is retiring: at PU scale (`n_charts=4`, matched 20-dim control, wall-clock ceiling disabled):

```
   model         cap  ran  stop   argmax mse      min mse   agree   sec
CAE nc=4          40   30  True   1.2474e-04   9.4677e-05  52.4%   641
CAE nc=4         300   30  True   1.2474e-04   9.4677e-05  52.4%   637
plainAE d=20      40   40 False   3.6117e-05            -      -    27
plainAE d=20     300  300 False   2.2646e-05            -      -   381
```

Raising the epoch cap 40 → 300 changes the CAE's result **bit-identically** — it early-stops at
epoch 30 either way, so the cap was never binding. The plain AE never early-stops and keeps
improving to 300 epochs, so the measured gap **widens** with budget (3.5x → 5.5x). `train_cae`
early-stops on TOTAL loss (reconstruction + cross-entropy + Lipschitz), so a plateau in terms
unrelated to reconstruction halts the CAE while its reconstruction is still descending. This is
a second, independent reason the CAE-vs-plain-AE comparison was the wrong instrument: the two
models were never on equal training, so even a same-bottleneck comparison confounds architecture
with training-length artifacts specific to each model's own stopping behaviour.

**Proposed absolute C0 threshold: `mse_per_dim < 2.5e-04`** — just above the measured CAE
ceiling (`2.438e-04`) on the corrected nine-cell grid, so every corrected-grid cell that has
actually been measured clears it, and the bar is a real ceiling rather than an untested
extrapolation. **PROPOSED, awaiting developer ratification at this task's checkpoint** — a
threshold chosen after seeing this distribution is weaker than one pre-registered, and that is a
known cost of replacing the criterion mid-phase rather than something to paper over. This
number is not settled by this note.

**C2 leg — curvature fidelity against analytic `H` on the Swiss roll clearing the existing
`ROLL_FLOOR = 0.65` on median `rho_chart`.** This bar is unchanged, already declared in
`swiss_roll_curvature_sweep_run.py`'s source before any Phase 3 number existed
(`ROLL_FLOOR = 0.65`, that module's D-15/D-02), and is simply pointed at rather than restated
with new values.

## 6. Consequences, named precisely

D-12's escalation trigger no longer drives any decision. `notebooks/diagnostics/curvature_field_pu_run.py`'s
`print_d12_trigger` stays in place and keeps printing — its output is now context, not a
trigger — and this note is what a reader must consult to interpret it. `03-08-PLAN.md` Task 3 is
the D-12 escalation checkpoint; it is **not rewritten**, and this note records that its decision
is answered here: no `d` sweep is escalated to on the strength of the retired trigger. Anyone
executing `03-08` must read this note first.

Every file that mentions D-12, checked directly, none edited by this note:

| File | D-12 reference |
|---|---|
| `03-CONTEXT.md` | D-12 itself (the decision text) |
| `03-07-PLAN.md` | computes the D-12 control, reports the trigger, does not act on it |
| `03-07-SUMMARY.md` | records the D-12 control was fitted and the trigger computed |
| `03-08-PLAN.md` | Task 3 is the D-12 escalation checkpoint |
| `03-08-DEFECTS-01.md` | defect 1 (unmatched D-12 control) and defect 2 (training-length confound) both corrupt D-12's inputs |
| `03-08-SUPPLEMENT-02.md` | the fix for defect 1's unmatched D-12 control |
| `03-VALIDATION.md` | re-fit controls at an escalated `d` if D-12 fires (row 4) |
| `03-GPU-RUNBOOK.md` | expects 3 D-12 control cells in the 12-record grid |
| `03-07-SUPPLEMENT-01.md` | device-mixing caveat extended to the D-12 control cells |
| `03-08-DECLARATION-01.md` | device-mixing caveat extended to the D-12 control cells |

**None of the ten is edited by this note.**

## 7. What this note does NOT do

- It does not reopen, soften, or reinterpret any sealed verdict.
- It does not change any recorded number.
- It does not select `n_charts`.
- It does not claim the CAE is good — it claims the plain-AE comparison was the wrong way to
  ask.
