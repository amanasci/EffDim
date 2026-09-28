---
quick_id: 20260809-cae-undertraining-test
description: Is the CAE a bad model for this data, or an under-trained one?
date: 2026-08-09
status: complete
branch: isomap-curvature
deliverable: notebooks/quick_cae_undertraining_test.ipynb
wall_clock: 39.0 min for 24 CAE fits + deduplicated plain-AE baselines
suite: 286 passed (120 root + 166 pu_manifold)
---

# Is the CAE a bad model for this data, or an under-trained one?

## One-line answer

**Neither, as posed.** The CAE *is* meaningfully under-trained for reconstruction (40 to 100
epochs improves relative error 3.2x), but that is not why its charts never die — **charts never
die in any arm at any dose, because the paper's pruning criterion is structurally incapable of
detecting a dead chart.** And on curvature, the thing that actually failed in 02.5-09, no amount
of training reaches the raw-point baseline.

---

## What was run

Six arms x four torch seeds = **24 CAE fits** on a Swiss roll (`make_swiss_roll`, n=3000,
`noise=0.0`, `random_state=20260807`, centred and divided by one global scalar std),
`chart_dim=2`, `n_charts=8`, `hidden=[64,64]`. 39.0 min wall-clock.

| arm | `max_epochs` | early stop | `wd` | `lip` | unused-chart shrink |
|---|---|---|---|---|---|
| A as-sealed | 40 | ON (pat 5) | 1e-4 | 1e-2 | 0.01 % |
| B paper-epochs (control) | 100 | OFF | 1e-4 | 1e-2 | 0.03 % |
| C decay 1e-1 | 100 | OFF | 1e-1 | 1e-2 | 26.7 % |
| C' decay 1e+0 | 100 | OFF | 1.0 | 1e-2 | 98.7 % |
| D lipschitz 1e-1 | 100 | OFF | 1e-4 | 1e-1 | 0.03 % |
| E 02.5-09 replica | 300 | ON (pat 25) | 1e-4 | 1e-3 | 0.09 % |

Deliverable: `notebooks/quick_cae_undertraining_test.ipynb`, executed end to end, committed
with outputs.

---

## Results

### The charts never die — anywhere

`cae.chart_survival(model, prune_tol=1e-2)` returns **8/8 in all 24 fits.**

| arm | survival (4 seeds) | **occupancy** (4 seeds) | min mass ratio | max gap |
|---|---|---|---|---|
| A as-sealed | 8/8/8/8 | 8/8/8/8 | 0.2575 | 1.80 |
| B paper-epochs | 8/8/8/8 | **6/7/6/6** | 0.1788 | 2.12 |
| C decay 1e-1 | 8/8/8/8 | **7/8/7/7** | 0.1620 | 3.15 |
| C' decay 1e+0 | 8/8/8/8 | **7/8/8/8** | 0.1093 | 2.34 |
| D lipschitz 1e-1 | 8/8/8/8 | 8/8/8/8 | 0.1552 | 2.61 |
| E 02.5-09 replica | 8/8/8/8 | **6/8/7/7** | 0.1931 | 1.87 |

Largest multiplicative gap in the sorted mass-ratio spectrum anywhere: **3.15** — unimodal, no
near-zero mode. Smallest ratio ever observed: **0.109**, still 10x above `prune_tol`.

**Prediction 1 (under-training kills charts): falsified.**
**Prediction 3 (occupancy collapses while weight mass does not): confirmed.**

### Why — measured, not assumed

An executor-side check **outside the notebook**, gating nothing and writing nothing (the
02.5-07 precedent), retrained arm B at seeds 0 and 3 and counted how many of the 3000 points
select each chart under each rule:

| | seed 0 | seed 3 |
|---|---|---|
| charts winning `argmin_alpha e` (what the eq.-3 recon gradient follows) | 7/8 | 7/8 |
| charts winning `argmax_alpha p` (occupancy at inference) | 6/8 | 6/8 |
| weight mass surviving at `prune_tol=1e-2` | **8/8** | **8/8** |
| points won by the emptiest chart | **0** | **0** |

Points per chart by `argmin_e`, seed 0: `[644, 334, 1, 1066, 371, 0, 6, 578]`.

Chart 5 wins **zero of 3000 points** at both seeds. It is exactly the paper's *"chart decoder
[that] is never utilized"*. Its weights did not go to zero. At `wd=1e-4` the only force on it
shrinks it by 0.0135 % over the entire run.

**And more decay cannot fix this.** `chart_survival` thresholds each chart's mass as a **ratio
to the largest chart's**. Decoupled AdamW decay applies to every parameter regardless of
gradient, so it shrinks live and dead charts *together* — and a uniform shrink **cancels in the
ratio**. Arm C' shrinks an ungradiented chart by 98.7 % and still reports 8/8, because the
reference shrank too. Decay compresses the spectrum monotonically with dose (min ratio
0.2575 -> 0.1788 -> 0.1620 -> 0.1093 across A -> B -> C -> C') but never separates dead from live.

**Prediction 4 resolves as: decay is not too weak — it is the wrong instrument.**

> **The finding is a critique of the paper's own pruning criterion.** It says to remove a chart
> when its decoder weight norm falls below tolerance. But a chart can win zero points,
> contribute nothing, and be functionally dead at full weight mass — and the one knob that
> could shrink it shrinks the reference too. On this manifold the criterion never fires, and
> it never could.

### Under-trained for reconstruction — yes

| arm | median rel. err | vs matched 2-D plain AE |
|---|---|---|
| A as-sealed (40 ep) | 0.1576 | 0.78-20.8x |
| B paper-epochs (100 ep) | **0.0485** | 1.8-40.0x |

3.2x better. Epochs matter a great deal for fitting the surface.

### A poor instrument for curvature — also yes

**No arm reaches the raw-point baseline `rho = 0.6712`** (reproduced exactly, seed-independent).

| arm | rho median | rho range |
|---|---|---|
| A | +0.1239 | -0.1994 .. +0.4702 |
| B | +0.0777 | **-0.7467** .. +0.1945 |
| C | +0.1622 | +0.1034 .. +0.4843 |
| C' | **+0.2585** | -0.1571 .. +0.4047 |
| D | -0.0282 | -0.1272 .. -0.0238 |
| E | +0.1919 | -0.0734 .. **+0.7284** |

The seed lottery 02.5-09 reported is not cured by training — arm B's within-arm range is 0.94,
*wider* than 02.5-09's. H0 merge retention loses to a dimension-matched plain AE in **all 24
fits** (CAE 0.56-0.93 vs plain 0.94-0.97).

### Arm D is a warning

Raising the eq.-4 Lipschitz weight to 1e-1 collapses the model: relative error 0.524 at three
of four seeds, **0.19x** the plain AE — worse than the trivial baseline. It *homogenises*
charts (min ratio rises to 0.79) rather than killing any. Lipschitz pressure is not a pruning
mechanism.

---

## The premise audit — three corrections before any compute

Recorded in full in `PLAN.md`. The design executed is **not** the one originally briefed.

**Confirmed.** The sealed fit's norm vector `31.54 ... 30.71` reproduces exactly (sum of
per-layer Frobenius norms, max/min 1.027). Sealed config confirmed: `lr=3e-4`, `wd=1e-4`,
`lip=1e-2`, `max_epochs=40`, `patience=5`. Decay arithmetic confirmed: 0.0135 % over ~4500 steps.

**P1 — two instruments, conflated.** The brief cited 02.5-09 as measuring "8/8 charts
surviving". That table's column is *charts used* (argmax occupancy) and read **8, 8, 3, 5**, not
8/8. Separately, `chart_survival` **had** been run on the roll — 8/8 at `prune_tol=1e-2`,
seed 0, cell 8. **My own first pass wrongly claimed it had never been run; the coordinator
corrected me.** The real error was conflating two different instruments, which this experiment
therefore measures separately per fit.

**P3 — the epochs hypothesis was already weak.** 02.5-09 recorded `epochs_run = 47,
early_stopped = True` at `max_epochs=300`. The roll already trained past the sealed 36/30/24.
Arm B was reframed as a control. Borne out: B changed reconstruction a lot and chart survival
not at all.

**P4 — the decay arm was under-powered by ~2 orders of magnitude.** As briefed, `wd=1e-2`
shrinks an unused chart by 1.34 %. It would have returned a null for reasons unrelated to the
question, and that null reads exactly like "the CAE is simply a poor fit here". Replaced with a
dose pair at `wd=0.1` and `wd=1.0`. **This mattered:** the dose pair is what showed decay
compresses the spectrum monotonically yet still cannot separate a dead chart — a result a
single under-powered point could not have produced.

---

## Deviations

1. **Worktree isolation disabled.** `.venv/` and `notebooks/.cache/` are gitignored and absent
   from a fresh worktree; the run needs `.venv/bin/python` and must hash the cache. Ran on the
   main checkout, branch `isomap-curvature`. Accepted by coordinator.

2. **`torch.set_num_threads(4)`.** Default threading oversubscribes batch-64 across 20 cores at
   3.52 s/epoch; 4 threads gives 1.03. Pinned in the notebook and recorded. Changes wall-clock
   only, and made the 24-fit budget feasible.

3. **Arm E did not reproduce 02.5-09 point-for-point, and could not.** It replicates every
   hyperparameter, but 02.5-09 trains on a **train/holdout split** and reports held-out
   reconstruction, while every fit here trains and measures on all 3000 points. `epochs_run`
   differs (38/32/31/157 vs 47) and Spearmans differ (+0.2651/-0.0734/+0.1187/+0.7284 vs
   -0.0604/-0.1444/+0.8665/+0.4250) — **comparable in spread and seed-lottery character, not in
   value.** The comparability goal is only partly met. All reconstruction numbers in this
   notebook are in-sample and therefore optimistic.

4. **The mechanism table comes from 2 retrained fits, not all 24** — run outside the notebook.

---

## Constraints — all verified

- **No protected module touched.** `git diff --name-only 06401cc..HEAD` over
  `notebooks/pu_manifold/`, `src/effdim/`, `pyproject.toml`: **empty**. Only two files changed
  in this task: `PLAN.md` and the new notebook.
- **No `.cache` write.** Tree hash byte-identical before and after:
  `a88da1f7208337ea8d5d25eab2ef3593688d91010e09766c3341370109438987`.
- **No sealed fit retrained, overwritten, or re-keyed.** All training is fresh and in-notebook.
- **Sealed verdicts untouched.** `CAE_VERDICT = FAIL`, the 02.4 TopoAE verdict, and
  `CURVATURE_VERDICT = FAIL` stand. No verdict artifact, no threshold table, no
  pre-registration, no cfg-hash cache keys.
- **`.planning/phases/02.5-*` untouched** — read during the audit, never written.
- **Additive only** — no existing notebook or runner deleted or rewritten.
- **Suite: 286 passed** (120 from `tests/` + 166 from `notebooks/pu_manifold/tests/`).
- **ROADMAP.md untouched.** One row added to STATE.md.

---

## What this licenses downstream

It reopens nothing. What it adds is a **mechanism** for the sealed FAIL: the 16 undying charts
on the PU fit were never evidence of a healthy atlas and never could have been, because the
instrument used to read them cannot distinguish a dead chart from a live one.

**If chart count is wanted as an outcome anywhere downstream, occupancy is the quantity that
moves — decoder weight norm is not.** `chart_curvature.chart_curvature_field(...)["n_charts_used"]`
already reports it, and the `argmin_alpha e` count is a two-line computation on any fit.

Untested here: whether a larger atlas on a manifold that genuinely needs one behaves
differently, and whether occupancy-based pruning would improve the curvature result. Both are
follow-ups, not conclusions.
