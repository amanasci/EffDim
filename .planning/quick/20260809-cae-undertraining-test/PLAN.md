---
quick_id: 20260809-cae-undertraining-test
description: Is the CAE a bad model for this data, or an under-trained one?
date: 2026-08-09
status: blocked-on-premise-audit
branch: isomap-curvature
deliverable: notebooks/quick_cae_undertraining_test.ipynb
---

# Quick Task: Is the CAE a bad model for this data, or an under-trained one?

**Status: HALTED before execution.** A pre-flight audit of the brief's premises
against the repository found three that do not hold as stated. Two of them would
have made the experiment produce a confidently wrong answer. Nothing has been
trained, no notebook written, no file outside this directory touched.

---

## Part 1 — Premise audit

Everything below is measured from the repository, read-only. No sealed fit was
retrained, re-keyed, or written to. `notebooks/.cache/` was opened for reading only.

### CONFIRMED — the brief is exactly right on these

**C1. The chart-decoder norm vector reproduces exactly.**
The quoted vector is the **sum of per-layer Frobenius norms** of the four
`nn.Linear` weight matrices in each chart decoder. Recomputed from
`notebooks/.cache/cae_fit_43cf438bc944c509_seed20260803.npz`:

```
31.54 30.90 31.21 30.97 31.06 31.43 30.98 31.10
31.41 31.03 31.06 30.92 30.99 30.71 31.01 31.03     max/min = 1.027
```

Byte-for-byte the brief's numbers. Not one chart died.

**C2. The sealed PU hyperparameters are as stated.** From the fit metadata:
`lr=3e-4`, `batch=64`, `weight_decay=1e-4`, `lip_weight=1e-2`, `max_epochs=40`,
`n_charts_init=16`, `d_chart=20`, `l_embed=40` (= 2·20, Nash–Kuiper), `hidden_width=250`,
`fps_pretrain_epochs=5`. The brief's reading of the implementation is accurate.

**C3. The decay arithmetic is right, and it is damning.** For a chart receiving no
gradient, decoupled AdamW shrinks its weights by exactly `(1 − lr·wd)^steps`.
At `lr=3e-4`, `wd=1e-4`, ~4,500 steps:

| `weight_decay` | multiplicative factor | total reduction |
|---|---|---|
| **1e-4 (as sealed)** | 0.999865 | **0.0135 %** |
| 1e-2 | 0.986591 | 1.34 % |
| 1e-1 | 0.873714 | 12.63 % |
| 1.0 | 0.259188 | 74.08 % |

The brief's conclusion — *"the charts were never going to die"* — is correct and now
has an exact number against it: 0.0135 %.

---

### CORRECTION P1 — "02.5-09 measured 8/8 charts surviving" is wrong on two counts

The brief cites plan 02.5-09 as having measured `8/8` charts surviving on a
contractible 2-D Swiss roll. The 02.5-09 SUMMARY seed table says something different.

**(a) Wrong instrument.** That table's column is **"charts used"** — the number of
*distinct charts selected by argmax of `p_alpha`* at inference. It is not
`cae.chart_survival`, which ranks charts by log-spectral-norm mass. These are
different measurements and they can disagree completely: a chart can hold full
weight mass while never winning an argmax.

**(b) Wrong values.** It was **not** 8/8. Across torch seeds 0–3 it read **8, 8, 3, 5**:

| torch seed | charts used | recon rel. err | `rho_chart` | max cond |
|---|---|---|---|---|
| 0 | 8 | 0.0521 | −0.0604 | 63.19 |
| 1 | 8 | 0.0859 | −0.1444 | 122.22 |
| 2 | **3** | 0.0108 | **0.8665** | 3.26 |
| 3 | **5** | 0.0550 | 0.4250 | 7.64 |

Charts *do* fall out of use on the roll, at two of four seeds. That variation is
the entire 02.5-09 finding — `rho_chart` is monotone in charts used
(3 → 0.8665, 5 → 0.4250, 8 → −0.0604 / −0.1444).

**Consequence.** The weight-mass survival instrument has **never been run on the
Swiss roll at all**. That question is genuinely open and worth answering — but the
premise "not one chart dies even on a contractible manifold, per 02.5-09" is not
something 02.5-09 established. The experiment must report *both* instruments and
must not conflate them.

---

### CORRECTION P2 — Arm A cannot be both "as-sealed" and "comparable to 02.5-09"

The design specifies arm A as the sealed protocol (`max_epochs=40`, early stopping
as configured, `wd=1e-4`, `lip=1e-2`) *and* states the arms will be "directly
comparable to [02.5-09's] −0.0604 / −0.1444 / 0.8665 / 0.4250". Those two protocols
differ in six knobs:

| knob | sealed PU fit | 02.5-09 Swiss roll |
|---|---|---|
| `lr` | 3e-4 | **1e-3** |
| `max_epochs` | 40 | **300** |
| `early_stop_patience` | 5 | **25** |
| `lip_weight` | 1e-2 | **1e-3** |
| `fps_pretrain_epochs` | 5 | **20** |
| `embed_dim` | 40 (= 2·20) | **8** (≠ 2·2) |

Only the three architecture items the brief names explicitly — `chart_dim=2`,
`n_charts=8`, `hidden=[64,64]` — actually match. Arm A as specified is the
**sealed-PU protocol**, so its Spearman is comparable to 02.5-09's *in kind*
(same fixture, same instrument, same architecture) but **not in value**.

The comparability claim has to be dropped, or a fifth 02.5-09-replica arm added.

---

### CORRECTION P3 — the epochs hypothesis already has evidence against it

02.5-09's roll fit ran with `max_epochs=300`, `patience=25`, and recorded:

```
CAE  epochs_run = 47  early_stopped = True
```

So on this fixture the CAE already trains to **47 epochs** — past the paper's
effective convergence point, and well past the sealed PU fits' 36 / 30 / 24
(which stopped early against an aggressive `patience=5`). At 47 epochs it still
used all 8 charts at two of four seeds.

Arm B (100 epochs, early stopping off) is therefore substantially a re-run of
territory 02.5-09 already covered at 300. It is cheap and still worth having as a
clean control — but it is unlikely to be where the answer is, and the notebook
should say so up front rather than presenting it as the live hypothesis.

---

### CORRECTION P4 — arm C at `wd=1e-2` is under-powered by ~2 orders of magnitude

**This is the one that would have wasted the run.**

Arm C exists to test whether stronger decay kills excess charts. But by the brief's
own arithmetic (table C3 above), `wd=1e-2` at `lr=3e-4` over ~4,700 steps shrinks an
unused chart's weights by **1.34 %**. That is not "weights go to zero" — it is
indistinguishable from the as-sealed arm at any threshold.

Arm C as specified would return a null **for a reason that has nothing to do with
the science question**, and that null would be read as "the CAE is simply a poor fit
here" — the brief's own second prediction. The experiment would confidently answer
its question wrongly.

To make the decay arm capable of falsifying anything, `wd` must satisfy
`lr·wd·steps = O(1)`. At `lr=3e-4` and ~4,700 steps that means **`wd ≈ 0.1` (12.6 %
shrinkage) or `wd ≈ 1.0` (74 % shrinkage)**.

---

## Part 2 — Proposed corrected design

Same shape, same cost envelope, three changes. Nothing here is executed until
approved.

### Arms — 4 arms × 4 torch seeds (0,1,2,3) = 16 fits

| arm | `max_epochs` | early stop | `weight_decay` | `lip_weight` | change |
|---|---|---|---|---|---|
| **A** as-sealed | 40 | ON (patience 5) | 1e-4 | 1e-2 | unchanged |
| **B** paper-epochs | 100 | OFF | 1e-4 | 1e-2 | unchanged; reframed per P3 as a control, not the live hypothesis |
| **C** strong-decay | 100 | OFF | **1.0** | 1e-2 | **`wd` 1e-2 → 1.0** per P4 — the only setting that can actually zero an unused chart |
| **D** strong-lipschitz | 100 | OFF | 1e-4 | 1e-1 | unchanged |

Fixed across arms: `make_swiss_roll(n≈3000, noise=0.0)` at a fixed `random_state`,
centred and divided by one global scalar std; `chart_dim=2`, `n_charts=8`,
`hidden=[64,64]`, `embed_dim=4` (= 2·chart_dim, the module's Nash–Kuiper default),
`lr=3e-4`, `batch=64`, `fps_pretrain_epochs=5`, `activation="silu"`.

Early stopping is disabled by `early_stop_patience = max_epochs + 1`.
`epochs_run` is recorded per fit and **asserted equal to the cap** in B/C/D; if
early stopping still fires, the notebook says so, because then the arm did not
test what it claims.

**Optional arm E (02.5-09 replica)** — `lr=1e-3`, `max_epochs=300`, `patience=25`,
`lip=1e-3`, `fps=20`, `embed_dim=8`. Add this only if direct numeric comparability
to −0.0604 / −0.1444 / 0.8665 / 0.4250 is wanted. It costs ~4 more fits (~5 min).
**Decision needed — see question Q2.**

### Measured per arm × seed

1. **Both norm distributions, in full** — the per-layer-Frobenius-sum vector (the
   brief's instrument, C1) *and* `cae.chart_survival`'s mass-ratio vector. Reported
   as vectors, never only as a count.
2. **`chart_survival(model, prune_tol)`** with `prune_tol` stated and justified —
   see the threshold note below.
3. **Argmax chart occupancy** — distinct charts winning `p_alpha` at inference.
   This is 02.5-09's instrument and the one that already moves (8/8/3/5).
4. **Reconstruction relative error**, vs a matched `cae.PlainAutoEncoder` at the
   same 2-D bottleneck and the same protocol per arm.
5. **H0 merge retention** vs the ambient roll — `topoae.persistence_pairs` MST
   edge-set instrument, reused unchanged from
   `notebooks/quick_topoae_vs_cae_persistence.ipynb`, with the dimension-matched
   plain AE as baseline.
6. **Curvature Spearman** through the chart decoder via
   `chart_curvature.chart_curvature_field`, against
   `curvature_probe.swiss_roll_analytic_H_scaled`.

### On the threshold trap

The brief's warning is correct and the audit sharpens it. Measured on the sealed
fit, the two instruments have very different spreads:

- per-layer-Frobenius sums: **30.71 – 31.54**, max/min = **1.027**
- `chart_survival` mass ratios: **0.498 – 1.000**, max/min = **2.01**

`chart_survival` spreads 2× where the Frobenius sum spreads 3 %, so it is the more
sensitive of the two — but it is still **unimodal with no near-zero mode**, and
16/16 survive at any `prune_tol < 0.498`. A count is therefore uninformative on the
sealed fit at *any* threshold.

So: `prune_tol = 1e-3` is reported as a fixed, pre-declared reference point (three
orders of magnitude below the sealed fit's observed minimum, so it can only fire on
a genuinely collapsed chart), and **the verdict is read off distribution shape** —
does a near-zero mode appear, does the distribution go bimodal — **never off the
count alone**. Every count is printed beside its full norm vector.

### Falsifiable predictions — written into the notebook before the results cell

- **If under-training is the cause:** arms B/C/D show a near-zero mode appearing in
  the norm distribution (charts collapsing), with reconstruction, H0 merge
  retention, and curvature Spearman improving relative to arm A.
- **If the CAE is simply a poor fit here:** the norm distribution stays unimodal and
  tight in every arm, occupancy stays at 8, and the downstream numbers do not move.
  Reported just as plainly.
- **Third outcome the audit makes live (new):** occupancy collapses (charts stop
  being *used*) while weight mass does not (charts do not *die*). Given 02.5-09's
  8/8/3/5, this is the most likely result, and it would mean the paper's pruning
  criterion does not detect the fragmentation that actually drives the curvature
  failure — a finding about the *instrument*, not the model.

### Cost

02.5-09's roll CAE took ~55 s for 47 epochs including its plain-AE baseline. At
100 epochs, 16 CAE fits + 16 matched plain AEs ≈ **20–30 min**, plus persistence and
curvature measurement. Inside the 45 min budget. If the projection exceeds it,
**seeds are cut before arms** (4 → 3 → 2), and the cut is stated in the notebook.
Wall-clock reported.

---

## Part 3 — Constraints carried forward (unchanged, all honoured so far)

- Never modify `cae.py`, `topoae.py`, `curvature.py`, `curvature_probe.py`,
  `chart_curvature.py`, `cache.py`, `mknn.py`, `geometry_probes.py`, `subsample.py`,
  `pyproject.toml`, `src/effdim/`. Import and call unmodified.
- Never retrain, overwrite, or re-key a sealed fit. Fresh in-notebook training only.
  **No writes to `notebooks/.cache/`** — tree hash verified byte-identical before
  and after.
- Sealed verdicts never reopened: `CAE_VERDICT = FAIL`, the 02.4 TopoAE verdict,
  `CURVATURE_VERDICT = FAIL`. No verdict artifact, no threshold table.
- `.planning/phases/02.5-*` untouched (02.5-09 is at an OPEN blocking checkpoint
  with 02.5-10..13 behind it). This audit **read** those files; it wrote nothing.
- Additive only. Full suite stays at **286 passed**.
- One row added to STATE.md "Quick Tasks Completed". `ROADMAP.md` untouched.

### Executed deviation from GSD defaults

**Worktree isolation disabled for this task.** `workflow.use_worktrees` defaults to
true, but both `.venv/` and `notebooks/.cache/` are gitignored and would be **absent
from a fresh worktree**. The environment requires `.venv/bin/python`, and the
no-cache-writes constraint requires hashing `notebooks/.cache/`. Neither is possible
in an isolated worktree. Execution runs on the main checkout, on branch
`isomap-curvature`.

---

## Open questions — blocking

**Q1 (blocking).** Arm C `weight_decay`: **1.0** (74 % shrinkage on an unused chart)
or **0.1** (12.6 %)? 1.0 is the setting most likely to produce a visible death and
so most likely to give a clean answer; 0.1 is the gentler probe and less likely to
degrade the *used* charts as a side effect. Running both as C and C′ costs 4 more
fits (~5 min). Recommendation: **run both** — the pair also shows whether any
degradation is decay-dose-dependent.

**Q2.** Add arm E (02.5-09 replica) for direct numeric comparability to
−0.0604 / −0.1444 / 0.8665 / 0.4250? Costs ~4 fits (~5 min).
Recommendation: **yes** — without it the brief's comparability goal is simply not met.

**Q3.** With Q1-both and Q2-yes the run is 24 fits, ~35–45 min. Acceptable, or cut
seeds to 3 (18 fits, ~30 min)?

No compute will be spent until these are answered.
