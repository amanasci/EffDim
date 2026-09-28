---
quick_id: 20260809-cae-undertraining-test
description: Is the CAE a bad model for this data, or an under-trained one?
date: 2026-08-09
status: approved-executing
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

**(c) My own error, corrected by the coordinator.** I initially wrote that the
weight-mass instrument had "never been run on the roll at all." That is **wrong**.
`notebooks/02.5_swiss_roll_chart_curvature_check.ipynb` cell 8 calls

```python
survival = cae.chart_survival(model, prune_tol=1e-2)   # -> "charts surviving = 8 / 8"
```

Verified in the committed output. So the weight-mass instrument has **one existing
observation on the roll: 8/8 at `prune_tol=1e-2`, seed 0**. This experiment
*reproduces and extends it across seeds and arms*; it does not open new ground.

**Consequence.** What was genuinely conflated is (a): "charts used" (8/8/3/5, the
seed table) and "charts surviving" (8/8, cell 8) are different quantities measured
by different instruments, and the brief treated them as one. The experiment must
report **both**, per fit, and never substitute one for the other.

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
showed **8/8 weight-mass survival** (cell 8) and used all 8 charts at seed 0.

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

## Part 2 — Settled design (approved by coordinator)

All three open questions answered: run **C and C′ both**, **add arm E**, **keep 4 seeds**.
Cut arm B before cutting a seed if the budget bites.

### Arms — 6 arms × 4 torch seeds (0,1,2,3) = 24 CAE fits

| arm | `lr` | `max_epochs` | early stop | `weight_decay` | `lip_weight` | `embed_dim` | fps | role |
|---|---|---|---|---|---|---|---|---|
| **A** as-sealed | 3e-4 | 40 | ON (patience 5) | 1e-4 | 1e-2 | 4 | 5 | reference |
| **B** paper-epochs | 3e-4 | 100 | OFF | 1e-4 | 1e-2 | 4 | 5 | **control** (P3: weak hypothesis) |
| **C** decay ×1e3 | 3e-4 | 100 | OFF | **1e-1** | 1e-2 | 4 | 5 | decay dose 1 (12.6 % shrink) |
| **C′** decay ×1e4 | 3e-4 | 100 | OFF | **1.0** | 1e-2 | 4 | 5 | decay dose 2 (74.1 % shrink) |
| **D** strong-lipschitz | 3e-4 | 100 | OFF | 1e-4 | **1e-1** | 4 | 5 | eq.-4 pressure |
| **E** 02.5-09 replica | **1e-3** | 300 | ON (patience 25) | 1e-4 | 1e-3 | **8** | 20 | comparability to −0.0604 / −0.1444 / 0.8665 / 0.4250 |

C and C′ are a **dose pair**: a single decay point cannot distinguish "decay too weak"
from "decay irrelevant." Two doses an order of magnitude apart can.

Fixed across all arms: `curvature_probe.make_swiss_roll_fixture(n=3000, seed=20260807)`
— which *is* `make_swiss_roll(noise=0.0)` centred and divided by one global scalar std,
CLAUDE.md's exact convention, and 02.5-09's fixture seed. `chart_dim=2`, `n_charts=8`,
`hidden=[64,64]`, `batch=64`, `activation="silu"`, `lip_every_n_steps=1`,
`early_stop_min_delta=1e-4`.

Early stopping is disabled by `early_stop_patience = max_epochs + 1`. `epochs_run` is
recorded per fit and **asserted equal to the cap** for B/C/C′/D; if early stopping
fires anyway the notebook prints a loud failure line, because then the arm did not
test what it claims.

### Primary hypothesis — promoted from footnote, per coordinator

> **Occupancy collapses while weight mass does not.**

Seed 0 already shows both instruments disagreeing in the direction that matters:
8 charts *used* alongside 8/8 *surviving*, while seed 2 drops to 3 charts used. If the
arms confirm this, the finding is a **critique of the paper's own pruning criterion**:
the paper says to remove a chart when its decoder weight norm falls below tolerance,
but a chart that has stopped winning any `argmax(p_alpha)` while its weights stay large
is **functionally dead and that criterion never fires**. It would mean the pruning rule
cannot see the atlas fragmentation that 02.5-09 showed actually drives the curvature
failure.

Stated in those terms **only if the data supports it**, and just as plainly if not.

### Falsifiable predictions — printed BEFORE the results cell

1. **Under-training is the cause:** arms B/C/C′/D develop a near-zero mode in the
   weight-mass distribution (charts genuinely dying), with reconstruction, H0 merge
   retention, and curvature Spearman improving relative to arm A.
2. **The CAE is simply a poor fit here:** every arm's norm distribution stays unimodal
   and tight, occupancy stays at 8, downstream numbers do not move. Reported just as
   plainly — an equally publishable answer.
3. **(Primary, per above)** Occupancy collapses in some arms/seeds while weight mass
   stays unimodal — the two instruments disagree, and the paper's pruning criterion is
   the thing that fails.
4. **Dose-dependence discriminator:** if C′ (74 % shrink) kills charts and C (12.6 %)
   does not, decay strength was the binding constraint. If *neither* does, decay is
   irrelevant on this manifold and P4's arithmetic was necessary but not sufficient.

### Measured per arm × seed

1. **Both norm vectors, in full** — per-layer-Frobenius sums (the brief's instrument)
   *and* `cae.chart_survival` mass ratios. Always printed as vectors.
2. **`chart_survival(model, prune_tol=1e-2)`** — `prune_tol` is **not a fresh pick**:
   it is the value 02.5-09 itself used, so the count is directly comparable to its
   existing 8/8. That is the justification.
3. **Argmax occupancy** — `chart_curvature.chart_curvature_field(...)["n_charts_used"]`,
   the instrument behind the 8/8/3/5 seed table.
4. **Reconstruction relative error**, vs a matched `cae.PlainAutoEncoder` at the 2-D
   bottleneck trained on the same protocol (CLAUDE.md's required baseline).
5. **H0 merge retention** vs the ambient roll — `topoae.persistence_pairs` MST edge-set
   instrument reused unchanged from `notebooks/quick_topoae_vs_cae_persistence.ipynb`,
   with a **dimension-matched** plain AE (latent = the arm's `embed_dim`) as baseline.
6. **Curvature Spearman** — `curvature_probe.spearman_gate_statistic` against
   `curvature_probe.swiss_roll_analytic_H_scaled`, plus the raw-point
   `centroid_mean_curvature(k=30)` reference.

### On the threshold trap

Measured on the sealed PU fit, the two instruments have very different spreads:

- per-layer-Frobenius sums: **30.71 – 31.54**, max/min = **1.027**
- `chart_survival` mass ratios: **0.498 – 1.000**, max/min = **2.01**

`chart_survival` is the more sensitive of the two, but still **unimodal with no
near-zero mode**; 16/16 survive at any `prune_tol < 0.498`. A count is therefore
uninformative at *any* threshold on that fit.

So the verdict is read off **distribution shape** — does a near-zero mode appear, does
the distribution go bimodal — **never off a count alone**. Every count is printed
beside its full vector, and a histogram per arm is plotted so the reader can see the
threshold is not doing the work.

### Cost — measured, not guessed

Timing probe on this machine (scratchpad, nothing kept):

| quantity | cost |
|---|---|
| CAE, 14 torch threads (default) | 3.52 s/epoch |
| CAE, **4 torch threads** | **1.03 s/epoch** |
| plain AE | ~0.1 s/epoch |
| `chart_curvature_field` (n=3000) | 2.7 s |
| `persistence_pairs` (n=3000) | 1.9 s |

Default threading oversubscribes a batch of 64 across 20 cores and costs **3.4×**.
The notebook sets `torch.set_num_threads(4)` explicitly and records it, so the run is
reproducible on this machine.

Projected: ~2,000 CAE epochs ≈ 34 min, plain baselines ≈ 5 min, measurement ≈ 4 min →
**~43 min**, inside the ~45 min bound. Plain-AE baselines are **deduplicated by config**
(arms B and D have identical plain-AE settings; `lip_weight` and `n_charts` do not exist
for a plain AE), which is exact, not an approximation. Wall-clock is reported. If the
bound is breached, **arm B is cut before any seed**, and the cut is stated in the
notebook.

---

## Part 3 — Constraints (unchanged, all honoured)

- Never modify `cae.py`, `topoae.py`, `curvature.py`, `curvature_probe.py`,
  `chart_curvature.py`, `cache.py`, `mknn.py`, `geometry_probes.py`, `subsample.py`,
  `pyproject.toml`, `src/effdim/`. Import and call unmodified.
- Never retrain, overwrite, or re-key a sealed fit. Fresh in-notebook training only.
  **No writes to `notebooks/.cache/`** — tree hash recorded before the run and verified
  byte-identical after. Baseline:
  `a88da1f7208337ea8d5d25eab2ef3593688d91010e09766c3341370109438987`
- Sealed verdicts never reopened: `CAE_VERDICT = FAIL`, the 02.4 TopoAE verdict,
  `CURVATURE_VERDICT = FAIL`. No verdict artifact, no threshold table, no
  pre-registration, no cfg-hash cache keys.
- `.planning/phases/02.5-*` untouched (02.5-09 sits at an OPEN blocking checkpoint with
  02.5-10..13 behind it). The audit **read** those files; it wrote nothing.
- Additive only — no existing notebook or runner deleted or rewritten.
- Full suite stays at **286 passed**.
- One row added to STATE.md "Quick Tasks Completed". `ROADMAP.md` untouched.

### Deviation from GSD defaults (accepted by coordinator)

**Worktree isolation disabled.** `workflow.use_worktrees` defaults to true, but `.venv/`
and `notebooks/.cache/` are both gitignored and would be **absent from a fresh
worktree**. The environment requires `.venv/bin/python`, and the no-cache-writes
constraint requires hashing `notebooks/.cache/`. Neither is possible in an isolated
worktree. Execution runs on the main checkout, branch `isomap-curvature`.

---

## Tasks

1. **Write** `notebooks/quick_cae_undertraining_test.ipynb` — predictions stated before
   the results cell.
2. **Execute** it end to end with
   `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace`, detached,
   polled by PID. Commit with outputs.
3. **Verify** cache tree hash byte-identical, suite at 286, no module edits.
4. **Write** `SUMMARY.md`, add one row to STATE.md, commit.
