---
phase: quick-20260809-topoae-vs-cae-persistence
plan: 01
subsystem: notebooks
tags: [persistence, topoae, cae, h0, h1, h2, instrument-validation, power-analysis, sensitivity-bound, swiss-roll, ripser, persim]
requires: [notebooks/pu_manifold/topoae.py, notebooks/pu_manifold/cae.py, "notebooks/.cache (read-only)", "ripser 0.6.15 (venv-local)", "persim 0.3.8 (venv-local)"]
provides: [notebooks/quick_topoae_vs_cae_persistence.ipynb]
affects: []
tech-stack:
  added: ["ripser 0.6.15 (venv-local, NOT declared in pyproject.toml)", "persim 0.3.8 (venv-local, NOT declared in pyproject.toml)"]
  patterns: [scale-free MST edge agreement, dimension-matched baseline ratios, disjoint half-split resampling null, ambient perturbation ladder, diameter-normalized persistence diagrams, known-answer-derived Betti threshold window, bottleneck saturation marker, known-cycle power fixture for bounding a null, control-derived detection ceiling]
key-files:
  created: [notebooks/quick_topoae_vs_cae_persistence.ipynb]
  modified: [.planning/STATE.md]
decisions:
  - "QUICK-TC-01 answered MIXED, not PASS: the scale-free merge instrument passes its known-answer check on both datasets, the topological_fidelity ratio fails it on the Swiss roll at 4/4 seeds"
  - "retained and spurious are algebraically complementary for MSTs on a fixed point set, so two new scale-free edge-length asymmetries carry the invents-vs-destroys direction instead"
  - "the CAE BOTH destroys and invents H0 structure, with destruction dominant, refining rather than confirming 02.5-09's INVENTS prior"
  - "H1/H2 extension (sections 12-19): the 'Swiss roll is contractible so beta_1 = 0' premise is FALSE for a Vietoris-Rips diagram of a finite sample -- the ambient roll measures beta_1 = 1, so the MEASURED ambient diagram is the null, not the textbook Betti number"
  - "bottleneck distance saturates at half the ambient's longest unmatched bar and then ranks nothing; every H1/H2 ranking is read off Wasserstein, with bottleneck still printed and marked 'sat'"
  - "the PU embedding has no H1 structure resolvable at n=383 or n=2000, so the loop question has no PU answer AT THIS RESOLUTION; the Swiss roll carries the H1 result"
  - "the PU H1 null is BOUNDED, not absolute: a measured power check (S^1 x B^18, beta_1=1 by construction, at d~20/D=768) detects a cycle down to 3.3x the manifold's transverse thickness at n=383 and 2.0x at n=2000, so a cycle at or below the manifold's own thickness is NOT ruled out"
  - "a null from an instrument of unmeasured power is not a finding -- every unqualified 'beta_1 = beta_2 = 0' was replaced with the bounded form, in notebook prose, printed read-outs and this SUMMARY"
  - "H2 has NO power analysis at all (an H2 diagram costs ~77x an H1 one at n=383); the H1 bound does NOT transfer, since detecting a void at fixed n is strictly harder, so every beta_2=0 here is an unbounded null and is the weakest claim in the notebook"
  - "ripser/persim are venv-local and NOT in pyproject.toml (CLAUDE.md bars editing it for v1.1) -- an accepted, documented reproducibility gap for sections 12-19"
metrics:
  duration: ~2h (H0) + ~3h (H1/H2 extension) + ~2h (power/sensitivity amendment)
  completed: 2026-08-10
status: complete
---

# Quick Task: TopoAE vs CAE — which preserves persistent homology better? Summary

One executed notebook measuring 0-dimensional persistence agreement for a Chart Auto-Encoder
and a Topological Auto-Encoder, on the PU embeddings and on the Swiss roll, calibrated against
three references external to both models — and it ended up being as much a result about the
measuring instrument as about either model.

## What was built

`notebooks/quick_topoae_vs_cae_persistence.ipynb` — **47 cells (24 code)**, committed executed
with outputs. It trains nothing on the PU side (all fits loaded read-only from
`notebooks/.cache/`) and trains 16 small models from scratch for the Swiss roll half
(4 models × 4 seeds).

- **§1–§11 — the original H0 half.** Unchanged in substance and re-executed identically:
  every number reproduces to six decimals (`TRIO … cae_retained=0.183246 …`,
  `ANSWERS q1=MIXED q2=0.404489 q3=BOTH resolved=true`).
- **§12–§19 — the H1/H2 extension**, added later on the user's instruction to cover loops and
  voids rather than connectedness alone. 16 appended cells; three existing prose passages
  amended because the extension made them false (below).
- **§14b and §15.5 — the power/sensitivity amendment**, added later still because the
  extension's headline PU claim (`beta_1 = beta_2 = 0`) was **stronger than its evidence**.
  4 appended cells; three printed read-outs and one limitations section amended in place so no
  reader meets the null without its bound (below, and deviation 14).

Full re-execution: **45.2 min wall-clock**, no errors, all 24 code cells executed.
(The pre-amendment notebook ran 36.9 min; §14b costs 363 s and §15.5 6 s of the difference.)

## The three questions, answered

**QUICK-TC-01 — does the instrument behave? MIXED.** This is the finding that reframed
everything else. The two instruments do not behave the same way:

| check | result |
|---|---|
| PU ladder, fidelity ratio | PASS — TopoAE beats its dimension-matched baseline at 6/6 rungs (0.888–0.996) |
| PU ladder, retained merges | PASS — 6/6 rungs |
| Swiss roll, retained merges | PASS — 4/4 seeds |
| Swiss roll, fidelity ratio | **FAIL — 0/4 seeds** (ratios 1.26–1.96) |

On the Swiss roll, where the answer is known in advance, the TopoAE keeps *more* of the true
merge structure than its matched baseline at every seed, yet `topological_fidelity` ranks it
*worse* than that baseline at every seed. So one instrument works and one does not, and
reporting this as "PASS" would have required scoring only the favourable subset.

**QUICK-TC-02 — how bad is the CAE, in units that mean something?** Decisively below its own
dimension-matched baseline, far above chance. On the 383-row primary set: CAE retains **0.183**
of ambient MST edges, plain-AE d40 **0.628**, TopoAE d40 **0.668**, chance floor **0.003**,
identity 1.000. In the perturbation ladder's external unit the CAE reads **worse than
displacing every ambient point by more than 2 nearest-neighbour spacings**, at every sealed
seed. Across the three sealed CAE seeds retained runs 0.102–0.217 (factor 2.1), so the value is
not seed-stable although the conclusion is. The Swiss roll agrees in direction (CAE 0.805–0.888
vs plain_d8 0.893–0.937, losing at 4/4 seeds) but with a far smaller magnitude.

This settles the distinction the calibration existed to make: not "slightly worse than TopoAE",
but **below the baseline** — while remaining far above chance, i.e. badly degraded rather than
random.

**QUICK-TC-03 — invents or destroys? BOTH, dominated by destruction.** `destroyed_stretch`
1.938, `invented_stretch` 1.261, against ~1.0 for both the TopoAE and the baseline; the two
fidelity directions as separate, never-summed ratios are 0.404 (x→z) and 0.098 (z→x). The
02.5-09 prior predicted INVENTS; the invented side *is* active, so the prior is not refuted,
but it is **incomplete** — on the PU embedding the CAE mainly fails to keep real merges. This
refines 02.5-09 rather than contradicting it.

The 20-draw disjoint-half-split null resolves the TopoAE-over-CAE gap (paired p05 0.394,
median 0.463, p95 0.517 — excludes zero on all 40 half-samples).

Secondary set (full 2000-row TopoAE holdout, 1617 of them CAE training rows, i.e. biased in
the CAE's favour): CAE 0.102, TopoAE 0.549, plain 0.532. Same ordering, despite the advantage.

---

# The H1/H2 extension (§12–§19) — loops and voids

Added after the H0 half was complete, on the user's instruction: *"you did not consider
anything above connectedness? … Not just connectedness."* **Additive** — no H0 number was
recomputed, contradicted or removed, and all of them reproduce exactly on re-execution.

## Reproducibility gap, accepted deliberately

`ripser 0.6.15` and `persim 0.3.8` are installed in `.venv` **only**. They are **not declared
in `pyproject.toml`**, because CLAUDE.md bars editing that file for the whole v1.1 milestone
and that constraint was not lifted. **Consequence: a clean checkout reproduces §1–§11 but
not §12–§19** without `.venv/bin/pip install ripser persim`. The notebook raises with exactly
that command on `ImportError` (§12.1) and states the gap in prose before any H1 result. Both
libraries compile from source on Python 3.14 (no cp314 wheel), so presence is checked by
importing, never by shelling out to `pip`.

## The premise that did not survive — a correction to the brief

**The brief stated:** *"The Swiss roll is the sharp test here and its expected answer is known:
it is contractible, so beta_1 = 0. Any long-lived H1 feature in a model's latent is an INVENTED
loop."*

**That is false as applied here, and the notebook measures it rather than assuming it.**
`beta_1 = 0` is a fact about the *continuous manifold*. What is computed is a **Vietoris–Rips**
diagram of a *finite sample*, and for a **rolled** sheet those are different objects: arms that
are far apart along the surface are close through the ambient space, so between the scale at
which neighbouring arms merge and the scale at which the roll fills in, the complex encircles
the roll's empty core.

Measured on the ambient roll — **the input data, no model involved** (§17.1, n = 600 holdout):

| quantity | value |
|---|---|
| longest ambient H1 bar (normalized life) | **0.1807** |
| next-longest | 0.0914 (a 2.0× gap) |
| `beta_1` at all three calibrated thresholds | **1**, not 0 |
| raw birth / death | 0.795 / 1.524 = **7.8 → 14.9 nearest-neighbour spacings** |
| `beta_1` after jittering every point by 2 nn | still 1 (robust to noise) |

Verified stable across `n` = 400/600/900/1200 (normalized life 0.175–0.181 throughout), so it
is not a small-sample fluke.

**Why this mattered rather than being a footnote.** Scoring against the textbook `0` would have
labelled *every model that faithfully reproduced the ambient diagram* as having invented a
loop — the exact inversion of the truth, printed with confidence. The notebook computes the
ambient diagram **first** and reads it before any model, so the framing error is caught by the
data rather than inherited. **The null used throughout §17 is the measured ambient diagram on
the same rows:** invented = more surviving features than ambient, destroyed = fewer. The
known-answer licence for the instrument comes from §14 instead, where a Rips complex genuinely
does recover the textbook Betti numbers.

## What the extension found

**QUICK-H-01 — does the H1/H2 instrument behave? PASS.** `ripser` recovers `beta_1 = 1` for a
circle, `beta_1 = 0` for a contractible disc and `beta_1 = 2, beta_2 = 1` for a torus, at every
threshold in the admissible window. `persim.bottleneck(d, d) == 0.0` exactly, and
`bottleneck(circle, disc) = 0.7942` = exactly half the circle's unmatched bar. The pre-flight
note's circle/disc/bottleneck figures reproduce to 4 decimals; its torus figures do **not**, and
that is the note being under-specified (it records neither `(R, r)` nor the RNG stream) rather
than a disagreement — the notebook states its own torus as `R=2, r=1, 900 points, default_rng(0)`
so it *is* reproducible, and asserts the torus's *structure* rather than its numbers.

**The Betti threshold is derived, not chosen.** Requiring all four known Betti numbers
simultaneously pins the admissible window to `(0.0935, 0.1263]` — only **1.35× wide**.
`BETTI_TAU = 0.1087` is its geometric centre, and every count is printed at three thresholds
spanning the window, all of which agree.

**QUICK-H-02 — loops and voids on the PU embedding? NO PU ANSWER EXISTS *AT THIS RESOLUTION*.**
The ambient embedding has no surviving H1 or H2 feature at n = 383: its longest H1 bar is
0.0582 of the diameter against **0.0686 for a contractible disc** — the same order as a cloud
with no loops at all. The null holds at n = 2000 too (§15.5: longest bar 0.0627, `b1 = 0` at
all three thresholds).

**The bounded claim — and it is the only form the evidence supports:**

> PU shows no cycle whose radius exceeds roughly **3× the manifold's transverse thickness**, at
> `n ≤ 2000`, `d ~ 20`, `D = 768`. A cycle comparable to or smaller than the manifold's own
> thickness would be **invisible** to this measurement and is **NOT ruled out**.

This replaces the earlier unqualified `beta_1 = beta_2 = 0`, which was a null from an
instrument of unmeasured power — see deviation 14 and the sensitivity section below. Within
that bound: there was no ambient loop *of that size* to destroy, and no model invents one
either (every latent's surviving H1 and H2 count is 0). What *is* measurable is fine-scale
geometric agreement in the noise band, and it resolves (H1 RESOLVED, H2 UNRESOLVED):

| model | H1 Wasserstein / plain-AE d40 baseline |
|---|---|
| CAE embed-40 (3 sealed seeds) | 1.484 / 1.440 / 1.147 |
| TopoAE d40 | 1.015 |
| plain AE d40 | 1.000 |
| random d40 latent | 3.920 |

Same direction as H0 (CAE worse than its own baseline, far better than chance) at much smaller
magnitude — and on H2 the ordering **reverses** (CAE 0.859). The notebook labours the point that
**this is not a statement about loops**: with no ambient feature resolvable at this `n`, every
bar is sub-threshold noise.

### The power check that bounds the null (§14b) — added by the amendment

A null is worth nothing without the power to back it, so the H1 instrument was measured against
a manifold **with a cycle by construction**, at the PU regime's own dimensions: `S^1 × B^18`,
`d = 20`, zero-padded and rotated into `D = 768` (the padding and rotation are asserted to be
*exactly* distance-preserving, so the bound is a statement about `d = 20`). `sigma` is the
transverse thickness; the cycle's radius is fixed at 1.0, so `radius/thickness = 1/sigma`. Every
rung has a **control** — the same cloud with the circle filled in, which is contractible.

Detection statistic: **longest H1 bar ÷ second-longest** (invariant under the diameter
normalization). The threshold is *derived*, not chosen, exactly as §14 derives `BETTI_TAU`: the
ceiling is the largest ratio any of **30 no-cycle control draws** produced (**1.34**), and a rung
counts as detected only if its **worst** draw clears it — scoring a rung that detects on some
seeds and not others as a miss, which can only widen the admitted blind region.

At **n = 383** (the primary PU set size), 5 draws per rung:

| sigma | radius/thickness | raw top H1 | raw 2nd | ratio min–med–max | ctrl worst | detected |
|---|---|---|---|---|---|---|
| 0.15 | 6.7× | 1.1247 | 0.1297 | 7.88 – 8.44 – 9.64 | 1.19 | **YES** |
| 0.30 | 3.3× | 0.8044 | 0.2611 | 3.03 – 3.09 – 3.47 | 1.13 | **YES** |
| 0.50 | 2.0× | 0.4836 | 0.4102 | 1.01 – 1.16 – 1.36 | 1.17 | no |
| 0.75 | 1.3× | 0.5703 | 0.5375 | 1.01 – 1.03 – 1.07 | 1.10 | no |
| 1.00 | 1.0× | 0.7855 | 0.7515 | 1.03 – 1.12 – 1.24 | 1.34 | no |
| 1.50 | 0.7× | 1.0993 | 1.0516 | 1.03 – 1.07 – 1.12 | 1.23 | no |

→ **detected down to 3.3× thickness; first missed at 2.0×.**

At **n = 2000** (the secondary set size), 1 draw per rung — so gap 1 is *measured*, not assumed:

| sigma | r/thick | ratio | ctrl | × ceiling | detected | sec |
|---|---|---|---|---|---|---|
| 0.15 | 6.7× | 8.80 | 1.03 | 6.59 | **YES** | 188.7 |
| 0.30 | 3.3× | 3.72 | 1.05 | 2.78 | **YES** | 45.0 |
| 0.50 | 2.0× | 1.39 | 1.04 | **1.04** | YES *(marginal)* | 21.4 |
| 0.75 | 1.3× | 1.05 | 1.04 | 0.79 | no | 20.5 |
| 1.00 | 1.0× | 1.01 | 1.14 | 0.76 | no | 19.4 |
| 1.50 | 0.7× | 1.00 | 1.06 | 0.75 | no | 19.3 |

→ **detected down to 2.0×; first missed at 1.33×.** The bound **tightens with n**, in the
expected direction and by a measured amount. But that finest rung clears the ceiling by only
**1.04×** on a **single draw**, which is inside the draw-to-draw spread the 5-draw n = 383 ladder
measured for the same statistic — it is flagged `marginal` in §14b.3 and could flip on another
seed. **The headline therefore quotes the conservative 3.3×**, the bound at the set where the
models are actually scored. A cycle at or below **1.3×** thickness was missed at *both* sizes.

**And the encouraging half, which is worth as much as the bound.** A prominent cycle (6.7×
thickness) is detected cleanly at **every** sample size tested — ratio 8.44 at n = 383, 6.90 at
n = 1000, 8.80 at n = 2000. So the `r/R` curse that defeated the point-cloud curvature estimator
at `d = 20` does **not** blind persistent homology at the same `d`. The limit is on the *size* of
a resolvable cycle, not on the sample being hopeless at these dimensions.

**Gap left open: `beta_2` has no power analysis at all.** Everything above is H1. No
`S^2 × B^17` fixture was built and no H2 sensitivity was measured, on cost grounds (§15.1
measures an H2 diagram at **~77×** an H1 one at n = 383). The H1 bound does **not** transfer —
detecting a void at fixed `n` is strictly harder — so the blind region for H2 is at least as
large and probably larger. **Every `beta_2 = 0` in this notebook is an unbounded null** and is
marked as such in §14b.5, §15.1, §16.1, §18 and §19.

**QUICK-H-03 — the Swiss roll. The CAE invents no loop; it exaggerates the one that is there.**
Against the measured ambient `beta_1 = 1`, across 4 seeds:

| model | `beta_1` per seed | verdict | longest-bar life ÷ ambient (median) |
|---|---|---|---|
| CAE_embed8 | 1, 1, 1, 1 | MATCHES | **1.25×** |
| plain_d8 (dim-matched) | 1, 1, 1, 1 | MATCHES | 1.09× |
| TopoAE_d2 | 1, 1, 0, 1 | MIXED | 0.63× |
| plain_d2 | 0, 0, 0, 0 | DESTROYS | 0.45× |
| random d8 | 0 | — | — |

On the **count**, `prior_02509 = NOT SEPARATED`: the CAE reproduces the ambient loop and adds
nothing, exactly as its baseline does. On the **lives**, the CAE renders that loop 1.25× as
persistent as ambient against 1.09× for its own baseline, larger at **3 of 4 seeds** — so
02.5-09's prior survives only in a weakened form: *not an invented feature, an exaggerated one*.
With 4 paired seeds, one fixture and no resampling null on this quantity, that is recorded as
**suggestive, not settled**.

## The most transferable finding: bottleneck distance saturates

Bottleneck is a *max* over a matching, so once a latent has nothing close enough to the
ambient's longest bar, that bar is sent to the diagonal at exactly half its life and **the
distance stops responding to anything else**. Measured, not asserted:

- Swiss roll: **8 of 16** model rows return the identical `0.0904`; the 16 rows take only
  **9 distinct bottleneck values against 16 distinct Wasserstein values**.
- PU: the perturbation ladder pins its top rung (`f = 2.0`) exactly on the saturation value
  `0.0291`, and 3 of 5 model rows sit on it. The H2 bottleneck ladder is **non-monotone in `f`**
  for the same reason, while all three Wasserstein ladders are monotone.

This is the H1/H2 counterpart of §6.3's H0 result about `topological_fidelity`, reached by a
different route on a different statistic: a number that looks like a ranking and is not one.
**Bottleneck is still computed and printed in every table, marked `sat`** — a saturated
instrument reported openly is a result; quietly swapping to another one would not be. All
rankings are read off Wasserstein and the Betti counts.

## How the two design traps were handled

1. **Scale.** Every diagram is divided by **its own cloud's diameter**, so lives are
   dimensionless fractions of that space's extent and bottleneck/Wasserstein are exactly
   invariant under a global rescale of either space (all three are 1-homogeneous). Diameter was
   preferred to median-nn because a Rips filtration terminates at the diameter, so every
   normalized life lands in `[0, 1]`. **Additionally** every ambient-vs-latent distance is
   reported as a **ratio against the plain-AE baseline at the same latent dimension** — the rule
   §2 already binds the notebook to. Both are stated and justified in §13.
2. **No distance in isolation.** Every H1/H2 number lands on the *existing* scaffolding:
   identity self-test, chance floor from a random latent, and the ambient perturbation ladder
   `sigma = f·nn/√D` with realized displacement asserted within 20% of nominal (PU realized
   0.250/0.499/0.999/2.000; Swiss 0.227/0.440/0.886/1.807). A **separate** ladder was built for
   the Swiss roll, per §9's rule that a nearest-neighbour spacing does not transfer between
   manifolds.

## Cost, and what it forced

No subsampling was needed: every diagram is computed on the full row set for its section
(PU n = 383, Swiss holdout n = 600, known-answer clouds 400/400/900). **H2 costs ~75× an H1
diagram** (PU: 0.19 s vs 14.2 s; Swiss: 1.7 s vs 26 s), which forced two *repetition* reductions,
both printed with their timings rather than taken silently:

- PU ladder: H2 at 2 seeds per rung against 5 for H1.
- Swiss models: H2 at the representative seed only.
- The two secondary CAE seeds carry H1 only.

Consequence recorded in §19: **no H2 number in this notebook carries an across-seed spread.**

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 2 — missing critical functionality] `retained` and `spurious` cannot answer QUICK-TC-03.**
The plan's `<measurement_design>` treats them as two independent directional rates ("Low
`retained` means DESTROYED. High `spurious` means INVENTED"). They are not independent: every
MST on `n` points has exactly `n − 1` edges, so `|E_x| = |E_z|` always and
`spurious ≡ 1 − retained` identically. The pair is one number and cannot separate the two
failure modes. Both are still reported and the identity is asserted in-notebook, but the
direction is carried by two genuinely independent scale-free edge-length asymmetries added for
the purpose (`destroyed_stretch`, `invented_stretch`), each a ratio taken within a single
space. Without this QUICK-TC-03 was unanswerable. Notebook §2, §6.2. Commit `89de96e`.

**2. [Rule 1 — bug] `realized_disp_ratio` was printed in the wrong units.**
First implementation printed realized-over-nominal (≈ 1.0); the plan's verifier expects the
realized displacement expressed in nearest-neighbour spacings, which must equal `f`. Fixed to
`disp / median_nn`, and the in-notebook assertion changed to compare against `f` rather than
against 1.0. The `sqrt(D)` scaling itself was correct throughout — realized ratios came out
0.250 / 0.499 / 0.999 / 2.000 against nominal 0.25 / 0.5 / 1.0 / 2.0. Commit `89de96e`.

**3. [Rule 1 — bug] `_amend01_` never appeared literally in source.**
The stems were built from an `AMEND = "amend01"` variable, so the tag never appeared as a
literal and the threat-model check T-TC-04 (proving only amend01 stems were read) could not
pass by grep. The tag is now written literally into every stem. Commit `89de96e`.

**4. [Rule 1 — bug] §6.1 asserted a conclusion its own numbers contradicted.**
The first version printed "The CAE's H0 behaviour is stable across its three sealed seeds"
while the measured retained values were 0.183 / 0.217 / 0.102 — a factor of 2.1, and
`loss_z_to_x` varied by 4.8×. Replaced with a computed spread table and an honest read-out
separating the unstable *value* from the seed-robust *conclusion*. The summary row also
overflowed its column widths and was rebuilt as a vertical table. Commit `6031bd4`.

**5. [Rule 1 — bug] §10 reported QUICK-TC-01 as PASS on a favourable subset.**
Q1 was computed from the PU fidelity ladder plus Swiss-roll *retained* only, which returned
PASS and silently omitted that the fidelity ratio loses to its own baseline at 4/4 Swiss roll
seeds. Q1 is now computed over both instruments on both datasets and returns MIXED, with the
four sub-checks printed individually. This is the single most consequential correction in the
task. Commit `9b44265`.

### Plan defects found (reported, not worked around)

**6. `<read_first>` names a file that does not exist.** Tasks 1 and 3 reference
`notebooks/02.2_chart_autoencoder.ipynb` for house style. It was renamed to
`notebooks/02.2_swiss_roll_cae_check.ipynb` in commit `cb2018f`. Used the renamed file plus
`notebooks/02_k_sensitivity_refit.ipynb` for the `§N` / provenance-cell idiom. No impact.

**7. Tasks 2 and 3's `<verify>` gate is unsatisfiable as written.** It asserts
`git diff --name-only --diff-filter=MDR HEAD -- . ':!.planning'` is empty, but Task 1 commits
the notebook, so Tasks 2 and 3 necessarily modify a tracked file — their own artifact. Ran the
gate with `':!notebooks/quick_topoae_vs_cae_persistence.ipynb'` added, which preserves the
threat-model intent of T-TC-02 (no *pre-existing* file is touched) and matches Task 3's own
`<done>` wording ("no **other** pre-existing tracked file is modified"). Verified separately
that the notebook is the only modified tracked file at each step.

**8. Swiss roll chart survival did not reproduce 02.5-09's instability.** Under this task's
matched-capacity protocol (`hidden=(64,64,64)`, `lr=3e-4`) all **8/8** charts survive at all
four torch seeds. 02.5-09 measured 8/8/3/5 on the same seeds using the 02.2 notebook's
`hidden=[64,64]`, `lr=1e-3`. The instability is a property of that protocol, not of the CAE as
such. Recorded in the notebook's §9 output. This does not reopen or revise 02.5-09, whose
numbers stand for its own configuration.

**9. STATE.md's row was committed separately from the notebook.** The plan asks for one commit
carrying notebook + plan directory + STATE.md row, but the row must cite the commit SHA of the
work it points to. Committed the notebook and plan first (`9b44265`), then the STATE.md row
citing that SHA. Two commits, but the citation is truthful.

### Deviations in the H1/H2 extension

**10. [premise correction — reported, not smoothed over] The brief's `beta_1 = 0` framing for
the Swiss roll is wrong for a Rips diagram.** Detailed above. The brief instructed that any
long-lived latent H1 feature be read as an invented loop; the ambient roll itself measures
`beta_1 = 1`, robustly. The extension uses the measured ambient diagram as the null instead.
This changes the *frame*, not the goal — items 1 and 3 of the brief asked for ambient-vs-latent
comparisons, which is exactly what is delivered.

**11. [Rule 1 — bug, caught in a dry run] Three prose blocks asserted conclusions their own
numbers would not always support.** Each was rewritten to be computed from the data rather than
hardcoded: (a) §15.2 asserted `wasserstein(d, d) == 0.0` exactly — it is an LP objective and
returns ~1e-7, so it is now checked against a stated `1e-5` tolerance while bottleneck is still
required to be exactly `0.0`, with the distinction explained; (b) §17.3's saturation paragraph
said "every row" when 8 of 16 saturate; (c) §18's amplification paragraph hardcoded the words
"stretches" and "more persistent", which the data happen to support (1.25× vs 1.09×) but which
were made branch-driven so they cannot print a false direction. All three were found by
executing the new cells against a shrunken fixture before the 37-minute real run.

**12. [Rule 2 — missing critical functionality] A single distance could not answer "is this
readable?"** The first §16 verdict asked only whether models beat the chance floor, which
returned RESOLVED for H2 on numbers whose between-model spread (0.065) is *smaller than the
chance floor's own draw-to-draw spread* (0.496). Split into two explicit questions — separation
from chance, and separation from each other by more than the chance spread — only the second of
which licenses a ranking. H2 correctly reads UNRESOLVED as a result.

**13. Three existing prose passages were amended because the extension made them false.**
The intro cell, §1's probe read-out and §11's first bullet all asserted "no persistence library
is installed and none is installed by this work". Each now records what changed, when, and why,
and points to §12/§19 for the reproducibility gap. **No H0 result was altered** — the amendments
are to claims about the environment, not to measurements.

**14. [Rule 1 — overclaim corrected, on instruction] `beta_1 = beta_2 = 0` was stronger than the
evidence supported, and one supporting sentence was unbacked.** The committed notebook and this
SUMMARY reported, as a *finding*, that the PU embeddings have `beta_1 = beta_2 = 0` — "no H1
answer at all". That is a **null from a finite sample in `d ~ 20`, `D = 768`, produced by an
instrument whose power had never been measured**. Nothing in the notebook established that a
cycle *could* have been detected had one been present, so the null was not a finding.

Fixed by measurement, not by softening the wording: §14b builds an `S^1 × B^18` fixture with
`beta_1 = 1` by construction at the PU regime's own `d` and `D`, and measures the detection
bound at both PU sample sizes (table above). Every unqualified assertion was then replaced with
the bounded form — **notebook prose (§19), printed read-outs (§15.1, §16.1, §18) and this
SUMMARY**. The H0 half is untouched.

Separately, this SUMMARY's sentence *"Confirmed at n = 800/1400/2000 as well"* had **no backing
anywhere in the notebook** — no such sweep existed in any cell or output. Rather than delete the
claim, §15.5 now *measures* the PU ambient null at the secondary n = 2000 set (reusing §8b's
distance matrix): longest H1 bar 0.0627 of the diameter, `b1 = 0` at all three thresholds. The
`n = 800/1400` half is dropped as unbacked; **n = 2000 is now a measured fact rather than an
assertion.**

**15. [Rule 1 — bug, caught by the first full run] `PU_H2_COST_FACTOR` was forward-referenced.**
§14b's gap-2 paragraph quoted the H2/H1 cost factor, which §15.1 defines — one cell *later*. The
run died with `NameError` after completing every §14b measurement. Reworded to point at §15.1's
printed factor instead of interpolating it. Found because the whole notebook was re-executed
rather than the new cells alone.

**16. [process — reported, not hidden] Two nbconvert runs briefly executed the same notebook
in parallel.** A shell-quoting mistake (`A && B && C & D` backgrounds the whole `&&` chain, so
`$S` was set only in the subshell) launched a run whose PID was never captured; a second,
tracked run was then started ~20 s later. Both were writing the same file `--inplace` and
competed for CPU, roughly doubling wall-clock and risking a corrupted write. Detected by process
inspection when the run overran its estimate, and both were killed. The notebook file was
verified unmodified (mtime unchanged, 47 cells, valid JSON) and the cache hash re-verified
before a single clean run was launched. **No result in this SUMMARY comes from the contended
runs** — every number is from the single 45.2-minute clean execution. Recorded because a
silently contended run would have mis-measured exactly the per-diagram timings the notebook uses
to justify its H2 scoping decisions, and because §15.1 asserts `H2 seconds < 90 s`, which
contention could have tripped spuriously.

### Pilot-vs-recomputed

Every pilot number in `<reference_facts>` reproduced exactly (383-row intersection, 1617-row
leak, plain-AE 277/893/1929 at d=8/20/40, MST retention 0.183 / 0.628 / 0.668, ladder
0.919/0.843/0.670 at f=0.25/0.5/1.0, median NN 0.251). No discrepancy to report.

The dispatch note's H1/H2 pre-flight figures also reproduced where they were fully specified —
circle H1 count 1 and life 1.5884, disc 94 features with max life 0.1364,
`bottleneck(d, d) == 0.0`, `bottleneck(circle, disc) = 0.7942`. Its **torus** figures did not,
because it records neither `(R, r)` nor the RNG stream; the notebook states its own torus
fixture explicitly and asserts the structure the check actually rests on.

**The amendment's sensitivity table vs the dispatch note's.** The note supplied an out-of-notebook
sweep; the notebook recomputed it from scratch. **Every detection verdict matches** — YES at
sigma 0.15 and 0.30, no at 0.50 and above, control indistinguishable from noise — so the bound
is the same. The individual ratios differ within draw-to-draw spread, which is why the notebook
runs 5 draws per rung and the note ran one:

| sigma | note ratio | notebook (5 draws) min–med–max | verdict |
|---|---|---|---|
| 0.15 | 7.86 | 7.88 – 8.44 – 9.64 | agree (YES) |
| 0.30 | 3.40 | 3.03 – 3.09 – 3.47 | agree (YES) |
| 0.50 | 1.20 | 1.01 – 1.16 – 1.36 | agree (no) |
| 0.75 | 1.18 | 1.01 – 1.03 – 1.07 | agree (no) |
| 1.00 | 1.02 | 1.03 – 1.12 – 1.24 | agree (no) |
| 1.50 | 1.02 | 1.03 – 1.07 – 1.12 | agree (no) |
| control | 1.06 | 1.00 – 1.06 – 1.34 (30 draws) | agree |

The note's 0.75 and 1.00 values sit just outside the notebook's 5-draw range in opposite
directions; both rungs are undetected either way, so nothing turns on it. Raw top-bar lives also
agree closely (note 1.1352 / 0.8377 / 0.4890 at sigma 0.15 / 0.30 / 0.50 against the notebook's
1.1247 / 0.8044 / 0.4836). Multi-`n` at sigma = 0.15: note 7.86 / 7.28 / 8.27 at n = 383 / 1000 /
2000, notebook 8.44 / 6.90 / 8.80 — same conclusion, detected cleanly at all three.

**One substantive difference, and it is new information rather than a disagreement:** the note
measured the sweep only at n = 383. Running it at n = 2000 as well showed the bound **tightens**
(3.3× → 2.0×), so the n = 383 bound does *not* transfer unchanged to the larger set — which is
precisely the gap the amendment was asked to stop glossing over. Nothing was tuned toward the
note's numbers.

### Scope pressure (recorded, not acted on)

`destroyed_stretch` / `invented_stretch` are the kind of quantity that would normally earn a
tested function in `notebooks/pu_manifold/`. Per the plan's constraint they were kept as
notebook-level arithmetic over `persistence_pairs`' tested output and no module was touched.
If this line of measurement continues, promoting them into `topoae.py` with tests is the
natural next step — and that is the signal it has outgrown quick mode.

## The most transferable finding (H0)

`topological_fidelity`'s baseline-relative ratio — the 02.4 T1 gate statistic — **can rank a
model best while its H0 merge structure is near-destroyed.** Measured cause (§6.3): it is a sum
of squared *absolute* edge-length differences, and `latent_unit_scale` equalises only the
*all-pairs* scale. The CAE's ambient-MST edges are 2.02 long in its latent against 3.25 for the
other two models at matched all-pairs scale, so it earns a small penalty for having a
compressed local scale rather than for preserving merges. The Swiss roll then confirmed this
from the opposite direction on a known-answer manifold.

It appears reliable *within* a model family at fixed `d` (the PU ladder, where it works) and
unreliable *across* families with different latent scale behaviour.

**This changes no sealed verdict.** It is a limitation of the statistic, recorded for whoever
reads it next. `CAE_VERDICT`, the 02.4 TopoAE verdict and `CURVATURE_VERDICT` are untouched,
and 02.5-09's checkpoint is still open with 02.5-10..13 still blocked.

**Both halves of this task ended up producing the same shape of result about a different
statistic** — H0's `topological_fidelity` ratio and H1/H2's bottleneck distance each rank
confidently while measuring something other than what they are read for. That is the durable
output of this task, more than any number about either model.

## Constraints honoured

- `notebooks/.cache/` tree hash (path, size, mtime) **byte-identical** before and after every
  execution — `dd0af6c3ee6328f0…` unchanged across the full 45.2-minute re-run, and a
  **content** hash (`sha256` of every file's bytes, `a88da1f7208337ea…`) was captured before the
  amendment and re-verified after it, so the read-only claim rests on file contents and not on
  metadata alone. No cache-write
  or handoff-delete entry point appears in executable source. (This hash uses this run's own
  `path size mtime` format; it is comparable before-vs-after, not to the earlier run's
  `e69b9d89…`, which was computed differently.)
- No sealed fit retrained; only `amend01`-tagged TopoAE stems read. The Swiss roll models are
  trained in-notebook as before and the H1/H2 sections **re-encode** them rather than retraining.
- No verdict artifact produced; no `*_VERDICT` key computed.
- `src/effdim/` untouched; **`pyproject.toml` untouched** — which is precisely why the
  ripser/persim reproducibility gap exists and is documented rather than resolved.
- **Two libraries WERE installed** (`ripser 0.6.15`, `persim 0.3.8`), venv-local, at the user's
  explicit instruction — a deliberate reversal of the H0 half's "nothing is installed" stance,
  recorded in the notebook's own text and above. Neither was re-installed or version-changed by
  this work; presence is proved by import, never by `pip`.
- `.planning/phases/02.5-*/` untouched.
- Additive only: 16 cells appended for the H1/H2 extension and **4 more for the power
  amendment** (§14b markdown + code, §15.5 markdown + code), 0 cells deleted, nothing reordered;
  `.planning/STATE.md` row amended in place, never duplicated.
- The power amendment changed **no measurement** — it added one, and rewrote the *claims* three
  printed read-outs (§15.1, §16.1, §18) and §19 draw from it. Every H0 and H1/H2 number in the
  notebook reproduces unchanged.
- Full suite still **286 passed** (`tests` + `notebooks/pu_manifold/tests`).
- `pyproject.toml` untouched; nothing installed — `ripser 0.6.15` / `persim 0.3.8` were already
  present and were used by import only.

## Known Stubs

None.

## Self-Check: PASSED

- `notebooks/quick_topoae_vs_cae_persistence.ipynb` — FOUND (1,778,333 bytes, 47 cells,
  24 code, **0 errors, 0 unexecuted**, execution counts 1–24 in order)
- H0 preservation verified by re-execution, not by assertion: `TRIO n_eval=383
  cae_retained=0.183246 topoae_retained=0.667539 plain_retained=0.628272 cae_ratio=0.404489`
  and `ANSWERS q1=MIXED q2=0.404489 q3=BOTH resolved=true` — identical to the pre-extension run
- `H_ANSWERS instrument=PASS pu_h1=RESOLVED pu_h2=UNRESOLVED pu_ambient_b1=0 pu_ambient_b2=0
  swiss_ambient_b1=1 swiss_ambient_b2=0 swiss_cae=MATCHES swiss_plain_d8=MATCHES
  swiss_topoae=MIXED swiss_plain_d2=DESTROYS prior_02509=NOT SEPARATED`
- `SW_AMP_CMP cae_median=1.2526 plain_d8_median=1.0903 cae_above_baseline_at=3/4`
- `SENS_SUMMARY d=20 D=768 ceiling=1.3355 bound_n383=3.33 bound_n2000=2.00 headline_bound=3.33
  blind_both_at_or_below=1.33 n_large_marginal=True h2_power=NOT_MEASURED wallclock=363s`
- `AMBIENT_PU_SECONDARY n=2000 diam=1.330560 h1_top=0.062743 b1=0 seconds=6.2` — the n = 2000
  confirmation that this SUMMARY previously asserted without any backing is now measured
- No unqualified `beta_1 = beta_2 = 0` remains: grep of every markdown cell, every code cell and
  every cell output finds the phrase only inside text that immediately bounds or negates it
- `notebooks/.cache/` content hash `a88da1f7208337ea…` and metadata hash `dd0af6c3ee6328f0…`
  both identical before and after; `286 passed`
