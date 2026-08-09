---
phase: quick-20260809-topoae-vs-cae-persistence
plan: 01
subsystem: notebooks
tags: [persistence, topoae, cae, h0, instrument-validation, swiss-roll]
requires: [notebooks/pu_manifold/topoae.py, notebooks/pu_manifold/cae.py, "notebooks/.cache (read-only)"]
provides: [notebooks/quick_topoae_vs_cae_persistence.ipynb]
affects: []
tech-stack:
  added: []
  patterns: [scale-free MST edge agreement, dimension-matched baseline ratios, disjoint half-split resampling null, ambient perturbation ladder]
key-files:
  created: [notebooks/quick_topoae_vs_cae_persistence.ipynb]
  modified: [.planning/STATE.md]
decisions:
  - "QUICK-TC-01 answered MIXED, not PASS: the scale-free merge instrument passes its known-answer check on both datasets, the topological_fidelity ratio fails it on the Swiss roll at 4/4 seeds"
  - "retained and spurious are algebraically complementary for MSTs on a fixed point set, so two new scale-free edge-length asymmetries carry the invents-vs-destroys direction instead"
  - "the CAE BOTH destroys and invents H0 structure, with destruction dominant, refining rather than confirming 02.5-09's INVENTS prior"
metrics:
  duration: ~2h
  completed: 2026-08-09
status: complete
---

# Quick Task: TopoAE vs CAE — which preserves persistent homology better? Summary

One executed notebook measuring 0-dimensional persistence agreement for a Chart Auto-Encoder
and a Topological Auto-Encoder, on the PU embeddings and on the Swiss roll, calibrated against
three references external to both models — and it ended up being as much a result about the
measuring instrument as about either model.

## What was built

`notebooks/quick_topoae_vs_cae_persistence.ipynb` — 27 cells (14 code), committed executed
with outputs, sections §1–§11. It trains nothing on the PU side (all fits loaded read-only
from `notebooks/.cache/`) and trains 16 small models from scratch for the Swiss roll half
(4 models × 4 seeds).

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

### Pilot-vs-recomputed

Every pilot number in `<reference_facts>` reproduced exactly (383-row intersection, 1617-row
leak, plain-AE 277/893/1929 at d=8/20/40, MST retention 0.183 / 0.628 / 0.668, ladder
0.919/0.843/0.670 at f=0.25/0.5/1.0, median NN 0.251). No discrepancy to report.

### Scope pressure (recorded, not acted on)

`destroyed_stretch` / `invented_stretch` are the kind of quantity that would normally earn a
tested function in `notebooks/pu_manifold/`. Per the plan's constraint they were kept as
notebook-level arithmetic over `persistence_pairs`' tested output and no module was touched.
If this line of measurement continues, promoting them into `topoae.py` with tests is the
natural next step — and that is the signal it has outgrown quick mode.

## The most transferable finding

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

## Constraints honoured

- `notebooks/.cache/` tree hash (path, size, mtime) **byte-identical** before and after every
  execution; no cache-write or handoff-delete entry point appears in executable source.
- No sealed fit retrained; only `amend01`-tagged TopoAE stems read.
- No verdict artifact produced; no `*_VERDICT` key computed.
- `src/effdim/` untouched; `pyproject.toml` untouched; **nothing installed** — all four
  persistence libraries probed and printed ABSENT, and the H0-only limitation is stated in the
  notebook's own text before any result.
- `.planning/phases/02.5-*/` untouched (`git status` empty).
- Additive only: one new notebook; `.planning/STATE.md` has exactly 1 added line, 0 deletions.
- Full suite still **286 passed**.

## Known Stubs

None.

## Self-Check: PASSED

- `notebooks/quick_topoae_vs_cae_persistence.ipynb` — FOUND (1,416,134 bytes, executed)
- Commits `89de96e`, `6031bd4`, `9b44265` — all FOUND in `git log`
- `TASK1_OK` / `TASK2_OK` / `TASK3_OK` all printed; final `ANSWERS q1=MIXED q2=0.404489
  q3=BOTH resolved=true`
