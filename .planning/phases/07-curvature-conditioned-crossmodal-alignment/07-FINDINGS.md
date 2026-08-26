# Phase 7 Findings — Curvature-Conditioned Crossmodal Alignment

**Date:** 2026-08-26. **Milestone:** v1.1 PU Manifold Curvature.

**Research question** (`07-CONTEXT.md` Sec 1): does the curvature of the PU embedding manifold
explain the weak crossmodal convergence reported by the Platonic Universe paper
(arXiv:2509.19453)? **Answer, verbatim from the frozen record's `verdict` row:
`ASSOCIATION DETECTED`.** All three `d` in `D_SWEEP = (20, 25, 32)` clear the negative tail of
`spearman(||H||, MKNN)` (`rho` -0.11218 at d=20, -0.12789 at d=25, -0.02373 at d=32), the
direction the research hypothesis predicted, licensed by a positive control that recovers a
planted effect as small as 0.05 on PU's own realized dynamic range. **This is not a clean
result.** A density partial collapses 49-78% of the raw association, and at d=20 and d=32 the
density-controlled residual sits only marginally above its own d's null threshold. d=32's raw
effect (-0.0237) sits close to the positive control's own un-cleared 0.02 target, in the band the
phase's own power analysis calls marginal. **So: the association is carried by d=20 and d=25, and
most of it there is density.** Every number below is quoted from `07-04-SUMMARY.md` and the
frozen record, not recomputed.

---

## 1. Measured results

### 1.1 The three-`d` sweep — headline statistic

| d | var_explained | cond(g) median | observed rho | direction | neg-tail thresh | clears neg | pos-tail thresh | clears pos | clears_either |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.98194 | 15.73 | -0.11218 | negative | 0.02062 | **True** | 0.02098 | False | **True** |
| 25 | 0.98432 | 17.98 | -0.12789 | negative | 0.01968 | **True** | 0.01890 | False | **True** |
| 32 | 0.98647 | 15.54 | -0.02373 | negative | 0.01873 | **True** | 0.01861 | False | **True** |

All three clear on the negative tail only — the sign matches the research hypothesis (more
curvature, worse crossmodal alignment) at every `d`, and none clears the positive tail at any
`d`. `||H||` medians: 37.19 (d=20), 41.41 (d=25), 47.03 (d=32).

### 1.2 Sensitivity grid — `MKNN_K_GRID`, point estimate only, non-gating

Sign and rough magnitude agree with the `HEADLINE_K=20` value at every neighboring `k`; this is
not a `k` artifact. Only `HEADLINE_K=20` carries a permutation null and feeds the verdict
(`SENSITIVITY_RULE`); the remaining `k` values below are point estimates that cannot overturn or
escalate it.

| d | k=5 | k=10 | k=20 (headline) | k=50 |
|---|---|---|---|---|
| 20 | -0.0938 | -0.1077 | -0.11218 | -0.1292 |
| 25 | -0.0846 | -0.1078 | -0.12789 | -0.1530 |
| 32 | -0.0189 | -0.0223 | -0.02373 | -0.0401 |

`mknn_n_distinct_by_k` (identical across `d`, since MKNN depends only on the frozen embeddings
and `k`): `{5: 5, 10: 9, 20: 15, 50: 36}`. At `HEADLINE_K=20`, the per-point MKNN statistic is
`j/k` for integer `j`, so there are at most `k+1=21` distinct values across the whole `n=10,000`
cloud — the measured 15 is why the significance route is permutation-based rather than
`spearmanr`'s asymptotic p-value (see Sec 3 below).

### 1.3 The D7-02 positive control — target vs. achieved

Planted at PU's own realized `d=20` `||H||` dynamic range (`h_norm` p95/p05 spread ≈ 1.5),
`HEADLINE_K=20`, `POSITIVE_CONTROL_SEED=20260825`, through the identical two-tailed permutation
machinery used on the real sweep:

| target rho | achieved rho | clears_either | direction |
|---|---|---|---|
| 0.02 | 0.02004 | False | neither |
| 0.05 | 0.05003 | **True** | positive |
| 0.10 | 0.10004 | **True** | positive |
| 0.20 | 0.20004 | **True** | positive |

`smallest_cleared_target = 0.05`.

### 1.4 Density and hubness diagnostics — reported, gating nothing (D7-03)

Computed once (density is `d`-independent) and recombined with each `d`'s field:

- `spearman(density, MKNN) = -0.2121` (constant across `d`).
- `density_ratio_p95_p05 = 5.98e7` — density p05=6.07e4, p50=2.29e9, p95=3.63e12. A striking
  diagnostic oddity, noted here and interpreted no further.
- `hubness_skewness_a = 1.0486` (HSC column), `hubness_skewness_b = 1.1880` (Legacy Survey
  column).
- `chance_floor = 0.002` at `n=10,000`, `k=20`.

The density partial, per `d` — `spearman(density, ||H||)` / `partial_rho_raw` /
`partial_rho_density_controlled`:

| d | spearman(density, \|\|H\|\|) | partial_raw | partial_density_controlled | % of raw explained by density |
|---|---|---|---|---|
| 20 | 0.4281 | -0.11218 | **-0.02419** | ~78% |
| 25 | 0.3150 | -0.12789 | **-0.06583** | ~49% |
| 32 | 0.0118 | -0.02373 | **-0.02172** | ~8% |

At d=20 and d=32 the density-controlled residual sits only marginally above its own `d`'s null
threshold rather than clearly above it (0.0242 vs. 0.0206 at d=20 — 17% above; 0.0217 vs. 0.0187
at d=32 — 16% above). At d=25 the residual stays clearly above its threshold (0.0658 vs. 0.0197,
over 3x). **This mirrors Phase 4's own recorded finding — its `HOLDS` verdict was 0.82 correlated
with density and its raw gap was mostly a region-size artifact (`04-FINDINGS.md` Sec 5).**
`DIAGNOSTICS_ARE_NON_GATING = True` is honored exactly here: none of this changes the verdict in
Sec 1.1 — `apply_verdict`'s signature has exactly two parameters, neither naming density, so the
non-gating property is structural, not a promise. Both facts stand side by side: the verdict is
`ASSOCIATION DETECTED` by the pre-registered rule, **and** most of the raw association at d=20 is
density, not curvature.

---

## 2. What licensed the reading — the D7-02 positive control

`VERDICT_RULE`'s D7-02 override exists because, without it, a null result and an underpowered
test are indistinguishable, and a null was the likely outcome absent this check. The positive
control planted a curvature-MKNN relationship at PU's own realized `d=20` `||H||` dynamic range —
not Phase 6's retired selfcheck, which planted `rng.random(n)`, a ~20x-spread field, against PU's
measured order-2x spread — and ran it through the identical two-tailed permutation machinery used
on the real sweep.

**The test recovers a planted effect as small as `rho=0.05`** (Sec 1.3). The observed magnitudes
at d=20 (-0.11218) and d=25 (-0.12789) comfortably exceed this floor — more than double it. d=32's
observed magnitude (-0.02373) sits close to the control's own un-cleared 0.02 target, in the band
the phase's own power analysis calls marginal: these are not strictly contradictory (the control's
null is built on a differently-planted array, and -0.02373 does exceed its own 0.01873 threshold),
but d=32 is where detection is weakest by the phase's own measurement.

One structural boundary on what this control can and cannot confirm: `plant_positive_control`
rank-transforms `h_real` before ever touching its raw values, so the mechanism is rank-invariant
by construction (07-03's own wide-vs-narrow separation test measured **no separation** between a
PU-matched narrow fixture and a Phase-6-matched wide one — both recovered
`smallest_cleared_target=0.10` at `n=500`). The control validates the rank-detection mechanism
itself; it cannot independently confirm that PU's particular narrow dynamic range was, or was not,
the limiting factor in any weaker result. Named here as a boundary, not closed.

---

## 3. The instrument — validated on a range, not a point

The curvature field is `decoder_curvature.plain_decoder_curvature` through a
`cae.PlainAutoEncoder(768 -> d -> 768)` trained by `cae.train_plain_ae` — differentiating
`model.decode` alone, never the encoder-composed round trip. It is validated against analytic
ground truth at `d=20` and `D=768` (`07-CONTEXT.md` Sec 4), and the fidelity is a **RANGE**:
`INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99)`. No single figure appears anywhere in this document, or
in the notebook, as the instrument's fidelity — quoting the high end alone would invite a reader
to find the one cell that scored it.

**Naming the cell a reviewer will find:** the `cubic@768` fixture reconstructs at 99.70% and
scores `rho=+0.5253` — the low end of the range. `ridge@768` reconstructs at 99.88% and scores
`rho=+0.9745` — the high end. **Reconstruction quality does not predict fidelity.** What separates
the two cells is the surface's own curvature variation (`II` coefficient of variation 0.104 vs.
0.483), which for PU is unknown. This phase's own holdout `var_explained` — 0.98194 (d=20),
0.98432 (d=25), 0.98647 (d=32) — is high, but per this same finding, high reconstruction does
**not** license a precise curvature-fidelity claim for PU's own field. The fidelity claim for any
PU cell in this record stays the range, `+0.53` to `+0.99`.

Estimated `||H||` spread is reported only as an **order of magnitude**, never a precise quantity:
the instrument's est/true `||H||` ratio was measured swinging 0.665 to 1.626 non-monotonically
across a 17-fold range of true spread (`07-CONTEXT.md` Sec 4). PU's own measured spread — p95/p05
ratios of 1.41 (d=20), 1.31 (d=25), 1.23 (d=32) — is read as "low, order 2" and nothing finer.

---

## 4. CLAUDE.md Swiss roll gate — declared satisfied, not skipped

CLAUDE.md requires a Swiss roll sanity-check notebook for every new manifold-learning or
representation-learning model. **Phase 7 introduces no new model.** It reuses
`cae.PlainAutoEncoder` and `decoder_curvature.plain_decoder_curvature` verbatim — the exact same
instrument, unedited, imported unchanged (`crossmodal_curvature.py`'s own module docstring names
both as sealed). `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb` already tests that
exact combination against a known analytic answer on the Swiss roll, and is the artifact that
satisfies CLAUDE.md's gate for this phase's instrument.

The 2026-08-25 analytic-fixture sweep behind `INSTRUMENT_FIDELITY_RANGE` (Sec 3 above) is, if
anything, a **stronger** check than a fresh Swiss roll run would be for this phase's purpose: it
has a closed-form answer at the same `d=20` and `D=768` PU actually uses, whereas the roll is a
2-D sheet embedded in 3-D — a different ambient dimension from anything this phase fits. This
section is a **declaration of an already-satisfied gate**, not a waiver: the obligation was met
before this phase began, by 02.6 and by the analytic-fixture sweep, and no new Swiss roll
notebook is owed here.

---

## 5. Accepted limitations

- **Single seed across the whole `d`-sweep.** `SEED_HANDLING_RULE = "single_seed_across_d_sweep"`
  is an ACCEPTED LIMITATION inherited from Phase 5's measured seed-instability of decoder
  curvature fields — never a silent assumption that one seed is representative. Agreement across
  all three `d` (Sec 1.1) is therefore **not** evidence of seed stability; it is agreement at one
  seed, sweeping only `d`.
- **The field is evaluated on all 10,000 rows, including the 8,000 the decoder trained on**
  (`FIELD_EVALUATED_ON = "all_10000_rows_including_the_8000_training_rows"`). This is not a
  fresh-holdout evaluation of the curvature field itself, only of the reconstruction loss used to
  fit it.
- **PU reconstruction shows no plateau anywhere through `d=48`** (`07-CONTEXT.md` Sec 5: 97.303%
  at d=10 climbing to 98.896% at d=48, with non-monotonic increments 0.214% → 0.194% → 0.270%).
  Every `d` in `D_SWEEP` therefore describes a truncated approximation of PU, not a converged fit
  — this is why the sweep exists at three `d` rather than one, and it tests whether the truncation
  matters for the conclusion rather than removing the truncation.
- **The density partial and hubness numbers (Sec 1.4) are reported and gate nothing.** They
  qualify the verdict's interpretation heavily — most of the d=20 association is density — without
  being permitted to change `ASSOCIATION DETECTED` itself, by the pre-registered rule's own
  design.
- **The per-point MKNN array has at most 21 distinct values across 10,000 points** (measured: 15
  at `HEADLINE_K=20`), because the statistic is `j/k` for integer `j`. This is why significance is
  established by `two_tailed_permutation_null` rather than `spearmanr`'s asymptotic p-value, which
  assumes no ties.

---

## 6. What this phase does not claim

- **That the field measures true PU curvature.** No ground truth for PU exists anywhere in this
  record. The analytic validation gives a RANGE (`+0.53` to `+0.99`), never a point estimate, and
  no rank statistic in this document is reported without that range stated beside it.
- **That a null result would mean no curvature-alignment relationship exists**, absent the D7-02
  positive-control evidence (Sec 2) that the test could have found one. (Moot here — the verdict
  is `ASSOCIATION DETECTED`, not a null — but the rule that would have governed a null is
  unchanged and recorded for completeness.)
- **Anything about CKA.** It is not implemented anywhere in this codebase (D7-07). MKNN
  (`ALIGNMENT_METRIC = "mknn"`, frozen and carried on every record row) is the source paper's
  headline probe and the only alignment metric this phase measured. This document says nothing
  about CKA or any other alignment metric.
- **Any extrapolation of these MKNN numbers to the source paper's `n=101,725`.** This milestone
  runs at `n=10,000`, where the `k/n` chance floor (`0.002` at `k=20`) is roughly ten times higher
  than it would be at the source paper's scale.
- **Any reinterpretation of a sealed verdict from Phases 2, 02.x, 3, 03.1, 4, 5, or 6.** None is
  reopened, softened, recomputed, or reinterpreted here.
- **That Phase 4's `HOLDS` is evidence of a curvature-alignment association.** It is explicitly
  not cited as one anywhere in this phase. Phase 4's split axis was measured
  `spearman(density, signed_projection) = +0.8208` (n=9500), and its own findings attribute
  nearly the entire raw-score gap to region-size imbalance between its two regions (n=6256 vs.
  n=3244) — the density confound Sec 1.4 measures here is the same category of artifact, on a
  different instrument, at a different unit of observation.

---

## 7. Provenance

- **Freeze commit (`preregistration_commit`, D7-06):** `f032745f6450068c63763993d39fa112fd36bb8c`
  — `crossmodal_curvature.py`'s constants block, `VERDICT_RULE`, `VERDICT_VALUES`, committed
  before any Phase 7 number existed.
- **Run commit (`run_commit`):** `a4537369be204b784d026ac36c6bfc7b14ea483d` — the code that
  produced the frozen record's numbers.
- **Strict-ancestor proof, both checks (a single `--is-ancestor` is insufficient — a commit is its
  own ancestor, so it alone would pass even if a number were produced in the freeze commit
  itself; `rev-list --count` closes that gap by requiring at least one commit strictly between
  them):**
  - `git merge-base --is-ancestor f032745f6450068c63763993d39fa112fd36bb8c a4537369be204b784d026ac36c6bfc7b14ea483d` — exit 0.
  - `git rev-list --count f032745f6450068c63763993d39fa112fd36bb8c..a4537369be204b784d026ac36c6bfc7b14ea483d` — **10** (>= 1).
- **Record:** `notebooks/.cache/07_crossmodal_curvature.jsonl` — 8 rows (3 sweep, 4
  positive-control, 1 verdict), gitignored per `CLAUDE.md`'s milestone-artifact convention; not
  tracked in git.
- **Fields:** `notebooks/.cache/07_crossmodal_curvature_fields.npz` — per-`d` `||H||` and `cond(g)`
  arrays; gitignored, not tracked in git.
- **The nine pre-existing `notebooks/diagnostics/07_*_run.py` spike scripts** (fixture sweep,
  low-spread control, noise calibration, PU latent recon sweep, PU plain AE fit, PU betti probe
  ×2, CAE divergence probe ×2) are **informational inputs that predate the freeze**. They satisfy
  nothing in this phase's pre-registration and were **not re-run for any number in this document**
  — every number here traces to the frozen record and `07-04-SUMMARY.md` alone.

---

## 8. Two planner-assumption resolutions, named for a later reader

Two edge-probe rows in this phase were resolved on planner assumption rather than explicit
developer instruction, ratified at plan 07-01's blocking checkpoint (`ratify-all`):

- **D7-03 — density sign convention.** The density statistic is reported on `1.0 / w`
  (`DENSITY_SIGN_CONVENTION`), not raw `w`, matching Phase 4's `REGN-01` sign convention.
  `curvature_probe.local_density_weights` returns the *inverse* density, so `density = 1.0 / w`
  converts it to a quantity that increases with local point density. **This is overturnable, and
  overturning it flips the sign of every `spearman(density, ...)` number in Sec 1.4** — it would
  not change the headline verdict (density diagnostics gate nothing), but it would change how the
  density-vs-curvature relationship reads in prose.
- **D7-07 — the `ALIGNMENT_METRIC` scope proof.** `ALIGNMENT_METRIC = "mknn"` is frozen as a
  checkable constant carried on every record row, proving CKA's exclusion positively (a fact a
  reader can grep the record for) rather than only in prose. This is a scope decision, not a
  measurement — it could be revisited by a future phase that implements CKA, which does not exist
  anywhere in this codebase today.

---

*Phase: 07-curvature-conditioned-crossmodal-alignment*
*Completed: 2026-08-26*
