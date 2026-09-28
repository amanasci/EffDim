# Phase 7 Context — Curvature-Conditioned Crossmodal Alignment

**Created:** 2026-08-25. **Milestone:** v1.1 PU Manifold Curvature.

**This document is self-contained on purpose.** Every number Phase 7 depends on was measured in
an interactive session on 2026-08-25 whose transcript will not survive. All of it is restated
here with its source script, so planning can proceed from this file alone. No sealed verdict from
any prior phase is reopened, softened, recomputed or reinterpreted.

---

## 1. The research question, stated plainly

**Does the curvature of the PU embedding manifold explain the weak crossmodal convergence
reported by the Platonic Universe paper (arXiv:2509.19453)?**

- **If NO** — curvature does not account for the poor crossmodal MKNN, and that explanation is
  removed from the table.
- **If YES** — there is a route to alignment measures that condition on curvature, which may
  recover convergence more in line with the Platonic Representation Hypothesis.

Either answer is a result. A **No** is only publishable with the D7-02 power evidence below.

## 2. Why the existing phases do not answer it

| phase | outcome variable | curvature variable | verdict | why it does not answer the question |
|---|---|---|---|---|
| 4 | crossmodal MKNN | direction (sign split) | HOLDS | split axis `spearman(density, ·) = +0.8208`; raw gap mostly region-size artifact — see §6 |
| 5 | **ridge residual** | 3 CAE decoder `‖H‖` | SPLIT ACROSS SEEDS | wrong outcome variable; the source paper probes MKNN, not linear decodability |
| 6 | **ridge residual** | point-cloud `‖H‖` | NO DETECTABLE RELATIONSHIP | same wrong outcome variable |

**The record contains no interpretable answer to the research question.** Phase 7 supplies one.

## 3. Locked decisions

- **D7-01 — the curvature field, from the validated instrument.**
  `cae.PlainAutoEncoder(in_dim=768, latent_dim=d, hidden=(250,250,250), activation="silu")`,
  trained with `cae.train_plain_ae`, curvature via
  `decoder_curvature.plain_decoder_curvature(model, model.encode(x))` — which differentiates
  `model.decode` ALONE, never the encoder-composed round trip.
  **Run the headline correlation at `d ∈ {20, 25, 32}` and report all three** (§5 explains why a
  single `d` cannot be defended). Same answer at each `d` ⇒ the truncation question is moot for
  the conclusion, which is stronger than picking one.

- **D7-02 — the positive control, and it is not optional.**
  Plant a curvature–MKNN relationship **at PU's realized `‖H‖` dynamic range** and show the test
  recovers it. Phase 6's existing selfcheck does NOT serve: it planted `rng.random(n)`, a
  ~20x-spread field, against PU's order-2x. Without this, a null cannot be distinguished from an
  underpowered test, and a null is the likely outcome.

- **D7-03 — density and hubness, reported and gating nothing.**
  `spearman(density, ‖H‖)`, the density partial on the headline correlation, and
  `mknn.hubness_skewness`. MKNN is a k-NN statistic and therefore mechanically density-sensitive;
  this is exactly how Phase 4's result became uninterpretable (§6).

- **D7-04 — per-point, not per-region.** `mknn.mknn_score` computes `(A & B).sum(axis=1) / k`, a
  per-point array, then averages it away. Retain it: **10,000 paired observations** instead of
  2–3 buckets. Spearman is scale-free, so this also sidesteps the low-dynamic-range problem that
  makes tertile bucketing underpowered here. Headline statistic:
  `spearman(‖H‖_i, MKNN_i)` over all points.

- **D7-05 — additive only.** `linear_probe.py` (Phase 5) and `pointcloud_probe.py` (Phase 6) are
  sealed; import, never edit. `src/effdim/` untouched. New constants live in a new module.

- **D7-06 — freeze before any number.** Constants and the verdict/reporting rule committed in
  source before the runner can produce a PU number, with an `assert_preregistered()` guard and
  git ancestry as the proof, exactly as Phases 5 and 6 established.

- **D7-07 — CKA is out of scope.** Not implemented anywhere in the codebase. MKNN is the source
  paper's headline probe and `notebooks/pu_manifold/mknn.py` is complete
  (`mknn_score`, `permutation_null`, `bootstrap_ci`, `chance_floor`, `hubness_skewness`).
  Adding CKA is a separate decision, not a Phase 7 task.

## 4. The instrument, and the evidence validating it

Script: `notebooks/diagnostics/07_instrument_fixture_sweep_run.py` →
`notebooks/.cache/07_plain_decoder_sweep.jsonl`

Analytic fixtures at `d=20`, `n=5000`, 400 epochs. Both arms measured on the identical cloud;
`ratio` is median est/true `‖H‖` magnitude.

| fixture | D | recon | cloud rho | **dec rho** | cloud cos | **dec cos** | cloud ratio | **dec ratio** |
|---|---|---|---|---|---|---|---|---|
| cubic | 28 | 99.91% | +0.6115 | **+0.8688** | +0.7700 | **+0.9813** | 0.0183 | **0.9558** |
| cubic | 768 | 99.70% | +0.6115 | **+0.5253** | +0.7700 | **+0.8934** | 0.0183 | **1.0771** |
| ridge | 28 | 99.94% | +0.4119 | **+0.9823** | +0.9173 | **+0.9996** | 0.0185 | **0.9697** |
| ridge | 768 | 99.88% | +0.4119 | **+0.9745** | +0.9173 | **+0.9994** | 0.0185 | **0.9741** |

`cond(g)` 2.2–7.8 in every cell (the CAE's was `1.88e+08`).

**Spread-regime sweep** (`07_instrument_low_spread_control_run.py` →
`07_low_spread_control.jsonl`), ridge at `d=20, D=768`, true `‖H‖` spread tuned by phase/frequency:

| true spread | est spread | est/true | **rho** | recon |
|---|---|---|---|---|
| 2.095 | 1.394 | 0.665 | **+0.9166** | 99.92% |
| 15.786 | 25.665 | 1.626 | **+0.9691** | 99.85% |
| 24.410 | 26.224 | 1.074 | **+0.9807** | 99.76% |
| 35.517 | 36.940 | 1.040 | **+0.9866** | 99.84% |
| 34.274 | 41.292 | 1.205 | **+0.9745** | 99.88% |

**Two readings, both load-bearing.** (a) `rho` sits at **+0.92 to +0.99 across a 17× range of
true spread**, including PU's low regime — ordering, the axis D7-04 uses, is robust. (b) `est/true`
swings 0.665 → 1.626 → 1.074 non-monotonically, so **estimated spread is a noisy readout and must
not be quoted as a precise quantity** — including PU's own 1.495, which should be read as "low,
order 2" and nothing finer.

**Honest range for the paper:** `rho` between **+0.53 and +0.99** on contractible `d=20` surfaces
at `D=768`. Quote the range; `+0.97` alone invites a reviewer to find the `cubic@768` cell.

**Reconstruction is not what predicts rho.** `cubic@768` reconstructs at 99.70% and scores
`+0.5253`; `ridge@768` reconstructs at 99.88% and scores `+0.9745`. What separates them is the
surface's own curvature variation (`II` CV 0.104 vs 0.483), which for PU is unknown. **Do not
infer "high reconstruction ⇒ high rho."**

## 5. PU itself — what is measured

**Reconstruction is DIMENSION-limited, not noise-limited**
(`07_pu_latent_recon_sweep_run.py` → `07_pu_latent_recon_sweep.jsonl`), 300 epochs, holdout:

| latent d | recon | gain |
|---|---|---|
| 10 | 97.303% | — |
| 15 | 97.874% | +0.571% |
| 20 | 98.217% | +0.343% |
| 25 | 98.431% | +0.214% |
| 32 | 98.626% | +0.194% |
| 48 | 98.896% | **+0.270%** (gain rose) |

**No plateau anywhere through `d=48`.** Consequences: (a) the `d=20` fit truncates PU, so numbers
taken from it describe a truncated approximation — this is why D7-01 sweeps `d`; (b) total gain
`d=20 → d=48` is only **+0.68 points** for a 2.4× larger bottleneck, and increments do not decay
cleanly (0.214 → 0.194 → 0.270), so single-seed run variance is comparable to the effect — do not
over-read it either; (c) it may be **capacity**-limited rather than dimension-limited, since
hidden width was fixed at 250×3 throughout. Untested.

**The `d=20` PU fit** (`07_pu_plain_ae_fit_run.py` → `07_pu_plain_ae_fit.jsonl`), for reference:
`recon 98.207%` (converged, flat over 4 blocks), `cond(g)` median **17.57** / p95 30.53 / max
63.03, `‖H‖` median 36.41, p05 29.69, p95 44.39, spread 1.495.

**PU topology: `β₁ = 0`, measured** (`07_pu_betti_probe_run.py`, `07_pu_betti_n800_run.py`),
Fasy bootstrap band, `B=10`, `alpha=0.05`, `maxdim=1`, `D=768`, 3 draws each:

| cloud | `n=400` | `n=800` |
|---|---|---|
| positive control `S¹×R¹⁹` (truth 1) | **1, 1, 1** | **1, 1, 1** |
| negative control ball²⁰ (truth 0) | **0, 0, 0** | 0, 0 |
| **PU legacysurvey** | **0, 0, 0** | **0, 0, 0** |

Controls validate the instrument's power at both `n`. PU's longest H₁ bar (0.0969 at `n=400`) is
below even the smallest band observed (0.1041). One honest detail: the `n=800` draw 2 had max life
`0.1658` against band `0.1658` — exactly equal, failing the strict `>` rather than clearing it.

**Trivial topology confirms a Euclidean-latent plain autoencoder is structurally correct.** It
does NOT revive the Acosta et al. (2212.10414) template approach: their library tops out at
`T² = S¹×S¹` (d=2) against PU's intrinsic dimension of 18–25, and a contractible `β = (1,0,0)`
means `Z` is flat, so their analytic curvature reduces to the pullback `g = JᵀJ` already computed.

## 6. The Phase 4 cautionary record — read before designing D7-03

Phase 4's `HOLDS` is **not** evidence of a curvature–alignment association, and must not be cited
as one. Its own `04-FINDINGS.md` establishes:

- Split axis `spearman(density, signed_projection) = +0.8208` (n=9500, p≈0), against
  `spearman(density, ‖H‖) = −0.0273` — the confound is specific to the **direction** axis, not
  magnitude. Region density medians differ ~**5,735×**.
- Region sizes 6256 vs 3244; MKNN's chance floor is `k/n_region`. Ratio-over-chance:
  52.1/64.2 (k=5), 36.6/42.9 (k=10), 25.5/28.2 (k=20), **15.6/15.7 (k=50)** — the gap collapses to
  nothing as `k` grows. Phase 4 states region size "accounts for nearly the entire raw-score gap."
- Direction was validated only on **codimension-1** fixtures, where `H = H_scalar · n̂` so
  "direction" is just normal orientation. PU is codimension ~748.
- Phase 4's own words: "no known-answer anchor at any point in the chain — estimator, field, or
  partition."

**Encouraging for Phase 7:** the confound was specific to direction. Magnitude looks clean —
`−0.0273` (centroid field) and `+0.0300` (Phase 6 point-cloud field). Measure it on the new
decoder field rather than assuming it.

## 7. Cost model — plan around this

**The curvature computation dominates, not training.** Measured on the `d=20` PU fit:
`1457s` for the curvature field over 10,000 points at `D=768`, against `374s` for all 600
training epochs. It scales as `D·d²`:

| d | relative cost | est. field time |
|---|---|---|
| 20 | 1.0× | ~24 min |
| 25 | 1.6× | ~38 min |
| 32 | 2.6× | ~62 min |
| 48 | 5.8× | ~140 min |

D7-01's three-`d` sweep is therefore ~2 hours of field computation plus ~15 min of training.

**Operational note:** three concurrent torch jobs on this 20-core machine drove load to 44 and
cost roughly a 10× slowdown. Run heavy jobs **serially** with `OMP_NUM_THREADS` capped. Also:
`pgrep -f <script>.py` matches the invoking shell's own command line — kill by real PID or you
will leave orphaned 53-thread processes running.

## 8. What Phase 7 will NOT claim

- That the field measures true curvature. No ground truth for PU exists; the analytic validation
  gives a **range** (`+0.53` to `+0.99`), not a point estimate.
- That a null means no relationship exists, absent D7-02's power evidence.
- Anything about CKA (D7-07), or about MKNN at the source paper's `n=101,725` — this milestone
  works at `n=10,000`, where the `k/n` chance floor is ~10× higher (Phase 4's D4-19).
- Any reinterpretation of Phases 2, 02.x, 3, 03.1, 4, 5 or 6.

## 9. Artifacts

| path | what |
|---|---|
| `notebooks/diagnostics/07_instrument_fixture_sweep_run.py` | fixture × ambient-D validation (§4) |
| `notebooks/diagnostics/07_instrument_low_spread_control_run.py` | spread-regime sweep, floor check (§4) |
| `notebooks/diagnostics/07_instrument_noise_calibration_run.py` | fidelity vs reconstruction quality |
| `notebooks/diagnostics/07_pu_latent_recon_sweep_run.py` | latent-dimension sweep (§5) |
| `notebooks/diagnostics/07_pu_plain_ae_fit_run.py` | the PU fit + `cond(g)` + `‖H‖` stats (§5) |
| `notebooks/diagnostics/07_pu_betti_probe_run.py` / `07_pu_betti_n800_run.py` | `β₁` with controls (§5) |
| `notebooks/diagnostics/07_cae_divergence_probe_run.py` / `..._plain_arm_run.py` | why the CAE was abandoned |
| `notebooks/.cache/07_*.jsonl` | all results above (gitignored, per CLAUDE.md) |

**Why the CAE was abandoned** (`07_cae_divergence_*`), `d=20` saddle, identical data/split/budget:
ChartAE `n_charts=1` went 21.1% → 35.1% → 18.2% → 6.2% → −2.1% → **−7.2%** variance explained over
600 epochs; `n_charts=4` went 33.7% → **1.9%**; the plain AE climbed monotonically to **98.64%**.
The CAE **diverges** with training rather than underfitting — `synthetic_control_run.py` already
notes the PU protocol "is known to diverge on the flat fixture at this budget."
