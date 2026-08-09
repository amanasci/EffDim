# Is the sparse fringe of the embedding manifold more curved?

**Run:** `outputs/run_20260809_212729/density_curvature/` · 2026-08-09
**Data:** `UniverseTBD/pu-embeddings` physics, `vit_base` (k_t=22) and `dinov3_vitb16` (k_t=21), 768-d L2-normalised, n_test=4916
**Code:** `curvature_core.py`, `density_stats.py`, `density_curvature_probe.py`, `stage_b_connectback.py` · gates in `tests/test_curvature_core.py`

---

## TL;DR

- **Sparse regions are more curved — for vit_base.** `kappa_ratio` rises monotonically 1.58 → 1.92 from densest to sparsest quartile (rank-biserial +0.590, ρ = +0.397), ~2.8× the largest control, stable across three scales. **For dinov3 the effect is inconclusive**: ρ = +0.196 against a flat-surrogate control of +0.198.
- **Every naive estimator gives the opposite answer.** On the real data `kappa_naive_ratio` says ρ = −0.643 ("dense more curved"); the calibrated metric says +0.397. On a manifold that is *exactly flat*, the naive metrics reach ρ = −0.99. Without the per-point permutation null this experiment would have concluded the reverse, confidently.
- **But curvature does not explain the SAE or probe failures.** ρ(kappa_ratio, SAE reconstruction error) = +0.309 collapses to **−0.172** once density is partialled out; against cleaned probe error, +0.118 → −0.032. Density, not bending, is the operative variable.
- **What does survive is `rf_k`** — the local PCA residual fraction — at partial ρ = +0.368 / +0.517 against SAE error and +0.139 / +0.295 against probe error. So the predictor of failure is *how much of a neighbourhood escapes its own tangent subspace*, not extrinsic curvature.
- **The probe-error target was substantially an artifact.** Restoring a dropped intercept cut `mean_residual_all` from mean 19.42 to 1.83, and the legacy target correlates +0.434 with mere label availability — for dinov3, the original headline correlation flips sign (+0.170 → −0.091) once that confound is removed.

---

## 1. Motivation

The previous run ([`run_20260802_222657`](outputs/run_20260802_222657/results.md)) found that SAE reconstruction error correlates with physics-probe error globally (mean |ρ| = 0.175) but that this collapses inside the densest quartile (0.077). The natural reading: a sparse "fringe" drives both failures at once.

That invites a geometric explanation — if the fringe is where the manifold **bends**, a locally-linear probe and a dictionary-based SAE would both degrade there for the same reason. Unlike raw density, curvature is something a model could be designed around.

This experiment measures local curvature per point, stratifies by local density, and then asks whether curvature actually accounts for the failures.

## 2. What κ is, and why the naive estimators fail

### 2.1 The quantity

Curvature is **1/radius**. A sphere of radius *R* has κ = 1/R: a big sphere is nearly flat, a small one bends hard. Units are 1/length. The estimator is anchored to exactly this — on a sphere of radius *R* it returns 1/R (verified to 3.5%, §4).

### 2.2 What is observable at a point

Take a point, take its *K* nearest neighbours, run local PCA. That splits each neighbour's displacement into two parts:

- **u** — the *tangential* part: movement within the flat sheet the manifold locally resembles
- **r** — the *normal* part: how far it lifts off that sheet

On a flat manifold r = 0 apart from noise. On a curved one, moving tangentially by ‖u‖ lifts you off by

```
‖r‖  ≈  ½ · κ · ‖u‖²
```

— the parabola `z = x²/2R` approximating a circle of radius R. Rearranged, **κ = 2‖r‖/‖u‖²**. Every metric below is a different way of extracting κ from this relation, or of failing to.

### 2.3 The metric ladder

**`rf_k`** — local PCA residual fraction: the share of local variance not captured by the top-`k_t` directions. Dimensionless, and it conflates three unrelated things — real bending, noise thickness, and local dimensionality. It is not a curvature at all, but "how much of this neighbourhood fails to lie flat."

**`kappa_naive`** = `median 2‖r‖/‖u‖²` — the formula applied literally. This is the one that looks correct and is not. Real manifolds have thickness σ, so even at zero curvature ‖r‖ ≈ σ, and

```
κ̂  ≈  2σ / ‖u‖²  ≈  2σ / R²
```

where R is the neighbourhood radius. Dense region → small R → **huge** κ̂. It is measuring the noise floor divided by neighbourhood size, i.e. a proxy for 1/R², i.e. a proxy for density.

**`kappa_slope`** — the obvious repair: regress ‖r‖ on ½‖u‖² *with an intercept*, letting the intercept absorb σ and reading κ off the slope. Better reasoning, still broken — and note the sign flips. It uses only magnitudes and discards direction.

**`kappa_jet`** — the real estimator. Instead of comparing magnitudes, fit the quadratic surface itself: regress the normal coordinates on `[1 | u | quadratic monomials of u]`. Those quadratic coefficients *are* the second fundamental form.

The gain over `kappa_naive` is that it uses **direction**. Genuine bending is systematic — stepping +u and −u lifts you the *same* way, which is what "quadratic" means — whereas noise does not. A magnitude ratio cannot separate those; a quadratic fit can.

But `kappa_jet` is still confounded, because **fitting a quadratic to pure noise still returns something nonzero**, and how much depends on the noise level and neighbourhood scale — both of which track density.

Measured on a synthetic manifold that is **exactly flat** (768-d ambient, 19-d subspace, lognormal density tilt, true κ = 0), all four are essentially pure density readouts:

| estimator | ρ(d_k, metric) on FLAT data |
|---|---|
| `rf_k` | **−0.999** |
| `kappa_jet` | **−0.996** |
| `kappa_naive` | **−0.973** |
| `kappa_slope` | **+0.999** |
| *`kappa_ratio` (calibrated, §3.1)* | *−0.109* |

Any of the first four would have given a confident, large, entirely artifactual answer — and note the sign disagreement, so the choice of estimator alone decides the direction of the conclusion. The calibrated metric in the last row is the subject of §3.1.

ε-balls don't rescue it. Neighbour count scales as K·(ε/d_K)^d, so at k_t ≈ 22 with the measured Q4/Q1 density ratio of 1.63, an ε giving a dense point 50 neighbours leaves a sparse point **0.001**. The same curse rules out density-matched subsampling.

## 3. Method

### 3.1 The fix: a per-point permutation null

Ask the question that actually matters: *how large would this quadratic fit be if there were no relationship between tangent position and normal displacement?* Answer it empirically, at each point:

1. Strip the constant and linear terms
2. **Permute which neighbour each normal residual belongs to**
3. Refit the *identical* quadratic design
4. Repeat 16× and average → `kappa_null`

```
kappa_ratio  =  kappa_jet / kappa_null
```

— "how many times larger is the observed bending than chance bending *here*."

**Why this cancels the confound.** Everything that varies with density — noise magnitude, neighbourhood radius, tangent spread, neighbour count — enters numerator and denominator identically. The permutation leaves the design matrix and the residual magnitudes exactly as they were; it destroys only the **pairing**. The density-dependent scale divides out, and what remains is purely whether systematic structure is present.

That is the essential move: rather than normalising by a theoretical scale (which would require knowing σ and R per point, both density-dependent and unmeasurable), the chance level is *measured separately at every point*. It is a per-point permutation test, not a per-point ratio.

### 3.2 Implementation

One eigendecomposition of the (*K*,*K*) neighbour Gram matrix yields the tangent basis, the normal coordinates and the PCA residual fraction together — ~3.7× faster than an SVD of the (*K*,768) displacement block and numerically identical. The quadratic coefficients are converted to a second-fundamental-form Hessian H, giving `kappa_jet = ‖H‖_F/√p_quad`, normalised so a sphere of radius *R* reads exactly 1/R.

**Three settings that matter, all fixed empirically:**
- **`p_lin` ≥ intrinsic dimension.** Allow too few tangent directions and the leftover real tangent directions are counted as normal and read as curvature — at `p_lin`=5 against true d=8, a *flat* manifold reported median κ = 2.15. So `p_lin` = Two-NN `k_t`.
- **`p_quad` = 3.** Curvature is measured only in the top 3 tangent directions — a sectional-curvature proxy. Acceptable because the null uses the identical design, so the ratio stays calibrated; larger `p_quad` needs far more neighbours.
- **K ≳ 6× design columns**, else the fit and its null both saturate and the null absorbs real signal. At k_t=22, p_quad=3 → 29 columns → K ≥ ~175. **This puts the K=50 scale of the density proxy out of reach for the jet fit** — `d_k` defines density only.

### 3.3 Why the flat surrogates are still needed

`kappa_ratio` is calibrated but not perfect: its own residual density bias is about ±0.10 (§4). So a real number is only interpretable beside the same number computed on data known to be flat while carrying the real density structure. That is what makes vit_base's +0.397 a result and dinov3's +0.196 not one.

The four controls: **`flat_gauss`** and **`flat_shuffle`** take the real embeddings, project them onto their global top-`k_t` PCA subspace and rebuild them there — exactly flat, but carrying the **real density structure** — adding matched Gaussian noise or row-permuted real residuals respectively. **`synthetic flat`** is the method's own bias floor. **`synthetic sphere`** has constant curvature with varying density: the strong test, since a good estimator must both detect the curvature and report it as constant across density quartiles.

### 3.4 Statistics

Quartiles of `d_k` (K=50), Q1 = densest; medians with 2000-sample bootstrap CIs; Mann-Whitney U for Q4 vs Q1 with rank-biserial *r*; Spearman ρ vs `d_k`; partial Spearman for the connect-back. Scales K ∈ {200, 300, 400}, `p_quad`=3, `m_norm`=5, `n_perm`=16, seed 0.

### 3.5 Probe-target correction

Only **8 of 38** probes reach `r2_cv > 0.1`, and those 8 are exactly the ones with ~93–100% label coverage; the other 30 have `n_valid` ≈ 725–1187 against 768 features, so the fit interpolates (`elpetro_theta`: r2_cv = −11117). A `nanmean` over all 38 partly encodes *whether a galaxy has an NSA/MaNGA cross-match*. Two bugs in `_common.compute_probe_residuals` were also fixed in the Stage B reimplementation (`_common` left untouched so the earlier run stays reproducible): the intercept was dropped, valid only if `Z_train` were mean-centred; and test targets were re-standardised with test-split statistics against train-fitted weights.

## 4. Validation

All 8 gates in `tests/test_curvature_core.py` pass. Sphere R=1 → `kappa_jet` 1.034 (**3.4% error**); R=3 → 0.345 (3.5%); plane → `kappa_ratio` **1.020**; `rf_k` matches the existing `residual_fraction` to 0.00e+00.

On the synthetic controls at K=200:

| control | true κ | metric | median | ρ(d_k) |
|---|---|---|---|---|
| **flat** | 0.00 | **kappa_ratio** | 1.062 | **−0.109** |
| flat | 0.00 | kappa_jet / rf_k / kappa_naive | — | −0.996 / −0.999 / −0.973 |
| **sphere** | 0.30 | **kappa_ratio** | **1.289** | **−0.065** |
| sphere | 0.30 | rf_k / kappa_naive | — | −0.525 / −0.849 |

On a manifold of *exactly constant* curvature with varying density, `kappa_ratio` detects the curvature and reports it as constant (ρ = −0.065); the naive metrics claim ρ ≈ −0.85, i.e. "dense regions are far more curved."

## 5. Findings

### 5.1 The confound is real and large here — and flips the answer

At K=200 on vit_base the median neighbourhood radius grows **39%** from densest to sparsest quartile (`R_med` 0.279 → 0.386, ρ = **+0.857**), with fitted local thickness tracking it (`noise_floor` ρ = +0.797). Uncalibrated estimators measure this and nothing else:

| metric | Q1 densest | Q4 sparsest | ρ(d_k) | implied verdict |
|---|---|---|---|---|
| `kappa_naive_ratio` | 4.220 | 3.277 | **−0.643** | "dense more curved" |
| `kappa_jet` | 1.787 | 1.646 | −0.162 | "dense more curved" |
| **`kappa_ratio`** | 1.580 | 1.924 | **+0.397** | **"sparse more curved"** |

Without the calibration this experiment would have concluded the opposite, confidently.

### 5.2 Sparse regions are more curved — clearly for vit_base, inconclusively for dinov3

**vit_base, K=200:**

| series | Q1 | Q2 | Q3 | Q4 | rank-biserial | ρ(d_k) |
|---|---|---|---|---|---|---|
| **Real** | 1.580 | 1.643 | 1.733 | **1.924** | **+0.590** | **+0.397** [.371, .420] |
| Flat surrogate (gauss) | 1.116 | 1.153 | 1.158 | 1.111 | −0.029 | −0.012 |
| Flat surrogate (shuffle) | 1.172 | 1.198 | 1.229 | 1.250 | +0.216 | +0.142 |
| Method bias floor (synthetic flat) | 1.123 | 1.066 | 1.058 | 1.057 | −0.186 | −0.109 |

Monotone, stable across scales (ρ = +0.397 / +0.361 / +0.326 at K = 200/300/400), ~2.8× the largest control.

**dinov3:** ρ = +0.196, but `flat_shuffle` gives +0.198 — the same number, at every scale (real +0.196/+0.192/+0.179 vs shuffle +0.198/+0.169/+0.198). **Not separable from the control**; inconclusive, not null.

**Qualification:** `kappa_ratio` is 1.6–2.6 *everywhere* on real data against ~1.0 on a flat manifold. The manifold is substantially curved at all densities — this is a gradient on a globally curved manifold, not a flat core with a curved fringe.

### 5.3 Curvature does not explain the failures

The direct answer to the motivating question, and it is negative (vit_base, K=200):

| curvature | target | ρ | **partial ρ given d_k** |
|---|---|---|---|
| `kappa_ratio` | SAE reconstruction error | +0.309 | **−0.172** (flips) |
| `kappa_ratio` | probe error (cleaned) | +0.118 | **−0.032** |
| `kappa_ratio` | `redshift` residual alone | +0.043 | +0.006 |

The raw association is entirely mediated by density. **Density, not bending, is the operative variable.**

### 5.4 What does survive: local off-subspace variance

`rf_k` retains its association after density is partialled out, in both models at every scale:

| | ρ (vit / dino) | **partial ρ given d_k** |
|---|---|---|
| vs SAE reconstruction error | +0.390 / +0.448 | **+0.368 / +0.517** |
| vs probe error, cleaned | +0.226 / +0.350 | **+0.139 / +0.295** |

Not a contradiction of §2: `rf_k` is uninterpretable as *curvature* because its density trend is artifactual, but partialling on `d_k` removes exactly that artifact. (On real data `rf_k` rises with sparsity, 0.213 → 0.234, while on the flat surrogate it falls steeply, 0.507 → 0.354 — the real trend runs opposite to the artifact.)

So what predicts both failures is **how much of a neighbourhood escapes its own tangent subspace** — local effective dimensionality and spread — not extrinsic bending. Only the calibrated estimator makes these separable.

### 5.5 The probe-error target was substantially an artifact

Restoring the intercept and using train-split standardisation dropped `mean_residual_all` from mean **19.42 to 1.83** (median 0.303) — most of that number was a missing constant offset, not probe difficulty. ρ(`mean_residual_all`, `n_valid_probes`) = **+0.434**. Checking the earlier run directly: for dinov3, ρ(reconstruction_error, probe error) = +0.170 overall but **−0.091 within the clean mode**, against +0.275 for the bare "has rare labels" indicator. Part of the §1 finding that motivated this work was a selection effect, not a difficulty effect.

## 6. Limitations

- **dinov3 is inconclusive** and should not be reported in either direction.
- **`flat_shuffle` is the binding control** and behaves unexpectedly (ρ ≈ +0.15–0.20 throughout, where the Gaussian surrogate gives ≈ 0) — heavy-tailed residuals appear to partly defeat the permutation calibration. vit_base clears it by ~2.8×; dinov3 does not.
- **K=50 is unreachable** for the jet fit at k_t=22, so persistence at small scales is untested.
- **The null assumes** normal residuals are exchangeable after linear removal; direction-dependent heteroscedastic thickness would misspecify it — likely the source of the `flat_shuffle` behaviour.
- **`p_quad`=3 is a sectional-curvature proxy** in the top-3 local-variance tangent directions, themselves mildly density-dependent.
- **Flat surrogates bound the artifact under the null only**, not residual scale sensitivity under a curved alternative.
- **The 8 surviving probes are redundant** (4 merger-family, 3 smooth/featured-family, redshift), so `mean_residual_good` is effectively a 3-concept average; `redshift` alone is reported separately.

## 7. Reproducing

```bash
# Full run (~15 min, no GPU; reuses existing SAE outputs rather than retraining)
python experiments/physics-probe-subspace/density_curvature_probe.py \
    --max-n 16384 --k-ladder 200,300,400 --n-perm 16 --n-boot 2000 \
    --nulls flat_gauss,flat_shuffle,synthetic --synthetic-n 4000 \
    --probe-r2-min 0.1 \
    --sae-npz-dir experiments/physics-probe-subspace/outputs/run_20260802_222657 \
    --output-dir experiments/physics-probe-subspace/outputs/run_<TS>/density_curvature

python experiments/physics-probe-subspace/density_curvature_probe.py --skip-probes ...  # Stage A only
pytest tests/test_curvature_core.py -q                                                  # gates
python experiments/physics-probe-subspace/render_report.py <run_dir>                    # rebuild results.md
```

BLAS threads are pinned to 1 before numpy is imported — the per-point matrices are small enough that multithreaded BLAS costs ~20× through oversubscription (observed load average 291); parallelism is taken over points via `--n-jobs`. Labels are now cached to `data_hf/physics/labels_test_n<N>.npz` instead of re-streamed each run.

## 8. Next steps

1. **Chase `rf_k`, not curvature** — it is what survives density control against both targets. Decompose it (local participation ratio, local intrinsic dimension, spectral tail mass) to identify which aspect of local spread hurts the SAE.
2. **Resolve dinov3** — more samples (`--max-n 32768`; the parquets hold 86471 rows), or a `--nulls rotation` mode (random tangent-plane rotation of **u**, preserving ‖u‖ and the radial ‖r‖–‖u‖ relation while destroying directional quadratic structure) to test whether `flat_shuffle` is misspecified.
3. **Re-run the original dense-region experiment with the corrected target** — §5.5 implies the "correlation vanishes in the dense quartile" headline may itself be partly a composition effect. Deliberately out of scope here.
