# Findings — Classical-MDS Validity of an Isomap Fit on PU Embeddings

**Project:** EffDim / milestone v1.1 "PU Manifold Curvature"
**Phase:** 2 — Eigenspectrum Audit & Validity Gate
**Date:** 2026-07-31
**Status:** complete and self-contained; the phase itself is not yet sealed (§10).

Four experiments, written for a reviewer with no prior context.

---

## 1. Summary

- Full eigenspectrum of the double-centred geodesic matrix of an Isomap fit on 10,000 PU
  astronomy image embeddings: **~half the eigenvalues negative, carrying ~41% of absolute
  mass. Pre-registered gate: FAIL.**
- Pre-registered k-sensitivity re-fit (k ∈ {5,10,30} vs incumbent 15): negative mass
  flat-to-slightly-increasing across a 6× k range while co-diagnostics confirm the graph
  genuinely densified. **kNN hop inflation (H2) not supported.**
- Diagnostic triage: removing L2 normalization moves m by **0.28%**; local intrinsic
  dimension is **stable and tight (~20–25, std 2.0)** — a genuine manifold, not a
  structureless cloud. Both remaining alternatives closed.
- Replication: paired HSC column gives m = 0.4226 (+2.55%); a ~90% disjoint resample gives
  m = 0.411948 (**−0.03%**) with *identical* positive/negative counts. Stable
  population-level property.

**Consequence: classical MDS does not describe this geometry.** Surviving explanation: a
real, stable ~20–25 dimensional manifold whose geodesic metric is strongly non-Euclidean.
**The one variable never varied is the model** (DINOv3 ViT-B/16 throughout) — the largest
open question (§8). **Correction (§6.4): the frozen elbow dimension of 5 disagrees with
every other estimate (18, 19.5, 25) and should be treated as suspect.**

---

## 2. Data and fit provenance

| Item | Value |
|---|---|
| Dataset | `UniverseTBD/pu-embeddings`, config `legacysurvey_dinov3_vitb16` |
| Population | 101,725 rows |
| Subsample | 10,000 rows, `seed = 20260729` |
| Preprocessing | L2 normalization (`normalize = True`) |
| Ambient dim | 768 |
| Method | `sklearn.manifold.Isomap`, `eigen_solver="dense"` |
| `n_neighbors` (k*) | 15 |
| `n_components` | 18 |
| Versions | numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, Python 3.14.6 |
| `fit_key` | `43cf438bc944c509` |

k*=15 from a plateau-stability criterion over k ∈ {5,8,10,15,20,30} (widest all-passing run
[10,15,30], centre 15). n_components=18 = ceil(median of 8 geometric estimators, 17.183).
Raw pre-normalization norms 16.029 ± 0.504 (cv 3.1%) — already near-constant-norm; matters
for §6.2.

---

## 3. Experiment 1 — Full eigenspectrum audit

**Method.** Spectrum computed by hand — `kernel_pca_.eigenvalues_` is truncated to 18 and
structurally cannot show a negative tail. `dist_matrix_` (10,000², float64) mmap-loaded;
symmetry measured chunk-wise before any eigensolve (max deviation 1.421e-14, bound
2.132e-09); in-place mean-form double-centring verified against the literal `-0.5·J D² J`
to rtol=atol=1e-12 on two 50×50 inputs; split eigensolve (`eigvalsh` all values, `eigh`
subset top 40); leading 18 cross-checked against sklearn at rtol=1e-8 (worst 8.532e-15);
float64 end-to-end; strict comparison against zero; length asserted exactly 10,000.

**Gate.** `r = |λ_min_neg|/λ_max_pos`, `m = Σ|λ_neg|/Σ|λ|`; verdict = worse of the two;
strict less-than throughout; classifier asserted on synthetic boundary cases (incl.
(0.10,0)→MARGINAL, (0.25,0)→FAIL, (0.05,0.20)→FAIL) before real data.

| | PASS | MARGINAL | else |
|---|---|---|---|
| `r` | < 0.10 | < 0.25 | FAIL |
| `m` | < 0.05 | < 0.15 | FAIL |

**Results.**

| Quantity | Value |
|---|---|
| Eigenvalues total | 10,000 |
| Strictly positive | 4,971 |
| Strictly negative | **5,029** |
| `λ_max_pos` | 3.230854e+03 |
| `λ_min_neg` | −1.693588e+02 |
| Noise floor (`n·eps·λ_max_pos`) | 7.173937e−09 |
| **`r`** | **0.052419** — passes (< 0.10) |
| **`m`** | **0.412071** — **fails even the 0.15 MARGINAL bound** |
| **Verdict** | **FAIL** |
| Steep-dropoff index / log-ratio | 2 / 2.4447 |

`|λ_min_neg|` sits ~10 orders above the float64 noise floor — real structure, not rounding.
The shape is the finding: not one short-circuit outlier but 5,029 negatives, none dominant,
collectively 41% of mass. `r` alone would have read this spectrum as clean. Independent
recompute from the persisted npz reproduces r=0.052419, m=0.412071 exactly.

---

## 4. Experiment 2 — Pre-registered k-sensitivity re-fit

**Hypotheses.** H1 intrinsic curvature (no k removes the tail) vs H2 kNN hop inflation
(densifying shrinks it) — opposite predictions for m(k).

**Confound controlled.** Larger k reduces hop inflation AND increases short-circuiting; both
lower m. Two descriptive co-diagnostics (no thresholds — k=15 values were already known)
over an identical pair sample at every k:
`GEO_AMBIENT_RATIO(k)` = median(geodesic/ambient distance), collapses toward 1 under
short-circuiting; `LONG_EDGE_FRACTION(k)` = fraction of edges beyond the k=15 p99.

**Pre-registration.** Design, k set {5,10,30}, interpretation rule, and threshold-revision
prohibition committed as `02-REFIT-PREREGISTRATION.md` (`057b084`) **before any re-fit
ran**. k=8,20 excluded (adding after a FAIL = widening the search in response to a result).
Same subsample/seed/rows/n_components/solver; only n_neighbors varies. n_components does not
affect r/m (full-spectrum statistics).

**Results — all four k, reported regardless of outcome.**

| k | `r(k)` | `m(k)` | positive | negative | `GEO_AMBIENT_RATIO` | `LONG_EDGE_FRACTION` | Verdict |
|---|---|---|---|---|---|---|---|
| 5 | 0.060312 | 0.406433 | 4972 | 5028 | 2.828727 | 0.006540 | FAIL |
| 10 | 0.058311 | 0.410187 | 4971 | 5029 | 2.320592 | 0.008620 | FAIL |
| **15** *(incumbent)* | 0.052419 | 0.412071 | 4971 | 5029 | 2.117401 | 0.010000 | FAIL |
| 30 | 0.050708 | 0.415735 | 4963 | 5037 | 1.864727 | 0.013923 | FAIL |

| k | `λ_max_pos` | `λ_min_neg` | noise floor | kNN edges | edge p99 | median geodesic |
|---|---|---|---|---|---|---|
| 5 | 5.432086e+03 | −3.276213e+02 | 1.206e−08 | 50,000 | 0.487021 | 1.593138 |
| 10 | 3.798254e+03 | −2.214809e+02 | 8.434e−09 | 100,000 | 0.504292 | 1.307802 |
| 15 | 3.230854e+03 | −1.693588e+02 | 7.174e−09 | 150,000 | 0.516666 | 1.192894 |
| 30 | 2.528065e+03 | −1.281927e+02 | 5.613e−09 | 300,000 | 0.539894 | 1.050865 |

`LONG_EDGE_FRACTION(15)=0.010000` holds by construction. All four graphs re-verified as one
connected component.

**Reading.** No k comes close (best m = 0.406 at k=5, still 2.7× the MARGINAL bound). m(k)
rises slightly and monotonically (spread 0.0093 over 6× k). Densification demonstrably
worked (`GEO_AMBIENT_RATIO` 2.83→1.86, `LONG_EDGE_FRACTION` 0.0065→0.0139) and bought no
reduction. A **controlled negative**: the rescue mechanism was actively engaged and produced
nothing. **H2 not supported.**

**Validity checks.** Reconstructed cfg reproduces fit_key 43cf438bc944c509; incumbent r/m
reproduce published values; the 200,000-pair sample re-drawn bit-identical to the cached
one; each spectrum's top 18 agree with sklearn to rtol=1e-8 (worst 5.6e-15); each array
exactly (10000,) float64; every |λ_min_neg| 10–11 orders above its noise floor.
**Cost.** Peak RSS 3.48 GiB; fits 78.5/87.9/104.5 s; eigensolves 122.9/120.4/122.8 s;
cache 6.4 GiB.

---

## 5. Corroborating evidence — residual-variance analysis

Deterministic max-curvature elbow, swept d = 1..40, ties to lower d:

| Quantity | Value |
|---|---|
| Elbow, Tenenbaum curve (first draw) | 5 |
| Elbow, second **disjoint** draw | 5 — exact agreement |
| Elbow, eigenvalue cross-check curve | 8 |
| **Max divergence between curves** | **0.697664**, at d=5 |

The divergence is the corroboration: both curves bounded in [0,1] and they disagree by 70
percentage points at the elbow — the eigenvalue curve normalizes by *positive* mass only, so
with 41% of mass negative it over-reports variance captured. The elbow of 5 sits far below
the ~18–25 estimates; §6.4 revises the reading. d=5 was frozen for record-keeping so the
verdict artifact is self-contained — **not** an endorsement of 5 as a working dimension.

---

## 6. Experiment 3 — Diagnostic triage

**Status: not pre-registered.** Post-hoc hypothesis-narrowing; identical r/m definitions and
bounds. Script (`gate_diagnostics.py`, since removed — see §10) first reproduced published
r/m from cache exactly.

### 6.2 L2 normalization?

Norms are cached, so normalization is exactly invertible. Refit on raw vectors, all else
identical:

| | normalized (published) | unnormalized |
|---|---|---|
| `r` | 0.052419 | 0.054843 |
| `m` | **0.412071** | **0.413239** |
| positive / negative | 4971 / 5029 | 4973 / 5027 |
| `λ_max_pos` | 3.230854e+03 | 8.475097e+05 |
| `λ_min_neg` | −1.693588e+02 | −4.648015e+04 |
| Verdict | FAIL | FAIL |

m moves **+0.001167 (+0.28%)**; `|λ_min_neg|` 2.47e+10 above the unnormalized noise floor. **Not the
cause.** Limit: raw norms cv 3.1% — the data already sat on a thin shell, which is *why*
the result is null. This closes "normalization introduced the curvature", NOT "shell
geometry contributes" (untested, different probe needed).

### 6.3 Is it a manifold?

TwoNN (Facco et al.) + local PCA (90% variance, k=60, 600 centres), both spaces:

| | normalized | unnormalized |
|---|---|---|
| TwoNN | 19.48 | 19.43 |
| Local PCA median | 25.0 | 25.0 |
| mean ± std | 24.52 ± 2.01 | 24.57 ± 1.98 |
| 5/25/50/75/95 pct | 21/23/25/26/28 | 21/23/25/26/27 |
| range | 19–29 | 19–29 |
| neighbourhoods > 18 dims | 100% | 100% |
| neighbourhoods > 40 dims | 0% | 0% |

Stable and tightly concentrated — the signature of a genuine manifold of roughly constant
dimension. **The manifold assumption holds.**

### 6.4 The frozen dimension is the outlier

| Estimate | Value |
|---|---|
| Local PCA 90% | ~25 |
| TwoNN | ~19.5 |
| Median of 8 geometric estimators | 18 |
| `n_components` | 18 |
| **Residual-curve elbow → frozen** | **5** |

With 41% of mass negative, the Tenenbaum curve saturates early because flat embedding fails
at every dimension — the elbow measured the *failure*, not the geometry (consistent with
CURVE_DIVERGENCE_MAX=0.698). Consequences: **d=5 is suspect, do not inherit**; and
n_components=18 sits *below* the intrinsic dimension — every fit was dimension-starved
(100% of neighbourhoods need >18 dims). Neither changes r/m.

---

## 7. Experiment 4 — Replication across survey column and disjoint sample

**Status: not pre-registered** (diagnostic, same standing as Exp 3).

### 7.2 Cross-survey — paired HSC column

Same objects row-for-row, same model, different imagery; identical settings:

| | `legacysurvey` (published) | `hsc` |
|---|---|---|
| `r` | 0.052419 | 0.062512 |
| `m` | **0.412071** | **0.422582** |
| positive / negative | 4971 / 5029 | 4965 / 5035 |
| `λ_max_pos` | 3.230854e+03 | 2.863319e+03 |
| `λ_min_neg` | −1.693588e+02 | −1.789906e+02 |
| Verdict | FAIL | FAIL |
| TwoNN | 19.48 | 17.56 |
| Local PCA median (std) | 25.0 (2.01) | 22.0 (2.57) |

m +2.55%, same signature; negative tail 2.8e+10 above the noise floor. **Not column-specific** — but a *weak* replication: the columns
are ~0.85 cosine-aligned (mean |hsc−ls| = 0.0156; alignment check s_true=0.8428).

### 7.3 Disjoint sample — new seed

10,000 fresh rows, `seed = 20260801`; overlap 1,002 rows (10.02%, chance expectation 9.83%):

| | seed 20260729 (published) | seed 20260801 |
|---|---|---|
| `r` | 0.052419 | 0.048304 |
| `m` | **0.412071** | **0.411948** |
| positive / negative | 4971 / 5029 | **4971 / 5029** |
| `λ_max_pos` | 3.230854e+03 | 3.203943e+03 |
| `λ_min_neg` | −1.693588e+02 | −1.547647e+02 |
| Verdict | FAIL | FAIL |
| TwoNN | 19.48 | 19.98 |
| Local PCA mean ± std | 24.52 ± 2.01 | 24.54 ± 2.02 |

**m differs by −0.03% with identical positive/negative counts on ~90% different objects.**
Closes sampling variance and object-set specificity at once. Also exercised the
pre-registration's remediation option 2.

---

## 8. Interpretation

**Established:** (1) negative mass real — 10–11 orders above noise floor at every k;
(2) large — ~50.3% of eigenvalues, ~41% of mass; (3) diffuse, not outlier-driven — r passes
at every k (max 0.060); (4) not neighbourhood-scale — flat across 6× k with confirmed
densification; (5) not L2 normalization — 0.28% move; (6) the cloud IS a manifold, ~20–25,
std 2.0; (7) not column-specific — HSC m=0.4226; (8) not sample-specific — disjoint draw
m=0.411948, identical counts; (9) therefore **classical MDS is not a valid description of
this geodesic metric**.

**Surviving explanation:** a genuine, stable ~20–25 dimensional manifold with strongly
non-Euclidean geodesic metric. Surviving, not positively demonstrated — alternatives were
eliminated; curvature was never directly measured.

**Untested:** the **model** (DINOv3 ViT-B/16 in every fit — the single largest open
question; other configs are embarrassingly parallel); **shell geometry** (cv 3.1% raw-norm
shell; needs a different probe); **estimator bias** (both §6.3 estimators
neighbourhood-based).

**On the thresholds:** project-specific tolerance, not a universal standard. Without appeal
to the bounds: at 41% of absolute mass the negative part is comparable in magnitude to the
positive part, not a correction to a Euclidean picture.

---

## 9. Limitations

- Sampling variance: tested and closed (§7.3, −0.03%).
- One model architecture across every fit; generalization untested (highest-value next
  compute).
- Three re-fit k values by deliberate pre-registration; k=8,20 excluded.
- Co-diagnostics descriptive by design (no thresholds).
- Experiments 3 and 4 not pre-registered; hypothesis-narrowing weight only.
- HSC replication weak (~0.85 aligned); disjoint-seed is the stronger.
- n_components=18 below the intrinsic dimension — fits dimension-starved (doesn't affect
  r/m).
- Curvature never directly measured.
- Documentation discrepancy: one summary says "nine synthetic boundary cases"; the
  implemented list has eight. Classifier exercised either way; no measured value depends on
  it.

---

## 10. Status and reproduction

Verdict artifact written; phase **not yet formally sealed** (human-verification step open).
A documented FAIL is a complete outcome. Remediations enumerated in the artifact: re-fit at
different k (closed by Exp 2), resample with new seed (exercised by §7.3), accept the
documented FAIL.

**Artifacts.** *(Note: a 2026-08-01 repo cleanup removed `01_manifold_and_gate.ipynb`,
`gate_diagnostics.py`, `hsc_crosscheck.py`, and `model_sweep.py`; they remain in git
history. `02_k_sensitivity_refit.ipynb` was made standalone and now fits k=15 itself;
`seed_crosscheck.py` survives.)*

| Path | Contents |
|---|---|
| `notebooks/01_manifold_and_gate.ipynb` §6.0–§6.9 *(removed)* | Experiment 1, residual/elbow, verdict |
| `notebooks/02_k_sensitivity_refit.ipynb` | Experiment 2 |
| `notebooks/diagnostics/gate_diagnostics.py` *(removed)* | Experiment 3 |
| `notebooks/diagnostics/hsc_crosscheck.py` *(removed)* | Experiment 4, cross-survey |
| `notebooks/diagnostics/seed_crosscheck.py` | Experiment 4, disjoint sample |
| `02-REFIT-PREREGISTRATION.md` | Experiment 2's pre-registration + outcome |
| `notebooks/.cache/gate_verdict_43cf438bc944c509.json` | Machine-readable verdict |
| `notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz` | Full spectrum |
| `notebooks/.cache/mds_residuals_43cf438bc944c509.npz` | Residual curves |

Headline statistics check independently of any notebook:

```python
import numpy as np
z = np.load("notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz")
ev = z["eigvals_all"]                      # (10000,) float64
neg, pos = ev[ev < 0], ev[ev > 0]
r = abs(neg.min()) / pos.max()             # 0.052419
m = np.abs(neg).sum() / np.abs(ev).sum()   # 0.412071
```

**Commit trail.**

| Commit | Contents |
|---|---|
| `3401c0c`, `108486e` | Experiment 1 |
| `057b084` | Experiment 2 pre-registration — **before any re-fit ran** |
| `6dcefba`, `380122e`, `f30a882` | Experiment 2, k = 5 / 10 / 30 |
| `c9f4ea7`, `6844624` | Comparison table, interpretation rule, outcome |
| `5cf9a19`, `539dafa` | Residual curves, elbow, frozen dimension |
| `aea04ff`, `a2ca11f` | Verdict artifact, downstream enforcement |
| `9c6e2b5` | Experiment 3 diagnostic script |
| `18bbaf4` | Experiment 4 replication scripts |
