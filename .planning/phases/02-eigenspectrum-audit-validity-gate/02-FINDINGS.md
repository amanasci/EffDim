# Findings — Classical-MDS Validity of an Isomap Fit on PU Embeddings

**Project:** EffDim / milestone v1.1 "PU Manifold Curvature"
**Phase:** 2 — Eigenspectrum Audit & Validity Gate
**Date:** 2026-07-31
**Status of this document:** complete and self-contained. The phase it reports on is **not yet
sealed** — see §9.

This document reports three experiments. It is written for a reviewer with no prior context on
the project. Everything needed to check the claims — data provenance, method, measured values,
cache keys, and reproduction commands — is included or pointed at explicitly.

---

## 1. Summary

An Isomap fit on 10,000 PU (astronomy) image embeddings was audited for classical-MDS validity
by computing the full eigenspectrum of its double-centred geodesic distance matrix. **Roughly
half the eigenvalues are negative and they carry ~41% of total absolute eigenvalue mass.** The
project's pre-registered validity gate returns **FAIL**.

A second, separately pre-registered experiment tested whether this was an artifact of the
neighbourhood-graph scale by re-fitting at three other values of `k`. **It is not.** Negative
mass is flat-to-slightly-increasing across a 6× range of `k`, while co-diagnostics confirm the
graph genuinely densified over that range. The kNN hop-inflation explanation is not supported.

A third, diagnostic experiment tested the two remaining alternative explanations. Neither
survives: removing L2 normalization moves `m` by **0.28%**, and the cloud's local intrinsic
dimension is **stable and tight** (~20–25, std 2.0), which is what a genuine manifold looks
like rather than a structureless cloud.

The practical consequence: **classical MDS does not describe this geometry**, and the downstream
work that assumed a valid flat embedding cannot proceed on that assumption. The surviving
explanation is a real, stable ~20–25 dimensional manifold whose geodesic metric is strongly
non-Euclidean.

**A correction this raises (§6.4):** the residual-curve elbow of 5, which was frozen as the
project's working dimension, disagrees with every other dimension estimate available (18, 19.5,
25). It should be treated as suspect.

---

## 2. Data and fit provenance

| Item | Value |
|---|---|
| Dataset | `UniverseTBD/pu-embeddings`, config `legacysurvey_dinov3_vitb16` |
| Population | 101,725 rows |
| Subsample | 10,000 rows, `seed = 20260729` |
| Preprocessing | L2 normalization (`normalize = True`) |
| Embedding dim (ambient) | 768 |
| Manifold method | `sklearn.manifold.Isomap`, `eigen_solver="dense"` |
| `n_neighbors` (k\*) | 15 |
| `n_components` | 18 |
| Library versions | numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, Python 3.14.6 |
| `fit_key` | `43cf438bc944c509` |

`k* = 15` was selected in a prior phase by a plateau-stability criterion over
`k ∈ {5, 8, 10, 15, 20, 30}` (all six connected; the widest run in which every adjacent pair
passed three stability metrics was `[10, 15, 30]`, centre 15). `n_components = 18` came from the
ceiling of the median over eight geometric intrinsic-dimension estimators (17.183 → 18).

**Note for interpretation (§6, §7):** rows are L2-normalized, so the points lie on the unit
hypersphere in R^768. The raw pre-normalization norms are tightly concentrated
(16.029 ± 0.504, cv = 3.1%) — the embeddings were already near-constant-norm before
normalization was applied. This matters for §6.2.

---

## 3. Experiment 1 — Full eigenspectrum audit

### 3.1 Question

Classical MDS is exact only when the input distance matrix is Euclidean-embeddable. Isomap's
graph geodesic distances need not be. Negative eigenvalues of the double-centred matrix measure
that failure. **How much negative mass does this fit carry?**

### 3.2 Method

The audited spectrum was computed by hand rather than read from sklearn, because
`Isomap.kernel_pca_.eigenvalues_` is truncated to `n_components = 18` and therefore
*structurally cannot* exhibit a negative eigenvalue. Reporting it would beg the question.

1. `dist_matrix_` (10,000 × 10,000, float64) loaded with `mmap_mode="r"`.
2. Symmetry measured chunk-wise **before** any symmetric eigensolver read a single triangle:
   max deviation `1.421e-14` against a bound of `2.132e-09`.
3. Double-centring applied in place in mean form. The optimisation was **verified, not asserted**:
   mean-form vs the literal `-0.5 · J D² J` form agreed to `rtol = atol = 1e-12` on two
   independent 50×50 inputs — one genuine metric, one non-metric symmetrised-random.
4. Split eigensolve: `scipy.linalg.eigvalsh` for all 10,000 eigenvalues (values only, avoiding a
   full 10,000² eigenvector materialization), plus
   `scipy.linalg.eigh(subset_by_index=...)` for the top 40 eigenpairs.
5. Leading 18 hand-rolled eigenvalues cross-checked against sklearn's own
   `kernel_pca_.eigenvalues_` at `rtol = 1e-8` — **worst measured relative difference
   `8.532e-15`**. This attribute was used for that cross-check only, never as the audited
   spectrum.
6. Float64 end to end. Negativity classified by strict comparison against zero, with no
   zero-thresholding.

Spectrum length was mechanically asserted to be exactly 10,000 — a length no truncated
attribute can produce.

### 3.3 Gate statistics

Two statistics, both pre-registered with their thresholds **before** the spectrum was computed:

```
r = |λ_min_neg| / λ_max_pos           # is there one dominant negative outlier?
m = Σ|λ_neg| / Σ|λ|                   # how much total mass is negative?
```

`r` catches a single large negative eigenvalue; `m` catches a long diffuse negative tail. The
verdict is the worse of the two. All comparisons are **strict** less-than.

| | PASS | MARGINAL | else |
|---|---|---|---|
| `r` | < 0.10 | < 0.25 | FAIL |
| `m` | < 0.05 | < 0.15 | FAIL |

The classifier was asserted against synthetic boundary cases — including `(r=0.10, m=0)` →
MARGINAL, `(r=0.25, m=0)` → FAIL, `(r=0.05, m=0.20)` → FAIL — **before** it was applied to real
data.

### 3.4 Results

| Quantity | Value |
|---|---|
| Eigenvalues, total | 10,000 |
| Strictly positive | 4,971 |
| Strictly negative | **5,029** |
| `λ_max_pos` | 3.230854e+03 |
| `λ_min_neg` | −1.693588e+02 |
| Float64 noise floor (`n · eps · λ_max_pos`) | 7.173937e−09 |
| **`r`** | **0.052419** — passes (< 0.10) |
| **`m`** | **0.412071** — **fails even the 0.15 MARGINAL bound** |
| **Verdict** | **FAIL** |
| Steep-dropoff index / log-ratio | 2 / 2.4447 |

`|λ_min_neg|` sits roughly **ten orders of magnitude above the float64 noise floor**. The
negative tail is real structure, not rounding.

### 3.5 Reading

`r` passes while `m` fails, and that shape is the finding. This is not one pathological
short-circuit edge inflating a single eigenvalue — it is 5,029 negative eigenvalues, none
individually dominant, collectively carrying 41% of absolute mass. Roughly half the spectrum is
negative.

A statistic reported alone would have missed it: `r` on its own reads this spectrum as clean.

### 3.6 Independent check

Recomputing both statistics outside the notebook, from the persisted `.npz` alone, reproduces
`r = 0.052419`, `m = 0.412071` exactly.

---

## 4. Experiment 2 — Pre-registered k-sensitivity re-fit

### 4.1 Question

A diffuse negative tail has (at least) two candidate mechanisms:

- **H1 — intrinsic curvature.** The manifold is genuinely curved, so its geodesic metric is not
  isometrically embeddable in flat space at any dimension. No choice of `k` removes it.
- **H2 — kNN hop inflation.** Discrete shortest paths over a sparse neighbour graph
  systematically overestimate true geodesics. Densifying the graph (larger `k`) should shrink it.

These make opposite predictions about `m(k)`.

### 4.2 The confound this design controls for

Larger `k` reduces hop inflation **and** increases short-circuiting. Short-circuit edges make
graph geodesics more chordal — more like the ambient Euclidean distance — which *also* drives
`m` down. Both a genuine improvement and a destroyed manifold lower `m`, and the gate alone
cannot distinguish them.

Without a control, the largest `k` would tend to score best on the gate while being the `k` most
likely to have flattened the very structure under study. Two descriptive co-diagnostics were
therefore specified in advance, over an identical point-pair sample at every `k`:

```
GEO_AMBIENT_RATIO(k)  = median( graph geodesic distance / ambient Euclidean distance )
                        # > 1 on a curved manifold; collapses toward 1 under short-circuiting

LONG_EDGE_FRACTION(k) = fraction of kNN-graph edges longer than the 99th percentile
                        of the k=15 graph's edge-length distribution
```

Deliberately **no pass/fail threshold** was attached to either. The `k=15` values were already
known, so setting a bound on them afterwards would be threshold-fitting to a seen result.

### 4.3 Design and pre-registration

The design, the `k` set, the interpretation rule, and the prohibition on threshold revision were
written to `02-REFIT-PREREGISTRATION.md` and **committed to git (`057b084`) before any re-fit was
run**. This matters: a `k`-sweep run without a rule fixed beforehand is indistinguishable from
retrying until something passes.

- **k set:** `{5, 10, 30}`, against the `k=15` incumbent. All drawn from the prior phase's already
  connectivity-verified range. Fixed in advance and not extensible without a new pre-registration.
- **Excluded:** `k = 8, 20`. Dropped by the prior phase's fit budget; adding them *after* a FAIL
  would be widening the search in response to a result.
- **Held constant:** same subsample, same `seed = 20260729`, same row indices, `n_components = 18`,
  dense solver. Only `n_neighbors` varies.
- **Thresholds:** unchanged, copied verbatim.

`n_components` does not affect the gate statistics — `r` and `m` come from the full 10,000-value
spectrum of the double-centred geodesic matrix regardless of embedding width.

### 4.4 Results — all four k, reported regardless of outcome

| k | `r(k)` | `m(k)` | positive | negative | `GEO_AMBIENT_RATIO` | `LONG_EDGE_FRACTION` | Verdict |
|---|---|---|---|---|---|---|---|
| 5 | 0.060312 | 0.406433 | 4972 | 5028 | 2.828727 | 0.006540 | FAIL |
| 10 | 0.058311 | 0.410187 | 4971 | 5029 | 2.320592 | 0.008620 | FAIL |
| **15** *(incumbent)* | 0.052419 | 0.412071 | 4971 | 5029 | 2.117401 | 0.010000 | FAIL |
| 30 | 0.050708 | 0.415735 | 4963 | 5037 | 1.864727 | 0.013923 | FAIL |

Supporting detail (descriptive, no thresholds attached):

| k | `λ_max_pos` | `λ_min_neg` | noise floor | kNN edges | edge p99 | median geodesic |
|---|---|---|---|---|---|---|
| 5 | 5.432086e+03 | −3.276213e+02 | 1.206e−08 | 50,000 | 0.487021 | 1.593138 |
| 10 | 3.798254e+03 | −2.214809e+02 | 8.434e−09 | 100,000 | 0.504292 | 1.307802 |
| 15 | 3.230854e+03 | −1.693588e+02 | 7.174e−09 | 150,000 | 0.516666 | 1.192894 |
| 30 | 2.528065e+03 | −1.281927e+02 | 5.613e−09 | 300,000 | 0.539894 | 1.050865 |

`LONG_EDGE_FRACTION(15) = 0.010000` holds by construction (the threshold is the `k=15` p99) and
is a check on the definition, not a result. All four graphs were independently re-verified as a
single connected component.

### 4.5 Reading

**No `k` comes close to passing.** The best `m` on the ladder is 0.406 at `k=5` — still 2.7× the
MARGINAL bound. No candidate arose, so the short-circuit control never needed to adjudicate one.

**`m(k)` does not fall with `k`.** It rises slightly and monotonically: 0.4064 → 0.4102 → 0.4121
→ 0.4157, a total spread of 0.0093 across a 6× change in `k`.

**And densification demonstrably worked.** `GEO_AMBIENT_RATIO` falls monotonically 2.83 → 1.86
(geodesics becoming measurably more chordal) while `LONG_EDGE_FRACTION` rises 0.0065 → 0.0139
(more long edges admitted). The graph did what larger `k` is supposed to do — and negative mass
still did not decrease.

This is a **controlled negative**, and stronger than a merely flat `m(k)` would have been: the
mechanism that would have rescued the gate was actively engaged across the range and produced
nothing. **H2 is not supported.**

### 4.6 Validity checks

- The reconstructed fit configuration reproduces the frozen `fit_key = 43cf438bc944c509` exactly.
- The incumbent's `r`/`m` reproduce the published `0.052419` / `0.412071`.
- The 200,000-pair geodesic sample re-drawn for the co-diagnostics is asserted **bit-identical**
  to the one cached by the first experiment — the pairs are provably the same at every `k`.
- Each spectrum's leading 18 eigenvalues agree with sklearn's `kernel_pca_.eigenvalues_` to
  `rtol = 1e-8` (worst 5.6e−15).
- Each eigenvalue array asserted exactly (10,000,) float64.
- Every `|λ_min_neg|` is 10–11 orders of magnitude above its own noise floor.

### 4.7 Cost

Peak RSS 3.48 GiB (one `k` at a time, bindings released between). Fits 78.5 / 87.9 / 104.5 s;
eigensolves 122.9 / 120.4 / 122.8 s. Cache grew to 6.4 GiB.

---

## 5. Corroborating evidence from the residual-variance analysis

A separate step of the same phase computed two residual-variance curves over embedding dimension:
the Tenenbaum residual (`1 − R²` between geodesic and embedded pairwise distances, the canonical
Isomap definition) and an eigenvalue-based residual
(`1 − cumsum(λ⁺) / sum(λ⁺)`). It located the elbow by a deterministic maximum-curvature criterion
fixed in advance, swept over `d = 1..40`, ties to the lower `d`.

| Quantity | Value |
|---|---|
| Elbow, Tenenbaum curve (first pair draw) | 5 |
| Elbow, second **disjoint** pair draw | 5 — exact agreement |
| Elbow, eigenvalue cross-check curve | 8 |
| **Max divergence between the two curves** | **0.697664**, at d=5 |

**The divergence is the corroboration.** Both curves are bounded in [0, 1] and they disagree by
70 percentage points at the elbow. The eigenvalue-based curve normalizes by *positive* mass only,
so with 41% of absolute mass negative it reports far more variance captured than the actual
geodesic-vs-embedded R² finds. The negative mass thus surfaces through a second, independent
route.

Two further notes for a reviewer:

- The residual elbow (5) sits far below the prior phase's provisional intrinsic-dimension
  estimate of 18 (median over eight geometric estimators). **§6.4 revises this reading** —
  direct measurement puts the intrinsic dimension at ~20–25, making the elbow the outlier
  against four other estimates rather than a competing estimate of equal standing.
- A dimension of 5 was frozen for record-keeping so the verdict artifact could be self-contained.
  It is **not** an endorsement of 5 as a working dimension — the FAIL halts the downstream work
  regardless, and §6.4 gives independent reason to treat the value as suspect.

---

## 6. Experiment 3 — Diagnostic triage of the two remaining explanations

### 6.1 Status of this experiment

Unlike Experiments 1 and 2 this was **not pre-registered**. It is post-hoc diagnostic triage
run to close two alternative explanations that Experiments 1 and 2 left open. It defines no new
gate statistic and revises no threshold — `r` and `m` are computed by the identical definitions
and compared against the identical bounds. A reviewer should weight it accordingly: it is
hypothesis-narrowing, not confirmatory.

Script: `notebooks/diagnostics/gate_diagnostics.py`. It begins by recomputing the published
`r`/`m` from the cached spectrum as a control, and reproduces `0.052419` / `0.412071` exactly
before running anything else.

### 6.2 Is the negative mass an artifact of L2 normalization?

The embeddings are L2-normalized onto the unit hypersphere, and spherical geodesics yield
negative eigenvalues by construction. The cached subsample stores the original norms, so
normalization is **exactly invertible** — no re-download or re-derivation required. The fit was
repeated on the recovered raw vectors with everything else identical (same rows, same seed,
k=15, `n_components=18`, dense solver).

| | normalized (published) | unnormalized |
|---|---|---|
| `r` | 0.052419 | 0.054843 |
| `m` | **0.412071** | **0.413239** |
| positive eigenvalues | 4971 | 4973 |
| negative eigenvalues | 5029 | 5027 |
| `λ_max_pos` | 3.230854e+03 | 8.475097e+05 |
| `λ_min_neg` | −1.693588e+02 | −4.648015e+04 |
| Verdict | FAIL | FAIL |

`m` changes by **+0.001167 (+0.28%)**. `|λ_min_neg|` sits 2.47e+10 above the unnormalized noise
floor. **The normalization step is not the cause.**

**Important limit on what this licenses.** The raw norms are 16.029 ± 0.504 (cv = 3.1%) — the
embeddings already concentrated on a thin shell before normalization, so removing normalization
barely moved the geometry. That is *why* the result is null. This experiment therefore closes
the claim *"the L2 normalization step introduced the curvature"*. It does **not** close the
broader claim *"the data occupies a shell and shell-like geometry contributes to the
negativity"*, which remains untested and would need a different probe.

### 6.3 Is the cloud a manifold at all?

If intrinsic dimension were high, unstable, or strongly varying across the cloud, graph
geodesics would be a poor metric and the negativity would be uninformative about curvature. Two
independent estimators were run on both spaces: the two-NN maximum-likelihood estimator
(Facco et al.) and local PCA (components for 90% of local variance, k=60 neighbourhoods,
600 sampled centres).

| | normalized | unnormalized |
|---|---|---|
| TwoNN intrinsic dimension | 19.48 | 19.43 |
| Local PCA dim, median | 25.0 | 25.0 |
| Local PCA dim, mean ± std | 24.52 ± 2.01 | 24.57 ± 1.98 |
| Percentiles (5 / 25 / 50 / 75 / 95) | 21 / 23 / 25 / 26 / 28 | 21 / 23 / 25 / 26 / 27 |
| Range (min–max) | 19–29 | 19–29 |
| Neighbourhoods needing > 18 dims | 100% | 100% |
| Neighbourhoods needing > 40 dims | 0% | 0% |

Local dimension is **stable and tightly concentrated** — standard deviation 2.0, with the entire
5th-to-95th percentile range spanning 7 dimensions and no neighbourhood exceeding 29. That is
the signature of a genuine manifold of roughly constant dimension. A cloud that was not a
manifold would show wide, unstable, strongly k-dependent local dimension. **The manifold
assumption holds.**

### 6.4 An inconsistency this surfaces — the frozen dimension

The intrinsic dimension is ~20–25, which places four available estimates in tension:

| Estimate | Value |
|---|---|
| Local PCA, 90% variance (Experiment 3) | ~25 |
| TwoNN (Experiment 3) | ~19.5 |
| Median of 8 geometric estimators (prior phase) | 18 |
| `n_components` used for every fit | 18 |
| **Residual-curve elbow → the project's frozen dimension** | **5** |

Three independent estimators cluster at 18–25. **The elbow of 5 is the outlier, and it is the
value that was frozen.**

This is explainable, and it connects to the `CURVE_DIVERGENCE_MAX = 0.698` reported in §5. With
41% of eigenvalue mass negative, the Tenenbaum residual curve saturates early not because 5 is
the intrinsic dimension, but because the flat embedding is failing at every dimension — adding
coordinates past ~5 buys little when the target space is the wrong kind of space. On this
reading the elbow measured the *failure*, not the geometry.

Two consequences a reviewer should note:

- **The frozen dimension of 5 should be treated as suspect**, not inherited by downstream work.
- **`n_components = 18` sits below the intrinsic dimension**, and 100% of neighbourhoods need
  more than 18 dimensions for 90% of local variance. Every fit in this phase was
  dimension-starved.

Neither consequence changes the gate result: `r` and `m` derive from the full 10,000-value
spectrum of the double-centred geodesic matrix and are independent of `n_components`.

---

## 7. Interpretation — what is established, and what is not

### Established by these experiments

1. The negative eigenvalue mass is **real**, not numerical: 10–11 orders of magnitude above the
   float64 noise floor at every `k`.
2. It is **large**: ~50.3% of eigenvalues negative, ~41% of absolute mass, at every `k` tested.
3. It is **diffuse, not outlier-driven**: `r` passes its own bound at every `k` (max 0.060 vs
   0.10).
4. It is **not an artifact of neighbourhood scale**: flat-to-slightly-increasing across a 6×
   range of `k`, while co-diagnostics confirm the graph genuinely densified over that range.
5. It is **not an artifact of L2 normalization**: `m` moves 0.28% when normalization is exactly
   inverted (§6.2).
6. The cloud **is** a manifold: local intrinsic dimension is stable and tightly concentrated at
   ~20–25 across both spaces (§6.3).
7. Therefore **classical MDS is not a valid description of this geodesic metric**, and any
   downstream method assuming a valid flat Isomap embedding is unsupported on this fit.

### The surviving explanation

**A genuine, stable ~20–25 dimensional manifold whose geodesic metric is strongly
non-Euclidean.** Both competing explanations available at the time of Experiments 1 and 2 —
normalization artifact, and absence of manifold structure — were tested in Experiment 3 and
neither survives.

This is stronger than the position after Experiment 2, but it is still the *surviving*
hypothesis rather than a positively demonstrated one: the experiments eliminated alternatives,
they did not independently measure curvature. Measuring it is the natural next piece of work,
and it now has to be done with a representation that does not assume flatness.

### Remaining untested alternatives

- **Shell geometry.** §6.2 closes "normalization caused it" but not "the data occupies a thin
  shell (cv = 3.1% in raw norm) and shell-like geometry contributes." Untested; needs a
  different probe than the one run here.
- **Estimator-specific behaviour.** Both dimension estimators in §6.3 are neighbourhood-based
  and could share a bias. They agree with the prior phase's eight geometric estimators, which
  is reassuring but not independent confirmation.

### On the thresholds

The four thresholds encode a **project-specific tolerance**, not a universal standard. Some
negative mass is expected for any curved manifold, and there is no field-wide convention for how
much is disqualifying. What can be said without appeal to the specific bounds: at 41% of
absolute mass, the negative part of this spectrum is not a correction to a Euclidean picture — it
is comparable in magnitude to the positive part.

---

## 8. Limitations

- **One subsample.** 10,000 of 101,725 rows, a single seed. Sampling variance was not tested; a
  new-seed re-draw is an enumerated remediation option that was not exercised.
- **One dataset config.** One of 163 configs in the source dataset. Whether this behaviour is
  specific to `legacysurvey_dinov3_vitb16` or general across PU embedding spaces is untested,
  and is the single highest-value use of additional compute — the runs are independent and
  embarrassingly parallel.
- **Three re-fit values of `k`,** by deliberate pre-registration. `k = 8, 20` excluded.
- **Co-diagnostics are descriptive.** `GEO_AMBIENT_RATIO` and `LONG_EDGE_FRACTION` carry no
  pre-registered thresholds by design; they support a qualitative reading only.
- **Experiment 3 was not pre-registered** (§6.1). It is hypothesis-narrowing triage, weaker
  evidence than Experiments 1 and 2.
- **Shell geometry untested** (§7). `n_components = 18` sits below the measured intrinsic
  dimension and every fit was dimension-starved (§6.4) — this does not affect `r`/`m`, which
  derive from the full spectrum, but it does constrain the embeddings.
- **Curvature was never directly measured.** The conclusion rests on eliminating alternatives,
  not on an independent curvature estimate.
- **Documentation discrepancy:** narrative text in one summary refers to "nine synthetic boundary
  cases"; the implemented list contains eight. The classifier is exercised either way and no
  measured value depends on this, but the count as written is wrong.

---

## 9. Status and reproduction

### Status

The verdict artifact has been written. The phase is **not yet formally sealed** — a
human-verification step remains open at the time of writing. The measured values reported here
are final and committed; the outstanding step is acceptance, not further computation.

Per the project's own design, a documented FAIL is a **complete and legitimate outcome**, not a
project failure. Three remediation options are enumerated in the verdict artifact itself: re-fit
at a different `k` (already explored, and closed by Experiment 2), resample with a new seed
(untested — a different axis), or accept the documented FAIL as the reported result.

### Artifacts

| Path | Contents |
|---|---|
| `notebooks/01_manifold_and_gate.ipynb` §6.0–§6.9 | Experiment 1, residual/elbow analysis, verdict |
| `notebooks/02_k_sensitivity_refit.ipynb` | Experiment 2 |
| `notebooks/diagnostics/gate_diagnostics.py` | Experiment 3 |
| `.planning/phases/02-eigenspectrum-audit-validity-gate/02-REFIT-PREREGISTRATION.md` | Experiment 2's pre-registration + outcome |
| `notebooks/.cache/gate_verdict_43cf438bc944c509.json` | Machine-readable verdict, self-contained |
| `notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz` | Full 10,000-value spectrum |
| `notebooks/.cache/mds_residuals_43cf438bc944c509.npz` | Residual curves |

`notebooks/.cache/` is gitignored; artifacts regenerate from the notebooks.

### Reproducing

Both notebooks execute end-to-end with zero error cells and were committed with real executed
output (every code cell carries a non-null execution count):

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_manifold_and_gate.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_k_sensitivity_refit.ipynb
```

Expect ~100 s per dense eigensolve and a comparable time per Isomap fit; Experiment 2 performs
three of each. Warm runs read from cache.

Experiment 3 is a plain script — one fit plus one eigensolve, then two cheap dimension
estimators (~2 s). It self-checks by reproducing the published `r`/`m` from cache before
running anything:

```
python notebooks/diagnostics/gate_diagnostics.py
```

The headline statistics can be checked independently of the notebooks, from the persisted
spectrum alone:

```python
import numpy as np
z = np.load("notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz")
ev = z["eigvals_all"]                      # (10000,) float64
neg, pos = ev[ev < 0], ev[ev > 0]
r = abs(neg.min()) / pos.max()             # 0.052419
m = np.abs(neg).sum() / np.abs(ev).sum()   # 0.412071
```

### Commit trail

| Commit | Contents |
|---|---|
| `3401c0c`, `108486e` | Experiment 1 |
| `057b084` | Experiment 2 pre-registration — **before any re-fit ran** |
| `6dcefba`, `380122e`, `f30a882` | Experiment 2, k = 5 / 10 / 30 |
| `c9f4ea7`, `6844624` | Comparison table, interpretation rule, outcome |
| `5cf9a19`, `539dafa` | Residual curves, elbow, frozen dimension |
| `aea04ff`, `a2ca11f` | Verdict artifact, downstream enforcement |
| `9c6e2b5` | Experiment 3 diagnostic script |
