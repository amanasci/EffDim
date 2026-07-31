---
status: pre-registered
phase: 02-eigenspectrum-audit-validity-gate
created: 2026-07-31
supersedes: nothing
amends: none
trigger: 02-01-SUMMARY.md measured GATE_VERDICT=FAIL on the frozen k*=15 fit
---

# Phase 2 k-Sensitivity Re-fit — Pre-Registration

**This document is written and committed before any re-fit is run.** Its purpose is to fix
the k set, the statistics, the co-diagnostic, and the interpretation rule in advance, so that
no result produced afterwards can be selected for being convenient. SPEC-07 prohibits
treating a FAIL as something to retry until it passes; a k-sweep run without a rule fixed
beforehand is exactly that pattern wearing a different name.

## 1. What triggered this

Plan 02-01 computed the full 10,000-value classical-MDS eigenspectrum of the frozen k*=15
Isomap fit (`fit_key=43cf438bc944c509`) and measured:

| Statistic | Value | PASS bound | MARGINAL bound | Result |
|---|---|---|---|---|
| `R_STAT` | 0.052419 | < 0.10 | < 0.25 | PASS |
| `M_STAT` | 0.412071 | < 0.05 | < 0.15 | **FAIL** |

Verdict is the worse of the two: `GATE_VERDICT = FAIL`.

Shape of the failure: 5029 of 10,000 eigenvalues are strictly negative, none individually
dominant (`r` passes), collectively carrying 41% of total absolute eigenvalue mass.
`|LAMBDA_MIN_NEG| = 169.36` against a float64 noise floor of `7.17e-09`, so the negative
tail is real structure and not rounding. This is a diffuse negative tail, the signature of
a shortest-path metric that is systematically non-Euclidean rather than locally corrupted.

## 2. Hypothesis under test

Two mechanisms both produce a diffuse negative tail:

- **H1 — intrinsic curvature.** The manifold is genuinely curved, so its geodesic metric is
  not isometrically embeddable in flat Euclidean space at any dimension. Under H1 the
  negativity is the object of study, not an artifact, and no choice of k removes it.
- **H2 — kNN-graph hop inflation.** Discrete shortest paths over a sparse neighbour graph
  systematically overestimate true geodesics. Under H2 the negativity is a graph artifact
  that shrinks as the graph densifies (larger k).

These make opposite predictions about `m(k)`, which is what this re-fit measures.

## 3. The confound this pre-registration exists to control

Larger k reduces hop inflation **and** increases short-circuiting. Short-circuit edges make
graph geodesics more chordal — that is, more like the ambient Euclidean distance — which
also drives `m` down. So both a genuine improvement and a destroyed manifold lower `m`, and
the gate alone cannot distinguish them.

Without a co-diagnostic, the largest k would tend to score best on the gate while being the
k most likely to have flattened the very curvature this milestone exists to measure. A PASS
obtained that way is a worse outcome than the honest k=15 FAIL.

**Phase 1's `short_circuit_risk` flag does not cover this.** It is derived from the
connectivity auto-extend path — it records whether k had to be pushed past the base range to
connect the graph. It reads `false` for every connected k, including k=30, so it carries no
information about shortcut edges at a given k.

## 4. Pre-registered design

### 4.1 The k set — fixed now, not extensible after seeing results

    K_REFIT = [5, 10, 30]

All three are drawn from Phase 1's `SWEEP_K_RANGE`, all are already connectivity-verified
(`connected_k = [5, 8, 10, 15, 20, 30]`, `n_components == 1` for each), and all three already
received Phase 1 stage-2 fits. k=15 is the incumbent and is not re-fit; its measured values
above are the comparison baseline.

k=8 and k=20 are deliberately excluded: Phase 1 dropped them under `STAGE2_MAX_FITS=4`, and
adding them now, after a FAIL, would be widening the search in response to a result.

**No k outside this set may be added to this analysis.** If the pre-registered set proves
uninformative, that is itself a finding and is reported as one. Testing a fourth k requires a
new, separately committed pre-registration stating why, and is recorded as an amendment.

### 4.2 What is held constant

Only `n_neighbors` varies. Everything else is pinned to the Phase 1 fit configuration:

    dataset      = "legacysurvey_dinov3_vitb16"
    n_rows       = 10000
    seed         = 20260729          # same subsample, same row_indices
    normalize    = True
    n_components = 18
    eigen_solver = "dense"

`n_components` is held at 18 for comparability. It does not affect the gate statistics:
`r` and `m` are computed from the eigenspectrum of the double-centred full geodesic matrix,
which has length 10,000 regardless of the embedding width.

### 4.3 Thresholds — copied verbatim from §6.0, unchanged

    R_MAX_PASS     = 0.10
    M_MAX_PASS     = 0.05
    R_MAX_MARGINAL = 0.25
    M_MAX_MARGINAL = 0.15

All comparisons remain strict less-than at every boundary. **These values are not revisable
by this document or by anything it produces.** If they must ever change, that is a separate
documented amendment with a stated reason and a full re-run recorded as a new
pre-registration — never a quiet edit.

### 4.4 Co-diagnostic — the short-circuit control

For each k, over the same pre-registered point-pair sample used for the residual curves
(`R2_PAIR_COUNT = 200_000`, `r2_pair_seed` from §6.0, identical pairs across all k):

    GEO_AMBIENT_RATIO(k) = median over sampled pairs of
                           ( graph geodesic distance / ambient Euclidean distance )

On a curved manifold this ratio exceeds 1 and grows with true separation. Short-circuiting
collapses it toward 1. Reported alongside:

    LONG_EDGE_FRACTION(k) = fraction of kNN-graph edges whose length exceeds the
                            99th percentile of the k=15 graph's edge-length distribution

Both are descriptive statistics, not gates. They carry no pass/fail threshold, because
inventing one now — with the k=15 values already known — would be threshold-setting against
a seen result. They enter the interpretation rule only through the qualitative test in §5.

## 5. Interpretation rule — fixed before any fit runs

Let `m(k)` be `M_STAT` at neighbour count k. Baseline: `m(15) = 0.412071`.

**Rule A — negativity robust across the ladder.**
If `m(k) >= M_MAX_MARGINAL` for all k in {5, 10, 15, 30}, the non-Euclideanity is not an
artifact of neighbourhood scale. H2 is not supported. `GATE_VERDICT = FAIL` stands as the
Phase 2 outcome, is written to `gate_verdict_43cf438bc944c509.json` against the incumbent
k*=15 fit, and the milestone proceeds to its documented-FAIL close-out. No further k is tried.

**Rule B — some k clears the gate.**
If any k has `m(k) < M_MAX_MARGINAL` and `r(k) < R_MAX_MARGINAL`, that k is a *candidate
only*, not an adopted k*. It must then pass the short-circuit test:

> `GEO_AMBIENT_RATIO(k)` must not fall materially below the k=15 value, and
> `LONG_EDGE_FRACTION(k)` must not rise materially above it.

If the candidate fails that test, its improvement is attributed to short-circuiting, the
candidate is rejected, and Rule A's outcome applies. This rejection is recorded with its
numbers, not silently dropped.

**Rule C — a candidate survives the short-circuit test.**
Adopting it is *not* automatic and is *not* done by this analysis. Changing k* invalidates
Phase 1's frozen fit, its `fit_key`, its handoff artifact, and the plateau-stability
selection that chose k*=15 through a blocking human-verify gate. A surviving candidate is
reported as a finding; adopting it requires re-running Phase 1's stage-2 stability selection
under the new k and a separate documented amendment to Phase 1.

**Rule D — monotone trend without a clear winner.**
If `m(k)` decreases monotonically with k but no k clears `M_MAX_MARGINAL`, report the trend
as evidence bearing on H1 vs H2 and apply Rule A. A trend in the predicted direction is not
a PASS.

## 6. What this analysis may not do

- May not revise any of the four thresholds, or add, drop, or reweight a gate statistic.
- May not add a k outside `K_REFIT` without a new committed pre-registration.
- May not adopt a new k* directly; see Rule C.
- May not treat a FAIL that survives this analysis as an error. A documented FAIL with its
  remediation options enumerated is a complete, legitimate, reportable milestone outcome.
- May not select among the three fits by which produced the most convenient verdict. The
  full `m(k)` / `r(k)` table is reported for all of {5, 10, 15, 30} regardless of outcome.

## 7. Expected cost

Three fresh Isomap fits at n=10,000, each producing a ~1.66 GiB joblib and requiring a
~104 s dense eigensolve of the 10,000x10,000 double-centred geodesic matrix, plus the fit
time itself. Approximately 5 GiB of additional cache. Measured against 212 GiB free.

## 8. Outcome

_Appended 2026-07-31 after all three re-fits completed. Executed in
`notebooks/02_k_sensitivity_refit.ipynb` (a new notebook; `01_manifold_and_gate.ipynb` was
not edited), run end-to-end with `jupyter nbconvert --to notebook --execute --inplace`:
27 cells, zero error cells, every code cell carrying a non-null execution count._

### 8.1 The measured table

All four k, reported regardless of outcome per §6. Thresholds unchanged: `r < 0.10` PASS /
`< 0.25` MARGINAL, `m < 0.05` PASS / `< 0.15` MARGINAL, strict less-than throughout.

| k | `r(k)` | `m(k)` | `n_positive` | `n_negative` | `GEO_AMBIENT_RATIO` | `LONG_EDGE_FRACTION` | Verdict |
|---|---|---|---|---|---|---|---|
| 5 | 0.060312 | 0.406433 | 4972 | 5028 | 2.828727 | 0.006540 | **FAIL** |
| 10 | 0.058311 | 0.410187 | 4971 | 5029 | 2.320592 | 0.008620 | **FAIL** |
| 15 *(incumbent)* | 0.052419 | 0.412071 | 4971 | 5029 | 2.117401 | 0.010000 | **FAIL** |
| 30 | 0.050708 | 0.415735 | 4963 | 5037 | 1.864727 | 0.013923 | **FAIL** |

Supporting detail (descriptive, no thresholds attached):

| k | `LAMBDA_MAX_POS` | `LAMBDA_MIN_NEG` | noise floor | kNN edges | edge p99 | median geodesic |
|---|---|---|---|---|---|---|
| 5 | 5.432086e+03 | -3.276213e+02 | 1.206e-08 | 50,000 | 0.487021 | 1.593138 |
| 10 | 3.798254e+03 | -2.214809e+02 | 8.434e-09 | 100,000 | 0.504292 | 1.307802 |
| 15 | 3.230854e+03 | -1.693588e+02 | 7.174e-09 | 150,000 | 0.516666 | 1.192894 |
| 30 | 2.528065e+03 | -1.281927e+02 | 5.613e-09 | 300,000 | 0.539894 | 1.050865 |

`LONG_EDGE_TAU = 0.516666` (the 99th percentile of the k=15 graph's edge lengths), so
`LONG_EDGE_FRACTION(15) = 0.010000` is ~0.01 by construction — a check on the definition,
not a result. All four graphs were independently re-verified connected
(`n_components == 1`). Every `|LAMBDA_MIN_NEG|` is 10-11 orders of magnitude above its
float64 noise floor: the negative tail is real structure at every k, not rounding.

Validity checks that the numbers had to pass: the reconstructed fit configuration reproduces
Phase 1's frozen `fit_key = 43cf438bc944c509` exactly; the incumbent's `r`/`m` reproduce
02-01's published `0.052419` / `0.412071`; the 200,000-pair geodesic sample re-drawn for the
co-diagnostics is bit-identical to the one already cached by 01 (so the pairs are provably
the same at every k); each hand-rolled spectrum's leading 18 eigenvalues agree with sklearn's
own `kernel_pca_.eigenvalues_` to `rtol=1e-8` (worst 5.6e-15); and each eigenvalue array was
asserted to be exactly 10,000 float64 values.

### 8.2 Which rule fired: **Rule A**

- **Rule B never engaged.** `CANDIDATES = []`. No k satisfies `m(k) < M_MAX_MARGINAL`; the
  smallest `m` measured is 0.406433 at k=5, which is 2.7x the 0.15 MARGINAL bound. Because
  no candidate exists, the short-circuit test had nothing to run on and Rule C's
  reporting-only path was never reached. No k* is adopted, and none was even reportable.
- **Rule D does not apply.** `m(k)` is not monotone decreasing in k. In ascending k it is
  0.406433, 0.410187, 0.412071, 0.415735 — flat to *slightly increasing*. The total spread
  across a 6x change in neighbourhood size is 0.0093, about 2.3% of the statistic's own
  value.
- **Rule A's precondition holds**: `m(k) >= M_MAX_MARGINAL` for all k in {5, 10, 15, 30}.

Therefore `GATE_VERDICT = FAIL` stands as the Phase 2 outcome against the incumbent k*=15
fit, and the milestone proceeds to its documented-FAIL close-out. No further k is tried.

### 8.3 What this says about H1 vs H2

H2 predicted that densifying the graph would shrink the negative mass. It did not. `m` moved
by 0.9 percentage points across `k = 5 -> 30` and moved in the *wrong direction*: the densest
graph has the largest negative mass, not the smallest.

The co-diagnostics show this is not because densification failed to do anything. It did
exactly what §3 warned it would do:

- `GEO_AMBIENT_RATIO` falls monotonically with k — 2.83, 2.32, 2.12, 1.86 — so graph
  geodesics become steadily more chordal, i.e. more like the ambient Euclidean distance.
- `LONG_EDGE_FRACTION` rises monotonically with k — 0.0065, 0.0086, 0.0100, 0.0139 — so
  the graph is taking in progressively more long edges relative to the k=15 reference.

Larger k measurably flattened the manifold and admitted more shortcut-length edges, and the
non-Euclidean mass still did not fall. That is the signature §3 was written to detect, and
here it lands on the side that *strengthens* the H1 reading rather than rescuing a candidate:
the negativity survives the very operation that was its most plausible artifactual
explanation. H2 (kNN-graph hop inflation) is not supported by this evidence.

This is a bound on what was tested, not a proof of H1. Three neighbour counts plus the
incumbent, one seed, one 10,000-row subsample, one dataset. What the analysis establishes is
narrower and sufficient for the gate: **the FAIL is not an artifact of neighbourhood scale.**

### 8.4 Conclusion

The k-sensitivity re-fit is complete and its result is unwelcome in the ordinary sense — no k
rescues the gate — and legitimate in the sense that matters. `GATE_VERDICT = FAIL` on the
frozen k*=15 fit is the real, measured Phase 2 outcome. Per §6 it is not an error to work
around: it is a complete, reportable milestone result whose remediation options belong to
plan 02-03's verdict artifact and the milestone-level decision, not to this analysis.

Nothing in §4.3 was revised, no k outside `K_REFIT` was tested, no k* was adopted, and no
pass/fail threshold was invented for either co-diagnostic.

**Artifacts** (all gitignored under `notebooks/.cache/`): `isomap_9db36086f7472619.joblib`
(k=5), `isomap_9fbaf46e3570c8b7.joblib` (k=10), `isomap_860e4b66f08af831.joblib` (k=30),
their `mds_eigenspectrum_{fit_key}.npz` spectra, `codiag_k{5,10,15,30}_{fit_key}.npz`, and
`k_sensitivity_refit_43cf438bc944c509.json` (the machine-readable table above). Measured
cost: fits 78.5 s / 87.9 s / 104.5 s, eigensolves 122.9 s / 120.4 s / 122.8 s, peak RSS
3.48 GiB, ~4.8 GiB of additional cache.
