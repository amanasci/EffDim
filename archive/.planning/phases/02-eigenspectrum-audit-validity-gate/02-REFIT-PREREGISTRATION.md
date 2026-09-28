---
status: pre-registered
phase: 02-eigenspectrum-audit-validity-gate
created: 2026-07-31
supersedes: nothing
amends: none
trigger: 02-01-SUMMARY.md measured GATE_VERDICT=FAIL on the frozen k*=15 fit
---

# Phase 2 k-Sensitivity Re-fit — Pre-Registration

**Written and committed before any re-fit ran.** Fixes the k set, statistics, co-diagnostic,
and interpretation rule in advance — a k-sweep without a prior rule is retry-until-pass
wearing a different name (SPEC-07 prohibits that).

## 1. Trigger

02-01 measured, on `fit_key=43cf438bc944c509`:

| Statistic | Value | PASS | MARGINAL | Result |
|---|---|---|---|---|
| `R_STAT` | 0.052419 | < 0.10 | < 0.25 | PASS |
| `M_STAT` | 0.412071 | < 0.05 | < 0.15 | **FAIL** |

`GATE_VERDICT = FAIL` (worse-of-two). Shape: 5029/10,000 eigenvalues strictly negative, none
dominant, 41% of absolute mass; `|LAMBDA_MIN_NEG| = 169.36` vs noise floor 7.17e-09 — a
diffuse negative tail, systematically non-Euclidean rather than locally corrupted.

## 2. Hypotheses

- **H1 intrinsic curvature:** geodesic metric not flat-embeddable at any dimension; no k
  removes the tail.
- **H2 kNN hop inflation:** sparse-graph shortest paths overestimate geodesics; densifying
  (larger k) shrinks the tail.

Opposite predictions for m(k).

## 3. The confound

Larger k reduces hop inflation AND increases short-circuiting — both lower m. Without a
control, the largest k scores best while most likely flattening the curvature under study; a
PASS obtained that way is worse than the honest k=15 FAIL. Phase 1's `short_circuit_risk`
flag does not cover this (it only records connectivity auto-extension; false for every
connected k).

## 4. Pre-registered design

- **§4.1 k set:** `K_REFIT = [5, 10, 30]`, all connectivity-verified in Phase 1. k=15 is the
  incumbent baseline, not re-fit. k=8, 20 deliberately excluded (dropped by
  `STAGE2_MAX_FITS=4`; adding them after a FAIL widens the search in response to a result).
  **No k outside this set** without a new committed pre-registration recorded as an
  amendment.
- **§4.2 held constant:** dataset `legacysurvey_dinov3_vitb16`, n_rows 10000, seed 20260729
  (same rows), normalize True, n_components 18, eigen_solver dense. Only n_neighbors varies.
  n_components does not affect r/m (full 10,000-value spectrum).
- **§4.3 thresholds, verbatim from §6.0, not revisable here:** `R_MAX_PASS=0.10`,
  `M_MAX_PASS=0.05`, `R_MAX_MARGINAL=0.25`, `M_MAX_MARGINAL=0.15`; strict less-than. A
  change is a separate documented amendment plus full re-run, never a quiet edit.
- **§4.4 co-diagnostics** (same 200,000-pair sample at every k; descriptive, no thresholds —
  the k=15 values are already known, so inventing bounds now would be threshold-setting
  against a seen result):
  `GEO_AMBIENT_RATIO(k)` = median(geodesic/ambient); >1 on a curved manifold, collapses
  toward 1 under short-circuiting. `LONG_EDGE_FRACTION(k)` = fraction of edges beyond the
  k=15 p99.

## 5. Interpretation rule — fixed before any fit

Baseline `m(15) = 0.412071`.

- **Rule A:** `m(k) >= M_MAX_MARGINAL` for all k ∈ {5,10,15,30} → not a neighbourhood-scale
  artifact; H2 unsupported; FAIL stands, written against the incumbent fit; no further k.
- **Rule B:** some k has `m(k) < M_MAX_MARGINAL` and `r(k) < R_MAX_MARGINAL` → *candidate
  only*; must pass the short-circuit test (GEO_AMBIENT_RATIO not materially below k=15,
  LONG_EDGE_FRACTION not materially above). Failing candidate = short-circuiting; rejected
  with its numbers recorded; Rule A applies.
- **Rule C:** a surviving candidate is *reported*, never adopted — adoption invalidates
  Phase 1's frozen fit/fit_key/handoff/plateau selection and requires re-running Phase 1's
  stage-2 selection plus a documented Phase 1 amendment.
- **Rule D:** monotone-decreasing m(k) with no k clearing the bound → report the trend, apply
  Rule A. A trend is not a PASS.

## 6. Prohibitions

No threshold revision; no k outside K_REFIT; no direct k* adoption; no treating a surviving
FAIL as an error; no selecting among fits by convenience — the full table is reported for
all of {5,10,15,30} regardless of outcome.

## 7. Expected cost

Three fits at n=10,000: ~1.66 GiB joblib each, ~104 s dense eigensolve each, ~5 GiB
additional cache (212 GiB free).

## 8. Outcome

_Appended 2026-07-31 after all three re-fits. Executed in `02_k_sensitivity_refit.ipynb`
(new notebook; 01 not edited); 27 cells, zero error cells, full nbconvert execution._

### 8.1 Measured table

| k | `r(k)` | `m(k)` | `n_positive` | `n_negative` | `GEO_AMBIENT_RATIO` | `LONG_EDGE_FRACTION` | Verdict |
|---|---|---|---|---|---|---|---|
| 5 | 0.060312 | 0.406433 | 4972 | 5028 | 2.828727 | 0.006540 | **FAIL** |
| 10 | 0.058311 | 0.410187 | 4971 | 5029 | 2.320592 | 0.008620 | **FAIL** |
| 15 *(incumbent)* | 0.052419 | 0.412071 | 4971 | 5029 | 2.117401 | 0.010000 | **FAIL** |
| 30 | 0.050708 | 0.415735 | 4963 | 5037 | 1.864727 | 0.013923 | **FAIL** |

| k | `LAMBDA_MAX_POS` | `LAMBDA_MIN_NEG` | noise floor | kNN edges | edge p99 | median geodesic |
|---|---|---|---|---|---|---|
| 5 | 5.432086e+03 | -3.276213e+02 | 1.206e-08 | 50,000 | 0.487021 | 1.593138 |
| 10 | 3.798254e+03 | -2.214809e+02 | 8.434e-09 | 100,000 | 0.504292 | 1.307802 |
| 15 | 3.230854e+03 | -1.693588e+02 | 7.174e-09 | 150,000 | 0.516666 | 1.192894 |
| 30 | 2.528065e+03 | -1.281927e+02 | 5.613e-09 | 300,000 | 0.539894 | 1.050865 |

`LONG_EDGE_TAU = 0.516666` (k=15 p99), so `LONG_EDGE_FRACTION(15)=0.010000` by construction.
All four graphs re-verified connected. Every `|LAMBDA_MIN_NEG|` 10-11 orders above its noise
floor. Validity: reconstructed cfg reproduces `43cf438bc944c509`; incumbent r/m reproduce
published values; pair sample bit-identical to 01's cache; top-18 vs sklearn rtol=1e-8
(worst 5.6e-15); every array exactly (10000,) float64.

### 8.2 Rule A fired

`CANDIDATES = []` — smallest m is 0.406433 at k=5, 2.7× the 0.15 bound; Rule B never
engaged, Rule C unreachable. Rule D inapplicable: m(k) ascending is 0.406433, 0.410187,
0.412071, 0.415735 — flat to slightly *increasing* (spread 0.0093, ~2.3% of the statistic).
**FAIL stands against the incumbent k*=15 fit; no further k.**

### 8.3 H1 vs H2

Densification measurably worked — GEO_AMBIENT_RATIO fell monotonically 2.83→1.86 (geodesics
more chordal), LONG_EDGE_FRACTION rose 0.0065→0.0139 (more long edges admitted) — and
negative mass still did not fall; the densest graph has the *largest* m. The negativity
survives its most plausible artifactual explanation. **H2 not supported.** A bound, not a
proof of H1: three k plus incumbent, one seed, one subsample, one dataset.

### 8.4 Conclusion

FAIL on the frozen k*=15 fit is the real, measured Phase 2 outcome — complete and
reportable; remediation belongs to 02-03's artifact and the milestone decision. Nothing in
§4.3 revised, no outside k tested, no k* adopted, no co-diagnostic threshold invented.

**Artifacts** (gitignored): `isomap_9db36086f7472619.joblib` (k=5),
`isomap_9fbaf46e3570c8b7.joblib` (k=10), `isomap_860e4b66f08af831.joblib` (k=30), their
`mds_eigenspectrum_{fit_key}.npz`, `codiag_k{5,10,15,30}_{fit_key}.npz`,
`k_sensitivity_refit_43cf438bc944c509.json`. Cost: fits 78.5/87.9/104.5 s, eigensolves
122.9/120.4/122.8 s, peak RSS 3.48 GiB, ~4.8 GiB cache.
