# Requirements: EffDim v1.1 — PU Manifold Curvature

**Defined:** 2026-07-29
**Core Value:** One call over an `(n_samples, n_features)` array returns a comparable panel of effective dimensionality estimates, so a researcher can see how spectral and geometric notions of dimension agree or disagree on the same data.

**Milestone Goal:** Reconstruct the PU embedding manifold via Isomap, build a smooth decoder parameterization of it, compute mean curvature, and test whether foundation-model representational alignment (MKNN) varies with local curvature.

**Scope note:** Notebook-only milestone. `src/effdim/` and `pyproject.toml` are not modified. Requirements below describe observable behaviours of the notebooks under `notebooks/` plus the local, non-installed helper package `notebooks/pu_manifold/`.

---

## v1.1 Requirements

### Data Loading (DATA)

- [x] **DATA-01**: Loads exactly one config of `UniverseTBD/pu-embeddings` (`legacysurvey_dinov3_vitb16`) without downloading the other 162 configs
- [x] **DATA-02**: Reproducible 10,000-row subsample from the 101,725 available rows, controlled by an explicit, recorded seed
- [x] **DATA-03**: `_hsc` and `_legacysurvey` arrays guaranteed row-aligned after subsampling, enforced by an assertion rather than convention
- [x] **DATA-04**: Embedding norm distribution shown plus an explicit statement of which metric (Euclidean vs cosine) the pipeline uses and why
- [x] **DATA-05**: Notebook states its own Python floor (3.11) and installs its dependencies from a notebook cell, leaving `pyproject.toml` untouched

### Manifold Reconstruction (ISO)

- [x] **ISO-01**: Connected-component count of the k-NN graph shown before Isomap is fitted, so silent bridging by `_fix_connected_components` cannot pass unnoticed
- [x] **ISO-02**: Embedding and eigenspectrum stability shown across at least three `n_neighbors` values, detecting short-circuit edges
- [x] **ISO-03**: `effdim.compute_dim` estimates on the raw 768-d embeddings compared against the dimension the Isomap eigenspectrum suggests
- [x] **ISO-04**: Isomap fit at n=10,000 completes and is cached, so downstream work never re-pays the fit cost
- [x] **ISO-05**: Re-running any notebook gets identical results from the cache; a config change produces a new cache key rather than silently reusing a stale one

### Eigenspectrum Audit and Validity Gate (SPEC)

- [x] **SPEC-01**: The **full** classical-MDS eigenspectrum shown, computed from `isomap.dist_matrix_` by manual double-centring — not from the truncated `kernel_pca_.eigenvalues_` attribute, which cannot show the negative tail
- [x] **SPEC-02**: Leading eigenvalues confirmed large and positive; steep dropoff location shown
- [x] **SPEC-03**: Negative-eigenvalue magnitude shown as an explicit reported statistic, with a stated and justified threshold
- [x] **SPEC-04**: Residual-variance-vs-dimension curve shown with the elbow identified by a stated criterion, not eyeballed
- [x] **SPEC-05**: Chosen embedding dimension `d` frozen and recorded before any decoder is trained
- [ ] **SPEC-06**: A PASS / MARGINAL / FAIL verdict written as a machine-readable artifact that downstream notebooks check before running
- [ ] **SPEC-07**: On FAIL, the notebook halts with remediation options enumerated, and that documented failure is itself a complete, reportable milestone outcome

### Geometry Representation (GEOM)

Added 2026-07-31 with Phase 02.1 (INSERTED), after Phase 2's `GATE_VERDICT = FAIL` established that classical MDS is an invalid description of the Isomap geodesic geometry for these embeddings (`m = 0.412071` against a 0.15 MARGINAL bound; ~41% of absolute eigenvalue mass negative, robust across `k`, survey column, disjoint resample, and normalization).

- [x] **GEOM-01**: Which manifold-learning methods share Isomap's flat-target assumption and would fail the same way on this data shown, establishing the failure as a property of the method class rather than an implementation choice
- [x] **GEOM-02**: Candidate representations that do not assume a flat target shown — Riemannian/hyperbolic and product-manifold embeddings, graph-native curvature (Ollivier-Ricci, Forman-Ricci) requiring no embedding, diffusion maps, pseudo-Euclidean/Krein-space treatments — each with its assumptions, cost, and what it would demand of the decoder phase
- [x] **GEOM-03**: What 41% negative eigenvalue mass means geometrically shown, with an argued judgment on whether indefinite MDS or distance-matrix correction is principled here or merely cosmetic — a method that hides the negativity rather than representing it must be named as such
- [x] **GEOM-04**: One recommended representation with its rationale, the alternatives it was chosen over, and the evidence it is expected to be judged against, sufficient for the decoder phase to be re-specified without re-opening the question
- [x] **GEOM-05**: What the recommendation implies for the working dimension shown. `D_FROZEN = 5` derives from a residual elbow flagged suspect against three other estimates clustering at 18-25; whether the chosen representation inherits, revises, or discards it must be stated explicitly

### Chart Autoencoder Validity (CAE)

Added 2026-08-03 with Phase 02.2 (INSERTED), to empirically test whether the Chart Auto-Encoder method of arXiv:1912.10094 — which assumes only local Euclidean charts, not the global Euclidean embeddability Phase 2's gate falsified — yields a representation Phase 3 can validly decode from.

- [x] **CAE-01**: Gate metrics and numeric PASS/FAIL thresholds pre-registered and committed before any CAE fit runs, with the ordering proved by git ancestry rather than asserted
- [x] **CAE-02**: Chart Auto-Encoder trained on the frozen Phase 1 10,000-row normalized subsample and reproducible from recorded seeds, with the full architecture named — initial encoder to R^l, N over-specified chart encoders into chart spaces `(0,1)^d`, per-chart decoders, one shared embedding decoder, and a partition-of-unity chart predictor — trained under the paper's loss with Lipschitz regularization on chart-encoder spectral norms and FPS-seeded per-chart pre-training
- [x] **CAE-03**: Held-out reconstruction error shown as an aggregate metric plus a per-output-dimension distribution against a matched-capacity single-chart autoencoder control and the classical-MDS/Isomap reconstruction, so a CAE that merely ties the control cannot be read as a success
- [x] **CAE-04**: Chart-transition cycle residual `R_cycle` shown over held-out points
- [x] **CAE-05**: Chart count obtained a posteriori after weight-decay pruning by a stated decoder-weight-norm tolerance, with the over-specified initial `N`, the surviving count, its stability across seeds, and the paper's unfaithfulness and coverage measures shown
- [x] **CAE-06**: Decoders trained with a C2-smooth activation rather than the reference implementation's ReLU, since DEC-02 and CURV-01..03 need a non-zero second derivative, with the substitution and any reconstruction cost stated; the compact-manifold-with-positive-reach assumption behind the paper's Thm 2 recorded as an unverified assumption about the PU point cloud, not a verified one
- [x] **CAE-07**: `cae_verdict_{fit_key}.json` written as a machine-readable PASS/FAIL artifact carrying every pre-registered metric and threshold, checked by downstream notebooks before any expensive cell; on FAIL the notebook halts, findings are documented, and that documented failure is itself a complete milestone outcome

### Topological Autoencoder Validity (TOPO)

Added 2026-08-07 with Phase 02.4 (INSERTED), to empirically test whether the Topological Auto-Encoder of arXiv:1906.00722 (Moor, Horn, Rieck, Borgwardt, ICML 2020) — which optimises topological signature matching rather than distance preservation — yields a representation Phase 3 can validly decode from.

- [x] **TOPO-01**: 0-dimensional persistence pairings computed from the minimum spanning tree of each batch's pairwise distance matrix in both input and latent space, ties broken lexicographically on distance then row index then column index, combined into the paper's topological loss
- [x] **TOPO-02**: A Topological Auto-Encoder trained on the frozen Phase 1 10,000-row normalized subsample, reproducible from recorded seeds, C2-smooth throughout encoder and decoder, across a pre-registered latent-dimension ladder, reusing `cae.py` by import and never editing it
- [x] **TOPO-03**: Gate metric definitions and numeric thresholds pre-registered and committed before any fit runs, with the ordering proved by git ancestry rather than asserted
- [x] **TOPO-04**: Exactly three gating metrics — topological fidelity, reconstruction margin against a matched-capacity plain autoencoder, and a rank-structure measure; geodesic distortion and the paper's unfaithfulness and coverage measures reported for comparability and never gating
- [x] **TOPO-05**: `topoae_verdict_{fit_key}.json` written on PASS and FAIL alike, self-contained and reproducible from its own recorded metrics and thresholds, with an absent or non-finite gating metric halting the run and writing nothing
- [x] **TOPO-06**: A Phase 3 handoff written on PASS only, with any existing handoff at that fit key actively deleted on FAIL rather than merely not written
- [x] **TOPO-07**: On PASS, a new dated amendment revisiting Phase 02.1's falsifier reading and a marker-guarded traceability update for the reinstated DEC and CURV requirements, both existence-guarded so a second run is a reported no-op and no sealed 02.1 artifact is edited
- [x] **TOPO-08**: The mandatory Swiss roll sanity check, `notebooks/02.4_swiss_roll_topoae_check.ipynb`, importing the model unchanged and compared against a matched `cae.PlainAutoEncoder` baseline at the same 2-D bottleneck

### Decoder (DEC)

> **SUPERSEDED 2026-08-17 by `## Phase 3 Requirement Re-Mint`.** The 2026-07-31 amendment asked a
> reader to re-read "Isomap coordinates" as the Phase 02.1 replacement. That patch is withdrawn:
> all 13 DEC / CURV requirements below are now re-minted with rewritten text under the same IDs,
> so nothing needs re-reading. Two premises died — *Isomap coordinates* and *a single global
> chart* — and the re-minted text names what replaced them. See the dated re-mint section at the
> end of this file for the old-to-new mapping. The requirement *intent* is unchanged: a C2-smooth
> decoder whose Jacobian yields an analytic curvature field, falsified against a synthetic control.

- [x] **DEC-01**: Train a per-chart decoder — `cae.ChartAutoEncoder`'s `chart_decoders[i]` composed with the single shared `embedding_decoder` — mapping each point's own chart coordinate to the 768-d embedding, with a C2-smooth activation throughout the forward path
- [x] **DEC-02**: No ReLU-family activation anywhere in the decoder, enforced by a guard that raises (`chart_curvature.assert_c2_activation`, `decoder_curvature.assert_c2_decoder`) rather than merely verified once
- [x] **DEC-03**: Held-out reconstruction quality shown per `n_charts` configuration in the PU sweep, not training loss and not from a single fit
- [x] **DEC-04**: Both an aggregate reconstruction metric and a per-output-dimension distribution shown for every configuration, so a good average cannot hide a subset failure
- [x] **DEC-05**: Every fit reproducible from a recorded torch seed, and every result reported across seeds — at least 5 on the Swiss roll gate, 3 on PU — never from a single draw

### Curvature Field (CURV)

- [x] **CURV-01**: First fundamental form `g = J^T J` from the chart decoder's Jacobian via `torch.func`, batched with `vmap` rather than looped, under a selectable reverse or forward differentiation mode whose outputs agree to float64 round-off
- [x] **CURV-02**: Second fundamental form as the normal-projected ambient Hessian, computed trace-first-then-project — the `g`-trace and the normal projection commute, so the trace is taken first and the projection applied by a `chart_dim` by `chart_dim` solve; no `(D, D)` projector and no full `II` tensor is ever materialized
- [x] **CURV-03**: Mean curvature **vector** field and its norm shown, labelled a vector norm and never Gaussian or principal curvature, under the pinned `CURVATURE_CONVENTION = "trace"`, `H = tr_g(II)` — a unit `d`-sphere gives a norm of `d`, not 1
- [x] **CURV-04**: Conditioning of the pullback metric shown as a distribution — histogram, median, 90th and 99th percentile, maximum — with points above a within-config percentile flagged and excluded from the reported `‖H‖` summary rather than averaged in; no fixed absolute threshold. **AND its absolute scale reported alongside the ratio** (`λ_min`, `λ_max` or `det(g)`), because `cond(g)` is scale-invariant and therefore cannot detect a uniformly collapsed metric — a near-non-immersion in every direction at once scores a *perfect* condition number while destroying the `g^-1` contraction
- [x] **CURV-05**: Decoder second derivatives verified non-zero and finite at held-out points, and independently cross-checked at PU scale against finite differences by `derivative_bridge.derivative_agreement`, reported and never gated on
- [x] **CURV-06**: PU curvature compared against the same architecture and training protocol fitted to flat plane, sphere and saddle at matched `chart_dim` and ambient 768, with the statement, alongside the numbers, that this control cannot detect parameterization damage
- [x] **CURV-07**: Whether the measured curvature is a property of the data manifold or an artifact of the fitted decoder, answered on CURV-06's evidence and explicitly conditioned on Phase 3's gate override, never presented as if the parameterization were independently validated
- [x] **CURV-08**: Curvature evaluated only at each point's own chart coordinate as assigned by `model.chart_probs(z).argmax(dim=1)`, never at an extrapolated, interpolated or grid coordinate

### Region Partitioning (REGN)

- [ ] **REGN-01**: Local sample-density measure per point in Isomap coordinate space shown
- [ ] **REGN-02**: Correlation between local density and curvature reported explicitly, before any region split is trusted
- [ ] **REGN-03**: Points partitioned into high- and low-curvature regions by quantile, never by a fixed absolute threshold
- [ ] **REGN-04**: Quantile threshold specified before regional alignment is computed, and that ordering visible in the notebook
- [ ] **REGN-05**: Each region's point count shown, since region size affects every downstream k-NN statistic

### Regional Alignment (MKNN)

- [ ] **MKNN-01**: MKNN score between two row-aligned embeddings computed as the k-normalized size of the k-NN set intersection, matching the origin paper
- [ ] **MKNN-02**: Global crossmodal MKNN number for HSC vs Legacy Survey reproduced and compared against the origin paper's published range
- [ ] **MKNN-03**: Per-region MKNN score for the high- and low-curvature regions shown
- [ ] **MKNN-04**: Each region gets its own permutation null, computed within that region's index set rather than reused from a global null
- [ ] **MKNN-05**: Bootstrap confidence intervals shown on every regional MKNN score
- [ ] **MKNN-06**: Whether the high-vs-low result holds across k = 5, 10, 20, 50 shown
- [ ] **MKNN-07**: Explicit verdict on whether regional difference is distinguishable from noise, where "no detectable difference" is a valid reported outcome
- [ ] **MKNN-08**: Hubness caveat for k-NN-based alignment metrics in high-dimensional spaces stated alongside the results

---

## Future Requirements

Deferred. Tracked but not in the v1.1 roadmap.

### Model Scale (SCALE)

- **SCALE-01**: Intramodal MKNN across a model-size ladder (the origin paper's stronger 28-56% signal), requiring a second model size
- **SCALE-02**: Curvature-stratified alignment compared across the size ladder, testing whether the curvature/alignment relationship itself scales with capacity

### Library Promotion (LIB)

- **LIB-01**: Mean curvature operator promoted into `src/effdim/`, unit-tested against analytically known curvature surfaces
- **LIB-02**: MDS eigenspectrum validity diagnostic promoted into `src/effdim/`
- **LIB-03**: `pyproject.toml` Python floor corrected from `>=3.8` to match the real scikit-learn 1.9.0 requirement of `>=3.11`

### Other Configs (CFG)

- **CFG-01**: Same pipeline run on `desi_*` (HSC vs DESI spectra), the strongest crossmodal claim in the origin paper
- **CFG-02**: Same pipeline run on `physics_*_test`, geometry only, since those configs are unpaired and cannot support MKNN

---

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Modifying `src/effdim/` or `pyproject.toml` | Milestone is notebook-only; promoting the curvature operator needs its own test suite first |
| Null-subtracted "excess MKNN" reporting statistic | Descoped by user decision; raw per-region MKNN plus its own null and CI is the reported quantity. The per-region null itself remains required (MKNN-04) |
| Gaussian curvature, principal curvatures, scalar curvature | Category error at codimension 768−d; each needs an arbitrary choice of normal direction and gives different, sometimes opposite-signed, answers at the same point |
| Fixed absolute curvature threshold | Curvature scale depends on the decoder run and Isomap's arbitrary global scale, so absolute thresholds are not comparable across reruns |
| k-means or other clustering to define curvature regions | Finds breakpoints in the marginal value distribution rather than spatial regions, adds a hyperparameter, and does not address the density confound |
| Landmark or Nyström-approximated Isomap | Landmark approximation distorts the MDS eigenspectrum, which is the exact object SPEC-01 through SPEC-04 exist to inspect |
| Alternative alignment metrics (CKA, mutual information) | Would break comparability with the origin paper's headline numbers; listed as the paper's own future work |
| Evaluating curvature on a dense grid beyond the coordinate support | Neural decoders extrapolate unreliably; off-manifold curvature reflects the decoder, not the data, and would dominate a min/max colour scale |
| Correcting for hubness in MKNN | Known open problem in the alignment literature; flagged as a caveat (MKNN-08) rather than solved |
| Model-size ladder / intramodal MKNN | Deferred to SCALE-01; single model chosen for v1.1 |

---

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| ISO-01 | Phase 1 | Complete |
| ISO-02 | Phase 1 | Complete |
| ISO-03 | Phase 1 | Complete |
| ISO-04 | Phase 1 | Complete |
| ISO-05 | Phase 1 | Complete |
| SPEC-01 | Phase 2 | Complete |
| SPEC-02 | Phase 2 | Complete |
| SPEC-03 | Phase 2 | Complete |
| SPEC-04 | Phase 2 | Complete |
| SPEC-05 | Phase 2 | Complete |
| SPEC-06 | Phase 2 | Pending |
| SPEC-07 | Phase 2 | Pending |
| GEOM-01 | Phase 02.1 | Complete |
| GEOM-02 | Phase 02.1 | Complete |
| GEOM-03 | Phase 02.1 | Complete |
| GEOM-04 | Phase 02.1 | Complete |
| GEOM-05 | Phase 02.1 | Complete |
| CAE-01 | Phase 02.2 | Complete |
| CAE-02 | Phase 02.2 | Complete |
| CAE-03 | Phase 02.2 | Complete |
| CAE-04 | Phase 02.2 | Complete |
| CAE-05 | Phase 02.2 | Complete |
| CAE-06 | Phase 02.2 | Complete |
| CAE-07 | Phase 02.2 | Complete |
| TOPO-01 | Phase 02.4 | Complete |
| TOPO-02 | Phase 02.4 | Complete |
| TOPO-03 | Phase 02.4 | Complete |
| TOPO-04 | Phase 02.4 | Complete |
| TOPO-05 | Phase 02.4 | Complete |
| TOPO-06 | Phase 02.4 | Complete |
| TOPO-07 | Phase 02.4 | Complete |
| TOPO-08 | Phase 02.4 | Complete |
| DEC-01 | Phase 3 | Complete |
| DEC-02 | Phase 3 | Complete |
| DEC-03 | Phase 3 | Complete |
| DEC-04 | Phase 3 | Complete |
| DEC-05 | Phase 3 | Complete |
| CURV-01 | Phase 3 | Complete |
| CURV-02 | Phase 3 | Complete |
| CURV-03 | Phase 3 | Complete |
| CURV-04 | Phase 03.1 | **Closed 2026-08-21** — `chart_curvature.chart_mean_curvature` and `chart_curvature_field` now return `lambda_min`, `lambda_max`, `det_g` and `log10_det_g` alongside `metric_condition_number`, from one reused `torch.linalg.eigvalsh(g)`; pinned by `test_chart_mean_curvature_reports_lambda_min_max_and_det_g`, `test_chart_curvature_field_reports_lambda_min_max_and_det_g` and `test_cond_is_blind_to_uniform_metric_collapse_but_det_g_is_not`. Standing limitation: Phase 3's recorded grids were not re-run, so no existing pre-03.1 record can be re-audited for uniform collapse |
| CURV-05 | Phase 3 | Complete |
| CURV-06 | Phase 3 | Complete (controls run; both curved fixtures failed at d=20) |
| CURV-07 | Phase 3 | Answered (negative, conditioned on the override) |
| CURV-08 | Phase 3 | Complete |
| REGN-01 | Phase 4 | Pending |
| REGN-02 | Phase 4 | Pending |
| REGN-03 | Phase 4 | Pending |
| REGN-04 | Phase 4 | Pending |
| REGN-05 | Phase 4 | Pending |
| MKNN-01 | Phase 4 | Pending |
| MKNN-02 | Phase 4 | Pending |
| MKNN-03 | Phase 4 | Pending |
| MKNN-04 | Phase 4 | Pending |
| MKNN-05 | Phase 4 | Pending |
| MKNN-06 | Phase 4 | Pending |
| MKNN-07 | Phase 4 | Pending |
| MKNN-08 | Phase 4 | Pending |

**Coverage:** v1.1 requirements: 55 total (DATA 5 + ISO 5 + SPEC 7 + GEOM 5 + CAE 7 + DEC 5 + CURV 8 + REGN 5 + MKNN 8). Mapped to phases: 55/55 ✓ · Unmapped: 0 ✓

Phase 1 (Data Loading & Manifold Reconstruction): DATA-01..05, ISO-01..05 (10 requirements)
Phase 2 (Eigenspectrum Audit & Validity Gate): SPEC-01..07 (7 requirements)
Phase 02.1 (Geometry Representation Research): GEOM-01..05 (5 requirements)
Phase 02.2 (Chart Autoencoder Validity Test): CAE-01..07 (7 requirements)
Phase 3 (Decoder & Curvature Field): DEC-01..05, CURV-01..08 (13 requirements)
Phase 4 (Region Partitioning & Regional Alignment / MKNN): REGN-01..05, MKNN-01..08 (13 requirements)

---

## Phase 3 Requirement Re-Mint (2026-08-13, executed 2026-08-17)

DEC-01..05 and CURV-01..08 were written against two premises that are both dead:

1. **"Isomap coordinates."** Phase 2's eigenspectrum gate FAILed (`GATE_VERDICT = FAIL`,
   `m = 0.412071`), invalidating Isomap as the representation. Phase 02.1 selected a graph-native
   replacement; Phase 3 decodes from a Chart Auto-Encoder's per-chart coordinates instead.

2. **"A single global chart."** 02.2's `CAE_VERDICT = FAIL` on T1 (geodesic distortion) and T3
   (held-out reconstruction margin) established that no single global coordinate patch is
   available. Phase 3 works per-chart, with each point measured in the chart the model assigns it.

**All 13 IDs are re-minted with rewritten text under the same `DEC-` / `CURV-` namespace. None was
retired, dropped or re-pointed.** The 2026-07-31 `AMENDED` blockquote, which asked readers to
mentally substitute the replacement representation, is superseded — the text now says what it means.

| ID | What changed | Supporting artifact |
|---|---|---|
| DEC-01 | Isomap coordinates → each point's own chart coordinate; single decoder → `chart_decoders[i]` composed with the shared `embedding_decoder` | `notebooks/pu_manifold/cae.py`, `03-08-SUPPLEMENT-03.md` |
| DEC-02 | One-time verification → a guard that raises | `chart_curvature.assert_c2_activation`, `decoder_curvature.assert_c2_decoder` |
| DEC-03 | One fit → per-`n_charts` held-out reconstruction across the sweep | `03-08-SUMMARY.md` §Task 2 |
| DEC-04 | Added "for every configuration" so the per-dimension distribution is not shown once | `03-08-SUMMARY.md` per-output-dimension table |
| DEC-05 | Recorded seed → recorded seed **plus** a reported multi-seed spread (≥5 roll, 3 PU) | `03-02-SUMMARY.md`, `03-08-SUMMARY.md` |
| CURV-01 | Named `g = J^T J`, `vmap` batching, and the reverse/forward mode toggle agreeing to float64 round-off | `chart_curvature.CURVATURE_MODES`, `03-05-SUMMARY.md` |
| CURV-02 | Added the trace-first-then-project form and the prohibition on materializing a `(D, D)` projector or a full `II` | `chart_curvature.chart_mean_curvature` |
| CURV-03 | Pinned `CURVATURE_CONVENTION = "trace"` so a unit `d`-sphere gives norm `d`, not 1 | `chart_curvature.py`, `synthetic_controls.py` (import-time agreement assert) |
| CURV-04 | "Flagged" → flagged **and excluded from the reported summary**, by a within-config percentile, with no absolute threshold. **Amended 2026-08-18** to also require the metric's absolute scale — the ratio alone is blind to uniform collapse | `curvature_field_pu_run.COND_FLAG_PERCENTILE`, `03-09-SUMMARY.md`, `03-FINDINGS.md` §8.3a |
| CURV-04 (closed 2026-08-21) | The absolute-scale fields (`λ_min`, `λ_max`, `log10 det(g)`) are promoted to the **primary** metric-health diagnostic; `cond(g)` is retained as one reported detail rather than relied on alone — `cond(g)` ranked Phase 3's two uniformly-collapsed seeds ahead of the only healthy one. The promotion is recorded in `03.1-01-PLAN.md`'s `<assumption_delta_decision>` | `03.1-FINDINGS.md` §8 |
| CURV-05 | Added the independent finite-difference cross-check at PU scale, reported and never gated on | `derivative_bridge.derivative_agreement`, `03-09-SUMMARY.md` §Task 2 |
| CURV-06 | "Matched dimension and ambient size" → matched architecture **and training protocol**, plus the mandatory damage caveat beside the numbers | `synthetic_control_run.py`, `03-10-SUMMARY.md` |
| CURV-07 | Added the explicit conditioning on Phase 3's gate override | `03-10-SUMMARY.md` §5 |
| CURV-08 | "At or near the Isomap coordinates" → at each point's own `chart_probs(z).argmax(dim=1)` chart coordinate, never extrapolated, interpolated or gridded | `03-09-SUMMARY.md` §Task 1 |

**Amendment, 2026-08-18.** CURV-04 is **reopened**. Its re-minted text specified conditioning as
a *ratio* only. Phase 3's three-seed spread then measured two fits with excellent `cond(g)`
(7.20e+02, 3.21e+03) whose entire metric spectrum had collapsed to `~1e-07` — a defect the
requirement as written cannot detect, and which no recorded run can be re-audited for, since the
runners store only the ratio. The requirement now also demands the absolute scale. This is a
defect in the re-mint itself, found the day after it was written.

**Outcome recorded honestly:** eleven of the thirteen are Complete. **CURV-04 is Reopened** (above). **CURV-07 is Answered
negatively** — the PU curvature field is *not validated*. The instrument is correct (validated
against analytic curvature at `d=4`, `rho = 0.989`, `R² = 0.980`), but the field **does not
reproduce across seeds**: a 52× range in `‖H‖` median with two of three fields piecewise-constant
on collapsed metrics. See `03-FINDINGS.md` §5-§6 and `03-10-SUMMARY.md` §5.

**Closure, 2026-08-21 (Phase 03.1).** CURV-04 is **closed**. `03.1-01` shipped `λ_min`, `λ_max`,
`det(g)` and `log10 det(g)` in `chart_curvature.py`; `03.1-03` recorded them on one full-cloud row
at the sealed protocol (the sealed `d=20` saddle fit reads `λ_min` median `9.448030e-07`, `λ_max`
median `1.744437e+02`, `log10 det(g)` median `-22.781268` — anisotropic, not uniform, collapse);
`03.1-04` recorded them on every ladder cell. Twelve of the thirteen re-minted DEC/CURV
requirements are now Complete; **CURV-07 remains Answered negatively** (above) — closing CURV-04
does not reopen or revisit CURV-07's answer. **Standing limitation, not work this phase bought:
Phase 3's recorded grids were not re-run with the new instrumentation, so no existing pre-03.1
record can be retroactively audited for uniform collapse.** See `03.1-FINDINGS.md` §8.

---
*Requirements defined: 2026-07-29*
*Last updated: 2026-08-03 — added CAE-01..07 for Phase 02.2 (INSERTED) and corrected coverage arithmetic, which had omitted the five GEOM requirements added for Phase 02.1 (55/55 requirements mapped)*
</content>
