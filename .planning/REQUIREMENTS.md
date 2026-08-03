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

- [ ] **CAE-01**: Gate metrics and numeric PASS/FAIL thresholds pre-registered and committed before any CAE fit runs, with the ordering proved by git ancestry rather than asserted
- [ ] **CAE-02**: Chart Auto-Encoder trained on the frozen Phase 1 10,000-row normalized subsample and reproducible from recorded seeds, with the full architecture named — initial encoder to R^l, N over-specified chart encoders into chart spaces `(0,1)^d`, per-chart decoders, one shared embedding decoder, and a partition-of-unity chart predictor — trained under the paper's loss with Lipschitz regularization on chart-encoder spectral norms and FPS-seeded per-chart pre-training
- [ ] **CAE-03**: Held-out reconstruction error shown as an aggregate metric plus a per-output-dimension distribution against a matched-capacity single-chart autoencoder control and the classical-MDS/Isomap reconstruction, so a CAE that merely ties the control cannot be read as a success
- [ ] **CAE-04**: Chart-transition cycle residual `R_cycle` shown over held-out points
- [ ] **CAE-05**: Chart count obtained a posteriori after weight-decay pruning by a stated decoder-weight-norm tolerance, with the over-specified initial `N`, the surviving count, its stability across seeds, and the paper's unfaithfulness and coverage measures shown
- [ ] **CAE-06**: Decoders trained with a C2-smooth activation rather than the reference implementation's ReLU, since DEC-02 and CURV-01..03 need a non-zero second derivative, with the substitution and any reconstruction cost stated; the compact-manifold-with-positive-reach assumption behind the paper's Thm 2 recorded as an unverified assumption about the PU point cloud, not a verified one
- [ ] **CAE-07**: `cae_verdict_{fit_key}.json` written as a machine-readable PASS/FAIL artifact carrying every pre-registered metric and threshold, checked by downstream notebooks before any expensive cell; on FAIL the notebook halts, findings are documented, and that documented failure is itself a complete milestone outcome

### Decoder (DEC)

> **AMENDED 2026-07-31 (Phase 2 FAIL).** DEC-01 and the CURV requirements below are written against *Isomap coordinates*. Those are the output of the step Phase 2 invalidated. Phase 02.1 selects the replacement representation; re-read "Isomap coordinates" throughout this section and the next as "the coordinates of the representation chosen in Phase 02.1," and re-plan Phase 3 against it. The requirement *intent* — a C2-smooth decoder whose Jacobian yields an analytic curvature field, falsified against a synthetic control — is unchanged.

- [ ] **DEC-01**: Train a decoder mapping Isomap coordinates to the original 768-d embedding, using a C2-smooth activation throughout the forward path
- [ ] **DEC-02**: Verify no ReLU-family activation appears anywhere in the decoder, since its second derivative is identically zero
- [ ] **DEC-03**: Held-out reconstruction quality shown, not just training loss
- [ ] **DEC-04**: Both an aggregate reconstruction metric and a per-output-dimension distribution shown, so good averages cannot hide subset failures
- [ ] **DEC-05**: Decoder training reproducible from a recorded torch seed

### Curvature Field (CURV)

- [ ] **CURV-01**: First fundamental form computed from the decoder Jacobian via `torch.func` autodiff, batched over all points rather than looped
- [ ] **CURV-02**: Second fundamental form computed as the normal-projected ambient Hessian of the decoder
- [ ] **CURV-03**: Mean curvature **vector** field and its norm shown, labelled as a vector norm and never as Gaussian or principal curvature
- [ ] **CURV-04**: Conditioning of the metric tensor shown, with near-singular points flagged, so a non-immersion point cannot silently corrupt the field
- [ ] **CURV-05**: Decoder's second derivatives verified non-zero and finite away from training nodes
- [ ] **CURV-06**: PU manifold's curvature compared against the same decoder architecture fitted to known-geometry synthetic manifolds (flat plane, sphere, saddle) at matched dimension and ambient size
- [ ] **CURV-07**: Whether the measured curvature is a property of the data manifold or an artifact of the fitted decoder shown, on the evidence of CURV-06
- [ ] **CURV-08**: Curvature only evaluated at or near the actual Isomap coordinates, never extrapolated beyond their support

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
| CAE-01 | Phase 02.2 | Pending |
| CAE-02 | Phase 02.2 | Pending |
| CAE-03 | Phase 02.2 | Pending |
| CAE-04 | Phase 02.2 | Pending |
| CAE-05 | Phase 02.2 | Pending |
| CAE-06 | Phase 02.2 | Pending |
| CAE-07 | Phase 02.2 | Pending |
| DEC-01 | Phase 3 | Pending |
| DEC-02 | Phase 3 | Pending |
| DEC-03 | Phase 3 | Pending |
| DEC-04 | Phase 3 | Pending |
| DEC-05 | Phase 3 | Pending |
| CURV-01 | Phase 3 | Pending |
| CURV-02 | Phase 3 | Pending |
| CURV-03 | Phase 3 | Pending |
| CURV-04 | Phase 3 | Pending |
| CURV-05 | Phase 3 | Pending |
| CURV-06 | Phase 3 | Pending |
| CURV-07 | Phase 3 | Pending |
| CURV-08 | Phase 3 | Pending |
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
*Requirements defined: 2026-07-29*
*Last updated: 2026-08-03 — added CAE-01..07 for Phase 02.2 (INSERTED) and corrected coverage arithmetic, which had omitted the five GEOM requirements added for Phase 02.1 (55/55 requirements mapped)*
</content>
