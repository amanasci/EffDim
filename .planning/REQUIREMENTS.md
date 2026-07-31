# Requirements: EffDim v1.1 — PU Manifold Curvature

**Defined:** 2026-07-29
**Core Value:** One call over an `(n_samples, n_features)` array returns a comparable panel of effective dimensionality estimates, so a researcher can see how spectral and geometric notions of dimension agree or disagree on the same data.

**Milestone Goal:** Reconstruct the PU embedding manifold via Isomap, build a smooth decoder
parameterization of it, compute mean curvature, and test whether foundation-model
representational alignment (MKNN) varies with local curvature.

**Scope note:** This milestone is notebook-only. `src/effdim/` and `pyproject.toml` are not
modified. Requirements below describe observable behaviours of the notebooks under
`notebooks/`, plus the local, non-installed helper package `notebooks/pu_manifold/`.

---

## v1.1 Requirements

### Data Loading (DATA)

- [x] **DATA-01**: A reader can load exactly one config of `UniverseTBD/pu-embeddings`
      (`legacysurvey_dinov3_vitb16`) without downloading the other 162 configs

- [x] **DATA-02**: A reader gets a reproducible 10,000-row subsample from the 101,725
      available rows, controlled by an explicit, recorded seed

- [x] **DATA-03**: The `_hsc` and `_legacysurvey` arrays are guaranteed row-aligned after
      subsampling, enforced by an assertion rather than by convention

- [x] **DATA-04**: A reader can see the embedding norm distribution and an explicit statement
      of which metric (Euclidean vs cosine) the pipeline uses and why

- [x] **DATA-05**: The notebook states its own Python floor (3.11) and installs its
      dependencies from a notebook cell, leaving `pyproject.toml` untouched

### Manifold Reconstruction (ISO)

- [x] **ISO-01**: A reader can see the connected-component count of the k-NN graph before
      Isomap is fitted, so silent bridging by `_fix_connected_components` cannot pass unnoticed

- [x] **ISO-02**: A reader can see whether the embedding and eigenspectrum are stable across
      at least three `n_neighbors` values, detecting short-circuit edges

- [x] **ISO-03**: A reader can compare `effdim.compute_dim` estimates on the raw 768-d
      embeddings against the dimension the Isomap eigenspectrum suggests

- [ ] **ISO-04**: An Isomap fit at n=10,000 completes and is cached, so downstream work never
      re-pays the fit cost

- [x] **ISO-05**: A reader can re-run any notebook and get identical results from the cache,
      with a config change producing a new cache key rather than silently reusing a stale one

### Eigenspectrum Audit and Validity Gate (SPEC)

- [ ] **SPEC-01**: A reader can see the **full** classical-MDS eigenspectrum, computed from
      `isomap.dist_matrix_` by manual double-centring — not from the truncated
      `kernel_pca_.eigenvalues_` attribute, which cannot show the negative tail

- [ ] **SPEC-02**: A reader can confirm the leading eigenvalues are large and positive and see
      where the steep dropoff occurs

- [ ] **SPEC-03**: A reader can see the negative-eigenvalue magnitude as an explicit reported
      statistic, with a stated and justified threshold

- [ ] **SPEC-04**: A reader can see the residual-variance-vs-dimension curve with the elbow
      identified by a stated criterion, not eyeballed

- [ ] **SPEC-05**: The chosen embedding dimension `d` is frozen and recorded before any
      decoder is trained

- [ ] **SPEC-06**: A PASS / MARGINAL / FAIL verdict is written as a machine-readable artifact
      that downstream notebooks check before running

- [ ] **SPEC-07**: On FAIL, the notebook halts with remediation options enumerated, and that
      documented failure is itself a complete, reportable milestone outcome

### Decoder (DEC)

- [ ] **DEC-01**: A reader can train a decoder mapping Isomap coordinates to the original
      768-d embedding, using a C2-smooth activation throughout the forward path

- [ ] **DEC-02**: A reader can verify no ReLU-family activation appears anywhere in the
      decoder, since its second derivative is identically zero

- [ ] **DEC-03**: A reader can see held-out reconstruction quality, not just training loss
- [ ] **DEC-04**: A reader can see both an aggregate reconstruction metric and a
      per-output-dimension distribution, so good averages cannot hide subset failures

- [ ] **DEC-05**: Decoder training is reproducible from a recorded torch seed

### Curvature Field (CURV)

- [ ] **CURV-01**: A reader can compute the first fundamental form from the decoder Jacobian
      via `torch.func` autodiff, batched over all points rather than looped

- [ ] **CURV-02**: A reader can compute the second fundamental form as the normal-projected
      ambient Hessian of the decoder

- [ ] **CURV-03**: A reader gets the mean curvature **vector** field and its norm, with the
      output labelled as a vector norm and never as Gaussian or principal curvature

- [ ] **CURV-04**: A reader can see the conditioning of the metric tensor, with near-singular
      points flagged, so a non-immersion point cannot silently corrupt the field

- [ ] **CURV-05**: A reader can verify the decoder's second derivatives are non-zero and
      finite away from training nodes

- [ ] **CURV-06**: A reader can compare the PU manifold's curvature against the same decoder
      architecture fitted to known-geometry synthetic manifolds (flat plane, sphere, saddle)
      at matched dimension and ambient size

- [ ] **CURV-07**: A reader can tell whether the measured curvature is a property of the data
      manifold or an artifact of the fitted decoder, on the evidence of CURV-06

- [ ] **CURV-08**: Curvature is only evaluated at or near the actual Isomap coordinates, never
      extrapolated beyond their support

### Region Partitioning (REGN)

- [ ] **REGN-01**: A reader can see a local sample-density measure per point in Isomap
      coordinate space

- [ ] **REGN-02**: A reader can see the correlation between local density and curvature
      reported explicitly, before any region split is trusted

- [ ] **REGN-03**: A reader can partition points into high- and low-curvature regions by
      quantile, never by a fixed absolute threshold

- [ ] **REGN-04**: The quantile threshold is specified before regional alignment is computed,
      and that ordering is visible in the notebook

- [ ] **REGN-05**: A reader can see each region's point count, since region size affects every
      downstream k-NN statistic

### Regional Alignment (MKNN)

- [ ] **MKNN-01**: A reader can compute the MKNN score between two row-aligned embeddings as
      the k-normalized size of the k-NN set intersection, matching the origin paper

- [ ] **MKNN-02**: A reader can reproduce a global crossmodal MKNN number for HSC vs Legacy
      Survey and compare it against the origin paper's published range

- [ ] **MKNN-03**: A reader gets a per-region MKNN score for the high- and low-curvature regions
- [ ] **MKNN-04**: Each region gets its own permutation null, computed within that region's
      index set rather than reused from a global null

- [ ] **MKNN-05**: A reader gets bootstrap confidence intervals on every regional MKNN score
- [ ] **MKNN-06**: A reader can see whether the high-vs-low result holds across k = 5, 10, 20, 50
- [ ] **MKNN-07**: A reader gets an explicit verdict on whether regional difference is
      distinguishable from noise, where "no detectable difference" is a valid reported outcome

- [ ] **MKNN-08**: A reader can see the hubness caveat for k-NN-based alignment metrics in
      high-dimensional spaces stated alongside the results

---

## Future Requirements

Deferred. Tracked but not in the v1.1 roadmap.

### Model Scale (SCALE)

- **SCALE-01**: Intramodal MKNN across a model-size ladder (the origin paper's stronger
  28–56% signal), requiring a second model size

- **SCALE-02**: Curvature-stratified alignment compared across the size ladder, testing
  whether the curvature/alignment relationship itself scales with capacity

### Library Promotion (LIB)

- **LIB-01**: Mean curvature operator promoted into `src/effdim/`, unit-tested against
  analytically known curvature surfaces

- **LIB-02**: MDS eigenspectrum validity diagnostic promoted into `src/effdim/`
- **LIB-03**: `pyproject.toml` Python floor corrected from `>=3.8` to match the real
  scikit-learn 1.9.0 requirement of `>=3.11`

### Other Configs (CFG)

- **CFG-01**: Same pipeline run on `desi_*` (HSC vs DESI spectra), the strongest crossmodal
  claim in the origin paper

- **CFG-02**: Same pipeline run on `physics_*_test`, geometry only, since those configs are
  unpaired and cannot support MKNN

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
| ISO-04 | Phase 1 | Pending |
| ISO-05 | Phase 1 | Complete |
| SPEC-01 | Phase 2 | Pending |
| SPEC-02 | Phase 2 | Pending |
| SPEC-03 | Phase 2 | Pending |
| SPEC-04 | Phase 2 | Pending |
| SPEC-05 | Phase 2 | Pending |
| SPEC-06 | Phase 2 | Pending |
| SPEC-07 | Phase 2 | Pending |
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

**Coverage:**

- v1.1 requirements: 43 total (corrected from an earlier miscount of 42 — the seven category
  subtotals stated at the top of this document, DATA 5 + ISO 5 + SPEC 7 + DEC 5 + CURV 8 +
  REGN 5 + MKNN 8, sum to 43, matching the 43 checklist items actually listed above)

- Mapped to phases: 43/43 ✓
- Unmapped: 0 ✓

Phase 1 (Data Loading & Manifold Reconstruction): DATA-01..05, ISO-01..05 (10 requirements)
Phase 2 (Eigenspectrum Audit & Validity Gate): SPEC-01..07 (7 requirements)
Phase 3 (Decoder & Curvature Field): DEC-01..05, CURV-01..08 (13 requirements)
Phase 4 (Region Partitioning & Regional Alignment / MKNN): REGN-01..05, MKNN-01..08
(13 requirements)

---
*Requirements defined: 2026-07-29*
*Last updated: 2026-07-29 after v1.1 phase renumber to 1-4 (43/43 requirements mapped)*
