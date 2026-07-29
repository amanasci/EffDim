# Roadmap: EffDim

## Overview

EffDim ships a working `compute_dim` API covering spectral and geometric effective
dimensionality estimators (v1.0 Phase 1, complete). This roadmap was bootstrapped
retroactively on 2026-07-27 from the existing repository state and now carries forward into
milestone v1.1, "PU Manifold Curvature": four new phases (5-8) reconstruct the PU
foundation-model embedding manifold via Isomap, gate its Euclidean-embeddability, fit a
smooth decoder to derive an analytic curvature field, and test whether crossmodal
representational alignment (MKNN) varies with local curvature — entirely inside notebooks
under `notebooks/`, with `src/effdim/` and `pyproject.toml` untouched throughout. v1.0's
Phase 2 (validation hardening) and Phase 4 (CI & packaging) remain outstanding and are
carried forward below as explicitly deferred, not dropped; v1.0's Phase 3 (applied analyses)
is effectively fulfilled by this milestone's notebook deliverables (see Phase 3 note below).

## Milestones

- 🚧 **v1.0 MVP** - Phases 1-4 (Phase 1 complete; Phase 3 superseded by v1.1; Phases 2 & 4
  deferred, not resumed in v1.1)
- 🚧 **v1.1 PU Manifold Curvature** - Phases 5-8 (in progress — planning)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, ...): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- Numbering is continuous across milestones — v1.1 does not restart at 1. It continues from
  v1.0's last phase (4), so v1.1's first phase is Phase 5.

### v1.0 MVP (Phases 1-4)

- [x] **Phase 1: Core ED Library** - Single-call API over spectral + geometric estimators
- [ ] **Phase 2: Validation Hardening** - Estimates checked against known-dimension manifolds
      *(deferred, carried forward from v1.0; not in v1.1 scope)*
- [ ] **Phase 3: Applied Analyses** - Notebooks demonstrating ED on real embedding corpora
      *(superseded — fulfilled by v1.1 Phases 5-8; see Phase 3 detail below)*
- [ ] **Phase 4: CI & Packaging** - Cross-platform test matrix and release pipeline
      *(deferred, carried forward from v1.0; not in v1.1 scope)*

### v1.1 PU Manifold Curvature (Phases 5-8)

- [ ] **Phase 5: Data Loading & Manifold Reconstruction** - Reproducible, row-aligned PU
      subsample loaded and an Isomap fit produced, validated for connectivity and
      `n_neighbors` stability
- [ ] **Phase 6: Eigenspectrum Audit & Validity Gate** - Full classical-MDS eigenspectrum
      audited by hand; a PASS/MARGINAL/FAIL gate freezes the embedding dimension `d`
- [ ] **Phase 7: Decoder & Curvature Field** - C2-smooth decoder trained and its analytic
      mean-curvature field validated against a synthetic-manifold falsification test
- [ ] **Phase 8: Region Partitioning & Regional Alignment (MKNN)** - Density-checked
      high/low-curvature regions compared on crossmodal MKNN alignment against permutation
      nulls and bootstrap CIs

## Phase Details

### Phase 1: Core ED Library
**Goal**: `effdim.compute_dim(array)` returns a dictionary of effective dimensionality estimates
**Depends on**: Nothing (first phase)
**Success Criteria** (what must be TRUE):
  1. A user can call `compute_dim` on a 2D numpy array and get spectral ED metrics
  2. The same call returns geometric/intrinsic dimension estimates
  3. Invalid input (wrong ndim, <2 samples, NaN/inf) raises a clear `ValueError`
  4. Large inputs use randomized SVD rather than full SVD
**Plans**: Complete (pre-GSD)

Plans:
- [x] 01-01: Spectral metrics in `metrics.py`
- [x] 01-02: Geometric estimators in `geometry.py`
- [x] 01-03: Orchestration, validation, SVD dispatch in `api.py`
- [x] 01-04: Benchmarks for runtime and accuracy

### Phase 2: Validation Hardening
**Goal**: ED estimates are demonstrably correct against data of known dimensionality
**Depends on**: Phase 1
**Success Criteria** (what must be TRUE):
  1. Isotropic Gaussian noise in D dims yields estimates approximating D
  2. Swiss Roll and similar manifolds yield estimates approximating their intrinsic dimension
  3. Tolerances are explicit per estimator, so a regression fails the suite
**Plans**: TBD
**Status**: Deferred — carried forward from v1.0. Real outstanding work, not resumed during
v1.1; not silently dropped. Revisit after v1.1 ships.

### Phase 3: Applied Analyses
**Goal**: Notebooks that apply EffDim to real high-dimensional embedding corpora
**Depends on**: Phase 1
**Success Criteria** (what must be TRUE):
  1. A reader can open a notebook and reproduce an ED analysis on a public embedding set
  2. Analyses relate ED estimates to a learned low-dimensional representation
  3. Notebooks install their own heavy dependencies without touching core package deps
**Plans**: TBD
**Status**: Superseded — v1.1 Phases 5-8 fulfill this phase's goal directly: the PU manifold
curvature notebooks under `notebooks/` are a reproducible ED-adjacent analysis of a public
embedding corpus (`UniverseTBD/pu-embeddings`), relate `compute_dim` estimates to a learned
low-dimensional (Isomap + decoder) representation (criterion 2), and install torch/datasets
from within the notebook, never touching `pyproject.toml` (criterion 3). No separate work
plan is needed for this phase; its success criteria are satisfied when Phases 5-8 complete.

### Phase 4: CI & Packaging
**Goal**: Tests run on every push across supported platforms; releases are reproducible
**Depends on**: Phase 2
**Success Criteria** (what must be TRUE):
  1. CI executes the test suite for the standard Python implementation
  2. CI covers any compiled extension across target platforms
  3. A tagged release publishes to PyPI without manual steps
**Plans**: TBD
**Status**: Deferred — carried forward from v1.0. Real outstanding work, not resumed during
v1.1; not silently dropped. Revisit after v1.1 ships (and after Phase 2, which it depends on).

### Phase 5: Data Loading & Manifold Reconstruction
**Goal**: A reproducible, row-aligned 10,000-row subsample of `legacysurvey_dinov3_vitb16` is
loaded and cached, and an Isomap fit on it is validated for connectivity and short-circuit
stability before any eigenspectrum audit is trusted.
**Depends on**: Phase 1 (uses `effdim.compute_dim` as a pre-audit input; already shipped, no
new library work needed)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, ISO-01, ISO-02, ISO-03,
ISO-04, ISO-05
**Success Criteria** (what must be TRUE):
  1. A reader can load exactly the `legacysurvey_dinov3_vitb16` config and get a reproducible,
     assertion-verified row-aligned 10,000-row HSC/Legacy-Survey subsample from a recorded
     seed, without downloading the dataset's other 162 configs (DATA-01, DATA-02, DATA-03)
  2. A reader can see the embedding norm distribution and an explicit, justified statement of
     which distance metric (Euclidean vs cosine) the pipeline uses, and confirm the notebook
     states its own Python 3.11 floor and installs its own dependencies (DATA-04, DATA-05)
  3. A reader can see the k-NN graph's connected-component count and an embedding/eigenspectrum
     stability comparison across at least 3 `n_neighbors` values before Isomap is trusted
     (ISO-01, ISO-02)
  4. A reader can compare `effdim.compute_dim` estimates on the raw 768-d embeddings against
     the dimension the Isomap eigenspectrum suggests, informing the candidate `n_components`
     (ISO-03)
  5. An Isomap fit at n=10,000 completes, is cached, and re-running the notebook reproduces
     identical results from cache, with any config change producing a new cache key
     (ISO-04, ISO-05)
**Plans**: TBD
**Research**: Standard patterns (sklearn `Isomap` internals, classical-MDS mechanics,
connectivity checks) — research pass can be skipped per SUMMARY.md.

### Phase 6: Eigenspectrum Audit & Validity Gate
**Goal**: The Isomap geodesic matrix's full classical-MDS eigenspectrum is audited by hand
(never via the truncated `kernel_pca_.eigenvalues_`), the embedding dimension `d` is frozen,
and a machine-readable PASS/MARGINAL/FAIL gate verdict is written that halts the milestone on
FAIL as a legitimate, complete outcome.
**Depends on**: Phase 5
**Requirements**: SPEC-01, SPEC-02, SPEC-03, SPEC-04, SPEC-05, SPEC-06, SPEC-07
**Success Criteria** (what must be TRUE):
  1. A reader can see the full classical-MDS eigenspectrum computed by manual double-centring
     of `isomap.dist_matrix_`, never from the truncated `kernel_pca_.eigenvalues_` (SPEC-01)
  2. A reader can confirm the leading eigenvalues are large and positive with the steep
     dropoff located, and see the negative-eigenvalue magnitude reported as an explicit
     statistic against a stated, justified threshold (SPEC-02, SPEC-03)
  3. A reader can see the residual-variance-vs-dimension curve with its elbow identified by a
     stated criterion, not eyeballed (SPEC-04)
  4. The chosen embedding dimension `d` is frozen and recorded before any decoder is trained
     (SPEC-05)
  5. A reader can see a PASS/MARGINAL/FAIL verdict written as a machine-readable artifact that
     downstream notebooks check before running; on FAIL the notebook halts with remediation
     options enumerated, and that documented failure is itself a complete, reportable
     milestone outcome (SPEC-06, SPEC-07)
**Plans**: TBD
**Research**: Standard patterns (classical-MDS double-centering, eigenspectrum audit) —
research pass can be skipped per SUMMARY.md. Together with Phase 5, this covers the scope
SUMMARY.md refers to as "the Isomap/gate phase."
**Hard gate**: This phase's terminal artifact is `gate_verdict.json` (PASS/MARGINAL/FAIL). A
FAIL halts the milestone here — Phase 7 must check this artifact before running any expensive
cell and must not proceed on FAIL.

### Phase 7: Decoder & Curvature Field
**Goal**: A C2-smooth decoder is trained from the frozen Isomap coordinates back to the 768-d
embedding, and its analytically-derived mean curvature field is validated against a
synthetic-control falsification test before being trusted as a property of the data manifold
rather than a decoder artifact.
**Depends on**: Phase 6 (requires a PASS or MARGINAL gate verdict and the frozen embedding
dimension `d`; does not proceed on FAIL)
**Requirements**: DEC-01, DEC-02, DEC-03, DEC-04, DEC-05, CURV-01, CURV-02, CURV-03, CURV-04,
CURV-05, CURV-06, CURV-07, CURV-08
**Success Criteria** (what must be TRUE):
  1. A reader can train a decoder mapping Isomap coordinates to the 768-d embedding using a
     C2-smooth activation throughout the forward path, with no ReLU-family activation
     anywhere, reproducible from a recorded torch seed (DEC-01, DEC-02, DEC-05)
  2. A reader can see held-out reconstruction quality as both an aggregate metric and a
     per-output-dimension distribution, not just training loss (DEC-03, DEC-04)
  3. A reader can compute the first and second fundamental forms from the decoder via batched
     `torch.func` autodiff and get the mean curvature vector field and its norm, labelled only
     as a vector norm and never as Gaussian or principal curvature (CURV-01, CURV-02, CURV-03)
  4. A reader can see the metric tensor's conditioning per point with near-singular points
     flagged, verify the decoder's second derivatives are non-zero and finite away from
     training nodes, and confirm curvature is only evaluated at or near actual Isomap
     coordinates, never extrapolated beyond their support (CURV-04, CURV-05, CURV-08)
  5. A reader can compare the PU manifold's curvature against the same decoder architecture
     fitted to known-geometry synthetic manifolds (flat plane, sphere, saddle) at matched
     dimension and ambient size, and tell whether the measured curvature reflects the data
     manifold or is an artifact of the fitted decoder (CURV-06, CURV-07)
**Plans**: TBD
**Research**: Needs a research pass during planning — SUMMARY.md flags the
mean-curvature-in-high-codimension math (first/second fundamental form, `‖H‖` derivation,
`torch.func` batched Jacobian/Hessian via `vmap`) as dense and easy to get subtly wrong on
tensor shapes/index conventions.
**Hard gate**: The synthetic-control falsification test (CURV-06, CURV-07) must complete and
be reported before Phase 8 starts, not run alongside it. If the control shows the decoder
manufactures comparable-magnitude curvature on a known-flat/known-curved synthetic target,
the real-data curvature signal cannot be trusted and Phase 8's regional comparison is not
meaningful.

### Phase 8: Region Partitioning & Regional Alignment (MKNN)
**Goal**: With all upstream hyperparameters (`n_neighbors`, `d`, decoder architecture,
curvature quantile threshold) frozen from Phases 5-7's own diagnostics and the synthetic-
control falsification test complete, points are pre-specified into density-checked high/low
curvature regions and crossmodal MKNN alignment is compared between them against
region-specific permutation nulls and bootstrap confidence intervals.
**Depends on**: Phase 7 (requires the synthetic-control falsification test (CURV-06, CURV-07)
to have already completed — not run in parallel with this phase)
**Requirements**: REGN-01, REGN-02, REGN-03, REGN-04, REGN-05, MKNN-01, MKNN-02, MKNN-03,
MKNN-04, MKNN-05, MKNN-06, MKNN-07, MKNN-08
**Success Criteria** (what must be TRUE):
  1. A reader can see a local sample-density measure per point in Isomap coordinate space and
     its correlation with curvature reported explicitly, before any region split is trusted
     (REGN-01, REGN-02)
  2. A reader can see points partitioned into high- and low-curvature regions by a
     pre-specified quantile threshold (never a fixed absolute value), with the threshold fixed
     before regional alignment is computed and each region's point count shown
     (REGN-03, REGN-04, REGN-05)
  3. A reader can compute the MKNN score between two row-aligned embeddings as the
     k-normalized k-NN intersection size, matching the origin paper, and reproduce a global
     crossmodal HSC-vs-Legacy-Survey MKNN number compared against the origin paper's published
     range (MKNN-01, MKNN-02)
  4. A reader gets a per-region MKNN score for the high- and low-curvature regions, each with
     its own permutation null computed within that region's index set (never reused from a
     global null) and bootstrap confidence intervals (MKNN-03, MKNN-04, MKNN-05)
  5. A reader can see whether the high-vs-low result holds across k = 5, 10, 20, 50, gets an
     explicit verdict on whether the regional difference is distinguishable from noise (where
     "no detectable difference" is a valid reported outcome), and sees the hubness caveat for
     k-NN-based alignment metrics stated alongside the results (MKNN-06, MKNN-07, MKNN-08)
**Plans**: TBD
**Research**: Needs a research pass during planning — SUMMARY.md flags the density-confound
control methodology (correlation check, centroid-distance check, partial regression,
density-matched stratification/null) as original synthesis, not a documented off-the-shelf
recipe.
**Ordering constraint**: Pre-specify the split, then compute. All upstream hyperparameters and
the curvature quantile threshold must be frozen using upstream-only diagnostics from Phases
5-7 *before* the first regional MKNN number is computed — this is a garden-of-forking-paths
guard against post-hoc tuning on a headline effect with thin statistical headroom (0.4-2% in
the origin paper), not an implementation detail to leave to mid-phase judgment.

## Progress

**Execution Order:**
v1.0 phases already exist (1 complete; 2 and 4 deferred; 3 superseded). v1.1 phases execute
in numeric order: 5 → 6 → 7 → 8, gated as described above.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Core ED Library | v1.0 | 4/4 | Complete | pre-GSD |
| 2. Validation Hardening | v1.0 | 0/? | Deferred | - |
| 3. Applied Analyses | v1.0 | 0/? | Superseded by v1.1 | - |
| 4. CI & Packaging | v1.0 | 0/? | Deferred | - |
| 5. Data Loading & Manifold Reconstruction | v1.1 | 0/TBD | Not started | - |
| 6. Eigenspectrum Audit & Validity Gate | v1.1 | 0/TBD | Not started | - |
| 7. Decoder & Curvature Field | v1.1 | 0/TBD | Not started | - |
| 8. Region Partitioning & Regional Alignment (MKNN) | v1.1 | 0/TBD | Not started | - |
