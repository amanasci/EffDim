# Roadmap: EffDim

## Overview

Milestone v1.1, "PU Manifold Curvature": four phases reconstruct the PU foundation-model
embedding manifold via Isomap, gate its Euclidean-embeddability, fit a smooth decoder to
derive an analytic curvature field, and test whether crossmodal representational alignment
(MKNN) varies with local curvature. All work lives in notebooks under `notebooks/`;
`src/effdim/` and `pyproject.toml` are untouched throughout.

Phase numbering restarts at 1 for this milestone. The core library that v1.1 builds on
(`effdim.compute_dim`) shipped before GSD adoption and is recorded under Shipped below.
Two items of unstarted pre-v1.1 work are tracked in Backlog — they are independent of v1.1
and are not numbered phases.

## Milestones

- ✅ **v1.0 MVP** — core `compute_dim` library, shipped pre-GSD (see Shipped)
- 🚧 **v1.1 PU Manifold Curvature** — Phases 1-4 (in progress — planning)

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, ...): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- Numbering restarts at 1 each milestone. v1.1's phases are 1-4.

- [x] **Phase 1: Data Loading & Manifold Reconstruction** - Reproducible, row-aligned PU (completed 2026-07-31)
      subsample loaded and an Isomap fit produced, validated for connectivity and
      `n_neighbors` stability

- [ ] **Phase 2: Eigenspectrum Audit & Validity Gate** - Full classical-MDS eigenspectrum
      audited by hand; a PASS/MARGINAL/FAIL gate freezes the embedding dimension `d`

- [ ] **Phase 02.1: Geometry Representation Research** (INSERTED) - A non-Euclidean-embeddable
      representation identified and justified against the literature, replacing the Isomap
      coordinates that Phase 2's gate invalidated

- [ ] **Phase 3: Decoder & Curvature Field** - C2-smooth decoder trained and its analytic
      mean-curvature field validated against a synthetic-manifold falsification test

- [ ] **Phase 4: Region Partitioning & Regional Alignment (MKNN)** - Density-checked
      high/low-curvature regions compared on crossmodal MKNN alignment against permutation
      nulls and bootstrap CIs

## Phase Details

### Phase 1: Data Loading & Manifold Reconstruction

**Goal**: A reproducible, row-aligned 10,000-row subsample of `legacysurvey_dinov3_vitb16` is
loaded and cached, and an Isomap fit on it is validated for connectivity and short-circuit
stability before any eigenspectrum audit is trusted.
**Depends on**: Nothing (first phase). Calls the shipped `effdim.compute_dim` API as a
pre-audit input; no library work needed.
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, ISO-01, ISO-02, ISO-03, ISO-04, ISO-05
**UI hint**: no
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
**Plans**: 4/4 plans executed

- [x] 01-01-PLAN.md — Scaffold `notebooks/pu_manifold/` and prove the pipeline end-to-end on a
      smoke config: subsample → alignment assert → L2 normalize → npz + joblib cache → read-back
      identity (DATA-01, DATA-03, DATA-05, ISO-05)

- [x] 01-02-PLAN.md — Scale to the real 10,000-row subsample, show the norm distribution and the
      locked metric statement, and derive `n_components` from the `compute_dim` panel
      (DATA-02, DATA-03, DATA-04, ISO-03)

- [x] 01-03-PLAN.md — Two-stage `n_neighbors` sweep: cheap connectivity scan across all six k,
      then full fits at 3-4 surviving k with the three-metric stability table (ISO-01, ISO-02)

- [x] 01-04-PLAN.md — Freeze `k*` by the pre-registered plateau rule, cache the fit, and write
      the Phase 1 → Phase 2 handoff artifact (ISO-04, ISO-05)
**Research**: Standard patterns (sklearn `Isomap` internals, classical-MDS mechanics,
connectivity checks) — research pass can be skipped per SUMMARY.md.

### Phase 2: Eigenspectrum Audit & Validity Gate

**Goal**: The Isomap geodesic matrix's full classical-MDS eigenspectrum is audited by hand
(never via the truncated `kernel_pca_.eigenvalues_`), the embedding dimension `d` is frozen,
and a machine-readable PASS/MARGINAL/FAIL gate verdict is written that halts the milestone on
FAIL as a legitimate, complete outcome.
**Depends on**: Phase 1
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
**Plans**: 2/3 plans executed

- [x] 02-01-PLAN.md — Pre-register the gate constants, then compute the full 10,000-value
      classical-MDS eigenspectrum by hand from the memory-mapped geodesic matrix and reduce it to
      the two negativity statistics `r` and `m` plus the leading-spectrum dropoff
      (SPEC-01, SPEC-02, SPEC-03)

- [x] 02-02-PLAN.md — Both residual-variance curves, the deterministic kneedle elbow with its
      pair-sample stability check, and the one-way freeze of the embedding dimension `d` — the
      nesting slice or the documented re-fit halt (SPEC-04, SPEC-05)

- [ ] 02-03-PLAN.md — The self-contained `gate_verdict_{fit_key}.json`, the copyable downstream
      enforcement block with its three-way FAIL-path self-test, and the phase close-out
      (SPEC-06, SPEC-07)
**Research**: Standard patterns (classical-MDS double-centering, eigenspectrum audit) —
research pass can be skipped per SUMMARY.md. Together with Phase 1, this covers the scope
SUMMARY.md refers to as "the Isomap/gate phase."
**Hard gate**: This phase's terminal artifact is `gate_verdict.json` (PASS/MARGINAL/FAIL). A
FAIL halts the milestone here — Phase 3 must check this artifact before running any expensive
cell and must not proceed on FAIL.

### Phase 02.1: Geometry Representation Research (INSERTED)

**Goal**: A representation of the PU embedding geometry that does not assume Euclidean
embeddability is identified and justified against the literature, and Phase 3 receives a
concrete, argued decision on what it should decode from — replacing the Isomap coordinates
that Phase 2's gate invalidated.

**Why inserted**: Phase 2 measured `GATE_VERDICT = FAIL` on the frozen k*=15 fit —
`m = 0.412071` against a 0.15 MARGINAL bound, with 5029 of 10,000 eigenvalues negative
carrying 41% of absolute eigenvalue mass. `r = 0.052419` passes its own bound, so the failure
is a long diffuse negative tail rather than one short-circuit edge. Four experiments ruled out
numerical error, implementation bug, kNN hop inflation (k ∈ {5,10,15,30}; `m` flat-to-rising
while co-diagnostics confirm the graph genuinely densified), L2 normalization (`m` moves 0.28%
when exactly inverted), absence of manifold structure (local intrinsic dimension stable and
tight at ~20–25, std 2.0), the specific survey column (paired HSC gives `m = 0.4226`), and the
specific 10,000 objects (a ~90% disjoint resample gives `m = 0.411948` with identical
positive/negative counts). See `.planning/phases/02-eigenspectrum-audit-validity-gate/02-FINDINGS.md`.

Phase 3 as originally specified decodes *from the Isomap coordinates* — the direct output of
the step that failed — so its mean-curvature field would conflate real curvature with
parameterization damage, and its own CURV-06/07 synthetic control cannot detect that because a
synthetic manifold passing the gate never reproduces the pathology. This phase exists to
replace that input, not to work around the FAIL.

**Depends on**: Phase 2 (consumes its FAIL verdict, the full eigenspectrum, and the intrinsic
dimension measurements as evidence). Does **not** require a PASS.

**Requirements**: GEOM-01, GEOM-02, GEOM-03, GEOM-04, GEOM-05 (to be added to REQUIREMENTS.md)

**Success Criteria** (what must be TRUE):

  1. A reader can see which manifold-learning methods share Isomap's flat-target assumption and
     would therefore fail the same way on this data, establishing that the failure is a property
     of the method class rather than an implementation choice (GEOM-01)

  2. A reader can see the candidate representations that do **not** assume a flat target,
     surveyed with their assumptions, costs, and what each would require of Phase 3 —
     covering at minimum Riemannian/hyperbolic and product-manifold embeddings, graph-native
     curvature (Ollivier-Ricci, Forman-Ricci) which needs no embedding at all, diffusion maps,
     and pseudo-Euclidean/Krein-space treatments of indefinite similarity (GEOM-02)

  3. A reader can see what 41% negative eigenvalue mass means geometrically, and an explicit
     argued judgment on whether indefinite MDS or distance-matrix correction is principled here
     or merely cosmetic — a method that hides the negativity rather than representing it must be
     named as such (GEOM-03)

  4. A reader gets one recommended representation with stated rationale, the alternatives it was
     chosen over, and the evidence it is expected to be judged against — sufficient for Phase 3
     to be re-specified without re-opening this question (GEOM-04)

  5. A reader can see what the recommendation implies for the working dimension. `D_FROZEN = 5`
     came from a residual-variance elbow that `02-FINDINGS.md` §6.4 flags as suspect against
     three other estimates clustering at 18–25, on the reading that the elbow measured the flat
     embedding's failure rather than the geometry. Whether the chosen representation inherits,
     revises, or discards that dimension must be stated, not left implicit (GEOM-05)

**Notes**: Literature-review and decision phase — no production pipeline code is expected. A
pre-registered 35-model cross-architecture sweep (`sweep/`, `02-MODEL-SWEEP-PREREGISTRATION.md`)
is packaged for external compute and not yet run; its result bears on how general the finding is
but does not block this phase's method survey.

**Plans:** 0 plans

Plans:

- [ ] TBD (run /gsd-plan-phase 02.1 to break down)

### Phase 3: Decoder & Curvature Field

**Goal**: A C2-smooth decoder is trained from the coordinates of Phase 02.1's chosen
representation back to the 768-d embedding, and its analytically-derived mean curvature field is
validated against a synthetic-control falsification test before being trusted as a property of
the data manifold rather than a decoder artifact.

> **AMENDED after Phase 2's FAIL.** This phase originally decoded from the frozen *Isomap*
> coordinates and depended on a PASS or MARGINAL verdict. Phase 2 returned FAIL
> (`m = 0.412071`), so those coordinates are the output of an invalidated step and cannot serve
> as the decoder's input: the pullback metric would conflate real curvature with the
> parameterization damage the decoder absorbs, and CURV-06/07's synthetic control cannot detect
> that, because a synthetic manifold that passes the gate never reproduces the pathology. The
> working dimension is also open — `D_FROZEN = 5` is flagged suspect in `02-FINDINGS.md` §6.4
> against three other estimates clustering at 18–25. Phase 02.1 supplies both the representation
> and the dimension. Re-plan this phase against its output before executing; the DEC and CURV
> requirement text still refers to Isomap coordinates and needs the same amendment.

**Depends on**: Phase 02.1 (requires its chosen representation and working dimension). Phase 2
supplies the eigenspectrum evidence and the FAIL verdict that motivated the change; a PASS is no
longer a precondition, because the flat-embedding path it would have gated is no longer the plan.
**Requirements**: DEC-01, DEC-02, DEC-03, DEC-04, DEC-05, CURV-01, CURV-02, CURV-03, CURV-04, CURV-05, CURV-06, CURV-07, CURV-08
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
be reported before Phase 4 starts, not run alongside it. If the control shows the decoder
manufactures comparable-magnitude curvature on a known-flat/known-curved synthetic target,
the real-data curvature signal cannot be trusted and Phase 4's regional comparison is not
meaningful.

### Phase 4: Region Partitioning & Regional Alignment (MKNN)

**Goal**: With all upstream hyperparameters (`n_neighbors`, `d`, decoder architecture,
curvature quantile threshold) frozen from Phases 1-3's own diagnostics and the synthetic-
control falsification test complete, points are pre-specified into density-checked high/low
curvature regions and crossmodal MKNN alignment is compared between them against
region-specific permutation nulls and bootstrap confidence intervals.
**Depends on**: Phase 3 (requires the synthetic-control falsification test (CURV-06, CURV-07)
to have already completed — not run in parallel with this phase)
**Requirements**: REGN-01, REGN-02, REGN-03, REGN-04, REGN-05, MKNN-01, MKNN-02, MKNN-03, MKNN-04, MKNN-05, MKNN-06, MKNN-07, MKNN-08
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
1-3 *before* the first regional MKNN number is computed — this is a garden-of-forking-paths
guard against post-hoc tuning on a headline effect with thin statistical headroom (0.4-2% in
the origin paper), not an implementation detail to leave to mid-phase judgment.

## Shipped

Work completed before GSD adoption. Not part of any numbered phase sequence.

### Core ED Library (v1.0, pre-GSD)

`effdim.compute_dim(array)` returns a dictionary of effective dimensionality estimates.

- Spectral metrics in `metrics.py`
- Geometric/intrinsic dimension estimators in `geometry.py`
- Orchestration, input validation, SVD dispatch in `api.py`
- Benchmarks for runtime and accuracy

Verified behaviours: spectral and geometric estimates returned for a 2D numpy array; invalid
input (wrong ndim, <2 samples, NaN/inf) raises `ValueError`; large inputs use randomized SVD.

## Backlog

Unstarted pre-v1.1 work. Independent of v1.1 — no v1.1 phase depends on any of it. Not
numbered, so it does not collide with the milestone phase sequence. Promote to a numbered
phase in a future milestone via `/gsd-phase` or `/gsd-review-backlog`.

### Validation Hardening

**Goal**: ED estimates are demonstrably correct against data of known dimensionality
**Success Criteria**:

  1. Isotropic Gaussian noise in D dims yields estimates approximating D
  2. Swiss Roll and similar manifolds yield estimates approximating their intrinsic dimension
  3. Tolerances are explicit per estimator, so a regression fails the suite

### CI & Packaging

**Goal**: Tests run on every push across supported platforms; releases are reproducible
**Success Criteria**:

  1. CI executes the test suite for the standard Python implementation
  2. A tagged release publishes to PyPI without manual steps

**Note**: An earlier version of this item also required CI coverage for a compiled Rust
extension. No Rust source exists in the repository and `pyproject.toml` builds with
setuptools, not maturin — the Rust references in `TODO.md`, `PYPI_SETUP.md`,
`docs/deployment.md` and `.gitignore` are stale. Reconcile those docs before promoting this
item to a phase.

## Progress

**Execution Order:** Phases execute in numeric order: 1 → 2 → 02.1 → 3 → 4, gated as described
above. Phase 02.1 was inserted after Phase 2 returned FAIL; Phase 3 now depends on 02.1's output
rather than on a Phase 2 PASS.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Data Loading & Manifold Reconstruction | v1.1 | 4/4 | Complete    | 2026-07-31 |
| 2. Eigenspectrum Audit & Validity Gate | v1.1 | 2/3 | In Progress|  |
| 02.1. Geometry Representation Research (INSERTED) | v1.1 | 0/TBD | Not started | - |
| 3. Decoder & Curvature Field | v1.1 | 0/TBD | Not started | - |
| 4. Region Partitioning & Regional Alignment (MKNN) | v1.1 | 0/TBD | Not started | - |
