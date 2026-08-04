# Roadmap: EffDim

## Overview

Milestone v1.1, "PU Manifold Curvature": four phases reconstruct the PU foundation-model embedding manifold via Isomap, gate its Euclidean-embeddability, fit a smooth decoder to derive an analytic curvature field, and test whether crossmodal representational alignment (MKNN) varies with local curvature. All work lives in notebooks under `notebooks/`; `src/effdim/` and `pyproject.toml` are untouched throughout.

Phase numbering restarts at 1 for this milestone. The core library v1.1 builds on (`effdim.compute_dim`) shipped before GSD adoption and is recorded under Shipped below. Two items of unstarted pre-v1.1 work are tracked in Backlog — independent of v1.1, not numbered phases.

## Milestones

- ✅ **v1.0 MVP** — core `compute_dim` library, shipped pre-GSD (see Shipped)
- 🚧 **v1.1 PU Manifold Curvature** — Phases 1-4 (in progress — planning)

## Phases

**Phase Numbering:** Integer phases (1, 2, 3, ...) = planned milestone work. Decimal phases (2.1, 2.2) = urgent insertions (marked INSERTED). Numbering restarts at 1 each milestone; v1.1's phases are 1-4.

- [x] **Phase 1: Data Loading & Manifold Reconstruction** — Reproducible, row-aligned PU subsample loaded and an Isomap fit produced, validated for connectivity and `n_neighbors` stability (completed 2026-07-31)
- [ ] **Phase 2: Eigenspectrum Audit & Validity Gate** — Full classical-MDS eigenspectrum audited by hand; a PASS/MARGINAL/FAIL gate freezes the embedding dimension `d`
- [ ] **Phase 02.1: Geometry Representation Research** (INSERTED) — A non-Euclidean-embeddable representation identified and justified against the literature, replacing the Isomap coordinates that Phase 2's gate invalidated
- [ ] **Phase 02.2: Chart Autoencoder Validity Test** (INSERTED) — The Chart Auto-Encoder method (arXiv:1912.10094) is empirically validity-tested on the PU data behind a PASS/FAIL gate that decides whether Phase 3 proceeds
- [ ] **Phase 3: Decoder & Curvature Field** — C2-smooth decoder trained and its analytic mean-curvature field validated against a synthetic-manifold falsification test
- [ ] **Phase 4: Region Partitioning & Regional Alignment (MKNN)** — Density-checked high/low-curvature regions compared on crossmodal MKNN alignment against permutation nulls and bootstrap CIs

## Phase Details

### Phase 1: Data Loading & Manifold Reconstruction

**Goal**: A reproducible, row-aligned 10,000-row subsample of `legacysurvey_dinov3_vitb16` is loaded/cached, and an Isomap fit on it is validated for connectivity and short-circuit stability before any eigenspectrum audit is trusted.
**Depends on**: Nothing (first phase). Calls the shipped `effdim.compute_dim` API as a pre-audit input; no library work needed.
**Requirements**: DATA-01..05, ISO-01..05
**UI hint**: no
**Success Criteria**:

  1. `legacysurvey_dinov3_vitb16` loads without downloading the other 162 configs, into a reproducible, assertion-verified row-aligned 10,000-row HSC/Legacy-Survey subsample from a recorded seed (DATA-01..03)
  2. Embedding norm distribution and a justified Euclidean-vs-cosine metric choice shown; notebook states its Python 3.11 floor and installs its own deps (DATA-04, DATA-05)
  3. k-NN graph's connected-component count and an embedding/eigenspectrum stability comparison across >= 3 `n_neighbors` values shown before Isomap is trusted (ISO-01, ISO-02)
  4. `effdim.compute_dim` estimates on raw 768-d embeddings compared against the Isomap-eigenspectrum-suggested dimension, informing candidate `n_components` (ISO-03)
  5. n=10,000 Isomap fit completes, is cached, and re-running the notebook reproduces identical results from cache, any config change producing a new cache key (ISO-04, ISO-05)

**Plans**: 4/4 executed

- [x] 01-01-PLAN.md — Scaffold `notebooks/pu_manifold/`, prove pipeline end-to-end on a smoke config: subsample -> alignment assert -> L2 normalize -> npz+joblib cache -> read-back identity (DATA-01, DATA-03, DATA-05, ISO-05)
- [x] 01-02-PLAN.md — Scale to the real 10,000-row subsample, show norm distribution + locked metric statement, derive `n_components` from `compute_dim` panel (DATA-02..04, ISO-03)
- [x] 01-03-PLAN.md — Two-stage `n_neighbors` sweep: cheap connectivity scan across all six k, then full fits at 3-4 surviving k with three-metric stability table (ISO-01, ISO-02)
- [x] 01-04-PLAN.md — Freeze `k*` by pre-registered plateau rule, cache the fit, write Phase 1->2 handoff artifact (ISO-04, ISO-05)

**Research**: Standard patterns (sklearn `Isomap` internals, classical-MDS mechanics, connectivity checks) — research pass skippable per SUMMARY.md.

### Phase 2: Eigenspectrum Audit & Validity Gate

**Goal**: The Isomap geodesic matrix's full classical-MDS eigenspectrum is audited by hand (never via truncated `kernel_pca_.eigenvalues_`), embedding dimension `d` frozen, and a machine-readable PASS/MARGINAL/FAIL gate verdict written that halts the milestone on FAIL as a legitimate outcome.
**Depends on**: Phase 1
**Requirements**: SPEC-01..07
**Success Criteria**:

  1. Full classical-MDS eigenspectrum computed by manual double-centring of `isomap.dist_matrix_`, never from truncated `kernel_pca_.eigenvalues_` (SPEC-01)
  2. Leading eigenvalues confirmed large/positive with steep dropoff located; negative-eigenvalue magnitude reported against a stated, justified threshold (SPEC-02, SPEC-03)
  3. Residual-variance-vs-dimension curve's elbow identified by a stated criterion, not eyeballed (SPEC-04)
  4. Chosen embedding dimension `d` frozen and recorded before any decoder is trained (SPEC-05)
  5. PASS/MARGINAL/FAIL verdict written as a machine-readable artifact downstream notebooks check before running; on FAIL the notebook halts with remediation options enumerated (SPEC-06, SPEC-07)

**Plans**: 2/3 executed

- [x] 02-01-PLAN.md — Pre-register gate constants, compute the full 10,000-value classical-MDS eigenspectrum by hand from the memory-mapped geodesic matrix, reduce to `r`/`m` plus leading-spectrum dropoff (SPEC-01..03)
- [x] 02-02-PLAN.md — Both residual-variance curves, the deterministic kneedle elbow with pair-sample stability check, one-way freeze of embedding dimension `d` (SPEC-04, SPEC-05)
- [ ] 02-03-PLAN.md — Self-contained `gate_verdict_{fit_key}.json`, copyable downstream enforcement block with three-way FAIL-path self-test, phase close-out (SPEC-06, SPEC-07)

**Research**: Standard patterns (classical-MDS double-centering, eigenspectrum audit) — skippable per SUMMARY.md. Together with Phase 1, covers what SUMMARY.md calls "the Isomap/gate phase."
**Hard gate**: Terminal artifact is `gate_verdict.json` (PASS/MARGINAL/FAIL). A FAIL halts the milestone here — Phase 3 must check this artifact before running any expensive cell and must not proceed on FAIL.

### Phase 02.1: Geometry Representation Research (INSERTED)

**Goal**: A representation of the PU embedding geometry that does not assume Euclidean embeddability is identified and justified against the literature; Phase 3 receives a concrete decision on what it should decode from — replacing the Isomap coordinates Phase 2's gate invalidated.

**Why inserted**: Phase 2 measured `GATE_VERDICT = FAIL` on the frozen k*=15 fit — `m = 0.412071` against a 0.15 MARGINAL bound, 5029 of 10,000 eigenvalues negative carrying 41% of absolute eigenvalue mass. `r = 0.052419` passes its own bound, so the failure is a long diffuse negative tail, not one short-circuit edge. Four experiments ruled out numerical error, implementation bug, kNN hop inflation (k in {5,10,15,30}; `m` flat-to-rising while co-diagnostics confirm the graph genuinely densified), L2 normalization (`m` moves 0.28% when exactly inverted), absence of manifold structure (local intrinsic dimension stable/tight ~20-25, std 2.0), the specific survey column (paired HSC gives `m = 0.4226`), and the specific 10,000 objects (a ~90% disjoint resample gives `m = 0.411948` with identical positive/negative counts). See `02-FINDINGS.md`.

Phase 3 as originally specified decodes *from the Isomap coordinates* — the direct output of the failed step — so its mean-curvature field would conflate real curvature with parameterization damage, and CURV-06/07's synthetic control cannot detect that (a synthetic manifold passing the gate never reproduces the pathology). This phase replaces that input, not works around the FAIL.

**Depends on**: Phase 2 (consumes its FAIL verdict, full eigenspectrum, intrinsic dimension measurements as evidence). Does **not** require a PASS.
**Requirements**: GEOM-01..05
**Success Criteria**:

  1. Manifold-learning methods sharing Isomap's flat-target assumption identified, establishing the failure as a property of the method class, not an implementation choice (GEOM-01)
  2. Candidate representations that do **not** assume a flat target surveyed with assumptions, costs, Phase 3 implications — at minimum Riemannian/hyperbolic and product-manifold embeddings, graph-native curvature (Ollivier-Ricci, Forman-Ricci, needing no embedding), diffusion maps, pseudo-Euclidean/Krein-space treatments (GEOM-02)
  3. What 41% negative eigenvalue mass means geometrically shown, with an argued judgment on whether indefinite MDS or distance-matrix correction is principled here or merely cosmetic (GEOM-03)
  4. One recommended representation delivered with stated rationale, alternatives it was chosen over, and evidence it is expected to be judged against (GEOM-04)
  5. What the recommendation implies for the working dimension shown. `D_FROZEN = 5` came from a residual-variance elbow `02-FINDINGS.md` §6.4 flags as suspect against three other estimates clustering at 18-25; whether the chosen representation inherits, revises, or discards that dimension must be stated (GEOM-05)

**Notes**: Literature-review/decision phase — no production pipeline code expected. A pre-registered 35-model cross-architecture sweep (`sweep/`, `02-MODEL-SWEEP-PREREGISTRATION.md`) is packaged for external compute, not yet run; its result bears on generality but doesn't block this phase's method survey.

**Plans:** 3/4 executed. **Wave dependencies:** Wave 1 = `02.1-01` alone (the fork; blocking decision checkpoint). Wave 2 = `02.1-02` and `02.1-03` in parallel, both depending on 01. Wave 3 = `02.1-04`, depending on all three.

**Cross-cutting constraints**: The coordinate-producing vs graph-native fork resolved in wave 1 gates everything after it (graph-native branch re-opens CURV-01..03, intrinsic Ricci != extrinsic mean-curvature vector). No package installs anywhere in this phase — all seven candidate libraries returned SUS from the legitimacy checker. Plan 03 requires the gitignored Phase 2 caches (`isomap_43cf438bc944c509.joblib`, 1.55 GiB, and the spectrum `.npz`) — not reproducible by any plan here, the runner halts rather than regenerating (would change provenance). Evaluation criterion and dimension rule pre-registered in wave 1, before any candidate compared, with ordering proved by git ancestry rather than asserted.

Plans:

- [x] 02.1-01-PLAN.md — Pre-register every decision rule, resolve the coordinate-producing vs graph-native fork with both branches' Phase 3 consequences by requirement ID; two commits with git-proved ordering, ratified at a blocking decision checkpoint (GEOM-02, GEOM-04)
- [x] 02.1-02-PLAN.md — Flat-target class-membership analysis + six-family non-flat-target candidate survey with assumptions, costs at n=10,000, maturity, Phase 3 demand (GEOM-01, GEOM-02)
- [x] 02.1-03-PLAN.md — Tested probe module with Wave 0 synthetic fixtures, then measured run over frozen Phase 2 cache: correction blindness on three spectra, pseudo-Euclidean (p,q) distortion ladder against Phase 2's own 200,000-pair sample, delta-hyperbolicity against tree and flat-Euclidean anchors (GEOM-03..05)
- [ ] 02.1-04-PLAN.md — Terminal artifact: geometric reading of 41% negative mass, argued correction-vs-retention judgment, one recommended representation with alternatives/evidence, re-derived working dimension, machine-readable Phase 3 handoff (GEOM-03..05)

### Phase 02.2: Chart Autoencoder Validity Test (INSERTED)

**Goal**: The Chart Auto-Encoder method of arXiv:1912.10094 (Schonsheck, Chen, Lai) is trained on the frozen Phase 1 10,000-row PU subsample and put through a pre-registered empirical validity test whose machine-readable PASS/FAIL verdict decides whether Phase 3 may decode a curvature field from a CAE representation, or the milestone remains at the phase-2 stage.

**Why inserted**: Phase 2 measured `GATE_VERDICT = FAIL` on the frozen k*=15 fit — `m = 0.412071` against a 0.15 MARGINAL bound, ~41% of absolute eigenvalue mass negative, robust across k in {5,10,30}, the survey column, a ~90% disjoint resample, and exact normalization inversion. See `02-FINDINGS.md`.

What failed is the assumption of a *global* Euclidean target for the geodesic metric. CAE assumes only that *local charts* are Euclidean — it is the assumption-relief the FAIL calls for, and a coordinate-producing candidate under Phase 02.1's coordinate-producing vs graph-native fork.

The paper's guarantee is a universal manifold approximation theorem (Thm 2): for a compact d-dimensional data manifold with reach tau, an epsilon-faithful representation exists with L > d charts, latent space homeomorphic to the manifold. Thm 1 is the contrast that motivates charts at all — a plain single-chart autoencoder *cannot* epsilon-faithfully represent non-contractible topology.

A theorem about compact manifolds with positive reach is not evidence about this point cloud, so this is a test, not an adoption. Phase 3 is expensive and its synthetic control (CURV-06/07) provably cannot detect a bad input parameterization — a synthetic manifold that passes the Phase 2 gate never reproduces the pathology. The input must be validated before Phase 3, not by it.

**Depends on**: Phase 02.1 (consumes its fork resolution and non-flat-target candidate survey; CAE sits on the coordinate-producing side of that fork) and Phase 2 (consumes its FAIL verdict, full eigenspectrum, intrinsic-dimension measurements as evidence, and the frozen Phase 1 subsample cache). Does **not** require a Phase 2 PASS — same posture as 02.1.
**Requirements**: CAE-01..07
**UI hint**: no
**Success Criteria**:

  1. Gate metrics and their numeric PASS/FAIL thresholds pre-registered and committed **before** any CAE fit runs, with the ordering proved by git ancestry rather than asserted — the `02-REFIT-PREREGISTRATION.md` precedent (CAE-01)
  2. A Chart Auto-Encoder trained on the frozen Phase 1 10,000-row normalized subsample, reproducible from recorded seeds: initial encoder E from R^768 to R^l with l near 2d per the Nash-Kuiper minimal near-isometric motivation, N over-specified chart encoders E_alpha into chart spaces (0,1)^d, per-chart decoders D_alpha, one shared embedding decoder D back to R^768, and a chart predictor P emitting partition-of-unity probabilities p_alpha; trained under the paper's loss (eq. 3, `min_alpha e_alpha` minus the cross-entropy term on softmax(-e)) with Lipschitz regularization on chart-encoder spectral norms (eq. 4) and FPS-seeded per-chart pre-training (eq. 5) (CAE-02)
  3. Held-out reconstruction error reported as both an aggregate metric and a per-output-dimension distribution, against two stated baselines at matched capacity: a plain single-chart autoencoder, and the classical-MDS/Isomap reconstruction at the working dimension. A CAE that only ties the single-chart control has not demonstrated that charts bought anything (CAE-03)
  4. Chart-transition cycle residual R_cycle (eq. 8) reported over held-out points — decoded data re-encoded through a second chart — as the direct measurement of whether the chart atlas is consistent where charts overlap (CAE-04)
  5. Chart count obtained **a posteriori**: the over-specified initial N stated, unused charts decayed under weight decay plus regularization and pruned by a stated decoder-weight-norm tolerance, the surviving count reported and its stability across seeds shown. Topology preservation shown by the paper's own empirical measures — unfaithfulness (distance of latent-sampled generations from the training set) and coverage (fraction of training modes hit) (CAE-05)
  6. Two stated deviations/assumptions carried forward for Phase 3, each with its consequence: (a) the reference implementation is TensorFlow with ReLU activations, while DEC-02 and CURV-01..03 require a C2-smooth activation throughout the decoder because ReLU's second derivative is identically zero — so per-chart and embedding decoders are trained with a C2-smooth substitute (tanh, softplus, or SiLU), the substitution and any reconstruction cost it carries reported; (b) Thm 2 covers *compact* manifolds with positive reach — PU point-cloud compactness is an assumption recorded, not verified (CAE-06)
  7. A machine-readable `cae_verdict_{fit_key}.json` written on the `gate_verdict_{fit_key}.json` pattern, carrying a `CAE_VERDICT` field of PASS or FAIL plus every pre-registered metric and threshold. Downstream notebooks check it before running any expensive cell. On FAIL the notebook halts, `02.2-FINDINGS.md` records the finding, and that documented failure is itself a complete, reportable milestone outcome (CAE-07)

**Notes**: CAE validates topology and geometry preservation *of a reconstruction*. It does not adjudicate the geometric reading of the 41% negative eigenvalue mass — plan 02.1-04's terminal artifact still owns that judgment, and this phase does not reopen it. No new package installs: implementation is in torch (`torch==2.13.0+cpu`, already vetted and pinned in `notebooks/requirements-notebooks.txt`); the paper's TensorFlow reference implementation is not installed and not a dependency. Cache posture inherited from 02.1 plan 03: the gitignored Phase 1/2 caches (the subsample `.npz`, and `isomap_43cf438bc944c509.joblib` at 1.55 GiB for the Isomap baseline in criterion 3) are not reproducible here; a runner that cannot find them halts rather than regenerating, because regenerating changes provenance. Working dimension d is not inherited from `D_FROZEN = 5`, flagged suspect in `02-FINDINGS.md` §6.4 against three estimates clustering at 18-25; d comes from Phase 02.1's GEOM-05 re-derivation, and the chart dimension used must be stated.

**Hard gate**: Terminal artifact is `cae_verdict.json` (PASS/FAIL). PASS unblocks Phase 3, which then decodes from the CAE representation. FAIL blocks Phase 3, the findings are documented, and the milestone remains at the phase-2 stage — a legitimate and complete outcome, not an error to be worked around.

**Plans**: 5/6 plans executed

Plans:
**Wave 1**

- [x] 02.2-01-PLAN.md — Pre-register the gate metrics, constants and three numeric thresholds; blocking D-10 ratification before any fit (CAE-01)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02.2-02-PLAN.md — Tracer: the Chart Auto-Encoder architecture and eq. 3 loss proven end to end into a manifest-guarded verdict artifact (CAE-02, CAE-06, CAE-07)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02.2-03-PLAN.md — eq. 4 Lipschitz penalty, FPS-seeded eq. 5 pre-training, the three-way stopping rule, and the matched-capacity baseline trainers (CAE-02, CAE-03, CAE-05, CAE-06)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 02.2-04-PLAN.md — A-posteriori chart pruning, eq. 8 cycle residual, eqs. 20/21 unfaithfulness and coverage, and the geodesic distortion on the global embedding (CAE-03, CAE-04, CAE-05, CAE-07)
- [x] 02.2-05-PLAN.md — Training runner with run-time ancestry and cache guards; execute 3 seeds plus the ReLU control and 4 baselines (CAE-02, CAE-03, CAE-05, CAE-06)

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 02.2-06-PLAN.md — Evaluate, write cae_verdict_{fit_key}.json and 02.2-FINDINGS.md, PASS-only Phase 3 handoff, blocking phase gate (CAE-03..07)

### Phase 3: Decoder & Curvature Field

**Goal**: A C2-smooth decoder trained from Phase 02.1's chosen representation back to the 768-d embedding, its analytically-derived mean curvature field validated against a synthetic-control falsification test before being trusted as a property of the data manifold rather than a decoder artifact.

> **AMENDED after Phase 2's FAIL.** This phase originally decoded from the frozen *Isomap* coordinates and depended on a PASS or MARGINAL verdict. Phase 2 returned FAIL (`m = 0.412071`), so those coordinates are the output of an invalidated step: the pullback metric would conflate real curvature with parameterization damage, undetectable by CURV-06/07's synthetic control (a synthetic manifold passing the gate never reproduces the pathology). Working dimension also open — `D_FROZEN = 5` flagged suspect in `02-FINDINGS.md` §6.4 against three estimates clustering at 18-25. Phase 02.1 supplies both the representation and the dimension. Re-plan against its output; DEC/CURV requirement text still refers to Isomap coordinates and needs the same amendment. **Further amended after Phase 02.2's insertion.** Phase 02.2 now tests whether a Chart Auto-Encoder representation is a valid input for this decoding step; Phase 3 does not start until `cae_verdict.json` reads PASS.

**Depends on**: Phase 02.1 (requires its chosen representation and working dimension) and Phase 02.2 **PASS** (a precondition — Phase 3 decodes from the CAE representation and must check `cae_verdict.json` before running any expensive cell; on FAIL, Phase 3 stays blocked and the milestone remains at the phase-2 stage). Phase 2 supplies the eigenspectrum evidence and FAIL verdict that motivated the change; a Phase 2 PASS is no longer a precondition.
**Requirements**: DEC-01..05, CURV-01..08
**Success Criteria**:

  1. A decoder maps the Phase 02.2-validated Chart Auto-Encoder chart representation to the 768-d embedding using a C2-smooth activation throughout, no ReLU-family activation, reproducible from a recorded torch seed (DEC-01, DEC-02, DEC-05)
  2. Held-out reconstruction quality shown as both aggregate metric and per-output-dimension distribution, not just training loss (DEC-03, DEC-04)
  3. First and second fundamental forms computed via batched `torch.func` autodiff, yielding the mean curvature vector field and its norm, labelled only as a vector norm and never as Gaussian or principal curvature (CURV-01..03)
  4. Metric tensor's per-point conditioning shown with near-singular points flagged; decoder second derivatives verified non-zero and finite away from training nodes; curvature confirmed evaluated only at/near actual Isomap coordinates, never extrapolated (CURV-04, CURV-05, CURV-08)
  5. PU manifold's curvature compared against the same decoder architecture fitted to known-geometry synthetic manifolds (flat plane, sphere, saddle) at matched dimension/ambient size (CURV-06, CURV-07)

**Plans**: TBD
**Research**: Needs a research pass during planning — SUMMARY.md flags the mean-curvature-in-high-codimension math (first/second fundamental form, `‖H‖` derivation, `torch.func` batched Jacobian/Hessian via `vmap`) as dense, easy to get subtly wrong on tensor shapes/index conventions.
**Hard gate**: The synthetic-control falsification test (CURV-06, CURV-07) must complete and be reported before Phase 4 starts, not run alongside it.

### Phase 4: Region Partitioning & Regional Alignment (MKNN)

**Goal**: With all upstream hyperparameters (`n_neighbors`, `d`, decoder architecture, curvature quantile threshold) frozen from Phases 1-3's own diagnostics and the synthetic-control falsification test complete, points are pre-specified into density-checked high/low curvature regions and crossmodal MKNN alignment compared between them against region-specific permutation nulls and bootstrap CIs.
**Depends on**: Phase 3 (requires the synthetic-control falsification test (CURV-06, CURV-07) to have already completed)
**Requirements**: REGN-01..05, MKNN-01..08
**Success Criteria**:

  1. Local sample-density measure per point in Isomap coordinate space and its correlation with curvature shown explicitly, before any region split is trusted (REGN-01, REGN-02)
  2. Points partitioned into high/low-curvature regions by a pre-specified quantile threshold (never a fixed absolute value), fixed before regional alignment is computed, each region's point count shown (REGN-03..05)
  3. MKNN score between two row-aligned embeddings computed as k-normalized k-NN intersection size, matching the origin paper; global crossmodal HSC-vs-Legacy-Survey MKNN number reproduced and compared against the origin paper's published range (MKNN-01, MKNN-02)
  4. Per-region MKNN score for high/low-curvature regions, each with its own permutation null computed within that region's index set (never reused from a global null) and bootstrap CIs (MKNN-03..05)
  5. Whether the high-vs-low result holds across k = 5, 10, 20, 50 shown, with an explicit verdict on whether the regional difference is distinguishable from noise ("no detectable difference" is a valid outcome), hubness caveat for k-NN-based alignment metrics stated alongside results (MKNN-06..08)

**Plans**: TBD
**Research**: Needs a research pass during planning — SUMMARY.md flags the density-confound control methodology (correlation check, centroid-distance check, partial regression, density-matched stratification/null) as original synthesis, not a documented off-the-shelf recipe.
**Ordering constraint**: Pre-specify the split, then compute. All upstream hyperparameters and curvature quantile threshold must be frozen using upstream-only diagnostics from Phases 1-3 *before* the first regional MKNN number is computed — a garden-of-forking-paths guard against post-hoc tuning on a headline effect with thin statistical headroom (0.4-2% in the origin paper).

## Shipped

Work completed before GSD adoption. Not part of any numbered phase sequence.

### Core ED Library (v1.0, pre-GSD)

`effdim.compute_dim(array)` returns a dictionary of effective dimensionality estimates. Spectral metrics in `metrics.py`; geometric/intrinsic dimension estimators in `geometry.py`; orchestration/validation/SVD dispatch in `api.py`; benchmarks for runtime and accuracy.

Verified behaviours: spectral and geometric estimates returned for a 2D numpy array; invalid input (wrong ndim, <2 samples, NaN/inf) raises `ValueError`; large inputs use randomized SVD.

## Backlog

Unstarted pre-v1.1 work. Independent of v1.1 — no v1.1 phase depends on any of it. Not numbered, so it does not collide with the milestone phase sequence. Promote to a numbered phase in a future milestone via `/gsd-phase` or `/gsd-review-backlog`.

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

**Note**: An earlier version of this item also required CI coverage for a compiled Rust extension. No Rust source exists in the repository and `pyproject.toml` builds with setuptools, not maturin — the Rust references in `TODO.md`, `PYPI_SETUP.md`, `docs/deployment.md` and `.gitignore` are stale. Reconcile those docs before promoting this item to a phase.

## Progress

**Execution Order:** Phases execute in numeric order: 1 -> 2 -> 02.1 -> 02.2 -> 3 -> 4, gated as described above. Phase 02.1 was inserted after Phase 2 returned FAIL; Phase 3 now depends on 02.1's output rather than on a Phase 2 PASS. Phase 02.2 was inserted after 02.1 to empirically test the Chart Auto-Encoder representation before Phase 3 commits to decoding from it; Phase 3 now also depends on a Phase 02.2 PASS.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Data Loading & Manifold Reconstruction | v1.1 | 4/4 | Complete | 2026-07-31 |
| 2. Eigenspectrum Audit & Validity Gate | v1.1 | 2/3 | In Progress | |
| 02.1. Geometry Representation Research (INSERTED) | v1.1 | 3/4 | In Progress | |
| 02.2. Chart Autoencoder Validity Test (INSERTED) | v1.1 | 5/6 | In Progress|  |
| 3. Decoder & Curvature Field | v1.1 | 0/TBD | Not started | - |
| 4. Region Partitioning & Regional Alignment (MKNN) | v1.1 | 0/TBD | Not started | - |
</content>
