# Roadmap: EffDim

## Overview

Milestone v1.1, "PU Manifold Curvature": four phases reconstruct the PU foundation-model embedding manifold via Isomap, gate its Euclidean-embeddability, fit a smooth decoder to derive an analytic curvature field, and test whether crossmodal representational alignment (MKNN) varies with local curvature. All work lives in notebooks under `notebooks/`; `src/effdim/` and `pyproject.toml` are untouched throughout.

Phase numbering restarts at 1 for this milestone. The core library v1.1 builds on (`effdim.compute_dim`) shipped before GSD adoption and is recorded under Shipped below. Two items of unstarted pre-v1.1 work are tracked in Backlog — independent of v1.1, not numbered phases.

## Milestones

- ✅ **v1.0 MVP** — core `compute_dim` library, shipped pre-GSD (see Shipped)
- 🚧 **v1.1 PU Manifold Curvature** — Phases 1-4 (in progress — planning)

## Phases

**Phase Numbering:** Integer phases (1, 2, 3, ...) = planned milestone work. Decimal phases (2.1, 2.2) = urgent insertions (marked INSERTED). Numbering restarts at 1 each milestone; v1.1's phases are 1-4.

> **THE PHASE-2 STAGE IS ON HOLD (2026-08-12).** Architecture selection is tabled by user
> decision; the **CAE** is the substrate carried into Phase 3, which is now the milestone's
> active work. Phases 02.3, 02.5, 02.6 and 02.7 stop where they stand — see
> `phases/02-eigenspectrum-audit-validity-gate/02-NOTE-phase-2-stage-on-hold.md` for the
> decision, the evidence for and against the CAE, exactly where each phase stopped, the
> carried debt, and what ends the hold. **No sealed verdict is reopened or reinterpreted.**
> Phase 3's hard gate names a PASS and no PASS exists, so Phase 3 starts on a **deliberate
> override**, not a satisfied precondition; the note states the consequence that override
> carries.

- [x] **Phase 1: Data Loading & Manifold Reconstruction** — Reproducible, row-aligned PU subsample loaded and an Isomap fit produced, validated for connectivity and `n_neighbors` stability (completed 2026-07-31)
- [x] **Phase 2: Eigenspectrum Audit & Validity Gate** — Full classical-MDS eigenspectrum audited by hand; a PASS/MARGINAL/FAIL gate freezes the embedding dimension `d` (sealed 2026-08-05, **GATE_VERDICT = FAIL**, `r=0.052419`, `m=0.412071`, `d_frozen=5` — see Phase Details)
- [x] **Phase 02.1: Geometry Representation Research** (INSERTED) — A non-Euclidean-embeddable representation identified and justified against the literature, replacing the Isomap coordinates that Phase 2's gate invalidated (sealed 2026-08-05, **GEOM-04 = Ollivier-Ricci graph-native**; the pre-registered falsifier fired and overturned the coordinate-producing fork — see Phase Details)
- [x] **Phase 02.2: Chart Autoencoder Validity Test** (INSERTED) — The Chart Auto-Encoder method (arXiv:1912.10094) empirically validity-tested on the PU data behind a pre-registered PASS/FAIL gate (completed 2026-08-04, **CAE_VERDICT = FAIL** — see Phase Details)
- [ ] **Phase 02.3: Chart Auto-Encoder Iteration** (INSERTED, proposed — not yet planned; **ON HOLD 2026-08-12**) — Investigate why the CAE failed (candidate axes: chart count, chart latent dimension, training budget/epochs, loss weighting, Lipschitz penalty schedule) and produce a fresh, separately-ratified pre-registration before any new fit
- [x] **Phase 02.4: Topological Auto-Encoder Validity Test** (INSERTED) — The Topological Auto-Encoder (Moor et al., arXiv:1906.00722) implemented and put through a pre-registered validity gate on the PU data (sealed 2026-08-07, **TOPOAE_VERDICT = FAIL** — both *global*-scoped gates failed, the *local*-scoped gate passed; see Phase Details and `02.4-FINDINGS.md`)
- [ ] **Phase 02.5: Local Curvature Feasibility & CAE Local Re-Gate** (INSERTED, not yet planned) — Establish empirically whether a local second fundamental form is estimable in the PU regime, then pre-register and run a *locally*-scoped gate on the Chart Auto-Encoder; resolves Phase 3's blocking dependency, which currently names a global-scoped PASS that no method has produced. **Stage 1 measured 2026-08-09, `CURVATURE_VERDICT = FAIL` under `02.5-PREREGISTRATION-AMENDMENT-01.md`'s 5-seed rule. PAUSED at plan 02.5-09's blocking checkpoint after the CAE chart decoder failed its Swiss roll check; stage-2 plans 02.5-10..13 waited on Phase 02.6 — that dependency is now SATISFIED (2026-08-11): `02.5-10` receives a ranking, a derivative-usability table, and the separating experiment's result, not a substrate — see Phase Details and `02.6-FINDINGS-02.md`** **ON HOLD 2026-08-12 at 9/13 plans with `02.5-09`'s Task 3 checkpoint still open; plans `02.5-10`..`13` are not scheduled. WR-01/02/03 (`derivative_bridge.py`) were routed here and now land on whoever next thresholds on the bridge — see `02-NOTE-phase-2-stage-on-hold.md`.**
- [x] **Phase 02.6: Decoder Substrate Screening** (INSERTED) — Screen candidate decoder substrates against the Swiss roll admission gate (known analytic `H`) and promote at most ONE to a full pre-registered validity gate; blocks Phase 02.5 stage 2, whose plan 02.5-10 is the last point the substrate can change before its thresholds seal. **HALTED 2026-08-10 at 3/6 plans, then REPLANNED onto persistent-homology agreement (D-01) and completed 2026-08-11 at 15/15 replan plans — no substrate promoted, none eliminated, ranking axis carries two named confounds. See `02.6-FINDINGS.md`, `02.6-FINDINGS-02.md`.** **Its selection question is TABLED 2026-08-12 — the substrate was chosen by user decision (CAE), not by this screening.**
- [ ] **Phase 02.7: Manifold-Template Inference Front End** (INSERTED) — A screening rule that infers a named manifold template from a point cloud, behind D-01/D-03's joint decision and D-16's in-library positive controls. **ON HOLD 2026-08-12 at 10/12 plans: `02.7-10` Tasks 2/3 (the ~17h benchmark grid) unrun, `02.7-11`/`02.7-12` unstarted, and `notebooks/02.7_swiss_roll_template_check.ipynb` prints 1 of 4 read-out lines true (GMST local-dispersion instability plus inflated banded β₀; both controls fail their labels). Does not block Phase 3.**
- [ ] **Phase 3: Decoder & Curvature Field** — Per-point mean-curvature field via autodiff through a C2-smooth CAE chart decoder, Swiss roll first, synthetic control last. **ACTIVE from 2026-08-12 on a deliberate override of its own PASS precondition — see `02-NOTE-phase-2-stage-on-hold.md` §3.**
- [x] **Phase 03.1: Decoder Metric Regularization** (INSERTED) — Add scale-aware and second-order priors to the CAE training objective and measure whether they fix the metric pathologies Phase 3 diagnosed. **Phase 3's field failed its own three-seed spread (52× range, two of three fields piecewise-constant on uniformly collapsed metrics); the training objective constrains no decoder derivative at any order, and `cond(g)` is scale-invariant so it cannot even detect the collapse.** **Sealed 2026-08-21: `scale` fully repairs the metric (`log10_det_g` -83.9 → +0.037, negative reconstruction cost) but only partially and non-seed-consistently moves ordering (`rho` -0.122 → +0.116); `christoffel` alone does not demonstrate its own mechanism under this ladder. Necessary but not sufficient — CURV-04 is closed, Phase 4 stays blocked. See Phase Details and `03.1-FINDINGS.md`.**
- [x] **Phase 4: Region Partitioning & Regional Alignment (MKNN)** — Density-checked high/low-curvature regions compared on crossmodal MKNN alignment against permutation nulls and bootstrap CIs (completed 2026-08-24)

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

**Plans**: 3/3 plans executed

- [x] 02-01-PLAN.md — Pre-register gate constants, compute the full 10,000-value classical-MDS eigenspectrum by hand from the memory-mapped geodesic matrix, reduce to `r`/`m` plus leading-spectrum dropoff (SPEC-01..03)
- [x] 02-02-PLAN.md — Both residual-variance curves, the deterministic kneedle elbow with pair-sample stability check, one-way freeze of embedding dimension `d` (SPEC-04, SPEC-05)
- [x] 02-03-PLAN.md — Self-contained `gate_verdict_{fit_key}.json`, copyable downstream enforcement block with three-way FAIL-path self-test, phase close-out (SPEC-06, SPEC-07)

**Research**: Standard patterns (classical-MDS double-centering, eigenspectrum audit) — skippable per SUMMARY.md. Together with Phase 1, covers what SUMMARY.md calls "the Isomap/gate phase."
**Hard gate**: Terminal artifact is `gate_verdict.json` (PASS/MARGINAL/FAIL). A FAIL halts the milestone here — Phase 3 must check this artifact before running any expensive cell and must not proceed on FAIL.

**Sealed 2026-08-05 — `GATE_VERDICT = FAIL`.** `r = 0.052419` (clears its 0.10 bound), `m = 0.412071` (fails even the 0.15 MARGINAL bound), `d_frozen = 5`, `fit_key = 43cf438bc944c509`. 5,029 of 10,000 eigenvalues negative carrying 41% of absolute eigenvalue mass. Remediation option 3 accepted — the documented FAIL is the milestone's reported outcome for this fit; options 1 and 2 were tested anyway and both returned FAIL. The hard gate fired as designed: Phase 3 is not planned against Isomap coordinates, and Phases 02.1/02.2/02.3 exist because of this verdict. `d_frozen = 5` is the dimension of record but is flagged suspect in `02-FINDINGS.md` §6.4 against three estimates clustering at 18–25 — **not to be inherited downstream**. The `02-03` phase-sealing checkpoint was approved on the surviving `gate_verdict_43cf438bc944c509.json` rather than a fresh notebook re-run, because quick task `260801-ovf` (commit `8958488`) deleted `notebooks/01_manifold_and_gate.ipynb` during the checkpoint hold; 8 of 10 verification steps re-verified, the 2 unrepeatable ones named in `02-03-SUMMARY.md`. Notebook recoverable at `a2ca11f`.

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

**Plans:** 4/4 executed. **Wave dependencies:** Wave 1 = `02.1-01` alone (the fork; blocking decision checkpoint). Wave 2 = `02.1-02` and `02.1-03` in parallel, both depending on 01. Wave 3 = `02.1-04`, depending on all three.

**Cross-cutting constraints**: The coordinate-producing vs graph-native fork resolved in wave 1 gates everything after it (graph-native branch re-opens CURV-01..03, intrinsic Ricci != extrinsic mean-curvature vector). No package installs anywhere in this phase — all seven candidate libraries returned SUS from the legitimacy checker. Plan 03 requires the gitignored Phase 2 caches (`isomap_43cf438bc944c509.joblib`, 1.55 GiB, and the spectrum `.npz`) — not reproducible by any plan here, the runner halts rather than regenerating (would change provenance). Evaluation criterion and dimension rule pre-registered in wave 1, before any candidate compared, with ordering proved by git ancestry rather than asserted.

Plans:

- [x] 02.1-01-PLAN.md — Pre-register every decision rule, resolve the coordinate-producing vs graph-native fork with both branches' Phase 3 consequences by requirement ID; two commits with git-proved ordering, ratified at a blocking decision checkpoint (GEOM-02, GEOM-04)
- [x] 02.1-02-PLAN.md — Flat-target class-membership analysis + six-family non-flat-target candidate survey with assumptions, costs at n=10,000, maturity, Phase 3 demand (GEOM-01, GEOM-02)
- [x] 02.1-03-PLAN.md — Tested probe module with Wave 0 synthetic fixtures, then measured run over frozen Phase 2 cache: correction blindness on three spectra, pseudo-Euclidean (p,q) distortion ladder against Phase 2's own 200,000-pair sample, delta-hyperbolicity against tree and flat-Euclidean anchors (GEOM-03..05)
- [x] 02.1-04-PLAN.md — Terminal artifact: geometric reading of 41% negative mass, argued correction-vs-retention judgment, one recommended representation with alternatives/evidence, re-derived working dimension, machine-readable Phase 3 handoff (GEOM-03..05)

**Outcome (sealed 2026-08-05): `GEOM-04 = Ollivier-Ricci discrete curvature on the frozen k\*=15 graph`,** Forman-Ricci as combinatorial cross-check. Retention stance: retain. Working dimension: **none — inapplicable on a per-edge branch**; `D_FROZEN = 5` discarded as inapplicable, not declared wrong, with the coordinate branch's re-derived `(8,0)` preserved.

The pre-registered falsifier **fired**, overturning the coordinate-producing fork verdict: condition (a) trips wide (`delta_rel_max = 0.383921` past a `0.360433` flat anchor, threshold `0.036043`), and condition (b) trips under `02.1-AMENDMENT-02.md`'s amended reading — requiring the ladder's drop be realisable in a decoder-consumable form. Krein `(40,25)` won the pre-registered criterion at `0.065190` (18.4% below the q=0 best) but was rejected: by user directive (`02.1-AMENDMENT-01.md` §1.3), and on measurement — a pre-registered decoder-side check gave it only **+1.44% / +0.10%** held-out reconstruction improvement against the **+18.37%** its distance advantage promised, with the matched-width signature control **negative** (−2.07% / −3.61%). Four structurally unrelated algorithm families wall at ~0.0796 distortion; metric SMACOF reaching it with no eigendecomposition and no PSD constraint shows the binding constraint is flatness of the target space, not the algorithm.

**Consequence:** per `02.1-FORK.md`'s graph-native accounting, all thirteen DEC/CURV requirements move — 9 dropped (DEC-01..05, CURV-01/02/04/05), 3 rewritten (CURV-03, CURV-06, CURV-07), 1 amended (CURV-08), 0 unchanged. Adopting this branch requires a REQUIREMENTS.md traceability update. Two SUS installs (`GraphRicciCurvature`, `POT`) stay behind blocking gates; `networkx` is now present at 3.6.1, so the branch costs two packages, not the three `02.1-FORK.md` recorded.

**Flagged for re-opening — read before inheriting this recommendation.** The decoder check surfaced, as an unplanned observation, that held-out reconstruction was nearly **decoupled** from the distance-distortion statistic this phase ranked representations by: classical `(40,0)`, worst distortion of the three ladder rungs at `0.179641`, reconstructed *best* of all four arms on both preprocessings. Distortion spanned 2.75×; MSE spanned ~6%. One seed was run, so few-percent gaps cannot be separated from initialisation noise and decoder capacity saturation is not excluded — an observation, not a verdict. `02.1-AMENDMENT-02.md` §6.4 records it as the strongest reason to doubt the amendment; §6.5 names the seed-sensitivity study that would settle it. **Not run.**

**Machinery validated:** `notebooks/02.1_swiss_roll_geometry_probes_check.ipynb` runs `pu_manifold.geometry_probes` unchanged on a developable surface under Phase 2's own thresholds — `m = 0.027292` (bound 0.05, PU 0.412071), `r = 0.002202` (bound 0.10), geodesics reconstructed to 0.65% distortion in exactly 2 dimensions, hand double-centring matching sklearn's Isomap to 1.8e-13. The 41% negative mass is a property of the data, not the instrument.

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

**Outcome (2026-08-04): `CAE_VERDICT = FAIL`.** T1 geodesic distortion 0.296981 (threshold `<0.15`, ~2x over, worse than both of 02.1's classical reference rungs) and T3 held-out reconstruction margin (worst-case ratio) 3.586350 (threshold `<0.90` — the CAE reconstructs materially worse than every matched-capacity control, not a near-miss) both failed; T2 chart-transition cycle residual passed (1.089366 `<2.0`). No `cae_handoff_{fit_key}.json` was written; no downstream artifact fell back to the 02.1 Krein representation (D-02). Full record: `02.2-FINDINGS.md`. **User decision at the phase gate: iterate.** The FAIL stands on the record exactly as measured — not reinterpreted, softened, or revisited. Rather than adopting the 02.1 Krein representation or stopping here, the user elected to open a follow-up investigation into *why* the CAE failed (candidate axes: chart count, chart latent dimension `D_CHART=20`, training budget/epochs, loss weighting, the Lipschitz penalty schedule) before any new pre-registration. Any changed constant requires a fresh, separately-ratified pre-registration and a full re-run per the sealed pre-registration's own prohibition (§6) — this FAIL is permanent history regardless of what that investigation finds. Phase 3's representation question remains open pending it; **Phase 02.2 does not hand off to Phase 3.** The follow-up was proposed as Phase 02.3, then superseded on 2026-08-05 when the user chose Phase 02.4 (Topological Auto-Encoder) as the next attempt instead; 02.3 stays on the roadmap unretracted as an available fallback.

**Plans**: 6/6 plans executed

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

- [x] 02.2-06-PLAN.md — Evaluate, write cae_verdict_{fit_key}.json and 02.2-FINDINGS.md, PASS-only Phase 3 handoff, blocking phase gate (CAE-03..07) — **CAE_VERDICT = FAIL**; user decision: iterate (see Outcome above)

### Phase 02.3: Chart Auto-Encoder Iteration (INSERTED, proposed — not yet planned)

**Goal**: Investigate why the pre-registered Chart Auto-Encoder fit failed (`02.2-06`'s `CAE_VERDICT = FAIL`) and produce a fresh, separately-ratified pre-registration before any new fit is run — never a quiet edit of the sealed `02.2-PREREGISTRATION.md`.

**Why proposed**: At the 02.2-06 phase gate the user chose to iterate rather than adopt 02.1's Krein representation or stop and report. The FAIL itself is not revisited by this phase — `02.2-FINDINGS.md` and `cae_verdict_43cf438bc944c509.json` stand as the permanent record of that measured outcome regardless of what this phase finds. Candidate investigation axes named at the decision point: chart count (`N_CHARTS_INIT=16`, all 16 survived pruning yet T3 failed by a wide margin), chart latent dimension (`D_CHART=20`), training budget/epochs, loss weighting between the reconstruction and cross-entropy terms (eq. 3), and the Lipschitz penalty schedule (eq. 4, `LIP_WEIGHT`/`LIP_EVERY_N_STEPS`).

**Depends on**: Phase 02.2 (consumes its FAIL finding, measured metrics, and the eight cached fit artifacts as evidence — does not require a PASS, same posture as 02.1/02.2's own insertion).
**Requirements**: Not yet defined — needs a `/gsd-discuss-phase` or `/gsd-phase` session to scope before planning.
**Status**: Proposed only. Not discussed, not researched, not planned. **Superseded as the milestone's next step by Phase 02.4 (2026-08-05)** — at the 02.1 seal gate the user elected to try a Topological Auto-Encoder rather than iterate on the CAE. Phase 3 no longer depends on this phase. It remains on the roadmap as an available line of attack if 02.4 also fails; nothing in it has been retracted, and the candidate investigation axes above stand as written.

**Plans**: TBD

### Phase 02.4: Topological Auto-Encoder Validity Test (INSERTED)

**UI hint**: no

**Goal**: The Topological Auto-Encoder (Moor, Horn, Rieck, Borgwardt — "Topological Autoencoders", ICML 2020, arXiv:1906.00722) is implemented and trained on the frozen Phase 1 10,000-row PU subsample, and put through a pre-registered validity gate whose machine-readable verdict decides whether Phase 3 may decode a curvature field from a TopoAE representation.

**Why inserted**: Chosen at the 02.1 seal gate in preference to 02.3's CAE iteration. TopoAE attacks the failure from a different direction than every method this milestone has tried: it optimises a **topological** signature-matching loss — 0-dimensional persistence pairs of the minibatch distance matrices, compared between input space and latent space — rather than preserving distances.

**Why this does not simply contradict 02.1.** Phase 02.1 sealed with its falsifier fired against the coordinate-producing branch, and TopoAE *is* coordinate-producing with a Euclidean latent, so the tension must be stated rather than glossed. The reconciliation: 02.1's ~0.0796 wall was measured on `DISTORTION = median(|d²_rep − d²_geo|/d²_geo)` — a **distance-preservation** statistic — and every arm that hit it was optimising distance preservation in some form (classical MDS, SMACOF, Isomap, Laplacian eigenmaps, LLE, ambient Riemannian). TopoAE does not attempt distance preservation and does not claim it. A method with a different objective is not obviously bound by a wall measured on the objective it declines to pursue. Two consequences follow and must be honoured:

  1. **TopoAE must not be scored primarily on `DISTORTION`.** Doing so would rank it on an axis it never optimised — the same category error the pre-registration forbids for graph-native candidates. Its gate needs topology-preservation measures, and 02.1's distortion number belongs beside them as context, not as the verdict.
  2. **If TopoAE reaches PASS, `02.1-AMENDMENT-02.md`'s falsifier firing should be revisited by a dated amendment** — a coordinate-producing representation clearing a validity gate is close to the pre-registration's *symmetric* falsifier, which overturns graph-native when "a coordinate candidate's DISTORTION drops materially below the classical failure at a tractable working dimension." Not identical, since the statistic differs, and that difference is exactly what the amendment would have to argue.

**Depends on**: Phase 1 (the frozen 10,000-row subsample cache), Phase 2 (its FAIL verdict and eigenspectrum as evidence), Phase 02.1 (its fork analysis, survey, and the flat-target finding this phase tests the boundary of), Phase 02.2 (the CAE gate's structure and its FAIL as the comparison point). Requires a PASS from none of them.

**Requirements**: `TOPO-01..08` (minted by plan `02.4-08`; see `.planning/REQUIREMENTS.md`). Locally scoped `R1..R8` live in `02.4-SPEC.md`.

**Status**: Sealed 2026-08-07. 8/8 plans executed, verification passed 12/12.

**Outcome (2026-08-07): `TOPOAE_VERDICT = FAIL` — but global-scoped, not generic.** The gate ran as pre-registered and returned FAIL. The three gates do not share a scope, and the record deliberately does not collapse them: `t1_topo_fidelity` 1.026379 (threshold `<0.90`, **global** — whole-held-out-set MST/0-dim persistence structure) FAILED; `t2_recon_margin` 1.211939 (threshold `<1.00`, **global** — held-out reconstruction) FAILED, which is the trade the method makes by design since the paper claims topological fidelity at *comparable*, not better, reconstruction; `t3_rank_structure` 0.671980 at `k=15` (threshold `<0.90`, **local** — k-neighbourhood rank ordering) **PASSED**. The verdict artifact carries an additive `gate_scope` field so no downstream consumer reads a bare FAIL. No `topoae_handoff_{fit_key}.json` was written; R6's stale-handoff deletion was exercised concretely. **Phase 3 remains blocked** — 02.4 does not hand off.

**The FAIL is interpretable, which is what the phase's machinery bought.** `topoae.py` was audited against the authors' reference implementation (github.com/BorgwardtLab/topological-autoencoders) and four fidelity gaps closed (missing jointly-trained `latent_norm`, missing per-batch `d_x/d_x.max()`, a spurious batch-size division, a missing ½ factor); the mandatory Swiss roll check then showed TopoAE beating its matched `PlainAutoEncoder` baseline on the topological structural check (r=0.680 vs 0.471) while losing on MSE. Two pre-registration corrections are on the record, each committed alone with proved ancestry: **Erratum 1** (`9f5bd9e`) — §1 falsely claimed `AMBIENT_DIST_NORM = "none"` applied identically at training and gate time, when it scopes to the T1 gate only; **Amendment 1** (`9f9a74a`) — the original stopping rule interacted with the λ warm-up/ramp so early stopping fired at warm-up + patience, meaning *no fit ever reached the pre-registered `LAMBDA_TOPO = 0.1`* and baselines received 2.7× the training budget, so all 16 fits were re-run under a corrected rule (valid fits carry an `amend01` cache tag; pre-amendment artifacts are preserved as the record of the defect); **Amendment 2** (`b4a7945`) — withdraws the claim that `LAMBDA_TOPO = 0.1` is "the paper's own minimum searched value" after a fifth fidelity gap was found (`topoae.py` **sums** the reconstruction term over ambient features where the reference **means** over them, reparameterizing λ by the ambient dimension). Gap #5 is recorded, deliberately **not** fixed — the topo-to-recon *ratio* is invariant under that reparameterization and was measured at 0.23–0.31 on PU versus 0.42 on the Swiss roll, so it does not explain T1's parity with the baseline and the FAIL is unaffected. Fixing it would require a fresh pre-registration and a full re-run.

**Cross-phase finding: every FAIL in this milestone is global-scoped.** `02.4-FINDINGS.md` establishes, with every number re-verified against its sealed source artifact, that no *local*-scoped gate has ever failed here — 02.2's chart-transition cycle residual (1.089366 `<2.0`) and 02.4's T3 both passed, while Phase 2's `m = 0.412071`, 02.2's T1/T3 and 02.4's T1/T2 are all whole-set statistics. It further records that Phase 2's negative-eigenvalue mass is the expected signature of a *curved* manifold (classical MDS assumes Euclidean distances; a curved manifold's geodesics are not Euclidean-embeddable), so that gate tested *flatness* and found not-flat — supporting only "not a globally flat, single-chart-Isomap-coordinatizable manifold," not "no usable structure"; and that Phase 02.1's falsifier fired on `DISTORTION`, a **global** pairwise statistic over 200,000 sampled pairs that never evaluates a per-neighbourhood quantity, with `02.1-AMENDMENT-02.md` §6.4 itself already flagging that statistic as "nearly decoupled" from held-out reconstruction. **No sealed verdict is reopened, softened, or recomputed by any of this** — Phase 2, 02.2 and 02.4 all remain FAIL exactly as measured. What changed is what those FAILs are read to mean downstream, which is why Phase 3 was re-scoped to local curvature (see its dated re-scope note).

**Constraints carried in from the sealed phases** — binding on whatever gets planned:

  - **Swiss roll sanity check is mandatory.** `CLAUDE.md` names topological auto-encoders explicitly. `notebooks/02.4_swiss_roll_topoae_check.ipynb` must exist and pass before the model is considered done, with `cae.PlainAutoEncoder` as the matched baseline. Reference implementations: `notebooks/02.2_swiss_roll_cae_check.ipynb`, `notebooks/02.1_swiss_roll_geometry_probes_check.ipynb`.
  - **Pre-register the gate before any fit**, with ordering proved by git ancestry — the `057b084` / `c2c4c93` precedent. A changed constant requires a fresh, separately-ratified pre-registration and a full re-run.
  - **C2-smooth activations throughout the decoder** (DEC-02, CURV-01..03): ReLU's second derivative is identically zero. Same substitution and the same reporting obligation as CAE-06.
  - **Working dimension is not inherited.** `D_FROZEN = 5` is flagged suspect and was discarded as inapplicable by 02.1; the intrinsic-dimension cluster sits at 18–25. The latent dimension used must be stated and justified, not inherited.
  - **No package installs without a blocking gate.** A persistent-homology dependency is likely needed and every candidate so far has returned SUS; `torch==2.13.0+cpu` is already vetted and pinned. The paper's 0-dimensional persistence is computable from a minimum spanning tree, which `scipy.sparse.csgraph` already provides — an implementation route with no new dependency, worth costing before requesting one.
  - **Cache posture**: the gitignored Phase 1/2 caches are not reproducible here; a runner that cannot find them halts rather than regenerating, because regenerating changes provenance.

**Hard gate**: a machine-readable verdict artifact on the `gate_verdict` / `cae_verdict` pattern. PASS unblocks Phase 3; FAIL blocks it and is a complete, reportable outcome rather than an error to work around.

**Plans**: 8/8 plans executed

- [x] 02.4-01-PLAN.md
- [x] 02.4-02-PLAN.md
- [x] 02.4-03-PLAN.md
- [x] 02.4-04-PLAN.md
- [x] 02.4-05-PLAN.md
- [x] 02.4-06-PLAN.md
- [x] 02.4-07-PLAN.md
- [x] 02.4-08-PLAN.md

**Wave 1**

- `02.4-01` — `topoae.py` tracer: hand-rolled Union-Find persistence pairs, symmetric topological loss, λ schedule, `train_topoae` *(R1, R2 — autonomous)*

**Wave 2** *(blocked on Wave 1 completion)*

- `02.4-02` — Gate layer: T1/T2/T3, `GATING_METRICS` positional-slot remap, delegating verdict wrapper, verdict and handoff writers *(R4, R5, R6 — autonomous)*
- `02.4-03` — λ sweep runner in `notebooks/diagnostics/`, plus the mandatory Swiss roll sanity notebook and its blocking visual check *(R8 — **human checkpoint**)*

**Wave 3** *(blocked on Wave 2 completion)*

- `02.4-04` — `02.4-PREREGISTRATION.md`, committed with git-ancestry proof; resolves the three RESEARCH.md open questions *(R3 — **human checkpoint**)*

**Wave 4** *(blocked on Wave 3 completion)*

- `02.4-05` — Gated PU train runner, halt-not-regenerate preconditions, timing probe, primary rung *(R2, R3 — **human checkpoint**)*

**Wave 5** *(blocked on Wave 4 completion)*

- `02.4-06` — Remaining 13 fits, registry audit *(R2 — autonomous)*

**Wave 6** *(blocked on Wave 5 completion)*

- `02.4-07` — Evaluate runner, verdict artifact, PASS handoff / FAIL deletion *(R4, R5, R6 — **human checkpoint**)*

**Wave 7** *(blocked on Wave 6 completion)*

- `02.4-08` — `TOPO-01`..`TOPO-08` register, reconciliation runner, idempotency proof *(R7 — autonomous)*

**Cross-cutting constraints** — `must_haves.truths` shared by two or more plans:

- `git diff --quiet -- notebooks/pu_manifold/cae.py` exits 0 after the plan — *`02.4-01`, `02.4-02`*
- Gate comparisons run in float64 and measured values are stored at full precision, never pre-rounded — *`02.4-02`, `02.4-07`*
- Metrics are stored as full-precision floats through `cae.to_native`, never pre-rounded — *`02.4-02`, `02.4-07`*
- Ladder rungs are independent fits; no rung interacts with another — *`02.4-05`, `02.4-06`*
- Rungs are trained in the pre-registered ladder order, each carrying its own recorded seed — *`02.4-05`, `02.4-06`*
- A rung whose loss goes non-finite halts the run rather than being dropped from the ladder — *`02.4-05`, `02.4-06`*

### Phase 02.5: Local Curvature Feasibility & CAE Local Re-Gate (INSERTED)

**UI hint**: no

**Goal**: Establish empirically whether a **local** second fundamental form is estimable at the PU data's sampling density, and — if it is — pre-register and run a **locally**-scoped validity gate on the Chart Auto-Encoder, producing a machine-readable verdict that either resolves or explicitly does not resolve Phase 3's blocking dependency.

**Why inserted**: Phase 3 is blocked on a **PASS** that no method has produced, and `02.4-FINDINGS.md` establishes why that gate may be asking the wrong question. Every FAIL in this milestone — Phase 2's `m = 0.412071`, 02.2's T1/T3, 02.4's T1/T2 — is a failure of a *global* statistic, while every *local*-scoped gate measured has passed (02.2's chart-transition cycle residual `1.089366 < 2.0`; 02.4's T3 `0.671980 < 0.90` at `k=15`). Mean curvature is a **local invariant**: the second fundamental form `II_p` depends only on an arbitrarily small neighbourhood of `p`, and a manifold need not admit any global chart to have well-defined curvature everywhere. So the milestone's repeated failure to obtain *global* coordinates does not by itself block a curvature field — but nothing measured so far licenses proceeding either, and that gap is what this phase closes.

**Why the CAE is the candidate vehicle**: it is an atlas of local charts by construction (arXiv:1912.10094), its *local* consistency gate passed on the real PU data, and it is the **only** model in this milestone to pass its mandatory Swiss roll check outright — 4.8% held-out relative reconstruction error against a `<10%` bound, 2.2× better than a matched plain-AE, 8/8 charts surviving pruning, printed `Swiss roll recovered: True`. Its sealed `CAE_VERDICT = FAIL` rests on T1 global geodesic distortion and T3 global reconstruction margin; neither is a property local curvature estimation requires. **This is a reason the CAE is not disqualified, not evidence it is licensed** — a locally-scoped PASS must be earned under its own fresh pre-registration, never inherited from 02.2's gate as measured.

**Depends on**: Phase 02.4 (its `02.4-FINDINGS.md` cross-phase re-reading and the sealed `TOPOAE_VERDICT` with its `gate_scope` annotation), Phase 02.2 (the sealed CAE gate, `cae.py`, and its passing Swiss roll notebook), Phase 02.1 (the sealed representation analysis, and its own §6.4 record that the `DISTORTION` statistic its falsifier fired on is "nearly decoupled" from held-out reconstruction), Phase 1 (the frozen 10,000-row subsample). Requires a PASS from none of them.

**Requirements**: No milestone-level REQ-IDs were minted for this phase. Its de-facto requirement set is `02.5-CONTEXT.md`'s sixteen implementation decisions, **D-00..D-15**, and every plan traces against those IDs. Recorded here so the absence is a stated choice rather than an omission.

**Status**: Inserted 2026-08-07. Discussed 2026-08-07 (`02.5-CONTEXT.md`, 16 decisions). Researched 2026-08-07 (`02.5-RESEARCH.md`, `02.5-PATTERNS.md`, `02.5-VALIDATION.md`). Planned 2026-08-07 — 13 plans across 12 waves, decision coverage 16/16.

**The two stages, in order** — the first gates the second:

  1. **Feasibility probe, on a manifold with a known answer.** Per `CLAUDE.md`'s standing rule: a Swiss roll with *analytic* mean curvature, estimating `II` by local PCA plus quadric/jet fitting, then degrading ambient dimension, intrinsic dimension and sample density toward the PU regime (768 ambient, `n = 10,000`, intrinsic 18–25) to find empirically where the estimator breaks. **If local curvature is not estimable in that regime, the CAE re-gate is moot and must not be run** — that negative result is itself a complete and reportable outcome for this phase.
  2. **Locally-scoped CAE re-gate**, only if stage 1 clears. Fresh, separately-ratified pre-registration with git-ancestry proof, gates chosen on *local* criteria, run against the sealed 02.2 fits or fresh ones as the pre-registration specifies.

**Open questions this phase must resolve, not assume**:

  - **Sample density is the binding constraint, not geometry.** A local quadratic fit needs `d(d+1)/2` coefficients per normal direction — 15 at `d=5`, 171 at `d=18`, 210 at `d=20`, 325 at `d=25` — against `k* = 15` and `n = 10,000` in ambient 768. At the intrinsic dimensions the evidence clusters around (18–25), this is badly underdetermined. Raising `k` buys equations but costs locality; that tradeoff is unresolved.
  - **Which working dimension.** `D_FROZEN = 5` is flagged suspect and **must not be inherited** — `02-FINDINGS.md` §6.4 and `STATE.md` record the residual-curve elbow saturating early under 41% negative eigenvalue mass, so it measured the flatness failure rather than the manifold's dimension. Three independent estimates cluster at 18–25 (local PCA 25.0, TwoNN 19.5, eight `compute_dim` geometric estimators at 18), and 02.4's T3 improving monotonically with latent dimension to `d=32` is weak independent corroboration that the true dimension sits well above 5.
  - **What a local-scoped PASS is allowed to unblock.** Phase 3's dependency currently names a global-scoped PASS. Whether a local PASS resolves it, or whether that dependency must itself be amended, is a decision this phase must make explicitly rather than by implication.
  - **Whether a PASS here should revisit 02.1's falsifier.** `02.1-AMENDMENT-02.md` carries a *symmetric* falsifier that overturns graph-native when a coordinate candidate clears a validity gate. A locally-scoped PASS is not identical to the condition it names — and arguing that difference is exactly what any such amendment would have to do.

**Constraints carried in from the sealed phases** — binding on whatever gets planned:

  - **Swiss roll sanity check is mandatory** for any new manifold model or curvature estimator (`CLAUDE.md` names curvature estimators explicitly). Stage 1 *is* a Swiss roll check by construction; any model introduced in stage 2 needs its own.
  - **Pre-register the gate before any fit**, ordering proved by git ancestry. A changed constant requires a fresh, separately-ratified pre-registration and a full re-run.
  - **No sealed verdict may be reopened, softened, or recomputed.** `GATE_VERDICT`, `CAE_VERDICT` and `TOPOAE_VERDICT` all remain FAIL as measured. This phase adds a new, differently-scoped measurement; it does not revise an old one.
  - **`notebooks/pu_manifold/cae.py` is Phase 02.2's sealed artifact** — import from it, never edit it. `src/effdim/` and `pyproject.toml` stay untouched for the v1.1 milestone.
  - **C2-smooth activations throughout** any decoder used for curvature (DEC-02, CURV-01..03): ReLU's second derivative is identically zero, which silently zeroes the second fundamental form.

**Plans**: 9/13 plans executed

Plans:
**Wave 1**

- [x] 02.5-01-PLAN.md — Tracer: Swiss roll → centroid/Laplace–Beltrami estimator → Spearman vs analytic H, end to end (D-00, D-01, D-03, D-05, D-07)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02.5-02-PLAN.md — Graph-of-function fixture family at arbitrary (d, D, codimension), non-uniform sampling, and the density correction (D-03, D-06, D-07)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02.5-03-PLAN.md — Non-gating quadric cross-check, estimator agreement, and permutation-null calibration (D-01, D-02, D-05)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 02.5-04-PLAN.md — Direction-aware verdict functions and the R6 verdict/handoff writers at 02.5 scope (D-01, D-12, D-15)
- [x] 02.5-05-PLAN.md — **[checkpoint]** Mandatory CLAUDE.md Swiss roll sanity notebook for the curvature estimator (D-03, D-05, D-07)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 02.5-06-PLAN.md — **[checkpoint]** Stage-1 pre-registration: ratified, committed alone, git-ancestry proved (D-00..D-08, D-12)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 02.5-07-PLAN.md — **[checkpoint]** Stage-1 feasibility sweep, the boundary report, and the **GO/NO-GO gate that decides whether stage 2 runs at all** (D-01..D-08)

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 02.5-08-PLAN.md — Exact chart-decoder curvature via `torch.func`, C2-smoothness guard, sealed-fit load check (D-09, D-10)

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 02.5-09-PLAN.md — **[checkpoint]** Mandatory Swiss roll sanity notebook for the chart-curvature model (D-03, D-09)

**Wave 9** *(blocked on Wave 8 completion)*

- [ ] 02.5-10-PLAN.md — **[checkpoint]** Stage-2 pre-registration: the D-09/D-10 reconciliation and D-12's neither-clears branch resolved (D-09..D-12, D-15)

**Wave 10** *(blocked on Wave 9 completion)*

- [ ] 02.5-11-PLAN.md — Gate A: CAE charts vs raw points, both scored against analytic H at the PU-matched regime (D-09, D-10)

**Wave 11** *(blocked on Wave 10 completion)*

- [ ] 02.5-12-PLAN.md — **[checkpoint]** Gate B seed stability on the three sealed fits, sealed stage-2 verdict, handoff or stale deletion, routing (D-10..D-12, D-15)

**Wave 12** *(blocked on Wave 11 completion)*

- [ ] 02.5-13-PLAN.md — **[checkpoint]** Retarget Phase 3's dead-pointer dependency, revisit 02.1's falsifier, complete the phase record (D-04, D-13, D-14, D-15)

### Phase 02.6: Decoder Substrate Screening (INSERTED)

**Goal**: Rank the three in-repo decoder substrates on **manifold preservation, measured by persistent homology agreement**, decided independently of how curvature is approximated — and separately retire the confound that halted the first attempt. This phase **measures and ranks. It promotes nothing and gates nothing.** No substrate decision is sealed here. `02.5-10` receives a ranking, a derivative-usability table, and the separating experiment's result, and makes the promotion decision itself under its own seal-before-measure discipline.

> **Goal and success criteria rewritten 2026-08-10** from `02.6-CONTEXT.md` (decisions D-01..D-22), which is authoritative for scope. The prior Goal and SC-1..SC-5 described the halted decoder-pullback-curvature axis and said "promote at most ONE", which D-10 reverses. The **Status** paragraph below is the original halt record, preserved verbatim. Nothing measured under the old axis is carried forward as evidence about any substrate.

**Why inserted**: Phase 02.5's Arm B was built against the Chart Auto-Encoder decoder, and plan 02.5-09 then measured that decoder failing its Swiss roll check — curvature Spearman `-0.0604` against the raw-point baseline's `0.6712`, alongside reconstruction 2.96x better than a matched plain AE. Arm B's structural argument, that a decoder escapes stage 1's locality wall by differentiating an analytic map rather than sampling a neighbourhood, **does not depend on charts**; it holds for any smooth decoder. The substrate is therefore open, and `02.5-10` is the last point it can change before stage-2 thresholds seal. Running stage 2 on a substrate that failed its admission gate is precisely what that gate exists to prevent.

**Depends on**: Phase 02.5 stage 1 (consumes `CURVATURE_VERDICT = FAIL` and its measured locality mechanism as motivation; requires no PASS). **Blocks**: Phase 02.5 plans 02.5-10..13.

> **Blocking note SATISFIED (2026-08-11).** `02.6-FINDINGS-02.md` unblocks `02.5-10`: what it
> receives is a ranking and two tables (the persistent-homology matrix and the
> derivative-usability bridge), not a substrate decision — the promotion decision stays with
> `02.5-10` under its own seal-before-measure discipline (D-09, D-10, D-19).

**Candidates — three in-repo only (D-20).** Adding a substrate not already in-repo is phase-sized work, not a substitution (`02.5-NOTE-substrate-selection.md` §5).

| substrate | status | known |
|---|---|---|
| plain AE | built, has Swiss roll check | best H0 retention per parameter (`~0.53-0.63` vs CAE `0.183`, at 1/8 the parameters) |
| TopoAE (Moor et al., 1906.00722) | built, sealed 02.4 fits | `TOPOAE_VERDICT = FAIL` globally but the ***local*-scoped gate PASSED** — the only model in this project to pass anything, in the scope 02.5 cares about |
| CAE (1912.10094) | built, sealed 02.2 fits | admission **FAILED** on the halted curvature axis (02.5-09); retained as the measured negative control |

**Deferred to a scoped follow-on phase**, not screened here: RTD-AE (ICLR 2023) — noted as having become *more* aligned the moment PH was chosen as the axis, since it constrains the map induced on homology groups toward an isomorphism, close to the criterion itself; Witness AE (k-NN witness complex, the scaling lever for the compute-bound quantity); GRAE (geometry- rather than topology-regularised). TopoAE++ (2502.20215) is resolved **NO-GO** in `02.6-RESEARCH.md` — its fast algorithm is planar-specific and does not apply at `d = 40`.

**Success criteria (ranking, explicitly non-gating)** — traced to `02.6-CONTEXT.md` decisions, cited per criterion.

1. **SC-1 — Ratified blind, and ratified as non-gating.** The criterion, its bars, and the read-out matrix are written down and committed **before any measurement on the new axis**, and ratified as **explicitly non-gating**: the criterion ranks, it does not promote. The `SPREAD > mean - floor` disqualifier proposed in the prior research is **dropped** — with a non-gating criterion there is nothing for a disqualifier to disqualify. (D-09, D-11)
2. **SC-2 — The separating experiment runs first, and is non-blocking.** A generously sized net is fit by regression on the analytic Swiss roll parameterization `(s, y) -> R^3` and curvature traced through it with the existing `decoder_curvature` tracer; the surface is correct by construction, so the only remaining variable is whether a *net's* second derivatives survive. PASS is **measured degradation from the analytic decoder's own exactness**, not a threshold set in the abstract. Its result is recorded and the screen proceeds regardless; the consequence for `02.5-10` is stated either way. (D-12..D-15)
3. **SC-3 — The full read-out matrix, reported separately, never collapsed.** Two references (the true 2-manifold in exact intrinsic coordinates `(arc-length s(t), y)`, and the ambient 3-D point cloud) × two degrees (H0, H1 — H2 excluded, no `beta_2` power analysis exists) × two distances (bottleneck and Wasserstein, each normalised by the reference diagram's own max persistence) × two spaces (encoder latent and decoder image) = **16 numbers per seed per candidate**, 64 across four seeds, 192 across the three candidates on the Swiss roll side. The runner emits this table and the findings document reads it. A correct unroll matches the intrinsic reference and diverges from the ambient one — reporting both makes that visible instead of scoring it as failure. (D-03..D-06)
4. **SC-4 — Four seeds minimum, spread reported.** Plan 02.5-09 measured Spearman running `-0.0604 / -0.1444 / 0.8665 / 0.4250` across torch seeds at one configuration; a single-seed result is a lucky draw. (D-11)
5. **SC-5 — The derivative-usability bridge, measured with no bar applied.** Autodiff versus finite-difference agreement on the trained decoder's second derivatives (`torch.func` against a finite-difference stencil) — it needs no ground truth, so it transfers to PU unchanged. Run on **both the Swiss roll and PU, reported separately**: the roll is the only place a method can be checked against a correct answer, but the phase's question is about PU. Agreement numbers are emitted with **no threshold**; `02.5-10` ratifies the threshold under its own discipline. (D-16..D-18)
6. **SC-6 — Every screened candidate gets a CLAUDE.md-mandated Swiss roll notebook, importing model code unchanged.** Bars on the roll are **absolute** (its answer is exact); bars on PU are **comparative**, against a matched plain-AE baseline and a null, because an absolute bar is not computable on PU (`02.5-NOTE-substrate-selection.md` §2). Any candidate implemented or modified carries a faithful-to-paper deviation audit — the CAE audit found two deviations, and one hid the effect an experiment was built to find. (D-07, D-20, project CLAUDE.md)
7. **SC-7 — The phase promotes no substrate.** `02.5-10` inherits a ranking, a derivative-usability table, and the separating experiment's result — not a substrate decision. A substrate that tops the PH ranking but fails the bridge is **not** promoted, and the ranking is **not** walked to the runner-up: that would turn the bridge into a second ranking axis applied after seeing results, structurally the failure `02.6-FINDINGS.md` §4 recorded. (D-10, D-19)

**Cross-cutting constraints**: No verdict artifacts, no sealed gates, no promotion; those belong to `02.5-10` and to the promoted model's own follow-on phase. **The original "no PU fits" constraint is amended (D-21)** to permit *new* PU fits **for the SC-5 bridge arm only** — without it the plain AE has no PU arm and the bridge is Swiss-roll-only, contradicting D-17. This widens the scope fence the phase was inserted with and PU fits are the expensive operation in this project; it is recorded as **costly to reverse**. Sealed verdicts (`GATE_VERDICT`, `CAE_VERDICT`, `TOPOAE_VERDICT`, `CURVATURE_VERDICT`) are never reopened, softened, or recomputed, and sealed fits (02.2 CAE, 02.4 TopoAE) remain **read-only** — the amendment permits new fits, it does not reopen sealed ones. The already-built distortion instruments (`cae.embedding_distortion`, `cae.fit_global_scale`, `geometry_probes.distortion_stats`) are **not computed in this phase** (D-08): a second geometric number invites post-hoc axis-switching, the exact failure `02.6-FINDINGS.md` §4 recorded. The Stage A / Stage B split from `02.5-NOTE-substrate-selection.md` §3 is kept, but **admission is descriptive** — failing Stage A no longer bars a substrate from Stage B (D-22). `src/effdim/` and `pyproject.toml` are not modified (`ripser`/`persim` are already venv-installed and undeclared — a known, recorded reproducibility gap). Additive only.

**Effect sizes this phase must argue against, not around**: the largest measured difference so far is **single-chart versus multi-chart (~3.4x on H0 retention)**, while the topological-loss axis is worth **~6%** (TopoAE `0.668` vs plain AE `0.628`). A candidate justified only by a better topological loss is arguing inside that 6%. Correspondingly, the PU topology beyond connectedness is bounded, not zero: no cycle above ~3x the manifold's local thickness at `n <= 2000`, with **no power analysis at all for `beta_2`** — so higher-homology methods cannot be dismissed on "there is nothing there to find," nor adopted on the assumption that there is.

**Design note**: `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-NOTE-substrate-selection.md` (commit `f8350b6`) sets out the Stage A admission / Stage B decision structure this phase implements, and Section 4 records three inferences that must **not** be drawn from a Swiss roll result. §4's two standing objections to topology-based scoring were surfaced during discussion and D-01 was chosen with them visible: the second — that the roll's ambient `beta_1 = 1` from near-touching spiral arms penalises correct unrolling when scoring a `d=2` latent on ambient H1 — is addressed by SC-3's two-reference design rather than ignored. **The first, that H0 preservation is not curvature fidelity, stands unretired** and is carried as assumption A-01 in `02.6-CONTEXT.md`.

**Status**: **HALTED 2026-08-10 mid-execution at 3/6 plans — see `02.6-FINDINGS.md`.** No substrate promoted, none eliminated. The ranking axis (agreement between decoder-pullback curvature and analytic `H`) was found to measure a composite of three separable properties — did the architecture learn the right surface, are that trained net's second derivatives trustworthy, is pullback the right approximator — so its measurements cannot be attributed to substrate choice. Ranking to be replanned on manifold/topology preservation, decided independently of curvature approximation. `02.6-SCREENING-RULE.md` was ratified blind before measurement but constrained no published conclusion; the axis change was made **after** an unfavourable plain-AE result and is recorded as such in `02.6-FINDINGS.md` §4. Phase 02.5 stage 2 remains blocked — `02.5-10` still has no substrate decision to inherit.

**Outcome (2026-08-11), replan complete — see `02.6-FINDINGS-02.md`.** The replan ran to
completion on the persistent-homology agreement axis: the criterion (`02.6-SCREENING-RULE-02.md`)
was ratified before any measurement on this axis and ratified explicitly non-gating; three
candidates were screened across four seeds, producing 192 separately reported numbers, none
weighted, summed, or collapsed; the separating experiment ran first and selected D-15's
PASSES branch (nets can carry usable second derivatives when the surface is right, on
evidence bounded to the general question, not the per-substrate one); the derivative-usability
bridge ran on both the Swiss roll and PU regimes, at both the full-Hessian and reduced
`H_vec`/`H_norm` levels, finding the two levels disagree by three to five orders of magnitude
in a substrate-dependent direction. **No substrate was promoted and none was eliminated.**
`02.6-FINDINGS-02.md` additionally names two confounds in the ranking axis itself — the CAE's
8-D latent cells are not dimensionally comparable to `plainae`/`topoae`'s 2-D cells, and
TopoAE's own training objective optimises ambient-space topological agreement while this
phase ranks on intrinsic-reference agreement — so even the ranking it hands forward carries
two caveats a reader must clear before trusting it. **Phase 02.5 stage 2 is unblocked**:
`02.5-10` inherits a ranking (with its disagreements and confounds named), a
derivative-usability table (both levels, both regimes), and the separating experiment's
result — not a substrate decision — under its own seal-before-measure discipline.

Prior history: Inserted 2026-08-10, scoped directly in this entry with no discuss pass — `02.6-RESEARCH.md`'s Seed Protocol / Promotion Rule were therefore **proposals**, ratified at plan `02` rather than inherited silently. Researched 2026-08-10. Planned 2026-08-10 — 6 plans across 3 waves. Executed to 3/6, then **halted** (see Status).

Replan history: `/gsd-discuss-phase` pass 2026-08-10 after the halt (`02.6-CONTEXT.md`, `02.6-DISCUSSION-LOG.md`) — the first this phase has had, and now authoritative for scope. Re-researched 2026-08-10 onto the persistent-homology agreement axis, overwriting `02.6-RESEARCH.md` (which carries a `## Retractions from Prior RESEARCH.md` table naming every withdrawn claim) and reseeding `02.6-VALIDATION.md`. `02.6-PATTERNS.md` is **partially superseded** — it and the prior research both claimed `chart_curvature.assert_c2_activation` can be called on a `cae.PlainAutoEncoder`; it cannot (`PlainAutoEncoder` sets no `self.activation`), and plan `01` introduced `assert_c2_decoder` instead. Plans `02.6-01`, `-02`, `-05` are **executed and retained** — `decoder_curvature.py`, the ratified rule, and the four-seed runner exist on disk and are reused as built assets. Plans `02.6-03`, `-04`, `-06` are **superseded by the axis change** and are not resumed; new plans are numbered from `02.6-07`. No milestone REQ-IDs; coverage is traced against the seven success criteria above as **SC-1..SC-7**, which map to `02.6-CONTEXT.md` decisions D-01..D-22.

**Plans**: 15/15 plans executed

> **The `(SC-N)` tags on plans `01`–`06` below refer to the OLD SC-1..SC-5**, which described the halted curvature axis. They are left as written so the historical record reads correctly. New plans trace against the current SC-1..SC-7.

Plans:

- [x] 02.6-03-PLAN.md
- [x] 02.6-04-PLAN.md
- [x] 02.6-06-PLAN.md

**Wave 1** *(halted-run history — retained, not re-executed)*

- [x] 02.6-01-PLAN.md — **[tracer]** `decoder_curvature.py`: exact decoder curvature with no chart routing, plus sphere/flat known-answer, C2-guard and bit-identity tests (SC-3)
- [x] 02.6-02-PLAN.md — **[checkpoint]** Ratify the screening bars, the seed-spread disqualifier and the promotion rule **blind**, before any 02.6 measurement (SC-4, SC-5)

**Wave 2** *(halted-run history — retained, not re-executed)*

- [~] 02.6-03-PLAN.md — **[checkpoint]** CLAUDE.md Swiss roll curvature admission notebook for the **plain-AE** decoder (SC-2, SC-3) — **PARTIAL at halt:** notebook written + executed with outputs (`bec17ef`, `028188f`), blocking `human-verify` gate never closed. **SUPERSEDED by the axis change** — not resumed.
- [~] 02.6-04-PLAN.md — **[checkpoint]** CLAUDE.md Swiss roll curvature admission notebook for the **TopoAE-trained** decoder, with a genuinely matched baseline at budget parity (SC-2, SC-3) — **NOT STARTED at halt. SUPERSEDED by the axis change** — not resumed.
- [x] 02.6-05-PLAN.md — Four-seed, four-axis screening runner in `notebooks/diagnostics/`, measuring and printing only — no bar, no verdict (SC-3, SC-4)

**Wave 3** *(halted-run history — retained, not re-executed)*

- [~] 02.6-06-PLAN.md — **[checkpoint]** `02.6-FINDINGS.md`: per-seed tables, bar-by-bar findings, all seven candidates dispositioned, at most one promoted argued against the effect sizes, Tier-2+ gated (SC-1, SC-4, SC-5) — **SUPERSEDED at halt:** never ran; `02.6-FINDINGS.md` exists instead as a halt record with no promotion argument. Its "promote at most one" premise is additionally reversed by D-10.

**Replan — new plans (PH-agreement axis).** Planned 2026-08-10 — 9 plans across 5 waves, numbered from `02.6-07`. Waves are numbered independently of the halted run; the halted run's Waves 1–3 above are history. Coverage traced against SC-1..SC-7 above: 7/7. Plans `07`, `08`, `12`, `13`, `15` are non-autonomous.

**Replan Wave 1** *(no dependencies — the ratification lands before any new-axis measurement, and the separator runs first and non-blocking)*

- [x] 02.6-07-PLAN.md — **[checkpoint:decision]** Ratify the PH-agreement criterion, its bars, the 16-cell read-out matrix and five open design questions **blind and explicitly non-gating**; `02.6-SCREENING-RULE-02.md` (SC-1)
- [x] 02.6-08-PLAN.md — **[tracer][checkpoint:human-verify]** The separating experiment: `analytic_param.py` (closed-form arc length, intrinsic plane, exact ambient map, the analytic Swiss roll decoder as the exactness floor, the generously-sized regression net) + tests + `02.6_swiss_roll_separator_check.ipynb` (SC-2)

**Replan Wave 2** *(blocked on Wave 1)*

- [x] 02.6-09-PLAN.md — **[tracer]** `persistence_probe.py` + tests: diagrams, bottleneck/Wasserstein normalised by the reference's own max persistence, saturation and thin-denominator guards, and one end-to-end 16-cell slice (SC-3)
- [x] 02.6-10-PLAN.md — **[tracer]** `derivative_bridge.py` + tests: finite-difference Hessian calibrated against this project's own known-answer fixtures, mirrored `tr_g(II)` reduction pinned to the sealed tracer, agreement at both levels, no bar (SC-5)

**Replan Wave 3** *(blocked on Wave 2)*

- [x] 02.6-11-PLAN.md — `decoder_substrate_ph_screen_run.py`: three candidates × four seeds × 16 separately labelled numbers, spread reported, measure-and-print only (SC-3, SC-4)
- [x] 02.6-12-PLAN.md — **[checkpoint:human-verify]** CLAUDE.md Swiss roll PH notebooks for the **plain-AE** and **TopoAE** arms, with a matched baseline at budget parity and a chance null (SC-6)
- [x] 02.6-14-PLAN.md — `derivative_bridge_run.py`: the bridge on the Swiss roll **and** PU, reported separately, step calibrated per model, no threshold (SC-5)

**Replan Wave 4** *(blocked on Wave 3)*

- [x] 02.6-13-PLAN.md — **[checkpoint:human-verify]** CLAUDE.md Swiss roll PH notebook for the **CAE**, the measured negative control (SC-6)

**Replan Wave 5** *(blocked on Wave 4)*

- [x] 02.6-15-PLAN.md — **[checkpoint:human-verify]** `02.6-FINDINGS-02.md`: the ordering proof, the full 192-number matrix, the ranking with its cell-level disagreements, both bridge tables, full disclosure for `02.5-10` — **promoting no substrate** (SC-7)

**Two source corrections carried into the replan's plans, both verified against source at planning time.** First, the one already recorded below: `chart_curvature.assert_c2_activation` cannot be called on a `cae.PlainAutoEncoder`, and `decoder_curvature.assert_c2_decoder` is the fix. Second, new this pass: **`cae.ChartAutoEncoder.forward(x)` returns no `"y"` key** — it returns `z`, `z_charts`, `y_charts`, `p`, `e`, and the CAE's decoder image is `model.reconstruct(x)`, the argmax-chart reconstruction that `notebooks/02.2_swiss_roll_cae_check.ipynb` itself uses. Both `02.6-RESEARCH.md` Pattern 3 and `02.6-PATTERNS.md`'s Shared Patterns state the opposite. Additionally, `02.6-RESEARCH.md`'s arc-length tolerance (`< 9e-14`) is below the measured deviation (`9.237e-14`) and its Pitfall 2 intrinsic-H1 value (`0.420`) is subsample-procedure-dependent and did not reproduce (measured `0.3348`); neither is pinned as a constant anywhere in the plans. A fourth planning-time measurement is new evidence rather than a correction: `persim.bottleneck` between the two references' own H1 diagrams already sits exactly at the ambient reference's saturation value (`0.35988`), so the `(H1, ambient)` bottleneck cell is expected to be saturated for a *correctly* unrolled latent — pinned as a regression test in plan `02.6-09` before any candidate is measured.

**Planning-time correction carried into the plans**: `chart_curvature.assert_c2_activation` **cannot** be called on a `cae.PlainAutoEncoder` — `ChartAutoEncoder` sets `self.activation`, `PlainAutoEncoder` does not, so the sealed guard hard-raises on every plain decoder. Both `02.6-RESEARCH.md` and `02.6-PATTERNS.md` state the opposite. Verified false during planning; plan `01` introduces `assert_c2_decoder`, which reaches the sealed `ZERO_SECOND_DERIVATIVE_ACTIVATIONS` frozenset by inspecting the decoder's own activation submodules instead of a recorded attribute. `cae.py` is not edited.

### Phase 02.7: Manifold-Template Inference Front End (INSERTED)

**Goal**: Determine **whether a latent manifold template can be inferred from an unlabelled point cloud at all**, and with what accuracy and abstention rate. This phase builds and measures the **inference front end only** — intrinsic dimension, then persistent homology under two distance metrics, then a template decision with an explicit **abstain** option. It trains no VAE, restricts no auto-encoder, and selects no template for PU. The topological-VAE arm is deferred to a follow-on phase, **gated on this front end working**.

**Why inserted**: Acosta et al. (`2212.10414`) constrain a VAE's latent space to a canonical template manifold `Z` and derive curvature through the decoder. The paper **assumes template selection as a preceding step** rather than solving it: it uses TDA to infer the topology of the neural manifold and then picks a `Z` homeomorphic to it, and in its own place-cell example it additionally has task knowledge that the animal runs a circular track — so `Z = S^1` is not a blind inference there. Their implemented library is essentially spheres and products of spheres (`S^1`, `S^2`, `T^2 = S^1 x S^1`). **The step the paper assumes is the step this project does not have**, and it is the one that decides whether the method is applicable to PU embeddings at all.

**Depends on**: Phase 02.6 (consumes `persistence_probe.py` and its raw-vs-normalised reading discipline; requires no verdict from it). **Blocks**: the deferred topological-VAE phase. Does **not** block Phase 02.5 or Phase 3.

**What already exists and is reused, not rebuilt**

| pipeline step | asset | status |
|---|---|---|
| intrinsic dimension | `src/effdim/geometry.py` — `mle_dimensionality`, `two_nn_dimensionality`, `danco_dimensionality`, `mind_mli/mlk_dimensionality`, `ess_dimensionality`, `tle_dimensionality`, `gmst_dimensionality` | built; **frozen this milestone — call, never modify** |
| persistent homology | `notebooks/pu_manifold/persistence_probe.py` (02.6) | built and tested; import unchanged |
| kNN graph | `notebooks/pu_manifold/mknn.py` | built; basis for the spectral-distance arm |
| template decision + abstain | — | **new** |
| spectral / diffusion distance | — | **new** |

**Scope decisions taken at insertion (2026-08-11)**

- **Front end only.** No VAE, no template-restricted auto-encoder, no curvature claim. The follow-on phase is gated on this one producing a front end whose accuracy and abstention rate are known.
- **Synthetic first, PU as a single final read-only probe.** Template-selection accuracy is only measurable where the template is known by construction — the same discipline CLAUDE.md's Swiss roll check enforces. PU is probed once at the end and its answer reported, whatever it is.
- **Both Euclidean and spectral/kNN distances, reported separately, never collapsed.** PU embeddings are `D = 768` ambient, and Euclidean persistent homology is documented to fail on a noisy circle in high ambient dimension where graph/spectral distances survive. Agreement between the two metrics is the phase's confidence signal, not a redundancy to average away.

**Success criteria**

1. **SC-1 — Template selection is model selection with an abstain option, ratified before measurement.** The decision rule, its thresholds, and the abstain condition are written down and committed **before any synthetic cloud is scored**, following `02.6-SCREENING-RULE-02.md`'s precedent. A front end that always emits a template is the wrong contract; "undetermined" is a valid and expected output.
2. **SC-2 — Intrinsic dimension is estimated across multiple neighbourhood sizes and several estimators, and the spread is reported.** A plateau in `d_hat(k)` over a non-trivial range of `k` is the signal; a single `k` from a single estimator is not. Strong variation of `d_hat` across the cloud is itself evidence that **one global smooth template is the wrong model**, and must be reported as such rather than averaged into a number.
3. **SC-3 — Persistent homology computed under BOTH Euclidean and spectral/kNN distances, reported separately.** Degrees `H_0` through `H_dhat`. Persistence must survive a non-trivial range of filtration scales **and** bootstrap/subsampling, not appear at one arbitrary radius.
4. **SC-4 — Selection accuracy and abstention rate measured on a synthetic benchmark with known ground truth.** Generate `Z_true -> random smooth immersion -> R^D -> non-uniform sampling + noise` across the library `{S^1, S^2, T^2, disk/ball}`, varying `D`, sample count, noise, and sampling density. Report accuracy **and** abstention rate as separate numbers; a selector that never abstains is not thereby better.
5. **SC-5 — A CLAUDE.md-mandated Swiss roll check.** The Swiss roll is a contractible 2-D sheet whose correct template is the **disk**, not `S^1/S^2/T^2` — so the check is that the front end selects the trivial template (or abstains) and that a wrong template is visibly detectable. This is why the library carries a disk/ball member.
6. **SC-6 — One read-only probe of the PU embeddings, reporting whatever it finds including "no template in library" or "abstain."** Sealed fits are read; nothing is refit.

**The prior this phase must be honest about, and is likely to confirm**: this ROADMAP already records that PU topology beyond connectedness is bounded — *no cycle above ~3x the manifold's local thickness at `n <= 2000`, with no power analysis at all for `beta_2`*. Run through the decision table that reads `beta_0 = 1` with no persistent `beta_1`, i.e. a **Euclidean/ball-like candidate, outside the `{S^1, S^2, T^2}` library**. **If PU is ball-like, "does restricting to a template help?" has no answer on PU**, because there is no non-trivial template to restrict to — and SC-6 finding exactly that is a complete, reportable outcome that legitimately terminates the follow-on arm. It is not a failure of this phase.

**Cross-cutting constraints**: `src/effdim/` and `pyproject.toml` are **frozen** — the eight intrinsic-dimension estimators are called, never edited. Sealed verdicts (`GATE_VERDICT`, `CAE_VERDICT`, `TOPOAE_VERDICT`, `CURVATURE_VERDICT`) are never reopened; sealed fits are read-only. **`d_frozen = 5` is not inherited** — Phase 02's own findings flag it as suspect, so this phase re-estimates intrinsic dimension rather than assuming it, and reports its own estimate against that record. `ripser`/`persim` remain venv-installed and undeclared (known, recorded gap — not fixed here). Additive only.

**Inherited discipline from Phase 02.6, which this phase must not relearn the hard way**:

- **Read-outs are reported separately and never collapsed into a score.** Accuracy and abstention are two numbers; Euclidean and spectral PH are two answers.
- **Normalised persistence distances are comparable only within a reference.** `02.6-FINDINGS-02.md` §5 measured intrinsic-`H1` and ambient-`H1` denominators differing by 2.15x; use raw distances for any cross-reference comparison.
- **Write at least one acceptance criterion that exercises production dimensionality.** Phase 02.6 produced three defects that passed every toy-scale check and failed at real scale — `torch.quantile`'s undocumented `2**24` cap, a training-budget asymmetry that silently made a comparison unfair, and a safety guard bypassed by passing a bound method. At `D = 768` this phase is squarely in that regime.
- **For any experiment that compares configurations, assert the comparison is fair** — same budget, same convergence rule, same stopping condition — not merely that each arm ran.

**Design note**: persistent homology yields **candidates, not a unique manifold**. Betti numbers do not determine homeomorphism type in general; the paper's approach works because its hypothesis class is small, and homology recovery from finite samples needs sampling-density and regularity assumptions that an arbitrary cloud does not guarantee. The front end is therefore a **classifier over a small named library with an abstain option**, and must never be described as recovering the manifold. If the cloud has boundary, self-intersections, multiple components, or mixed intrinsic dimension, a single global template is inappropriate and the correct output is abstention.

**Status**: **Planned 2026-08-11.** Discussed (`02.7-CONTEXT.md`: sixteen decisions D-01..D-16 plus assumptions A-01..A-05), researched (`02.7-RESEARCH.md`, carrying measured `ripser` H2 cost curves), pattern-mapped and validation-seeded. **12 plans across 8 waves.** Decision coverage **16/16** against D-01..D-16; success-criterion coverage **6/6** against SC-1..SC-6. Next step: `/gsd-execute-phase 02.7`.

**Wave order — the calibrate → ratify-blind → measure ordering is enforced structurally by `depends_on`, not by convention.** `01` **[tracer + blocking `checkpoint:decision`]** one `S^1` cloud end to end at `D = 768` under both metrics, with D-01's one-way mechanism confirmed before any lookup-table code exists → `02` geodesic `k`-sweep ∥ `03` `local_dimension.py` ∥ `04` `confidence_band.py` ∥ `05` `template_immersion.py` + Jacobian rank ∥ `06` `template_decision.py` abstain conditions and tally → `07` H2 budget calibration at the real `D = 768` → `08` **[blocking `checkpoint:decision`]** `02.7-SCREENING-RULE.md` committed **alone**, ancestry asserted → `09` **[checkpoint:human-verify]** the CLAUDE.md Swiss roll notebook with in-library `S^1`/`T^2` positive controls → `10` **[checkpoint:human-verify]** the benchmark grid and its three-way tally → `11` **[checkpoint:human-verify]** the read-only PU probe → `12` **[checkpoint:human-verify]** `02.7-FINDINGS.md` with the ancestry re-check. Plans `01`, `08`, `09`, `10`, `11`, `12` are non-autonomous. Wave 2's five plans have zero `files_modified` overlap and run in parallel; every later wave is serialized by the ratification ordering.

**The single hardest sizing fact, measured rather than assumed.** `02.7-RESEARCH.md` timed unthresholded `ripser(..., maxdim=2)` on real clouds this session: `S^2` costs `1.85 s / 8.59 s / 26.55 s` at `n = 200/300/400` and does not finish inside a 90-second wall at `n = 500`; `T^2` reaches `74.66 s` by `n = 800`. Cost grows roughly as `n^4` and is **template-dependent**, so `n_ph` is sized against `S^2` rather than an average. Those numbers were taken at `D = 3`, so plan `07` re-measures at the real `D = 768` with the real warp **before** plan `08` ratifies `n_ph` and `B` — because assumption **A-05**, that the pinned budget may not resolve `H_2` on `S^2`/`T^2` at all, would mean the benchmark measures the budget rather than the front end. RESEARCH.md's worked grid arithmetic comes to roughly **8.25 hours for the `H_2` degree alone**; plan `08` resolves that as an explicitly recorded choice, never a silent reduction, and plan `10`'s runner is resumable and per-cell timed so a multi-hour grid cannot lose its work or hide its cost.

**Two source corrections carried into the plans, both verified against `src/effdim/geometry.py` and measured at planning time.** First: **`tle_dimensionality` is mathematically identical to `mle_dimensionality`** — the same expression, the same `1e-10` epsilon, the same `np.mean` reduction — returning bit-identical values in three regimes including `D = 768`. The library's TLE is Levina-Bickel MLE under another name, not Amsaleg's Tight Localities Estimator. Second: **`two_nn_dimensionality` and `mind_mli_dimensionality` are invariant in `k` by construction**, reading only the leading two and the leading one columns of `precomputed_knn_dist_sq` respectively; slicing to different widths returns bit-identical values, measured constant across `k in {5,10,20,30}`. Together with `gmst_dimensionality`, which accepts neither `precomputed_knn_dist_sq` nor `k` (already flagged in RESEARCH.md), that makes **three of eight estimators unable to vary with `k`** — and two of them register a *perfect* plateau at every `k` for reasons that have nothing to do with the data. Consequence for D-09/D-10: the reported spread is over **seven** distinct estimators with one vote doubled, and a naive "majority of the eight plateau at the same value" can be carried by vacuous plateaus. Both live in frozen code this milestone may not fix; both are pinned as regression tests in plan `03` (`test_tle_is_identical_to_mle`, `test_two_nn_and_mind_mli_are_k_invariant`), and plan `08`'s ratified consensus rule is written against them, blind, before any measurement.

**Assumption-delta:** recorded in `02.7-08` as one **`promote`** — the primary noun generalizes from a *dimension estimate* to an *estimator-spread-over-`k`*, and from a *persistence diagram* to a *(metric, diagram)* pair. `d_hat` survives only as the conditional output of D-10's consensus rule, which returns `None` plus a reason rather than a number the estimators do not support. Encoded by the invariant test `test_no_readout_without_metric_label`.

**Plans**: 9/12 plans executed
**Wave 1**

- [x] 02.7-01-PLAN.md — **[tracer + checkpoint:decision]** one `S^1` cloud end to end at `D = 768`, both metrics, `maxdim=2`, per-metric band, Betti lookup; the symmetrization regression test; no default for any ratifiable threshold in any module (D-01, D-02, D-04, D-05, D-07, D-11)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02.7-02-PLAN.md — `geodesic_graph.py`: the `k` sweep, the component curve, dropped fractions, no bridging (D-05, D-06, D-07)
- [x] 02.7-03-PLAN.md — `local_dimension.py`: eight estimators over `k` with no aggregation, plateau-consensus, anchor-neighbourhood local estimates, and both frozen-estimator corrections pinned (D-09, D-10, D-12)
- [x] 02.7-04-PLAN.md — `confidence_band.py`: the Fasy per-cloud per-metric bootstrap band, and the `beta_1 = 0` disc case a gap cut cannot return (D-02)
- [x] 02.7-05-PLAN.md — `template_immersion.py`: four templates, orthogonal lift, named warp, Jacobian rank checked at `D = 768`, negative control (D-14, D-15)
- [x] 02.7-06-PLAN.md — `template_decision.py`: joint `(Betti, d_hat)` key, four named abstain conditions, three-way tally, metric-label invariant (D-01, D-03, D-04, D-13)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02.7-07-PLAN.md — `ph_budget_calibration_run.py` + `02.7-BUDGET-CALIBRATION.md`: `H_2` cost, memory, one complete grid cell and the `H_2` power check, measured at the real `D = 768` (D-08, D-11, D-15)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 02.7-08-PLAN.md — **[checkpoint:decision]** `02.7-SCREENING-RULE.md` committed **alone**, ancestry asserted, every constant ratified blind (SC-1, D-01..D-16)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 02.7-09-PLAN.md — **[checkpoint:human-verify]** the CLAUDE.md Swiss roll notebook with in-library `S^1`/`T^2` positive controls, a live test of abstain condition (c) (SC-5, D-16)

**Wave 6** *(blocked on Wave 5 completion)*

- [ ] 02.7-10-PLAN.md — **[checkpoint:human-verify]** `template_benchmark_run.py` + `02.7-BENCHMARK-RESULTS.md`: the ratified grid with `D = 768` executing, budget parity asserted in code (SC-4, D-13, D-15)

**Wave 7** *(blocked on Wave 6 completion)*

- [ ] 02.7-11-PLAN.md — **[checkpoint:human-verify]** `pu_template_probe_run.py` + `02.7-PU-PROBE.md`: one read-only PU probe, refitting nothing (SC-6)

**Wave 8** *(blocked on Wave 7 completion)*

- [ ] 02.7-12-PLAN.md — **[checkpoint:human-verify]** `02.7-FINDINGS.md`: the ancestry re-check against every scored commit, SC-1..SC-6 answered, the claim bounded

### Phase 3: Decoder & Curvature Field

**Goal**: A per-point mean-curvature field over the PU manifold, computed by autodiff through a C2-smooth CAE chart decoder, checked on a known-answer manifold before it is computed on PU and against a synthetic control after.

**Substrate**: the **Chart Auto-Encoder**, fixed by user decision 2026-08-12. Selection is tabled, not resolved — the choice rests on readiness and a clean defect ledger, **not** on measured superiority. Full record, including the evidence against: `phases/02-eigenspectrum-audit-validity-gate/02-NOTE-phase-2-stage-on-hold.md`.

**Depends on**: nothing outstanding. The phase's original precondition — a **PASS** from 02.4 — is **deliberately overridden**; no PASS exists anywhere in this milestone (02, 02.2, 02.4, 02.5 stage 1 are all FAIL). The override and its consequence must be recorded in this phase's own artifacts, never inherited as a silent green light.

**Scope**: curvature is **local**. `II_p` depends only on a neighbourhood of `p`, so the milestone's global-scoped FAILs do not block a piecewise field (`02.4-FINDINGS.md`). No global parameterization is attempted and none is claimed.

**Requirements**: DEC-01..05 and CURV-01..08 are **stale** — written against Isomap coordinates and a global chart. Re-mint at plan time; do not re-point.

**Start simple.** Each step is cheap, readable, and gates the next. Nothing below is pre-registered; add gate machinery only when a step's result would otherwise be untrustworthy.

  1. **Swiss roll first, known answer.** Existing `chart_curvature.py` + a fresh 2-chart CAE fit on the roll, curvature vs analytic `H` by Spearman. `02.5-09` measured `-0.0604` here against a raw-point baseline of `0.6712` — reproduce that measurement first, then try to beat it. **If it cannot clear the raw-point baseline, Phase 3 stops here and reports** — that is a complete outcome, not a failure to work around.
  2. **PU curvature field, one fit, one seed.** Sealed 02.2 CAE fit read-only or one fresh fit; `torch.func` Jacobian/Hessian per chart; emit `‖H‖` per point plus metric-tensor conditioning. Descriptive only — a histogram and a map, no verdict.
  3. **Seeds and sanity.** Repeat across ≥3 seeds, report spread. Flag near-singular metric points; confirm second derivatives are non-zero and finite; evaluate only at/near data, never extrapolated.
  4. **Synthetic control.** Same architecture and protocol fitted to flat plane / sphere / saddle at matched `d` and ambient 768. States what the PU numbers can and cannot mean.

**Success criteria**

  1. A curvature Swiss roll check passes with the CAE chart decoder, or the phase reports the failure and stops (step 1).
  2. `‖H‖` per point over the PU cloud, from batched `torch.func` autodiff, labelled as a mean-curvature **vector norm** — never Gaussian or principal curvature.
  3. C2-smooth activation throughout the decoder, asserted in code (`decoder_curvature.assert_c2_decoder`); ReLU's second derivative is identically zero and silently zeroes `II`.
  4. Seed spread reported, not a single draw; near-singular metric points flagged rather than averaged in.
  5. Synthetic control completed and reported **before** Phase 4 starts, with the parameterization-damage caveat stated plainly: a synthetic manifold that passes never reproduces the pathology the override carries.

**Assets to reuse, not rebuild**: `chart_curvature.py` (exact chart curvature via `torch.func`, 02.5-08), `decoder_curvature.py` (curvature with no chart routing + C2 guard, 02.6-01), `derivative_bridge.py` (autodiff vs finite-difference agreement, 02.6-10), `cae.py` (sealed — import, never edit), `notebooks/02.5_swiss_roll_chart_curvature_check.ipynb`.

**Constraints**: `src/effdim/` and `pyproject.toml` frozen this milestone. Sealed verdicts and sealed fits are read-only. `d_frozen = 5` is **not** inherited — state and justify the working dimension. `k* = 15` against `d(d+1)/2` coefficients is badly underdetermined at `d` in 18–25; if the decoder route is used, that constraint is the reason.

**Research**: needed at plan time for the high-codimension mean-curvature math — first/second fundamental form, `‖H‖`, batched `vmap` Jacobian/Hessian — easy to get subtly wrong on shapes and index conventions.

**Status**: **SEALED 2026-08-23. 11/11 plans. The curvature field is NOT validated** — that is
the phase's outcome, not a failure to work around. `CURV-07` answers "neither established": the
instrument is correct at `d=4` (`rho = 0.989`) but the field does not reproduce across seeds
(`||H||` median spans **52x** over three converged draws; two of the three are numerically
degenerate). The phase's most transferable result is the `cond(g)` -> artifact-curvature band
table, monotone across four decades. Record: `03-FINDINGS.md`, `03-11-SUMMARY.md`, and
`03-FINDINGS-SUPPLEMENT-01.md` (which withdraws one supporting clause in §6 point 3 after spike
003 showed the `d=20` saddle control is unrankable by construction; the conclusion is unchanged).

**Plans**: 11/11 plans executed

Plans:
**Wave 1**

- [x] 03-01-PLAN.md — Tracer: declare the Step-1 floor and the `n_charts` scope ruling, build the roll sweep runner, prove the whole path end to end (CURV-01, CURV-03, DEC-05)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-02-PLAN.md — Run the 4×5 Swiss roll sweep, apply the floor to the best config, decide the gate or take the D-05a branch (DEC-05, CURV-03, CURV-04)
- [x] 03-03-PLAN.md — Close WR-01/02/03 in `derivative_bridge.py` with regression tests (CURV-05)
- [x] 03-04-PLAN.md — `synthetic_controls.py`: flat / sphere / saddle at `d=20, D=768`, saddle cross-checked against finite differences (CURV-06)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03-05-PLAN.md — D-08 forward-mode toggle (spike first, reverse stays default) and D-09 equivalence, reverse path pinned bit-identical (CURV-01, CURV-02)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03-06-PLAN.md — CLAUDE.md-mandated `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb` (DEC-01, DEC-02, CURV-03)
- [x] 03-07-PLAN.md — Build the PU sweep runner: `chart_dim=20` justified, timing probe, four D-07 diagnostics, selection rule declared up front (DEC-01, DEC-03, DEC-04)

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 03-08-PLAN.md — Execute the 9-fit PU grid, apply the pre-declared rule, decide D-12 escalation (DEC-03, DEC-04, DEC-05)

**Wave 6** *(blocked on Wave 5 completion)*

- [ ] 03-09-PLAN.md — Steps 2–3: PU `‖H‖` field across 3 seeds, `cond(g)` distribution with percentile flagging, no-extrapolation proof, bridge at PU scale (CURV-03, CURV-04, CURV-05, CURV-08)

**Wave 7** *(blocked on Wave 6 completion)*

- [ ] 03-10-PLAN.md — Step 4: matched synthetic control fits, four-axis comparison, parameterization-damage caveat beside the numbers (CURV-06, CURV-07)

**Wave 8** *(blocked on Wave 7 completion)*

- [ ] 03-11-PLAN.md — `03-FINDINGS.md`, the presentation notebook, and the 13-requirement re-mint (DEC-01..05, CURV-01..08)

**Next**: `/gsd-execute-phase 3`.

**History**: this entry was rewritten 2026-08-12 to drop four superseded amendment layers (Isomap decoding, the 02.2 PASS precondition, the 02.1 graph-native rewrite of DEC/CURV, the 02.4 wait) and the 2026-08-07 local re-scope note, all of which are now either resolved by the substrate decision or restated above. Prior text is in git history and `02-NOTE-phase-2-stage-on-hold.md`; no sealed verdict or measured number is changed by the rewrite.

### Phase 03.1: Decoder Metric Regularization (INSERTED)

**Goal**: Make the CAE produce a decoder parameterization whose pullback metric is usable for
curvature — well-scaled and well-conditioned — and measure the effect on a fixture with known
analytic curvature, so the answer is not another unfalsifiable field.

**Depends on**: Phase 3 complete. Inherits its validated instrument (`chart_curvature.py`
recovers analytic mean curvature at `d=4` to `rho = 0.989`, `R² = 0.980`) and its two diagnosed
defects.

**Why this phase exists.** Phase 3 delivered a curvature field that does not survive its own
declared reporting unit: across three converged seeds the `‖H‖` median spans **52×**
(1.36e+03, 5.14e+04, 7.08e+04), and **two of the three fields are piecewise-constant** — one
value per chart — produced by metrics whose entire spectrum collapsed to `~1e-07`
(`det(g) ~ 1e-162`, `‖J‖_F ~ 1e-03`). Two root causes, both measured:

1. **Nothing in the objective constrains the decoder's derivatives at any order.**
   `cae.lipschitz_penalty` regularizes `chart_encoders`; curvature is decoded through
   `chart_decoders` composed with `embedding_decoder`; the two sets share no parameter.
   Removing total-loss early stopping cut held-out reconstruction **62.2%** and left `cond(g)`
   **unmoved** — a C0 objective cannot bound a C2 quantity.

2. **The conditioning diagnostic is blind to the failure.** `cond(g) = λ_max/λ_min` is
   scale-invariant, so a uniformly collapsed metric scores a *perfect* condition number. On the
   real fits it **ranked the two degenerate seeds ahead of the only healthy one** (1.0e+03 and
   1.8e+03 against 9.6e+06). CURV-04 is reopened for this.

**Two arms, because the two failures are different and each candidate penalty is blind to one
of them.** Measured under a uniform rescaling `J -> cJ`:

| penalty | behaviour under collapse | targets |
|---|---|---|
| `christoffel` (C2, tangential `D²F`) | **exactly invariant** | anisotropy / fragmentation |
| `conformal` | **minimized by collapse** — rewards it | anisotropy only |
| `isometry` | saturates at `‖I‖²_F = d`, parameter gradient → 0 | both, weakly |
| `scale` (`(log det g / d)²`) | **diverges**, gradient `∝ g⁻¹` | absolute scale |

So the phase measures **`scale`** (against the collapse that killed seeds 14/15) and
**`christoffel`** (against the anisotropy seed 13 exhibits, `λ_min = 1e-07` at `λ_max = 3.35`),
separately and in combination — never collapsed into one weight.

**Scope notes.** `decoder_priors.py` already implements both modes, opt-in and default-off, via
a contextmanager that never edits `cae.py` (the sealed 02.2 architecture). The Christoffel term
is proven by test not to bias the estimand — it penalizes only the tangential part of `D²F`,
never the normal part, which is the curvature. So this phase is **measurement, not
implementation**: a weight ladder, not new machinery.

**The test bed is the `d=20` sphere or saddle, not PU.** Both have analytic `H` *and* currently
fail, so "does the prior help?" is directly readable — cosine, magnitude ratio, `R²` and `rho`
moving toward 1 as the metric improves — rather than another field nothing can check. PU comes
after, if and only if a control clears.

**Also in scope**: close CURV-04 by recording `λ_min` / `λ_max` / `det(g)` alongside `cond(g)`
in the runners. No existing record can be re-audited for the collapse, because only the ratio
was ever stored.

**Not in scope**: reopening any sealed verdict; the phase-2 stage stays on hold; `cae.py` is not
edited.

**Plans:** 5/5 plans executed
architecture, so no two plans may be co-scheduled.

**Outcome (sealed 2026-08-21).** `scale`'s Tier-1 verdict is **MECHANISM DEMONSTRATED** — it
repairs the pullback metric completely (`log10_det_g -83.9 → +0.037`, `cond(g) 1.7e8 → 5.7e2`) at
*negative* reconstruction cost. `christoffel`'s Tier-1 verdict is **MECHANISM NOT DEMONSTRATED**
under this ladder's rung resolution (F1's relief cut removed the resolution needed to locate its
`cond(g)` minimum), though it halves `cond(g)` again in the post-hoc combination. Tier 2 (the
estimand) moved only partially and not seed-consistently: rank `rho` from `-0.122` to at most
`+0.116`, far short of the `rho = 0.989` this same fixture yields at `d=4`. **Regularizing the
decoder's parameterization is necessary but not sufficient — it does not recover curvature
ordering at `d=20`.** CURV-04 is closed (`03.1-FINDINGS.md` §8). Full record:
`03.1-FINDINGS.md`.

Plans:
**Wave 1**

- [x] 03.1-01-PLAN.md — CURV-04's absolute-scale instrumentation in `chart_curvature.py`, and the end-to-end tracer: one prior-active cell measured and recorded at smoke scale

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03.1-02-PLAN.md — nested `decoder_prior_active` composition proven by test (D-12), and the mandatory Swiss roll decoder-prior check notebook (D-14, one-way gated)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03.1-03-PLAN.md — the exact-equality faithfulness anchor, the two-sided cost probe (training per mode, curvature evaluation per row), and a blocking ratification of the run sizing

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03.1-04-PLAN.md — the D-08 two-tier read-out implemented and proven on a synthetic record, then the ladder: fresh `weight=0` baseline plus the `scale` and `christoffel` arms

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 03.1-05-PLAN.md — the post-hoc best-of-each combination cell, `03.1-FINDINGS.md`, and CURV-04's closure

### Phase 4: Region Partitioning & Regional Alignment (MKNN)

**Goal**: With all upstream hyperparameters (`n_neighbors`, `d`, decoder architecture, curvature quantile threshold) frozen from Phases 1-3's own diagnostics and the synthetic-control falsification test complete, points are pre-specified into density-checked high/low curvature regions and crossmodal MKNN alignment compared between them against region-specific permutation nulls and bootstrap CIs.
**Depends on**: Phase 3 (requires the synthetic-control falsification test (CURV-06, CURV-07) to have already completed). **Still gated on Phase 03.1's outcome (sealed 2026-08-21, not resolved in Phase 4's favor)**: Phase 03.1 measured whether decoder-parameterization regularization fixes the ordering defect Phase 3's handoff flagged, and it does not, at the dimension tested — `scale` repairs the metric completely but moves rank `rho` only to `+0.116` (combination `+0.013`) against the `-0.0151` Phase 3 measured, far short of the `rho = 0.989` the same fixture yields at `d=4`. **Phase 4 stays blocked, with no proposed route out (D-11).** The developer-directed next step is a local-polynomial geometry-teacher spike scoring `(P̂, ÎI)` against the same four axes on the `d=20` saddle, to test feasibility before any architecture change — not yet run, and not itself a Phase 4 precondition change. See `03.1-FINDINGS.md` §10 and `03-FINDINGS.md` §9.
**UNBLOCKED FOR PLANNING 2026-08-23.** D-11 ("Phase 4 stays blocked, no route out proposed") is **discharged** — the paragraph above is retained as the historical record of why it was raised, not as current status. D4-02 Amendment 02 supplies the route out: the instrument is `curvature_probe.centroid_mean_curvature` applied directly to the point cloud, which forms no pullback metric and cannot suffer the `cond(g)` pathology, so Phase 03.1's metric regularization is **optional, not blocking** and Phase 3's non-reproducing decoder field is off the critical path. **What is unblocked is planning, not claiming:** the field remains unvalidated (`CURV-07` answered negatively) and Phase 4's record must state that, the codimension gap, and the density confound in its own words. See `STATE.md` § Phase 4 and `phases/03-decoder-curvature-field/03-NOTE-phase-4-decisions.md`.

**Requirements**: REGN-01..06, MKNN-01..08 — REGN-01/03/04 **re-minted** and REGN-06 **added** 2026-08-23 by D4-11; see `REQUIREMENTS.md` § Phase 4 Requirement Re-Mint
**Success Criteria**:

  1. Local sample-density measure per point ~~in Isomap coordinate space~~ **in the ambient 768-d embedding space the curvature field is estimated in** (re-minted 2026-08-23 by D4-11/D4-13 — the estimator runs directly on the normalized embeddings, so the density that could masquerade as curvature is 768-d density) and its correlation with curvature shown explicitly, before any region split is trusted (REGN-01, REGN-02)
  2. **SUPERSEDED 2026-08-23 by D4-01** (`phases/03-decoder-curvature-field/03-NOTE-phase-4-decisions.md`). Points partitioned by clustering curvature **DIRECTION** `H/||H||`, not by `|H|` quantiles — at `d=20` direction is recoverable (cosine 0.77–1.000) while magnitude is ~50x attenuated and its ordering saturates at `rho ~ 0.5–0.65` (spike 003). The partition is still **pre-specified and frozen** before any regional MKNN number is computed; only the quantity being partitioned changes. ~~Direction-clustering must first be validated against `varying_ii_controls.make_ridge_graph_control`, whose single bending direction `w` is an exact known answer.~~ **OVERRIDDEN 2026-08-23 by D4-10**: no known-answer fixture validation runs before the PU split is frozen. Developer rationale, recorded: narrowing codimension 1 to at most 8 against PU's ~748 risks reading as closure rather than narrowing, and D4-01 was already adopted on partial evidence with that gap named. The fixtures exist and are tested; the check remains available later without touching anything Phase 4 produces. Original criterion, for the record: *points partitioned into high/low-curvature regions by a pre-specified quantile threshold, each region's point count shown* (REGN-03..05)
  3. MKNN score between two row-aligned embeddings computed as k-normalized k-NN intersection size, matching the origin paper; global crossmodal HSC-vs-Legacy-Survey MKNN number reproduced and compared against the origin paper's published range (MKNN-01, MKNN-02)
  4. Per-region MKNN score for high/low-curvature regions, each with its own permutation null computed within that region's index set (never reused from a global null) and bootstrap CIs (MKNN-03..05)
  5. Whether the high-vs-low result holds across k = 5, 10, 20, 50 shown, with an explicit verdict on whether the regional difference is distinguishable from noise ("no detectable difference" is a valid outcome), hubness caveat for k-NN-based alignment metrics stated alongside results (MKNN-06..08)

**Decisions taken 2026-08-23** (`phases/03-decoder-curvature-field/03-NOTE-phase-4-decisions.md`): **D4-01** partition on curvature direction, not `|H|` quantiles — **adopted on PARTIAL evidence**: the partition-fidelity validation was built then deliberately scoped out (both schemes read the same field at the same points, so location error cancels). **Unclosed codimension gap:** every spike 003 fixture is a codimension-1 graph where `H = H_scalar * n_hat`, so "direction" IS the surface normal and the cosine 1.000 result shows normal-ORIENTATION recovery, not resolution within a normal space; PU's codimension is ~748. **D4-02 RESOLVED 2026-08-23 to the point-cloud `centroid_mean_curvature`** — three cells, both arms on identical data at `d=20`: cloud `rho` +0.41..+0.61 with cosine +0.77..+0.92 in **2s**, decoder `rho` +0.002..+0.018 with cosine ~0 (twice negative) in ~358s, its magnitude inflated 12,000-42,000x consistent with `cond(g)` 4e11-1.6e12. *Caveat: the decoder arm is undertrained vs Phase 3's sealed fits (mse_per_dim 0.23-0.32 against 1.6e-02), so this is not a clean disqualification of a well-trained decoder — but 200->400 epochs moved its `rho` only +0.0019 -> +0.0072.* **Consequence: Phase 03.1's metric regularization is OPTIONAL, not blocking**, and Phase 3's non-reproducing field is off the critical path. `k` becomes the main free parameter (hundreds at `d=20`). **D4-03** PU split-half reliability (`R_H = 0.589` at `k=231`) is accepted as sufficient; cross-estimator agreement on PU is NOT required. That is a **deliberately accepted blind spot** — split-half reliability cannot detect a bias both halves share (measured: `R_H = 0.990` with `rho = 0.469` on the Swiss roll) — and any Phase 4 result inherits an unvalidated field, which Phase 4's record must state in its own words.

**Plans**: 6/6 plans executed

Plans:
**Wave 1**

- [x] 04-01-PLAN.md — TRACER: fill `mknn.py`'s three stubs on a shared k-NN-membership-matrix architecture and produce the global crossmodal MKNN end to end (MKNN-01, MKNN-02, MKNN-05, MKNN-08)
- [x] 04-02-PLAN.md — Density-corrected `R_H` sweep past k=231 and the D4-07 `k` freeze, rule declared before the first corrected number (REGN-04)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 04-03-PLAN.md — The diametrical sign-split helper with a known-answer test, then the blocking pre-registration checkpoint and freeze (REGN-03, REGN-04, MKNN-07)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 04-04-PLAN.md — PU field at the frozen `k`, the sign split frozen as an artifact, and the density confound reported (REGN-01, REGN-02, REGN-05, REGN-06)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 04-05-PLAN.md — The eight-cell regional MKNN grid with region-scoped nulls, bootstrap CIs, hubness, and the pre-registered verdict (MKNN-03..08)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 04-06-PLAN.md — The phase record: three accepted gaps in the phase's own words, the Swiss-roll reasoning, and honest requirement outcomes (REGN-02, MKNN-07, MKNN-08)

**Research**: Completed 2026-08-23 — `phases/04-region-partitioning-regional-alignment-mknn/04-RESEARCH.md`. The density-confound battery is **descoped to the REGN-02 correlation only** by D4-14 (no centroid-distance check, no partial regression, no density-matched stratification or null), so the original-synthesis risk this line flagged no longer applies; the accepted consequence — a regional MKNN difference cannot be separated from a regional density difference by anything in this phase — is carried in the phase record instead.
**Ordering constraint**: Pre-specify the split, then compute. All upstream hyperparameters and curvature quantile threshold must be frozen using upstream-only diagnostics from Phases 1-3 *before* the first regional MKNN number is computed — a garden-of-forking-paths guard against post-hoc tuning on a headline effect with thin statistical headroom (**0.34%-2.25%** in the origin paper — arXiv:2509.19453 Table 2, Legacy-vs-HSC column, read directly from the PDF by the 2026-08-23 research pass; supersedes the rounded "0.4-2%" written here originally). **The freeze covers the curvature-field `k`, the partition rule, and MKNN-07's verdict rule, all written into the notebook and into committed source before the first regional MKNN number exists.** The global MKNN-02 reproduction is region-blind and partition-blind and is therefore not gated by this constraint.

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

**Execution Order:** Phases 1 -> 2 -> 02.1 -> 02.2 -> 02.4 ran and are closed. The phase-2 stage (02.3, 02.5, 02.6, 02.7) is **ON HOLD from 2026-08-12** — architecture selection tabled, nothing further scheduled there. Phase 3 ran on a CAE substrate under a deliberate override of its PASS precondition and **closed 2026-08-23 with the field NOT validated**; Phase 03.1 repaired the pullback metric without moving rank correlation and closed 2026-08-22. **Phase 4 remains BLOCKED** (D-11: no route out proposed by Phase 3 itself). Spike 003 (2026-08-22) proposes candidate routes; none is ratified.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Data Loading & Manifold Reconstruction | v1.1 | 4/4 | Complete | 2026-07-31 |
| 2. Eigenspectrum Audit & Validity Gate | v1.1 | 3/3 | Complete (FAIL verdict) | 2026-08-05 |
| 02.1. Geometry Representation Research (INSERTED) | v1.1 | 4/4 | Complete (graph-native) | 2026-08-05 |
| 02.2. Chart Autoencoder Validity Test (INSERTED) | v1.1 | 6/6 | Complete    | 2026-08-04 |
| 02.3. Chart Auto-Encoder Iteration (INSERTED, proposed) | v1.1 | 0/TBD | Proposed — not planned | - |
| 02.4. Topological Auto-Encoder Validity Test (INSERTED) | v1.1 | 8/8 | Complete    | 2026-08-07 |
| 02.5. Local Curvature Feasibility & CAE Re-Gate (INSERTED) | v1.1 | 9/13 | On hold 2026-08-12 | - |
| 02.6. Decoder Substrate Screening (INSERTED) | v1.1 | 15/15 | Complete (no substrate promoted) | 2026-08-11 |
| 02.7. Manifold-Template Inference Front End (INSERTED) | v1.1 | 10/12 | On hold 2026-08-12 | - |
| 3. Decoder & Curvature Field | v1.1 | 11/11 | Complete (field NOT validated) | 2026-08-23 |
| 03.1. Decoder Metric Regularization (INSERTED) | v1.1 | 5/5 | Complete (metric repaired, ordering unmoved) | 2026-08-22 |
| 4. Region Partitioning & Regional Alignment (MKNN) | v1.1 | 6/6 | Complete    | 2026-08-24 |
| 5. Curvature-Conditioned Linear Decodability | v1.1 | 6/6 | Complete (SPLIT ACROSS SEEDS) | 2026-08-24 |
| 6. Point-Cloud Curvature-Conditioned Linear Decodability | v1.1 | n/a | Complete (NO DETECTABLE RELATIONSHIP) | 2026-08-24 |
</content>

### Phase 5: Curvature-Conditioned Linear Decodability

**Goal:** Measure whether linear crossmodal decodability degrades as decoder-side manifold curvature
magnitude increases — one global ridge map `hsc -> legacysurvey` on frozen PU embeddings, held-out
per-point residuals bucketed independently by each of three per-seed decoder-side `||H||` fields,
judged per seed under a rule frozen before any PU probe number exists and combined into one phase
read-out by a frozen combination rule. *(Amended 2026-08-24: the seeds are NOT pooled. See
`05-03-DECISION.md`, which superseded `05-CONTEXT.md` D5-04 at the `05-03` Task 1 one-way blocking
checkpoint on measured inter-seed disagreement.)*
**Requirements**: D5-01 .. D5-13 (no milestone REQ-IDs were minted for this phase; `05-CONTEXT.md`'s
thirteen locked decisions are the de-facto requirement set, following Phase 02.5's precedent)
**Depends on:** Phase 4
**Plans:** 6/6 plans complete

Plans:
**Wave 1**

- [x] 05-01-PLAN.md — Tracer: the whole machine wired end to end on planted data, constants unset so the bucketed path is provably dead

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 05-02-PLAN.md — Extract three seeds' decoder-side curvature fields (~2.6 h, per-seed cached) and measure inter-seed agreement with its direction axis

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 05-03-PLAN.md — Carry the ratified no-pooling decision into the code: restructure the pre-registration block for three verdicts (constants still unset), build three per-seed bucketings, re-measure the density confound per seed

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 05-04-PLAN.md — The pre-registration freeze: all 31 constants, the full VERDICT_RULE and the SEED_VERDICT_COMBINATION_RULE defining what a split outcome means, in committed source plus 05-PREREGISTRATION.md

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 05-05-PLAN.md — The bucketed probe run: one global fit, three per-seed bucketings, three per-seed verdicts and one phase verdict, all applied mechanically from the committed rules

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 05-06-PLAN.md — Executed notebook, 05-FINDINGS.md reporting three verdicts and their spread, and the mechanical git-ancestry proof of the ordering guarantee

### Phase 6: Point-Cloud Curvature-Conditioned Linear Decodability

**Goal:** Re-run Phase 5's probe with the curvature field swapped for the training-free
point-cloud estimator, changing nothing else. One global ridge map `hsc -> legacysurvey` on the
same frozen PU embeddings, the same 70/30 split, the same `RIDGE_ALPHA_GRID` and selection rule,
the same per-point squared-L2 residual metric, the same tertile bucketing — but bucketed by the
**density-corrected `centroid_mean_curvature` field at Phase 4's frozen `K_FROZEN = 500`**
(already computed and sealed as `h_norm` in `notebooks/.cache/04_region_partition.npz`) instead of
the three CAE decoder-side `||H||` fields. Judged under a single verdict rule frozen before any
Phase 6 number exists.

**Why inserted:** three independent reasons, all from the sealed record. (1) **Consistency** —
Phase 4 already partitions on `centroid_mean_curvature`, density-corrected, at `k = K_FROZEN = 500`
(`04-FINDINGS.md`); Phase 5 reverted to the CAE decoder route. (2) **The decoder route is not
reproducible** — `05-03-DECISION.md` measured the three seeds' fields mutually anti-correlated on
rank (pairwise Spearman `-0.1402`, `+0.2019`, `-0.2725`) and directionally orthogonal (median
cosine `0.0007`–`0.0039`, 46–48% of points anti-aligned). The point-cloud estimator has no
training and no seeds, so `SPLIT ACROSS SEEDS` cannot arise as an outcome. (3) **It closes D4-08**
— cross-estimator agreement was recommended at the Phase 3 close and declined twice (D4-03,
D4-08); running the probe on both fields over the same held-out residuals measures it directly.

**What this phase does NOT claim:** the point-cloud field is validated only by split-half
reliability on PU (`04-FINDINGS.md` Gap 1), which cannot detect a bias both halves share —
measured on the Swiss roll at `R_H = 0.990` alongside `rho = 0.469`. Phase 6 inherits that gap
verbatim and does not close it.

**Requirements**: D6-01 .. D6-NN (`06-CONTEXT.md`'s locked decisions are the de-facto requirement
set, following Phase 5's and Phase 02.5's precedent)
**Depends on:** Phase 4 (frozen `K_FROZEN`, density correction, and the sealed `h_norm` field),
Phase 5 (frozen split, ridge protocol, residual metric, and bucketing machinery in
`notebooks/pu_manifold/linear_probe.py`)
**Outcome (2026-08-24): `NO DETECTABLE RELATIONSHIP`.** Fails on criterion (a) alone — the
highest and lowest tertile's 95% bootstrap CIs overlap by `0.000914`; criteria (b) and (c) both
hold. The pattern is **not monotone** (bucket 1 sits below bucket 0; only the top tertile is
elevated). Protocol inheritance is exact and checkable: `mean_residual_overall =
0.06642936194948156`, byte-identical to Phase 5's, so both phases scored the same 3,000 held-out
residuals. **The answer is instrument-dependent** — Phase 5 returned `SPLIT ACROSS SEEDS` on the
same residuals. Neither verdict upgrades or downgrades the other.

**D4-08 closed, with a negative answer.** Spearman between Phase 4's point-cloud field and the
three Phase 5 decoder fields: `-0.0875`, `+0.0487`, `-0.1177` — all below `|0.12|` and
sign-inconsistent. The two instruments are not two measurements of one quantity.

**Plans:** executed without formal plan files (autonomous run at the developer's standing
instruction). Artifacts: `06-CONTEXT.md`, `06-PREREGISTRATION.md`,
`06-PREREGISTRATION-AMENDMENT-01.md`, `06-FINDINGS.md`, `06-VERIFICATION.md`.

Plans:

- [x] Freeze (`c11218c`) — 32 pre-registration constants and `VERDICT_RULE` in committed source
- [x] Runner (`37d1ba8`) — `--selfcheck` and `--mode bucketed`, with the D6-01 provenance guard
- [x] Amendment 01 (`62dc10a`) — `R2_MULTIOUTPUT` restored; phase re-run
- [x] Findings and verification — 9/9 conduct checks reproduced; awaiting developer review

### Phase 7: Curvature-Conditioned Crossmodal Alignment

**Goal:** Answer the milestone's actual research question — **does the curvature of the PU
embedding manifold explain the weak crossmodal convergence reported by the Platonic Universe
paper (arXiv:2509.19453)?** Measure `spearman(||H||_i, MKNN_i)` over all 10,000 points, using a
curvature field from an instrument validated against analytic ground truth, with a positive
control establishing the test's power and density/hubness reported as diagnostics.

**Why this phase exists, and why it is not a fourth re-run.** Phases 5 and 6 both bucketed
*ridge-regression residual* by `||H||` tertiles. That is the wrong outcome variable — the source
paper's probe is **MKNN**, not linear decodability — so neither phase speaks to the claim the
milestone set out to test. Phase 4 did use MKNN but partitioned on curvature *direction*, on an
axis measured `+0.8208` correlated with density, with a raw-score gap its own findings attribute
mostly to region-size imbalance. **The record therefore contains no interpretable answer to the
research question**, and this phase supplies one.

**Two design changes from Phases 4-6, both of which increase power:**

1. **Per-point, not per-region.** `mknn.mknn_score` already computes
   `(A & B).sum(axis=1) / k` — a per-point array — before averaging it away. Retaining it gives
   **10,000 paired observations** instead of 2-3 buckets. Spearman is scale-free, so this also
   sidesteps the near-constant-field problem that makes tertile bucketing underpowered on PU
   (`||H||` spread measured at 1.5x by the plain-AE decoder, 3.94x by Phase 4's centroid).

2. **A validated instrument.** The plain-AE decoder + `decoder_curvature.plain_decoder_curvature`
   scores `rho = +0.9745` (ridge, `D=768`), `+0.9166` (ridge at PU's own low-spread regime) and
   `+0.5253` (cubic, `D=768`) against analytic truth at `d=20` — the first instrument in this
   milestone validated against a known answer at PU's dimension and ambient size.

**The three deliverables (D7-01..D7-03):**

- **D7-01 — the curvature field**, from the frozen instrument. Latent dimension set by measured
  reconstruction (`pu_latent_sweep`), not defaulted to 20; PU's intrinsic-dimension estimates run
  18-25 and the `d=20` fit converged at 98.207% with `cond(g) = 17.6`.

- **D7-02 — the positive control.** Plant a curvature-MKNN relationship **at PU's realized
  `||H||` spread** and show the test recovers it. Without this a null is uninterpretable, and a
  null is the likely outcome. Phase 6's selfcheck does not serve: it planted a ~20x-spread field,
  not PU's ~1.5x.

- **D7-03 — density and hubness diagnostics.** `spearman(density, ||H||)`, the density partial on
  the headline correlation, and `mknn.hubness_skewness`. **Reported, gating nothing** — MKNN is a
  k-NN statistic and therefore mechanically density-sensitive, which is precisely how Phase 4's
  result became uninterpretable.

**What this phase will NOT claim:** that either field measures true curvature (no ground truth
for PU exists; the analytic validation gives a *range*, `+0.53` to `+0.97`, not a point
estimate); that a null means no relationship exists absent D7-02's power evidence; or anything
about CKA, which is not implemented anywhere in the codebase.

**Requirements**: D7-01 .. D7-07 (`07-CONTEXT.md` §3's locked decisions are the de-facto
requirement set, following Phases 5, 6 and 02.5)
**Depends on:** Phase 4 (`mknn.py`, and its density-confound cautionary record), Phase 6
(`linear_probe.py` split/CI/verdict machinery), and the 2026-08-25 instrument-validation work
**Plans:** 5/5 plans complete

Plans:
**Wave 1**

- [x] 07-01-PLAN.md — Ratify the six open decisions blind, then commit the pre-registration constants block as THE FREEZE COMMIT (D7-06)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 07-02-PLAN.md — Tracer: one `d`, one verdict, end to end — the runner, the per-point MKNN gap-fill (D7-04), and the two-tailed permutation wrapper

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 07-03-PLAN.md — The D7-02 positive control at PU's realized `||H||` dynamic range, and the D7-03 density/hubness diagnostics that gate nothing

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 07-04-PLAN.md — The real serial `d ∈ {20,25,32}` sweep (~2h), the positive control run, and the mechanically applied verdict

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 07-05-PLAN.md — The reporting notebook (committed with outputs) and `07-FINDINGS.md`

**Cross-cutting constraints:**

- Importing crossmodal_curvature never mutates module-level state in any sealed module (no monkeypatching, no attribute assignment onto mknn / cae / decoder_curvature / curvature_probe / cross_split_curvature), regardless of import order.

### Phase 07.1: Density-Stratified Null and Seed Stability (INSERTED)

**Goal:** Settle whether Phase 7's curvature-MKNN association survives its density confound, and
whether the one cell that does is seed-stable. Phase 7 returned `ASSOCIATION DETECTED` with
`partial_rho_density_controlled` collapsing ~78% of the raw association at `d=20` and ~49% at
`d=25` — and with that residual judged against a threshold built for the **raw** statistic,
because the partial has no pre-registered null of its own.

**Two deliverables:**

- **D7.1-01 — a density-stratified null for the partial.** Permute MKNN *within* density strata so
  the null preserves density structure and breaks only the curvature link, giving
  `partial_rho_density_controlled` a legitimate threshold instead of borrowing the raw statistic's.
  Reuses the frozen `07_crossmodal_curvature_fields.npz` — no decoder retraining. Decides whether
  the `d=20` (-0.02419) and `d=32` (-0.02172) residuals, both sitting within ~20% of the ≈0.0205
  null-band edge, are real.

- **D7.1-02 — seed stability at `d=25`.** `d=25` is the only cell whose density-controlled residual
  (-0.06583, ~3.2x the null-band edge) clears with room. Phase 7 ran
  `SEED_HANDLING_RULE = "single_seed_across_d_sweep"`, an accepted limitation inherited from Phase
  5's measured seed instability. Three seeds at `d=25` alone (~2h, versus ~7h for the full 3x3
  grid) tests whether that one surviving result holds.

**Carries forward from Phase 7:** the freeze-before-compute discipline (D7-06) and the mechanical
verdict rule. Phase 7's CR-01/CR-02/CR-03 runner guards were hardened in-phase (`c92260f`); the
remaining review debt (WR-01, WR-02, WR-04) is unclaimed and may be folded in here.
**Requirements**: D7.1-01, D7.1-02
**Depends on:** Phase 7
**Plans:** 6/6 plans executed

Plans:
**Wave 1**

- [x] 07.1-01-PLAN.md — Freeze 07.1's own gating constants and both verdict rules before any number exists (D-08, D-11, D-14, D-15, D-16)
- [x] 07.1-02-PLAN.md — Close Phase 7 review debt WR-01, WR-02 and WR-04, each under a demonstrated no-op proof (D-17, D-18)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 07.1-03-PLAN.md — Tracer: the whole D7.1-01 path end to end at d=20 on real PU data, plus the null-routine test group (D-01, D-05, D-06, D-07)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 07.1-04-PLAN.md — D7.1-01: the direction-matched positive control, the 3-d x 3-S null grid with its null-mean bias diagnostic, and the verdict (D-02, D-03, D-04)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 07.1-05-PLAN.md — D7.1-02: three serial d=25 fits at three TORCH_INIT_SEED values and the unanimity verdict (D-09..D-13)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 07.1-06-PLAN.md — 07.1-FINDINGS.md and the executed notebook, with the Swiss-roll gate declared satisfied

### Phase 8: Curvature-Conditioned CKA Alignment

**Goal:** Add CKA as a second alignment probe and test whether Phase 7's curvature–MKNN association
is MKNN-specific: split points by `||H||` magnitude into density-matched, equal-n subsets within
density strata, compute unbiased-HSIC CKA on each, and test the tertile-3-minus-tertile-1 difference
against a density-stratified permutation null — with a validation ladder and a frozen unconditional
reporting block beside every verdict.
**Requirements**: D8-01..D8-24 (`08-CONTEXT.md`'s locked decisions are this phase's de-facto
requirement set — `REQUIREMENTS.md` maps no `REQ-` IDs to Phase 8, the same arrangement Phase 7 used
with `07-CONTEXT.md` §3)
**Depends on:** Phase 7, Phase 07.1
**Plans:** 4/6 plans executed

Plans:
**Wave 1**

- [x] 08-01-PLAN.md — Tracer: unbiased-HSIC CKA estimator end to end on synthetic pairs, D8-16's invariance ladder, every constant UNSET

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 08-02-PLAN.md — Within-stratum tertile split, tertile-difference panel, label-permutation null, and the verdict/seed rules
- [x] 08-03-PLAN.md — D8-23 import-purity test, runner production data layer, and the D8-03 global sigma measurement

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 08-04-PLAN.md — The D8-22 freeze: ratify every constant at a blocking checkpoint, then one commit fills all 45

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 08-05-PLAN.md — Real-data run: planted-effect ladder, shuffled-`||H||` calibration, then the 18-cell sweep

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 08-06-PLAN.md — Reporting notebook and `08-FINDINGS.md` with D8-21's frozen block and caveat-bearing verdict

### Phase 9: Curvature-Conditioned Label Decodability (Physics Replication)

**Goal:** Recreate the curvature–decodability experiment on the colleague's `origin/curvature-experiments`
branch with this milestone's instrument. His frozen result: controlled Spearman
`rho(K_H, local OOF R^2 of a global ridge probe for r-band magnitude) = -0.240` (raw `-0.412`) at chart
rank `d=16`, `k=2048` neighbours, `n=512` anchors, on ViT-B Physics embeddings, with `+0.143` at `d=12`
and `-0.233` at `d=20`. Phase 9 asks whether the plain-autoencoder decoder curvature
(`cae.PlainAutoEncoder` + `decoder_curvature.plain_decoder_curvature`, trace convention, Phase 7's frozen
fit protocol, `D_SWEEP=(20,25,32)`) shows the same sign and a comparable magnitude against the same
outcome on the same data, in place of his `k=2048` nested-PCA quadratic-chart estimator.

- **Data.** `UniverseTBD/pu-embeddings` config `physics_vit_base_test` (86,471 rows, single-modality
  `<model>_galaxies`, 768-D) joined to labels from the `Smith42/galaxies` test split (86,471 rows;
  `mag_r` primary; `photo_z`, `smooth_fraction`, `stellar_mass` secondary). pu-embeddings carries no
  ids, so row alignment must be PROVEN inside the phase (his own audit rule: equal row count is not the
  proof). No proof, no Physics number.

- **Outcome.** One global 5-fold out-of-fold ridge probe embedding -> label, frozen, scored locally
  (R^2, MSE, SST) inside k-NN neighbourhoods of anchors; denominator check (SST) and direct-error check
  (MSE) beside R^2 exactly as his `METHODS_FOR_PAPER.md` §9-§11.

- **Controls.** log kNN radius, local label variance, evaluation count; rank-partial Spearman AND
  Phase 07.1's within-density-stratum permutation null; Freedman–Lane FWER across `d`. Both nulls
  reported unconditionally.

- **Radial term.** Report `||H||` and the sphere-tangential `||H_tan||` side by side. His estimator
  removes the sphere-radial component by construction; `08-DIAGNOSTICS.md` §2 measured our `d=25`
  partial collapsing 2.8x under the same substitution. A replication that ignores this is not a
  replication.

- **Gates before any real number.** Planted-effect positive control at the realized `||H||` dynamic
  range, shuffled-label calibration, constants frozen in committed source before any Physics number
  exists (D7-06 / D8-22 pattern). Fit-quality read-out (`var_explained`, `cond(g)`) at every `d`,
  since the instrument was validated at `d=20` on 10,000 Legacy rows, not on Physics rows.

- **Reference material on his branch.** Per-anchor table
  `paper/curvature_neurreps/audit_outputs/multilabel_chart_screen/mag_r_desi/global_anchor_metrics.csv`
  (512 anchors: `K_H_cross`, `log_knn_radius`, `r2_G`, `mse_G`); methods
  `paper/curvature_neurreps/audit_outputs/submission_validation/METHODS_FOR_PAPER.md`; inference code
  `experiments/geometry/physics_curvature_probe_rank_sweep/inference.py`. His `sample_id`s index his
  own `selection.npz` subset, not ours, so a per-anchor instrument comparison is optional and only
  possible if that file is obtained.
**Requirements**: D9-xx (`09-CONTEXT.md`'s locked decisions will be this phase's requirement set, the
arrangement Phases 7 and 8 used)
**Depends on:** Phase 7 (instrument and fit protocol), Phase 07.1 (stratified null), Phase 8
(density-matched machinery)
**Plans:** 1/10 plans executed

Plans:
**Wave 1**

- [x] 09-01-PLAN.md — Tracer: the whole statistical path end to end on synthetic data, every gating constant UNSET
- [ ] 09-02-PLAN.md — Instrument fidelity at `d=16` on the analytic fixtures, plus the phase's API-coverage declaration

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 09-03-PLAN.md — Revision-pinned column-projected loaders for both HF datasets and the row-alignment proof runner

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 09-04-PLAN.md — Full-scale data manifest, and the blocking ruling on the raw-column mapping, sentinels and alignment margin

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 09-05-PLAN.md — The freeze: every constant filled in one commit, `09-PREREGISTRATION.md`, SHA wired and proved from a fresh clone

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 09-06-PLAN.md — Execution-host hand-off: artifact bundling, per-thread cost model, runbook, and a green smoke run on the host

**Wave 6** *(blocked on Wave 5 completion)*

- [ ] 09-07-PLAN.md — The row-alignment proof on the execution host, and the ruling on its outcome (D9-08 adoption branch included)

**Wave 7** *(blocked on Wave 6 completion)*

- [ ] 09-08-PLAN.md — Wave A: the four-`d` sweep, the positive control, the shuffled-label calibration and the verdict print

**Wave 8** *(blocked on Wave 7 completion)*

- [ ] 09-09-PLAN.md — Wave B: three seeds at every fired `d`, combined by the frozen unanimity rule, never pooled

**Wave 9** *(blocked on Wave 8 completion)*

- [ ] 09-10-PLAN.md — Reporting notebook, `09-FINDINGS.md` with the caveat-bearing verdict and its accepted gaps, ROADMAP and STATE
