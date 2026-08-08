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
- [x] **Phase 2: Eigenspectrum Audit & Validity Gate** — Full classical-MDS eigenspectrum audited by hand; a PASS/MARGINAL/FAIL gate freezes the embedding dimension `d` (sealed 2026-08-05, **GATE_VERDICT = FAIL**, `r=0.052419`, `m=0.412071`, `d_frozen=5` — see Phase Details)
- [x] **Phase 02.1: Geometry Representation Research** (INSERTED) — A non-Euclidean-embeddable representation identified and justified against the literature, replacing the Isomap coordinates that Phase 2's gate invalidated (sealed 2026-08-05, **GEOM-04 = Ollivier-Ricci graph-native**; the pre-registered falsifier fired and overturned the coordinate-producing fork — see Phase Details)
- [x] **Phase 02.2: Chart Autoencoder Validity Test** (INSERTED) — The Chart Auto-Encoder method (arXiv:1912.10094) empirically validity-tested on the PU data behind a pre-registered PASS/FAIL gate (completed 2026-08-04, **CAE_VERDICT = FAIL** — see Phase Details)
- [ ] **Phase 02.3: Chart Auto-Encoder Iteration** (INSERTED, proposed — not yet planned) — Investigate why the CAE failed (candidate axes: chart count, chart latent dimension, training budget/epochs, loss weighting, Lipschitz penalty schedule) and produce a fresh, separately-ratified pre-registration before any new fit
- [x] **Phase 02.4: Topological Auto-Encoder Validity Test** (INSERTED) — The Topological Auto-Encoder (Moor et al., arXiv:1906.00722) implemented and put through a pre-registered validity gate on the PU data (sealed 2026-08-07, **TOPOAE_VERDICT = FAIL** — both *global*-scoped gates failed, the *local*-scoped gate passed; see Phase Details and `02.4-FINDINGS.md`)
- [ ] **Phase 02.5: Local Curvature Feasibility & CAE Local Re-Gate** (INSERTED, not yet planned) — Establish empirically whether a local second fundamental form is estimable in the PU regime, then pre-register and run a *locally*-scoped gate on the Chart Auto-Encoder; resolves Phase 3's blocking dependency, which currently names a global-scoped PASS that no method has produced
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

**Plans**: 3/13 plans executed

Plans:
**Wave 1**

- [x] 02.5-01-PLAN.md — Tracer: Swiss roll → centroid/Laplace–Beltrami estimator → Spearman vs analytic H, end to end (D-00, D-01, D-03, D-05, D-07)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02.5-02-PLAN.md — Graph-of-function fixture family at arbitrary (d, D, codimension), non-uniform sampling, and the density correction (D-03, D-06, D-07)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02.5-03-PLAN.md — Non-gating quadric cross-check, estimator agreement, and permutation-null calibration (D-01, D-02, D-05)

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 02.5-04-PLAN.md — Direction-aware verdict functions and the R6 verdict/handoff writers at 02.5 scope (D-01, D-12, D-15)
- [ ] 02.5-05-PLAN.md — **[checkpoint]** Mandatory CLAUDE.md Swiss roll sanity notebook for the curvature estimator (D-03, D-05, D-07)

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 02.5-06-PLAN.md — **[checkpoint]** Stage-1 pre-registration: ratified, committed alone, git-ancestry proved (D-00..D-08, D-12)

**Wave 6** *(blocked on Wave 5 completion)*

- [ ] 02.5-07-PLAN.md — **[checkpoint]** Stage-1 feasibility sweep, the boundary report, and the **GO/NO-GO gate that decides whether stage 2 runs at all** (D-01..D-08)

**Wave 7** *(blocked on Wave 6 completion)*

- [ ] 02.5-08-PLAN.md — Exact chart-decoder curvature via `torch.func`, C2-smoothness guard, sealed-fit load check (D-09, D-10)

**Wave 8** *(blocked on Wave 7 completion)*

- [ ] 02.5-09-PLAN.md — **[checkpoint]** Mandatory Swiss roll sanity notebook for the chart-curvature model (D-03, D-09)

**Wave 9** *(blocked on Wave 8 completion)*

- [ ] 02.5-10-PLAN.md — **[checkpoint]** Stage-2 pre-registration: the D-09/D-10 reconciliation and D-12's neither-clears branch resolved (D-09..D-12, D-15)

**Wave 10** *(blocked on Wave 9 completion)*

- [ ] 02.5-11-PLAN.md — Gate A: CAE charts vs raw points, both scored against analytic H at the PU-matched regime (D-09, D-10)

**Wave 11** *(blocked on Wave 10 completion)*

- [ ] 02.5-12-PLAN.md — **[checkpoint]** Gate B seed stability on the three sealed fits, sealed stage-2 verdict, handoff or stale deletion, routing (D-10..D-12, D-15)

**Wave 12** *(blocked on Wave 11 completion)*

- [ ] 02.5-13-PLAN.md — **[checkpoint]** Retarget Phase 3's dead-pointer dependency, revisit 02.1's falsifier, complete the phase record (D-04, D-13, D-14, D-15)

### Phase 3: Decoder & Curvature Field

**Goal**: A C2-smooth decoder trained from Phase 02.1's chosen representation back to the 768-d embedding, its analytically-derived mean curvature field validated against a synthetic-control falsification test before being trusted as a property of the data manifold rather than a decoder artifact.

> **AMENDED after Phase 2's FAIL.** This phase originally decoded from the frozen *Isomap* coordinates and depended on a PASS or MARGINAL verdict. Phase 2 returned FAIL (`m = 0.412071`), so those coordinates are the output of an invalidated step: the pullback metric would conflate real curvature with parameterization damage, undetectable by CURV-06/07's synthetic control (a synthetic manifold passing the gate never reproduces the pathology). Working dimension also open — `D_FROZEN = 5` flagged suspect in `02-FINDINGS.md` §6.4 against three estimates clustering at 18-25. Phase 02.1 supplies both the representation and the dimension. Re-plan against its output; DEC/CURV requirement text still refers to Isomap coordinates and needs the same amendment. **Further amended after Phase 02.2's insertion.** Phase 02.2 tested whether a Chart Auto-Encoder representation is a valid input for this decoding step; Phase 3 does not start until a CAE verdict reads PASS. **Further amended after Phase 02.2's FAIL (2026-08-04).** `CAE_VERDICT = FAIL` (see Phase 02.2's Outcome above); the user chose to iterate rather than adopt Krein or stop. **Further amended after Phase 02.1 sealed (2026-08-05).** Phase 02.1's falsifier fired and its GEOM-04 answer is **Ollivier-Ricci graph-native**, which drops or rewrites all thirteen DEC/CURV requirements: 9 dropped (DEC-01..05, CURV-01/02/04/05), 3 rewritten (CURV-03's extrinsic mean-curvature vector becomes an intrinsic Ricci scalar — a different mathematical object, not a renaming; CURV-06/07 re-architect onto a tree-and-expander fixture), 1 amended (CURV-08). **The success criteria and requirement list below are stale against that outcome and must be re-planned, not merely re-pointed.** At the same gate the user chose Phase 02.4 (Topological Auto-Encoder) as the next attempt, so this phase now waits on 02.4's verdict rather than 02.3's — and 02.4, being coordinate-producing, would if it PASSes reinstate much of the DEC/CURV set that 02.1's graph-native answer drops. Which representation Phase 3 decodes from is therefore still open, and depends on 02.4's outcome.

**Depends on**: Phase 02.4 **PASS** (the operative precondition — Phase 3 must check its verdict artifact before running any expensive cell; until 02.4 exists and passes, Phase 3 stays blocked). Phase 02.1 supplies the sealed representation analysis and the graph-native recommendation that applies if 02.4 fails. Phase 02.2's verdict is sealed at FAIL and is not a precondition; Phase 02.3 is superseded as the next step and is not a precondition. Phase 2 supplies the eigenspectrum evidence and FAIL verdict that motivated the original change; a Phase 2 PASS was never a precondition.
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

> **RE-SCOPED to local curvature (2026-08-07, additive — nothing above is deleted or retracted).**
> Everything above this note describes Phase 3 as it was specified before this re-scope: a *global*
> C2-smooth decoder over a single parameterization, with mean curvature computed via first/second
> fundamental forms over that global chart. The re-scope changes the target of what Phase 3
> estimates, not whether it can proceed — Phase 3's blocking dependency on a PASS verdict (line
> above) is untouched by this note and is not resolved here.
>
> **Why.** Mean curvature is a *local* invariant — the second fundamental form `II_p` at a point `p`
> depends only on an arbitrarily small neighbourhood of `p`. A manifold need not admit any single
> global chart to have a well-defined curvature field everywhere (the sphere is the standard
> counterexample: no single chart covers it, yet its curvature is defined and known at every point).
> So the milestone's repeated failure to obtain *global* coordinates — Phase 2's `GATE_VERDICT =
> FAIL`, Phase 02.2's `CAE_VERDICT = FAIL`, Phase 02.4's `TOPOAE_VERDICT = FAIL` — does not, by
> itself, block a curvature field built from *local* charts. See
> `.planning/phases/02.4-topological-auto-encoder-validity-test-inserted/02.4-FINDINGS.md` for the
> full cross-phase argument: every one of those three FAILs is a failure of a *global* statistic
> (flat-target Euclidean embeddability, whole-embedding distance/reconstruction fidelity,
> whole-held-out-set topological/reconstruction fidelity), and every *local*-scoped gate measured in
> this milestone (02.2's chart-transition cycle residual, 02.4's rank-structure statistic) passed.
>
> **Re-scoped goal.** Phase 3 now targets estimating the second fundamental form **locally, per
> point**, and assembling the curvature field piecewise across the point cloud — not a single global
> parameterization's Jacobian/Hessian. Phase 4 is **preserved unchanged**: it partitions on curvature
> quantiles and compares regional crossmodal MKNN, a question that survives intact under a
> piecewise-local curvature field exactly as it did under a global one.
>
> **Open questions this re-scope inherits, recorded without resolving them:**
>
> - **Sample density is the binding constraint, not geometry.** A local quadratic fit needs
>   `d(d+1)/2` coefficients per normal direction — 15 at `d=5`, 171 at `d=18`, 210 at `d=20`, 325 at
>   `d=25` — against the `k*=15` neighbourhood size this milestone's gates have used throughout and
>   `n=10,000` points in ambient dimension 768. At the intrinsic dimensions the measured evidence
>   actually clusters around (18–25, per `02.4-FINDINGS.md` §2.2), this is badly underdetermined at
>   `k*=15`. Raising `k` buys more equations but costs locality — the tradeoff is unresolved.
> - **Which vehicle.** Two candidate routes are open and neither is licensed by anything measured so
>   far: (a) local PCA plus quadric/jet fitting, which needs no learned model; (b) using a learned
>   decoder as the local parameterization. On (b): 02.4's TopoAE decoder is C-infinity under SiLU and
>   passed its own *local* gate (T3, rank structure) while failing the *global* ones (T1, T2) — so the
>   gate that failed tested a property local curvature estimation does not require. This is noted as
>   a reason (b) is not obviously disqualified, **not** as evidence it is licensed — any such use
>   would need its own locally-scoped pre-registration, not an inheritance from 02.4's gate as
>   measured.
> - **Feasibility should be settled on a manifold with a known answer first**, per `CLAUDE.md`'s
>   standing Swiss roll rule: a Swiss roll with analytic mean curvature, degrading ambient dimension,
>   intrinsic dimension, and sample density toward the PU regime, to find empirically where the local
>   curvature estimator breaks down — before it is trusted on data with no known answer.
>
> None of these questions is resolved by this note; they are the questions Phase 3's next
> `/gsd-discuss-phase` or `/gsd-spec-phase` session must take up before planning.

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

**Execution Order:** Phases execute in numeric order: 1 -> 2 -> 02.1 -> 02.2 -> 02.3 -> 3 -> 4, gated as described above. Phase 02.1 was inserted after Phase 2 returned FAIL; Phase 3 now depends on 02.1's output rather than on a Phase 2 PASS. Phase 02.2 was inserted after 02.1 to empirically test the Chart Auto-Encoder representation before Phase 3 commits to decoding from it. Phase 02.2 completed all six plans and returned `CAE_VERDICT = FAIL`; at that phase gate the user chose to iterate rather than adopt 02.1's Krein representation or stop, so Phase 02.3 (Chart Auto-Encoder Iteration) is proposed — not yet discussed, researched, or planned — and Phase 3 now depends on *that* phase's PASS rather than on 02.2's (sealed, permanently FAIL) verdict.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Data Loading & Manifold Reconstruction | v1.1 | 4/4 | Complete | 2026-07-31 |
| 2. Eigenspectrum Audit & Validity Gate | v1.1 | 3/3 | Complete (FAIL verdict) | 2026-08-05 |
| 02.1. Geometry Representation Research (INSERTED) | v1.1 | 4/4 | Complete (graph-native) | 2026-08-05 |
| 02.2. Chart Autoencoder Validity Test (INSERTED) | v1.1 | 6/6 | Complete    | 2026-08-04 |
| 02.3. Chart Auto-Encoder Iteration (INSERTED, proposed) | v1.1 | 0/TBD | Proposed — not planned | - |
| 02.4. Topological Auto-Encoder Validity Test (INSERTED) | v1.1 | 8/8 | Complete    | 2026-08-07 |
| 3. Decoder & Curvature Field | v1.1 | 0/TBD | Not started (blocked on 02.4) | - |
| 4. Region Partitioning & Regional Alignment (MKNN) | v1.1 | 0/TBD | Not started | - |
</content>
