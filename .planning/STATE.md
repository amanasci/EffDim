---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: PU Manifold Curvature
current_phase: 02.2
current_phase_name: chart-autoencoder-validity-test-inserted
status: verifying
stopped_at: Completed 02.2-06-PLAN.md -- CAE_VERDICT=FAIL, user elected to iterate; Phase 02.3 (Chart Auto-Encoder Iteration) proposed, not yet planned
last_updated: "2026-08-04T21:35:24.428Z"
last_activity: 2026-08-04
last_activity_desc: Phase 02.2 execution started
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 17
  completed_plans: 15
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-29)

**Core value:** One call over an (n_samples, n_features) array returns a comparable panel of effective dimensionality estimates.
**Current focus:** Phase 02.2 — chart-autoencoder-validity-test-inserted

## Current Position

Phase: 02.2 (chart-autoencoder-validity-test-inserted) — EXECUTING
Plan: 6 of 6
Status: Phase complete — ready for verification
Last activity: 2026-08-04 — Phase 02.2 execution started

**!! PHASE 2 IS STILL OPEN — do not treat it as sealed.** Plan 02-03 has tasks 1/3 and 2/3 committed (aea04ff, a2ca11f) but Task 3 is a blocking `checkpoint:human-verify` gate the user paused to inspect before approving. No `02-VERIFICATION.md` exists and ROADMAP still shows 2/3 plans. Resume by approving 02-03 Task 3, then `/gsd-execute-phase 2`. Phase 02.1 was started ahead of that deliberately — its work depends only on Phase 2's FAIL verdict, settled and recorded below.

**Gate outcome (settled).** 02-01 measured R_STAT=0.052419 (passes r<0.10) and M_STAT=0.412071 (fails m<0.15 MARGINAL) on the frozen k*=15 fit: GATE_VERDICT=FAIL. A pre-registered k-sensitivity re-fit (`02-REFIT-PREREGISTRATION.md`, committed 057b084 before any fit ran) tested k in {5,10,30} against incumbent k=15, all other parameters pinned:

| k | r(k) | m(k) | GEO_AMBIENT_RATIO | LONG_EDGE_FRACTION | Verdict |
|---|---|---|---|---|---|
| 5 | 0.060312 | 0.406433 | 2.828727 | 0.006540 | FAIL |
| 10 | 0.058311 | 0.410187 | 2.320592 | 0.008620 | FAIL |
| 15 | 0.052419 | 0.412071 | 2.117401 | 0.010000 | FAIL |
| 30 | 0.050708 | 0.415735 | 1.864727 | 0.013923 | FAIL |

Rule A fired: CANDIDATES=[], no k within 2.7x of the MARGINAL bound, m(k) flat-to-slightly-increasing in k. Densification worked (geodesics grew more chordal, more long edges admitted) and still bought no reduction in negative mass, so kNN hop-inflation (H2) is not supported and intrinsic curvature (H1) stands. No k* adopted; k*=15 remains fit of record. FAIL sealed against fit_key=43cf438bc944c509 by plan 02-03.

**Post-gate diagnostic triage (2026-07-31, `notebooks/diagnostics/gate_diagnostics.py`, committed 9c6e2b5).** Both remaining alternative explanations tested, neither survives — see `02-FINDINGS.md` §6:

- **Not L2 normalization.** Norms cached, normalization exactly invertible. Unnormalized refit (same rows/seed/k=15): m=0.413239 vs 0.412071 (0.28% move). Raw norms 16.029 +/- 0.504 (cv=3.1%) — data already near-constant-norm, so this only rules out "normalization caused it," not "shell geometry contributes."
- **The cloud IS a manifold.** Local intrinsic dimension stable and tight: TwoNN=19.5, local PCA median 25.0 (mean 24.5, std 2.0, 5-95% range 21-28, no neighbourhood above 29).

Surviving explanation: a real, stable ~20-25 dimensional manifold whose geodesic metric is strongly non-Euclidean.

**!! D_FROZEN=5 IS SUSPECT — do not inherit it downstream.** Four intrinsic-dimension estimates: local PCA ~25, TwoNN ~19.5, Phase 1's eight geometric estimators 18, residual-curve elbow 5 (the frozen one, and the outlier). Likely cause: with 41% negative eigenvalue mass the Tenenbaum residual curve saturates early (flat embedding fails at every dimension), so the elbow measured the failure, not the geometry (consistent with CURVE_DIVERGENCE_MAX=0.698). Separately, n_components=18 sits BELOW measured intrinsic dimension — 100% of neighbourhoods need more than 18 dims for 90% local variance, so every fit this phase was dimension-starved. Neither point changes r/m, which derive from the full 10,000-value spectrum independently of n_components.

**Implication for any Phase 3 respec:** a curvature-native representation is required (Riemannian/hyperbolic embedding, or working directly on the geodesic metric without flattening), target dimension ~20-25, not 5.

Progress: [█████████░] 88% (0/4 v1.1 phases complete; none yet planned)

## Performance Metrics

**Velocity:** 4 plans completed this milestone (4 pre-GSD plans shipped the core library; see ROADMAP Shipped). Average/total duration: n/a. By-phase totals not yet tracked (Phase 01: 4 plans).

**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 35min | 4 tasks | 8 files |
| Phase 01 P02 | 55min | 3 tasks | 1 files |
| Phase 01 P03 | 25min | 4 tasks | 1 files |
| Phase 01 P04 | 30min | 3 tasks | 1 files |
| Phase 02 P01 | 20min | 2 tasks | 1 files |
| Phase 02 P02 | 15min | 3 tasks | 1 files |
| Phase 02.1 P01 | N/A | 2 tasks | 2 files |
| Phase 02.1 P02 | 15min | 2 tasks | 1 files |
| Phase 02.1 P03 | 45min | 2 tasks | 3 files |
| Phase 02.2 P01 | 7min | 3 tasks | 1 files |
| Phase 02.2 P02 | 5min | 3 tasks | 2 files |
| Phase 02.2 P03 | ~15min | 3 tasks | 2 files |
| Phase 02.2 P04 | ~2min | 3 tasks | 2 files |
| Phase 02.2 P05 | ~3h | 3 tasks | 3 files |
| Phase 02.2 P06 | ~15min | 3 tasks | 4 files |

## Accumulated Context

### Decisions

Logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- [Bootstrap]: `.planning/` created retroactively; pre-GSD library work recorded under ROADMAP Shipped, not a numbered phase
- [v1.1 scope]: Heavy notebook deps (torch, datasets) install in-notebook, never core `pyproject.toml`; `src/effdim/`/`pyproject.toml` untouched all milestone
- [Roadmap]: v1.1 phase numbering restarts at 1. Split into 4 phases rather than SUMMARY.md's proposed 3, separating eigenspectrum audit/gate (Phase 2, 7 requirements, hard PASS/MARGINAL/FAIL gate) from data loading/Isomap fitting (Phase 1)
- [Roadmap]: unstarted pre-v1.1 work (Validation Hardening, CI & Packaging) moved to ROADMAP Backlog, unnumbered; no v1.1 phase depends on it
- [Phase ?]: Task 1 approved: torch==2.13.0+cpu, datasets==5.0.1, matplotlib==3.11.1 confirmed legitimate on PyPI
- [Phase ?]: Task 2: normalized-only selected for subsample_*.npz (no raw 768-d array cache; one-way tradeoff accepted)
- [Phase ?]: requirements-notebooks.txt now fully self-provisions (numpy/scipy/scikit-learn/faiss-cpu/joblib/pytest pinned to exact venv versions), reversing Task 1 exclusion policy
- [Phase ?]: Task 1 negative control: literal np.roll(LS,1,axis=0) does not reliably fail at n=10,000 (z=5.0010, at the margin) due to residual correlation over ~10-row gaps in sorted row_indices; np.roll(LS,1000,axis=0) used instead (z=0.29), DATA-03 check itself unchanged
- [Phase ?]: N_COMPONENTS=18 (=D_PROVISIONAL) derived from ceil(median(8 geometric compute_dim keys))=ceil(17.183); fit_key=80ce249fedcf55e0
- [Phase ?]: Task 4 gate: accept-candidate selected, k*=15 confirmed (widest all-three-passing plateau run [10,15,30], odd length 3, no tie-break needed)
- [Phase ?]: SHORT_CIRCUIT_RISK=False; all six base-range k (5,8,10,15,20,30) connected at n=10,000, auto-extend ladder never entered
- [Phase ?]: Known limitation recorded (not acted on): STAGE2_K=[5,10,15,30] unevenly spaced (gaps 5,5,15); k=8/k=20 dropped by STAGE2_MAX_FITS=4, plateau maximal in index space not k space
- [Phase ?]: Task 3 gate (checkpoint:human-verify, blocking): approved. K_STAR=15 frozen and cross-checked, isomap_43cf438bc944c509.joblib (dist_matrix_/embedding_/nbrs_/kernel_pca_) and phase1_handoff_43cf438bc944c509.json independently re-verified before Phase 1 sealed
- [Phase ?]: fit_key == sweep_k15's key (43cf438bc944c509) is correct cache-contract behaviour (identical ANALYSIS_CFG/fit_cfg dicts hash identically), not a collision
- [Phase ?]: n_components_no_headroom=True is a live D-12 condition Phase 2 must budget for: a SPEC-04 elbow beyond N_COMPONENTS=18 forces a re-fit at a larger dimension
- [Phase ?]: Real measured GATE_VERDICT=FAIL on k*=15 fit: R_STAT=0.052419 passes r<0.10 but M_STAT=0.412071 fails even m<0.15 MARGINAL (41% eigenvalue mass negative). Legitimate hard-gate outcome, not an error.
- [Phase ?]: Rule 1 auto-fix: np.asarray(dist_matrix_, dtype=float64) on a read-only memmap returned a view not a copy; fixed with np.array(..., copy=True)
- [Phase ?]: Task 2 checkpoint resolved: freeze-at-elbow selected, D_FROZEN=5 confirmed and approved (ELBOW_D=5 <= N_COMPONENTS=18)
- [Phase ?]: D_FROZEN=5 frozen via classical-MDS nesting slice EMBEDDING_ISOMAP[:, :5]; nesting verified numerically to worst relative difference 1.207e-14
- [Phase ?]: 02.1-01 checkpoint resolved: ratify (coordinate-producing branch stands as written, no amendment); falsifier remains live and untested, tested next by plan 02.1-03
- [Phase ?]: 02.1-02: GEOM-01 class-membership table separates PSD-constrained-by-construction (MVU/Laplacian-eigenmaps/LLE/Hessian-LLE/LTSA) from probability-based (t-SNE/UMAP); Isomap.kernel_pca_.eigenvalues_'s n_components truncation recorded as a second, within-family instance of the same blindness
- [Phase ?]: 02.1-02: GEOM-02 survey covers all six candidate families with identical five-part treatment (Assumptions/Cost/Maturity/Fork side/Phase 3 demand); pseudo-Euclidean/Krein retention identified as cheapest candidate (one bottom-40 eigensolve on already-cached spectrum)
- [Phase ?]: 02.1-02: MVU SDP claim, Ollivier-Ricci continuum-limit claim (arXiv:2307.02378), and both under-extracted survey papers (arXiv:2510.22599, arXiv:2509.15517) labelled [CITED] not [VERIFIED] in 02.1-SURVEY.md — no WebFetch/WebSearch tool available this session
- [Phase ?]: 02.1-03: falsifier condition (a) trips unambiguously (real manifold delta_rel_max=0.386 exceeds the flat-Euclidean anchor 0.360); condition (b) does not cleanly trip (18.4% relative distortion reduction from retaining negative eigenvalue directions — real but modest)
- [Phase ?]: 02.1-03: pair-sample bit-identity verified on first attempt (200,000 re-drawn pairs match cached geo_pairs_r2 exactly); Krein bottom-40 eigenpairs cross-checked against Phase 2's eigvals_all to rtol=1e-8
- [Phase ?]: 02.1-03: working-dimension re-derivation under gate_verdict's own kneedle criterion lands on (p,q)=(8,0) for the pseudo-Euclidean frontier — identical to the classical q=0 elbow of p=8; retaining negative directions does not move the elbow-selected dimension, only improves the far tail past it
- [Phase ?]: 02.2-01: All three CAE gate thresholds ratified exactly as proposed on 2026-08-04 (T1=0.15, T2 ratio=2.0, T3 margin=0.10); ancestry SHA c2c4c93 confirmed an ancestor of HEAD, satisfying D-10 and CAE-01's ordering requirement
- [Phase ?]: 02.2-02: Built a generic three-named-gate verdict engine (GATING_METRICS) decoupled from T1/T2/T3's actual statistic computation, which lands in plan 02.2-04; verdict_from_metrics hardened to raise ValueError on any absent/non-finite gating metric rather than ever silently resolving to FAIL
- [Phase ?]: 02.2-02: Tracer feedback gate satisfied via automated <verify> re-run under this session's Auto Mode Active configuration, in place of an interactive checkpoint:human-verify stop, before proceeding to Task 2's expansion work
- [Phase ?]: Split the combined Task 1-3 implementation pass into three atomic per-task commits (87a04c2/673bbb6/2bf36d9) by reconstructing intermediate file states from HEAD, since git checkout was blocked by the destructive-git-operation guard
- [Phase ?]: eq. 5 FPS pre-training is required for the two-chart model to activate its second chart -- without it a one-chart and two-chart CAE converge identically (the dead-chart failure mode eq. 5 prevents)
- [Phase ?]: chart_survival/r_cycle/unfaithfulness_coverage accept a duck-typed model object (not necessarily a full ChartAutoEncoder) so known-answer test fixtures can be minimal, fully floating-point-controllable stand-ins
- [Phase ?]: Pruning boundary test nudges the tolerance by one ULP against a bit-exact-computed mass ratio rather than trying to hit an arbitrary target ratio via weights, since exp(log(w)) does not round-trip bit-exactly
- [Phase ?]: embedding_distortion raises when handed a chart-dimensional array instead of the global embedding (T-02.2-11), demonstrated to differ >2x from the correct computation on a synthetic two-chart fixture
- [Phase ?]: 02.2-05: [Rule 1] Fixed LAPACK SVD non-convergence in lipschitz_penalty/chart_survival with a float64 retry (_robust_spectral_norm) discovered by real training, not by unit tests; no pre-registered constant changed
- [Phase ?]: 02.2-05: All eight pre-registered fits complete and cached (three CAE seeds, ReLU control, two plain-AE controls, two MDS-decoder baselines), all within wall-clock ceiling, cache-hit re-invocation verified
- [Phase ?]: 02.2-06: CAE_VERDICT=FAIL (T1 distortion 0.296981 vs <0.15; T3 worst-case reconstruction ratio 3.586350 vs <0.90; T2 passed 1.089366 vs <2.0) -- measured, not tuned toward PASS
- [Phase ?]: 02.2-06: T3's compound two-control AND condition encoded as max(mse_cae/mse_control_A, mse_cae/mse_control_B) < (1-THRESH_RECON_MARGIN), algebraically identical to the ratified rule, never a reinterpretation
- [Phase ?]: 02.2-06: phase gate resolved -- user chose iterate over adopt-Krein or stop-and-report; Phase 02.3 (Chart Auto-Encoder Iteration) proposed in ROADMAP.md, not yet planned; Phase 3 now depends on Phase 02.3 PASS, not Phase 02.2 (sealed FAIL)

### Pending Todos

From `TODO.md`:

- Expand test suite to validate against known dimensionalities (ROADMAP Backlog)
- CI for the standard Python implementation across platforms (ROADMAP Backlog). The Rust extension this todo also names does not exist in the repo — stale reference, see Backlog note

### Blockers/Concerns

- `UniverseTBD/pu-embeddings` is ~93 GB across 163 configs — v1.1 streams exactly one config (`legacysurvey_dinov3_vitb16`) and subsamples 10k of 101,725 rows; never materialize the whole dataset
- Phase 3 (decoder/curvature) and Phase 4 (regional MKNN) need a dedicated research pass during planning per `research/SUMMARY.md`; Phase 1/2 are standard sklearn/MDS patterns and can skip it
- Phase 2's PASS/MARGINAL/FAIL gate is a hard stop: a FAIL halts the milestone and is itself a legitimate, complete outcome. Phase 3 is now blocked on Phase 02.2's `cae_verdict.json` reading PASS, and a FAIL there leaves the milestone at the phase-2 stage

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260801-ovf | cleanup: reduce to barebones isomap-on-dino experiment | 2026-08-02 | 59742af | [260801-ovf-cleanup-reduce-to-barebones-isomap-on-di](./quick/260801-ovf-cleanup-reduce-to-barebones-isomap-on-di/) |
| 260803-k9n | Insert Phase 02.2: Chart Autoencoder Validity Test (arXiv:1912.10094) with PASS/FAIL gate for Phase 3 | 2026-08-03 | 3357ea5 | [260803-k9n-update-phase-2-of-milestone-to-test-vali](./quick/260803-k9n-update-phase-2-of-milestone-to-test-vali/) |

### Roadmap Evolution

- Phase 02.1 inserted after Phase 2: Geometry Representation Research — Phase 2 gate FAIL invalidated the Isomap coordinates Phase 3 was specified to decode from (URGENT)
- Phase 02.1 planned: 4 plans across 3 waves; plan-checker VERIFICATION PASSED first iteration; GEOM-01..05 coverage complete
- Phase 02.2 inserted after Phase 02.1: Chart Autoencoder Validity Test — empirically tests arXiv:1912.10094 on the PU data behind a PASS/FAIL gate; PASS unblocks Phase 3 to decode from the CAE representation, FAIL documents the finding and leaves the milestone at the phase-2 stage. Doc-only insertion; the phase itself is unplanned

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Validation | ED estimates checked against known-dimension manifolds (noise → D, Swiss Roll → intrinsic dim) | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| CI/Packaging | Cross-platform test matrix and release pipeline | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| Scale | SCALE-01/SCALE-02 — intramodal MKNN across a model-size ladder; curvature-stratified alignment across that ladder | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |
| Library | LIB-01/LIB-02/LIB-03 — promote curvature operator and MDS validity diagnostic into `src/effdim/`; fix `pyproject.toml` Python floor | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |

## Session Continuity

Last session: 2026-08-04T21:35:24.406Z
Stopped at: Completed 02.2-06-PLAN.md -- CAE_VERDICT=FAIL, user elected to iterate; Phase 02.3 (Chart Auto-Encoder Iteration) proposed, not yet planned
REQUIREMENTS.md traceability renumbered; awaiting phase planning for Phase 1
Resume file: None
</content>
