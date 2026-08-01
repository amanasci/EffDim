---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: PU Manifold Curvature
current_phase: 02.1
current_phase_name: geometry-representation-research
status: executing
stopped_at: "Completed 02.1-01-PLAN.md (fork ratified: coordinate-producing branch)"
last_updated: "2026-08-01T19:37:38.727Z"
last_activity: 2026-08-01
last_activity_desc: Phase 02.1 execution started
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 11
  completed_plans: 7
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-29)

**Core value:** One call over an (n_samples, n_features) array returns a comparable panel of effective dimensionality estimates.
**Current focus:** Phase 02.1 — geometry-representation-research

## Current Position

Phase: 02.1 (geometry-representation-research) — EXECUTING
Plan: 2 of 4 (wave 1 of 3)
Status: Ready to execute
Last activity: 2026-08-01 — Phase 02.1 execution started

**!! PHASE 2 IS STILL OPEN — do not treat it as sealed.** Plan 02-03 has tasks 1/3 and 2/3
committed (aea04ff, a2ca11f) but Task 3 is a blocking `checkpoint:human-verify` gate that the
user paused to inspect before approving. Phase 2 is NOT sealed and NOT verified; no
`02-VERIFICATION.md` exists and the ROADMAP still shows 2/3 plans. Resume by approving 02-03
Task 3, then `/gsd-execute-phase 2`. Phase 02.1 was started ahead of that deliberately — its
work does not depend on Phase 2 being sealed, only on Phase 2's FAIL verdict, which is settled
and recorded below.

**Gate outcome (settled).** 02-01 measured R_STAT=0.052419 (passes r<0.10) and
M_STAT=0.412071 (fails the m<0.15 MARGINAL bound) on the frozen k*=15 fit:
GATE_VERDICT=FAIL. A pre-registered k-sensitivity re-fit
(`02-REFIT-PREREGISTRATION.md`, committed 057b084 before any fit ran) then tested
k in {5,10,30} against the incumbent k=15 with all other parameters pinned:

| k | r(k) | m(k) | GEO_AMBIENT_RATIO | LONG_EDGE_FRACTION | Verdict |
|---|---|---|---|---|---|
| 5 | 0.060312 | 0.406433 | 2.828727 | 0.006540 | FAIL |
| 10 | 0.058311 | 0.410187 | 2.320592 | 0.008620 | FAIL |
| 15 | 0.052419 | 0.412071 | 2.117401 | 0.010000 | FAIL |
| 30 | 0.050708 | 0.415735 | 1.864727 | 0.013923 | FAIL |

Rule A fired: CANDIDATES=[], no k comes within 2.7x of the MARGINAL bound, and m(k) is
flat-to-slightly-increasing in k rather than decreasing. The co-diagnostics show
densification measurably worked (geodesics grew more chordal, more long edges admitted)
and still bought no reduction in negative mass, so the kNN hop-inflation hypothesis (H2)
is not supported and intrinsic curvature (H1) stands. No k* adopted; k*=15 remains the
fit of record. FAIL is sealed against fit_key=43cf438bc944c509 by plan 02-03.

**Post-gate diagnostic triage (2026-07-31, `notebooks/diagnostics/gate_diagnostics.py`,
committed 9c6e2b5).** Both remaining alternative explanations were tested and neither
survives — see `02-FINDINGS.md` §6:

- **Not L2 normalization.** Norms are cached, so normalization is exactly invertible. An
  unnormalized refit (same rows, seed, k=15) gives m=0.413239 vs 0.412071 — a 0.28% move.
  Caveat: raw norms are 16.029 +/- 0.504 (cv=3.1%), so the data was already near-constant-norm
  and removing normalization barely moved the geometry. This closes "normalization caused it",
  not "shell geometry contributes".

- **The cloud IS a manifold.** Local intrinsic dimension is stable and tight: TwoNN=19.5,
  local PCA median 25.0 (mean 24.5, std 2.0, 5-95% range 21-28, no neighbourhood above 29).
  That is a genuine manifold of roughly constant dimension, not a structureless cloud.

Surviving explanation: a real, stable ~20-25 dimensional manifold whose geodesic metric is
strongly non-Euclidean.

**!! D_FROZEN=5 IS SUSPECT — do not inherit it downstream.** Four estimates of intrinsic
dimension now exist: local PCA ~25, TwoNN ~19.5, Phase 1's eight geometric estimators 18,
and the residual-curve elbow 5. The elbow is the outlier and it is the value that was frozen.
Likely cause: with 41% of eigenvalue mass negative the Tenenbaum residual curve saturates
early because flat embedding fails at every dimension, so the elbow measured the failure
rather than the geometry (consistent with CURVE_DIVERGENCE_MAX=0.698). Separately,
n_components=18 sits BELOW the measured intrinsic dimension — 100% of neighbourhoods need
more than 18 dimensions for 90% of local variance, so every fit this phase was
dimension-starved. Neither point changes r/m, which derive from the full 10,000-value
spectrum independently of n_components.

**Implication for any Phase 3 respec:** a curvature-native representation is required
(Riemannian/hyperbolic embedding, or working directly on the geodesic metric without
flattening), and the target dimension is ~20-25, not 5.
ROADMAP Shipped, unstarted pre-v1.1 work moved to ROADMAP Backlog (unnumbered)

Progress: [██████░░░░] 64% (0/4 v1.1 phases complete; none yet planned)

## Performance Metrics

**Velocity:**

- Total plans completed: 4 this milestone (4 pre-GSD plans shipped the core library; see
  ROADMAP Shipped)

- Average duration: n/a
- Total execution time: n/a

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| (none yet) | - | - | - |
| 01 | 4 | - | - |

**Recent Trend:**

- Last 5 plans: n/a
- Trend: n/a

*Updated after each plan completion*
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

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Bootstrap]: `.planning/` created retroactively; pre-GSD library work is recorded under
  ROADMAP Shipped rather than as a numbered phase

- [v1.1 scope]: Heavy notebook deps (torch, datasets) install in-notebook, never in core
  `pyproject.toml`; `src/effdim/` and `pyproject.toml` untouched all milestone

- [Roadmap]: v1.1 phase numbering restarts at 1. Split into 4 phases rather than
  SUMMARY.md's proposed 3, separating the eigenspectrum audit/gate (Phase 2, 7 requirements,
  the hard PASS/MARGINAL/FAIL gate) from data loading/Isomap fitting (Phase 1) given SPEC's
  size and gate role

- [Roadmap]: unstarted pre-v1.1 work (Validation Hardening, CI & Packaging) moved to ROADMAP
  Backlog, unnumbered, so it neither collides with v1.1's phase sequence nor gets dropped. No
  v1.1 phase depends on it. Pre-v1.1 applied-analysis intent is fulfilled by v1.1 itself

- [Phase ?]: Task 1 approved: torch==2.13.0+cpu, datasets==5.0.1, matplotlib==3.11.1 confirmed legitimate on PyPI
- [Phase ?]: Task 2: normalized-only selected for subsample_*.npz (no raw 768-d array cache; one-way tradeoff accepted)
- [Phase ?]: User-directed deviation: requirements-notebooks.txt now fully self-provisions (numpy/scipy/scikit-learn/faiss-cpu/joblib/pytest pinned to exact venv versions), reversing the Task 1 exclusion policy
- [Phase ?]: Task 1 negative control: literal np.roll(LS,1,axis=0) does not reliably fail at n=10,000 (z=5.0010, at the margin) due to residual correlation over ~10-row gaps in sorted row_indices; np.roll(LS,1000,axis=0) used as the asserted control instead (z=0.29), DATA-03 check itself unchanged
- [Phase ?]: N_COMPONENTS=18 (=D_PROVISIONAL) derived from ceil(median(8 geometric compute_dim keys))=ceil(17.183); ANALYSIS_CFG[n_components] set; fit_key=80ce249fedcf55e0
- [Phase ?]: Task 4 gate: accept-candidate selected, k*=15 confirmed (widest all-three-passing plateau run [10,15,30], odd length 3, no tie-break needed)
- [Phase ?]: SHORT_CIRCUIT_RISK=False; all six base-range k (5,8,10,15,20,30) are connected at n=10,000, auto-extend ladder never entered
- [Phase ?]: Known limitation recorded (not acted on): STAGE2_K=[5,10,15,30] is unevenly spaced (gaps 5,5,15); k=8/k=20 dropped by STAGE2_MAX_FITS=4, so the plateau is maximal in index space not k space
- [Phase ?]: Task 3 gate (checkpoint:human-verify, gate=blocking): approved. K_STAR=15 frozen and cross-checked, isomap_43cf438bc944c509.joblib (dist_matrix_/embedding_/nbrs_/kernel_pca_) and phase1_handoff_43cf438bc944c509.json independently re-verified by coordinator before Phase 1 was sealed
- [Phase ?]: fit_key == sweep_k15's key (43cf438bc944c509) is correct cache-contract behaviour (identical ANALYSIS_CFG/fit_cfg dicts hash identically), not a collision -- the joblib and npz artifacts remain distinct files under distinct stems
- [Phase ?]: n_components_no_headroom=True is a live D-12 condition Phase 2 must budget for: a SPEC-04 elbow beyond N_COMPONENTS=18 forces a re-fit at a larger dimension (cheap to invalidate correctly via a new fit_key, but real wall-clock work)
- [Phase ?]: Real measured GATE_VERDICT=FAIL on k*=15 fit: R_STAT=0.052419 passes r<0.10 but M_STAT=0.412071 fails even the m<0.15 MARGINAL bound (41% of eigenvalue mass is negative). Legitimate hard-gate outcome per phase design, not an error.
- [Phase ?]: Rule 1 auto-fix: np.asarray(dist_matrix_, dtype=float64) on a read-only memmap returned a view not a copy (dtype already matched); fixed with np.array(..., copy=True).
- [Phase ?]: Task 2 checkpoint resolved: freeze-at-elbow selected, D_FROZEN=5 confirmed and approved by human (ELBOW_D=5 <= N_COMPONENTS=18)
- [Phase ?]: D_FROZEN=5 frozen via classical-MDS nesting slice EMBEDDING_ISOMAP[:, :5]; nesting claim verified numerically to worst relative difference 1.207e-14, not merely argued in prose
- [Phase ?]: 02.1-01 checkpoint resolved: ratify (coordinate-producing branch stands as written, no amendment); falsifier remains live and untested, tested next by plan 02.1-03

### Pending Todos

From `TODO.md`:

- Expand test suite to validate against known dimensionalities (ROADMAP Backlog)
- CI for the standard Python implementation across platforms (ROADMAP Backlog). The Rust
  extension this todo also names does not exist in the repo — stale reference, see Backlog note

### Blockers/Concerns

- `UniverseTBD/pu-embeddings` is ~93 GB across 163 configs — v1.1 streams exactly one config
  (`legacysurvey_dinov3_vitb16`) and subsamples 10k of 101,725 rows; never materialize the
  whole dataset.

- Phase 3 (decoder/curvature) and Phase 4 (regional MKNN) are flagged in
  `research/SUMMARY.md` as needing a dedicated research pass during planning; Phase 1/2
  (Isomap + gate) are standard sklearn/MDS patterns and can skip that pass.

- Phase 2's PASS/MARGINAL/FAIL gate is a hard stop: a FAIL halts the milestone and is itself
  a legitimate, complete outcome. Phase 3 must not be planned in detail until Phase 2's gate
  outcome is known.

### Roadmap Evolution

- Phase 02.1 inserted after Phase 2: Geometry Representation Research - Phase 2 gate FAIL invalidated the Isomap coordinates Phase 3 was specified to decode from (URGENT)
- Phase 02.1 planned: 4 plans across 3 waves; plan-checker VERIFICATION PASSED first iteration; GEOM-01..05 coverage complete

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Validation | ED estimates checked against known-dimension manifolds (noise → D, Swiss Roll → intrinsic dim) | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| CI/Packaging | Cross-platform test matrix and release pipeline | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| Scale | SCALE-01/SCALE-02 — intramodal MKNN across a model-size ladder; curvature-stratified alignment across that ladder | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |
| Library | LIB-01/LIB-02/LIB-03 — promote curvature operator and MDS validity diagnostic into `src/effdim/`; fix `pyproject.toml` Python floor | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |

## Session Continuity

Last session: 2026-08-01T19:37:38.707Z
Stopped at: Completed 02.1-01-PLAN.md (fork ratified: coordinate-producing branch)
REQUIREMENTS.md traceability renumbered; awaiting phase planning for Phase 1
Resume file: None
