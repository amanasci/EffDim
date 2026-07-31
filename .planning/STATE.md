---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: PU Manifold Curvature
current_phase: 02
current_phase_name: eigenspectrum-audit-validity-gate
status: paused
stopped_at: Halted after 02-01 by user decision — GATE_VERDICT=FAIL (m=0.412071) measured; remediation decision pending before waves 2-3 run
last_updated: "2026-07-31T19:37:46.581Z"
last_activity: 2026-07-31
last_activity_desc: Phase 02 execution started
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 7
  completed_plans: 5
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-29)

**Core value:** One call over an (n_samples, n_features) array returns a comparable panel of effective dimensionality estimates.
**Current focus:** Phase 02 — eigenspectrum-audit-validity-gate

## Current Position

Phase: 02 (eigenspectrum-audit-validity-gate) — PAUSED (user-directed halt)
Plan: 1 of 3 complete; 02-02 and 02-03 not dispatched
Status: Awaiting remediation decision on the measured GATE_VERDICT=FAIL
Last activity: 2026-07-31 — 02-01 complete, phase halted before d-freeze and verdict artifact

**Halt context:** 02-01 measured R_STAT=0.052419 (passes r<0.10) and M_STAT=0.412071
(fails the m<0.15 MARGINAL bound), giving GATE_VERDICT=FAIL on the frozen k*=15 fit.
5029 of 10,000 eigenvalues are strictly negative, none individually dominant, collectively
carrying 41% of total absolute eigenvalue mass; |LAMBDA_MIN_NEG|=169.36 sits ~24 orders of
magnitude above the float64 noise floor, so the negative tail is real non-Euclidean
structure and not rounding. User elected to decide remediation before 02-02 freezes `d`
and 02-03 writes gate_verdict_{fit_key}.json. No verdict artifact exists yet — this FAIL
is recorded only in 02-01-SUMMARY.md and the cached spectrum npz.
ROADMAP Shipped, unstarted pre-v1.1 work moved to ROADMAP Backlog (unnumbered)

Progress: [███████░░░] 71% (0/4 v1.1 phases complete; none yet planned)

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

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Validation | ED estimates checked against known-dimension manifolds (noise → D, Swiss Roll → intrinsic dim) | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| CI/Packaging | Cross-platform test matrix and release pipeline | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| Scale | SCALE-01/SCALE-02 — intramodal MKNN across a model-size ladder; curvature-stratified alignment across that ladder | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |
| Library | LIB-01/LIB-02/LIB-03 — promote curvature operator and MDS validity diagnostic into `src/effdim/`; fix `pyproject.toml` Python floor | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |

## Session Continuity

Last session: 2026-07-31T19:37:46.563Z
Stopped at: Completed 02-01-PLAN.md (eigenspectrum audit, GATE_VERDICT=FAIL measured)
REQUIREMENTS.md traceability renumbered; awaiting phase planning for Phase 1
Resume file: None
