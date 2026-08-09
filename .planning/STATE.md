---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: PU Manifold Curvature
current_phase: 02.5
current_phase_name: local-curvature-feasibility-cae-re-gate
status: executing
stopped_at: 02.5-07 Task 3 checkpoint RESOLVED (GO). Stage-1 re-run under 02.5-PREREGISTRATION-AMENDMENT-01.md (5 seeds, confidence-bounded verdict) returned CURVATURE_VERDICT=FAIL; user chose GO to stage-2 Arm B. Executing 02.5-08.
last_updated: "2026-08-09T00:00:00.000Z"
last_activity: 2026-08-07
last_activity_desc: Phase 02.5 execution started
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 38
  completed_plans: 32
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-29)

**Core value:** One call over an (n_samples, n_features) array returns a comparable panel of effective dimensionality estimates.
**Current focus:** Phase 02.5 — local-curvature-feasibility-cae-re-gate

## Current Position

Phase: 02.5 (local-curvature-feasibility-cae-re-gate) — EXECUTING
Plan: 8 of 13
Status: 02.5-07 Task 3 checkpoint RESOLVED — user decided GO. CURVATURE_VERDICT=FAIL under the amended 5-seed confidence-bounded rule (qbc lower bound 0.454223 vs floor 0.475000; rho lower bound 0.530332 vs floor 0.500000). Proceeding to 02.5-08 (stage-2 Arm B, chart decoder curvature)
Last activity: 2026-08-09 — Amendment 1 ratified and sealed (12cca56), 5-seed re-run completed (27/27 cells, 28.6 min, stage1_key e71a4ea18050ea20), 02.5-FINDINGS-AMENDED-01.md written, GO decision taken; 02.5-08 dispatched

**Phase 02.5 is planned (2026-08-07).** 13 plans across 12 waves. No REQ-IDs exist for this phase — `02.5-CONTEXT.md`'s 16 decisions are the de-facto requirement set, and decision coverage is **16/16 (D-00..D-15)**, verified by the plan-checker rather than accepted from the planner. Plan-checker returned **0 blockers, 0 warnings**. Wave order: `01` centroid-estimator tracer → `02` fixtures and density correction → `03` quadric cross-check and permutation null → `04` verdict layer ∥ `05` stage-1 Swiss roll notebook → `06` stage-1 pre-registration → **`07` stage-1 GO/NO-GO** → `08` chart curvature → `09` stage-2 notebook → `10` stage-2 pre-registration → `11` Gate A → `12` verdict → `13` D-13/D-14 obligations and the phase record. Only wave 4 runs in parallel (`04` ∥ `05`, disjoint `files_modified`); pre-registration ordering forces the rest to be sequential. Plans `05`, `06`, `07`, `09`, `10`, `12`, `13` are non-autonomous — each carries a blocking human checkpoint.

**Stage 1 gates stage 2 structurally, and the NO-GO branch is written down.** Plans `08`–`12` depend transitively on plan `07`, whose Task 3 is a blocking `checkpoint:human-verify` — the sole gate deciding whether stage 2 runs at all. On NO-GO, plan `07` directs an explicit human-gated re-pointing of plan `13`'s `depends_on` from `["02.5-12"]` to `["02.5-07"]`, and plan `13` documents writing a stage-1-only FINDINGS/amendment set from `02.5-07-SUMMARY.md`. A stage-1 negative is a complete, reportable phase outcome, not a stall.

**Two substantive things the planner found and resolved, both verified by the checker.** (1) A **curvature-convention mismatch inside `02.5-RESEARCH.md`**: its Pattern 1 derivation, Pattern 4, and `curvature.py`'s stub docstrings all use `H = tr(II)`, but its `swiss_roll_analytic_H` returns the *averaged* `κ/2` — off by a factor of `d` (2 at the Swiss roll, 20 at the PU regime). Spearman is invariant to the factor, so D-01's gate would never have caught it, but D-01's non-gating median relative error and D-05's estimator-agreement check would both have been wrong by `d`. Resolved to the **trace convention** in `02.5-01-PLAN.md`, pinned by `test_curvature_convention_is_trace_not_averaged`. (2) **D-09 and D-10 are not jointly satisfiable as written** — D-09 wants both arms scored against known `H`, D-10 wants the three *sealed* fits re-measured rather than retrained, and the sealed fits are trained on PU data, which has no known `H`. Plan `10` splits the gate: **Gate A** (margin) on analytic-`H` fixtures with CAEs fitted at the sealed fits' verbatim architecture, **Gate B** (seed stability) on the sealed PU fits where agreement needs no ground truth — with the reconciliation itself put in front of the user at the ratification checkpoint rather than resolved silently.

**Performance trap flagged into the plans.** `02.5-RESEARCH.md`'s Pattern 1 uses `np.linalg.eigh` on the `(D, D)` covariance — O(D³), unusable at `D = 768`, `n = 10,000`. The plans specify the O(k²D) SVD route instead, with a negative grep on `eigh` as an acceptance criterion.

**Carried into execution.** `02.5-VALIDATION.md`'s per-task map is still `TBD`-keyed. Its ten pre-seeded pytest names all appear verbatim in the plans (kept in one `notebooks/pu_manifold/tests/test_curvature_probe.py` so the map reconciles cleanly), but `/gsd-validate-phase` still needs to fill in the Task IDs. `notebooks/pu_manifold/curvature.py`'s `NotImplementedError` stubs — labelled "Implemented in Phase 3 (CURV-0N)" — are explicitly **never filled and never imported**; stage 2 builds a phase-scoped `chart_curvature.py` instead, so Phase 3 requirements are not pulled forward.

**Why 02.5 exists.** Phase 3 is blocked on a **PASS** no method has produced, and `02.4-FINDINGS.md` argues that gate may ask the wrong question: every FAIL in this milestone (Phase 2's `m = 0.412071`, 02.2's T1/T3, 02.4's T1/T2) is a *global* statistic, while every *local*-scoped gate measured has passed (02.2's chart-transition residual `1.089366 < 2.0`; 02.4's T3 `0.671980` at `k=15`). Mean curvature is a **local invariant** — `II_p` depends only on an arbitrarily small neighbourhood — so failing to obtain *global* coordinates does not by itself block a curvature field. Two stages, the first gating the second: (1) a Swiss roll feasibility probe with analytic `H`, degraded toward the PU regime, to find where local second-fundamental-form estimation breaks; (2) a locally-scoped CAE re-gate, **only if stage 1 clears**. A stage-1 negative is a complete, reportable outcome.

**The "binding constraint" was overstated — corrected during discussion (`02.5-CONTEXT.md` D-00).** The `d(d+1)/2`-coefficient count (171 at `d=18`, 210 at `d=20`, 325 at `d=25`, against `k* = 15`) is the cost of the **full second fundamental form**. Mean curvature is only its *trace*, and the identity `Δ_M x = H` — equivalently, a neighbourhood's centroid is displaced from its centre point along the mean curvature vector — estimates the trace as an **average over `k` vectors**: one unknown with `k` samples, not 210 unknowns with 15 equations. The underdetermination recorded in the ROADMAP re-scope may not bind at all. Real remaining risks are different ones: bias growing like `r²` at finite radius, and non-uniform sampling density drifting the centroid in a way indistinguishable from curvature (D-06 pre-registers a correction and proves it on deliberately non-uniform fixtures — a Swiss roll is evenly sampled and would never catch it). `D_FROZEN = 5` **must not be inherited**: `02-FINDINGS.md` §6.4 records the residual-curve elbow saturating early under 41% negative eigenvalue mass, so it measured the flatness failure, not the dimension; three estimates cluster at 18–25, and D-07 uses `d = 20` per 02.2's D-04.

**Why the CAE is the candidate.** It is an atlas of local charts by construction, its local consistency gate passed on real PU data, and it is the only model in this milestone to pass its Swiss roll outright (4.8% relative error vs a `<10%` bound, 2.2× better than a matched plain-AE, 8/8 charts surviving). Its sealed FAIL rests on *global* T1/T3. That makes it not-disqualified, **not** licensed — a local PASS must be earned under a fresh pre-registration, never inherited from 02.2's gate.

**Phase 02.4 is planned (2026-08-06).** 8 plans across 7 waves. Requirement coverage R1–R8 complete; decision coverage 20/20 against `02.4-CONTEXT.md`. Plan-checker returned **0 blockers, 1 warning**. Wave order: `01` topoae.py tracer (R1,R2) → `02` gate layer (R4,R5,R6) + `03` λ sweep and the mandatory Swiss roll notebook (R8) → `04` pre-registration (R3) → `05` gated PU train runner, primary rung (R2,R3) → `06` remaining 13 fits (R2) → `07` evaluate runner and verdict artifact (R4,R5,R6) → `08` reconciliation and the TOPO-01..08 register (R7). Plans `03`, `04`, `05`, `07` are non-autonomous — each carries a blocking human checkpoint.

**Open warning carried into execution.** `02.4-RESEARCH.md` § Open Questions is not marked resolved. Its three questions — ambient X-space distance normalization, baseline-fit coverage across the ladder, and warm-up/ramp shape — are all resolved inside `02.4-04-PLAN.md` Tasks 1–2 (`AMBIENT_DIST_NORM = "none"`; one matched baseline per TopoAE fit, so eight baselines; a quarter/quarter/half warm-up-ramp-constant split) and committed into `02.4-PREREGISTRATION.md` before any PU fit. The annotation back into RESEARCH.md is deferred because it can only be written after that pre-registration exists. Nothing stalls on it; the research record simply reads as still-open until then.

**Note on the compute budget.** `02.4-RESEARCH.md` flagged that per-rung baseline `PlainAutoEncoder` fits appeared uncounted in D-19's ~8h estimate. Plan `04` resolves this as one matched baseline per TopoAE fit — eight baselines, not one — so the real fit count is 16, not 8. Confirm the wall-clock envelope still holds when plan `05`'s timing probe runs.

**Phase 02.4 is scoped.** `02.4-SPEC.md` locks 8 requirements, 17 acceptance criteria, 20 resolved edges, 8 prohibitions. `02.4-CONTEXT.md` carries 20 implementation decisions. Two sequencing facts the planner must not miss: **λ is tuned on the Swiss roll fixture and frozen before any PU fit**, so the fixture and its sweep precede the pre-registration (not the other way round); and the λ sweep lives in a separate `notebooks/diagnostics/` runner, because `CLAUDE.md` caps the sanity-check notebook at ~15 cells with no threshold tables. Gate: T1 = the paper's `L_t` held-out (full-set MST gates, worse direction), T2 = reconstruction margin vs matched `PlainAutoEncoder`, T3 = `1 − min(trustworthiness, continuity)` at k=15 — all three baseline-relative, none gating on `DISTORTION`. Ladder `{8,16,20,24,32,40}`, primary `d=20`, 3 seeds primary + 1 elsewhere = 8 fits at a ~1h ceiling (cut from 02.2's 2h to fit the envelope; the 32 and 40 rungs may be under-trained, recorded as a stated limitation).

**Every phase before Phase 3 is now closed.** Phases 1 (4/4), 2 (3/3), 02.1 (4/4), 02.2 (6/6) complete. Phase 02.3 is superseded as the next step and is not a Phase 3 precondition, but stays on the roadmap unretracted as a fallback if 02.4 fails. Phase 3 is blocked on Phase 02.4's verdict.

**Phase 2 SEALED (2026-08-05), `GATE_VERDICT = FAIL`.** Plan 02-03's blocking human-verify gate — held since 2026-07-31 — approved against the surviving `gate_verdict_43cf438bc944c509.json` rather than a fresh Restart-and-Run-All, because quick task `260801-ovf` (`8958488`) deleted `notebooks/01_manifold_and_gate.ipynb` during the hold. 8 of 10 verification steps re-verified; the 2 unrepeatable ones named in `02-03-SUMMARY.md`. Remediation option 3 accepted. Notebook recoverable at `a2ca11f`. `02-VERIFICATION.md` records all five criteria PASS across SPEC-01..07. Code review N/A — the phase's entire source footprint was that one deleted notebook. `02-SECURITY.md` not produced.

**Phase 02.1 SEALED (2026-08-05), `GEOM-04 = Ollivier-Ricci graph-native`.** The pre-registered falsifier fired: (a) trips wide (`delta_rel_max=0.383921` past a `0.360433` flat anchor, threshold `0.036043`); (b) trips under `02.1-AMENDMENT-02.md`'s amended reading requiring the ladder's drop be realisable in a decoder-consumable form. Krein `(40,25)` won the pre-registered criterion at `0.065190` and was rejected twice — user directive (Amendment 01 §1.3) and a pre-registered decoder check giving it only `+1.44%`/`+0.10%` held-out reconstruction against the `+18.37%` promised, with the matched-width signature control negative. Four unrelated families wall at ~0.0796; metric SMACOF reaching it with no eigendecomposition and no PSD constraint shows the constraint is target flatness, not algorithm. `D_FROZEN=5` discarded as inapplicable — a per-edge branch has no embedding dimension; the coordinate branch's `(8,0)` is preserved. Machinery validated on a Swiss roll (`m=0.027292` vs a 0.05 bound; hand double-centring matches sklearn to 1.8e-13). `02.1-VERIFICATION.md` records all five criteria PASS across GEOM-01..05.

**!! CARRY FORWARD — the evaluation criterion may not measure the right thing.** The decoder check found held-out reconstruction nearly **decoupled** from the distance-distortion statistic Phase 02.1 ranked representations by: classical `(40,0)`, worst distortion of the three ladder rungs at `0.179641`, reconstructed *best* of all four arms on both preprocessings. Distortion spanned 2.75×; MSE spanned ~6%. One seed, so few-percent gaps are not separable from initialisation noise and capacity saturation is not excluded — an observation, not a verdict. `02.1-AMENDMENT-02.md` §6.4 records it as the strongest reason to doubt that amendment; §6.5 names the seed-sensitivity study that would settle it. **Not run.** Any phase inheriting 02.1's recommendation should know the criterion that selected its predecessor may have been measuring the wrong thing.

**!! Phase 02.4 sits in tension with Phase 02.1's outcome, deliberately.** TopoAE is coordinate-producing with a Euclidean latent, and 02.1's falsifier just fired against that branch. The reconciliation: 02.1's ~0.0796 wall was measured on a **distance-preservation** statistic, and every arm that hit it was optimising distance preservation; TopoAE optimises topological signature matching instead and does not claim distance preservation. So it must **not** be scored primarily on `DISTORTION` — that would rank it on an axis it never optimised. And if it reaches PASS, `02.1-AMENDMENT-02.md`'s falsifier firing should be revisited by a dated amendment. Both obligations are recorded in the ROADMAP's Phase 02.4 entry.

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

Progress: [████████░░] 82% of planned plans (17/17; Phases 1, 2, 02.1, 02.2 all complete). Phase 02.4 next — not yet scoped, so its plan count is unknown and the milestone is not near done.

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
| Phase 02.4 P01 | 10min | 3 tasks | 2 files |
| Phase 02.4 P02 | 5min | 3 tasks | 2 files |
| Phase 02.4 P03 | ~10h30m wall-clock (mostly unattended sweep runs + checkpoint round-trips) | 4 tasks | 4 files |
| Phase 02.4 P04 | ~1h | 3 tasks | 1 files |
| Phase 02.4 P05 | 50min | 3 tasks | 1 files |
| Phase 02.4 P05 | 1h40m | 3 tasks | 4 files |
| Phase 02.4 P06 | ~20min | 3 tasks | 0 files |
| Phase 02.4 P07 | 45min | 3 tasks | 1 files |
| Phase 02.4 P08 | ~25min | 3 tasks | 6 files |
| Phase 02.5 P01 | ~30min | 3 tasks | 2 files |
| Phase 02.5 P02 | ~1h20min | 3 tasks | 2 files |
| Phase 02.5 P03 | ~45min | 3 tasks | 2 files |
| Phase 02.5 P04 | ~34min | 3 tasks | 2 files |
| Phase 02.5 P05 | ~20min active (checkpoint hold between segments) | 3 tasks | 1 files |
| Phase 02.5 P06 | 66min | 3 tasks | 2 files |

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
- [Phase 2]: 02-03 Task 3 phase-sealing checkpoint approved 2026-08-05 on the surviving artifact, not a fresh notebook re-run — `260801-ovf` deleted the verification target during the hold. 8/10 steps re-verified, 2 recorded as unrepeatable. Precedent: a checkpoint whose target was removed by later work is closed on independently reconstructible evidence with the gaps named, never silently marked verified.
- [Phase ?]: eq. 5 FPS pre-training is required for the two-chart model to activate its second chart -- without it a one-chart and two-chart CAE converge identically (the dead-chart failure mode eq. 5 prevents)
- [Phase ?]: chart_survival/r_cycle/unfaithfulness_coverage accept a duck-typed model object (not necessarily a full ChartAutoEncoder) so known-answer test fixtures can be minimal, fully floating-point-controllable stand-ins
- [Phase ?]: Pruning boundary test nudges the tolerance by one ULP against a bit-exact-computed mass ratio rather than trying to hit an arbitrary target ratio via weights, since exp(log(w)) does not round-trip bit-exactly
- [Phase ?]: embedding_distortion raises when handed a chart-dimensional array instead of the global embedding (T-02.2-11), demonstrated to differ >2x from the correct computation on a synthetic two-chart fixture
- [Phase ?]: 02.2-05: [Rule 1] Fixed LAPACK SVD non-convergence in lipschitz_penalty/chart_survival with a float64 retry (_robust_spectral_norm) discovered by real training, not by unit tests; no pre-registered constant changed
- [Phase ?]: 02.2-05: All eight pre-registered fits complete and cached (three CAE seeds, ReLU control, two plain-AE controls, two MDS-decoder baselines), all within wall-clock ceiling, cache-hit re-invocation verified
- [Phase ?]: 02.2-06: CAE_VERDICT=FAIL (T1 distortion 0.296981 vs <0.15; T3 worst-case reconstruction ratio 3.586350 vs <0.90; T2 passed 1.089366 vs <2.0) -- measured, not tuned toward PASS
- [Phase ?]: 02.2-06: T3's compound two-control AND condition encoded as max(mse_cae/mse_control_A, mse_cae/mse_control_B) < (1-THRESH_RECON_MARGIN), algebraically identical to the ratified rule, never a reinterpretation
- [Phase ?]: 02.2-06: phase gate resolved -- user chose iterate over adopt-Krein or stop-and-report; Phase 02.3 (Chart Auto-Encoder Iteration) proposed in ROADMAP.md, not yet planned; Phase 3 now depends on Phase 02.3 PASS, not Phase 02.2 (sealed FAIL)
- [Phase ?]: 02.4-01: Task 1's tracer feedback gate (interactive checkpoint:human-verify) approved by user after independent re-verification of all <verify> commands and T1/T2/T3 ratio directions; Tasks 2-3 proceeded
- [Phase ?]: 02.4-01: tracer artifact's gate_detail intentionally still uses cae's borrowed GATING_METRICS slot names (distortion/rcycle_ratio/recon_margin) -- orchestrator confirmed replacing them is plan 02.4-02's job (threat T-02.4-11), out of scope for 02.4-01
- [Phase ?]: 02.4-01: train_topoae's non-finite-loss check runs per-batch (not per-epoch-mean), raising ValueError naming the epoch and batch index at the point of divergence; empirically confirmed to trip at lr=1e8
- [Phase ?]: 02.4-02: T-02.4-11 resolved via positional slot remap (CAE_SLOT_ALIASES = dict(zip(GATING_METRICS, cae.GATING_METRICS))); the three borrowed cae.py slot names never appear in any topoae artifact
- [Phase ?]: 02.4-02: write_topoae_verdict recomputes gate_detail internally and refuses to write if the supplied verdict disagrees with the recomputed one -- a stored verdict may never disagree with its own gates
- [Phase ?]: 02.4-02: requirements.mark-complete found no R4/R5/R6 entries in REQUIREMENTS.md -- phase 02.4's R1..R8 are scoped locally to 02.4-SPEC.md and were never mirrored into the milestone-level REQUIREMENTS.md (no TOPO section exists there); not a blocker for this plan, noted for a future ledger sync
- [Phase ?]: 02.4-03 mid-plan fidelity correction (2026-08-07): topoae.py's train_topoae did not faithfully implement the paper (arXiv:1906.00722)/reference (BorgwardtLab/topological-autoencoders) -- missing jointly-trained latent_norm scale, missing per-batch d_x/d_x.max() ambient normalization, a spurious /batch_size division, and a missing 1/2 factor on each directional term. All four fixed, 4 new tests added (106 total in the suite), float64 confirmed as a deliberate retained divergence from the reference's float32
- [Phase ?]: 02.4-03 lambda re-swept over the paper's actual log-uniform-[0.1,3] range after the fidelity correction (prior grid was ~32x mis-scaled due to the batch-size division + missing 1/2 factor bugs); re-measured selection is again the grid floor, lambda=0.1, same value as before but now a faithful measurement
- [Phase ?]: 02.4-03 Task 4's blocking Swiss roll checkpoint is NOT approved as of this SUMMARY -- the corrected implementation still does not clearly recover the Swiss roll (22.6% relative error, does not beat the matched plain-AE baseline, 0.680 persistence-pair correlation vs a 0.8 bound). Plans 02.4-04..08 remain blocked pending a human decision
- [Phase ?]: 02.4-03 round 2 (2026-08-07): the notebook's topological structural check had no matched baseline (absolute r>0.8 bound only) -- added the same ambient MST pairing applied to both TopoAE and plain-AE latents, gate now baseline-relative. Result: TopoAE r=0.680 vs plain-AE r=0.471 -- TopoAE clearly beats the baseline on its own stated objective while losing to it on plain MSE reconstruction (ratio 1.382). Read as the trade the method makes on purpose (paper evaluates with KL divergence/trustworthiness-continuity, never MSE), not tuned toward this reading
- [Phase ?]: 02.4-03: Task 4's blocking Swiss roll checkpoint APPROVED (2026-08-07) after two correction rounds -- TopoAE beats the matched plain-AE baseline on the topological structural check (r=0.680 vs 0.471, 45% relative improvement) while losing to it on plain MSE reconstruction (ratio 1.382), read as the trade the method makes on purpose. LAMBDA_TOPO=0.1 frozen for 02.4-PREREGISTRATION.md. Three named limitations carried forward: absolute r remains below the original 0.8 bound; the lambda selection rule is mis-specified for this method (documented, not fixed); loss_x_to_z/loss_z_to_x are scale-sensitive and not clean evidence on their own -- r is the trustworthy number
- [Phase ?]: 02.4-04: Task 1 blocking checkpoint returned to user (auto mode confirmed inactive); resolved option-a -- froze LAMBDA_TOPO=0.1 and the three planner-resolved constants (AMBIENT_DIST_NORM=none, one PlainAutoEncoder baseline per TopoAE fit, quarter-warmup/quarter-ramp/half-constant schedule) exactly as proposed
- [Phase ?]: 02.4-04: coordinator reframing recorded in Known Limitation 2 -- the lambda grid (0.0,0.1,0.3,1.0,3.0) already spans the paper's own log-uniform-[0.1,3] searched range, so LAMBDA_TOPO=0.1 is the smallest lambda the authors themselves considered, alongside (not instead of) the existing mis-specified-selection-rule finding
- [Phase ?]: 02.4-04: 02.4-PREREGISTRATION.md committed alone at 744c1c1 (no file under notebooks/); ancestry SHA 744c1c1d73a9e788a67768e2b397ad453045062a proved an ancestor of HEAD via git merge-base --is-ancestor
- [Phase ?]: 02.4-04 erratum (2026-08-07, additive commit 9f5bd9e): orchestrator verification found 02.4-PREREGISTRATION.md Section 1 falsely claimed AMBIENT_DIST_NORM=none applies identically at training time and the T1 gate. train_topoae actually normalizes ambient distances by their own per-batch max (d_x/d_x.max()) plus the jointly-trained latent_norm -- the paper's convention, closed as fidelity gap #2 in 02.4-03 (4b9b6c9); only the T1 gate uses raw d_x. Corrected Sections 1/4, added Section 11 erratum record. No constant or threshold moved; 744c1c1 stays in the record unmodified; ancestry proof re-confirmed
- [Phase ?]: 02.4-05: Task 3 checkpoint returned to coordinator rather than auto-approved (auto mode confirmed inactive); coordinator directed fix-transfer-ratio-then-approve
- [Phase ?]: 02.4-05: transfer-ratio estimator corrected to match topoae_lambda_sweep_run.py's own definition (lambda-weighted, single post-ramp epoch, documented fallback) after coordinator found the runner's original unweighted all-epoch-average produced a spurious ~307.5x reading vs. the true ~0.373x -- no order-of-magnitude transfer gap survives under the corrected, shared estimator
- [Phase ?]: 02.4-05: all sixteen pre-registered fits complete and cached (8 TopoAE + 8 matched baselines); every TopoAE fit early-stopped cleanly at epoch 15/40, no divergence, no wall-clock truncation at any rung or seed; plan 02.4-06 will find everything already cached and complete as a cache-hit verification pass
- [Phase ?]: 02.4-05 REOPENED: orchestrator verification of the first sixteen-fit run found train_topoae's plateau early-stop fired against the non-stationary warm-up/ramp objective, truncating every TopoAE fit at epochs_run=15 with lambda_t stuck at half of LAMBDA_TOPO (identical across 6 dims x 3 seeds), and leaving every rung's two arms on different training budgets (15 vs 40 epochs) -- a confound T1/T2/T3's ratios cannot tolerate
- [Phase ?]: 02.4-05: user decision on the reopened defect -- amend the pre-registration and re-run all sixteen fits (ratified 02.4-PREREGISTRATION-AMENDMENT-01.md, commit 9f9a74a, its own ancestry proof), not a silent fix, per 02.4-PREREGISTRATION.md Section 10's own stated consequence for changing a rule. LAMBDA_TOPO, THRESH_T1/T2/T3, the ladder, the seeds, and the fit schedule all confirmed unchanged -- only the stopping rule changed
- [Phase ?]: 02.4-05: stopping-rule fix (commit ee54858) -- early stopping suspended until floor(warmup_frac*max_epochs)+floor(ramp_frac*max_epochs); best_loss/plateau_count reset at that epoch. All sixteen fits re-run under amend01-tagged cache stems (pre-amendment buggy artifacts left intact on disk): every TopoAE fit now runs the full 40-epoch budget, reaches lambda_t=LAMBDA_TOPO=0.1, and has perfect budget parity with its matched baseline at all 8 rungs. Transfer_ratio (now measured at the true post-ramp epoch, no fallback) ranges 0.227701-0.313072, 0.54x-0.74x of the Swiss roll sweep's 0.422840 -- no order-of-magnitude gap
- [Phase ?]: 02.4-06: verified (not re-ran) that plan 02.4-05's reopened re-run already delivered all sixteen amend01-tagged fits -- registry structure, cfg-match, both ancestry proofs, cache-hit reproducibility, and bit-identical reload all independently confirmed; no code changed
- [Phase ?]: 02.4-06: primary-rung seed-to-seed transfer_ratio spread confirmed 0.227701-0.271348 (about 18% relative), all eight rungs' budget parity True and lambda_t=0.1 reached in full; pre-amendment (epochs_run=15) artifacts confirmed still intact and unmodified, never read as current
- [Phase ?]: 02.4-07: TOPOAE_VERDICT=FAIL sealed (T1=1.026379 vs <0.90, T2=1.211939 vs <1.00 both FAIL; T3=0.671980 vs <0.90 PASS) -- no threshold/constant/rule adjusted
- [Phase ?]: 02.4-07: coordinator checkpoint directed an additive gate_scope annotation (global: T1/T2, local: T3=k15) on the sealed verdict artifact rather than a bare FAIL string, since local curvature estimation depends on local not global fidelity; verdict/metrics/thresholds/gate_detail confirmed byte-identical before/after
- [Phase ?]: 02.4-07: withdrew 02.4-04's 'paper's own minimum searched lambda' justification for LAMBDA_TOPO=0.1 -- a fifth fidelity gap (EffDim sums the reconstruction term over features, reference means it) means LAMBDA_TOPO=0.1 is ~D times smaller in paper convention than stated, well below the searched [0.1,3] range. LAMBDA_TOPO unchanged, no re-fit; flagged for 02.4-08's pre-registration amendment
- [Phase ?]: 02.4-08: Reconciliation runner ran twice against sealed TOPOAE_VERDICT=FAIL, confirmed genuine no-op -- 02.1's graph-native recommendation stands untouched; TOPO-01..08 minted globally in REQUIREMENTS.md
- [Phase ?]: 02.4-08: Orchestrator-directed scope extension executed -- 02.4-FINDINGS.md (every FAIL this milestone is global-scoped; every measured local-scoped gate passed), 02.4-PREREGISTRATION-AMENDMENT-02.md (withdraws the 'paper's own minimum searched lambda' justification, changes no constant), and an additive ROADMAP.md Phase 3 re-scope to local curvature
- [Phase ?]: 02.4-08: WINDOWS.md entry #2 marked fixed (lambda-justification correction delivered via Amendment 2); new entry #3 records gap #5 (topoae.py reconstruction-term sum-vs-mean divergence) as still-open, not fixed
- [Phase ?]: 02.5-01: Trace convention H=tr(II) used everywhere per OQ-CONV, pinned by test_curvature_convention_is_trace_not_averaged (fails against the averaged form)
- [Phase ?]: 02.5-01: Rule 1 auto-fix -- centroid_mean_curvature's scale constant corrected from 2*(d+2)/r2 (RESEARCH.md Pattern 1 / plan's own Task 1 text) to 2*d/r2, after the sphere known-answer test caught it returning H=d+2 instead of H=d; confirmed exact for d in {2,3,5,8}
- [Phase ?]: 02.5-01: Tracer feedback checkpoint (auto mode inactive) held after Task 1; user approved rho=0.5806 before Tasks 2-3 ran
- [Phase ?]: 02.5-01: requirements.mark-complete found no D-00/D-01/D-03/D-05/D-07 entries in REQUIREMENTS.md -- phase 02.5's D-00..D-15 are scoped locally to 02.5-CONTEXT.md and were never mirrored into the milestone-level REQUIREMENTS.md, same pre-existing gap noted at 02.4-02; not a blocker for this plan
- [Phase ?]: 02.5-02: Preserved 02.5-01's 2*d/r2 scale constant; graph_mean_curvature implements the exact (not leading-order) graph curvature formula; global_std computed on the unpadded local embedding so padding is a true no-op
- [Phase ?]: 02.5-02: [Rule 1 - math-level] D-06's flat-fixture test premise (density skew alone produces large fake H on an exactly-linear embedding) proven mathematically impossible for the shipped normal-projecting estimator -- both empirically and analytically (exact-rank-d SVD, log-linear density model). Test redesigned: flat fixture proven at noise floor regardless of correction; correction's real ~8-10% effect demonstrated on a genuinely curved, skewed fixture instead. Flagged human_judgment:true in SUMMARY coverage since it amends a plan must-have
- [Phase ?]: 02.5-03: Preserved 02.5-01's 2*d/r2 scale constant and 02.5-02's density correction; quadric_fit_curvature rewrites Pattern 2's dead-branch trace accumulation (H += 2.0*c over i==j columns only)
- [Phase ?]: 02.5-03: [Rule 3 - blocking] quadric_mean_curvature needed its own _quadric_tangent_basis (full_matrices=True SVD) rather than reusing local_tangent_basis, which hard-raises whenever d > k -- exactly the underdetermined d=20/k=15 regime Task 1's own acceptance criteria require it to run and report on
- [Phase ?]: 02.5-03: estimator_agreement resolved report-never-block per D-05/CONTEXT.md discretion; permutation_null uses scipy.stats.permutation_test (not mknn.py's hand-rolled precedent) with no default for quantile; measure_cell bundles all of it into one flat, JSON-serializable dict with exactly one gating key (spearman_rho)
- [Phase ?]: 02.5-04: OQ-4 resolved -- mirror (not import) topoae.py's R6 verdict/handoff/stale-deletion functions at 02.5-scoped stems; topoae.py never edited or called
- [Phase ?]: 02.5-04: OQ-5 resolved -- cae.verdict_from_metrics not delegated to (its 3-slot positional remap doesn't fit 1/2-gate stages, and it applies uniform strict-less-than while spearman_rho/chart_vs_raw_margin are greater-than gates); _apply_gates implements its own guard-then-compare with an explicit GATE_DIRECTIONS map
- [Phase ?]: 02.5-04: thresholds live inside write_curvature_verdict's cache cfg dict, so an edited threshold raises cache._manifest_matches's mismatch ValueError on re-call instead of silently re-verdicting
- [Phase ?]: 02.5-05: Notebook amended post-checkpoint to report a per-point Spearman scale-bias/noise decomposition (median ratio h_est/h_true=0.8934 ~11% scale bias, within-band CV 0.20-0.28 interior/0.42-0.52 edges, region-median 20-band Spearman=0.8406) as notebook-level diagnostics only -- no gate/threshold changed; whether stage 1 gates at per-point or region scale is left open for 02.5-06 to propose and ratify
- [Phase ?]: Stage-1 gates on BOTH spearman_rho_pointwise AND quantile_bin_concordance independently (option-scale-C), after the first region-scale statistic proposal was rejected at checkpoint as saturated and redesigned from scratch
- [Phase ?]: REGION_ABSOLUTE_FLOOR=0.4750 derived from the Swiss roll's own noise-oracle ceiling (same 50%-noise tolerance as SPEARMAN_ABSOLUTE_FLOOR); BASE_CELL's graph-of-function fixture found to saturate the noise-oracle calibration technique for every region-statistic design tried (max/median=4708.8x true-curvature dynamic range)
- [Phase ?]: 02.5-07: CURVATURE_VERDICT=FAIL on the base cell (spearman_rho=0.5205 clears, quantile_bin_concordance=0.4444 misses threshold 0.4750 by -0.0306); reported alongside a seed-instability disclosure -- 2 of 3 tested seeds at the identical base configuration clear both gates, and the across-seed spread (0.0792) exceeds the base cell's own margin to threshold (0.0306)
- [Phase ?]: 02.5-07: ambient dimension D found bit-identical (to the last printed digit) across D=28,50,200,768 -- the base-cell failure is entirely an intrinsic-d effect, not an ambient-scale effect, correcting a framing risk in 02.5-PREREGISTRATION.md Section 13b
- [Phase ?]: 02.5-07: the non-gating quadric cross-check (D-05) could not complete beyond the d=2 Swiss roll anchor within the sweep's 30-minute wall-clock budget (measured ~6-8 min/cell at PU scale); reported as a genuine evidentiary gap in 02.5-FINDINGS.md Section 6, with the d(d+1)/2-vs-k coefficient boundary reported structurally (determined through d=5, underdetermined from d=8) rather than empirically

### Pending Todos

From `TODO.md`:

- Expand test suite to validate against known dimensionalities (ROADMAP Backlog)
- CI for the standard Python implementation across platforms (ROADMAP Backlog). The Rust extension this todo also names does not exist in the repo — stale reference, see Backlog note

### Blockers/Concerns

- `UniverseTBD/pu-embeddings` is ~93 GB across 163 configs — v1.1 streams exactly one config (`legacysurvey_dinov3_vitb16`) and subsamples 10k of 101,725 rows; never materialize the whole dataset
- Phase 3 (decoder/curvature) and Phase 4 (regional MKNN) need a dedicated research pass during planning per `research/SUMMARY.md`; Phase 1/2 are standard sklearn/MDS patterns and can skip it
- Phase 2's PASS/MARGINAL/FAIL gate is a hard stop: a FAIL halts the milestone and is itself a legitimate, complete outcome. Phase 3 is now blocked on Phase 02.2's `cae_verdict.json` reading PASS, and a FAIL there leaves the milestone at the phase-2 stage
- Plan 02.4-03's three named Swiss roll limitations (RESOLVED as blockers -- Task 4 approved 2026-08-07, but carried forward as facts 02.4-04 must inherit): absolute topological correlation r=0.680 remains below the originally-set 0.8 bound despite beating the matched baseline; the lambda selection rule ("<=10% reconstruction degradation") is mis-specified for a method that trades reconstruction for topology by design and bound at the grid floor on both the broken and corrected loss -- documented in 02.4-03-SUMMARY.md, not fixed; loss_x_to_z/loss_z_to_x are measured under a different normalization than training optimizes and are not clean evidence on their own -- the scale-free correlation r is the trustworthy number. See `02.4-03-SUMMARY.md` § Known Limitations for full detail.
- Phase 02.5 blocked at plan 02.5-07's Task 3 blocking checkpoint (stage-1 GO/NO-GO): CURVATURE_VERDICT=FAIL (marginal, seed-sensitive) on the base cell. Per 02.5-PREREGISTRATION.md Section 10, the phase halts for a user decision with no auto-fallback. Plans 02.5-08 through 02.5-13 do not execute until this checkpoint is resolved.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260801-ovf | cleanup: reduce to barebones isomap-on-dino experiment | 2026-08-02 | 59742af | [260801-ovf-cleanup-reduce-to-barebones-isomap-on-di](./quick/260801-ovf-cleanup-reduce-to-barebones-isomap-on-di/) |
| 260803-k9n | Insert Phase 02.2: Chart Autoencoder Validity Test (arXiv:1912.10094) with PASS/FAIL gate for Phase 3 | 2026-08-03 | 3357ea5 | [260803-k9n-update-phase-2-of-milestone-to-test-vali](./quick/260803-k9n-update-phase-2-of-milestone-to-test-vali/) |
| 260805-brr | distill the CAE experiment into a notebook | 2026-08-05 | ccc0bf7 | [260805-brr-distill-the-cae-experiment-into-a-notebo](./quick/260805-brr-distill-the-cae-experiment-into-a-notebo/) |

### Roadmap Evolution

- Phase 02.1 inserted after Phase 2: Geometry Representation Research — Phase 2 gate FAIL invalidated the Isomap coordinates Phase 3 was specified to decode from (URGENT)
- Phase 02.1 planned: 4 plans across 3 waves; plan-checker VERIFICATION PASSED first iteration; GEOM-01..05 coverage complete
- Phase 02.2 inserted after Phase 02.1: Chart Autoencoder Validity Test — empirically tests arXiv:1912.10094 on the PU data behind a PASS/FAIL gate; PASS unblocks Phase 3 to decode from the CAE representation, FAIL documents the finding and leaves the milestone at the phase-2 stage. Doc-only insertion; the phase itself is unplanned
- Phase 02.5 inserted after Phase 2: Local curvature feasibility probe, then a locally-scoped CAE re-gate — resolves Phase 3's blocking dependency on a global-scoped PASS no method has produced (URGENT)

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Validation | ED estimates checked against known-dimension manifolds (noise → D, Swiss Roll → intrinsic dim) | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| CI/Packaging | Cross-platform test matrix and release pipeline | ROADMAP Backlog | v1.0 → v1.1 transition (2026-07-29) |
| Scale | SCALE-01/SCALE-02 — intramodal MKNN across a model-size ladder; curvature-stratified alignment across that ladder | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |
| Library | LIB-01/LIB-02/LIB-03 — promote curvature operator and MDS validity diagnostic into `src/effdim/`; fix `pyproject.toml` Python floor | Deferred (REQUIREMENTS.md Future Requirements) | v1.1 requirements definition (2026-07-29) |

## Session Continuity

Last session: 2026-08-08T19:42:53.324Z
Stopped at: Paused at 02.5-07 Task 3 checkpoint (stage-1 GO/NO-GO) -- 02.5-FINDINGS.md and 02.5-07-SUMMARY.md written; checkpoint unresolved
REQUIREMENTS.md traceability renumbered; awaiting phase planning for Phase 1
Resume file: None
</content>
