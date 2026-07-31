---
phase: 02-eigenspectrum-audit-validity-gate
plan: 02
subsystem: data
tags: [numpy, scipy, sklearn, isomap, classical-mds, kneedle, jupyter]

# Dependency graph
requires:
  - phase: 02-eigenspectrum-audit-validity-gate
    provides: "02-01: the full 10,000-value classical-MDS eigenspectrum (EIGVALS_ALL, EIGVALS_TOP, EIGVECS_TOP, MDS_COORDS), N_POSITIVE, K_EFF, R_STAT, M_STAT, GATE_VERDICT=FAIL, and mds_eigenspectrum_43cf438bc944c509.npz"
provides:
  - "Both residual-variance curves (Tenenbaum + eigenvalue cross-check), a from-scratch deterministic kneedle elbow finder, and its two-disjoint-draw stability check (§6.5)"
  - "The frozen embedding dimension D_FROZEN=5 via the freeze-at-elbow branch, with the classical-MDS nesting claim verified numerically to worst relative difference 1.207e-14 (§6.6)"
  - "mds_residuals_43cf438bc944c509.npz — cached residual curves plan 02-03 can read for the verdict artifact"
affects: [02-03-verdict-artifact-enforcement, phase-3-decoder-curvature, phase-4-region-partition]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "From-scratch deterministic kneedle/maximum-curvature elbow finder (_kneedle): normalize both axes to [0,1], perpendicular distance from the endpoint chord, tie-break to lower index, explicit raise on <3 points or zero y-range"
    - "Rotation-invariant nesting verification: compare pairwise distances (not raw coordinates) between a sliced sklearn embedding and a sliced hand-rolled classical-MDS solution via np.allclose, immune to eigenvector sign flips"
    - "Vectorised R^2 residual curve via cumulative squared coordinate differences (one pass over the coordinate axis) rather than a loop over d"

key-files:
  created:
    - "notebooks/.cache/mds_residuals_43cf438bc944c509.npz (gitignored, 2.6 KB)"
    - "notebooks/.cache/mds_residuals_43cf438bc944c509.meta.json (gitignored)"
  modified:
    - "notebooks/01_manifold_and_gate.ipynb (105 -> 107 cells; appended §6.5-§6.6)"

key-decisions:
  - "Checkpoint decision (Task 2, gate=blocking): freeze-at-elbow selected by the human. D=5 CONFIRMED and APPROVED, since ELBOW_D=5 <= N_COMPONENTS=18. The halt-for-refit and halt-elbow-unstable branches were both inapplicable (their preconditions did not occur) and were not taken."
  - "D_FROZEN=5 is the dimension of record inside a gate that already FAILed (M_STAT=0.412071 against the 0.15 MARGINAL bound). Freezing it satisfies SPEC-05/D-07 so plan 02-03's verdict artifact is self-contained under SPEC-06 — it is not an endorsement of d=5 as a Phase 3 decoder width, because the FAIL halts Phase 3 regardless of what d this section froze."
  - "ELBOW_D_EIGEN=8 (the eigenvalue cross-check curve's own elbow) was reported alongside ELBOW_D=5 but never substituted as the freeze source, per D-06."

requirements-completed: [SPEC-04, SPEC-05]

coverage:
  - id: D1
    description: "Both residual-variance curves computed, cached, and plotted on one axis; the elbow is read from the Tenenbaum curve only, with the eigenvalue curve carried as a labelled cross-check; the two-disjoint-draw stability check passes (ELBOW_D == ELBOW_D_CHECK == 5); CURVE_DIVERGENCE_MAX=0.697664 reported alongside R_STAT/M_STAT as corroborating, non-verdict-entering evidence"
    requirement: "SPEC-04"
    verification:
      - kind: other
        ref: "jupyter nbconvert --execute --inplace notebooks/01_manifold_and_gate.ipynb; §6.5 executed output shows ELBOW_D=5, ELBOW_D_CHECK=5, ELBOW_D_EIGEN=8, CURVE_DIVERGENCE_MAX=0.697664 at d=5; independent recompute from the persisted npz outside the notebook reproduces elbow(tenenbaum)=5, elbow(check draw)=5, elbow(eigen)=8, max divergence=0.6976644052911366 exactly"
        status: pass
    human_judgment: false
  - id: D2
    description: "d frozen at D_FROZEN=5 before any decoder exists via the freeze-at-elbow branch (ELBOW_D=5 <= N_COMPONENTS=18), cross-checked against the human-confirmed checkpoint value, with the classical-MDS nesting claim verified numerically (not only argued) and the invariant D_FROZEN <= N_COMPONENTS asserted; the halt-for-refit branch is implemented with the full required remediation content but was not exercised"
    requirement: "SPEC-05"
    verification:
      - kind: other
        ref: "§6.6 executed output: D_FROZEN=5, ISOMAP_COORDS_D.shape=(10000, 5), nesting max relative difference=1.207e-14 over the 200,000-pair sample; automated verify block confirms 'assert False' halt branch present with n_components/fit_key/1.55/ANALYSIS_CFG all in the halt message"
        status: pass
    human_judgment: false

duration: ~15min active execution across two sessions (Task 1 + blocking checkpoint in a prior session; Task 3 + summary in this continuation session)
completed: 2026-07-31
status: complete
---

# Phase 2 Plan 2: Residual Elbow and Frozen d Summary

**D_FROZEN=5 via the freeze-at-elbow branch — deterministic kneedle elbow on the Tenenbaum residual curve, stable across two disjoint 200,000-pair draws, with the classical-MDS nesting slice verified numerically to 1.207e-14 relative difference**

## D_FROZEN and the branch taken

**`D_FROZEN = 5`**, via the **freeze-at-elbow** branch (`§6.6`, `ELBOW_D=5 <= N_COMPONENTS=18`). This is the one-way value Phases 3 and 4 are defined on: the decoder's input width, the curvature field's coordinate domain, and the basis for Phase 4's region partition. It was confirmed at the Task 2 `checkpoint:decision` gate — the human selected `freeze-at-elbow` and explicitly confirmed the integer `d=5` — and `§6.6` cross-checks the computed `D_FROZEN` against that confirmed value, halting on divergence (it did not diverge).

`d = 5` is the dimension of record **inside a gate that has already `FAIL`ed** (`GATE_VERDICT=FAIL`, measured in 02-01 and confirmed robust across k in `{5,10,30}` by the pre-registered k-sensitivity re-fit). Freezing it here is required so plan 02-03's verdict artifact is self-contained under SPEC-06; it is not an endorsement of `d=5` as Phase 3's decoder width, because the FAIL halts Phase 3 regardless of what `d` this section froze.

## Performance

- **Duration:** ~15 min active execution (Task 1 and the Task 2 checkpoint occurred in a prior session; Task 3 and this SUMMARY were completed in this continuation session after the human's `freeze-at-elbow` decision)
- **Completed:** 2026-07-31
- **Tasks:** 3/3 complete (Task 1: auto, Task 2: checkpoint:decision — resolved, Task 3: auto)
- **Files modified:** 1 (notebook), 2 cache artifacts created (gitignored)

## Accomplishments

- **§6.5** computes both residual-variance curves on the pre-registered `d`-grid
  `D_GRID = 1..40` (the full `D_SWEEP_MAX` ceiling — `K_EFF=40` was not bounded short by
  `N_POSITIVE`). `RESIDUAL_EIGEN` (`1 - cumsum(positive descending)/sum(all positive)`) is
  asserted non-increasing and in `[0, 1]`. `RESIDUAL_TENENBAUM` (`1 - Pearson r^2` between
  geodesic and embedded pair distances) is computed vectorised via a cumulative sum over
  coordinates on a 200,000-pair sample (61.0 MiB intermediate array, printed before
  allocation and released before the second draw). Both curves, the second-draw curve, and
  the `d`-grid are cached in `mds_residuals_43cf438bc944c509.npz` (2.6 KB) via `npz_cache`,
  sharing the spectrum artifact's `fit_key`-plus-ceiling cfg contract.
- **The kneedle elbow finder** (`_kneedle`) is a from-scratch deterministic
  maximum-curvature implementation with no in-repo analog: normalizes both axes to `[0,1]`,
  takes the perpendicular distance from the endpoint chord, breaks ties to the lower index,
  and raises explicitly on degenerate input (fewer than three points, or zero `y`-range).
  Its body contains no gate-threshold literal (`0.10`, `0.05`, `0.25`, `0.15` all absent),
  verified by an automated grep-style check.
- **`ELBOW_D = 5`**, agreeing exactly with **`ELBOW_D_CHECK = 5`** on the second,
  independently seeded, disjoint 200,000-pair draw — the stability check passed with no
  halt. The eigenvalue cross-check curve's own elbow, **`ELBOW_D_EIGEN = 8`**, is reported
  alongside but never substituted as the freeze source (D-06): the two differ by +3
  dimensions, read as coordinate variance not buying geodesic fidelity.
- **`CURVE_DIVERGENCE_MAX = 0.697664`** at `d=5`, reported next to `R_STAT=0.052419` and
  `M_STAT=0.412071` as corroborating evidence for the already-FAILed gate — very large on
  curves bounded in `[0,1]`, reinforcing the non-Euclideanity `m` already flagged. It does
  not enter the verdict (D-01 fixed the verdict as a function of `r`/`m` only).
- **The Task 2 checkpoint** (`checkpoint:decision`, `gate="blocking"`) was resolved by the
  human: **`freeze-at-elbow` selected, `d=5` CONFIRMED and APPROVED.** The `halt-for-refit`
  branch (requires `ELBOW_D > 18`) and `halt-elbow-unstable` branch (requires the stability
  assertion to have fired) were both inapplicable — neither precondition occurred.
- **§6.6** implements the freeze as a single `if/else` on `ELBOW_D <= N_COMPONENTS`, no third
  path, no silent capping at 18. On the freeze branch: `D_FROZEN = int(ELBOW_D) = 5`,
  `ISOMAP_COORDS_D = EMBEDDING_ISOMAP[:, :5]` (shape `(10000, 5)`, float64), cross-checked
  against the human-confirmed `D_FROZEN_CONFIRMED = 5` with a halt on divergence. The D-07
  classical-MDS nesting claim was **verified numerically, not only argued**: pairwise
  distances from `ISOMAP_COORDS_D` and from `MDS_COORDS[:, :5]` over the 200,000-pair sample
  agree to a worst relative difference of **`1.207e-14`** (`np.allclose`, rotation-invariant
  so immune to eigenvector sign flips) — essentially machine precision, confirming the
  slice is the exact 5-dimensional classical-MDS solution, not a projection. The invariant
  `D_FROZEN <= N_COMPONENTS` (`5 <= 18`) is asserted explicitly.
- **The halt branch** (`ELBOW_D > N_COMPONENTS`) was implemented in full per D-08 — observed
  elbow, required `n_components`, the ~1.55 GiB / one-fresh-Isomap-fit cost, the exact
  `ANALYSIS_CFG["n_components"]` constant to edit, the automatic `fit_key` change, the need
  to regenerate every `§6` artifact, and a quote of Phase 1's `notes_for_phase2` line
  anticipating this branch — but was **not exercised** this run, since `ELBOW_D=5 <= 18`.
- The notebook (107 cells) was executed end-to-end via
  `jupyter nbconvert --to notebook --execute --inplace` with zero error cells and every code
  cell carrying a non-null execution count. An independent recompute of both `mds_residuals`
  curves outside the notebook (reimplementing the chord-distance kneedle from scratch against
  the persisted npz) reproduced `elbow(tenenbaum)=5`, `elbow(check draw)=5`,
  `elbow(eigen)=8`, `max divergence=0.6976644052911366` exactly.

## The measured elbow and freeze outcome

| Quantity | Value | Reading |
|---|---|---|
| `ELBOW_D` | 5 | Tenenbaum curve, first draw, seed `R2_PAIR_SEED=20260731` |
| `ELBOW_D_CHECK` | 5 | second disjoint draw, seed `R2_PAIR_SEED_CHECK=20260732` — exact agreement, no halt |
| `ELBOW_D_EIGEN` | 8 | eigenvalue cross-check curve's own elbow; NOT the freeze source |
| `CURVE_DIVERGENCE_MAX` | 0.697664 | at `d=5`; large on `[0,1]`-bounded curves, reinforcing the FAIL |
| `K_EFF` | 40 | = `D_SWEEP_MAX`; the `d`-grid reached the full pre-registered ceiling, not bounded short |
| `N_COMPONENTS` | 18 | Phase 1's fit capacity, not the elbow's source |
| `R_STAT` / `M_STAT` / `GATE_VERDICT` | 0.052419 / 0.412071 / FAIL | unchanged from 02-01, carried forward for context only |
| **Task 2 checkpoint outcome** | **freeze-at-elbow, `d=5` confirmed** | human decision, blocking gate |
| **`D_FROZEN`** | **5** | frozen via `EMBEDDING_ISOMAP[:, :5]` |
| Nesting cross-check worst relative diff | `1.207e-14` | `np.allclose` between `ISOMAP_COORDS_D` and `MDS_COORDS[:, :5]` pairwise distances, 200,000-pair sample |
| `mds_residuals_{fit_key}.npz` size | 2.6 KB | well under the 10 MB budget |

## Task Commits

Each task was committed atomically:

1. **Task 1: Section 6.5 — both residual curves, the deterministic elbow, and its stability check** — `5cf9a19` (feat, prior session)
2. **Task 2: Confirm the one-way freeze of the embedding dimension d** — `checkpoint:decision`, `gate="blocking"`; resolved by human: `freeze-at-elbow`, `d=5` confirmed (no commit; decision recorded here and cross-checked in code)
3. **Task 3: Section 6.6 — freeze d, or halt with the re-fit decision documented** — `539dafa` (feat, this session)

**Plan metadata:** committed separately after this SUMMARY (docs commit).

## Files Created/Modified

- `notebooks/01_manifold_and_gate.ipynb` — grown from 105 to 107 cells; `§6.6` appended
  (1 markdown + 1 code cell) with real executed output from a full
  `jupyter nbconvert --execute --inplace` run (§6.5's 3 cells were appended in the prior
  session's commit `5cf9a19`).
- `notebooks/.cache/mds_residuals_43cf438bc944c509.npz` (gitignored, 2.6 KB) —
  `d_grid`, `residual_tenenbaum`, `residual_tenenbaum_check`, `residual_eigen`,
  `residual_seconds` (created in the prior session, read here).
- `notebooks/.cache/mds_residuals_43cf438bc944c509.meta.json` (gitignored) — the cfg
  manifest (`fit_key`, `d_sweep_max`, `k_eff`, `r2_pair_count`, `r2_pair_seed`,
  `r2_pair_seed_check`).

## Decisions Made

- Followed the plan's pre-registered `d`-range, criterion, and pair sample exactly: no
  threshold, criterion, or sample size was adjusted after seeing the real curves.
- Task 2's checkpoint resolved to `freeze-at-elbow` with `d=5` confirmed by the human, which
  is the branch `§6.6` implements as the freeze path; the two halt branches were implemented
  for completeness per D-08/the plan's must_haves but neither was exercised this run.
- `_NESTING_RTOL = 1e-6` / `_NESTING_ATOL = 1e-8` were chosen as a stated, generous tolerance
  for the D-07 nesting verification; the actual measured worst relative difference
  (`1.207e-14`) came in roughly 8 orders of magnitude tighter than the stated bound,
  consistent with `§6.2`'s `rtol=1e-8` sklearn cross-check on the same fit.

## Deviations from Plan

None — plan executed exactly as written. Both tasks (`Task 1`, `Task 3`) implemented every
`must_haves.truths` item and every `acceptance_criteria` bullet listed in `02-02-PLAN.md`;
the checkpoint (`Task 2`) was resolved by an explicit human decision rather than an
auto-fix, so no deviation rule applies to it.

## Issues Encountered

None. The prior session's `np.asarray`-on-read-only-memmap trap (already fixed in 02-01)
did not recur here — `EMBEDDING_ISOMAP` was already a resident `float64` array bound in
`§4.2`, and `§6.6` only slices it, so no memmap/view issue arose.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **`D_FROZEN = 5`** is bound and printed in the committed notebook output, ready for plan
  02-03 to write into `gate_verdict_{fit_key}.json` as `d_frozen`.
- `ISOMAP_COORDS_D` (shape `(10000, 5)`, float64) is bound and available for any downstream
  consumer that reads the notebook's live state; the authoritative machine-readable record
  is still the plan 02-03 verdict artifact per SPEC-06, not this notebook binding alone.
- **`GATE_VERDICT = FAIL` remains the settled outcome.** This plan froze `d` because SPEC-05
  requires it to be frozen and recorded before any decoder exists, regardless of the gate's
  own verdict — but the FAIL itself is not reopened or re-litigated here, and Phase 3 must
  not proceed past plan 02-03's sealed verdict artifact without a milestone-level decision on
  the FAIL (re-fit at a different k, resample with a new seed, or accept the documented FAIL
  and stop — per 02-01-SUMMARY.md and the k-sensitivity re-fit's Rule A outcome).
- `pyproject.toml`, `src/effdim/`, and `notebooks/pu_manifold/` are confirmed byte-identical
  to their pre-plan state (`git diff --quiet` passed after Task 3).
- Plan 02-03 (verdict artifact enforcement) can now read `D_FROZEN`, `ELBOW_D`,
  `ELBOW_D_CHECK`, `ELBOW_D_EIGEN`, `CURVE_DIVERGENCE_MAX`, and
  `mds_residuals_{fit_key}.npz` directly — no re-run of the residual computation needed.

## Self-Check: PASSED

- FOUND: `notebooks/01_manifold_and_gate.ipynb` (107 cells, zero error cells)
- FOUND: `notebooks/.cache/mds_residuals_43cf438bc944c509.npz`
- FOUND: `notebooks/.cache/mds_residuals_43cf438bc944c509.meta.json`
- FOUND: commit `5cf9a19` (Task 1)
- FOUND: commit `539dafa` (Task 3)

---
*Phase: 02-eigenspectrum-audit-validity-gate*
*Completed: 2026-07-31*
