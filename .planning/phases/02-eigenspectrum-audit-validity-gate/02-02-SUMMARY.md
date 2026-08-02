---
phase: 02-eigenspectrum-audit-validity-gate
plan: 02
subsystem: data
tags: [numpy, scipy, sklearn, isomap, classical-mds, kneedle, jupyter]

requires:
  - phase: 02-eigenspectrum-audit-validity-gate
    provides: "02-01: full 10,000-value spectrum (EIGVALS_ALL/TOP, EIGVECS_TOP, MDS_COORDS), N_POSITIVE, K_EFF, R_STAT, M_STAT, GATE_VERDICT=FAIL, mds_eigenspectrum_43cf438bc944c509.npz"
provides:
  - "Both residual-variance curves (Tenenbaum + eigenvalue cross-check), deterministic kneedle elbow finder, two-disjoint-draw stability check (§6.5)"
  - "D_FROZEN=5 via freeze-at-elbow, nesting claim verified numerically to worst relative difference 1.207e-14 (§6.6)"
  - "mds_residuals_43cf438bc944c509.npz — cached residual curves for 02-03"
affects: [02-03-verdict-artifact-enforcement, phase-3-decoder-curvature, phase-4-region-partition]

tech-stack:
  added: []
  patterns:
    - "From-scratch deterministic kneedle (_kneedle): axes normalized to [0,1], perpendicular distance from endpoint chord, ties to lower index, raises on degenerate input"
    - "Rotation-invariant nesting verification: pairwise distances (not raw coordinates) via np.allclose, immune to eigenvector sign flips"
    - "Vectorised R² residual curve via cumsum over coordinates, no loop over d"

key-files:
  created:
    - "notebooks/.cache/mds_residuals_43cf438bc944c509.npz (gitignored, 2.6 KB)"
    - "notebooks/.cache/mds_residuals_43cf438bc944c509.meta.json (gitignored)"
  modified:
    - "notebooks/01_manifold_and_gate.ipynb (105 -> 107 cells; appended §6.5-§6.6)"

key-decisions:
  - "Task 2 checkpoint (blocking): freeze-at-elbow selected by the human, d=5 CONFIRMED (ELBOW_D=5 <= N_COMPONENTS=18). Halt branches inapplicable."
  - "D_FROZEN=5 is the dimension of record inside an already-FAILed gate (M_STAT=0.412071 vs 0.15). Freezing satisfies SPEC-05/D-07 so 02-03's artifact is self-contained under SPEC-06 — not an endorsement of d=5 as decoder width; the FAIL halts Phase 3 regardless."
  - "ELBOW_D_EIGEN=8 reported alongside ELBOW_D=5, never substituted as freeze source (D-06)."

requirements-completed: [SPEC-04, SPEC-05]

coverage:
  - {id: D1, description: "Both curves computed/cached/plotted; elbow from Tenenbaum only; stability check passes (ELBOW_D == ELBOW_D_CHECK == 5); CURVE_DIVERGENCE_MAX=0.697664 reported beside R_STAT/M_STAT as non-verdict-entering", requirement: "SPEC-04", verification: [{kind: other, ref: "nbconvert --execute --inplace; §6.5 output: ELBOW_D=5, ELBOW_D_CHECK=5, ELBOW_D_EIGEN=8; independent npz recompute reproduces all, max divergence=0.6976644052911366", status: pass}], human_judgment: false}
  - {id: D2, description: "D_FROZEN=5 frozen before any decoder via freeze-at-elbow, cross-checked against human-confirmed value, nesting verified numerically, D_FROZEN <= N_COMPONENTS asserted; halt branch implemented but not exercised", requirement: "SPEC-05", verification: [{kind: other, ref: "§6.6 output: D_FROZEN=5, ISOMAP_COORDS_D.shape=(10000, 5), nesting max relative difference=1.207e-14; halt content (n_components/fit_key/1.55/ANALYSIS_CFG) verified present", status: pass}], human_judgment: false}

duration: ~15min
completed: 2026-07-31
status: complete
---

# Phase 2 Plan 2: Residual Elbow and Frozen d Summary

**D_FROZEN=5 via freeze-at-elbow — kneedle elbow on the Tenenbaum residual curve, stable across two
disjoint 200,000-pair draws, nesting slice verified to 1.207e-14.**

## D_FROZEN and the branch taken

**`D_FROZEN = 5`** (`ELBOW_D=5 <= N_COMPONENTS=18`), confirmed at the blocking Task 2 checkpoint;
§6.6 cross-checks against the confirmed integer and did not diverge. `d=5` is the dimension of record
**inside a gate that already FAILed** — freezing satisfies SPEC-05/D-07 so 02-03's verdict artifact
is self-contained, not an endorsement of d=5 as Phase 3's decoder width.

## Measured outcome

| Quantity | Value | Reading |
|---|---|---|
| `ELBOW_D` | 5 | Tenenbaum curve, seed `R2_PAIR_SEED=20260731` |
| `ELBOW_D_CHECK` | 5 | disjoint draw, seed `R2_PAIR_SEED_CHECK=20260732` — exact agreement |
| `ELBOW_D_EIGEN` | 8 | eigenvalue cross-check curve; NOT the freeze source |
| `CURVE_DIVERGENCE_MAX` | 0.697664 | at d=5; large on [0,1]-bounded curves, reinforces the FAIL |
| `K_EFF` | 40 | = `D_SWEEP_MAX`; full pre-registered ceiling reached |
| `R_STAT` / `M_STAT` / verdict | 0.052419 / 0.412071 / FAIL | unchanged from 02-01 |
| **`D_FROZEN`** | **5** | `EMBEDDING_ISOMAP[:, :5]` |
| Nesting worst relative diff | 1.207e-14 | vs `MDS_COORDS[:, :5]` pairwise distances, 200,000 pairs |
| npz size | 2.6 KB | under the 10 MB budget |

## What ran

- §6.5: `D_GRID = 1..40`; `RESIDUAL_EIGEN` asserted non-increasing in [0,1];
  `RESIDUAL_TENENBAUM` vectorised (61.0 MiB intermediate, printed before allocation); both curves +
  check draw cached. `_kneedle` contains no gate-threshold literal (verified).
- §6.6: single `if/else`, no third path, no capping at 18. Freeze branch:
  `ISOMAP_COORDS_D` shape (10000, 5) float64, invariant `5 <= 18` asserted. Halt branch implemented
  in full per D-08 (elbow, required n_components, ~1.55 GiB cost, `ANALYSIS_CFG["n_components"]`,
  automatic fit_key change, Phase 1 notes_for_phase2 quote) — not exercised.
- Notebook (107 cells) executed end-to-end, zero error cells. Independent kneedle reimplementation
  against the persisted npz reproduces elbow(tenenbaum)=5, elbow(check)=5, elbow(eigen)=8,
  max divergence=0.6976644052911366 exactly.

## Commits

1. §6.5 (Task 1) — `5cf9a19`.
2. Task 2 checkpoint — human decision, no commit.
3. §6.6 (Task 3) — `539dafa`.

## Decisions / Deviations

- Pre-registered d-range, criterion, and pair sample followed exactly; nothing adjusted after seeing
  curves. `_NESTING_RTOL=1e-6`/`_NESTING_ATOL=1e-8` stated; measured 1.207e-14 came in ~8 orders
  tighter. No deviations; no issues.

## Next Phase Readiness

`D_FROZEN=5` ready for 02-03 to write as `d_frozen` in `gate_verdict_{fit_key}.json`.
`GATE_VERDICT=FAIL` remains settled — Phase 3 must not proceed past 02-03's sealed artifact without a
milestone-level decision on the FAIL. `pyproject.toml`/`src/effdim/`/`notebooks/pu_manifold/`
byte-identical to pre-plan state.

## Self-Check: PASSED

FOUND: notebook (107 cells), npz + meta.json, commits `5cf9a19`, `539dafa`.

---
*Phase: 02-eigenspectrum-audit-validity-gate* · *Completed: 2026-07-31*
