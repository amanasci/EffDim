---
phase: 02-eigenspectrum-audit-validity-gate
plan: 01
subsystem: data
tags: [scipy, numpy, sklearn, isomap, classical-mds, eigendecomposition, jupyter]

requires:
  - {phase: "01-data-loading-manifold-reconstruction", provides: "notebooks/.cache/isomap_43cf438bc944c509.joblib (frozen k*=15 Isomap fit) and phase1_handoff_43cf438bc944c509.json"}
provides:
  - "Full 10,000-value classical-MDS eigenspectrum (§6.0-§6.2), hand double-centred, cross-checked against sklearn's leading 18 eigenvalues"
  - "r/m negativity statistics and PASS/MARGINAL/FAIL gate verdict (§6.3)"
  - "Leading-spectrum table, steep-dropoff location, two diagnostic figures (§6.4)"
  - "mds_eigenspectrum_43cf438bc944c509.npz — cached spectrum read by 02-02/02-03"
affects: [02-02-residual-elbow-freeze-d, 02-03-verdict-artifact-enforcement, phase-3-decoder-curvature]

tech-stack:
  added: []
  patterns:
    - "Split eigensolve: eigvalsh (all values) + eigh(subset_by_index) (top-K vectors); no full 10,000x10,000 eigenvector array"
    - "In-place mean-form double-centring, verified equal to the literal J-form on two 50x50 inputs"
    - "Cell-index self-assertion proving pre-registered constants execute before consumers"

key-files:
  created: ["notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz (gitignored, 9.23 MiB)", "notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.meta.json (gitignored)"]
  modified: ["notebooks/01_manifold_and_gate.ipynb (90 -> 102 cells; appended §6.0-§6.4)"]

key-decisions:
  - "Measured gate outcome: R_STAT=0.052419 (passes r<0.10), M_STAT=0.412071 (fails the m<0.15 MARGINAL bound) -> GATE_VERDICT=FAIL. Legitimate hard-gate outcome, not an error."
  - "Rule 1 auto-fix: np.asarray on a read-only float64 memmap returns a view; in-place D2 **= 2 raised 'output array is read-only'. Fixed with np.array(..., copy=True)."

requirements-completed: [SPEC-01, SPEC-02, SPEC-03]

coverage:
  - {id: D1, description: "Full 10,000-value spectrum, length asserted, cross-checked against kernel_pca_.eigenvalues_ at rtol=1e-8 (worst diff 8.532e-15)", requirement: "SPEC-01", verification: [{kind: other, ref: "nbconvert --execute --inplace; §6.2 output", status: pass}], human_judgment: false}
  - {id: D2, description: "Leading spectrum large-and-positive; steep dropoff by largest log-ratio gap, distinguished from SPEC-04 elbow", requirement: "SPEC-02", verification: [{kind: other, ref: "§6.4 output: DROPOFF_INDEX=2, DROPOFF_RATIO=2.4447", status: pass}], human_judgment: false}
  - {id: D3, description: "r/m vs pre-registered thresholds, worse-of-two rule; classifier boundaries asserted on nine synthetic cases first", requirement: "SPEC-03", verification: [{kind: other, ref: "§6.3 output; recompute from npz reproduces R_STAT/M_STAT exactly", status: pass}], human_judgment: false}

duration: ~20min
completed: 2026-07-31
status: complete
---

# Phase 2 Plan 1: Eigenspectrum Audit Summary

**Full 10,000-eigenvalue classical-MDS spectrum from the k*=15 fit: GATE_VERDICT=FAIL
(R_STAT=0.052419, M_STAT=0.412071).**

## What ran

§6.0 pre-registers all gate constants (`R_MAX_PASS=0.10`, `M_MAX_PASS=0.05`,
`R_MAX_MARGINAL=0.25`, `M_MAX_MARGINAL=0.15`, `D_SWEEP_MAX=40`, `R2_PAIR_COUNT=200_000`,
`EQUIV_N=50`, `EQUIV_TOL=1e-12`, `SYMMETRY_RTOL=1e-10`, `ELBOW_TIE_BREAK="lower"`), guarded by a
cell-index assertion (index 90 < 93). §6.1 releases Phase 1 bindings, memory-maps `dist_matrix_`,
measures symmetry chunk-wise (max deviation 1.421e-14, bound 2.132e-09), double-centres in place.
§6.2 runs the split eigensolve, caches `mds_eigenspectrum_43cf438bc944c509.npz` (9.23 MiB),
cross-checks the top 18 against sklearn at rtol=1e-8 (worst diff 8.532e-15). §6.3 asserts nine
synthetic boundary cases (incl. `(0.10,0.0)->MARGINAL`, `(0.25,0.0)->FAIL`, `(0.05,0.20)->FAIL`)
before real data. §6.4 reports the top-20 (all strictly positive), locates the dropoff, renders
two figures. Notebook executed end-to-end twice via `nbconvert --execute --inplace`, zero error
cells.

## Measured gate outcome

| Statistic | Value | Bound | Reading |
|---|---|---|---|
| `N_POSITIVE` / negative | 4971 / 5029 | — | fewer than half positive |
| `LAMBDA_MAX_POS` | 3.230854e+03 | — | — |
| `LAMBDA_MIN_NEG` | -1.693588e+02 | — | — |
| `NOISE_FLOOR` | 7.173937e-09 | — | negative tail ~24 orders above float64 rounding — real structure |
| `R_STAT` | 0.052419 | < 0.10 PASS | passes r alone |
| `M_STAT` | 0.412071 | < 0.15 MARGINAL | fails — 41% of absolute eigenvalue mass negative |
| `K_EFF` | 40 | = `D_SWEEP_MAX` | — |
| `DROPOFF_INDEX` / `RATIO` | 2 / 2.4447 | — | runner-up (dim 8, log-ratio 0.4256) less than half the max (0.8939) |

**`GATE_VERDICT = FAIL`**, determined by `M_STAT` alone — `r` catches one large negative outlier,
`m` catches the long diffuse tail (5029 negatives, none dominant, 41% of mass). Independent
recompute from the persisted npz reproduces `r=0.052419 m=0.412071` exactly.

## Commits

1. §6.0-§6.3 (tracer) — `3401c0c`.
2. §6.4 — `108486e`.

## Artifacts

`mds_eigenspectrum_43cf438bc944c509.npz`: `eigvals_all` (10,000 float64),
`eigvals_top`/`eigvecs_top`/`mds_coords` (top-40), `geo_pairs_r2`/`geo_pairs_r2_check` (200,000
each), `n_positive`, `k_eff`, `spectrum_seconds`; `.meta.json` sidecar.

## Decisions / Deviations

- No threshold adjusted after seeing real r/m. Eigensolve `spectrum_seconds=103.61s` fresh; warm
  re-run 68s vs ~170s.
- `mds_coords = eigvecs_top * sqrt(eigvals_top)` literal, no `np.clip` — top-40 all positive.
- [Rule 1 bug] `np.asarray` on the read-only memmap returned a view; `D2 **= 2` raised
  `ValueError: output array is read-only`. Fixed with `np.array(..., copy=True)` (`3401c0c`).

## Next Phase Readiness

Spectrum cached for 02-02 (residual curves/elbow) and 02-03 (verdict artifact). Phase 1's
`! grep -q 'eigvalsh'` negative check now fails by design (SPEC-01 requires eigvalsh) — the
documented T-01-09 phase-boundary crossing, not a regression. `pyproject.toml`/`src/effdim/`/
`notebooks/pu_manifold/` byte-identical to pre-plan state.

## Self-Check: PASSED

FOUND: notebook, npz, meta.json, commits `3401c0c`, `108486e`.

---
*Phase: 02-eigenspectrum-audit-validity-gate* · *Completed: 2026-07-31*
