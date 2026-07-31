---
phase: 02-eigenspectrum-audit-validity-gate
plan: 01
subsystem: data
tags: [scipy, numpy, sklearn, isomap, classical-mds, eigendecomposition, jupyter]

# Dependency graph
requires:
  - phase: 01-data-loading-manifold-reconstruction
    provides: "notebooks/.cache/isomap_43cf438bc944c509.joblib (frozen k*=15 Isomap fit, dist_matrix_/embedding_/nbrs_/kernel_pca_) and phase1_handoff_43cf438bc944c509.json"
provides:
  - "The full 10,000-value classical-MDS eigenspectrum (§6.0-§6.2), computed by hand double-centring dist_matrix_, cross-checked against sklearn's own leading 18 eigenvalues"
  - "The r/m negativity statistics and PASS/MARGINAL/FAIL gate verdict (§6.3), with a synthetic-boundary-tested classifier"
  - "The leading-spectrum table, steep-dropoff location, and two diagnostic figures (§6.4)"
  - "mds_eigenspectrum_43cf438bc944c509.npz — cached spectrum artifact plan 02-02/02-03 read"
affects: [02-02-residual-elbow-freeze-d, 02-03-verdict-artifact-enforcement, phase-3-decoder-curvature]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Split eigensolve: scipy.linalg.eigvalsh (values-only, full spectrum) + scipy.linalg.eigh(subset_by_index=...) (top-K eigenvectors), avoiding a full 10,000x10,000 eigenvector materialization"
    - "In-place mean-form double-centring, verified equal to the literal J-form on two independent 50x50 inputs (metric + non-metric) via np.testing.assert_allclose"
    - "Cell-index self-assertion (notebook reads its own JSON) proving pre-registered constants execute before the cells that consume them"

key-files:
  created:
    - "notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz (gitignored, 9.23 MiB)"
    - "notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.meta.json (gitignored)"
  modified:
    - "notebooks/01_manifold_and_gate.ipynb (90 -> 102 cells; appended §6.0-§6.4)"

key-decisions:
  - "Real measured gate outcome: R_STAT=0.052419 (PASSes r<0.10) but M_STAT=0.412071 (fails even the m<0.15 MARGINAL bound) -> GATE_VERDICT=FAIL. This is a legitimate, complete phase outcome per the plan's hard-gate design, not an error to work around."
  - "Rule 1 auto-fix: np.asarray(dist_matrix_, dtype=float64) on a read-only float64 memmap returns a view (no copy) since dtype already matches, so the in-place D2 **= 2 raised 'output array is read-only'. Fixed with np.array(..., dtype=np.float64, copy=True)."

requirements-completed: [SPEC-01, SPEC-02, SPEC-03]

coverage:
  - id: D1
    description: "Full 10,000-value classical-MDS eigenspectrum computed by hand double-centring, length mechanically asserted, cross-checked against sklearn's truncated kernel_pca_.eigenvalues_ at rtol=1e-8 (worst diff 8.5e-15)"
    requirement: "SPEC-01"
    verification:
      - kind: other
        ref: "jupyter nbconvert --execute --inplace notebooks/01_manifold_and_gate.ipynb; §6.2 executed cell output shows EIGVALS_ALL.shape=(10000,), dtype=float64, and the 18-row sklearn cross-check table"
        status: pass
    human_judgment: false
  - id: D2
    description: "Leading spectrum reported as large-and-positive with a stated steep-dropoff criterion (largest log-ratio gap), explicitly distinguished in prose from the SPEC-04 elbow"
    requirement: "SPEC-02"
    verification:
      - kind: other
        ref: "§6.4 executed cell output: top-20 table all strictly positive, DROPOFF_INDEX=2/DROPOFF_RATIO=2.4447, two rendered figures (image/png) present"
        status: pass
    human_judgment: false
  - id: D3
    description: "r/m negativity statistics against pre-registered thresholds with the worse-of-two verdict rule; classifier boundaries asserted on nine synthetic cases (all strict less-than) before applying to the real spectrum"
    requirement: "SPEC-03"
    verification:
      - kind: other
        ref: "§6.3 executed cell output: all nine synthetic boundary cases pass; independent recompute from the persisted npz (outside the notebook) reproduces R_STAT=0.052419, M_STAT=0.412071 exactly"
        status: pass
    human_judgment: false

duration: ~20min (across two sessions; resumed after an account session limit cut off context-gathering before any file was written)
completed: 2026-07-31
status: complete
---

# Phase 2 Plan 1: Eigenspectrum Audit Summary

**Full 10,000-eigenvalue classical-MDS spectrum computed by hand double-centring of the memory-mapped Isomap geodesic matrix, cross-checked against sklearn to rtol=1e-8, yielding a real measured GATE_VERDICT=FAIL (R_STAT=0.052, M_STAT=0.412) on the k*=15 fit**

## Performance

- **Duration:** ~20 min of active execution this session (plus prior context-gathering that was cut off with no files written)
- **Completed:** 2026-07-31
- **Tasks:** 2/2 completed
- **Files modified:** 1 (notebook), 2 cache artifacts created (gitignored)

## Accomplishments

- **§6.0** pre-registers every gate constant (`R_MAX_PASS=0.10`, `M_MAX_PASS=0.05`,
  `R_MAX_MARGINAL=0.25`, `M_MAX_MARGINAL=0.15`, `D_SWEEP_MAX=40`, `R2_PAIR_COUNT=200_000`,
  `EQUIV_N=50`, `EQUIV_TOL=1e-12`, `SYMMETRY_RTOL=1e-10`, `ELBOW_TIE_BREAK="lower"`) in one
  cell, guarded by a cell-index self-assertion (`GATE_PREREG` index 90 < `SPECTRUM_COMPUTE`
  index 93) that reads the notebook's own JSON and halts if the ordering is ever violated.
- **§6.1** releases the Phase 1 `isomap_kstar`/`isomap_kstar_reloaded` bindings on purpose
  (peak RSS printed before/after), memory-maps `dist_matrix_` (`mmap_mode="r"`), measures
  symmetry chunk-wise (measured max deviation `1.421e-14`, bound `2.132e-09`) before any
  symmetric eigensolver reads it, double-centres in place in the mean form, and demonstrates
  the D-09 equivalence guard — mean-form vs. literal `-0.5 * J D^2 J` — agreeing to
  `rtol=atol=1e-12` on two independent 50x50 inputs (a genuine metric one and a
  non-metric symmetrised-random one).
- **§6.2** runs the split eigensolve — `scipy.linalg.eigvalsh` for all 10,000 values,
  `scipy.linalg.eigh(subset_by_index=...)` for the top-40 eigenpairs — caches the result
  through `pu_manifold.npz_cache` under `mds_eigenspectrum_43cf438bc944c509.npz` (9.23 MiB,
  under the 10 MB budget), asserts `EIGVALS_ALL.shape == (10_000,)` (the mechanical proof no
  truncated array could produce that length) and float64 dtype, and cross-checks the top 18
  hand-rolled eigenvalues against sklearn's `kernel_pca_.eigenvalues_` at `rtol=1e-8`
  (worst measured relative difference `8.532e-15`).
- **§6.3** defines `_gate_classify` reading its thresholds from §6.0 only, asserts all nine
  synthetic boundary cases (including the three named in the plan:
  `(0.10, 0.0) -> MARGINAL`, `(0.25, 0.0) -> FAIL`, `(0.05, 0.20) -> FAIL`) before touching
  real data, then computes and prints the real statistics.
- **§6.4** reports the top-20 eigenvalues with per-value and cumulative positive-mass share
  (all strictly positive, asserted), locates the steep dropoff (`DROPOFF_INDEX=2`,
  `DROPOFF_RATIO=2.4447`), and renders two figures: the leading spectrum on a log scale with
  the dropoff and Phase 1's `d=18` marked, and a two-panel full-spectrum-plus-negative-tail
  figure with `LAMBDA_MIN_NEG` and `-NOISE_FLOOR` annotated.
- The entire notebook (102 cells) was executed end-to-end twice via
  `jupyter nbconvert --to notebook --execute --inplace` (once per task) with zero error
  cells and every code cell carrying a non-null execution count; the committed outputs are
  real, not placeholders.

## The measured gate outcome

| Statistic | Value | PASS bound | MARGINAL bound | Reading |
|---|---|---|---|---|
| `N_POSITIVE` | 4971 | — | — | fewer than half the 10,000 eigenvalues are positive |
| count strictly negative | 5029 | — | — | — |
| `LAMBDA_MAX_POS` | 3.230854e+03 | — | — | — |
| `LAMBDA_MIN_NEG` | -1.693588e+02 | — | — | — |
| `NOISE_FLOOR` | 7.173937e-09 | — | — | `abs(LAMBDA_MIN_NEG)` is **~24 orders of magnitude above** the float64 rounding floor — the negative tail is real non-Euclidean structure, not rounding noise |
| `R_STAT` | 0.052419 | < 0.10 | < 0.25 | **PASSes** the r threshold alone |
| `M_STAT` | 0.412071 | < 0.05 | < 0.15 | **FAILs** even the loose MARGINAL bound — 41% of total absolute eigenvalue mass is negative |
| `K_EFF` | 40 | = `D_SWEEP_MAX` | — | did not fall short of the sweep ceiling |
| `DROPOFF_INDEX` / `DROPOFF_RATIO` | 2 / 2.4447 | — | — | decisive: the runner-up drop (after dim 8, log-ratio 0.4256) is less than half the size of the maximum (log-ratio 0.8939) |

**`GATE_VERDICT = FAIL`**, determined entirely by `M_STAT` — `r` alone would have read this
spectrum as clean, which is exactly the blind spot D-01 designed `m` to close. This is the
worked example of `m`'s purpose, not a surprise: `r` catches one large negative outlier;
`m` catches a long diffuse negative tail, and this fit has the latter (5029 negative
eigenvalues, none individually dominant, collectively carrying 41% of the total mass).

**Independent recompute (outside the notebook, from the persisted npz alone) reproduces
both statistics exactly:** `r=0.052419 m=0.412071`, matching the notebook's own printed
values, confirming the artifact and the printed statistics have not diverged.

This FAIL is a real, complete outcome per the phase's hard-gate design (ROADMAP.md
"Hard gate" note) — plan 02-03 writes it into `gate_verdict_{fit_key}.json` with its
enumerated remediation options; it is not something this plan works around.

## Task Commits

Each task was committed atomically:

1. **Task 1: Sections 6.0-6.3 — pre-registered constants through the verdict rule (tracer)**
   - `3401c0c` (feat) — 998 insertions, 138 deletions in `notebooks/01_manifold_and_gate.ipynb`
2. **Task 2: Section 6.4 — leading spectrum, steep dropoff, two figures**
   - `108486e` (feat) — 397 insertions, 177 deletions in `notebooks/01_manifold_and_gate.ipynb`

**Plan metadata:** committed separately after this SUMMARY (docs commit).

_Note: Task 1's `type="tracer"` feedback gate was satisfied by its own comprehensive
automated `<verify>` block (all 5 automated checks passed against the real executed
notebook and the real cached npz) before Task 2 began — logged here as the equivalent of
"Tracer verified end-to-end — expanding" since this run proceeded autonomously per the
orchestrator's directive to execute the full plan without pausing._

## Files Created/Modified

- `notebooks/01_manifold_and_gate.ipynb` — grown from 90 to 102 cells; `§6.0`-`§6.4`
  appended with real executed outputs from two full `jupyter nbconvert --execute --inplace`
  runs. Zero error cells; every new code cell carries a non-null execution count.
- `notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz` (gitignored, 9.23 MiB) —
  `eigvals_all` (10,000 float64), `eigvals_top`/`eigvecs_top`/`mds_coords` (top-40),
  `geo_pairs_r2`/`geo_pairs_r2_check` (200,000 each), `n_positive`, `k_eff`,
  `spectrum_seconds`
- `notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.meta.json` (gitignored) — the cfg
  manifest (`fit_key`, `d_sweep_max`, `r2_pair_count`, `r2_pair_seed`,
  `r2_pair_seed_check`, `scipy_version`, `numpy_version`)

## Decisions Made

- Followed the plan's pre-registered literals exactly: no threshold was adjusted after
  seeing the real `r`/`m` values (which would have violated the must_haves prohibition on
  post-hoc threshold revision).
- Eigensolve wall-clock: `spectrum_seconds = 103.61s` (fresh, on this 20-core machine with
  OpenBLAS); Task 2's re-run hit the npz cache and skipped it entirely (68s total notebook
  re-run vs. ~170s for Task 1's fresh run).
- `mds_coords = eigvecs_top * np.sqrt(eigvals_top)` was implemented literally per the plan
  text (no defensive `np.clip` before the sqrt) — the top-40 eigenvalues are all positive
  in practice (confirmed by §6.4's strictly-positive assertion on the top 20), so clipping
  was unnecessary and would have silently masked a real finding had it been needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `np.asarray` on a read-only memmap returned a view, not a copy**
- **Found during:** Task 1, first execution attempt of §6.1
- **Issue:** `_dist_matrix` is a `numpy.memmap` opened with `mmap_mode="r"` (read-only) and
  is already `float64`. `np.asarray(_dist_matrix, dtype=np.float64)` therefore returns the
  memmap itself (no copy is made when the requested dtype already matches), so the
  subsequent in-place `D2 **= 2` raised `ValueError: output array is read-only`.
- **Fix:** Changed to `np.array(_dist_matrix, dtype=np.float64, copy=True)`, which forces
  an actual resident copy regardless of dtype match, documented inline with a comment
  explaining why `asarray` was wrong here.
- **Files modified:** `notebooks/01_manifold_and_gate.ipynb` (§6.1 code cell)
- **Verification:** Full notebook re-execution completed with zero error cells; the
  squared-then-centred array (`D2`/`B_CENTERED`) is confirmed resident and mutable by the
  successful in-place centring and the equivalence-guard assertions that follow it.
- **Committed in:** `3401c0c` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug)
**Impact on plan:** Necessary for the tracer to run at all; no scope creep. The plan's own
text already specified `joblib.load(..., mmap_mode="r")`, so this fix is an implementation
detail of realizing that spec correctly, not a design change.

## Issues Encountered

None beyond the deviation above. `.venv` already had `jupyter`/`nbconvert`/`nbformat`
installed (not listed in `requirements-notebooks.txt` but present from environment setup),
so no new dependency was installed and the Package Legitimacy Gate did not fire.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `mds_eigenspectrum_43cf438bc944c509.npz` and `EIGVALS_ALL`/`EIGVALS_TOP`/`EIGVECS_TOP`/
  `MDS_COORDS`/`N_POSITIVE`/`K_EFF` are all bound and cached for plan 02-02's residual
  curves and kneedle elbow to consume directly — no re-run of LAPACK needed on a warm
  Restart-and-Run-All (confirmed: Task 2's full re-run took 68s total, vs. ~170s including
  the fresh eigensolve in Task 1).
- **Phase 1's negative grep is now expected to fail, as designed.** `01-04-PLAN.md`'s
  verification block asserts `! grep -q 'eigvalsh' notebooks/01_manifold_and_gate.ipynb` —
  this notebook now contains `scipy.linalg.eigvalsh` in §6.2 by design (SPEC-01 requires
  it). This is the documented phase-boundary crossing T-01-09's mitigation was written to
  detect, not a regression. Re-running Phase 1's verification suite will correctly flag this
  one line; it should not be "fixed" by removing the Phase 2 call.
- **`GATE_VERDICT = FAIL` is the real, measured outcome for this fit.** Per the phase's hard
  gate (ROADMAP.md), this is itself a legitimate, complete, reportable milestone outcome.
  Plan 02-02 (elbow/frozen-d) and 02-03 (verdict artifact) still need to run to produce the
  complete audit trail and the self-contained `gate_verdict_{fit_key}.json` — but whoever
  reviews this milestone should be aware, from this plan onward, that the spectral gate on
  the current `k*=15` fit does not pass. The remediation options (re-fit at a different k,
  resample with a new seed, accept the documented FAIL) belong to plan 02-03's D-16 artifact
  and the eventual milestone-level decision, not to this plan.
- `pyproject.toml`, `src/effdim/`, and `notebooks/pu_manifold/` are confirmed byte-identical
  to their pre-plan state (`git diff --quiet` passed after both tasks).

## Self-Check: PASSED

- FOUND: `notebooks/01_manifold_and_gate.ipynb`
- FOUND: `notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz`
- FOUND: `notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.meta.json`
- FOUND: commit `3401c0c` (Task 1)
- FOUND: commit `108486e` (Task 2)

---
*Phase: 02-eigenspectrum-audit-validity-gate*
*Completed: 2026-07-31*
