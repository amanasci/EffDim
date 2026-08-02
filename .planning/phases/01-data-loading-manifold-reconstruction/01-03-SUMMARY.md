---
phase: 01-data-loading-manifold-reconstruction
plan: 03
subsystem: data-loading
tags: [isomap, scikit-learn, scipy, connectivity, stability-sweep, notebooks]

# Dependency graph
requires:
  - phase: 01-data-loading-manifold-reconstruction (plan 02)
    provides: "N_COMPONENTS=18, D_PROVISIONAL=18, ANALYSIS_CFG, fit_key=80ce249fedcf55e0, real 10,000-row LS/HSC arrays"
provides:
  - "notebooks/01_manifold_and_gate.ipynb §4.0 -- all twelve pre-registered sweep constants, including the three D-10 plateau thresholds, fixed and cell-index-asserted before any stage-2 fit"
  - "notebooks/01_manifold_and_gate.ipynb §4.1 -- the full six-k connectivity scan (ISO-01), CONNECTED_K, K_SMALLEST_CONNECTED=5, SHORT_CIRCUIT_RISK=False"
  - "notebooks/01_manifold_and_gate.ipynb §4.2-4.3 -- stage-2 fits at STAGE2_K=[5,10,15,30], STABILITY_TABLE, PLATEAU_RUNS (ISO-02)"
  - "notebooks/.cache/sweep_k{5,10,15,30}_*.npz -- slim (~1.07 MB each) per-k stage-2 fit records"
  - "Confirmed k*=15 (Task 4 checkpoint, accept-candidate), SHORT_CIRCUIT_RISK=False -- both consumed directly by plan 04"
affects: [01-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pre-registration-by-cell-index: every sweep constant (including all three D-10 plateau thresholds) is fixed in a §4.0 cell, and a live notebook cell plus an automated verify script both assert PLATEAU_THRESH's cell index < STAGE2_SWEEP's cell index -- the garden-of-forking-paths guard is structural, not disciplinary."
    - "Slim npz + in-loop sampling: dist_matrix_ (~763 MiB) is never persisted for a swept k; geo_pairs (100,000 sampled distances) are extracted from dist_matrix_ while it is still in memory inside the fit closure, then only the ~1 MB slim record is cached."
    - "Independent per-fit connectivity re-verification: each stage-2 fit re-checks connected_components on its own fitted model.nbrs_ graph, so the stage-1 scan result and what the fitted estimator actually used cannot silently diverge (T-01-08)."

key-files:
  created: []
  modified:
    - notebooks/01_manifold_and_gate.ipynb

key-decisions:
  - "Task 4 gate (checkpoint:decision, gate=blocking): accept-candidate selected. k*=15 confirmed by the coordinator/user after independent spot-check of the four sweep npz files (1,122,020 bytes each, correct 7-key field set, no dist_matrix_ leak) and full disclosure of the uneven-STAGE2_K-spacing caveat (see Known Limitations below). The pre-registered plateau rule (§4.0) was applied exactly as written -- no post-hoc threshold or range adjustment."
  - "Uneven STAGE2_K spacing ([5,10,15,30], gaps 5/5/15) was disclosed to the user before the Task 4 decision and is recorded here as a documented limitation of the plateau evidence, per explicit instruction: it must be recorded, not acted upon. Reopening a frozen §4.0 sweep parameter (STAGE2_MAX_FITS, or refitting the skipped connected values k=8/k=20) after seeing sweep results is exactly the garden-of-forking-paths failure §4.0's pre-registration exists to prevent, so the caveat stands as a known gap in the evidence rather than a trigger for rework."

requirements-completed: [ISO-01, ISO-02]

coverage:
  - id: D1
    description: "§4.0: all twelve sweep constants (SWEEP_K_RANGE, K_EXTENSIONS, K_CEILING, K_WARN_ABOVE, PLATEAU_THRESH with all 3 threshold keys/values as literals, GEO_PAIR_COUNT/SEED, MIN_PLATEAU_RUN, STAGE2_MAX_FITS, PLATEAU_TIE_BREAK, GEO_PAIR_ROWS/COLS) fixed and printed in a cell that executes before any Isomap fit; cell-index ordering (PLATEAU_THRESH < STAGE2_SWEEP) asserted both as a live notebook cell and as an automated task-gate script"
    requirement: "ISO-02"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb §4.0 (executed cell outputs, committed 37f65cf); re-run via jupyter nbconvert --to notebook --execute --inplace notebooks/01_manifold_and_gate.ipynb"
        status: pass
    human_judgment: false
  - id: D2
    description: "§4.1: kneighbors_graph + connected_components scan across all six SWEEP_K_RANGE values (no Isomap fit), component count and size distribution (np.bincount) tabulated for every k, bounded D-11 auto-extend as explicit for/else with three-remediation-option halt message, SHORT_CIRCUIT_RISK strict-comparison flag with warnings.warn + boxed print"
    requirement: "ISO-01"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb §4.1 (executed cell outputs, committed cde9ed9); real result: all six base-range k connected (n_components=1 for k in {5,8,10,15,20,30}), K_SMALLEST_CONNECTED=5, SHORT_CIRCUIT_RISK=False, auto-extend ladder not entered"
        status: pass
    human_judgment: false
  - id: D3
    description: "§4.2: full Isomap fits (eigen_solver=\"dense\") at STAGE2_K=[5,10,15,30] (evenly-spaced-with-endpoints selection from all 6 connected values, STAGE2_MAX_FITS=4), each persisted as a slim ~1.07 MB sweep_k{K}_{key}.npz (embedding, eigenvalues_truncated, n_connected_components, fit_seconds, geo_pairs, geo_pair_count, geo_pair_seed -- dist_matrix_ never persisted), independent per-fit connectivity re-verification"
    requirement: "ISO-02"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb §4.2 (executed cell outputs, committed 0f7ae6b); npz files independently spot-checked post-commit: 1,122,020 bytes each, 7-key field set confirmed, no dist_matrix_ present"
        status: pass
    human_judgment: false
  - id: D4
    description: "§4.3: three D-10 metrics (Procrustes disparity, self-normalized relative eigenvalue change, geodesic Spearman on the fixed seeded pair sample) computed for every adjacent STAGE2_K pair against the §4.0 thresholds, geo_pair_count/geo_pair_seed staleness assertion before each Spearman comparison, PLATEAU_RUNS (maximal all-three-passing runs) computed and printed; k* deliberately not bound or printed in this section"
    requirement: "ISO-02"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb §4.3 (executed cell outputs, committed 0f7ae6b); real result: (5,10) fails on Procrustes only (0.1316 > 0.10), (10,15) and (15,30) pass all three; PLATEAU_RUNS=[{k_values:[10,15,30], length:3}], reaches MIN_PLATEAU_RUN=3"
        status: pass
    human_judgment: false
  - id: D5
    description: "Task 4 checkpoint:decision (gate=blocking) -- candidate k*=15 (centre of the widest all-three-passing run [10,15,30]) presented to a human with the full CONNECTIVITY_SCAN, STABILITY_TABLE, PLATEAU_RUNS, and the uneven-spacing caveat; accept-candidate selected"
    requirement: "ISO-02"
    verification:
      - kind: other
        ref: "Coordinator/user response: 'Gate answered: accept-candidate. k* = 15 is confirmed. Proceed.' Independent spot-check of the four sweep npz files and execution-count completeness (27/27 code cells) performed and reported before the decision was made."
        status: pass
    human_judgment: true
    rationale: "k* selection is explicitly a human-confirmed, irreversible decision per this plan's own Task 4 design (D-10 locks the criterion; the resulting value is what the checkpoint confirms) -- not something a test can certify as 'correct' independent of that confirmation."

# Metrics
duration: ~25min active work (3 task commits across ~10 min, then a Task 4 checkpoint pause awaiting human confirmation, then this wrap-up)
completed: 2026-07-30
status: complete
---

# Phase 1 Plan 3: n_neighbors Sweep -- Connectivity Scan, Stage-2 Fits, and Stability Table Summary

**All six candidate k values are connected at n=10,000 (no auto-extend, no short-circuit risk), and the pre-registered D-10 plateau criterion finds one real maximal run `[10, 15, 30]` (length 3) among the four stage-2 fits -- confirmed candidate `k* = 15`, human-approved via the Task 4 checkpoint after an independent artifact spot-check.**

## Performance

- **Duration:** ~25 min active work: three task commits (`37f65cf`, `cde9ed9`, `0f7ae6b`) spanning ~10 minutes plus roughly the real wall-clock time of four dense Isomap fits at n=10,000 (fit_seconds: 62.71s / 69.54s / 94.76s / 75.60s for k=5/10/15/30 -- observed, not published; STACK.md flagged the absence of a benchmark at this n as a known gap, and these are the measured numbers filling it), then a Task 4 `checkpoint:decision` pause awaiting human confirmation, then this closeout.
- **Started:** 2026-07-30 (continuation from plan 02)
- **Completed:** 2026-07-30 (checkpoint answered same session)
- **Tasks:** 4/4 complete (3 code tasks + 1 human-confirmed decision gate)
- **Files modified:** 1 (`notebooks/01_manifold_and_gate.ipynb`, grown from 59 to 74 cells)

## Accomplishments

- **§4.0 (Task 1):** Added all twelve pre-registered sweep constants -- `SWEEP_K_RANGE=(5,8,10,15,20,30)`, `K_EXTENSIONS=(40,50)`, `K_CEILING=50` (inclusive), `K_WARN_ABOVE=30` (strict), `PLATEAU_THRESH={"procrustes_disparity_max":0.10,"eig_rel_change_max":0.15,"geodesic_spearman_min":0.95}`, `GEO_PAIR_COUNT=100_000`, `GEO_PAIR_SEED=20260730`, `MIN_PLATEAU_RUN=3`, `STAGE2_MAX_FITS=4`, `PLATEAU_TIE_BREAK="lower"`, and the seeded `GEO_PAIR_ROWS`/`GEO_PAIR_COLS` (100,000 off-diagonal index pairs) -- fixed and printed in a cell that runs before any stage-2 fit exists, with full threshold-by-threshold justification in the accompanying markdown.
- **§4.1 (Task 2):** Ran the full six-k connectivity scan (`kneighbors_graph` + `connected_components`, no Isomap fit). **Real result: every base-range k (5, 8, 10, 15, 20, 30) yields exactly one connected component** at n=10,000 -- the D-11 auto-extend ladder was never entered. `K_SMALLEST_CONNECTED=5`, `SHORT_CIRCUIT_RISK=False`.
- **§4.2-4.3 (Task 3):** Selected `STAGE2_K=[5,10,15,30]` (evenly-spaced-with-endpoints from all 6 connected values), ran four full `Isomap(eigen_solver="dense")` fits, and computed the three-metric adjacent-k stability table. **Real result:** the (5,10) pair fails on Procrustes disparity alone (0.1316 > 0.10) while passing the other two metrics; (10,15) and (15,30) pass all three. `PLATEAU_RUNS` finds exactly one maximal run, `[10,15,30]` (length 3), which reaches `MIN_PLATEAU_RUN=3`.
- **Task 4 (checkpoint:decision, gate=blocking):** Presented the full evidence to the coordinator/user, who independently spot-checked the four `sweep_k*.npz` artifacts (1,122,020 bytes each, correct 7-key field set, no `dist_matrix_` leak) and confirmed 27/27 notebook code cells carry real execution counts, then selected **`accept-candidate`**: **`k* = 15`** is confirmed.
- Re-executed the entire notebook end-to-end via `jupyter nbconvert --execute --inplace` three times (once per task) against the real cached 10,000-row analysis subsample; all 14 `pu_manifold` pytest tests pass throughout; `pyproject.toml`/`src/effdim/` verified byte-identical to their pre-plan state after every task.

## Full Evidence (verbatim, for plan 04 and Phase 2)

### CONNECTIVITY_SCAN (§4.1, all six base-range k)

| k | n_components | largest_component | smallest_component | scan_seconds |
|---|---|---|---|---|
| 5 | 1 | 10000 | 10000 | 3.013 |
| 8 | 1 | 10000 | 10000 | 3.150 |
| 10 | 1 | 10000 | 10000 | 3.336 |
| 15 | 1 | 10000 | 10000 | 3.143 |
| 20 | 1 | 10000 | 10000 | 3.115 |
| 30 | 1 | 10000 | 10000 | 3.124 |

`CONNECTED_K = [5, 8, 10, 15, 20, 30]` (all six). `K_SMALLEST_CONNECTED = 5`. Auto-extend ladder (`K_EXTENSIONS = (40, 50)`) **not entered** -- the base range was fully connected. `SHORT_CIRCUIT_RISK = False` (`K_SMALLEST_CONNECTED=5 <= K_WARN_ABOVE=30`, strict comparison).

### STAGE2_K and per-k fit summary (§4.2)

`STAGE2_K = [5, 10, 15, 30]` (4 of 6 connected values, `STAGE2_MAX_FITS=4`).

| k | n_connected_components | fit_seconds | npz size (MiB) | eigenvalues_truncated[:5] |
|---|---|---|---|---|
| 5 | 1 | 62.71 | 1.070 | [5432.09, 3731.44, 1544.05, 1049.73, 666.27] |
| 10 | 1 | 69.54 | 1.070 | [3798.25, 2575.84, 1047.77, 723.45, 488.25] |
| 15 | 1 | 94.76 | 1.070 | [3230.85, 2149.13, 879.09, 586.71, 414.91] |
| 30 | 1 | 75.60 | 1.070 | [2528.07, 1698.27, 691.70, 458.85, 321.96] |

Each fit's own `model.nbrs_` graph independently re-verified `n_connected_components=1`, matching §4.1. `dist_matrix_` (~763 MiB) was never persisted for any swept k -- only the ~1 MB slim record.

### STABILITY_TABLE (§4.3, all three adjacent pairs)

| k1 | k2 | procrustes_disparity | eig_rel_change | geodesic_spearman | procrustes_pass (≤0.10) | eig_pass (≤0.15) | spearman_pass (≥0.95) | all_three_pass |
|---|---|---|---|---|---|---|---|---|
| 5 | 10 | 0.1316 | 0.0100 | 0.9676 | N | Y | Y | **fail** |
| 10 | 15 | 0.0699 | 0.0130 | 0.9815 | Y | Y | Y | **PASS** |
| 15 | 30 | 0.0803 | 0.0066 | 0.9793 | Y | Y | Y | **PASS** |

### PLATEAU_RUNS

```
[{"k_values": [10, 15, 30], "length": 3}]
```

Widest run length = 3, reaches `MIN_PLATEAU_RUN=3`: **True**. This is the sole maximal run.

### Task 4 checkpoint outcome

- **Decision:** `accept-candidate`
- **Confirmed `k* = 15`** (centre of the widest run `[10, 15, 30]`; odd length 3, no `PLATEAU_TIE_BREAK` needed since there is a unique middle element)
- **`SHORT_CIRCUIT_RISK = False`** -- carries forward unchanged to Phase 2's gate as a clean (non-elevated) flag
- Both values -- **`k* = 15`** and **`SHORT_CIRCUIT_RISK = False`** -- are what plan 04 reads directly to freeze the cached Isomap fit and build the Phase 1 to Phase 2 handoff artifact.

## Known Limitations

**Uneven `STAGE2_K` spacing (disclosed to the user before the Task 4 decision, recorded per explicit instruction -- not acted upon).**

`STAGE2_K = [5, 10, 15, 30]` is unevenly spaced (gaps 5, 5, 15). Connected values k=8 and k=20 were never fit — `STAGE2_MAX_FITS=4` dropped them (evenly-spaced-with-endpoints selection over the 6 connected base-range values). The plateau run `[10, 15, 30]` is therefore maximal in *index* space, not *k* space — real but coarse-grained. The user was shown this and chose to apply the pre-registered rule unchanged rather than fit the two skipped values: reopening a frozen sweep parameter after seeing results is exactly the garden-of-forking-paths failure §4.0's pre-registration exists to prevent. Documented gap in evidence granularity, not a computation defect — carried forward for Phase 2, not triggering rework here.

## Task Commits

Each code task was committed atomically; Task 4 is a checkpoint-only gate (no commit of its own beyond this closeout):

1. **Task 1: §4.0 -- pre-registered sweep constants** -- `37f65cf` (feat)
2. **Task 2: §4.1 -- connectivity scan across all six k** -- `cde9ed9` (feat)
3. **Task 3: §4.2-4.3 -- stage-2 fits and adjacent-k stability table** -- `0f7ae6b` (feat)
4. **Task 4: checkpoint:decision (gate=blocking)** -- no commit (gate only); **`accept-candidate`, `k*=15`** confirmed by the coordinator/user in this session.

**Plan metadata:** committed separately after this Summary (see final commit below).

## Files Created/Modified

- `notebooks/01_manifold_and_gate.ipynb` -- grown from 59 to 74 cells; `§4` header, `§4.0`-`§4.3` appended; every cell carries real, committed execution outputs from an actual `Restart and Run All` (via `jupyter nbconvert --execute --inplace`) against the live cached dataset. All 27 code cells carry non-null execution counts.
- `notebooks/.cache/sweep_k5_9db36086f7472619.npz`, `sweep_k10_9fbaf46e3570c8b7.npz`, `sweep_k15_43cf438bc944c509.npz`, `sweep_k30_860e4b66f08af831.npz` -- 1,122,020 bytes each, with sidecar `.meta.json` manifests. Gitignored (`notebooks/.cache/` already covered).

## Decisions Made

- **Task 4 (`accept-candidate`, confirmed `k*=15`):** the pre-registered D-10 plateau rule applied exactly as written to the real sweep result. No threshold, range, or selection-rule adjustment was made after seeing the data.
- **Uneven-spacing caveat:** documented as a known limitation (see above), not acted upon, per explicit coordinator instruction and consistent with the plan's own prohibition against post-hoc reopening of pre-registered constants.

## Deviations from Plan

None -- plan executed exactly as written. All `<verify>` automated checks passed on first execution for all three code tasks (cell-ordering assertions, token-presence checks, `eigvalsh`-absence, `pyproject.toml`/`src/effdim/` untouched, 14/14 pytest). The Task 4 checkpoint resolved with the pre-registered rule's own candidate, requiring no remediation branch (no auto-extend was entered, no short-circuit risk, a valid plateau existed).

## Issues Encountered

None. `git diff` on the notebook was checked clean (no stray IDE/`ipykernel` corruption) both before this closeout and is reconfirmed by the Self-Check below.

## User Setup Required

None -- no external service configuration required. All work used the already-warm `notebooks/.cache/` artifacts from plans 01-02; no new network access was needed.

## Next Phase Readiness

- **`k* = 15`** and **`SHORT_CIRCUIT_RISK = False`** are frozen findings plan 04 consumes directly to build `ANALYSIS_CFG["n_neighbors"] = 15`, fit and cache the full `isomap_{fit_key}.joblib` (~1 GB), and write the Phase 1 to Phase 2 handoff artifact.
- The uneven `STAGE2_K` spacing limitation (above) should be carried into Phase 2's gate-weighing narrative as context for how coarse-grained the plateau evidence is, even though `SHORT_CIRCUIT_RISK` itself is clean.
- `STAGE2_RECORDS`, `STABILITY_TABLE`, and `PLATEAU_RUNS` remain live in-notebook (and reloadable from the four `sweep_k*.npz` files) for any future re-derivation without re-fitting.
- No blockers. `pyproject.toml` and `src/effdim/` remain byte-identical to their pre-plan state (verified via `git diff --quiet` after every task).

---
*Phase: 01-data-loading-manifold-reconstruction*
*Completed: 2026-07-30*

## Self-Check: PASSED

- `notebooks/01_manifold_and_gate.ipynb` (74 cells, 27/27 code cells with execution counts) verified present and committed.
- All four `sweep_k*.npz` files verified present at `notebooks/.cache/` with the correct 7-key field set and 1,122,020-byte size (no `dist_matrix_` leak).
- All 3 task commits (`37f65cf`, `cde9ed9`, `0f7ae6b`) verified present in `git log --oneline --all`.
- `git diff` on the notebook re-confirmed clean immediately before this closeout (no stray `ipykernel`/IDE corruption).
- No missing items.
