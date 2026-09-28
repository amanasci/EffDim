---
phase: 01-data-loading-manifold-reconstruction
plan: 03
subsystem: data-loading
tags: [isomap, scikit-learn, scipy, connectivity, stability-sweep, notebooks]
requires:
  - {phase: "01-data-loading-manifold-reconstruction (plan 02)", provides: "N_COMPONENTS=18, D_PROVISIONAL=18, ANALYSIS_CFG, fit_key=80ce249fedcf55e0, real 10,000-row LS/HSC arrays"}
provides:
  - "01_manifold_and_gate.ipynb §4.0-4.3 -- all twelve pre-registered sweep constants (incl. D-10's three plateau thresholds, cell-index-asserted), the full six-k connectivity scan (ISO-01, K_SMALLEST_CONNECTED=5, SHORT_CIRCUIT_RISK=False), stage-2 fits at STAGE2_K=[5,10,15,30], STABILITY_TABLE, PLATEAU_RUNS (ISO-02)"
  - "notebooks/.cache/sweep_k{5,10,15,30}_*.npz -- slim (~1.07 MB each) per-k stage-2 fit records, GEO_PAIR_SEED=20260730"
  - "Confirmed k*=15 (Task 4 checkpoint, accept-candidate), SHORT_CIRCUIT_RISK=False -- both consumed directly by plan 04"
affects: [01-04]
tech-stack:
  added: []
  patterns:
    - "Pre-registration-by-cell-index: PLATEAU_THRESH's cell index asserted < STAGE2_SWEEP's, live + automated — structural, not disciplinary"
    - "Slim npz + in-loop sampling: dist_matrix_ (~763 MiB) never persisted; geo_pairs (100,000 samples) extracted in-loop, only the slim record cached"
    - "Independent per-fit connectivity re-verification on each fit's own model.nbrs_ graph (T-01-08)"
key-files:
  created: []
  modified: [notebooks/01_manifold_and_gate.ipynb]
key-decisions:
  - "Task 4 gate: accept-candidate. k*=15 confirmed after independent spot-check of the four sweep npz files (1,122,020 bytes each, correct 7-key field set, no dist_matrix_ leak). Pre-registered plateau rule (§4.0) applied exactly as written. Uneven STAGE2_K spacing ([5,10,15,30], gaps 5/5/15) disclosed before the decision, recorded as a known limitation, not acted on (reopening a frozen §4.0 parameter after seeing results is the garden-of-forking-paths failure pre-registration prevents)"
requirements-completed: [ISO-01, ISO-02]
coverage:
  - {id: D1, description: "§4.0: twelve sweep constants fixed before any fit; cell-index ordering asserted live + automated", requirement: "ISO-02", verification: [{kind: integration, ref: "§4.0 (committed 37f65cf)", status: pass}], human_judgment: false}
  - {id: D2, description: "§4.1: six-k connectivity scan, bounded D-11 auto-extend, SHORT_CIRCUIT_RISK flag — see CONNECTIVITY_SCAN table below", requirement: "ISO-01", verification: [{kind: integration, ref: "§4.1 (committed cde9ed9)", status: pass}], human_judgment: false}
  - {id: D3, description: "§4.2-4.3: full Isomap fits at STAGE2_K, each a slim npz (dist_matrix_ never persisted); three D-10 metrics per adjacent pair vs §4.0 thresholds, PLATEAU_RUNS computed; k* deliberately not bound here — see tables below", requirement: "ISO-02", verification: [{kind: integration, ref: "§4.2-4.3 (committed 0f7ae6b); npz spot-checked 1,122,020 bytes each, 7-key set", status: pass}], human_judgment: false}
  - {id: D4, description: "Task 4 checkpoint:decision (blocking) — candidate k*=15 presented with full evidence; accept-candidate selected", requirement: "ISO-02", verification: [{kind: other, ref: "coordinator: 'accept-candidate. k*=15 confirmed.' after independent spot-check", status: pass}], human_judgment: true, rationale: "k* selection is an explicitly human-confirmed, irreversible decision — not test-certifiable independent of confirmation."}
duration: ~25min active work
completed: 2026-07-30
status: complete
---

# Phase 1 Plan 3: n_neighbors Sweep -- Connectivity Scan, Stage-2 Fits, and Stability Table Summary

**All six candidate k values are connected at n=10,000 (no auto-extend, no short-circuit risk), and the pre-registered D-10 plateau criterion finds one real maximal run `[10, 15, 30]` (length 3) among the four stage-2 fits -- confirmed candidate `k* = 15`, human-approved via the Task 4 checkpoint after an independent artifact spot-check.**

## Performance / Accomplishments

~25 min active work (`37f65cf`/`cde9ed9`/`0f7ae6b` over ~10 min, plus real wall-clock of four
dense Isomap fits, then a Task 4 checkpoint pause). Completed 2026-07-30. 4/4 tasks. 1 file
modified, grown 59->74 cells. §4.0 constants fixed before any fit; §4.1 six-k scan, all connected;
§4.2-4.3 stage-2 fits and stability table (see tables below). Task 4: coordinator independently
spot-checked npz artifacts and execution counts, selected accept-candidate. Notebook re-executed
end-to-end three times against real data; 14/14 pytest pass; core untouched.

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

`CONNECTED_K = [5, 8, 10, 15, 20, 30]`. `K_SMALLEST_CONNECTED = 5`. `K_EXTENSIONS=(40,50)` not
entered. `SHORT_CIRCUIT_RISK = False` (`5 <= 30`, strict).

### STAGE2_K and per-k fit summary (§4.2)

`STAGE2_K = [5, 10, 15, 30]` (4 of 6 connected values, `STAGE2_MAX_FITS=4`).

| k | n_connected_components | fit_seconds | npz size (MiB) | eigenvalues_truncated[:5] |
|---|---|---|---|---|
| 5 | 1 | 62.71 | 1.070 | [5432.09, 3731.44, 1544.05, 1049.73, 666.27] |
| 10 | 1 | 69.54 | 1.070 | [3798.25, 2575.84, 1047.77, 723.45, 488.25] |
| 15 | 1 | 94.76 | 1.070 | [3230.85, 2149.13, 879.09, 586.71, 414.91] |
| 30 | 1 | 75.60 | 1.070 | [2528.07, 1698.27, 691.70, 458.85, 321.96] |

Each fit's own `model.nbrs_` graph independently re-verified `n_connected_components=1`. `dist_matrix_` (~763 MiB) never persisted for any swept k.

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

Widest run length = 3, reaches `MIN_PLATEAU_RUN=3`: **True**. Sole maximal run.

Task 4 checkpoint `accept-candidate`, `k* = 15`, `SHORT_CIRCUIT_RISK = False` — both consumed
directly by plan 04. Known limitation (disclosed, recorded, not acted on): `STAGE2_K` gaps 5/5/15
(k=8/k=20 never fit, `STAGE2_MAX_FITS=4`), so the plateau run is maximal in *index* space not *k*
space — carried forward for Phase 2, no rework (garden-of-forking-paths guard).

## Commits / Files / Next Phase Readiness

`37f65cf`/`cde9ed9`/`0f7ae6b` (Tasks 1-3); Task 4 checkpoint, no commit. `01_manifold_and_gate.ipynb`
grown 59->74 cells (27/27 with execution counts). Four `sweep_k{5,10,15,30}_*.npz` files, 1,122,020
bytes each with sidecar manifests, gitignored. No deviations, issues, or setup — plan executed
exactly as written, `<verify>` passed first execution. `k*=15`/`SHORT_CIRCUIT_RISK=False` are what
plan 04 sets `ANALYSIS_CFG["n_neighbors"]` from, fits/caches `isomap_{fit_key}.joblib` (~1 GB), and
writes the Phase 1->2 handoff from. No blockers; core untouched.

---
*Phase: 01-data-loading-manifold-reconstruction* · *Completed: 2026-07-30*

## Self-Check: PASSED

Notebook and all four npz files present with correct sizes/fields; all 3 task commits present in
`git log --oneline --all`; `git diff` on the notebook clean. No missing items.
</content>
