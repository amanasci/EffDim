---
phase: 01-data-loading-manifold-reconstruction
plan: 01
subsystem: data-loading
tags: [huggingface-datasets, sklearn, isomap, joblib, numpy, faiss, pytest, notebooks]

requires: []
provides:
  - "pu_manifold/cache.py -- config-hash-keyed npz/joblib/json cache, sidecar-manifest verification, CACHE_DIR containment guard"
  - "pu_manifold/subsample.py -- seeded, row-alignment-safe subsampling of legacysurvey_dinov3_vitb16 with a strict statistical alignment assertion"
  - "pu_manifold/curvature.py, mknn.py -- Phase 3/4 stub surfaces (NotImplementedError, real signatures)"
  - "01_manifold_and_gate.ipynb §0-1 -- env/repro header, dep install, permanent smoke-config self-test, executed end-to-end with real HF data"
  - "requirements-notebooks.txt -- fully self-provisioning pinned notebook venv"
affects: [01-02, 01-03, 01-04]

tech-stack:
  added: [torch==2.13.0+cpu, datasets==5.0.1, matplotlib==3.11.1, "numpy/scipy/scikit-learn/faiss-cpu/joblib/pytest (pinned in requirements-notebooks.txt, per user-directed deviation)"]
  patterns: ["config-hash cache: sha256(json.dumps(cfg, sort_keys=True))[:16] + sidecar .meta.json manifest asserted on every load", "two-tier cache key: narrow subsample_key vs full D-14 fit_key", "row-alignment invariant: one np.sort(rng.choice(replace=False)) index array, one indexing pass, structural + permutation-z-score assertion (strict >)", "notebook-scoped pu_manifold/ package, never installed, lazy import of heavy deps"]

key-files:
  created: [notebooks/pu_manifold/cache.py, notebooks/pu_manifold/subsample.py, notebooks/pu_manifold/curvature.py, notebooks/pu_manifold/mknn.py, notebooks/pu_manifold/__init__.py, notebooks/pu_manifold/tests/test_pu_manifold.py, notebooks/requirements-notebooks.txt, notebooks/01_manifold_and_gate.ipynb]
  modified: []

key-decisions:
  - "Task 1 gate: approved. torch==2.13.0+cpu, datasets==5.0.1, matplotlib==3.11.1 confirmed legitimate on PyPI; numpy/scipy/scikit-learn/faiss-cpu already-core, excluded from re-pinning at that time"
  - "Task 2 gate: normalized-only. subsample_{seed}_{key}.npz stores only L2-normalized hsc/legacysurvey plus hsc_norms/ls_norms; no raw 768-d arrays. One-way: raw vectors would need re-streaming the 553 MiB parquet"
  - "User-directed deviation (post-tracer): requirements-notebooks.txt reversed the exclusion policy, also pins numpy==2.5.1, scipy==1.18.0, scikit-learn==1.9.0, faiss-cpu==1.14.3, joblib==1.5.3, pytest==9.1.1 (actual venv versions) so it fully provisions a venv. D-14's cache-key contract unaffected"
  - "Cache-key refinement (already flagged in the plan): load_subsample keys off a narrower subsample_cfg, not the full cfg, so a fit-only field like n_neighbors never invalidates the subsample artifact"

requirements-completed: [DATA-01, DATA-03, DATA-05, ISO-05]

coverage:
  - {id: D1, description: "Config-hash-keyed cache with sidecar manifest + CACHE_DIR containment guard (T-01-01/T-01-03)", requirement: "ISO-05", verification: [{kind: unit, ref: "test_pu_manifold.py cache round-trip/manifest-mismatch/traversal/config-key tests", status: pass}], human_judgment: false}
  - {id: D2, description: "Seeded row_indices, L2 normalize at cache-write, structural + statistical alignment assertion, strict > margin", requirement: "DATA-03", verification: [{kind: unit, ref: "test_pu_manifold.py draw_row_indices/l2_normalize/assert_alignment (incl. off-by-one control)", status: pass}], human_judgment: false}
  - {id: D3, description: "requirements-notebooks.txt: Task 1 pins, later fully provisioned per user-directed deviation", requirement: "DATA-05", verification: [{kind: other, ref: "grep-based pin checks, re-run after deviation; all pass", status: pass}], human_judgment: true, rationale: "Both pin decisions were explicit human/coordinator calls, not test-derivable."}
  - {id: D4, description: "Notebook §0-1 executed end-to-end via real Restart-and-Run-All (HF stream, subsample, Isomap fit, cache round-trips)", requirement: "DATA-01, ISO-05", verification: [{kind: integration, ref: "jupyter nbconvert --execute --inplace notebooks/01_manifold_and_gate.ipynb", status: pass}], human_judgment: true, rationale: "Tracer slice — required human sign-off on real executed outputs before Task 4 expansion (tracer-feedback-gate protocol)."}
  - {id: D5, description: "Phase 3/4 stubs (curvature.py, mknn.py): real signatures, docstrings, NotImplementedError naming phase/requirement, no torch/faiss import at module level", verification: [{kind: unit, ref: "inline stub-contract check + pytest re-run", status: pass}], human_judgment: false}

duration: ~35min active work across two sessions (paused between commits e584180 and c83045a for the tracer-feedback-gate checkpoint)
completed: 2026-07-30
status: complete
---

# Phase 1 Plan 1: Environment, Cache/Subsample Package, and Smoke-Config Tracer Summary

**Config-hash-keyed cache + seeded row-aligned subsample of `UniverseTBD/pu-embeddings`, proven end-to-end on a real 500-row smoke fit through a real Isomap round-trip before the 10k-row analysis artifact is ever built.**

## Performance

~35 min across two sessions (quota interruption mid-Task-3, paused at the tracer-feedback-gate
checkpoint). Completed 2026-07-30 (`a4de638`). 4/4 tasks. 8 files created, 0 modified.

## Accomplishments

`cache.py` (`config_key`/`cache_path`/`npz_cache`/`joblib_cache`/`json_cache`, sidecar-asserted,
`CACHE_DIR` containment guard, `KEY_LEN=16` deliberate deviation from ARCHITECTURE.md's `[:8]`).
`subsample.py` (`draw_row_indices`, `l2_normalize`, structural+statistical alignment assertion
`z > 5.0` vs 50-permutation null, `load_subsample` asserting exactly 101,725 rows). 14 synthetic
pytest tests incl. an off-by-one negative control (`z`~0, correctly raises). **Actually executed**
notebook §0-1 (`jupyter nbconvert --execute`): real HF stream, real 500-row subsample, real
npz/joblib round-trips (bit-identical), real `Isomap` fit (`eigen_solver="dense"`, 1 connected
component). Observed: `s_true=0.8456`, `mu_perm=0.7171`, `sd_perm=0.003075`, `z=41.78` (vs
`margin_z=5.0`). Scaffolded `curvature.py`/`mknn.py` stubs with the S^767 and per-region-null
notes carried forward. User-directed `requirements-notebooks.txt` full self-provisioning (below).

## Task Commits

Tasks 1-2 (checkpoints): no commit, approved/decided prior session. Task 3 (tracer): `e584180`
(feat), resumed from quota-interrupted state. Tracer-gate deviation: `c83045a` (fix),
self-provisioning `requirements-notebooks.txt`. Task 4 (stubs): `a4de638` (feat).

## Deviations from Plan

**[User-directed] Reverse the Task 1 exclusion policy in `requirements-notebooks.txt`** — found
during tracer-feedback-gate review after `e584180`; user wants the file self-sufficient for a
pre-existing venv (policy reversal, not a defect). Added
`numpy==2.5.1`/`scipy==1.18.0`/`scikit-learn==1.9.0`/`faiss-cpu==1.14.3`/`joblib==1.5.3`/`pytest==9.1.1`;
D-14's cache-key contract unaffected. Verified via plan's `<verify>` re-run. Committed `c83045a`.

## Issues Encountered

Provider-quota interruption mid-Task-3 (zero commits; resumed work reviewed the untracked files
as correct). Stray kernel corruption: a system `ipykernel` re-executed the notebook and overwrote
real outputs; detected via `git diff`, restored via `git checkout -- notebooks/01_manifold_and_gate.ipynb`.

## User Setup Required / Next Phase Readiness

None — `huggingface.co`/PyPI network access required and available. `pu_manifold/{cache,subsample}.py`
cache-key contracts are the load-bearing surface plans 02-04 build on. `ANALYSIS_CFG` defined with
`n_neighbors`/`n_components` as `None`. No blockers.

---
*Phase: 01-data-loading-manifold-reconstruction*
*Completed: 2026-07-30*

## Self-Check: PASSED

All 8 created files and 3 task/deviation commits (`e584180`, `c83045a`, `a4de638`) verified present on disk / in `git log --oneline --all`. No missing items.
</content>
