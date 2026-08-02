---
phase: 01-data-loading-manifold-reconstruction
plan: 01
subsystem: data-loading
tags: [huggingface-datasets, sklearn, isomap, joblib, numpy, faiss, pytest, notebooks]

# Dependency graph
requires: []
provides:
  - "notebooks/pu_manifold/cache.py -- config-hash-keyed npz/joblib/json cache with sidecar-manifest verification and CACHE_DIR containment guard"
  - "notebooks/pu_manifold/subsample.py -- seeded, row-alignment-safe subsampling of UniverseTBD/pu-embeddings legacysurvey_dinov3_vitb16, with a strict statistical alignment assertion"
  - "notebooks/pu_manifold/curvature.py, mknn.py -- Phase 3/4 stub surfaces (NotImplementedError, real signatures)"
  - "notebooks/01_manifold_and_gate.ipynb sections 0-1 -- environment/reproducibility header, dependency install, permanent smoke-config self-test, executed end-to-end with real HF data"
  - "notebooks/requirements-notebooks.txt -- fully self-provisioning pinned notebook venv"
affects: [01-02, 01-03, 01-04]

# Tech tracking
tech-stack:
  added: [torch==2.13.0+cpu, datasets==5.0.1, matplotlib==3.11.1, "numpy/scipy/scikit-learn/faiss-cpu/joblib/pytest (pinned in requirements-notebooks.txt only, per user-directed deviation)"]
  patterns:
    - "config-hash cache: sha256(json.dumps(cfg, sort_keys=True))[:16] + sidecar .meta.json manifest asserted on every load, never trusted from filename alone"
    - "two-tier cache key: a narrow subsample_key (dataset/seed/n_rows/normalize/datasets_version/numpy_version) separate from the full D-14 fit_key, so changing a fit-only param never busts the subsample cache"
    - "row-alignment invariant: one np.sort(rng.choice(..., replace=False)) index array, one indexing pass for both paired columns, structural + permutation-null-z-score statistical assertion, strict > comparison"
    - "notebook-scoped helper package (pu_manifold/), never installed, never imported from src/effdim/, lazy import of heavy notebook-only deps (datasets, eventually torch/faiss) inside function bodies not at module top level"

key-files:
  created:
    - notebooks/pu_manifold/cache.py
    - notebooks/pu_manifold/subsample.py
    - notebooks/pu_manifold/curvature.py
    - notebooks/pu_manifold/mknn.py
    - notebooks/pu_manifold/__init__.py
    - notebooks/pu_manifold/tests/test_pu_manifold.py
    - notebooks/requirements-notebooks.txt
    - notebooks/01_manifold_and_gate.ipynb
  modified: []

key-decisions:
  - "Task 1 gate (package legitimacy): approved. torch==2.13.0+cpu, datasets==5.0.1, matplotlib==3.11.1 all confirmed legitimate on PyPI before install; numpy/scipy/scikit-learn/faiss-cpu confirmed already-core and deliberately excluded from re-pinning at that time"
  - "Task 2 gate (D-05 artifact shape): normalized-only selected. subsample_{seed}_{key}.npz stores only the L2-normalized hsc/legacysurvey arrays plus hsc_norms/ls_norms (raw norms); no raw 768-d arrays cached. One-way tradeoff accepted: a later need for raw vectors means re-streaming the 553 MiB parquet"
  - "User-directed deviation (post-tracer, pre-Task-4): requirements-notebooks.txt reversed the Task 1 exclusion policy and now also pins numpy==2.5.1, scipy==1.18.0, scikit-learn==1.9.0, faiss-cpu==1.14.3, joblib==1.5.3, pytest==9.1.1 -- all read from the actual venv, matching the versions the tracer ran under and the notebook's own §0.3 header -- so the file fully provisions a user-supplied venv without first needing effdim installed from pyproject.toml. D-14's cache-key contract is unaffected: config_key hashes the installed versions at runtime, not this file's pins."
  - "Cache-key refinement (stated deviation from a literal reading of D-14, already flagged in the plan's own artifacts_this_phase_produces section): load_subsample computes its cache key from a narrower subsample_cfg dict, not the full caller-supplied cfg, so a fit-only field like n_neighbors never invalidates the subsample artifact"

requirements-completed: [DATA-01, DATA-03, DATA-05, ISO-05]

coverage:
  - id: D1
    description: "Config-hash-keyed npz/joblib/json cache helpers with sidecar manifest verification and CACHE_DIR containment guard (T-01-01, T-01-03 mitigations)"
    requirement: "ISO-05"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_pu_manifold.py::test_npz_cache_round_trip_is_bit_identical_and_computes_once, ::test_joblib_cache_round_trip_is_bit_identical_and_computes_once, ::test_manifest_mismatch_raises_instead_of_silently_reusing, ::test_cache_path_rejects_traversal_stem, ::test_config_key_length_and_insertion_order_stability, ::test_config_key_changes_with_any_single_field"
        status: pass
    human_judgment: false
  - id: D2
    description: "Seeded row_indices draw, L2 normalization at cache-write time, structural + statistical (permutation z-score) row-alignment assertion with a strict > margin comparison"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_pu_manifold.py::test_draw_row_indices_sorted_unique_and_reproducible, ::test_draw_row_indices_rejects_n_rows_below_two, ::test_draw_row_indices_rejects_n_rows_above_max, ::test_draw_row_indices_rejects_n_total_below_n_rows, ::test_l2_normalize_returns_unit_rows_and_correct_raw_norms, ::test_l2_normalize_rejects_zero_norm_row, ::test_assert_alignment_passes_on_synthetic_aligned_pair, ::test_assert_alignment_raises_on_off_by_one_negative_control"
        status: pass
    human_judgment: false
  - id: D3
    description: "notebooks/requirements-notebooks.txt: Task 1-approved pins (torch/datasets/matplotlib), later fully provisioned per user-directed deviation (numpy/scipy/scikit-learn/faiss-cpu/joblib/pytest added, exact venv versions)"
    requirement: "DATA-05"
    verification:
      - kind: other
        ref: "grep-based pin presence/absence checks in notebooks/requirements-notebooks.txt, re-run after the deviation; all pass"
        status: pass
    human_judgment: true
    rationale: "Both the initial pin set (Task 1) and the later full-provisioning reversal were explicit human/coordinator decisions, not something a test can validate as 'correct' independent of that instruction."
  - id: D4
    description: "notebooks/01_manifold_and_gate.ipynb sections 0-1: environment/reproducibility header, dependency install, permanent smoke-config self-test -- executed end-to-end via a real Restart-and-Run-All (real HF network stream, real subsample, real Isomap fit, real cache round-trips)"
    requirement: "DATA-01, ISO-05"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb (committed with real executed outputs); re-run via `jupyter nbconvert --to notebook --execute --inplace notebooks/01_manifold_and_gate.ipynb`"
        status: pass
    human_judgment: true
    rationale: "This is the tracer slice; per the executor's tracer-feedback-gate protocol it required explicit human sign-off on the real executed outputs before Task 4 (expansion) could proceed. That sign-off was given (coordinator's message: 'Tracer gate: NOT approved as-is' pending one change, then 'approved to continue' after the requirements-notebooks.txt deviation was applied)."
  - id: D5
    description: "Phase 3/4 stub modules (curvature.py, mknn.py): real signatures, full docstrings, NotImplementedError bodies naming the owning phase/requirement, no torch/faiss import at module level"
    verification:
      - kind: unit
        ref: "inline verification script (7 stub functions checked for docstring + NotImplementedError-naming-phase contract); also `python -m pytest notebooks/pu_manifold/tests/test_pu_manifold.py -q` re-run afterward to confirm no regression"
        status: pass
    human_judgment: false

# Metrics
duration: ~35min active work across two sessions (paused between commits e584180 and c83045a for the tracer-feedback-gate checkpoint)
completed: 2026-07-30
status: complete
---

# Phase 1 Plan 1: Environment, Cache/Subsample Package, and Smoke-Config Tracer Summary

**Config-hash-keyed cache + seeded row-aligned subsample of `UniverseTBD/pu-embeddings`, proven end-to-end on a real 500-row smoke fit through a real Isomap round-trip before the 10k-row analysis artifact is ever built.**

## Performance

- **Duration:** ~35 min active work, spanning two executor sessions (this run resumed after a provider-quota interruption mid-Task-3, then paused again at the tracer-feedback-gate checkpoint awaiting coordinator confirmation)
- **Started (this session):** 2026-07-30 (continuation)
- **Completed:** 2026-07-30T23:04:12-04:00 (last commit, `a4de638`)
- **Tasks:** 4/4 complete (2 gate-only checkpoints already answered when this session began; Task 3 finished from a partial state; Task 4 executed fresh)
- **Files modified:** 8 created, 0 modified (plus a follow-up edit to 1 of the 8 for the user-directed deviation)

## Accomplishments

- Built `notebooks/pu_manifold/cache.py`: `config_key`/`cache_path`/`npz_cache`/`joblib_cache`/`json_cache`, each backed by a sidecar `.meta.json` manifest that is asserted (not just filename-matched) on every load, plus a `CACHE_DIR` containment guard against path traversal. `KEY_LEN=16` (64-bit) is a stated, deliberate deviation from ARCHITECTURE.md's illustrative 8.
- Built `notebooks/pu_manifold/subsample.py`: `draw_row_indices` (`np.sort(default_rng(seed).choice(n_total, n_rows, replace=False))`), `l2_normalize`, the structural + statistical (`z`-score against a 50-permutation null, strict `z > 5.0`) row-alignment assertion, and `load_subsample`, which streams the `legacysurvey_dinov3_vitb16` config, asserts the loaded row count is exactly 101,725, and caches the L2-normalized result.
- Wrote 14 synthetic-array pytest tests covering the full cache and subsample contract, including an off-by-one negative control that drives the alignment `z`-score to ~0 and correctly raises.
- Wrote and **actually executed** `notebooks/01_manifold_and_gate.ipynb` §0-§1 end-to-end (`jupyter nbconvert --execute`, not a dry run): real network stream from `huggingface.co`, real `%pip install`, a real 500-row subsample, real L2 normalization, a real npz cache write/read-back, a real k-NN connectivity check (1 component), a real `Isomap` fit (`eigen_solver="dense"`), and a real joblib cache write/read-back with bit-identical reload. Observed alignment stats: `s_true=0.8456`, `mu_perm=0.7171`, `sd_perm=0.003075`, `z=41.78` (far above the strict `margin_z=5.0` threshold).
- Scaffolded `notebooks/pu_manifold/curvature.py` (CURV-01..04) and `notebooks/pu_manifold/mknn.py` (MKNN-01/04/05): real signatures, full `Parameters:`/`Returns:` docstrings, `NotImplementedError` bodies naming the owning phase/requirement, no `torch`/`faiss` import at module level. The S^767 sphere-matching consequence (CURV-06) and the per-region permutation-null requirement (MKNN-04) are carried forward in the module docstrings so they aren't lost before Phase 3/4.
- Applied a user-directed deviation to `notebooks/requirements-notebooks.txt`, reversing the original deliberate-exclusion policy for `numpy`/`scipy`/`scikit-learn`/`faiss-cpu` and adding `joblib`/`pytest`, so the file fully provisions a user-supplied venv on its own (see Deviations below).

## Task Commits

Each task was committed atomically:

1. **Task 1: Package legitimacy verification** (`checkpoint:human-verify`, gate `blocking-human`) — no commit (gate only); **approved** in a prior session.
2. **Task 2: D-05 normalization artifact shape** (`checkpoint:decision`, gate `blocking`) — no commit (gate only); **`normalized-only`** selected in a prior session.
3. **Task 3: End-to-end smoke-config tracer** — `e584180` (feat) — resumed from a quota-interrupted partial state; `requirements-notebooks.txt` and `cache.py` were reviewed and kept as-is (correct); `subsample.py`, `__init__.py`, the pytest suite, and the notebook §0-§1 were written and the notebook was actually executed.
4. **Tracer-gate deviation: fully provision requirements-notebooks.txt** — `c83045a` (fix) — user-directed reversal of the Task 1 exclusion policy, applied as its own atomic commit per the coordinator's explicit instruction.
5. **Task 4: Scaffold Phase 3/4 stub modules** — `a4de638` (feat) — `curvature.py` and `mknn.py` created; `__init__.py` already satisfied Task 4's docstring requirement from Task 3, verified unchanged.

**Plan metadata:** committed separately after this Summary (see final commit below).

## Files Created/Modified

- `notebooks/pu_manifold/cache.py` — config-hash-keyed npz/joblib/json cache, sidecar manifest, containment guard
- `notebooks/pu_manifold/subsample.py` — seeded subsample, L2 normalize, structural + statistical alignment assertion, `load_subsample`
- `notebooks/pu_manifold/__init__.py` — package surface, four-module map, curvature/mknn not imported eagerly
- `notebooks/pu_manifold/curvature.py` — Phase 3 stub surface (CURV-01..04)
- `notebooks/pu_manifold/mknn.py` — Phase 4 stub surface (MKNN-01/04/05)
- `notebooks/pu_manifold/tests/test_pu_manifold.py` — 14 synthetic-array tests, no network/torch dependency
- `notebooks/requirements-notebooks.txt` — pinned notebook-venv requirements, later made fully self-provisioning
- `notebooks/01_manifold_and_gate.ipynb` — sections 0-1, executed with real outputs

## Decisions Made

- **Task 1 (approved):** `torch==2.13.0+cpu`, `datasets==5.0.1`, `matplotlib==3.11.1` confirmed legitimate on PyPI; `numpy`/`scipy`/`scikit-learn`/`faiss-cpu` deliberately excluded from re-pinning at that time (already core `effdim` deps); `huggingface_hub`/`hf_xet`/`pyarrow` accepted as transitive.
- **Task 2 (`normalized-only`):** `subsample_*.npz` stores only the L2-normalized `hsc`/`legacysurvey` arrays plus `hsc_norms`/`ls_norms`. One-way tradeoff accepted per user instruction: no raw-array cache; a future need for raw vectors re-streams the parquet.
- **Cache-key refinement:** `load_subsample` keys the subsample cache on a narrower `{dataset, seed, n_rows, normalize, datasets_version, numpy_version}` dict rather than the full caller-supplied cfg, so a fit-only parameter (e.g. `n_neighbors`) changing never forces a re-download. This implements the plan's own "Cache-key refinement (stated deviation from D-14)" section rather than the more literal `config_key(cfg)` phrasing in Task 3's prose action text, since the two are in tension and the dedicated refinement section is the more detailed, deliberate design.
- **Post-tracer, coordinator-directed:** `requirements-notebooks.txt` now also pins `numpy==2.5.1`, `scipy==1.18.0`, `scikit-learn==1.9.0`, `faiss-cpu==1.14.3`, `joblib==1.5.3`, `pytest==9.1.1` (all read from the actual venv via `importlib.metadata.version`, matching the versions already exercised by the tracer and printed in the notebook's §0.3 header) so a fresh user-supplied venv is fully provisioned without needing `effdim`'s `pyproject.toml` installed first. See Deviations below.

## Deviations from Plan

### Auto-fixed / User-directed Issues

**1. [User-directed, coordinator-instructed] Reverse the Task 1 exclusion policy in `requirements-notebooks.txt`**
- **Found during:** Tracer-feedback-gate review, after Task 3's commit (`e584180`)
- **Issue:** Task 1's approved policy left `numpy`/`scipy`/`scikit-learn`/`faiss-cpu` unpinned (already core `effdim` deps). User runs notebooks in a pre-existing venv and wants the file self-sufficient — a policy reversal, not a defect.
- **Fix:** Added `numpy==2.5.1`, `scipy==1.18.0`, `scikit-learn==1.9.0`, `faiss-cpu==1.14.3` (exact tracer/§0.3-header versions), plus `joblib==1.5.3` (direct `cache.py` import, previously transitive) and `pytest==9.1.1` (pyproject.toml's `dev` extra). Rewrote the header comment to state the new "fully provision a user-supplied venv" policy and clarify D-14's cache-key contract is unaffected (hashes installed versions at runtime, not pins).
- **Files modified:** `notebooks/requirements-notebooks.txt`
- **Verification:** Plan's automated `<verify>` block re-run; the original "these four packages are absent" assertion is superseded (documented, not silently weakened) — all other checks (`trust_remote_code`, `eigvalsh`, `pyproject.toml`/`src/effdim/` untouched, 14/14 pytest, cache-key contract, notebook §0 assertions) pass unchanged. No re-execution needed: new pins match versions already exercised.
- **Committed in:** `c83045a` (separate atomic commit, as instructed)

---

**Total deviations:** 1 user-directed (requirements-notebooks.txt full provisioning), plus one internal cache-key-scoping refinement already anticipated in the plan's own "Cache-key refinement" section (not counted as a deviation, since the plan names it as deliberate).
**Impact on plan:** No scope creep. The requirements-notebooks.txt change is additive and does not alter any code path, cache contract, or test. All plan `<verify>` checks pass except the one superseded assertion, explicitly documented.

## Issues Encountered

- **Provider-quota interruption:** A prior executor was cut off mid-Task-3 with zero commits, leaving `requirements-notebooks.txt` and `cache.py` untracked. Both reviewed against the plan spec on resume and found correct; no rework needed.
- **Stray kernel corruption during the deviation fix:** Between `e584180` and the coordinator's requested change, a stray `ipykernel` process (system Python 3.13.14, not the project `.venv`, likely an IDE auto-opening the notebook) re-executed `01_manifold_and_gate.ipynb` in place and overwrote its real outputs with a broken run (`ModuleNotFoundError: No module named 'scipy'`). Detected via `git diff` before staging; the stray process was killed and the notebook restored via `git checkout -- notebooks/01_manifold_and_gate.ipynb` (a targeted single-file restore, the destructive-git-prohibition's sanctioned exception) to the verified committed state. No data lost — corruption only touched the uncommitted working tree.

## User Setup Required

None — no external service configuration required. Network access to `huggingface.co` and PyPI (including `download.pytorch.org`) was required and available in this environment; `~/.cache/huggingface` was writable.

## Next Phase Readiness

- `notebooks/pu_manifold/{cache,subsample}.py` and their cache-key contracts (`subsample_key`, `fit_key`, sidecar manifests) are the load-bearing surface Plans 02-04 build on. Both are proven end-to-end at smoke scale with real data, not just unit-tested in isolation.
- `ANALYSIS_CFG` is defined in the notebook with `n_neighbors`/`n_components` left as `None`, ready for Plan 03's connectivity sweep and D-12 derivation.
- `curvature.py`/`mknn.py` stubs are in place with the cross-phase notes (S^767 sphere-matching, per-region null) Phase 3/4 need preserved.
- No blockers. `pyproject.toml` and `src/effdim/` remain byte-identical to their pre-plan state (verified via `git diff --quiet`).

---
*Phase: 01-data-loading-manifold-reconstruction*
*Completed: 2026-07-30*

## Self-Check: PASSED

All 8 created files and 3 task/deviation commits (`e584180`, `c83045a`, `a4de638`) verified present on disk / in `git log --oneline --all`. No missing items.
