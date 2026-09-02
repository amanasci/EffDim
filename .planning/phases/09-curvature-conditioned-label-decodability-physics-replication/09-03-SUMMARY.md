---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 03
subsystem: data-loading
tags: [pyarrow, pandas, huggingface, parquet, row-alignment, ridge-regression]

# Dependency graph
requires:
  - phase: 09-01
    provides: "physics_labels.py module scaffold (assert_preregistered, mask_sentinels, canonical_label, shifted_pairing, alignment_r2_curve, alignment_verdict, resolve_hf_cache_dir), physics_curvature_probe.py (oof_ridge_predictions, record_path, _assert_inside_output_root, resolve_output_root), and 09_physics_curvature_run.py's runner shape to mirror"
provides:
  - "physics_labels.py: load_physics_embeddings, load_label_table, label_missingness_report, _shard_url -- revision-pinned, column-projected real HuggingFace loaders; every value each needs is either an explicit override or read from a still-UNSET frozen constant, so none can complete before the 09-05 freeze"
  - "09_row_alignment_proof_run.py: --mode smoke/manifest/proof/search runner, smoke-verified end to end (ALIGNMENT SMOKE PASS on both the aligned and injected-offset synthetic cases); manifest mode's CLI/record contract (row_kind key, comma-joined --candidate-columns, notebooks/.cache/09_data_manifest.jsonl default path) matches 09-04-PLAN.md's own hardcoded invocation exactly"
  - "10 new tests in test_physics_labels.py pinning the loaders' failure modes (missing column, row-count mismatch, empty read, shard order) and the manifest mode's metadata-only rule behaviourally, not by source grep"
  - "Live-network validation: the full --mode manifest exercise completed against BOTH real datasets at full scale (86,471x768 embeddings, all 16 v2.0 label shards) in 25m47s wallclock; mass_med_photoz's post-masking finite count (79,490) reproduces the colleague's own reported 79,490/86,471 figure exactly"
affects: [09-04, 09-05, 09-06, 09-07, 09-08, 09-09, 09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pre-freeze override parameters (parquet_path/column/expected_rows on load_physics_embeddings; expected_rows on load_label_table) let --mode manifest run before assert_preregistered() would pass, without weakening any post-freeze call site (which passes nothing and gets the frozen values)"
    - "row_kind provenance tag on every JSONL row this runner writes (column/summary/curve/verdict/search_classification), read by 09-04's own automated verify step"
    - "_hf_cache_env_override context manager: exports resolve_hf_cache_dir()'s resolved value into HF_HOME only when HF_HOME is not already set, restores the prior (absent) state on exit -- never overrides a value the execution host chose"
    - "_load_label_table_with_overrides: temporarily monkeypatches physics_labels' LABEL_REPO/LABEL_REVISION/LABEL_SPLIT/LABEL_N_SHARDS module globals around one load_label_table call, then restores them exactly -- lets --mode manifest source the label catalog entirely from the CLI without widening load_label_table's plan-specified (columns, expected_rows) signature or leaving any gating constant filled on exit"

key-files:
  created:
    - notebooks/diagnostics/09_row_alignment_proof_run.py
  modified:
    - notebooks/pu_manifold/physics_labels.py
    - notebooks/pu_manifold/tests/test_physics_labels.py

key-decisions:
  - "_load_label_table_with_overrides monkeypatches physics_labels' four label-source module globals for the duration of one call rather than widening load_label_table's signature -- the plan fixed that function's signature at (columns, expected_rows); this is the only way to give --mode manifest CLI-sourced repo/revision/split/shard-count without touching it"
  - "The --mode smoke synthetic fixture's OOF ridge uses alpha=1.0, not the real ALPHA_RIDGE=100.0 -- measured directly: alpha=100.0 over-shrinks this fixture's unit-norm-row feature scale to R2~0.29 for even the perfectly-aligned case, under the 0.30 smoke margin. This is a smoke-fixture-local constant, not a Phase 9 gating constant, and has no bearing on the frozen ALPHA_RIDGE the real proof mode will use"
  - "Renamed record_kind -> row_kind everywhere in the runner (manifest/proof/search rows alike) and made --candidate-columns split every argv token on ',' -- discovered by reading 09-04-PLAN.md while preparing this plan's <output> section: it already hardcodes both this runner's exact --mode manifest invocation (--candidate-columns passed as ONE comma-joined token) and an automated verify step reading r.get('row_kind')=='summary'. Both would have silently broken 09-04's verification without this fix"
  - "The full-scale manifest evidence-gathering run (09-04 Task 1's own job) was exercised here only as an exploratory validation, written to a scratch path (09_data_manifest_evidence.jsonl) distinct from the official 09_data_manifest.jsonl -- confirming the real code path works end to end without pre-empting or duplicating 09-04's deliverable"

requirements-completed: [D9-01, D9-02, D9-05, D9-06, D9-07, D9-08, D9-16]

coverage:
  - id: D1
    description: "Both HuggingFace datasets are readable by column projection at a pinned revision, with exact row-count and schema assertions, and neither loader can run before the freeze"
    requirement: "D9-05"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_labels.py::test_shard_url_pins_revision"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_labels.py::test_load_label_table_raises_on_missing_column"
        status: pass
      - kind: other
        ref: ".venv/bin/python -c \"...; p.load_label_table(['mag_r_desi'])\" exits 1 with RuntimeError naming LABEL_REPO UNSET"
        status: pass
    human_judgment: false
  - id: D2
    description: "The row-alignment proof runner exists with all four modes, and --mode smoke proves the machinery on both a known-aligned and a known-offset synthetic pair"
    requirement: "D9-06"
    verification:
      - kind: automated_ui
        ref: ".venv/bin/python notebooks/diagnostics/09_row_alignment_proof_run.py --mode smoke --record-path notebooks/.cache/09_scratch_alignment.jsonl (last line: ALIGNMENT SMOKE PASS)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The strict-ancestor gate rejects a missing freeze commit and a freeze commit equal to HEAD"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "--mode proof (no --freeze-commit) exits 1 naming D9-18; --mode proof --freeze-commit HEAD exits 1 (not a strict ancestor)"
        status: pass
    human_judgment: false
  - id: D4
    description: "--mode manifest's metadata-only rule is pinned behaviourally: no written row ever carries an r2/rho/p/passed key"
    requirement: "D9-16"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_physics_labels.py::test_manifest_mode_writes_no_statistic_key"
        status: pass
    human_judgment: false
  - id: D5
    description: "No machine-specific absolute path and no torch import exist in either new file"
    verification:
      - kind: other
        ref: "grep -nE '/home/|/Users/|C:\\\\\\\\' + grep -n 'import torch' on both files -- both print nothing"
        status: pass
    human_judgment: false

# Metrics
duration: 35min
completed: 2026-09-02
status: complete
---

# Phase 9 Plan 3: Real Data-Acquisition Layer and Row-Alignment Proof Runner Summary

**Revision-pinned, column-projected HuggingFace loaders plus the four-mode row-alignment proof runner, with the manifest mode's CLI/record contract matched exactly to what 09-04 already depends on.**

## Performance

- **Duration:** 35 min
- **Started:** 2026-09-02T20:56:18Z
- **Completed:** 2026-09-02T21:31:33Z
- **Tasks:** 3 completed (plus 1 post-commit interface fix)
- **Files modified:** 3

## Accomplishments
- Added `load_physics_embeddings`, `load_label_table`, `label_missingness_report` and `_shard_url` to `physics_labels.py`; every value each needs is an explicit override or a still-UNSET frozen constant, so none can complete before the 09-05 freeze
- Built `notebooks/diagnostics/09_row_alignment_proof_run.py` with `--mode smoke/manifest/proof/search`; `--mode smoke` verified `ALIGNMENT SMOKE PASS` on both the aligned (argmax shift 0) and injected-offset (single clearing alignment) synthetic cases
- Discovered, by reading 09-04-PLAN.md's own hardcoded invocation while preparing this plan's `<output>`, that this runner's original `record_kind` key and space-separated `--candidate-columns` would have silently broken 09-04's automated verify step; fixed both to match
- Verified the full `--mode manifest` exercise end to end against BOTH real datasets at full scale: `UniverseTBD/pu-embeddings` physics parquet (86,471 rows x 768 features, downloaded and cached at 531,970,096 bytes) and all 16 `Smith42/galaxies@v2.0` label shards (86,471 rows), completing in 25m47s wallclock with `mag_r_desi` 100% populated and `mass_med_photoz`'s post-masking finite count landing at exactly 79,490 -- matching the colleague's own reported figure
- 10 new tests added to `test_physics_labels.py`; full `notebooks/pu_manifold/tests/` suite: 909 passed, 2 skipped

## Task Commits

1. **Task 1: Revision-pinned, column-projected loaders for both HuggingFace datasets** - `9637c99` (feat)
2. **Task 2: The row-alignment proof runner — smoke, manifest, proof and search modes** - `e5ba34c` (feat)
3. **Task 3: Tests for the loaders and the manifest-mode statistic prohibition** - `cb1c02e` (test)
4. **Fix: match 09-04's dependent CLI/record contract exactly** - `6c96ec3` (fix)

## Files Created/Modified
- `notebooks/pu_manifold/physics_labels.py` - added the three loaders, `_shard_url`, `_require_label_source_constants`, `_hf_cache_env_override`
- `notebooks/diagnostics/09_row_alignment_proof_run.py` - new runner: all four modes, `_strict_ancestor_or_exit`, `resolve_record_path`, `append_record_row`, `_read_last_verdict_row`, `_load_label_table_with_overrides`
- `notebooks/pu_manifold/tests/test_physics_labels.py` - 10 new test functions plus a module-scoped `runner` fixture loading the new runner by file path

## Decisions Made
- `_load_label_table_with_overrides` monkeypatches `physics_labels`' four label-source module globals for the duration of one call rather than widening `load_label_table`'s plan-fixed `(columns, expected_rows)` signature
- The smoke fixture's synthetic OOF ridge uses `alpha=1.0` (measured), not the real `ALPHA_RIDGE=100.0` -- a fixture-local tuning choice with no bearing on any Phase 9 gating constant
- `record_kind` renamed to `row_kind` throughout, and `--candidate-columns` now accepts a single comma-joined token, matching 09-04-PLAN.md's own hardcoded invocation and verify step
- The exploratory full-scale manifest run was written to a scratch path, distinct from the official `09_data_manifest.jsonl` 09-04's own Task 1 produces, so it validates the code without pre-empting that plan's deliverable

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `LABEL_REPO` docstring line failed this plan's own revision-pairing prohibition**
- **Found during:** Task 1 acceptance-criteria check
- **Issue:** 09-01's pre-existing `LABEL_REPO` docstring (`"""HuggingFace repo id for the label catalog, e.g. "Smith42/galaxies"."""`) mentions the repo without `v2.0`/`LABEL_REVISION` on the same line, failing this plan's own acceptance criterion ("every line matching `Smith42/galaxies` also matches `LABEL_REVISION` or `v2.0`")
- **Fix:** Reworded the docstring to state the revision requirement explicitly on the same line
- **Files modified:** `notebooks/pu_manifold/physics_labels.py`
- **Verification:** `grep -n 'Smith42/galaxies'` shows both matching lines also contain `v2.0`/`LABEL_REVISION`
- **Committed in:** `9637c99` (Task 1 commit)

**2. [Rule 1 - Bug] Smoke fixture's ridge alpha over-shrunk the synthetic OOF fit below the smoke margin**
- **Found during:** Task 2, first `--mode smoke` run (`ALIGNMENT SMOKE FAIL`, aligned-case R2=0.29)
- **Issue:** `alpha=100.0` (borrowed from `09_physics_curvature_run.py`'s smoke fixture) over-regularizes this fixture's unit-norm-row feature scale; measured R2 sweep showed `alpha=100.0` gives R2=0.29 vs `alpha<=1.0` giving R2>0.99 for the identical aligned case
- **Fix:** Smoke fixture now uses `alpha=1.0`, documented inline as fixture-local and distinct from the real `ALPHA_RIDGE`
- **Files modified:** `notebooks/diagnostics/09_row_alignment_proof_run.py`
- **Verification:** `--mode smoke` now prints `ALIGNMENT SMOKE PASS`, exits 0
- **Committed in:** `e5ba34c` (Task 2 commit)

**3. [Rule 1 - Bug] Runner's record key and CLI flag shape didn't match 09-04-PLAN.md's hardcoded dependency**
- **Found during:** Preparing this plan's `<output>` section (reading 09-04-PLAN.md, which is not in this plan's own `<read_first>` but is directly downstream)
- **Issue:** 09-04-PLAN.md's Task 1 already hardcodes the exact invocation `--mode manifest --candidate-columns mag_r_desi,mag_r,photo_z,...` (one comma-joined token) and its automated verify reads `r.get('row_kind')=='summary'`. This runner originally wrote `record_kind` and only accepted space-separated `--candidate-columns` tokens -- both would have silently broken 09-04's verify step
- **Fix:** Renamed `record_kind` to `row_kind` everywhere in the runner; `--candidate-columns` now splits every argv token on `,` so both comma- and space-separated invocations produce the same column list
- **Files modified:** `notebooks/diagnostics/09_row_alignment_proof_run.py`
- **Verification:** Re-ran `--mode smoke` (still PASS) and the full `test_physics_labels.py` suite (49 passed) and full notebooks suite (909 passed, 2 skipped) after the rename
- **Committed in:** `6c96ec3` (post-Task-3 fix commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 - bugs; one docstring wording fix, one fixture-tuning fix, one cross-plan interface-contract fix)
**Impact on plan:** All three were necessary for correctness or for not silently breaking 09-04's dependent verify step. No scope creep, no gating constant touched, no sealed interface redesigned.

## Issues Encountered
The exploratory full-scale `--mode manifest` validation run took considerably longer than 09-04-PLAN.md's own stated 10-15-minute estimate: it completed in **25m47s wallclock** (real; 2m29s user, 0m33s sys -- almost entirely network-bound), against both datasets at full scale. It ran to completion successfully and produced this measured per-column table (86,471 rows total on both sides):

| raw column | n_finite_raw | n_sentinel | n_finite_masked |
|---|---|---|---|
| `mag_r_desi` | 86,471 | 0 | 86,471 |
| `mag_r` | 5,970 | 0 | 5,970 |
| `photo_z` | 80,035 | 0 | 80,035 |
| `smooth-or-featured_smooth_fraction` | 86,471 | 0 | 86,471 |
| `mass_med_photoz` | 80,102 | 612 | **79,490** |
| `elpetro_mass_log` | 5,972 | 0 | 5,972 |
| `total_sfr_median` | 7,771 | 465 | 7,306 |

`mass_med_photoz`'s post-masking count (79,490) reproduces the colleague's own reported 79,490/86,471 figure exactly. This is exploratory validation only, written to a scratch path (`notebooks/.cache/09_data_manifest_evidence.jsonl`, gitignored) distinct from the official `notebooks/.cache/09_data_manifest.jsonl` 09-04's own Task 1 will produce -- 09-04 still runs the manifest fresh and writes its own record and `09-DATA-MANIFEST.md`; this run's only purpose was to confirm the runner's real-data code path and CLI contract work end to end before 09-04 depends on them.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 09-04 can run its Task 1 exactly as documented: `.venv/bin/python notebooks/diagnostics/09_row_alignment_proof_run.py --mode manifest --candidate-columns mag_r_desi,mag_r,photo_z,smooth-or-featured_smooth_fraction,mass_med_photoz,elpetro_mass_log,total_sfr_median --record-path notebooks/.cache/09_data_manifest.jsonl` -- the CLI/record-shape mismatch that would have broken its automated verify step is fixed, and this plan's own exploratory run already confirms the command completes successfully end to end (25m47s wallclock) and reproduces the colleague's `mass_med_photoz` figure exactly (79,490/86,471)
- Both `assert_preregistered()` functions still raise; no Physics number exists anywhere in the tree (D9-18 held)
- No blockers

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-02*

## Self-Check: PASSED

All 3 created/modified files found on disk; all 4 commits (`9637c99`, `e5ba34c`, `cb1c02e`,
`6c96ec3`) found in git history.
