---
phase: 01-data-loading-manifold-reconstruction
verified: 2026-07-31T04:37:52Z
status: passed
score: 24/24 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 1: Data Loading & Manifold Reconstruction Verification Report

**Phase Goal:** A reproducible, row-aligned 10,000-row subsample of `legacysurvey_dinov3_vitb16`
is loaded and cached, and an Isomap fit on it is validated for connectivity and short-circuit /
`n_neighbors` stability.

**Verified:** 2026-07-31T04:37:52Z · **Status:** passed · **Re-verification:** No — initial

## Method

Installed the exact pinned runtime and, independently of SUMMARY.md's prose: loaded
`isomap_43cf438bc944c509.joblib` and `phase1_handoff_43cf438bc944c509.json` directly; loaded
`subsample_20260729_a79b3460b838fd0a.npz` and recomputed row norms/monotonicity; inspected the
four `sweep_k{5,10,15,30}_*.npz` field sets; re-ran the pytest suite; read the notebook's committed
cell **outputs** (not source) for §0-§6, cross-checking numeric claims; cross-referenced
`01-REVIEW.md` against current contents; confirmed `pyproject.toml`/`src/effdim/` byte-identical.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Loads exactly `legacysurvey_dinov3_vitb16`, asserts 101,725 rows (DATA-01) | ✓ | `EXPECTED_N_TOTAL=101_725` assert; single-config `load_dataset` call |
| 2 | 10,000-row subsample, seeded, reproducible, cached (DATA-02) | ✓ | npz on disk: `(10000,768)` both arrays; §1.6 CACHE HIT on 2nd call |
| 3 | HSC/LS row-aligned, assertion not convention (DATA-03) | ✓ | `row_indices` strictly increasing; `z=203.93`; `roll=1000` control raises |
| 4 | Norm distribution + explicit metric statement (DATA-04) | ✓ | Cell 43 CV 3.24%/3.14%; cell 46 states L2-then-Euclidean unconditionally |
| 5 | Python floor stated, own deps installed, `pyproject.toml` untouched (DATA-05) | ✓ | Cell 3 assert, `requirements-notebooks.txt` present, `git diff` empty |
| 6 | Connected-component count before Isomap fit (ISO-01) | ✓ | Cell 63: all six k give `n_components=1` before any stage-2 fit |
| 7 | Stability across ≥3 `n_neighbors` values (ISO-02) | ✓ | Cell 70: 3 adjacent pairs, 3 metrics each; `PLATEAU_RUNS=[10,15,30]` len 3 |
| 8 | `compute_dim` vs Isomap-suggested dimension compared (ISO-03) | ✓ | Cell 54: 8 geometric keys, `median=17.183`, `N_COMPONENTS=18` |
| 9 | Isomap fit at n=10,000 completes and is cached (ISO-04) | ✓ | joblib loaded directly: shapes/params match, `nbrs_`/`kernel_pca_` present |
| 10 | Re-run identical from cache; config change → new key (ISO-05) | ✓ | 14/14 pytest; CACHE HIT + `n_neighbors+1` → `53a54bf5917e48d0` ≠ `43cf438bc944c509` (see CR-01 caveat) |

**Score:** 10/10 roadmap-level truths verified against on-disk artifacts and cell outputs, not
SUMMARY.md prose.

### Frozen Values — Independently Re-Derived

| Field | Claimed | Confirmed |
|---|---|---|
| `k_star` | 15 | ✓ joblib `n_neighbors==15`; handoff JSON matches |
| `n_components` | 18 | ✓ joblib + handoff JSON match |
| `d_provisional` | 18 | ✓ handoff JSON |
| `subsample_key` | a79b3460b838fd0a | ✓ handoff JSON, matches npz filename |
| `fit_key` | 43cf438bc944c509 | ✓ handoff JSON, matches joblib filename |
| flags | `{short_circuit_risk:false, k_auto_extended:false, n_components_no_headroom:true}` | ✓ exact match |
| `fit_seconds` | 66.86 | ⚠️ NOT independently confirmable — appears only in `01-04-SUMMARY.md` prose; the committed notebook's §5.2 cell output reads "fit_seconds not available this run (already cached)" in both `1535010` and current HEAD. Not a blocker — no must-have truth depends on it. |

### Required Artifacts

All ✓ VERIFIED: `isomap_43cf438bc944c509.joblib` (1,664,401,892 bytes, shapes/attrs confirmed);
`phase1_handoff_43cf438bc944c509.json` (14 keys populated); `subsample_20260729_a79b3460b838fd0a.npz`
(shapes/unit-norm/sorted `row_indices`); four `sweep_k{5,10,15,30}_*.npz` (1,122,020 bytes each,
7-key set, no `dist_matrix_` leak); `01_manifold_and_gate.ipynb` (90 cells, 32 code cells all
non-null execution counts, zero errors); `pu_manifold/cache.py`/`subsample.py` (all named functions
present, exercised by passing tests); `curvature.py`/`mknn.py` stubs (raise `NotImplementedError`
naming phase, no torch/faiss at module level); `test_pu_manifold.py` (14/14 pass under pinned
versions); `requirements-notebooks.txt` (matches installed versions).

### Key Link Verification

All ✓ WIRED except one narrow-scope caveat: notebook→`subsample.py` (plain import, cell 20);
`subsample.py`→`cache.py` (`npz_cache` wrap); notebook→`isomap_{fit_key}.joblib`
(`joblib_cache`, §5.2); notebook→`phase1_handoff_{fit_key}.json` (`json_cache` — **but** the `cfg`
passed (`ANALYSIS_CFG`) omits the §4.0 sweep constants determining the cached `k_star_selection`
payload, see CR-01 below); notebook→`compute_dim` (read-only, cell 54, real ~48s runtime); §4.0→§4.2
cell-index self-check (cell 73 asserts mechanically).

### Requirements Coverage

All ten requirements ✓ SATISFIED: DATA-01 (§0/§1.6, plan 01-01), DATA-02 (§1.6, cached npz, 01-02),
DATA-03 (subsample.py+§1.6-1.7, 01-01/02), DATA-04 (§2.1-2.2, 01-02), DATA-05 (§0.1-0.2, `git diff`
clean, 01-01), ISO-01 (§4.1, 01-03), ISO-02 (§4.2-4.3, 01-03), ISO-03 (§3, 01-02), ISO-04
(§5.1-5.2 + joblib, 01-04), ISO-05 (pytest + §1.3/§5.2 CACHE HIT demos, 01-01/04, see CR-01
caveat). No orphaned requirements — `REQUIREMENTS.md`'s Traceability table maps exactly
DATA-01..05/ISO-01..05 to Phase 1, all ten IDs appear across the four plans' `requirements:`
frontmatter.

### Anti-Patterns Found

All findings from `01-REVIEW.md` confirmed still open, none blocking: **CR-01** (WARNING, notebook
§5.3 cell 84) — cache key too narrow, no commits since `5ef9fc6`; relaxing a §4.0 threshold
without changing `K_STAR`/`N_COMPONENTS` would leave `fit_key` bit-identical and `json_cache` would
silently return the prior run's `thresholds`/`plateau_runs`. Does NOT corrupt the artifact
currently on disk. Not entered in `.planning/WINDOWS.md` (only STAGE2_K is) — recommend fixing or
waiving before Phase 2 touches §4.0 constants. **WR-01/02/03, IN-01..04** (INFO) — non-atomic cache
writes, `l2_normalize` exact-zero-only guard (norms well within range, min 13.4), dead/inverted
`_stage2_k_selection` fallback (provably unreachable for calls this milestone makes), magic-number
duplication/narrow test coverage/unenforced constant/overstated cwd comment — none affect
observable truths. No debt markers (`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`) found (grep;
only false positives).

### Known Limitation (already tracked)

`STAGE2_K=[5,10,15,30]` unevenly spaced (gaps 5,5,15); k=8/k=20 never fit, so the plateau run
`[10,15,30]` that froze `K_STAR=15` is maximal in *index* space not *k* space. Confirmed present in
`.planning/WINDOWS.md` id 1, disclosed and accepted at plan 03's Task 4 checkpoint. Correctly
documented; not re-litigated here.

### Behavioral Spot-Checks

All ✓ PASS: pytest suite `14 passed, 1 warning` (unrelated deprecation); joblib load matches claimed
shapes/params; handoff JSON has the claimed 14-key structure and frozen values; `row_indices`
strictly increasing/unit-norm on the real cached array; `pyproject.toml`/`src/effdim/` `git diff`
empty across the phase.

### Human Verification Required / Gaps Summary

None required — all must-have truths independently confirmable from on-disk artifacts and cell
outputs; existing human-verify/decision checkpoints (plan 01 Tasks 1/2, plan 03 Task 4, plan 04
Task 3, all recorded approved) already covered behavior-dependent judgment during execution. No
blocking gaps; CR-01 and the `fit_seconds` evidentiary note (both above) are the only carry-forwards.

---
_Verified: 2026-07-31T04:37:52Z_ · _Verifier: Claude (gsd-verifier)_
</content>
