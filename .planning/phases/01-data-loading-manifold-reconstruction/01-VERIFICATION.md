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

**Verified:** 2026-07-31T04:37:52Z
**Status:** passed
**Re-verification:** No — initial verification

## Method

This phase's deliverable is a notebook plus a small support package; the scientific artifacts
live in `notebooks/.cache/` (gitignored by design). Verification installed the exact pinned
runtime (`numpy==2.5.1`, `scipy==1.18.0`, `scikit-learn==1.9.0`, `joblib==1.5.3`,
`pytest==9.1.1` — matching `notebooks/requirements-notebooks.txt` and the notebook's own §0.3
header) and then, independently of SUMMARY.md's prose claims:

- Loaded `notebooks/.cache/isomap_43cf438bc944c509.joblib` directly with `joblib.load` and
  inspected its attributes.
- Parsed `notebooks/.cache/phase1_handoff_43cf438bc944c509.json` directly.
- Loaded `notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz` and recomputed row norms /
  monotonicity checks from the raw arrays.
- Inspected the four `sweep_k{5,10,15,30}_*.npz` field sets.
- Re-ran `python -m pytest notebooks/pu_manifold/tests/test_pu_manifold.py -q` (14/14 pass).
- Read `notebooks/01_manifold_and_gate.ipynb`'s committed cell **outputs** (not just source) for
  every section §0–§6, cross-checking specific numeric claims (connectivity counts, alignment
  z-scores, compute_dim panel values, plateau table) against SUMMARY.md's prose.
- Cross-referenced `01-REVIEW.md` (committed code review, 1 Critical / 3 Warning / 4 Info) against
  the current file contents to confirm which findings remain open.
- Confirmed `pyproject.toml` and `src/effdim/` are byte-identical to pre-phase state
  (`git diff` empty for both across the whole phase's commit range).

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Loads exactly `legacysurvey_dinov3_vitb16`, asserts 101,725 rows (DATA-01) | ✓ VERIFIED | `subsample.py` `EXPECTED_N_TOTAL=101_725` assert; notebook cell 33 references it; `load_dataset(..., name=cfg["dataset"])` — single config only |
| 2 | 10,000-row subsample, seeded, reproducible, cached (DATA-02) | ✓ VERIFIED | `subsample_20260729_a79b3460b838fd0a.npz` on disk: `hsc`/`legacysurvey` both `(10000,768)`; §1.6 CACHE HIT demonstrated on 2nd call |
| 3 | HSC/Legacy-Survey arrays row-aligned, enforced by assertion not convention (DATA-03) | ✓ VERIFIED | `row_indices` strictly increasing (`np.diff > 0` confirmed on the actual cached array); full-scale alignment `z=203.93` (recomputed value in handoff JSON matches SUMMARY); negative control (`roll=1000`) raises `ValueError` (cell 39 output confirmed) |
| 4 | Norm distribution shown, metric choice explicitly stated with rationale (DATA-04) | ✓ VERIFIED | Cell 43 prints HSC/LS min/median/mean/max/std/CV (CV 3.24%/3.14%); cell 46 markdown states L2-normalize-then-Euclidean as a locked, unconditional decision with the cosine-equivalence proof |
| 5 | Notebook states Python floor (3.11), installs own deps, `pyproject.toml` untouched (DATA-05) | ✓ VERIFIED | Cell 3: `assert sys.version_info >= (3, 11)` naming LIB-03 drift, executed output `Python 3.14.6 OK`; `requirements-notebooks.txt` present with pinned deps; `git diff` empty for `pyproject.toml`/`src/effdim/` across the phase |
| 6 | Connected-component count shown before Isomap fit (ISO-01) | ✓ VERIFIED | Cell 63 output: all six k in {5,8,10,15,20,30} give `n_components=1`, printed with size distribution, before any stage-2 Isomap fit runs |
| 7 | Embedding/eigenspectrum stability checked across ≥3 `n_neighbors` values (ISO-02) | ✓ VERIFIED | Cell 70 STABILITY_TABLE: 3 adjacent pairs (5,10)/(10,15)/(15,30), 3 metrics each; PLATEAU_RUNS=`[10,15,30]` len 3, matches SUMMARY |
| 8 | `effdim.compute_dim` on raw 768-d embeddings compared against Isomap eigenspectrum-suggested dimension (ISO-03) | ✓ VERIFIED | Cell 54 output: 8 geometric keys printed, `median=17.183`, `N_COMPONENTS=ceil=18`; matches SUMMARY exactly |
| 9 | Isomap fit at n=10,000 completes and is cached (ISO-04) | ✓ VERIFIED | `isomap_43cf438bc944c509.joblib` loaded directly: `n_neighbors=15`, `n_components=18`, `dist_matrix_.shape=(10000,10000)`, `embedding_.shape=(10000,18)`, `nbrs_`/`kernel_pca_` present |
| 10 | Re-run gets identical cached results; config change → new cache key (ISO-05) | ✓ VERIFIED | 14/14 `pu_manifold` pytest tests pass including manifest-mismatch-raises and cache-round-trip-bit-identical; notebook cell 79 demonstrates CACHE HIT and a config-key change on `n_neighbors+1` (`53a54bf5917e48d0` ≠ `43cf438bc944c509`) — see also CR-01 caveat below for one narrow call-site gap |

**Score:** 10/10 roadmap-level truths verified (all DATA/ISO requirement IDs independently confirmed against on-disk artifacts and committed notebook cell outputs, not merely SUMMARY.md prose).

### Frozen Values — Independently Re-Derived

All ground-truth values named in the verification brief were independently confirmed against the
artifacts on disk (not taken from SUMMARY.md):

| Field | Claimed | Independently confirmed |
|---|---|---|
| `k_star` | 15 | ✓ `isomap_43cf438bc944c509.joblib.n_neighbors == 15`; handoff JSON `k_star: 15` |
| `n_components` | 18 | ✓ `isomap.n_components == 18`; handoff JSON `n_components: 18` |
| `d_provisional` | 18 | ✓ handoff JSON `d_provisional: 18` |
| `subsample_key` | a79b3460b838fd0a | ✓ handoff JSON `subsample_key`; matches cached npz filename |
| `fit_key` | 43cf438bc944c509 | ✓ handoff JSON `fit_key`; matches joblib filename |
| flags | `{short_circuit_risk: false, k_auto_extended: false, n_components_no_headroom: true}` | ✓ handoff JSON `flags` block, exact match |
| `fit_seconds` | 66.86 | ⚠️ **NOT independently confirmable.** See Anti-Patterns / Gaps below — this number appears only in `01-04-SUMMARY.md` prose; the committed notebook's own §5.2 cell output reads `"fit_seconds not available this run (isomap_{fit_key}.joblib was already cached)"` in every commit of the notebook that touches this cell (checked `1535010` and the current `HEAD`). Not a blocker — it does not correspond to any must-have truth — but it means one specific SUMMARY.md-cited number cannot be verified from committed evidence. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `notebooks/.cache/isomap_43cf438bc944c509.joblib` | Full Isomap fit, ~1.55 GiB | ✓ VERIFIED | 1,664,401,892 bytes on disk; loaded directly, all attributes/shapes confirmed |
| `notebooks/.cache/phase1_handoff_43cf438bc944c509.json` | 14-key Phase 1→2 interface | ✓ VERIFIED | Loaded directly; exactly 14 top-level keys present and populated |
| `notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz` | 10,000-row analysis subsample | ✓ VERIFIED | `hsc`/`legacysurvey` `(10000,768)`, unit-norm confirmed, `row_indices` sorted |
| `notebooks/.cache/sweep_k{5,10,15,30}_*.npz` | Per-k sweep artifacts, ~1.1 MB each | ✓ VERIFIED | 1,122,020 bytes each; 7-key field set confirmed (`embedding`, `eigenvalues_truncated`, `n_connected_components`, `fit_seconds`, `geo_pairs`, `geo_pair_count`, `geo_pair_seed`); no `dist_matrix_` leak |
| `notebooks/01_manifold_and_gate.ipynb` | Committed with executed outputs, §0–§6 | ✓ VERIFIED | 90 cells, 32 code cells, all carry non-null `execution_count`, zero error-type outputs |
| `notebooks/pu_manifold/cache.py` | Config-hash-keyed cache | ✓ VERIFIED | `config_key`/`cache_path`/`npz_cache`/`joblib_cache`/`json_cache`/`_assert_inside_cache` all present, exercised by passing tests |
| `notebooks/pu_manifold/subsample.py` | Seeded, aligned subsampling | ✓ VERIFIED | All named functions present; behavior confirmed against real cached artifacts |
| `notebooks/pu_manifold/curvature.py`, `mknn.py` | Phase 3/4 stubs | ✓ VERIFIED | Both raise `NotImplementedError` naming owning phase; neither imports torch/faiss at module level (grep confirmed) |
| `notebooks/pu_manifold/tests/test_pu_manifold.py` | Fast synthetic-array test suite | ✓ VERIFIED | 14/14 pass under the exact pinned runtime versions |
| `notebooks/requirements-notebooks.txt` | Pinned notebook deps | ✓ VERIFIED | Present, matches the versions actually installed and used for this verification |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `notebooks/01_manifold_and_gate.ipynb` | `notebooks/pu_manifold/subsample.py` | plain relative import | ✓ WIRED | Cell 20 `from pu_manifold import load_subsample, ...`, executed successfully in committed run |
| `notebooks/pu_manifold/subsample.py` | `notebooks/pu_manifold/cache.py` | `npz_cache` wrap | ✓ WIRED | `load_subsample` body wraps compute path in `cache.npz_cache` |
| `notebooks/01_manifold_and_gate.ipynb` | `notebooks/.cache/isomap_{fit_key}.joblib` | `joblib_cache` | ✓ WIRED | §5.2 confirmed; artifact loads with expected shape/attributes |
| `notebooks/01_manifold_and_gate.ipynb` | `notebooks/.cache/phase1_handoff_{fit_key}.json` | `json_cache` | ✓ WIRED, ⚠️ narrow-scope cfg | Artifact written and readable; **but** the `cfg` passed to `json_cache` (`ANALYSIS_CFG`) omits the §4.0 sweep constants that determine the `k_star_selection` payload being cached — see CR-01 below |
| `notebooks/01_manifold_and_gate.ipynb` | `src/effdim/api.py` `compute_dim` | read-only call | ✓ WIRED | Cell 54 output shows real 8-key geometric panel + 11-key spectral panel from a real `compute_dim(LS)` call (~48s runtime, matches SUMMARY) |
| `notebooks/01_manifold_and_gate.ipynb` §4.0 | §4.2 stage-2 sweep | cell-index self-check | ✓ WIRED | Cell 73 asserts `_threshold_cell_idx < _sweep_cell_idx` mechanically, not by reading discipline |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| DATA-01 | 01-01 | Single-config load, 101,725-row assert | ✓ SATISFIED | §0/§1.6 |
| DATA-02 | 01-02 | Reproducible 10,000-row subsample, explicit seed | ✓ SATISFIED | §1.6, cached npz |
| DATA-03 | 01-01, 01-02 | Row-alignment assertion, not convention | ✓ SATISFIED | subsample.py + §1.6–1.7 |
| DATA-04 | 01-02 | Norm distribution + explicit metric statement | ✓ SATISFIED | §2.1–2.2 |
| DATA-05 | 01-01 | Python floor stated, notebook-cell install, `pyproject.toml` untouched | ✓ SATISFIED | §0.1–0.2, `git diff` clean |
| ISO-01 | 01-03 | Connected-component count before fit | ✓ SATISFIED | §4.1 |
| ISO-02 | 01-03 | Stability across ≥3 `n_neighbors` values | ✓ SATISFIED | §4.2–4.3 |
| ISO-03 | 01-02 | `compute_dim` on raw embeddings vs. Isomap-suggested dim | ✓ SATISFIED | §3 |
| ISO-04 | 01-04 | Isomap fit completes and is cached | ✓ SATISFIED | §5.1–5.2, joblib artifact |
| ISO-05 | 01-01, 01-04 | Reproducible from cache; config change → new key | ✓ SATISFIED (see CR-01 caveat) | pytest suite, §1.3/§5.2 CACHE HIT demos |

No orphaned requirements: `REQUIREMENTS.md`'s Traceability table maps exactly DATA-01..05 and
ISO-01..05 to Phase 1, and all ten IDs appear in the union of the four plans' `requirements:`
frontmatter fields.

### Anti-Patterns Found

| File | Line/Cell | Pattern | Severity | Impact |
|---|---|---|---|---|
| `notebooks/01_manifold_and_gate.ipynb` §5.3, cell 84 | `PHASE1_HANDOFF = json_cache(f"phase1_handoff_{fit_key}", ANALYSIS_CFG, ...)` | Cache key too narrow (CR-01, from `01-REVIEW.md`) | WARNING | Confirmed still present and unfixed in the current file (no commits since `5ef9fc6`). If a reader relaxes a §4.0 plateau threshold (the notebook's own remediation text in cell 76 suggests exactly this) and the relaxation happens not to change `K_STAR`/`N_COMPONENTS`, `ANALYSIS_CFG` and `fit_key` stay bit-identical, so `json_cache` would silently return the **prior** run's `k_star_selection.thresholds`/`plateau_runs` from disk without recomputing. **Does not corrupt the current, frozen handoff artifact** — the JSON on disk right now is correct for the run that produced it — but it is a real, documented, open gap in the "config change → new cache key" guarantee for this one artifact. Not entered in `.planning/WINDOWS.md` (only the STAGE2_K-spacing item is). Recommend: either fix per the review's suggested remediation, or explicitly waive/track it in `WINDOWS.md` before Phase 2 (or any later re-run) touches §4.0 constants. |
| `notebooks/pu_manifold/cache.py` :151-167, :193-194, :226-227, :254-255 | Non-atomic writes (WR-01) | INFO | From `01-REVIEW.md`, still open; an interrupted write can crash-loop on `JSONDecodeError`. Low likelihood in this milestone's actual usage pattern; not a functional-correctness issue for the artifacts already on disk. |
| `notebooks/pu_manifold/subsample.py` :96-121 | `l2_normalize` only guards exact-zero norm (WR-02) | INFO | Still open; real DINOv3 embeddings are very unlikely to trigger this, and the committed run's norms are all well within range (min 13.4, confirmed). |
| `notebooks/01_manifold_and_gate.ipynb` §4.2, cell 66 | `_stage2_k_selection` dead/inverted fallback branch (WR-03) | INFO | Still open; provably unreachable for every `(n, max_fits)` this milestone actually calls it with. |
| Various (IN-01..04) | — | Magic-number duplication, narrow test coverage, unenforced-alternative constant, overstated `cwd` comment | INFO | All still open per `01-REVIEW.md`; none affect the phase's observable truths. |

No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` debt markers found in any file modified by this
phase (grep run against `pu_manifold/*.py`, the test file, and the notebook's source cells; the
only substring hits were the dataset name `UniverseTBD` and base64 image bytes in cell outputs,
both false positives).

### Known Limitation (already tracked, confirmed properly documented)

`STAGE2_K = [5, 10, 15, 30]` is unevenly spaced (gaps 5, 5, 15); connected values k=8 and k=20 were
never fit, so the plateau run `[10, 15, 30]` that froze `K_STAR=15` is maximal in *index* space,
not *k* space. Confirmed present in `.planning/WINDOWS.md` as open item id 1, and confirmed
disclosed to and accepted by the user at the Plan 03 Task 4 checkpoint (per `01-03-SUMMARY.md`)
before the decision was made. Correctly documented; not re-reported as a new gap; not re-litigated
here per the user's own prior instruction not to act on it.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| `pu_manifold` cache/subsample contract | `python -m pytest notebooks/pu_manifold/tests/test_pu_manifold.py -q` | `14 passed, 1 warning` (unrelated joblib/numpy deprecation warning) | ✓ PASS |
| `isomap_43cf438bc944c509.joblib` loads and matches claimed shape/params | `joblib.load(...)` then attribute inspection | `n_neighbors=15, n_components=18, dist_matrix_=(10000,10000), embedding_=(10000,18)` | ✓ PASS |
| `phase1_handoff_43cf438bc944c509.json` has the claimed 14-key structure and frozen values | `json.load` then key/value inspection | 14 keys, all frozen values match | ✓ PASS |
| Subsample row-alignment invariant on the real cached array | `np.diff(row_indices) > 0` check on the actual npz | strictly increasing, unit-norm rows confirmed | ✓ PASS |
| `pyproject.toml`/`src/effdim/` untouched across the whole phase | `git diff` against pre-phase commit | empty diff | ✓ PASS |

### Human Verification Required

None. All must-have truths are independently confirmable from on-disk artifacts and committed
notebook cell outputs; no behavior-dependent truth required human judgment beyond what the
existing human-verify/decision checkpoints (Task 1/2 of plan 01, Task 4 of plan 03, Task 3 of
plan 04 — all already recorded as approved in the SUMMARYs) already covered during execution.

### Gaps Summary

No blocking gaps. Two informational items are worth carrying forward for whoever plans or executes
Phase 2:

1. **CR-01 (open, WARNING):** the `phase1_handoff` cache key is narrower than the data it caches
   (see Anti-Patterns above). Does not affect the correctness of the artifact currently on disk;
   does create a latent staleness risk the moment someone tunes a §4.0 constant without also
   changing `ANALYSIS_CFG`. Recommend fixing or formally waiving via `gsd-tools windows` before
   that scenario arises.
2. **`fit_seconds=66.86` is not independently verifiable** from any committed artifact — it is a
   real number reported in `01-04-SUMMARY.md` prose, but the corresponding notebook cell's
   committed output shows a cache-hit path ("`fit_seconds not available this run`") in every commit
   inspected. This does not correspond to any must-have truth (the fit's completion, caching, and
   round-trip identity are independently verified through other means above), so it is reported
   here as an evidentiary note rather than a gap.

---

_Verified: 2026-07-31T04:37:52Z_
_Verifier: Claude (gsd-verifier)_
