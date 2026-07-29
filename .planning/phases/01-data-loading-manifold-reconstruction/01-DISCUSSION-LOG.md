# Phase 1: Data Loading & Manifold Reconstruction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-29
**Phase:** 1-Data Loading & Manifold Reconstruction
**Areas discussed:** Notebook↔phase split, Metric & normalization, n_neighbors sweep, What the cache commits to

**Gray areas offered:** all four listed above; the user selected all four.

---

## Notebook↔phase split

### Q1 — How should Phase 1's work map onto notebook files, given Phase 2's audit + gate is a separate phase?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep 3-notebook plan | P1 writes §1-3 of `01_manifold_and_gate.ipynb`; P2 appends its audit + gate to the same file. Matches ARCHITECTURE.md. Cost: P1 ships a file whose name promises a gate it does not yet contain. | ✓ |
| One notebook per phase | 4 notebooks, each phase's deliverable a complete file. Cost: an extra cache-boundary crossing research called friction without benefit. | |
| You decide | Leave to planning. | |

**User's choice:** Keep 3-notebook plan
**Notes:** Resolves the 3-notebooks-vs-4-phases mismatch in favour of the research layout.

### Q2 — How much of `notebooks/pu_manifold/` does Phase 1 build?

| Option | Description | Selected |
|--------|-------------|----------|
| Only what P1 needs | `__init__.py`, `subsample.py`, `cache.py`; later phases add their own modules. | |
| Scaffold all 4 modules | Also stub `curvature.py` and `mknn.py` so the package shape is visible from the start. Cost: empty stubs with no callers. | ✓ |
| No package, notebook-local | Helpers in notebook cells. Cost: Anti-Pattern 3 — the row-alignment invariant gets copy-pasted with drift. | |

**User's choice:** Scaffold all 4 modules

### Q3 — DATA-05: how should the dependency install cell work?

| Option | Description | Selected |
|--------|-------------|----------|
| `requirements-notebooks.txt` | Pinned versions in one file; every notebook's first cell runs `%pip install -r`. | ✓ |
| Inline `%pip` per notebook | Self-contained per notebook. Cost: three places to update a pin. | |
| Both | Shared file plus per-notebook extras. Cost: two places to check. | |

**User's choice:** `requirements-notebooks.txt`
**Notes:** Accepted side effect — notebook 01 installs `torch` even though only notebook 02 needs it.

### Q4 — How should the Python 3.11 floor be handled?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard assert + repro header | `assert sys.version_info >= (3, 11)` explaining the `pyproject.toml` drift, then print seed / versions / git SHA. | ✓ |
| Markdown note + header | State it, do not enforce. Cost: opaque pip failure on 3.10. | |
| You decide | Leave to planning. | |

**User's choice:** Hard assert + repro header

---

## Metric & normalization

### Q1 — DATA-04: which metric, and is the array L2-normalized before Isomap?

| Option | Description | Selected |
|--------|-------------|----------|
| Decide from norm histogram | Pre-state a CV threshold, plot norms, branch. Most literal reading of DATA-04. | |
| Raw Euclidean, always | Keeps ambient space flat ℝ⁷⁶⁸. Cost: magnitude can dominate the geodesic graph. | |
| L2-normalize, always | Unit-norm then Euclidean (monotone with cosine). Standard for foundation-model embeddings. Cost: manifold lives on S⁷⁶⁷, so Phase 3's ‖H‖ includes the sphere's own curvature. | ✓ |

**User's choice:** L2-normalize, always
**Notes:** Flagged at the time — Phase 3's synthetic controls (CURV-06) must now be matched on the sphere, not a flat plane. Carried into CONTEXT.md D-05.

### Q2 — Does normalization apply to both paired columns, and where?

| Option | Description | Selected |
|--------|-------------|----------|
| Both columns, at cache write | `subsample.py` normalizes both before writing the npz, and stores raw norms separately for the DATA-04 histogram. | ✓ |
| Both columns, at use site | Cache raw, normalize on load. Cost: convention-over-enforcement at every call site. | |
| Legacysurvey only | Cost: breaks MKNN-02 comparability — normalized k-NN set vs raw one. | |

**User's choice:** Both columns, at cache write

### Q3 — How is the 10k subsample drawn? (STACK.md and ARCHITECTURE.md disagree)

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit `row_indices` | `default_rng(seed).choice(...)`, sorted, applied once to both columns; indices cached as a first-class artifact. | ✓ |
| `datasets` shuffle+select | Fewer lines. Cost: reproducibility depends on library-internal permutation; no index array to assert on. | |
| You decide | Leave to planning. | |

**User's choice:** Explicit `row_indices`
**Notes:** ARCHITECTURE.md wins over STACK.md on this conflict.

### Q4 — DATA-03: what should the alignment assertion check, with no join key?

| Option | Description | Selected |
|--------|-------------|----------|
| Structural + statistical | Shapes, shared `row_indices`, sha256 re-verified on load, plus a true-pair-vs-permuted cosine smoke test. Cost: margin must not false-alarm on a 0.4-2% signal. | ✓ |
| Structural only | Proves same rows, same order — all positional alignment can mean. | |
| Shape equality only | Passes even after an independent re-sort. | |

**User's choice:** Structural + statistical

---

## n_neighbors sweep

### Q1 — ISO-01 + ISO-02: how should connectivity check and stability sweep be structured?

| Option | Description | Selected |
|--------|-------------|----------|
| Two-stage: cheap scan, then full fits | Stage 1 k-NN graph + `connected_components` across all six k (seconds); stage 2 full fits at 3-4 surviving k. | ✓ |
| Full fits across all six k | Most complete. Cost: ~6× fit time, up to ~6 GB cache. | |
| Pilot sweep on a subsample | Cheapest. Cost: k-NN density scales with n, so it answers a different question. | |

**User's choice:** Two-stage: cheap scan, then full fits

### Q2 — What criterion freezes the final `n_neighbors`?

| Option | Description | Selected |
|--------|-------------|----------|
| Smallest connected k | Conservative against short-circuits. Cost: sits at the fragmentation boundary, noisiest choice. | |
| Middle of stable plateau | Procrustes disparity + relative eigenvalue change + geodesic Spearman; pick the centre of the widest passing run. Cost: three thresholds to justify. | ✓ |
| Smallest connected + plateau check | Smallest connected k, but required to fall inside a plateau. | |

**User's choice:** Middle of stable plateau
**Notes:** Circularity flagged at the time — the eigenvalue metric needs a `d`, which Phase 2 freezes, so the sweep must use a provisional `d` from the ISO-03 `compute_dim` pre-audit.

### Q3 — What if no k gives `connected_components == 1`?

| Option | Description | Selected |
|--------|-------------|----------|
| Halt with enumerated options | Assert and stop, listing remediation choices. Mirrors Phase 2's gate philosophy. Cost: Phase 1 can end with no Isomap fit. | |
| Auto-extend k upward | Increase k until connected, report the k needed. Cost: forcing connectivity on a fragmented graph lands deep in short-circuit territory — the failure ISO-02 exists to detect. | ✓ |
| Restrict to largest component | Drop outliers, re-derive `row_indices`. Cost: n < 10,000 by automatic rule. | |

**User's choice:** Auto-extend k upward
**Notes:** The short-circuit tension was stated in the option text and the user selected it anyway. Recorded as a deliberate decision in CONTEXT.md D-11, with the mitigation from Q4 attached.

### Q4 — Should the upward extension have a ceiling?

| Option | Description | Selected |
|--------|-------------|----------|
| Ceiling, then halt | Extend to k=50; if still fragmented, halt with remediation options. Any k > 30 reported as a short-circuit-risk flag for Phase 2's gate. | ✓ |
| No ceiling | Simplest. Cost: at large k, Isomap stops being a manifold reconstruction. | |
| Ceiling, then largest component | Fall back to the largest component at the ceiling. Cost: n drops automatically. | |

**User's choice:** Ceiling, then halt

---

## What the cache commits to

### Q1 — What `n_components` does the cached fit use, given Phase 2 freezes `d`?

| Option | Description | Selected |
|--------|-------------|----------|
| Generous fit, Phase 2 re-slices | Fit at a ceiling (~30); Phase 2 takes the leading `d` columns, no refit. Cost: commits to a ceiling before the audit. | |
| Fit at the `compute_dim` estimate | Data-informed, per ARCHITECTURE.md's stated primary use of the library. Cost: a Phase 2 elbow above it forces a refit. | ✓ |
| Cache `dist_matrix_`, defer embedding | Nothing committed early. Cost: Phase 1 needs an embedding for the ISO-02 Procrustes check anyway. | |

**User's choice:** Fit at the `compute_dim` estimate

### Q2 — What exact rule turns the panel into one `n_components`, and is headroom added?

| Option | Description | Selected |
|--------|-------------|----------|
| Geometric median + headroom | `ceil(median(geometric)) + 5`, guarding against a Phase 2 refit. | |
| Geometric median, no headroom | Exactly `ceil(median(geometric))` — the library's own unpadded answer. Cost: any elbow above it forces a refit plus a new ~1 GB cache entry. | ✓ |
| Geometric max | Headroom by construction. Cost: driven by whichever estimator is least reliable. | |

**User's choice:** Geometric median, no headroom
**Notes:** Refit risk explicitly accepted. Spectral estimators excluded from the rule but still reported for the ISO-03 comparison.

### Q3 — ISO-04/ISO-05: what gets persisted, and what happens to sweep fits?

| Option | Description | Selected |
|--------|-------------|----------|
| Full joblib for k*, slim npz for sweep | Only k* pickled whole (~1 GB, carries `dist_matrix_` for Phase 2); sweep fits keep embedding + eigenvalues only. Peak ~1 GB. | ✓ |
| Full joblib for every fit | Any non-chosen k answerable later. Cost: ~5 GB. | |
| npz only, no joblib | Avoids pickle version fragility. Cost: `nbrs_` and `kernel_pca_` lost. | |

**User's choice:** Full joblib for k*, slim npz for sweep

### Q4 — What goes into the ISO-05 cache key?

| Option | Description | Selected |
|--------|-------------|----------|
| Params + library versions | Analysis params plus `sklearn`/`numpy`/`scipy` versions, so an Isomap-internals change busts the key and covers pickle reload fragility. | ✓ |
| Params only | Stable across dependency bumps. Cost: a numerics-changing sklearn upgrade reuses the old artifact. | |
| Params + versions + git SHA | Strictest. Cost: a docstring commit invalidates a 1 GB artifact. | |

**User's choice:** Params + library versions

---

## Claude's Discretion

The user selected a concrete option on every question — no "you decide" answers. Items left
open for planning (named during discussion, never locked) are listed in CONTEXT.md
§Claude's Discretion: the three plateau thresholds, the alignment smoke-test margin, the
provisional `d`, notebook output hygiene, section numbering, and the written Phase 1 → Phase 2
interface.

## Deferred Ideas

None — no scope creep was raised; discussion stayed inside DATA-01..05 / ISO-01..05.
