# Phase 1: Data Loading & Manifold Reconstruction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-29
**Phase:** 1-Data Loading & Manifold Reconstruction
**Areas discussed:** Notebook↔phase split, Metric & normalization, n_neighbors sweep, What the
cache commits to. Gray areas offered: all four; the user selected all four.

---

## Notebook↔phase split

| Q | Selected | Runner-up cost |
|---|---|---|
| How should Phase 1's work map onto notebook files? | Keep 3-notebook plan: P1 writes §1-3 of `01_manifold_and_gate.ipynb`, P2 appends its audit+gate | Rejected: one-notebook-per-phase (4 notebooks) — an extra cache-boundary crossing research called friction without benefit |
| How much of `pu_manifold/` does Phase 1 build? | Scaffold all 4 modules — stub `curvature.py`/`mknn.py` too | Rejected: notebook-local helpers — Anti-Pattern 3, the alignment invariant gets copy-pasted with drift |
| DATA-05: how should the dependency install cell work? | `requirements-notebooks.txt`, every notebook's first cell `%pip install -r`s it | Accepted side effect: notebook 01 installs `torch` though only 02 needs it |
| How should the Python 3.11 floor be handled? | Hard assert + repro header naming the `pyproject.toml` drift | Rejected: markdown note only — opaque pip failure on 3.10 |

---

## Metric & normalization

| Q | Selected | Runner-up cost |
|---|---|---|
| DATA-04: which metric, normalized before Isomap? | L2-normalize always, then Euclidean (monotone with cosine) | Flagged: manifold now lives on S⁷⁶⁷, so Phase 3's ‖H‖ includes the sphere's own curvature — carried into CONTEXT.md D-05 |
| Does normalization apply to both columns, and where? | Both columns, at cache write in `subsample.py`; raw norms stored separately for the DATA-04 histogram | Rejected: normalize at use site — convention-over-enforcement at every call site |
| How is the 10k subsample drawn (STACK.md vs ARCHITECTURE.md conflict)? | Explicit `row_indices` (`default_rng(seed).choice`, sorted, applied once), cached as a first-class artifact | ARCHITECTURE.md wins over STACK.md's shuffle+select — reproducibility can't depend on library-internal permutation |
| DATA-03: what should the alignment assertion check, no join key? | Structural + statistical (shapes/sha256 + true-pair-vs-permuted cosine smoke test) | Margin must not false-alarm on the paper's 0.4-2% signal |

---

## n_neighbors sweep

| Q | Selected | Runner-up cost |
|---|---|---|
| ISO-01+02: how structure connectivity check + stability sweep? | Two-stage: cheap k-NN+`connected_components` scan across all six k, then full fits at 3-4 surviving k | Rejected: full fits across all six k — ~6x fit time, up to ~6 GB cache |
| What criterion freezes final `n_neighbors`? | Middle of stable plateau (Procrustes disparity + relative eigenvalue change + geodesic Spearman, centre of widest passing run) | Rejected: smallest connected k — sits at the fragmentation boundary, noisiest choice |
| What if no k gives `connected_components == 1`? | Auto-extend k upward, report the k needed | Short-circuit tension stated and selected anyway — recorded as CONTEXT.md D-11 |
| Should the upward extension have a ceiling? | Ceiling then halt: extend to k=50, halt with remediation options; any k>30 is a short-circuit-risk flag | Rejected: no ceiling — at large k, Isomap stops being a manifold reconstruction |

Circularity flagged at Q2: the eigenvalue metric needs a `d`, which Phase 2 freezes, so the sweep
uses a provisional `d` from the ISO-03 `compute_dim` pre-audit.

---

## What the cache commits to

| Q | Selected | Runner-up cost |
|---|---|---|
| What `n_components` for the cached fit, given Phase 2 freezes `d`? | Fit at the `compute_dim` estimate (data-informed, per ARCHITECTURE.md) | Rejected: generous fit at a ceiling (~30) — commits to a ceiling before the audit |
| What exact rule turns the panel into one `n_components`? Headroom? | Geometric median, no headroom — `ceil(median(geometric))` exactly | Refit risk explicitly accepted; spectral estimators excluded but still reported |
| ISO-04/05: what gets persisted, sweep fits? | Full joblib for k* only (~1 GB, `dist_matrix_`), slim npz for sweep fits — peak ~1 GB | Rejected: full joblib for every fit — ~5 GB |
| What goes into the ISO-05 cache key? | Analysis params + `sklearn`/`numpy`/`scipy` versions | Rejected: params+versions+git SHA — a docstring commit would invalidate a 1 GB artifact |

---

## Claude's Discretion

The user selected a concrete option on every question. Items left open for planning (named during
discussion, never locked): listed in CONTEXT.md §Claude's Discretion — the three plateau
thresholds, alignment smoke-test margin, provisional `d`, notebook output hygiene, section
numbering, and the written Phase 1 → Phase 2 interface.

## Deferred Ideas

None — no scope creep raised; discussion stayed inside DATA-01..05 / ISO-01..05.
</content>
