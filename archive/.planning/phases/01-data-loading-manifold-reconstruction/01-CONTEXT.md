# Phase 1: Data Loading & Manifold Reconstruction - Context

**Gathered:** 2026-07-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 delivers a reproducible, row-aligned, cached 10,000-row subsample of
`UniverseTBD/pu-embeddings` config `legacysurvey_dinov3_vitb16`, plus an Isomap fit validated for
k-NN graph connectivity and `n_neighbors` (short-circuit) stability. Covers DATA-01..05 and
ISO-01..05 (10 requirements).

**Not in this phase:** the classical-MDS eigenspectrum audit, negative-eigenvalue statistic,
residual-variance elbow, freezing `d`, `gate_verdict.json` — all Phase 2 (SPEC-01..07). Phase 1
produces the cached Isomap fit Phase 2 audits.

Milestone-wide bounds: notebook-only (`src/effdim/`, `pyproject.toml` untouched); exact Isomap
only (landmark/Nyström Out of Scope — distorts the eigenspectrum Phase 2 inspects).

</domain>

<decisions>
## Implementation Decisions

| ID | Decision |
|---|---|
| D-01 | Three-notebook layout kept; Phase 1 writes §1-3 of `01_manifold_and_gate.ipynb`, Phase 2 appends. Reversibility: costly (all cache paths/doc refs written against these filenames) |
| D-02 | Scaffold all four `pu_manifold/` modules; `cache.py`+`subsample.py` implemented, `curvature.py`+`mknn.py` stubbed (package shape visible from start, filled by Phases 3-4). Never installed, never imported from `src/effdim/` |
| D-03 | Deps pinned in `requirements-notebooks.txt`, every notebook's first cell `%pip install`s it. `torch` only needed by notebook 02 but shared file installs it for 01 too — accepted |
| D-04 | Hard-assert `sys.version_info >= (3,11)` (pyproject's `>=3.8` is stale drift, LIB-03), print repro header (seed, lib versions, git SHA). DATA-05 |
| D-05 | **L2-normalize always, then Euclidean** (monotone with cosine once normalized), stated unconditionally not decided from the histogram. One-way: normalized array is the Phase 3/4 target; ambient space becomes S⁷⁶⁷ — Phase 3's CURV-06 controls must match on the sphere, not a flat plane in ℝ⁷⁶⁸ |
| D-06 | Normalize both paired columns at cache-write time; cache holds only normalized arrays + raw `hsc_norms`/`ls_norms` (DATA-04 histogram reproducible without re-streaming) — Phase 4 can't mix normalized/raw |
| D-07 | `row_indices = np.sort(default_rng(SEED).choice(n_total, 10_000, replace=False))`, one indexing pass both columns. ARCHITECTURE.md wins over STACK.md's `.shuffle().select()` — reproducibility can't depend on `datasets`'s internal permutation. DATA-02 |
| D-08 | DATA-03 assertion is structural (shapes, one indexing call, sha256 re-verified) + statistical (true-pair cosine vs permuted baseline, stated margin). Crossmodal signal is weak (0.4-2% MKNN, arXiv:2509.19453) so the margin must catch gross misalignment without false-alarming on weak-but-correct pairing |
| D-09 | Two-stage sweep (ISO-01/02): stage 1 (seconds) k-NN graph + `connected_components` across k∈{5,8,10,15,20,30}; stage 2 (minutes each) full fits at 3-4 surviving k. Rejected: six full fits (~6 GB); a pilot sweep at n=2,000 (density scales with n) |
| D-10 | `n_neighbors` frozen by a stable-plateau criterion (not smallest connected k): Procrustes disparity, relative leading-eigenvalue change, geodesic Spearman between adjacent k; `k*` = centre of widest all-passing run. One-way: baked into the cached artifact/cache key. Circularity: metric 2 needs a provisional `d` from the ISO-03 `compute_dim` pre-audit |
| D-11 | If no base k connects, auto-extend (40, then 50), ceiling 50; at ceiling halt with 3 remediation options. `k*` above 30 is a loud short-circuit-risk flag Phase 2's gate must weigh — auto-extend chosen over halt-first, bounded ceiling over unbounded |
| D-12 | `n_components = ceil(median(...))` over geometric/intrinsic estimators only (TwoNN/MLE/ESS/TLE/GMST/DANCo/MiND-ML*), no headroom; spectral estimators excluded from the rule but still reported. Accepted risk: a Phase 2 elbow above this forces a re-fit — unpadded data-informed dimension chosen over a headroom constant |
| D-14 | ISO-05 cache key = sha256 over `dataset`/`seed`/`n_rows`/`normalize`/`n_neighbors`/`n_components`/`eigen_solver`/`sklearn`/`numpy`/`scipy` versions. Git SHA deliberately excluded (a docstring commit must not invalidate a 1 GB artifact) |
| D-15 | `eigen_solver="dense"` pinned explicitly: `Isomap` has no `random_state`; `"auto"`/`"arpack"` use ARPACK with a random start vector. At n=10,000 `dense` is deterministic LAPACK, feasible as a one-time cached cost |

**D-13 persisted artifacts:**

| File | Contents | Approx size |
|---|---|---|
| `subsample_{seed}.npz` | normalized `hsc`/`legacysurvey`, `hsc_norms`, `ls_norms`, `row_indices` | ~60 MB |
| `effdim_panel_{seed}.json` | full `compute_dim` panel on normalized legacysurvey | <1 KB |
| `isomap_{key}.joblib` | fitted `Isomap` at `k*` only — `dist_matrix_`/`embedding_`/`nbrs_`/`kernel_pca_` | ~1 GB |
| `sweep_k{K}_{key}.npz` | per swept k: embedding, eigenvalues, component count, timing | ~1 MB each |

Only `k*` gets the full joblib (Phase 2 needs `dist_matrix_`). Peak cache ~1 GB not ~5 GB.
Gitignored under `notebooks/.cache/`.

**Claude's Discretion** (resolved by plans 01-04): the three D-10 plateau thresholds (pre-stated,
not tuned after); the D-08 margin; the provisional `d` for D-10; output-hygiene policy;
section-numbering convention; the Phase 1->2 handoff interface.

</decisions>

<canonical_refs>
## Canonical References

**Requirements/scope:** `REQUIREMENTS.md` §Data Loading/Manifold Reconstruction, §Out of Scope;
`ROADMAP.md` §Phase 1; `PROJECT.md` §Key Decisions, §PU embeddings dataset (row-aligned, no join
key, no labels — D-08's basis).

**Implementation:** `research/ARCHITECTURE.md` (Project Structure for D-01/D-02, Pattern 1 for
D-14, Caching Strategy for D-13, Integration Points for D-12, Determinism for D-15, Anti-Patterns
1/3 for Restart&RunAll/no-independent-column-sampling); `research/STACK.md` (pins for D-03, floor
for D-04; shuffle/select recipe superseded by D-07); `research/PITFALLS.md` (1-2 bridging/
short-circuit are this phase's, 3 truncated `kernel_pca_.eigenvalues_` constrains caching);
`research/FEATURES.md` (P1 six-value sweep range); `research/SUMMARY.md` (synthesis).

**External:** arXiv:2509.19453 (origin paper, MKNN definition, published crossmodal ranges — D-08's
margin calibrated against its 0.4-2% Legacy Survey number).
`https://huggingface.co/datasets/UniverseTBD/pu-embeddings` config `legacysurvey_dinov3_vitb16`,
101,725 rows, ~553 MiB parquet.

</canonical_refs>

<code_context>
## Existing Code Insights

No `.planning/codebase/` maps exist; from a direct scan. `src/effdim/api.py` `compute_dim(...)`
called read-only for the ISO-03 pre-audit/D-12; `_do_svd` (`api.py:144`) uses full SVD when
`min(n,d)<1000` — at 10,000x768 this phase takes the full-SVD path, not randomized.
`src/effdim/geometry.py` geometric/intrinsic estimators (D-12's median); `metrics.py` spectral
estimators, reported but excluded from the rule. `faiss-cpu` already core, available for k-NN.
Core deps stay light; every new dep is notebook-scoped, `%pip`-installed. `.gitignore` already
covers `notebooks/.cache/`. `notebooks/` empty — Phase 1 builds from zero, D-01..D-04 conventions
inherited by Phases 2-4. Integration: notebook -> `effdim` (read-only), notebook -> `pu_manifold/`
(never installed); Phase 1 -> Phase 2 via the same notebook file + D-13's cache artifacts.
`pyproject.toml`/`src/effdim/` not modified.

</code_context>

<specifics>
## Specific Ideas

DATA-03 assertion: equal shapes -> sha256 match -> true-pair cosine vs permuted baseline clears a
stated margin. Connectivity scan cheap enough for the full k range before any fit:
`kneighbors_graph`+`connected_components`, tabulate, then spend the fits. k-ceiling as an explicit
`for/else`: break on connected (warn if k>30), `assert False` in `else` with the three remediation
options. Cache key built from `json.dumps(..., sort_keys=True)` before hashing.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed in scope. Two downstream consequences recorded inline above, must not be
lost: Phase 3 (CURV-06) must match synthetic controls on S⁷⁶⁷, not a flat plane in ℝ⁷⁶⁸ (D-05);
Phase 2 must treat "elbow exceeds the Phase 1 fit dimension -> re-fit" as a real branch (D-12), and
a `k*` above 30 arrives carrying a short-circuit-risk flag the gate must weigh (D-11).

</deferred>

---
*Phase: 1-Data Loading & Manifold Reconstruction* · *Context gathered: 2026-07-29*
</content>
