# Phase 1: Data Loading & Manifold Reconstruction - Context

**Gathered:** 2026-07-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 delivers a reproducible, row-aligned, cached 10,000-row subsample of
`UniverseTBD/pu-embeddings` config `legacysurvey_dinov3_vitb16`, plus an Isomap fit on it
that has been validated for k-NN graph connectivity and `n_neighbors` (short-circuit)
stability. It covers DATA-01..05 and ISO-01..05 (10 requirements).

**Explicitly NOT in this phase:** the classical-MDS eigenspectrum audit, the negative-
eigenvalue statistic, the residual-variance elbow, freezing `d`, and the
PASS/MARGINAL/FAIL `gate_verdict.json`. All of that is Phase 2 (SPEC-01..07). Phase 1
produces the cached Isomap fit that Phase 2 audits.

Milestone-wide constraints that bound this phase: notebook-only work — `src/effdim/` and
`pyproject.toml` are not modified; exact Isomap only (landmark/Nyström is Out of Scope
because it distorts the very eigenspectrum Phase 2 inspects).

</domain>

<decisions>
## Implementation Decisions

### Notebook & package structure

- **D-01:** Keep ARCHITECTURE.md's three-notebook layout. Phase 1 writes sections 1-3 of
  `notebooks/01_manifold_and_gate.ipynb`; **Phase 2 appends its audit and gate sections to
  that same file.** The 4-phase roadmap does not become a 4-notebook layout.
  — **Reversibility:** costly — every cache path, cross-notebook gate check, and doc
  reference is written against these three filenames; renaming later touches all of them
  plus any committed notebook outputs.

- **D-02:** Phase 1 scaffolds all four `notebooks/pu_manifold/` modules, not just the ones
  it uses. `__init__.py`, `subsample.py` (seeded subsample + row-alignment asserts) and
  `cache.py` (config-hash npz/joblib helpers) are **implemented**; `curvature.py` and
  `mknn.py` are **stubbed** so the package shape is visible from the start. Phases 3 and 4
  fill in the stubs. The package is never installed and never imported from `src/effdim/`.

- **D-03:** Notebook dependencies are pinned in `notebooks/requirements-notebooks.txt`; the
  first cell of every notebook runs `%pip install -r requirements-notebooks.txt`. One
  diffable place for versions across all three notebooks.
  *Planner note:* `torch` is only needed by notebook 02, but the shared file installs it for
  notebook 01 too. That is accepted — do not split the file to avoid it.

- **D-04:** The notebook hard-asserts `sys.version_info >= (3, 11)` with a message
  explaining that `pyproject.toml`'s `>=3.8` is stale core-dep drift (tracked as LIB-03),
  then prints a reproducibility header: seed(s), `sklearn`/`numpy`/`scipy`/`faiss` versions,
  and the git commit SHA. A reader on 3.10 gets a clear failure, not an opaque pip
  resolution error. (DATA-05)

### Metric & normalization

- **D-05:** **L2-normalize always**, then Euclidean (`Isomap` default `minkowski, p=2`,
  which is monotone with cosine once normalized, so k-NN sets match cosine exactly). This
  is the DATA-04 answer and it is stated unconditionally — not decided at runtime from the
  norm histogram. The histogram is still shown, as the justification.
  — **Reversibility:** one-way — the normalized array is the 768-d reconstruction target
  for the Phase 3 decoder and the input to Phase 4's MKNN. Changing it after Phase 2
  invalidates the gate verdict, the decoder, the curvature field, and every MKNN number,
  and forces a full re-fit of the ~1 GB Isomap artifact.
  **Downstream consequence the planner must carry forward:** the ambient space is now the
  unit sphere S⁷⁶⁷, which is itself intrinsically curved. Phase 3's synthetic-control
  falsification manifolds (CURV-06) must therefore be matched **on the sphere**, not against
  a flat plane in ℝ⁷⁶⁸, or the control compares against the wrong baseline.

- **D-06:** Normalization applies to **both** paired columns, performed in `subsample.py`
  **at cache-write time**. `subsample_{seed}.npz` stores normalized `hsc` and
  `legacysurvey`, plus `hsc_norms` and `ls_norms` (the raw norms) so the DATA-04 histogram
  is reproducible without re-streaming, plus `row_indices`. Because the cache holds only
  normalized arrays, Phase 4 cannot accidentally compare a normalized embedding against a
  raw one.

- **D-07:** The subsample is drawn with an explicit index array —
  `row_indices = np.sort(np.random.default_rng(SEED).choice(n_total, 10_000, replace=False))`
  — applied once to both columns. **This resolves a direct conflict in the research files:
  ARCHITECTURE.md's explicit-`row_indices` approach wins over STACK.md's
  `.shuffle(seed).select(range(10000))`.** Rationale: reproducibility must not depend on the
  `datasets` library's internal permutation staying stable across versions, and DATA-03
  needs a concrete index array to assert on and cache. (DATA-02)

- **D-08:** The DATA-03 row-alignment assertion is **structural + statistical**, because the
  dataset ships no `object_id` and no join key — positional order is the only alignment
  that exists.
  - Structural: equal shapes; both arrays produced by one indexing call on one
    `row_indices`; sha256 of `row_indices` recorded and re-verified on every cache load.
  - Statistical smoke test: mean per-row cosine similarity of true pairs must exceed the
    permuted-pair baseline by a stated margin — catches a gross off-by-one or independent
    re-sort that shape checks cannot see.
  *Planner note:* the crossmodal signal is weak (0.4-2% MKNN in the origin paper), so the
  margin must be chosen to catch gross misalignment without false-alarming on a genuinely
  weak-but-correct pairing. Pick and justify the margin explicitly.

### n_neighbors sweep (ISO-01, ISO-02)

- **D-09:** Two-stage sweep. **Stage 1 (seconds):** build only the k-NN graph and run
  `scipy.sparse.csgraph.connected_components` across k ∈ {5, 8, 10, 15, 20, 30}, answering
  ISO-01 for every k and showing exactly where the graph stops fragmenting. **Stage 2
  (minutes each):** full `Isomap` fits at 3-4 surviving k values for the ISO-02
  embedding/eigenspectrum stability comparison. Expensive fits are spent only where they
  buy the ISO-02 answer. Rejected: six full fits (~6 GB), and a pilot sweep at n=2,000
  (k-NN density scales with n, so the pilot answers a different question).

- **D-10:** `n_neighbors` is frozen by a **stable-plateau criterion**, not by taking the
  smallest connected k. Three quantitative stability metrics are computed between adjacent
  swept k values, and `k*` is the centre of the widest run where all three pass stated
  thresholds:
  1. Procrustes disparity between embeddings (handles the inherent sign/rotation ambiguity)
  2. relative change in the leading eigenvalues
  3. Spearman correlation of the flattened geodesic distance matrices
  — **Reversibility:** one-way — `k*` is baked into the cached ~1 GB Isomap artifact and
  into the cache key that Phases 2-4 all load from. Changing it invalidates the gate
  verdict, the decoder, and every downstream MKNN number.
  **Circularity the planner must resolve:** metric 2 needs a `d` to define "leading"
  eigenvalues, but `d` is not frozen until Phase 2. The sweep must use a **provisional `d`**
  taken from the ISO-03 `compute_dim` pre-audit, and say so explicitly in the notebook.

- **D-11:** If no k in {5, 8, 10, 15, 20, 30} yields `connected_components == 1`, the
  notebook **auto-extends k upward** (e.g. 40, then 50) until connected, **bounded by a
  stated ceiling of k=50**. At the ceiling it halts with remediation options enumerated
  (widen the range, resample with a new seed, or restrict to the largest component and
  re-derive `row_indices`). Any `k*` found above 30 is reported prominently as a
  **short-circuit-risk flag that Phase 2's gate must weigh**.
  *Tension recorded deliberately, user decision:* auto-extending k to force connectivity
  moves toward the short-circuit regime that ISO-02 exists to detect. The user chose
  auto-extend over halting immediately, and chose a bounded ceiling over unbounded
  extension. Do not silently swap in the halt-first behaviour — but do make the
  above-30 warning loud, and carry it into whatever Phase 1 hands Phase 2.

### Cache & artifacts (ISO-03, ISO-04, ISO-05)

- **D-12:** `n_components` for the cached fit comes from the ISO-03 `compute_dim`
  pre-audit: `ceil(median(...))` over the **geometric/intrinsic estimators only** (TwoNN,
  MLE, ESS, TLE, GMST, DANCo, MiND-ML*), **with no headroom added**. Spectral estimators are
  excluded from the selection rule because they are ambient-linear and inflate under
  curvature (ARCHITECTURE.md §Integration Points) — but the **full** panel, spectral and
  geometric, is still reported for the ISO-03 comparison against what the Isomap
  eigenspectrum suggests.
  **Accepted risk, stated explicitly:** if Phase 2's residual-variance elbow lands above
  this `n_components`, the Isomap fit must be **re-fit** — re-paying minutes of compute and
  a new ~1 GB cache entry. The user chose the unpadded, data-informed dimension over adding
  a headroom constant. Phase 2 planning should treat "elbow exceeds the Phase 1 fit
  dimension" as a real branch, not an edge case.

- **D-13:** Persisted artifacts:

  | File | Contents | Approx size |
  |---|---|---|
  | `subsample_{seed}.npz` | normalized `hsc`, normalized `legacysurvey`, `hsc_norms`, `ls_norms`, `row_indices` | ~60 MB |
  | `effdim_panel_{seed}.json` | full `compute_dim` panel on the normalized legacysurvey array | <1 KB |
  | `isomap_{key}.joblib` | the fitted `Isomap` object at `k*` only — carries `dist_matrix_`, `embedding_`, `nbrs_`, `kernel_pca_` | ~1 GB |
  | `sweep_k{K}_{key}.npz` | per swept k: embedding, eigenvalues, component count, timing | ~1 MB each |

  Only `k*` gets the full joblib pickle, because Phase 2 needs `dist_matrix_` for its
  hand-rolled full spectrum. Sweep fits persist slim npz only, so the ISO-02 stability table
  stays reproducible without carrying 4× the geodesic matrices. Peak cache ~1 GB, not ~5 GB.
  All under `notebooks/.cache/`, already gitignored.

- **D-14:** The ISO-05 cache key is `sha256` over analysis parameters **plus library
  versions**: `dataset` config name, `seed`, `n_rows`, `normalize` flag, `n_neighbors`,
  `n_components`, `eigen_solver`, and `sklearn`/`numpy`/`scipy` versions. A sklearn upgrade
  that changes Isomap internals therefore produces a new key instead of silently reusing an
  incompatible pickle — which also covers joblib reload fragility. The git SHA is
  deliberately **not** in the key (a docstring commit must not invalidate a 1 GB artifact).

- **D-15:** `eigen_solver="dense"` is pinned explicitly (carried forward from research, not
  re-litigated): `Isomap` has no `random_state`, and `"auto"`/`"arpack"` uses ARPACK with a
  random start vector. At n=10,000 `dense` is fully deterministic LAPACK and feasible as a
  one-time cached cost.

### Claude's Discretion

The user selected a concrete option on every question — no "you decide" answers. The
following are left to planning because they were named as open sub-decisions and never
locked:

- The three numeric thresholds for the D-10 plateau criterion (Procrustes disparity,
  relative eigenvalue change, geodesic Spearman). **Given the milestone's explicit
  garden-of-forking-paths concern, these should be stated in the notebook before the sweep
  runs, not tuned after seeing the results.**
- The D-08 statistical smoke-test margin.
- The provisional `d` used for the D-10 leading-eigenvalue metric (derived from D-12's
  `compute_dim` panel — the planner picks how).
- Notebook output-stripping / cell-output hygiene policy.
- Section numbering convention inside `01_manifold_and_gate.ipynb` so Phase 2's appended
  sections slot in cleanly.
- The concrete written interface Phase 1 hands Phase 2 (which cache keys, which flags,
  where the above-30 `k*` warning surfaces).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

ROADMAP.md carries no `Canonical refs:` line for this phase; the list below was accumulated
from REQUIREMENTS.md, PROJECT.md, and the `.planning/research/` set.

### Requirements & scope (read first)
- `.planning/REQUIREMENTS.md` §Data Loading (DATA), §Manifold Reconstruction (ISO) — the 10
  requirements this phase must satisfy, verbatim
- `.planning/REQUIREMENTS.md` §Out of Scope — the exclusion table (landmark Isomap, fixed
  absolute thresholds, alternative alignment metrics) that bounds this phase
- `.planning/ROADMAP.md` §Phase 1 — goal and the five success criteria
- `.planning/PROJECT.md` §Key Decisions — the locked milestone-level decisions carried
  forward here (config choice, 10k, notebook-only, single model)
- `.planning/PROJECT.md` §PU embeddings dataset — config families, row counts, and the
  "row-aligned, no join key, no labels" fact that D-08 is built around

### Implementation guidance
- `.planning/research/ARCHITECTURE.md` §Recommended Project Structure — the three-notebook
  + `pu_manifold/` layout D-01 and D-02 adopt
- `.planning/research/ARCHITECTURE.md` §Architectural Patterns Pattern 1 — config-hash-keyed
  checkpointing, the basis for D-14
- `.planning/research/ARCHITECTURE.md` §Caching Strategy — artifact table and sizes, refined
  by D-13
- `.planning/research/ARCHITECTURE.md` §Integration Points — why the **geometric** estimators
  (not spectral) drive `n_components` in D-12, and the honest caveat about comparing the
  panel against the eigenspectrum
- `.planning/research/ARCHITECTURE.md` §Determinism and Reproducibility — the
  `eigen_solver="dense"` rationale (D-15) and the axis sign/rotation ambiguity note
- `.planning/research/ARCHITECTURE.md` §Anti-Patterns 1 and 3 — "Restart & Run All" as the
  iteration loop; sampling `hsc` and `legacysurvey` independently
- `.planning/research/STACK.md` — pinned versions for `requirements-notebooks.txt` (D-03)
  and the Python 3.11 floor (D-04). **Note:** its `.shuffle(seed).select(...)` subsample
  recipe is superseded by D-07.
- `.planning/research/PITFALLS.md` — pitfalls 1-2 (silent graph bridging, short-circuit
  edges) are this phase's; pitfall 3 (truncated `kernel_pca_.eigenvalues_`) belongs to
  Phase 2 but constrains what Phase 1 must cache
- `.planning/research/FEATURES.md` — P1 table-stakes list, including the six-value
  `n_neighbors` sweep range D-09 adopts for stage 1
- `.planning/research/SUMMARY.md` — cross-cutting synthesis and the confidence/gaps
  assessment

### External
- arXiv:2509.19453 — Duraphe, Smith, Sourav & Wu, *The Platonic Universe: Do Foundation
  Models See the Same Sky?* Origin paper; MKNN definition and the published crossmodal
  ranges. Not needed to implement Phase 1, but D-08's smoke-test margin has to be calibrated
  against its 0.4-2% Legacy Survey number.
- `https://huggingface.co/datasets/UniverseTBD/pu-embeddings` — config
  `legacysurvey_dinov3_vitb16`, 101,725 rows, columns `dinov3_vitb16_hsc` /
  `dinov3_vitb16_legacysurvey`, ~553 MiB parquet

</canonical_refs>

<code_context>
## Existing Code Insights

No `.planning/codebase/` maps exist; this section comes from a direct scan.

### Reusable Assets
- `src/effdim/api.py` — `compute_dim(data: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]`.
  Called **read-only** by notebook 01 for the ISO-03 pre-audit and to drive `n_components`
  (D-12). Validates 2-D, `n >= 2`, finite; switches to randomized SVD at `min(n, d) >= 1000`,
  so the 10,000×768 call takes that path.
- `src/effdim/geometry.py` — the geometric/intrinsic estimators (TwoNN, MLE, ESS, TLE, GMST,
  DANCo, MiND) whose median D-12 uses. The planner must read this to get the exact result-dict
  key names before writing the selection rule.
- `src/effdim/metrics.py` — spectral estimators. Reported in the ISO-03 comparison but
  deliberately excluded from the `n_components` rule.
- `faiss-cpu` is already a core dependency — available for k-NN work without adding a
  notebook dep.

### Established Patterns
- Core deps stay light (numpy, scipy, scikit-learn, faiss-cpu). Every new dependency in this
  phase is notebook-scoped, installed via `%pip`, never added to `pyproject.toml`.
- `.gitignore` already covers `notebooks/.cache/`, `.planning/research/.cache/`, and
  `.ipynb_checkpoints/` — no gitignore change needed for D-13's artifacts.
- `notebooks/` exists and is **empty**. Phase 1 builds the entire directory from zero, so
  there is no in-repo notebook convention to match — the conventions D-01..D-04 establish
  are the ones Phases 2-4 inherit.

### Integration Points
- `notebooks/01_manifold_and_gate.ipynb` → `import effdim` (installed package, read-only)
- `notebooks/01_manifold_and_gate.ipynb` → `notebooks/pu_manifold/` (plain import, never
  installed, never referenced from `src/effdim/`)
- Phase 1 → Phase 2: same notebook file (D-01) **and** the `notebooks/.cache/` artifacts in
  D-13. Phase 2 reads `isomap_{key}.joblib` for `dist_matrix_`.
- `pyproject.toml` and `src/effdim/` are **not modified** by this phase.

</code_context>

<specifics>
## Specific Ideas

- The DATA-03 assertion should look roughly like: assert equal shapes → assert
  `sha256(row_indices)` matches the cached hash → compute mean true-pair cosine similarity
  against a permuted-pair baseline and assert it clears a stated margin.
- The connectivity scan is deliberately cheap enough to cover the full k range before any
  Isomap is fitted: build `kneighbors_graph(X, k, mode="distance")`, call
  `connected_components`, tabulate. Only then spend the fits.
- The k-ceiling branch should read as an explicit `for/else`: iterate k, break on connected
  (warning loudly if `k > 30`), and `assert False` in the `else` with the three remediation
  options spelled out in the message — the same "documented halt is a legitimate outcome"
  posture as Phase 2's gate.
- The cache key is built from a plain dict serialized with `json.dumps(..., sort_keys=True)`
  before hashing, so key stability does not depend on dict insertion order.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. No scope creep was raised.

Two items surfaced as **downstream consequences** rather than deferred ideas; they are
already recorded inline above and must not be lost:

- Phase 3 (CURV-06): because of D-05, the synthetic-control manifolds must be matched on the
  unit sphere S⁷⁶⁷, not against a flat plane in ℝ⁷⁶⁸.
- Phase 2: because of D-12 (no headroom), "residual-variance elbow exceeds the Phase 1 fit
  dimension → re-fit Isomap" is a real branch Phase 2 planning must handle. Because of D-11,
  a `k*` above 30 arrives at Phase 2 carrying a short-circuit-risk flag the gate must weigh.

</deferred>

---

*Phase: 1-Data Loading & Manifold Reconstruction*
*Context gathered: 2026-07-29*
