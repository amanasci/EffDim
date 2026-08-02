# Phase 1: Data Loading & Manifold Reconstruction - Pattern Map

**Mapped:** 2026-07-29
**Files analyzed:** 7 (1 notebook + 6 package/config files)
**Analogs found:** 3 / 7 (all from `src/effdim/`; `notebooks/` was empty — no in-repo notebook convention existed)

**Scope note:** `notebooks/` did not exist yet at mapping time. Every code-style analog below comes
from `src/effdim/` (imports, docstrings, typing, error handling) — not notebook structure, which
CONTEXT.md's D-01..D-04 establish from scratch. Cache/config-hash patterns sourced from
ARCHITECTURE.md's Pattern 1 worked example, since no cache module existed in the repo either.

## File Classification

| New/Modified File | Role | Closest Analog | Match Quality |
|---|---|---|---|
| `notebooks/01_manifold_and_gate.ipynb` (§1-3) | notebook/orchestration | none in-repo | new convention, D-01..D-04 define it |
| `notebooks/pu_manifold/__init__.py` | package init | `src/effdim/__init__.py` | style-only |
| `notebooks/pu_manifold/subsample.py` | utility (data loading+validation) | `src/effdim/geometry.py` (style) + ARCHITECTURE.md Pattern 4 (row-alignment invariant) | role-match (style), no in-repo I/O analog |
| `notebooks/pu_manifold/cache.py` | utility (config-hash persistence) | ARCHITECTURE.md §Architectural Patterns Pattern 1 example | research-authored reference, no in-repo analog |
| `notebooks/pu_manifold/curvature.py`/`mknn.py` (stubs) | utility (stubs, Phase 3/4 fill) | `src/effdim/metrics.py` (stub-worthy shape/docstring style) | style-only |
| `notebooks/requirements-notebooks.txt` | config | `pyproject.toml` `[project] dependencies` | role-match, different format (plain pip-freeze, not TOML) |

## Pattern Assignments

**`01_manifold_and_gate.ipynb`:** no analog. Structure fully specified by CONTEXT.md D-01..D-15 and
ARCHITECTURE.md §Recommended Project Structure/§Suggested Build Order. Key anchors: first cell
`assert sys.version_info >= (3, 11)` (D-04) then a repro header (seed, lib versions, git SHA);
second cell `%pip install -r requirements-notebooks.txt` (D-03); import `effdim` (read-only) and
`pu_manifold` (plain relative import); `effdim.compute_dim(legacysurvey)` is the ISO-03 pre-audit
on the **normalized** array (D-05/D-06), feeding D-12's median-of-geometric-keys rule. The
gate/verdict machinery belongs to Phase 2 — do not implement it here.

**`subsample.py`:** style analog `src/effdim/geometry.py` (imports/typing/docstrings); row-alignment
invariant from ARCHITECTURE.md §Data Flow "Where alignment can silently break" and Pattern 4.
Convention: `src/effdim/geometry.py` uses short imperative docstrings, `metrics.py` uses the fuller
Parameters/Returns block — new `pu_manifold` modules should pick the fuller style since these
functions are the shared-invariant surface. Validation style follows `api.py:50-56`
(`raise ValueError(f"...")`, not bare `assert`) for structural checks (D-08); CONTEXT.md's
`<specifics>` calls for `assert` in the connectivity-sweep `for/else` remediation branch — mixed
deliberately (ValueError for reusable validation, assert for notebook-cell-local invariants,
matching ARCHITECTURE.md's own guard-cell example). Row-alignment invariant: `row_indices` selected
once (D-07), both `hsc`/`legacysurvey` read off the *same* array in one indexing pass — never two
independent `.shuffle().select()` calls (the literal Anti-Pattern 3); normalized at cache-write
time (D-06) into `subsample_{seed}.npz` alongside raw norms and `row_indices`.

**`cache.py`:** no in-repo analog — used ARCHITECTURE.md's own worked config-hash-cache example
verbatim as the starting shape, adapted for D-13's artifact list and D-14's key composition (must
include library versions, must NOT include git SHA). Required deviations: `config_key`'s cfg dict
must include `dataset`/`seed`/`n_rows`/`normalize`/`n_neighbors`/`n_components`/`eigen_solver`/lib
versions, excluding git SHA; needs a `joblib`-backed variant alongside `npz_cache` for
`isomap_{key}.joblib` (only `k*` gets the full pickle). `.gitignore` already covered
`notebooks/.cache/` — no change needed.

**`curvature.py`/`mknn.py` stubs:** analog `src/effdim/metrics.py` — full NumPy-style docstring
even for small functions, explicit Parameters/Returns, defensive zero-division guards. D-02
requires these stubbed not empty ("package shape visible from the start"): real signatures Phase
3/4 will implement, docstring in `metrics.py`'s style, body `raise NotImplementedError(...)` naming
the phase — not silent `pass`, so an accidental early call fails loudly.

**`requirements-notebooks.txt`:** analog `pyproject.toml`'s `[project] dependencies` for what's
core vs. notebook-scoped. `numpy`/`scipy`/`scikit-learn`/`faiss-cpu` already core deps —
`requirements-notebooks.txt` pins `torch`, `datasets`, `matplotlib`, and other notebook-only libs
per D-03. Format is plain pip-requirements, deliberately not TOML — do not fold into
`[project.optional-dependencies]`.

## Shared Patterns

**Result-dict key names for D-12's `n_components` rule (CRITICAL — exact strings), from
`src/effdim/api.py:75-123`.** Geometric/intrinsic estimators INCLUDED in the `ceil(median(...))`
rule: `mle_dimensionality`, `two_nn_dimensionality`, `danco_dimensionality`,
`mind_mli_dimensionality`, `mind_mlk_dimensionality`, `ess_dimensionality`, `tle_dimensionality`,
`gmst_dimensionality`. CONTEXT.md's D-12 prose lists "TwoNN, MLE, ESS, TLE, GMST, DANCo, MiND-ML*"
— `MiND-ML*` maps to **two** distinct keys, so the median is over **8 values**, not 7; the planner
must use this literal 8-key list, not the paraphrased 7-name prose. Spectral estimators EXCLUDED
from the rule, reported for ISO-03 only (`api.py:75-88`): `pca_explained_variance_95`,
`participation_ratio`, `shannon_entropy`, `stable_rank`, `numerical_rank`,
`cumulative_eigenvalue_ratio`, `renyi_eff_dimensionality_alpha_2/_3/_4/_5`,
`geometric_mean_eff_dimensionality`.

**`compute_dim` signature/validation** (`api.py:29-60`): accepts `np.ndarray` or `List[np.ndarray]`
(vstacked); raises `ValueError` on empty list, wrong type, `ndim != 2`, `n_samples < 2`, non-finite.
Centers internally (`_ensure_centered`, tol `1e-5`) — notebook does not pre-center. Randomized-SVD
switch (`api.py:143-147`): full SVD when `min(n,d) < 1000`, else randomized. At `(10000, 768)`,
`min=768 < 1000`, so **full SVD is used, not randomized** — correcting CONTEXT.md's
`<code_context>` note, which is true in general but not for this phase's actual input. Also
computes k-NN distances once via FAISS internally, shared across geometric estimators — not
something the notebook replicates.

**Docstring/typing conventions:** type-annotate all public signatures; use full NumPy-style
Parameters/Returns blocks (metrics.py style) for `subsample.py`/`cache.py` public functions, since
they encode shared invariants. Defensive returns over exceptions for numeric edge cases inside pure
math helpers, but `raise ValueError` for input-contract violations — apply the latter to
`subsample.py`'s structural checks.

**Error handling — no repo-wide try/except convention:** none of `api.py`/`geometry.py`/`metrics.py`
use `try/except` — all upfront `if ...: raise ValueError(...)`. `pu_manifold` follows this:
validate eagerly and raise, don't wrap computation in try/except.

## No Analog Found / Metadata

`01_manifold_and_gate.ipynb` (structure entirely from CONTEXT.md D-01..D-15/ARCHITECTURE.md);
`cache.py` (no config-hash utility anywhere in `src/effdim/`; used Pattern 1 example);
`requirements-notebooks.txt` (no prior notebook-scoped requirements; `pyproject.toml` deps closest
reference). **Search scope:** `src/effdim/`, `notebooks/` (confirmed empty), `pyproject.toml`,
`.gitignore`. **Files scanned:** `__init__.py`, `api.py`, `geometry.py`/`metrics.py` (partial),
`pyproject.toml`, `.gitignore`. **Pattern extraction date:** 2026-07-29
</content>
