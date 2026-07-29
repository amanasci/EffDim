# Phase 1: Data Loading & Manifold Reconstruction - Pattern Map

**Mapped:** 2026-07-29
**Files analyzed:** 7 (1 notebook + 6 package/config files)
**Analogs found:** 3 / 7 (all from `src/effdim/`; `notebooks/` is empty — no in-repo notebook convention exists)

**Important scope note:** `notebooks/` does not exist yet (confirmed: `ls notebooks/` returns empty). There is
no in-repo notebook, `pu_manifold`-style helper package, or requirements-pinning convention to copy. Every
analog below comes from `src/effdim/` (the sibling installable package) purely for **code style** (imports,
docstrings, typing, error handling) — not for notebook structure, which D-01..D-04 in CONTEXT.md establish
from scratch. Cache/config-hash patterns are sourced from ARCHITECTURE.md's Pattern 1 (research-authored
example code), since no cache module exists in the repo yet either.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `notebooks/01_manifold_and_gate.ipynb` (sections 1-3) | notebook / orchestration script | batch (stream → subsample → fit → cache) | none in-repo | no analog — new convention, D-01..D-04 define it |
| `notebooks/pu_manifold/__init__.py` | package init | n/a | `src/effdim/__init__.py` | style-only |
| `notebooks/pu_manifold/subsample.py` | utility (data loading + validation) | batch / file-I/O | `src/effdim/geometry.py` (function style, typing, docstrings) + ARCHITECTURE.md Pattern 4 (row-alignment invariant) | role-match (style), no in-repo I/O analog |
| `notebooks/pu_manifold/cache.py` | utility (config-hash keyed persistence) | file-I/O | ARCHITECTURE.md §Architectural Patterns Pattern 1 (`cache.py` example, lines 96-116 of ARCHITECTURE.md) | research-authored reference, no in-repo analog |
| `notebooks/pu_manifold/curvature.py` (stub) | utility (stub, Phase 3 fills) | transform | `src/effdim/metrics.py` (stub-worthy function shape/docstring style) | style-only |
| `notebooks/pu_manifold/mknn.py` (stub) | utility (stub, Phase 4 fills) | transform | `src/effdim/metrics.py` (stub-worthy function shape/docstring style) | style-only |
| `notebooks/requirements-notebooks.txt` | config | n/a | `pyproject.toml` `[project] dependencies` (lines 18-23) | role-match, different format (plain pip-freeze-style, not TOML) |

## Pattern Assignments

### `notebooks/01_manifold_and_gate.ipynb` (notebook, batch)

**No analog** — `notebooks/` is empty. Structure is fully specified by CONTEXT.md D-01..D-15 and
ARCHITECTURE.md §Recommended Project Structure / §Suggested Build Order. Key concrete anchors the planner
must honor, not invented from an analog:

- First cell: `assert sys.version_info >= (3, 11), "..."` (D-04) then a reproducibility header printing
  seed(s), `sklearn`/`numpy`/`scipy`/`faiss` versions, and git SHA (`subprocess` or `!git rev-parse --short HEAD`).
- Second cell: `%pip install -r requirements-notebooks.txt` (D-03).
- Import `effdim` (installed, read-only) and `notebooks/pu_manifold` (plain relative import, not installed).
- `effdim.compute_dim(legacysurvey)` is the ISO-03 pre-audit — called with the **normalized** legacysurvey
  array (D-05/D-06), and its result dict is what D-12's median-of-geometric-keys selection rule operates on.
- The gate/verdict machinery (`gate_verdict.json`, negative-eigenvalue audit) belongs to **Phase 2**, not
  this file's sections 1-3 — do not implement it here (CONTEXT.md domain boundary, line 14-17).

---

### `notebooks/pu_manifold/subsample.py` (utility, batch/file-I/O)

**Analog for code style:** `src/effdim/geometry.py` (imports, typing, docstring, validation style) —
lines 1-32 below. **Analog for the row-alignment invariant:** ARCHITECTURE.md §Data Flow "Where alignment
can silently break" (lines 172-178) and §Architectural Patterns Pattern 4 (lines 151-155).

**Imports/typing/docstring style** (`src/effdim/geometry.py:1-13`):
```python
import numpy as np
import faiss
from typing import Optional
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from sklearn.neighbors import kneighbors_graph

def compute_knn_distances(data: np.ndarray, k: int) -> np.ndarray:
    """
    Compute k nearest neighbors distances for each point in data.
    Returns squared distances.
    Excludes the point itself (distance 0).
    """
```
Convention: short imperative docstrings (no full NumPy-style Parameters/Returns block in `geometry.py`;
`metrics.py` uses the fuller Parameters/Returns block — see below). New `pu_manifold` modules should pick
the fuller `metrics.py`-style docstring since these functions are the shared-invariant surface other
modules/notebooks depend on and benefit from explicit contracts.

**Validation style** (`src/effdim/api.py:50-56`, from `compute_dim`):
```python
if data.ndim != 2:
    raise ValueError(f"Input data must be a 2D array, got {data.ndim}D.")
if data.shape[0] < 2:
    raise ValueError(f"Input data must have at least 2 samples, got {data.shape[0]}.")
if not np.all(np.isfinite(data)):
    raise ValueError("Input data contains NaN or infinity.")
```
Use this `raise ValueError(f"...")` style (not bare `assert`) for `subsample.py`'s **structural** checks
(D-08: equal shapes, sha256 match) — but CONTEXT.md's own `<specifics>` explicitly calls for `assert` in
the connectivity-sweep `for/else` remediation branch, so mix deliberately: `ValueError` for library-style
reusable validation helpers, `assert` for notebook-cell-local invariant checks (matches ARCHITECTURE.md's
own recommended guard cell: `assert hsc.shape[0] == legacysurvey.shape[0] == row_indices.shape[0]`, line 178).

**Row-alignment invariant pattern** (ARCHITECTURE.md Pattern 4, paraphrased into required shape):
- Select `row_indices` once via `np.sort(np.random.default_rng(SEED).choice(n_total, 10_000, replace=False))` (D-07).
- Read both `hsc` and `legacysurvey` off the *same* `row_indices` in one indexing pass — never two independent
  `.shuffle(seed).select(...)` calls (this is the literal Anti-Pattern 3 in ARCHITECTURE.md, lines 262-266).
- Normalize both columns at cache-write time (D-06), storing normalized arrays plus `hsc_norms`/`ls_norms`
  (raw norms, for the reproducible histogram) plus `row_indices` in `subsample_{seed}.npz`.
- Every function that touches both arrays takes/returns them together or takes a shared mask — never as two
  independently obtained objects (Pattern 4, line 153).

---

### `notebooks/pu_manifold/cache.py` (utility, file-I/O)

**Analog:** No in-repo file does config-hash caching. Use ARCHITECTURE.md's own worked example verbatim as
the starting shape (ARCHITECTURE.md lines 96-116), adapted for D-13's specific artifact list and D-14's key
composition (must include library versions, must NOT include git SHA):

```python
# notebooks/pu_manifold/cache.py
import hashlib, json
from pathlib import Path
import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def config_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def npz_cache(name: str, cfg: dict, compute_fn):
    path = CACHE_DIR / f"{name}_{config_key(cfg)}.npz"
    if path.exists():
        return dict(np.load(path))
    arrays = compute_fn()
    np.savez(path, **arrays)
    return arrays
```

**Required deviations from this example, per CONTEXT.md:**
- D-14: the `cfg` dict passed to `config_key` must include `dataset` config name, `seed`, `n_rows`,
  `normalize` flag, `n_neighbors`, `n_components`, `eigen_solver`, and `sklearn`/`numpy`/`scipy` versions —
  explicitly **not** the git SHA (a docstring commit must not invalidate a 1 GB artifact).
- D-13: needs a `joblib`-backed variant alongside `npz_cache` for `isomap_{key}.joblib` (the fitted `Isomap`
  object — `dist_matrix_`, `embedding_`, `nbrs_`, `kernel_pca_`), since only `k*` gets the full pickle.
- `.gitignore` already covers `notebooks/.cache/` (confirmed: `.gitignore` line "notebooks/.cache/" present) —
  no gitignore change needed.

---

### `notebooks/pu_manifold/curvature.py` and `notebooks/pu_manifold/mknn.py` (stubs)

**Analog for stub shape:** `src/effdim/metrics.py:5-49` — full NumPy-style docstring even for small
functions, explicit `Parameters`/`Returns` blocks, defensive zero-division guards returning a sentinel
(`0.0` or `1`) rather than raising:

```python
def pca_explained_variance(spectrum: np.ndarray, threshold: float = 0.95) -> int:
    """
    Compute the number of principal components required to explain a given
    threshold of variance.

    Parameters:
    -----------
    spectrum : np.ndarray
        Array of eigenvalues (explained variance) from PCA.
    threshold : float
        The cumulative variance threshold to reach (between 0 and 1).

    Returns:
    --------
    int
        Number of principal components needed to reach the threshold.
    """
```

D-02 requires these two modules be **stubbed, not empty** — "so the package shape is visible from the start."
Stub functions should declare the real signature Phase 3/4 will implement (per ARCHITECTURE.md's
`curvature.py` = "torch.func fundamental-form / mean-curvature helpers", `mknn.py` = "MKNN, permutation null,
bootstrap CI"), with a docstring in the `metrics.py` Parameters/Returns style and a body of
`raise NotImplementedError("Implemented in Phase 3 (CURV-*)")` / `"Implemented in Phase 4 (MKNN-*)"` —
not silent `pass`, so an accidental early call fails loudly rather than returning `None`.

---

### `notebooks/requirements-notebooks.txt` (config)

**Analog:** `pyproject.toml` `[project] dependencies` (lines 18-23) for what belongs in core vs. what stays
notebook-scoped:
```toml
dependencies = [
    "numpy",
    "scipy",
    "scikit-learn",
    "faiss-cpu"
]
```
`numpy`/`scipy`/`scikit-learn`/`faiss-cpu` are already core deps (installed via `effdim`'s own install) —
`requirements-notebooks.txt` should pin `torch`, `datasets` (HF), `matplotlib`, and any other notebook-only
library named in `.planning/research/STACK.md`, per D-03. Format is plain pip-requirements
(`package==x.y.z`), not TOML — this is a deliberate divergence from `pyproject.toml`'s format, not an
oversight; do not attempt to fold this into `[project.optional-dependencies]`.

## Shared Patterns

### Result-dict key names for D-12's `n_components` selection rule (CRITICAL — exact strings)

**Source:** `src/effdim/api.py:75-123`, the literal `results[...]` assignments inside `compute_dim`.

**Geometric/intrinsic estimators — INCLUDED in D-12's `ceil(median(...))` rule** (7 keys):
```python
results["mle_dimensionality"]        # Levina-Bickel MLE
results["two_nn_dimensionality"]     # TwoNN
results["danco_dimensionality"]      # DANCo
results["mind_mli_dimensionality"]   # MiND-ML1
results["mind_mlk_dimensionality"]   # MiND-MLk
results["ess_dimensionality"]        # ESS
results["tle_dimensionality"]        # TLE
results["gmst_dimensionality"]       # GMST
```
Note: CONTEXT.md's D-12 prose lists "TwoNN, MLE, ESS, TLE, GMST, DANCo, MiND-ML*" — the `MiND-ML*` maps to
**two** distinct keys (`mind_mli_dimensionality` and `mind_mlk_dimensionality`), so the median in D-12 is
over **8 values**, not 7. The planner must use this literal 8-key list, not the paraphrased 7-name prose.

**Spectral estimators — EXCLUDED from D-12's rule, reported for ISO-03 comparison only** (`src/effdim/api.py:75-88`):
```python
results["pca_explained_variance_95"]
results["participation_ratio"]
results["shannon_entropy"]
results["stable_rank"]
results["numerical_rank"]
results["cumulative_eigenvalue_ratio"]
results["renyi_eff_dimensionality_alpha_2"]   # and _alpha_3, _alpha_4, _alpha_5
results["geometric_mean_eff_dimensionality"]
```

**`compute_dim` signature and validation branches** (`src/effdim/api.py:29-60`):
```python
def compute_dim(data: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
```
- Accepts `np.ndarray` or `List[np.ndarray]` (list gets `np.vstack`'d).
- Raises `ValueError` on: empty list, wrong type, `ndim != 2`, `n_samples < 2`, non-finite values.
- Internally centers data (`_ensure_centered`, tol `1e-5`) before SVD — the notebook does **not** need to
  pre-center `legacysurvey` before calling `compute_dim`.
- Randomized-SVD switch (`src/effdim/api.py:143-147`): `if min(n_samples, n_features) < 1000: full SVD else: randomized_svd(n_components=min(n,d)-1)`.
  At `legacysurvey` shape `(10000, 768)`, `min(n,d) = 768 < 1000`, so **full SVD is used, not randomized** —
  correcting CONTEXT.md's `<code_context>` note ("switches to randomized SVD at `min(n, d) >= 1000`" — true
  in general, but for this phase's actual 768-dim input the threshold is NOT crossed). The planner should not
  assume randomized-SVD timing/nondeterminism applies to the ISO-03 pre-audit call.
- Also computes k-NN distances once via FAISS (`compute_knn_distances(data_f32, k=10)`, `src/effdim/geometry.py:8-31`)
  and shares them across the geometric estimators — this is internal to `compute_dim`, not something the
  notebook needs to replicate.

### Docstring/typing conventions to echo in `pu_manifold/`

**Source:** `src/effdim/metrics.py:1-49` (fuller style, preferred for shared-invariant modules) vs.
`src/effdim/geometry.py:1-13` (terser style, used for FAISS/scipy-heavy numerics).
- Type-annotate all public function signatures (`data: np.ndarray`, `-> float`/`-> Dict[str, Any]`).
- Use full NumPy-style `Parameters:`/`Returns:` blocks (metrics.py) for `subsample.py` and `cache.py` public
  functions, since these encode the row-alignment and cache-key invariants other modules rely on.
- Defensive returns over exceptions for numeric edge cases inside pure-math helpers (e.g. `if denominator == 0: return 0.0`,
  `metrics.py:47-49`), but `raise ValueError(f"...")` for input-contract violations (`api.py:50-56`) — apply
  the latter to `subsample.py`'s row-alignment structural checks.

### Error handling — no repo-wide try/except convention

**Finding:** Neither `api.py`, `geometry.py`, nor `metrics.py` use `try/except` anywhere — all error handling
is upfront `if ...: raise ValueError(...)` validation, not caught/wrapped exceptions. New `pu_manifold`
modules should follow this: validate inputs eagerly and raise, do not wrap computation in try/except blocks.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `notebooks/01_manifold_and_gate.ipynb` | notebook | batch | `notebooks/` is empty; no in-repo notebook exists. Structure comes entirely from CONTEXT.md D-01..D-15 and ARCHITECTURE.md §Recommended Project Structure / §Suggested Build Order, not from a codebase analog. |
| `notebooks/pu_manifold/cache.py` | utility | file-I/O | No config-hash caching utility exists anywhere in `src/effdim/`. Use ARCHITECTURE.md's Pattern 1 worked example as the base, adapted per D-13/D-14. |
| `notebooks/requirements-notebooks.txt` | config | n/a | No prior notebook-scoped requirements file exists (notebooks/ is new). `pyproject.toml` deps list is the closest reference for which packages are already core vs. need pinning here. |

## Metadata

**Analog search scope:** `src/effdim/` (only non-planning source directory), `notebooks/` (confirmed empty),
`pyproject.toml`, `.gitignore`
**Files scanned:** `src/effdim/__init__.py`, `src/effdim/api.py`, `src/effdim/geometry.py` (partial, lines 1-80),
`src/effdim/metrics.py` (partial, lines 1-60), `pyproject.toml`, `.gitignore`
**Pattern extraction date:** 2026-07-29
</content>
