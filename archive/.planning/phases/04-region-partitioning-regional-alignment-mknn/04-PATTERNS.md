# Phase 4: Region Partitioning & Regional Alignment (MKNN) - Pattern Map

**Mapped:** 2026-08-23
**Files analyzed:** 5 (3 code files, 1 requirements doc edit, no notebook Swiss-roll check per D4-12)
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `notebooks/pu_manifold/mknn.py` (fill 3 stubs: `mknn_score`, `permutation_null`, `bootstrap_ci`) | utility/service (statistical estimator) | CRUD-like batch transform (point cloud in, scalar/dict out) | `notebooks/pu_manifold/curvature_probe.py` — specifically its `permutation_null` (lines 1021-1147) and its `NearestNeighbors` k-NN idiom | exact (same package, same statistical-primitive-wrapping style) |
| `notebooks/pu_manifold/mknn.py` — region-partition helper (D4-09 sign split; no existing home, planner's call whether it lives here or a new module) | utility (transform) | batch transform | `notebooks/diagnostics/direction_partition_run.py` (`_unit`, k-NN-consuming clustering logic) | role-match (same "cluster unit-H vectors" concern, different algorithm — k-means vs. sign split) |
| `notebooks/diagnostics/region_partition_mknn_run.py` (new runner) | route/CLI entrypoint (batch job) | batch + file I/O (JSONL append) | `notebooks/diagnostics/pu_curvature_rankability_run.py` (whole file — `load_pu`, `run_cell`, `build_arg_parser`, `main`, `--smoke`) | exact |
| `notebooks/04_region_partition_mknn.ipynb` (new notebook) | component/report (read-out) | request-response-style batch read + plot | `notebooks/02.2_swiss_roll_cae_check.ipynb` for shape/pre-registration-cell discipline; no Swiss-roll notebook itself required (D4-12) | role-match (structural shape only, not content) |
| `.planning/REQUIREMENTS.md` (edit: re-mint REGN-01/03/04, add REGN-06) | config/contract doc | N/A (static edit) | same file, existing REGN-02/05 and MKNN-01..08 rows (Phase 3 re-mint precedent already in the doc) | exact |

## Pattern Assignments

### `notebooks/pu_manifold/mknn.py` — `mknn_score`, `permutation_null`, `bootstrap_ci` (utility, batch transform)

**Analog:** `notebooks/pu_manifold/curvature_probe.py`

**Current stub file in full** (`notebooks/pu_manifold/mknn.py`, all 32 lines) — this is what must be filled in, signatures must not change:
```python
from typing import Any


def mknn_score(z1: Any, z2: Any, k: Any) -> Any:
    """Mean MKNN score over all points. Caller guarantees row alignment — there is
    no object_id in this dataset to catch a mismatch."""
    raise NotImplementedError("Implemented in Phase 4 (MKNN-01)")


def permutation_null(z1: Any, z2: Any, k: Any, n_permutations: Any, seed: Any) -> Any:
    """Permutation-null MKNN distribution, drawn *within* the region's own index set —
    a global null would not control for the region's local density."""
    raise NotImplementedError("Implemented in Phase 4 (MKNN-04)")


def bootstrap_ci(z1: Any, z2: Any, k: Any, n_resamples: Any, seed: Any) -> Any:
    """Bootstrap (low, high) CI on the regional MKNN score, resampling within region."""
    raise NotImplementedError("Implemented in Phase 4 (MKNN-05)")
```
Module docstring already states the metric and the no-module-level-faiss-import constraint — preserve both.

**k-NN idiom to continue** (verified directly against source, three call sites all identical):
- `curvature_probe.py:190` — `nbrs = NearestNeighbors(n_neighbors=k_density + 1).fit(X)` then `dist, _ = nbrs.kneighbors(X)` and `r = dist[:, k_density]` (self is `dist[:, 0] == 0`).
- `curvature_probe.py:283-284` — `nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)`; `_, idx = nbrs.kneighbors(X)  # idx[:, 0] is the point itself`; neighbours used as `idx[i, 1:]`.
- `curvature_probe.py:959` (`quadric_mean_curvature`) — same `n_neighbors=k + 1` / drop-column-0 idiom a third time.
This is a fixed, three-times-repeated codebase convention: **always `NearestNeighbors(n_neighbors=k+1)`, always drop `idx[:, 0]`/`dist[:, 0]` as self.** `mknn.py`'s new k-NN membership-matrix code must do the same, not `n_neighbors=k`.

**Imports pattern** (`curvature_probe.py` lines 26-38):
```python
import math
import signal
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import permutation_test, spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

from . import cache
```
`mknn.py` should follow the same relative-import convention (`from . import cache` if it needs artifact freezing) and only import what it uses (`from sklearn.neighbors import NearestNeighbors`, `from scipy.stats import permutation_test, bootstrap`, `import numpy as np`).

**Core permutation-null pattern to mirror** (`curvature_probe.py` lines 1120-1147, condensed — this is the pattern `mknn.permutation_null` must reproduce, not a hand-rolled loop):
```python
def _stat(x: np.ndarray, y: np.ndarray) -> float:
    if statistic_fn is None:
        return float(spearmanr(x, y).statistic)
    return float(statistic_fn(x, y))

rng = np.random.default_rng(seed)
result = permutation_test(
    (h_true_norm, h_est_norm),
    _stat,
    permutation_type="pairings",
    alternative="greater",
    n_resamples=n_resamples,
    rng=rng,
)
observed_rho = float(result.statistic)
null_threshold = float(np.quantile(result.null_distribution, quantile))
return {
    "observed_rho": observed_rho,
    "null_quantile": float(quantile),
    "null_threshold": null_threshold,
    "null_mean": float(np.mean(result.null_distribution)),
    "null_std": float(np.std(result.null_distribution)),
    "n_resamples": int(n_resamples),
    "seed": int(seed),
    "clears_null": bool(observed_rho > null_threshold),
}
```
Note the codebase's own explicit warning inside this docstring (`curvature_probe.py` line 1062-1063): *"`mknn.permutation_null`'s hand-rolled loop (read for this codebase's naming convention only, never copied) is not repeated a third, different way here."* — i.e. the CURRENT stub docstring in `mknn.py` was written anticipating a hand-rolled loop; the sealed module already flags that as the wrong pattern to copy. **Use `scipy.stats.permutation_test`, not a manual shuffle loop**, matching `curvature_probe.permutation_null`, not the older informal style the comment refers to.

**Error/validation-guard pattern to mirror** (`curvature_probe.py` lines 1097-1118 — input guards before computing anything):
```python
h_true_norm = np.asarray(h_true_norm, dtype=np.float64)
h_est_norm = np.asarray(h_est_norm, dtype=np.float64)
if not np.all(np.isfinite(h_true_norm)):
    raise ValueError("permutation_null: h_true_norm contains a non-finite value.")
if not np.all(np.isfinite(h_est_norm)):
    raise ValueError("permutation_null: h_est_norm contains a non-finite value.")
if h_true_norm.shape[0] != h_est_norm.shape[0]:
    raise ValueError(f"permutation_null: h_true_norm (len={h_true_norm.shape[0]}) and "
                      f"h_est_norm (len={h_est_norm.shape[0]}) have different lengths.")
```
`mknn.py`'s functions should raise `ValueError` naming the offending argument the same way (e.g. length mismatch between `z1`/`z2`, non-finite values) rather than letting a `NaN` or shape mismatch propagate silently.

**No-default-on-pre-registered-constant pattern** (`centroid_mean_curvature`'s `d` argument, lines 204-219, and `permutation_null`'s `quantile` argument, lines 1065-1068): pre-registered constants (`quantile`, `d`) are REQUIRED positional arguments with no default, specifically so they cannot be inherited by accident. `mknn.py`'s `k`, `n_permutations`, `n_resamples`, `seed` should follow the same discipline — no silent defaults for values D4-17 pre-registers (1,000/1,000) or D4-06/07 freezes (`k`).

**`density_correct` flag/value pairing pattern** (`centroid_mean_curvature` lines 278-281): when a boolean flag requires a companion value, validate the PAIR, not the value alone:
```python
if density_correct and k_density is None:
    raise ValueError(
        "centroid_mean_curvature: k_density must be given when density_correct=True."
    )
```
Reusable if `mknn.py`'s region-partition helper needs a similar optional/required pairing (e.g. a near-zero exclusion threshold that is only meaningful when exclusion is enabled).

---

### `notebooks/pu_manifold/local_density_weights` and `centroid_mean_curvature` — CONSUMED UNCHANGED, not rewritten

**Exact signatures (verified against source, no defaults where noted):**
```python
def local_density_weights(X: np.ndarray, k_density: int, d: int) -> np.ndarray:
    ...  # returns (n,) weights, mean-normalized to 1

def centroid_mean_curvature(
    X: np.ndarray,
    k: int,
    d: int,                              # REQUIRED, no default (D-07)
    density_correct: bool = False,
    k_density: Optional[int] = None,     # REQUIRED when density_correct=True
) -> np.ndarray:
    ...  # returns (n, D) mean-curvature vectors, trace convention
```
Phase 4's field computation (step [1] of the pipeline) is a straight call:
```python
H = curvature_probe.centroid_mean_curvature(X, k=k, d=20, density_correct=True, k_density=30)
```
Never re-derive `d` (D-07 bars this); never edit `curvature_probe.py`.

---

### `notebooks/diagnostics/region_partition_mknn_run.py` (new runner, route/CLI, batch + file I/O)

**Analog:** `notebooks/diagnostics/pu_curvature_rankability_run.py` (whole-file shape)

**Header/import pattern** (lines 41-58):
```python
import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np

from pu_manifold import curvature_probe
from pu_manifold import cross_split_curvature as csc

DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "03.2_pu_curvature_rankability.jsonl"
```
The new runner follows this exactly, swapping the imports for `from pu_manifold import mknn` (plus `curvature_probe` for the field) and `DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "04_region_partition_mknn.jsonl"`.

**Data-loading pattern** (`load_pu`, lines 74-90) — the ONLY way this codebase reads the frozen 10k subsample; must be reused/adapted verbatim (it globs `subsample_*.npz`, verifies the requested column exists, and prints what it loaded):
```python
def load_pu(column: str) -> np.ndarray:
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
    if not cands:
        raise FileNotFoundError("no subsample_*.npz in notebooks/.cache/")
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if column in z.files and z[column].shape[0] > best_n:
                best, best_n = c, z[column].shape[0]
    if best is None:
        raise KeyError(f"no cached subsample carries column {column!r}")
    with np.load(best) as z:
        X = np.asarray(z[column], dtype=np.float64)
    print(f"loaded {column} {X.shape} from {Path(best).name}")
    return X
```
For MKNN the runner needs BOTH columns (`hsc`, `legacysurvey`) row-aligned from the same file — a two-column variant of this function is the natural extension, not a rewrite of the globbing/verification logic.

**CLI/argparse pattern** (lines 179-188):
```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, nargs="+", default=[30, 60, 120, 231])
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--column", type=str, default="legacysurvey")
    p.add_argument("--n-anchor", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p
```
New runner's `--k` default should be `[5, 10, 20, 50]` per D4-17's MKNN grid (distinct from the curvature-field `k` sweep values 30/60/120/231/350/500 — do not conflate the two `k`s; name them distinctly, e.g. `--mknn-k` vs `--field-k`, in the new runner).

**`--smoke` + JSONL-append + `main()` pattern** (lines 191-222):
```python
def main() -> None:
    a = build_arg_parser().parse_args()
    X = load_pu(a.column)

    if a.smoke:
        print("SMOKE: 800 rows, k=30 -- proves the path runs, measures nothing.\n")
        _header()
        _row(run_cell(X[:800], 30, a.d, a.seed, 200))
        return

    record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    ...
    with record_path.open("a") as fh:
        for k in a.k:
            r = run_cell(X, k, a.d, a.seed, a.n_anchor)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            records.append(r)
            _row(r)
    summarize(records)

if __name__ == "__main__":
    main()
```
`json.dumps(r, default=float)` is the codebase's numpy-scalar-safe JSON serialization idiom — reuse it verbatim for the new runner's per-cell (region × k) records.

---

### Region-partition (sign split, D4-09) — no direct analog, closest is a different-algorithm sibling

**Analog:** `notebooks/diagnostics/direction_partition_run.py` (role-match only — it clusters `H/‖H‖`, but via k-means, not the locked sign-split; still the closest example of "consume `centroid_mean_curvature`'s output and cluster the unit vectors").

**Unit-vector normalization helper to mirror exactly** (lines 78-79):
```python
def _unit(H: np.ndarray) -> np.ndarray:
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
```
This `np.maximum(..., 1e-12)` guard is the codebase's established idiom for avoiding divide-by-zero on a near-zero-norm vector — directly relevant to Pitfall 2 (near-zero `‖H‖` exclusion) since it shows the existing convention is a numerical floor, not a percentile exclusion; D4-09/Pitfall-2's percentile exclusion should be a separate, explicit filter applied BEFORE this normalization, not folded into it.

**Quantile-binning helper if REGN-02/discretion items need equal-count bins** (lines 82-89, matches `curvature_probe._quantile_bin_labels`'s own construction so bin membership is well-defined under ties):
```python
def _quantile_labels(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.shape[0], dtype=np.int64)
    for b, grp in enumerate(np.array_split(order, n_bins)):
        labels[grp] = b
    return labels
```

**What D4-09's sign split itself looks like** (new code, no existing analog — composed from the above primitives per the RESEARCH.md architecture diagram):
```python
def region_partition(H: np.ndarray, min_norm_percentile: float) -> dict:
    """Diametrical sign-split (D4-09): excludes near-zero-||H|| points below
    `min_norm_percentile` of the field's OWN ||H|| distribution (never an absolute
    threshold — Pitfall 2), then labels the rest by sign(<H_i/||H_i||, v>) where v is
    the top eigenvector of Cov(H_i/||H_i||) over the surviving points."""
    norm = np.linalg.norm(H, axis=1)
    floor = np.percentile(norm, min_norm_percentile)
    keep = norm >= floor
    unit = H[keep] / norm[keep, None]
    cov = np.cov(unit, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    proj = unit @ v
    labels = np.where(proj >= 0, 0, 1)
    return {"v": v, "labels": labels, "keep_idx": np.flatnonzero(keep),
            "excluded_idx": np.flatnonzero(~keep)}
```
This is this document's own composition (flagged as new, not copied from an existing function) — it should be placed in `mknn.py` or a small new module per RESEARCH.md's explicit "planner's call" note, and its output (`v`, `labels`, `excluded_idx`) is exactly what REGN-06 requires freezing via `cache.py`.

---

### `.planning/REQUIREMENTS.md` re-mint (config/contract edit)

**Analog:** the file's own existing Phase-3 re-mint precedent (same file — grep for how REGN-01/03/04's predecessor requirements or Phase 3's re-minted IDs are worded, and follow that exact table-row format; re-mint text is already given verbatim in RESEARCH.md's `<phase_requirements>` table for REGN-01/03/04/06). This is a same-file, same-convention edit — no external analog needed.

## Shared Patterns

### k-NN via sklearn, self excluded — the one idiom every estimator here uses
**Source:** `notebooks/pu_manifold/curvature_probe.py` lines 190, 283-284, 959 (three independent, identical call sites)
**Apply to:** `mknn.py`'s membership-matrix construction, and any k-NN call the region-partition helper needs.
```python
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
_, idx = nbrs.kneighbors(X)   # idx[:, 0] is the point itself
neigh_idx = idx[:, 1:]        # excludes self
```
Never `n_neighbors=k` with no self-exclusion step — that would silently include self as a "neighbour," breaking every k-normalized statistic downstream (MKNN's own `k^-1|N_k∩N_k|` included).

### Permutation testing — `scipy.stats.permutation_test`, never hand-rolled
**Source:** `curvature_probe.permutation_null`, lines 1120-1133
**Apply to:** `mknn.permutation_null`
```python
result = permutation_test(
    (arr1, arr2), _stat,
    permutation_type="pairings", alternative="greater",
    n_resamples=n_resamples, rng=np.random.default_rng(seed),
)
```
`permutation_type="pairings"` is explicitly required — the codebase's own docstring (lines 1054-1063) warns that omitting it silently computes a different null, and that the intersection statistic must treat both permuted arguments as an already-paired dataset, never assume either is the caller's original array.

### Config-hash-keyed cache containment guard
**Source:** `notebooks/pu_manifold/cache.py` lines 39-47
**Apply to:** REGN-06's frozen-artifact write (`v`, `labels`, `excluded_idx`) and any `.npz`/JSONL the new runner or partition helper writes.
```python
def _assert_inside_cache(path: Path) -> None:
    resolved = path.resolve()
    resolved_cache_dir = CACHE_DIR.resolve()
    if resolved_cache_dir not in resolved.parents and resolved != resolved_cache_dir:
        raise ValueError(
            f"Refusing to use path outside CACHE_DIR: {resolved} is not inside "
            f"{resolved_cache_dir}."
        )
```
Every write must go through `cache.cache_path(stem, ext)` (which calls this guard internally), never a raw `Path(...)` construction, so a stem with a `..` segment cannot escape `notebooks/.cache/`.

### JSONL runner output, numpy-safe serialization
**Source:** `pu_curvature_rankability_run.py` line 213
**Apply to:** `region_partition_mknn_run.py`
```python
fh.write(json.dumps(r, default=float) + "\n")
fh.flush()
```
`default=float` coerces numpy scalars (e.g. `np.float64`) to plain floats, avoiding `TypeError: Object of type float64 is not JSON serializable`.

### Pre-registration / ordering-proof cell discipline
**Source:** RESEARCH.md's own derivation from the 02.2 CAE precedent (not a literal code excerpt from an existing file, since no prior notebook in this repo has needed exactly this ordering assertion for a *verdict rule* rather than a *threshold*) — apply the same spirit as `curvature_probe.permutation_null`'s own docstring insistence that "CALIBRATES, does NOT SET, the gate": a cell computing a number is not the same as a cell fixing the rule that number will be judged by. The verdict rule (MKNN-07) must be printed and asserted in a cell strictly before the first regional-MKNN-number cell — this ordering is REGN-04's and the ROADMAP Ordering constraint's requirement, and it is what distinguishes this phase's discipline from a plain analysis notebook.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| D4-09's diametrical sign-split assignment logic itself | utility (transform) | batch transform | No existing function performs a covariance-eigenvector sign split in this codebase; `direction_partition_run.py` clusters via k-means instead. Composed above from `np.cov`/`np.linalg.eigh` primitives — use RESEARCH.md's Architecture Patterns diagram (step [4]) and this document's own composed snippet as the basis, not a copied analog. |
| `tests/test_mknn.py` | test | N/A | Deliberately NOT added per D4-18 (locked decision) — no analog needed because no file is created. |
| Swiss-roll sanity notebook for the region partition or MKNN | test/notebook | N/A | Deliberately NOT added per D4-12 (locked) and RESEARCH.md's Project Constraints — neither the sign split nor MKNN itself is a representation-learning/manifold-recovery model under CLAUDE.md's rule. |

## Metadata

**Analog search scope:** `notebooks/pu_manifold/` (all `.py` modules and `tests/`), `notebooks/diagnostics/` (all runner scripts), `.planning/REQUIREMENTS.md`. No search was needed outside these directories — `src/effdim/` is off-limits this milestone (CLAUDE.md) and was not searched.
**Files scanned:** `mknn.py`, `curvature_probe.py` (full, both `local_density_weights`/`centroid_mean_curvature`/`permutation_null` sections), `pu_curvature_rankability_run.py` (full), `direction_partition_run.py` (partial — enough to extract `_unit`/`_quantile_labels`/run shape), `cache.py` (header + containment guard), `notebooks/pu_manifold/tests/test_curvature_probe.py` (header only, for test-file convention), directory listings of `notebooks/pu_manifold/tests/` and `notebooks/diagnostics/`.
**Pattern extraction date:** 2026-08-23
