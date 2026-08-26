# Phase 7: Curvature-Conditioned Crossmodal Alignment - Pattern Map

**Mapped:** 2026-08-25
**Files analyzed:** 4 (new/modified) + 9 unchanged imports whose exact signatures are pinned below
**Analogs found:** 4 / 4

RESEARCH.md's "Architecture Patterns" section already named every analog. This file makes each
one concrete with real, line-numbered excerpts so the planner can write `<action>` fields
directly from it, without re-reading source.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `notebooks/pu_manifold/crossmodal_curvature.py` | utility/config (frozen constants + pure compute) | transform / statistical | `notebooks/pu_manifold/pointcloud_probe.py` (structure) + `notebooks/pu_manifold/linear_probe.py` (origin of inherited constants) | exact |
| `notebooks/diagnostics/07_crossmodal_curvature_run.py` | controller/runner (owns all I/O) | batch / request-response (CLI `--mode`) | `notebooks/diagnostics/region_partition_mknn_run.py` | exact |
| `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` | test | CRUD-of-assertions | `notebooks/pu_manifold/tests/test_pointcloud_probe.py` | exact |
| `notebooks/07_crossmodal_curvature_check.ipynb` (or similarly named) | notebook (committed with outputs) | transform / report | prior phase notebooks (e.g. `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb` for shape only — **not a new Swiss Roll requirement**, see Pitfall 4 in RESEARCH.md) | role-match |

Files imported UNCHANGED (D7-05 seals them; never edit, only call):
`mknn.py`, `linear_probe.py`, `pointcloud_probe.py`, `cae.py`, `decoder_curvature.py`, `cache.py`,
`curvature_probe.py`, `cross_split_curvature.py`, everything under `src/effdim/`.

## Pattern Assignments

### `notebooks/pu_manifold/crossmodal_curvature.py` (new frozen module)

**Analog:** `notebooks/pu_manifold/pointcloud_probe.py` (structural template), inheriting the
"re-declare, don't import, for constants" discipline from `notebooks/pu_manifold/linear_probe.py`.

**Docstring / provenance block pattern** (`pointcloud_probe.py` lines 1-64): open with a
docstring that states (a) this module adds, never edits sealed modules; (b) exactly what changes
relative to the prior phase's instrument and the exhaustive list of what does not; (c) every
`Dxx-yy` decision ID inline in prose, e.g.:

```python
"""Phase 7 curvature-conditioned crossmodal alignment: the pre-registration constants block,
its guard, the per-point MKNN gap-fill, and the verdict rule.

**This module adds; it does not edit.** `mknn.py`, `cae.py`, `decoder_curvature.py`,
`curvature_probe.py`, `cross_split_curvature.py`, `linear_probe.py` and `pointcloud_probe.py`
are all imported here and never modified (D7-05).
"""
```

**Constants block pattern** (`pointcloud_probe.py` lines 66-131, `linear_probe.py`'s equivalent
section is the origin these values were first declared in): flat `UPPER_CASE` literals, no
computed defaults, e.g. for Phase 7:

```python
D_SWEEP = (20, 25, 32)                      # D7-01
HEADLINE_K = 20                             # inherit Phase 4's headline k (Open Question 1)
MKNN_K_GRID = (5, 10, 20, 50)               # sensitivity grid, non-gating
CAE_HIDDEN = (250, 250, 250)
CAE_ACTIVATION = "silu"
CURVATURE_SOURCE_FUNCTION = "decoder_curvature.plain_decoder_curvature"
CURVATURE_CONVENTION = "trace"              # matches decoder_curvature.py's own convention name
DENSITY_K = 30                              # for curvature_probe.local_density_weights, pin explicitly
N_PERMUTATIONS = 999                        # or whatever value is chosen; must be a literal, not computed
PERMUTATION_SEED = 20260825
NULL_QUANTILE = 0.95
POSITIVE_CONTROL_TARGET_RHOS = (0.05, 0.10, 0.20)   # D7-02 effect-size grid
POSITIVE_CONTROL_SEED = 20260825
SEED_HANDLING_RULE = "single_seed_across_d_sweep"   # per RESEARCH.md Open Question 3 — name the limitation
```

**`VERDICT_RULE` string-literal pattern** (`pointcloud_probe.py` lines 138-176): a triple-quoted
prose block naming the exact headline statistic, the exact pass/fail criteria as a lettered list,
the terminal outcome vocabulary, and every inherited caveat restated in the rule's own text (not
only in surrounding prose) — e.g. carry forward Phase 4's density-confound caveat (D7-03) and the
"field measures true curvature" disclaimer (§8 of CONTEXT.md) verbatim inside `VERDICT_RULE`,
exactly as `pointcloud_probe.py` restates G6-01/G6-03/G6-04 inside its own rule text rather than
only in the module docstring.

**`_REQUIRED_CONSTANTS` + `assert_preregistered()` pattern** (`pointcloud_probe.py` lines
178-204, copy verbatim except the tuple contents and module name):

```python
_REQUIRED_CONSTANTS = (
    "D_SWEEP", "HEADLINE_K", "MKNN_K_GRID", "CAE_HIDDEN", "CAE_ACTIVATION",
    "CURVATURE_SOURCE_FUNCTION", "CURVATURE_CONVENTION", "DENSITY_K",
    "N_PERMUTATIONS", "PERMUTATION_SEED", "NULL_QUANTILE",
    "POSITIVE_CONTROL_TARGET_RHOS", "POSITIVE_CONTROL_SEED", "SEED_HANDLING_RULE",
    "VERDICT_RULE", "VERDICT_VALUES",
)

def assert_preregistered() -> None:
    g = globals()
    missing = []
    for name in _REQUIRED_CONSTANTS:
        if name not in g:
            missing.append(f"{name} (absent)")
            continue
        value = g[name]
        if value is None:
            missing.append(f"{name} (None)")
        elif isinstance(value, str) and not value.strip():
            missing.append(f"{name} (empty string)")
        elif isinstance(value, (tuple, list)) and len(value) == 0:
            missing.append(f"{name} (empty sequence)")
    if missing:
        raise RuntimeError(
            "crossmodal_curvature.assert_preregistered: Phase 7 is not frozen -- the "
            "following pre-registered constants are unset: " + ", ".join(missing) + ". "
            "No probe number may be computed before the freeze (D7-06)."
        )
```

**D7-04 gap-fill — `per_point_mknn`** (source of the gap: `mknn.py` lines 47-59, the exact
final line that discards the per-point array):

```python
# notebooks/pu_manifold/mknn.py, lines 47-59 (read this session; NEVER edit)
def mknn_score(z1: Any, z2: Any, k: Any) -> Any:
    ...
    A = _membership_matrix(z1, k)
    B = _membership_matrix(z2, k)
    return float(((A & B).sum(axis=1) / k).mean())   # <-- per-point array discarded here
```

New function to add in `crossmodal_curvature.py`, composing `mknn._membership_matrix` (a plain,
unmangled top-level function, importable) exactly as `mknn_score` does but returning the array:

```python
from pu_manifold import mknn

def per_point_mknn(z1: Any, z2: Any, k: Any) -> np.ndarray:
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    if z1.shape[0] != z2.shape[0]:
        raise ValueError(
            f"per_point_mknn: z1 has {z1.shape[0]} rows but z2 has {z2.shape[0]} rows; "
            "rows must be row-aligned."
        )
    A = mknn._membership_matrix(z1, k)
    B = mknn._membership_matrix(z2, k)
    return (A & B).sum(axis=1) / k
```

Regression pin: `per_point_mknn(z1, z2, k).mean() == mknn.mknn_score(z1, z2, k)` on a small
fixture (see test file section below).

**Wrapping `curvature_probe.permutation_null` for D7-04 significance** — exact signature
(`curvature_probe.py` lines 1021-1028):

```python
def permutation_null(
    h_true_norm: np.ndarray,
    h_est_norm: np.ndarray,
    n_resamples: int,
    seed: int,
    quantile: float,
    statistic_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> dict:
```

`statistic_fn=None` reproduces `lambda x, y: spearmanr(x, y).statistic` under
`permutation_type="pairings"`. This is the function to call for D7-04's headline
significance test — NOT `mknn.permutation_null` (a different, MKNN-only function with a
different signature: `mknn.permutation_null(z1, z2, k, n_permutations, seed, quantile)`,
which shuffles a k-NN *pairing* directly, not a generic paired-array statistic). Call site:

```python
result = curvature_probe.permutation_null(
    H_norm, mknn_per_point, N_PERMUTATIONS, PERMUTATION_SEED, NULL_QUANTILE,
)
# result: {"observed_score", "null_distribution", "p_value", "null_mean", "null_std",
#          "null_threshold", "null_quantile", "clears_null", "n_permutations", "seed", "n", "k"}
```

(Confirm exact return-dict key names against `curvature_probe.py`'s actual body before wiring —
the excerpt above is from the analogous `mknn.permutation_null`'s return shape; `curvature_probe
.permutation_null`'s return dict was documented but its keys were not fully re-read this pass —
grep `curvature_probe.py` around line 1021-1147 for the literal `return {` block before writing
the runner.)

**Wrapping `cross_split_curvature.partial_spearman` for D7-03** — exact signature
(`cross_split_curvature.py` lines 232-234):

```python
def partial_spearman(x: Any, y: Any, controls: Optional[Any] = None) -> float:
    """Rank-transforms x, y, and each control column; residualizes x and y against the
    rank-transformed controls (with intercept) by least squares; returns Pearson
    correlation of the residuals. controls=None returns the raw Spearman rho."""
```

Call sites:

```python
raw_rho = cross_split_curvature.partial_spearman(H_norm, mknn_per_point, controls=None)
density_controlled_rho = cross_split_curvature.partial_spearman(
    H_norm, mknn_per_point, controls=density
)
```

**D7-02 positive control** — no existing helper (RESEARCH.md Pattern 3 flags this as the one
genuinely new piece of statistical engineering). Build a `plant_positive_control(h_real, k,
target_rhos, seed) -> Dict[str, Any]` pure function in the new module: rank-transform
`h_real` to `[0,1]`, draw `MKNN_planted ~ Binomial(k, p_i)/k` with `p_i` a function of rank tuned
to each `target_rho` in `POSITIVE_CONTROL_TARGET_RHOS`, then run the SAME
`curvature_probe.permutation_null` call used for the headline test on `(h_real,
MKNN_planted)` and report at which target rho `clears_null` first turns `True`. Pre-register
`POSITIVE_CONTROL_TARGET_RHOS`/`POSITIVE_CONTROL_SEED` in the same freeze commit (RESEARCH.md
Open Question 2's recommendation).

---

### `notebooks/diagnostics/07_crossmodal_curvature_run.py` (new runner)

**Analog:** `notebooks/diagnostics/region_partition_mknn_run.py` (full file read this session).

**Module docstring + `--mode` CLI pattern** (lines 1-17):

```python
"""Phase 4 region-partitioning MKNN runner. `--mode global` ... `--mode partition` ...
`--mode regional` requires both `region_partition.assert_preregistered()` and the frozen
partition artifact to exist before it will even attempt a regional MKNN cell.

    python notebooks/diagnostics/region_partition_mknn_run.py --selfcheck
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global --smoke
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global
"""
```
For Phase 7, mirror this exactly with modes like `--mode dsweep` (the real 2-hour D7-01 compute),
`--mode positive-control`, `--selfcheck` — and add the new
`OMP_NUM_THREADS`/`torch.set_num_threads` cap at the very top of the file, before any torch
import triggers thread-pool initialization (RESEARCH.md Pitfall 1 — grep confirmed ZERO prior use
of `OMP_NUM_THREADS` anywhere in `notebooks/`, so this is new, not copy-paste).

**sys.path bootstrap + imports pattern** (lines 20-33):

```python
import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from pu_manifold import cache
from pu_manifold import curvature_probe
from pu_manifold import mknn
from pu_manifold import region_partition   # Phase 7: import crossmodal_curvature instead

DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "04_region_partition_mknn.jsonl"
```

For Phase 7, `DEFAULT_RECORD` should point at `notebooks/.cache/07_crossmodal_curvature.jsonl`
(the FROZEN record — distinct from the pre-existing spike scripts' own
`notebooks/diagnostics/07_*.jsonl` outputs, which are informational only per RESEARCH.md Pitfall
3/CONTEXT.md §9 and must not be treated as this phase's frozen artifact).

**`load_pu_pair()` — copy verbatim** (lines 42-70):

```python
def load_pu_pair(
    column_a: str = "hsc", column_b: str = "legacysurvey"
) -> Tuple[np.ndarray, np.ndarray, str]:
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
    if not cands:
        raise FileNotFoundError("no subsample_*.npz in notebooks/.cache/")
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if column_a in z.files and column_b in z.files and z[column_a].shape[0] > best_n:
                best, best_n = c, z[column_a].shape[0]
    if best is None:
        raise KeyError(f"no cached subsample carries both {column_a!r} and {column_b!r} columns")
    with np.load(best) as z:
        Xa = np.asarray(z[column_a], dtype=np.float64)
        Xb = np.asarray(z[column_b], dtype=np.float64)
    if Xa.shape[0] != Xb.shape[0]:
        raise ValueError(...)
    print(f"loaded {column_a} {Xa.shape} and {column_b} {Xb.shape} from {Path(best).name}")
    return Xa, Xb, best
```

**"One flat JSONL-serializable row per cell" pattern** (`run_global_cell`, lines 73-90 onward):
each `--mode` computes a dict of plain-Python/JSON-serializable values (never raw numpy arrays —
Phase 6's amendment `fix(06): serialize numpy arrays in the Phase 6 record` is the exact
cautionary precedent) and appends it as one line via a `cache.json_cache`-style or manual
`jsonl` append, timed with `time.monotonic()`.

**D7-01's decoder + curvature call** (exact signatures, confirmed directly against source this
pass):

```python
# cae.py line 585: class PlainAutoEncoder(nn.Module)
# cae.py line 605: def encode(self, x: torch.Tensor) -> torch.Tensor
# cae.py line 608: def decode(self, z: torch.Tensor) -> torch.Tensor
# cae.py line 727: def train_plain_ae(model: "PlainAutoEncoder", x_train: torch.Tensor,
#                                      cfg: Dict[str, Any]) -> Dict[str, Any]
#   -- identical protocol/return shape as train_cae; cfg["protocol_difference"] records
#      "no pre-training stage, no chart predictor, no cross-entropy term, no Lipschitz penalty"

model = cae.PlainAutoEncoder(in_dim=768, latent_dim=d, hidden=CAE_HIDDEN, activation=CAE_ACTIVATION)
fit = cae.train_plain_ae(model, x_train32, cfg)
model.eval().double()
with torch.no_grad():
    z = model.encode(x64)                                # (10000, 768) -> (10000, d)

# decoder_curvature.py line 161: def plain_decoder_curvature(model: Any, z: torch.Tensor) -> Dict[str, Any]
#   guard order: assert_c2_decoder(model) first, then _assert_float64
#   differentiates model.decode ALONE -- never the encoder-composed round trip (D7-01's own text)
field = decoder_curvature.plain_decoder_curvature(model, z)
H_norm = field["H_norm"].detach().cpu().numpy()          # (10000,)
```

`decoder_curvature.assert_c2_decoder(model)` (line 80) branches on whether `model` has an
`.activation` attribute (`ChartAutoEncoder` shape) or not (`PlainAutoEncoder`'s actual shape) —
`PlainAutoEncoder` takes the second branch, inspecting `getattr(model, "decoder", model).modules()`.

**Cache/containment discipline** — route any cached artifact through `cache.py`'s helpers
rather than raw paths (`cache.py`, full file, 121 lines):

```python
# notebooks/pu_manifold/cache.py
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"

def config_key(cfg: Dict[str, Any]) -> str: ...          # sha256(json.dumps(cfg, sort_keys=True))[:16]
def _assert_inside_cache(path: Path) -> None: ...          # T-01-01 containment guard
def cache_path(stem: str, ext: str) -> Path: ...
def npz_cache(stem, cfg, compute_fn) -> Dict[str, np.ndarray]: ...
def joblib_cache(stem, cfg, compute_fn) -> Any: ...
def json_cache(stem, cfg, compute_fn) -> Dict[str, Any]: ...
```

Note: RESEARCH.md's Recommended Project Structure states the new runner writes to
`notebooks/diagnostics/*.jsonl` per the existing `07_*_run.py` spike convention — but
CONTEXT.md's own §9 table and the Architectural Responsibility Map both treat `07-CONTEXT.md`'s
spike scripts as writing to `notebooks/diagnostics/`. **Resolve at plan time**: if the frozen
Phase 7 record is meant to be a `cache.json_cache`/`npz_cache`-backed artifact (matching every
sealed Phase 4/5/6 precedent, which write into `notebooks/.cache/`), route it through `cache.py`;
if it is deliberately kept alongside the spike scripts' own `*.jsonl` outputs for continuity,
that is a plan-level decision the planner should state explicitly, not inherit silently.

---

### `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` (new test file)

**Analog:** `notebooks/pu_manifold/tests/test_pointcloud_probe.py` (full file, 137 lines).

**Structure pattern**: `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` then
`from pu_manifold import crossmodal_curvature as cc`. One `test_assert_preregistered_passes_when_frozen`
and one `test_assert_preregistered_names_every_unset_constant(monkeypatch)` mirroring:

```python
def test_assert_preregistered_passes_when_frozen():
    pp.assert_preregistered()

def test_assert_preregistered_names_every_unset_constant(monkeypatch):
    # monkeypatch.delattr / monkeypatch.setattr one _REQUIRED_CONSTANTS entry to None,
    # assert RuntimeError raised and the constant's name appears in str(exc)
```

**D7-04 regression pin** (new, no direct analog — compose from Pattern 2's own text):

```python
def test_per_point_mknn_mean_matches_mknn_score():
    rng = np.random.default_rng(0)
    z1 = rng.normal(size=(400, 16))
    z2 = rng.normal(size=(400, 16))
    per_point = cc.per_point_mknn(z1, z2, k=10)
    assert per_point.mean() == pytest.approx(mknn.mknn_score(z1, z2, 10))
```

**Known-answer MKNN test** — mirror `mknn.py`'s own selfcheck shape as exercised by
`region_partition_mknn_run.py --selfcheck` (identical pair scores 1.0; independent Gaussian
clouds land near `mknn.chance_floor(n, k)`).

---

## Shared Patterns

### Freeze-then-compute discipline (D7-06)
**Source:** `notebooks/pu_manifold/pointcloud_probe.py` lines 1-262 (full file), origin
constants in `notebooks/pu_manifold/linear_probe.py`.
**Apply to:** `crossmodal_curvature.py` in full — constants block, `VERDICT_RULE` string,
`assert_preregistered()`, and "all downstream compute functions are pure, no file I/O" (the
runner alone owns I/O).

### Cache containment
**Source:** `notebooks/pu_manifold/cache.py` (full file, `_assert_inside_cache`, `npz_cache`,
`json_cache`).
**Apply to:** `07_crossmodal_curvature_run.py` for any artifact under `notebooks/.cache/`.

### Row-aligned PU pair loading
**Source:** `notebooks/diagnostics/region_partition_mknn_run.py` lines 42-70 (`load_pu_pair`).
**Apply to:** `07_crossmodal_curvature_run.py`'s data-loading step; call once, reuse the loaded
`(hsc, legacysurvey)` pair across all three `d` fits (`per_point_mknn` does not depend on `d`).

### Guard against sealed-module edits
**Source:** D7-05 (CONTEXT.md), reinforced by `pointcloud_probe.py`'s own opening docstring
sentence ("This module adds; it does not edit.").
**Apply to:** every new file. `mknn.py`, `cae.py`, `decoder_curvature.py`, `curvature_probe.py`,
`cross_split_curvature.py`, `linear_probe.py`, `pointcloud_probe.py` are import-only.

### Serial d-sweep with capped threads
**Source:** No prior in-repo pattern (RESEARCH.md Pitfall 1 confirms zero prior
`OMP_NUM_THREADS` usage) — this is new engineering for Phase 7, not a copy.
**Apply to:** `07_crossmodal_curvature_run.py`'s `--mode dsweep` entry point: loop
`for d in D_SWEEP:` in-process, single script invocation, `torch.set_num_threads(N)` set before
first torch use.

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| D7-02 positive-control planting helper (function inside `crossmodal_curvature.py`) | utility | transform | No prior module in this codebase plants a curvature-MKNN relationship at a specified dynamic range; Phase 6's `rng.random(n)` selfcheck is explicitly rejected as underpowered by D7-02 and is not a usable analog. Original design work — see RESEARCH.md Assumptions Log A1. |
| Committed notebook `notebooks/07_crossmodal_curvature_check.ipynb` | notebook | report | No fresh Swiss Roll check is required (RESEARCH.md Pitfall 4) — `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb` already satisfies CLAUDE.md's gate for the exact `cae.PlainAutoEncoder` + `decoder_curvature.plain_decoder_curvature` combination Phase 7 reuses unchanged. The new notebook's job is reporting the frozen runner's own record, not a new sanity check — closest structural analog is simply "the phase's own prior notebooks" generically, not a single strong match. |

## Metadata

**Analog search scope:** `notebooks/pu_manifold/`, `notebooks/pu_manifold/tests/`,
`notebooks/diagnostics/`
**Files read this session:** `pointcloud_probe.py` (full), `cache.py` (full), `mknn.py` (full),
`cae.py` (targeted: `PlainAutoEncoder`, `train_plain_ae`), `decoder_curvature.py` (targeted:
`plain_decoder_curvature`, `assert_c2_decoder`), `curvature_probe.py` (targeted:
`permutation_null` signature/docstring), `cross_split_curvature.py` (targeted:
`partial_spearman`), `test_pointcloud_probe.py` (partial), `region_partition_mknn_run.py`
(partial: header, `load_pu_pair`, `run_global_cell` start)
**Pattern extraction date:** 2026-08-25
