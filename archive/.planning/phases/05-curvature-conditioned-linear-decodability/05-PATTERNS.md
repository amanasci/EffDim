# Phase 5: Curvature-Conditioned Linear Decodability - Pattern Map

**Mapped:** 2026-08-24
**Files analyzed:** 3 new files (+ 1 notebook, pattern-mapped loosely, not a "file to classify"
in the controller/service sense)
**Analogs found:** 3 / 3

All three files are **additive** — no existing sealed module is modified. This matches
CLAUDE.md's additive-only rule and 05-CONTEXT.md/05-RESEARCH.md's own framing.

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|-----------------|---------------|
| `notebooks/pu_manifold/linear_probe.py` | config/model (pre-registration constants + pure functions) | transform (batch, no file I/O) | `notebooks/pu_manifold/region_partition.py` | exact (same role: sealed constants module + `assert_preregistered()` + pure numpy transform functions) |
| `notebooks/diagnostics/curvature_probe_decodability_run.py` | route/CLI runner (orchestration) | batch, event-driven (`--mode` dispatch) | `notebooks/diagnostics/region_partition_mknn_run.py` | exact (same role: argparse `--mode`/`--selfcheck`/`--smoke` runner, JSONL provenance, pre-registration guard) |
| `notebooks/pu_manifold/tests/test_linear_probe.py` | test | request-response (pytest assertions) | `notebooks/pu_manifold/tests/test_region_partition.py` | exact (same role: known-answer + boundary + guard tests for a sealed constants/pure-function module) |

Two secondary sources feed the runner's curvature-extraction step, distinct from the primary
analog above:

| Capability needed by the runner | Source module | Role |
|----------------------------------|----------------|------|
| Loading a sealed CAE checkpoint + calling the correct curvature extractor | `notebooks/diagnostics/curvature_field_pu_run.py` (`load_converged_model`, `build_cae`) + `notebooks/pu_manifold/chart_curvature.py` (`chart_curvature_field`) | model-loading helper + estimator call site |
| Bootstrap CI on a per-bucket statistic | `notebooks/pu_manifold/mknn.py` (`bootstrap_ci`) | statistic helper, copy-the-shape only |
| Cache path / manifest conventions | `notebooks/pu_manifold/cache.py` | cache tier, imported directly (not copied) |

## Pattern Assignments

### `notebooks/pu_manifold/linear_probe.py` (config/model, transform)

**Analog:** `notebooks/pu_manifold/region_partition.py` (251 lines, read in full)

**Module docstring / caveat-disclosure convention** (lines 1-41): the sealed module opens with
a long docstring stating exactly what the module does NOT resolve (codimension gap, covariance
form ambiguity, which precondition check was skipped and why) — not just what it does. Phase 5's
`linear_probe.py` docstring should carry the equivalent disclosures inline: the Pitfall 1
correction (call `chart_curvature.chart_curvature_field`, never
`decoder_curvature.plain_decoder_curvature`), the seed-pooling normalization choice and why raw
averaging fails (Pitfall 2), and D5-11's "no known-answer anchor" caveat repeated in the module
itself, not only in 05-CONTEXT.md.

**Pre-registration constants block** (lines 48-114):
```python
# --- Pre-registration (D4-11, ratified at this plan's blocking checkpoint) -----------------
#
# PRE-REGISTERED under the ROADMAP's Ordering constraint: every constant below, and
# VERDICT_RULE's full text, were ratified at this plan's Task 2 blocking decision checkpoint
# BEFORE any regional MKNN number existed. Amending any of ... after a regional MKNN number
# has been computed invalidates the phase ...

MIN_NORM_PERCENTILE = 5.0
MIN_REGION_N = 500
...
SEED = 20260822
COVARIANCE_FORM = "mean_centered"

VERDICT_RULE = """MKNN-07 verdict rule -- ratified at this plan's Task 2 blocking checkpoint,
before any regional MKNN number existed.
...
"NO DETECTABLE DIFFERENCE" at the headline k is a complete, valid outcome. It is never treated
as a phase failure ...
"""
```
Copy this exact shape for Phase 5: `TRAIN_FRACTION`, `SPLIT_SEED`, `N_BUCKETS`,
`RIDGE_ALPHA_GRID`, `POOLING_METHOD`, `SEED_STEMS`, `CURVATURE_CONVENTION = "trace"`, and a full
`VERDICT_RULE` string that states — in its own text, per the sealed precedent — that "NO
DETECTABLE DIFFERENCE" is a complete, valid, non-escalated outcome (05-CONTEXT.md's own
requirement on the pre-registered rule).

**`assert_preregistered()` shape** (lines 117-136):
```python
def assert_preregistered() -> None:
    """Raise ``RuntimeError`` unless the pre-registration is intact: ``VERDICT_RULE`` is a
    non-empty string naming ``HEADLINE_K``, ``K_FROZEN`` is a positive int, and
    ``MIN_REGION_N`` is a positive int. Called at the top of the runner's ``--mode regional``
    branch so the regional path fails loudly rather than computing anything when the
    pre-registration is absent or malformed."""
    if not isinstance(VERDICT_RULE, str) or not VERDICT_RULE.strip():
        raise RuntimeError("assert_preregistered: VERDICT_RULE is empty or not a string.")
    if "HEADLINE_K" not in VERDICT_RULE:
        raise RuntimeError("assert_preregistered: VERDICT_RULE does not name HEADLINE_K.")
    if not isinstance(K_FROZEN, int) or isinstance(K_FROZEN, bool) or K_FROZEN <= 0:
        raise RuntimeError(f"assert_preregistered: K_FROZEN={K_FROZEN!r} is not a positive int.")
    ...
```
Phase 5's `assert_preregistered()` should raise `RuntimeError` (matching this exact exception
type, not `ValueError`) unless: `VERDICT_RULE` is a non-empty string naming e.g. `N_BUCKETS`;
`TRAIN_FRACTION` is a float in `(0, 1)`; `SPLIT_SEED`/`SEED_STEMS` are well-typed; `N_BUCKETS`
is a positive int. Raise on the FIRST failing check, one check per constant, each with its own
`f"assert_preregistered: {name}={value!r} is ..."` message — this module's exact idiom.

**Pure-function signature convention, no default on pre-registered args** (lines 149-153,
docstring): `region_partition(H, min_norm_percentile)` takes `min_norm_percentile` as a
**required** positional argument with no default — "following this module's convention for a
pre-registered constant: a default value is exactly how such a value gets inherited by
accident rather than by an explicit call-site choice." Apply the identical discipline to
`fit_probe(X_train, Y_train, alpha_grid)`, `pool_seed_fields(fields, method)`,
`bucket_by_field(pooled_H_norm, n_buckets)` — every pre-registered constant is passed in by the
caller (the runner, reading it off the module's own constant), never defaulted inside the
function.

**`ValueError` input-validation convention** (lines 168-179): finite-check, dimensionality
check, then a domain check on the tuning parameter, each raising `ValueError` naming the
offending argument by name and value:
```python
H = np.asarray(H, dtype=np.float64)
if H.ndim != 2:
    raise ValueError(f"region_partition: H must be two-dimensional, got shape {H.shape}.")
if not np.all(np.isfinite(H)):
    raise ValueError("region_partition: H contains a non-finite value.")
if not (0.0 <= min_norm_percentile < 100.0):
    raise ValueError(f"region_partition: min_norm_percentile={min_norm_percentile} must be in [0, 100).")
```
Reuse this exact validation idiom in `linear_probe.py`'s functions (shape checks on `X`/`Y`
row-alignment, `np.isfinite` checks, `n_buckets >= 1`, `train_fraction` domain).

**Result-dict convention**: `region_partition` returns a single flat `Dict[str, Any]` with every
intermediate value the caller or a test might want (`floor`, `keep_idx`, `mean_unit_norm`, the
full eigenvalue spectrum) rather than only the headline output. `linear_probe.py`'s
`fit_probe`, `pool_seed_fields`, `bucket_residuals` should return the same kind of
everything-included dict (fitted `W`/`b`/selected `alpha`, per-point residuals, bucket
assignment array, bucket `n` counts) so the runner and the tests can inspect intermediate state
without recomputing it.

---

### `notebooks/diagnostics/curvature_probe_decodability_run.py` (route/CLI runner, batch + event-driven)

**Analog:** `notebooks/diagnostics/region_partition_mknn_run.py` (941 lines; read the module
docstring, `load_pu_pair`, `run_global_cell`, `_spearman_report`, `selfcheck`, the JSONL append
loop, and `build_arg_parser`/`main`)

**Module docstring + usage lines** (lines 1-17):
```python
"""Phase 4 region-partitioning MKNN runner. `--mode global` ... `--mode partition` ...
`--mode regional` ... requires both `region_partition.assert_preregistered()` and the
frozen partition artifact to exist before it will even attempt a regional MKNN cell ...

    python notebooks/diagnostics/region_partition_mknn_run.py --selfcheck
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global --smoke
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global
    python notebooks/diagnostics/region_partition_mknn_run.py --mode partition --smoke
    python notebooks/diagnostics/region_partition_mknn_run.py --mode partition
    python notebooks/diagnostics/region_partition_mknn_run.py --mode regional
"""
```
Copy this shape: a one-paragraph docstring naming every `--mode` value and what guard it
enforces, followed by the exact runnable command lines (e.g. `--mode field`, `--mode bucketed`,
`--selfcheck`, `--smoke`) — this doubles as both documentation and the manual smoke-test script.

**Imports / sys.path convention** (lines 19-37):
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
from pu_manifold import region_partition
```
Copy verbatim except the `pu_manifold` imports become `from pu_manifold import cache,
linear_probe, chart_curvature` (+ `cae` for `build_cae`) and add `import torch`.

**`load_pu_pair` — the two-column PU loader** (lines 42-70, cited verbatim in RESEARCH.md):
copy unchanged (only the two default column names `"hsc"`/`"legacysurvey"` already match
D5-01's target). This is the exact function to reuse for Phase 5's `hsc -> legacysurvey`
loading — no reimplementation needed, just import or literally copy this one function into (or
next to) the new runner.

**`_spearman_report`** (lines 125-136) — reused verbatim per RESEARCH.md's own citation for
D5-05's inter-seed diagnostics and D5-07's continuous `spearman(||H||, residual)` report:
```python
def _spearman_report(a: np.ndarray, b: np.ndarray, name: str) -> Dict[str, Any]:
    """One plain Spearman correlation, printed with its p-value and point count. When
    either input is constant, ``scipy.stats.spearmanr`` returns NaN rather than raising;
    that case is reported with an explicit undefined marker rather than a number ..."""
    n_pts = int(a.shape[0])
    rho, p = spearmanr(a, b)
    if np.isnan(rho):
        print(f"... UNDEFINED (constant input, spearmanr returned NaN) -- n={n_pts}")
        return {"rho": None, "p_value": None, "n": n_pts, "undefined": True}
    print(f"... rho={rho:+.4f}  p={p:.4g}  n={n_pts}")
    return {"rho": float(rho), "p_value": float(p), "n": n_pts, "undefined": False}
```

**Pre-registration guard (D5-10)** — the exact `--mode` dispatch shape (lines 795-808 area, and
the generalized pattern seen at lines 850-905):
```python
if a.mode == "regional":
    region_partition.assert_preregistered()
    partition_artifact = cache.cache_path(...)
    if not partition_artifact.exists():
        raise FileNotFoundError(
            f"--mode regional requires the frozen partition artifact at "
            f"{partition_artifact}, which does not exist. Run --mode partition first ..."
        )
```
Phase 5's analogue: `--mode bucketed` calls `linear_probe.assert_preregistered()`, then checks
`cache.cache_path("05_curvature_field", "npz").exists()` before computing anything, raising
`FileNotFoundError` with the identical "run X first" message shape.

**JSONL provenance append loop** (lines 926-932, and the `--smoke` early-return pattern at
lines 861-876, `--mode partition --smoke` shows the pattern for a non-"global" mode too):
```python
record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
record_path.parent.mkdir(parents=True, exist_ok=True)
...
with record_path.open("a") as fh:
    for k in a.mknn_k:
        r = run_global_cell(...)
        fh.write(json.dumps(r, default=float) + "\n")
        fh.flush()
        records.append(r)
        _row(r)
```
`json.dumps(r, default=float)` is the exact idiom that lets a row dict containing numpy scalars
(`np.float64`, `np.int64`) serialize without a manual cast on every field — reuse this, not a
custom encoder.

**`--smoke` early-return convention** (lines 861-876): every mode has a `--smoke` branch that
runs on a small slice (`X_hsc[:800]`, or explicit small n/k/d), prints a one-line "SMOKE: ..."
banner naming exactly what's reduced, and returns without writing to the JSONL record —
"writes nothing." Phase 5's `--mode field --smoke` / `--mode bucketed --smoke` should follow
the identical shape (e.g., run curvature extraction on 200 rows and one seed only).

**`selfcheck()` — known-answer self-check function** (lines 643-698, full function read):
```python
def selfcheck() -> bool:
    """... This runner flag is the phase's automated implementation check ..."""
    ok = True
    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            ok = False
    rng = np.random.default_rng(20260822)
    ...
    for name, fn in [
        ("k + 1 > n raises ValueError", lambda: mknn.mknn_score(X[:5], Y[:5], 10)),
        ...
    ]:
        try:
            fn()
            check(name, False)
        except ValueError:
            check(name, True)
    return ok
```
Copy this `check(name, cond)` closure + `nonlocal ok` + PASS/FAIL-line pattern exactly for
Phase 5's `--selfcheck`: a planted `y = A @ x + b + tiny_noise` linear relationship recovered
by `fit_probe`, plus a fabricated `||H||` array correlated with residual by construction to
prove the verdict rule fires correctly on a known-answer case (per RESEARCH.md's Validation
Architecture table).

**`build_arg_parser` / `main` dispatch structure** (lines 773-941): `argparse.ArgumentParser`
with `--mode` as a `choices=[...]` argument, `--selfcheck`/`--smoke` as `store_true` flags,
numeric frozen-constant overrides as named flags with defaults matching the module's own
pre-registered constants (e.g. `--seed`, `--confidence-level`), then `main()` dispatches
`if a.selfcheck: ...`, `if a.mode == "X": ...` in sequence, each mode's block self-contained.
Copy this exact `choices`/flag/dispatch shape for `--mode field` / `--mode bucketed`.

**Second reusable source — loading a sealed CAE checkpoint** (from
`notebooks/diagnostics/curvature_field_pu_run.py`, cited by RESEARCH.md at lines 1510-1530,
487-500, verified against RESEARCH.md's own quoted excerpt rather than independently re-read
this session since the exact excerpt was already reproduced verbatim in 05-RESEARCH.md's Code
Example 2):
```python
def build_cae(n_charts, device=torch.device("cpu")):
    model = cae.ChartAutoEncoder(
        in_dim=768, embed_dim=40, chart_dim=20, n_charts=n_charts,
        hidden=[250, 250, 250], activation="silu",
    )
    return model.to(device)

def load_converged_model(n_charts, seed, device):
    ckpt_path = cache.cache_path(f"03_converged_cae_pu_nc{n_charts}_seed{seed}", "pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No converged checkpoint at {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_cae(n_charts, device=device).double()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
```
followed by the CORRECT curvature-extraction call (this is the load-bearing correction — see
"Critical Correction" section below):
```python
from pu_manifold import chart_curvature
field = chart_curvature.chart_curvature_field(model, x64, mode="reverse")
H_norm = field["H_norm"].detach().cpu().numpy().astype(np.float64)
```

**Bootstrap CI on a per-bucket statistic** — `notebooks/pu_manifold/mknn.py::bootstrap_ci`
(lines 124-167, read in full):
```python
def bootstrap_ci(z1, z2, k, n_resamples, seed, confidence_level) -> Dict[str, Any]:
    """... `n_resamples` and `confidence_level` are required arguments with no default ..."""
    ...
    rng = np.random.default_rng(seed)
    result = bootstrap(
        (per_point,), np.mean, method="percentile",
        n_resamples=n_resamples, confidence_level=confidence_level, rng=rng,
    )
    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)
    return {
        "score": float(per_point.mean()), "ci_low": ci_low, "ci_high": ci_high,
        "degenerate": bool(ci_low == ci_high), "confidence_level": float(confidence_level),
        "n_resamples": int(n_resamples), "seed": int(seed), "n": int(n), "k": int(k),
    }
```
Same shape for Phase 5's `bucket_residual_ci(residuals_in_bucket, n_resamples, seed,
confidence_level)`: required args with no default, `scipy.stats.bootstrap(...,
method="percentile", rng=np.random.default_rng(seed))`, a flat result dict including
`degenerate` (CI collapsed to a point — worth carrying forward as a diagnostic here too).

---

### `notebooks/pu_manifold/tests/test_linear_probe.py` (test)

**Analog:** `notebooks/pu_manifold/tests/test_region_partition.py` (157 lines, read in full)

**Module docstring / invocation note** (lines 1-15):
```python
"""
Known-answer and boundary tests for ``pu_manifold.region_partition`` ...

No HuggingFace access, no torch, no fixtures beyond synthetic point clouds generated
in-test. Not collected by the core `effdim` test suite (``pyproject.toml``'s
``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_region_partition.py -q
"""
```
Copy this framing exactly (adjusted for `linear_probe`), including the explicit note that
`pyproject.toml`'s `testpaths` excludes this directory and the file must be run explicitly.

**`sys.path` bootstrap** (lines 17-26):
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from pu_manifold import region_partition as rp
```
Copy the `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` line verbatim (same
directory depth: `notebooks/pu_manifold/tests/test_linear_probe.py` is the same nesting level
as `test_region_partition.py`), then `from pu_manifold import linear_probe as lp`.

**Known-answer test shape** (lines 32-54): construct a synthetic fixture with a KNOWN ground
truth (here, two antipodal cones with a known axis `w`), run the function, assert exact or
near-exact recovery (`ari == 1.0`, `abs(dot(v, w)) > 0.99`). For `linear_probe.py`'s
`fit_probe`, the equivalent is: construct `X` with a known linear map `y = A @ x + b + tiny
noise`, fit, assert recovered `W` is close to `A` and held-out residual is small — this is
exactly what RESEARCH.md's Validation Architecture table already names as
`test_fit_probe_shape_and_row_alignment`.

**Boundary/tie-rule test shape** (lines 60-80): construct input so a percentile/threshold lands
exactly on a data point, assert the documented tie rule (`>=`, inclusive) is honored. Reuse for
Phase 5's `test_bucket_assignment_known_answer` — construct `pooled_H_norm` values so a bucket
edge lands exactly on a point, assert the documented inclusive/exclusive convention.

**Reproducibility test shape** (lines 86-99): call the function twice on identical input,
assert identical output (`np.array_equal`). Reuse for `pool_seed_fields` and `fit_probe` given
a fixed `SPLIT_SEED`/`SEED_STEMS`.

**Counts-sum-to-total test shape** (lines 105-134): construct a fixture designed so an edge
case (here, `n_zero_projection`) is forced, assert the returned counts partition the total
exactly. Reuse for `test_size_matched_check_uses_test_split_counts` (D5-08/Pitfall 4) — assert
bucket counts on the realized test split sum to `len(test_idx)` exactly, not to `10000`.

**Guard test shape** (lines 140-158):
```python
def test_region_partition_raises_on_one_dimensional_H():
    with pytest.raises(ValueError):
        rp.region_partition(np.zeros(10), min_norm_percentile=0.0)

def test_region_partition_raises_on_non_finite_H():
    H = np.ones((10, 3))
    H[3, 1] = np.nan
    with pytest.raises(ValueError):
        rp.region_partition(H, min_norm_percentile=0.0)

def test_region_partition_raises_on_percentile_out_of_range():
    H = np.random.default_rng(2).normal(size=(10, 3))
    with pytest.raises(ValueError):
        rp.region_partition(H, min_norm_percentile=100.0)
    with pytest.raises(ValueError):
        rp.region_partition(H, min_norm_percentile=-1.0)
```
Copy this exact `pytest.raises(ValueError)` shape for `linear_probe.py`'s input-validation
guards, and add the `assert_preregistered` guard test (D5-10) as
`test_assert_preregistered_raises_when_absent`:
```python
def test_assert_preregistered_raises_when_missing_verdict_rule(monkeypatch):
    monkeypatch.setattr(lp, "VERDICT_RULE", "")
    with pytest.raises(RuntimeError):
        lp.assert_preregistered()
```
(`monkeypatch.setattr` on the module-level constant, matching how `region_partition.py`'s own
constants are module-level and would be patched the same way — no existing test in
`test_region_partition.py` exercises `assert_preregistered` directly since region_partition's
own guard is exercised only via the runner, but the shape follows directly from
`assert_preregistered`'s own raise conditions.)

## Critical Correction (verified independently this session)

RESEARCH.md's Pitfall 1 is confirmed correct against direct reads this session:

- `notebooks/pu_manifold/decoder_curvature.py`'s own module docstring (lines 1-8, read
  directly): *"Phase 02.6 decoder-substrate screening: EXACT mean curvature through a decoder
  that has no chart routing ... This module is `chart_curvature.py` with the
  `chart_decoders[chart_idx]` two-hop composition removed ... both free candidate substrates
  screened by Phase 02.6 (a plain autoencoder, and a `PlainAutoEncoder` trained under
  `topoae.train_topoae`) decode through ONE smooth MLP end to end and have no chart index at
  all."*
- Its `plain_decoder_map` (lines 140-152, read directly) calls `model.decode(z.unsqueeze(0))`
  directly, with a docstring warning against wrapping `model.forward` because
  `PlainAutoEncoder.forward` "re-derives ``z`` internally from ``x``". `ChartAutoEncoder` has no
  bare `.decode(z)` matching this single-hop signature.
- `notebooks/pu_manifold/chart_curvature.py::chart_curvature_field(model, x, batch_size=32,
  mode="reverse")` (signature confirmed directly at lines 513-515; docstring at 516-552, read
  directly) is the function that internally does `chart_probs(z).argmax(dim=1)` chart
  assignment, per-chart curvature, and row-order reassembly — exactly the function
  `curvature_field_pu_run.py` uses (per RESEARCH.md's citation, not independently re-read this
  session, but consistent with the confirmed docstrings above).

**Conclusion: RESEARCH.md's correction is correct.** `05-PATTERNS.md`'s pattern assignment above
maps Phase 5's decoder-curvature call site to `chart_curvature.chart_curvature_field`, not
`decoder_curvature.plain_decoder_curvature`, and the runner's analog section states the correct
call explicitly.

## Shared Patterns

### Pre-registration freeze + guard
**Source:** `notebooks/pu_manifold/region_partition.py` (constants block, `VERDICT_RULE`,
`assert_preregistered`) + `notebooks/diagnostics/region_partition_mknn_run.py` (the `--mode`
guard calling it before computing)
**Apply to:** `linear_probe.py` (constants + guard) and
`curvature_probe_decodability_run.py` (`--mode bucketed` guard branch)

### Cache path / manifest conventions
**Source:** `notebooks/pu_manifold/cache.py` — `cache.cache_path(stem, ext)`,
`cache.npz_cache(stem, cfg, compute_fn)`, `cache.json_cache(...)`, all containment-checked via
`_assert_inside_cache`
**Apply to:** every cache write/read in the new runner — `cache.cache_path("05_curvature_field",
"npz")`, JSONL record path under `.cache/`, never a raw `Path` construction. Note: no `cache.py`
helper exists for JSONL append specifically (confirmed — `cache.py`'s three cache functions are
npz/joblib/json only), so the JSONL append loop is hand-rolled exactly as
`region_partition_mknn_run.py` does it (`record_path.open("a")`, `json.dumps(r,
default=float)`, `fh.flush()`), per RESEARCH.md's own note on this.

### `--selfcheck` / `--smoke` / real-mode structure
**Source:** `notebooks/diagnostics/region_partition_mknn_run.py` (`selfcheck()`, the `if
a.smoke:` early-return blocks, `build_arg_parser`/`main`)
**Apply to:** `curvature_probe_decodability_run.py` in full

### Spearman reporting
**Source:** `notebooks/diagnostics/region_partition_mknn_run.py::_spearman_report`
**Apply to:** D5-05 inter-seed diagnostics, D5-07 continuous statistic, D5-13 density-confound
re-measurement — reuse the same function (copy or share) for all three call sites so the
"UNDEFINED on constant input" handling is consistent everywhere.

### Bootstrap CI
**Source:** `notebooks/pu_manifold/mknn.py::bootstrap_ci`
**Apply to:** `linear_probe.py`'s per-bucket residual CI (D5-08)

## No Analog Found

None — all three files have a strong, same-role, same-data-flow analog in the codebase (Phase
4's `region_partition.py` / `region_partition_mknn_run.py` / `test_region_partition.py` triad).
The only genuinely novel logic (per RESEARCH.md's own summary) is the seed-pooling
normalization and the ridge-probe fit itself, both of which are standard library calls
(`sklearn.linear_model.RidgeCV`) with no existing project-internal analog needed — RESEARCH.md's
Code Examples and Don't-Hand-Roll table already cover these directly and are sufficient without
a further codebase analog.

## Metadata

**Analog search scope:** `notebooks/pu_manifold/`, `notebooks/diagnostics/`,
`notebooks/pu_manifold/tests/`
**Files read in full or targeted-range this session:** `region_partition.py` (251/251 lines),
`test_region_partition.py` (157/157 lines), `cache.py` (121/121 lines),
`region_partition_mknn_run.py` (targeted: 1-140, 643-722, 850-941 of 941),
`chart_curvature.py` (targeted: 1-90, 513-552 of 907), `decoder_curvature.py` (targeted: 1-30,
140-170 of 319), `mknn.py` (targeted: 1-50, 100-167 of 180)
**Pattern extraction date:** 2026-08-24
