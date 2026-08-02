---
phase: 01-data-loading-manifold-reconstruction
reviewed: 2026-07-31T04:30:58Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - notebooks/pu_manifold/__init__.py
  - notebooks/pu_manifold/cache.py
  - notebooks/pu_manifold/subsample.py
  - notebooks/pu_manifold/curvature.py
  - notebooks/pu_manifold/mknn.py
  - notebooks/pu_manifold/tests/test_pu_manifold.py
  - notebooks/requirements-notebooks.txt
  - notebooks/01_manifold_and_gate.ipynb
findings:
  critical: 1
  warning: 3
  info: 4
  total: 8
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-07-31T04:30:58Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

`cache.py` and `subsample.py` are solid: the path-containment guard in
`_assert_inside_cache` was verified empirically (both a `../`-relative traversal stem and
an absolute-path stem such as `/etc/passwd` are correctly rejected — it resolves the fully
composed path and checks containment rather than pattern-matching the stem string), the
alignment smoke test's statistics are sound, and the 14-test suite passes cleanly against
the pinned dependency versions. `curvature.py` and `mknn.py` are intentional Phase 3/4
stubs per D-02, out of scope for findings.

The one Critical finding is in the notebook, not the library modules: the Phase 1 -> Phase
2 handoff artifact (`phase1_handoff_{fit_key}.json`, §5.3) is cached under a key that omits
the §4.0 pre-registered constants (`PLATEAU_THRESH`, `SWEEP_K_RANGE`,
`GEO_PAIR_COUNT`/`SEED`, etc.) that determine the `k_star_selection` data being recorded —
exactly the "silently reuse a stale artifact" threat (T-01-03) `cache.py`'s manifest
system exists to prevent. The notebook's stage-2 sweep cells defend against an analogous
gap (§4.3's `GEO_PAIR_COUNT`/`GEO_PAIR_SEED` re-verification asserts) but §5.3 has no
equivalent guard, despite being the most-trusted, most downstream artifact of the phase.

Three Warnings and four Info items round out the report: non-atomic cache writes in
`cache.py` (verified to crash-loop with an unhandled `JSONDecodeError` on an interrupted
manifest write), an unguarded near-zero-norm precision hazard in `l2_normalize` (verified:
a `1e-100`-magnitude row passes the exact-zero check but silently degrades to visible
imprecision), and a mathematically-dead but logically-inverted fallback branch in the
notebook's `_stage2_k_selection` helper.

## Critical Issues

### CR-01: `phase1_handoff` cache key omits the pre-registered sweep constants it records, risking silent stale-provenance reuse

**File:** `notebooks/01_manifold_and_gate.ipynb:§5.3 (code cell index 84)`
**Issue:**

```python
PHASE1_HANDOFF = json_cache(
    f"phase1_handoff_{fit_key}", ANALYSIS_CFG, lambda: _phase1_handoff_built
)
```

`json_cache`'s cache-hit path (`cache.py:251-252`) is:

```python
if path.exists() and _manifest_matches(stem, cfg):
    return json.loads(path.read_text())
```

The `cfg` passed here is `ANALYSIS_CFG` (`dataset`, `seed`, `n_rows`, `normalize`,
`n_neighbors`, `n_components`, `eigen_solver`, plus three library versions). It does
**not** include `PLATEAU_THRESH`, `SWEEP_K_RANGE`, `K_EXTENSIONS`, `K_CEILING`,
`K_WARN_ABOVE`, `MIN_PLATEAU_RUN`, `STAGE2_MAX_FITS`, `PLATEAU_TIE_BREAK`,
`GEO_PAIR_COUNT`, or `GEO_PAIR_SEED` -- the §4.0 pre-registered constants that actually
determine `STABILITY_TABLE`, `PLATEAU_RUNS`, `_selected_run`, and therefore `K_STAR`
itself (`_k_star_selection` and `"thresholds": dict(PLATEAU_THRESH)` baked into
`_phase1_handoff_built`, cell 84).

If any of those §4.0 constants is tuned between two notebook runs (e.g. relaxing
`PLATEAU_THRESH["procrustes_disparity_max"]` per the remediation instructions the
notebook itself prints in cell 76: *"relax one of the thresholds ... and re-run the
ENTIRE sweep"*) and the tuning happens not to change the resulting `K_STAR`/`N_COMPONENTS`
(entirely plausible for a small threshold relaxation), `ANALYSIS_CFG` -- and hence
`fit_key` -- is bit-identical to the prior run. `_manifest_matches` then returns `True`,
`compute_fn` (the lambda producing the freshly-built `_phase1_handoff_built`) is never
even invoked, and `json_cache` returns the **old** `phase1_handoff_{fit_key}.json` from
disk verbatim -- silently keeping the prior run's `thresholds`, `plateau_runs`,
`chosen_run`, and `k_star_selection.criterion` fields even though this run used
different §4.0 constants to reach the same `k*`.

This is precisely the threat `cache.py`'s docstring calls T-01-03 ("a filename match
alone is never trusted... see PITFALLS.md Pitfall 10") -- and the notebook demonstrably
knows how to guard against this exact class of staleness, because §4.3 (cell 70) does
guard an analogous gap in the stage-2 sweep npz records:

```python
assert int(rec1["geo_pair_count"]) == GEO_PAIR_COUNT and int(
    rec2["geo_pair_count"]
) == GEO_PAIR_COUNT, (...)
assert int(rec1["geo_pair_seed"]) == GEO_PAIR_SEED and int(
    rec2["geo_pair_seed"]
) == GEO_PAIR_SEED, (...)
```

The §5.3 handoff cell has no equivalent check, and `phase1_handoff_{fit_key}.json` is
explicitly documented as "the concrete, machine-readable interface Phase 2's planner and
gate read." A downstream phase trusting a silently stale `thresholds`/`plateau_runs`
record is a real correctness/provenance-integrity risk, not a cosmetic one.

**Fix:** Either (a) widen the `cfg` passed to this `json_cache` call to include every
§4.0 pre-registered constant that feeds `_k_star_selection` (mirroring the pattern
already used for `fit_cfg` in §4.2, or simply `dict(ANALYSIS_CFG, **PLATEAU_THRESH,
sweep_k_range=SWEEP_K_RANGE, k_extensions=K_EXTENSIONS, geo_pair_count=GEO_PAIR_COUNT,
geo_pair_seed=GEO_PAIR_SEED, min_plateau_run=MIN_PLATEAU_RUN,
stage2_max_fits=STAGE2_MAX_FITS, plateau_tie_break=PLATEAU_TIE_BREAK)`), or (b) add an
explicit re-verification assert before trusting a cache hit, the same way §4.3 does for
`geo_pair_count`/`geo_pair_seed`:

```python
if path.exists() and _manifest_matches(...):
    cached = json.loads(path.read_text())
    assert cached["k_star_selection"]["thresholds"] == dict(PLATEAU_THRESH), (
        "Cached phase1_handoff thresholds diverge from the current PLATEAU_THRESH -- "
        "refusing to silently reuse a stale handoff."
    )
```

## Warnings

### WR-01: Cache writes are not atomic; an interrupted manifest write crash-loops with an unhandled `JSONDecodeError`

**File:** `notebooks/pu_manifold/cache.py:151-167` (`_write_manifest`), `:116-148`
(`_manifest_matches`), and the corresponding write calls in `npz_cache` (`:193-194`),
`joblib_cache` (`:226-227`), `json_cache` (`:254-255`)
**Issue:** `np.savez(path, **arrays)`, `joblib_dump(obj, path)`, and
`manifest_path.write_text(...)` all write directly to their final on-disk path -- none
use a temp-file-then-rename pattern. `manifest_path.write_text` in particular is not
atomic: an interruption partway through (kernel restart, `SIGKILL`/OOM-kill -- the
notebook's own §4.2/§5.2 comments explicitly flag "a transient peak of roughly 2-3 GiB"
during the multi-minute Isomap fits this cache wraps, so an OOM kill mid-write is a
realistic failure mode here, not a contrived one) leaves a truncated `.meta.json` on
disk. Verified concretely:

```
$ python -c "... write_text('{\"a\": 1' ) ... npz_cache('stem1', {'a':1}, compute)"
JSONDecodeError: Expecting ',' delimiter: line 1 column 8 (char 7)
```

`_manifest_matches` calls `json.loads(manifest_path.read_text())` with no
try/except, so every subsequent call to `npz_cache`/`joblib_cache`/`json_cache` for that
stem raises an unhandled `JSONDecodeError` instead of the module's own descriptive
`ValueError` -- and the corrupted manifest persists, so the crash repeats on every
re-run until a human manually finds and deletes the file. For a multi-minute, ~1 GB
Isomap fit artifact, this is a meaningfully worse failure mode than the module's other
error paths, all of which are designed to fail with an actionable message.
**Fix:** Write to a temp file in `CACHE_DIR` and `os.replace()` it into place, for both
the artifact and the manifest:

```python
def _write_manifest(stem: str, cfg: Dict[str, Any]) -> None:
    manifest_path = _manifest_path(stem)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    tmp_path.replace(manifest_path)  # atomic on POSIX and Windows
```
and wrap `_manifest_matches`'s `json.loads` call to raise the module's own descriptive
`ValueError` (naming the corrupt path and instructing deletion) rather than letting a
raw `JSONDecodeError` propagate.

### WR-02: `l2_normalize` only guards exact-zero norms; near-zero norms silently degrade precision without warning

**File:** `notebooks/pu_manifold/subsample.py:96-121`
**Issue:** The only safety check is `if np.any(norms == 0): raise ValueError(...)`
(`:116-120`). Verified that a row with a genuinely tiny but non-zero norm passes this
check and produces a numerically degraded "unit" vector with no error and no warning:

```
x = [[1e-100, 0, 0], [1, 0, 0]]  ->  norms = [1e-100, 1.0]  (passes the check)
```

and, more concretely, for a near-zero-but-multi-component row (`~2.77e-159` norm, 768
components):

```
unit row norm (should be exactly 1.0): 1.0000030063809708   # off by ~3e-6
```

That ~3e-6 relative error is *smaller* than the notebook's own §1.6 unit-norm sanity
check (`abs(hsc_row_norms.min() - 1.0) < 1e-5`, cell 32) -- meaning a row with a
sufficiently degenerate raw norm could pass every guard in this pipeline while carrying
materially corrupted directional information into the alignment smoke test, the Isomap
fit, and (eventually) Phase 3's curvature estimates. Real DINOv3 embeddings are very
unlikely to have near-zero norms, which is presumably why this was not caught by the
committed run, but the function itself provides no defense beyond the exact-zero
boundary case.
**Fix:** Add a relative floor, not just an exact-zero check:

```python
MIN_NORM = 1e-8  # or derived from float64 eps relative to the expected norm scale
if np.any(norms < MIN_NORM):
    raise ValueError(
        f"l2_normalize received at least one near-zero-norm row (min norm "
        f"{norms.min():.3e} < {MIN_NORM}); refusing to normalize a numerically "
        f"degenerate vector."
    )
```

### WR-03: `_stage2_k_selection`'s duplicate-index fallback is dead code with inverted logic that contradicts its own docstring

**File:** `notebooks/01_manifold_and_gate.ipynb:§4.2 (code cell index 66)`
**Issue:**

```python
def _stage2_k_selection(connected_k, max_fits):
    """
    ... take `max_fits` values evenly spaced across the sorted range, always including
    both endpoints, which maximizes the span the plateau criterion (§4.0) can see.
    """
    values = sorted(connected_k)
    if len(values) <= max_fits:
        return values
    n = len(values)
    idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, max_fits)))
    while len(idx) < max_fits:
        remaining = [i for i in range(n) if i not in idx]
        idx.append(min(remaining, key=lambda i: min(abs(i - j) for j in idx)))
        idx = sorted(idx)
    return [values[i] for i in idx]
```

The `while` loop's fallback -- meant to top up `idx` if rounding produced fewer than
`max_fits` distinct indices -- picks the remaining index that **minimizes** distance to
the existing selection (`min(remaining, key=lambda i: min(abs(i - j) for j in idx))`).
That is the opposite of the docstring's stated goal ("maximizes the span"); a correct
implementation would `max` over the same distance metric.

In this specific function this branch is provably unreachable: this code path is only
entered when `n > max_fits`, which (since `n` and `max_fits` are integers) guarantees
`n - 1 >= max_fits`, so `step = (n-1)/(max_fits-1) > 1` always, so consecutive
`np.linspace` points are always more than 1 apart and therefore always round to distinct
integers. Verified empirically for every `(n, max_fits)` pair with `4 <= n <= 39` and
`3 <= max_fits < n`: zero cases produced a duplicate. So `len(idx) == max_fits`
immediately after the `set()` comprehension in every case this function is actually
called with today, and the `while` loop body never executes. It is a latent landmine,
not a live bug: if the rounding strategy above it is ever changed (e.g. swapped for
`math.floor`, or `max_fits` is ever allowed to approach `n`), the fallback will silently
start clustering the extra selections next to existing ones instead of spreading them,
directly undermining the "maximize the span" contract §4.3's plateau criterion depends
on -- with no test coverage to catch the regression.
**Fix:** Either delete the dead branch (replace with an `assert len(idx) == max_fits,
"unreachable: linspace with n > max_fits always yields max_fits distinct rounded
indices"`) or fix the direction if it's being kept as defensive code:

```python
idx.append(max(remaining, key=lambda i: min(abs(i - j) for j in idx)))
```

## Info

### IN-01: Notebook duplicates the `ALIGNMENT_MARGIN_Z` magic number instead of importing the constant

**File:** `notebooks/01_manifold_and_gate.ipynb:§1.6 (code cell index 31)`
**Issue:** `assert ALIGNMENT_STATS["z"] > 5.0, (...)` hardcodes `5.0` rather than using
`subsample.ALIGNMENT_MARGIN_Z`, which is already exported through `pu_manifold.__init__`
(`__all__` includes `"ALIGNMENT_MARGIN_Z"`). If the library constant is ever changed,
this notebook assertion silently drifts out of sync with the actual enforced margin.
**Fix:** `from pu_manifold import ALIGNMENT_MARGIN_Z` and assert against that name
instead of the literal `5.0`.

### IN-02: `test_cache_path_rejects_traversal_stem` only exercises relative `../` traversal, not an absolute-path stem

**File:** `notebooks/pu_manifold/tests/test_pu_manifold.py:102-104`
**Issue:** The only containment-guard test is:

```python
def test_cache_path_rejects_traversal_stem():
    with pytest.raises(ValueError):
        cache_mod.cache_path("../escape", "npz")
```

`_assert_inside_cache` was independently verified (against the real `.venv`, not from
memory) to also correctly reject an absolute-path stem such as `"/etc/passwd"` --
`Path(CACHE_DIR) / "/etc/passwd.npz"` resolves to `/etc/passwd.npz` because pathlib's
`/` operator discards the left operand when the right operand is absolute, and
`_assert_inside_cache` does catch this via its resolved-path containment check. The
implementation is correct today, but this is exactly the kind of guard whose test
coverage should be as broad as its threat surface (per the module's own T-01-01/T-01-03
framing) -- a future refactor of `_assert_inside_cache` (e.g. switching from
`.resolve()`-based containment to a naive string-prefix check) would not be caught by
the existing test suite.
**Fix:** Add a parametrized case for an absolute-path stem, e.g.
`cache_mod.cache_path("/etc/passwd", "npz")`, asserting it also raises `ValueError`.

### IN-03: `PLATEAU_TIE_BREAK` is declared as a configurable constant but only one value is actually supported, enforced by a message-less `assert`

**File:** `notebooks/01_manifold_and_gate.ipynb:§4.0 (code cell index 61)` (declaration)
and `§5.1 (code cell index 76)` (enforcement)
**Issue:** `PLATEAU_TIE_BREAK = "lower"` is declared alongside the other genuinely
tunable §4.0 pre-registered constants, but §5.1 enforces it with a bare
`assert PLATEAU_TIE_BREAK == "lower"` -- no other value is implemented anywhere, and the
assert carries no message, so setting it to anything else produces an opaque
`AssertionError` with no explanation of what went wrong or what values are valid.
**Fix:** Either implement the alternative ("higher") branch this constant implies exists,
or replace the pretense of configurability with a plain code comment, and if kept, give
the assert a descriptive message (`assert PLATEAU_TIE_BREAK == "lower", f"Only
PLATEAU_TIE_BREAK='lower' is implemented, got {PLATEAU_TIE_BREAK!r}"`).

### IN-04: "regardless of how the kernel was started" overstates what §0.1's `sys.path` fix-up actually guarantees

**File:** `notebooks/01_manifold_and_gate.ipynb:§0.1 (code cell index 3)`
**Issue:**

```python
NOTEBOOK_DIR = Path.cwd()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))
```

with the comment "Make the notebook-local pu_manifold package importable regardless of
how the kernel was started." This only works if the kernel's current working directory
happens to already equal the notebook's own directory. Standard interactive Jupyter
(Notebook/Lab) generally does start kernels with cwd set to the notebook's directory, so
the common case is fine, but the guarantee does not hold "regardless of how the kernel
was started": `jupyter nbconvert --execute` and `papermill`, for example, commonly run
with cwd equal to wherever the command was invoked from, not the notebook's directory --
in which case this cell inserts the *wrong* directory into `sys.path`, `from
pu_manifold import ...` in cell 20 fails, and separately §4.0's self-referential
pre-registration check (cell 73's `Path.cwd() / "01_manifold_and_gate.ipynb"`) also
fails to locate the notebook file.
**Fix:** Either derive the notebook's directory more robustly (e.g. via the kernel
connection file, or `%pip`/`%config`-provided path in environments that expose one), or
narrow the comment's claim to match what the code actually guarantees (cwd-at-kernel-start
equals the notebook's directory) so a future reader troubleshooting an import failure
under a headless execution path isn't misled by the stronger claim.

---

_Reviewed: 2026-07-31T04:30:58Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
