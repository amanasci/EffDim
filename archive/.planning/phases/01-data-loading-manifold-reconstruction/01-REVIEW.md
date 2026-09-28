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

**Reviewed:** 2026-07-31T04:30:58Z · **Depth:** standard · **Files Reviewed:** 8 · **Status:** issues_found

## Summary

`cache.py`/`subsample.py` are solid: `_assert_inside_cache`'s path-containment guard verified
empirically against both a `../`-relative traversal stem and an absolute-path stem (e.g.
`/etc/passwd`); the alignment smoke test's statistics are sound; the 14-test suite passes cleanly.
`curvature.py`/`mknn.py` are intentional Phase 3/4 stubs per D-02, out of scope.

The one Critical finding is in the notebook: the Phase 1->2 handoff artifact
(`phase1_handoff_{fit_key}.json`, §5.3) is cached under a key that omits the §4.0 pre-registered
constants (`PLATEAU_THRESH`, `SWEEP_K_RANGE`, `GEO_PAIR_COUNT`/`SEED`, etc.) that determine the
`k_star_selection` data being recorded — exactly the "silently reuse a stale artifact" threat
T-01-03 `cache.py`'s manifest system exists to prevent. §4.3's stage-2 sweep cells defend against
an analogous gap; §5.3 has no equivalent guard despite being the most-trusted, most downstream
artifact of the phase.

## Critical Issues

### CR-01: `phase1_handoff` cache key omits the pre-registered sweep constants it records, risking silent stale-provenance reuse

**File:** `01_manifold_and_gate.ipynb §5.3 (cell 84)`. `json_cache(f"phase1_handoff_{fit_key}",
ANALYSIS_CFG, ...)` keys its cache-hit check on `ANALYSIS_CFG` alone (`dataset`/`seed`/`n_rows`/
`normalize`/`n_neighbors`/`n_components`/`eigen_solver` + 3 lib versions) — it does **not**
include `PLATEAU_THRESH`, `SWEEP_K_RANGE`, `K_EXTENSIONS`, `K_CEILING`, `K_WARN_ABOVE`,
`MIN_PLATEAU_RUN`, `STAGE2_MAX_FITS`, `PLATEAU_TIE_BREAK`, `GEO_PAIR_COUNT`, or `GEO_PAIR_SEED` —
the §4.0 constants that actually determine `STABILITY_TABLE`/`PLATEAU_RUNS`/`K_STAR`. If any of
those is tuned between runs (e.g. relaxing `PLATEAU_THRESH["procrustes_disparity_max"]` per the
notebook's own remediation text) and the tuning happens not to change the resulting
`K_STAR`/`N_COMPONENTS`, `ANALYSIS_CFG` and hence `fit_key` are bit-identical to the prior run, so
`json_cache` silently returns the **old** JSON verbatim — stale `thresholds`/`plateau_runs`/
`chosen_run`/`criterion` fields even though this run used different §4.0 constants to reach the
same `k*`. §4.3 (cell 70) already guards an analogous gap for the stage-2 sweep npz records via
explicit `geo_pair_count`/`geo_pair_seed` re-verification asserts; §5.3 has no equivalent.

**Fix:** either (a) widen the `cfg` passed to this `json_cache` call to include every §4.0
constant feeding `k_star_selection` (mirror §4.2's `fit_cfg` pattern), or (b) add an explicit
re-verification assert on cache hit comparing `cached["k_star_selection"]["thresholds"]` against
the current `PLATEAU_THRESH`, refusing to silently reuse a stale handoff.

## Warnings

### WR-01: Cache writes are not atomic; an interrupted manifest write crash-loops with an unhandled `JSONDecodeError`

**File:** `pu_manifold/cache.py` (`_write_manifest`, `_manifest_matches`, and the write calls in
`npz_cache`/`joblib_cache`/`json_cache`). None use a temp-file-then-rename pattern; an interruption
mid-write (kernel restart, OOM-kill — the notebook's own §4.2/§5.2 comments flag a ~2-3 GiB
transient peak during the multi-minute Isomap fits this cache wraps, so this is a realistic
failure mode) leaves a truncated `.meta.json`. Verified concretely: a manually truncated manifest
raises `JSONDecodeError: Expecting ',' delimiter: line 1 column 8 (char 7)`. `_manifest_matches`
has no try/except around `json.loads`, so every subsequent cache call for that stem raises the
same unhandled error and the corrupted manifest persists until a human deletes it — for a
multi-minute ~1 GB artifact, a meaningfully worse failure mode than the module's other,
descriptively-`ValueError`'d error paths.

**Fix:** write to a temp file in `CACHE_DIR` and `os.replace()` it into place for both artifact and
manifest (atomic on POSIX/Windows); wrap `_manifest_matches`'s `json.loads` to raise the module's
own descriptive `ValueError` (naming the corrupt path) instead of letting `JSONDecodeError` propagate.

### WR-02: `l2_normalize` only guards exact-zero norms; near-zero norms silently degrade precision without warning

**File:** `pu_manifold/subsample.py:96-121`. The only check is `if np.any(norms == 0): raise
ValueError(...)`. Verified a row with a tiny non-zero norm (e.g. `1e-100`) passes and produces a
numerically degraded "unit" vector with no error; for a near-zero 768-component row (`~2.77e-159`
norm), the resulting unit-row norm is `1.0000030063809708` — off by ~3e-6, which is *smaller* than
the notebook's own §1.6 unit-norm sanity check (`< 1e-5`), so a sufficiently degenerate raw norm
could pass every guard while carrying corrupted directional information into alignment, the Isomap
fit, and eventually Phase 3's curvature estimates. Real DINOv3 embeddings are unlikely to hit this,
which is presumably why the committed run didn't catch it, but the function itself has no defense
beyond the exact-zero boundary.

**Fix:** add a relative floor (e.g. `MIN_NORM = 1e-8`) raising `ValueError` naming the offending
min norm, not just an exact-zero check.

### WR-03: `_stage2_k_selection`'s duplicate-index fallback is dead code with inverted logic that contradicts its own docstring

**File:** `01_manifold_and_gate.ipynb §4.2 (cell 66)`. The `while` loop's fallback — meant to top
up `idx` if rounding produced fewer than `max_fits` distinct indices — picks the remaining index
that **minimizes** distance to the existing selection, the opposite of the docstring's stated goal
("maximizes the span"); a correct implementation would `max` over the same distance metric. This
branch is provably unreachable today: `n > max_fits` guarantees consecutive `np.linspace` points
are always >1 apart and round to distinct integers (verified empirically for every `(n, max_fits)`
pair with `4<=n<=39`, `3<=max_fits<n`: zero duplicates). Latent landmine, not a live bug — if the
rounding strategy changes, the fallback would silently cluster extra selections instead of
spreading them, undermining the "maximize the span" contract §4.3's plateau criterion depends on,
with no test coverage to catch the regression.

**Fix:** delete the dead branch (`assert len(idx) == max_fits, "unreachable: ..."`) or fix the
direction if kept as defensive code (`max` instead of `min`).

## Info

- **IN-01** (`01_manifold_and_gate.ipynb §1.6, cell 31`): `assert ALIGNMENT_STATS["z"] > 5.0`
  hardcodes `5.0` instead of importing `subsample.ALIGNMENT_MARGIN_Z` (already exported via
  `pu_manifold.__init__`). Fix: import and assert against the named constant.
- **IN-02** (`test_pu_manifold.py:102-104`): `test_cache_path_rejects_traversal_stem` only exercises
  relative `../` traversal, not an absolute-path stem. `_assert_inside_cache` was independently
  verified to also correctly reject `/etc/passwd`-style absolute stems, but a future refactor
  (e.g. switching to a naive string-prefix check) wouldn't be caught by the existing suite. Fix:
  add a parametrized absolute-path-stem case.
- **IN-03** (`§4.0 cell 61` decl, `§5.1 cell 76` enforcement): `PLATEAU_TIE_BREAK = "lower"` is
  declared as if configurable but only one value is implemented; enforced by a message-less
  `assert PLATEAU_TIE_BREAK == "lower"`. Fix: implement the alternative or replace with a plain
  comment; give the assert a descriptive message if kept.
- **IN-04** (`§0.1 cell 3`): the `sys.path` fix-up comment claims to work "regardless of how the
  kernel was started," but only holds when cwd equals the notebook's directory — `jupyter
  nbconvert --execute`/`papermill` commonly run from elsewhere, in which case the `pu_manifold`
  import and §4.0's self-referential pre-registration check both fail. Fix: derive the notebook
  directory more robustly, or narrow the comment's claim.

---
_Reviewed: 2026-07-31T04:30:58Z_ · _Reviewer: Claude (gsd-code-reviewer)_ · _Depth: standard_
</content>
