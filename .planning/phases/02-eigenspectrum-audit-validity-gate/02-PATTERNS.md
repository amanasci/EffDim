# Phase 2: Eigenspectrum Audit & Validity Gate - Pattern Map

**Mapped:** 2026-07-31
**Files analyzed:** 1 (all work is new §6+ content inside one existing notebook; no new source files)
**Analogs found:** 6 / 6 (all analogs are sections of the same notebook, `notebooks/01_manifold_and_gate.ipynb`)

## Scope note

This phase has an unusual file inventory: CONTEXT.md's `<code_context>` and `<decisions>`
sections are explicit that **no new file is created** except two gitignored runtime
artifacts (`spectrum_{key}.npz`, `gate_verdict_{fit_key}.json`) and the notebook itself is
only *appended to*, never newly created. `src/effdim/`, `pyproject.toml`, and
`notebooks/pu_manifold/*.py` are explicitly **not modified** (D-14 declines the `gate.py`
module rejected alternative). There is therefore no "new file vs. existing file" analog
search across the repo tree — the correct analog source is **prior sections of the same
notebook**, because D-01/D-03/D-08/D-11/D-13 all say "mirror the Phase 1 mechanism exactly."

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `notebooks/01_manifold_and_gate.ipynb` §6.0 (pre-registration cell) | config | batch (constants only) | `notebooks/01_manifold_and_gate.ipynb` §4.0 (cell 60-61) | exact |
| §6.x spectrum computation (double-centring, split eigensolve) | transform/utility | batch (dense linear algebra) | §5.2 k* fit cell (cell 79) + `PITFALLS.md` Pitfall 3 recipe | role-match (no prior dense-eigensolve cell exists; closest is the Isomap fit-and-cache cell) |
| §6.x spectrum npz cache call | service (cache I/O) | CRUD (load-or-compute) | §4.2 stage-2 sweep npz cache (cell 67) | exact |
| §6.x R² residual / Tenenbaum curve (point-pair subsampling) | transform | batch | §4.0 `GEO_PAIR_ROWS`/`GEO_PAIR_COLS` subsampling (cell 61) | exact |
| §6.x kneedle elbow + `d`-freeze branch (halt-or-continue) | controller (branch/gate logic) | request-response (assert/halt) | §4.1 connectivity-scan `K_CEILING` halt `for/else` (cell 63) | exact |
| §6.x verdict computation + `gate_verdict.json` write | service (artifact writer) | CRUD (write self-contained JSON) | §5.3 `phase1_handoff_{fit_key}.json` build (cell 84) | exact |
| §6.x D-09 mean-form vs `J`-form equivalence guard | test (inline assertion) | transform + assert | §1.7 alignment negative control (cell 37-40) — the notebook's one small-array self-test pattern | role-match |
| Notebook 03 (Phase 3) first cell — inline gate enforcement (D-14) | controller (guard) | request-response | none in-repo yet (forward-looking, out of this phase's scope but named in D-14/D-16) | no analog (future file) |

## Pattern Assignments

### §6.0 Pre-registration cell (constants + cell-index assertion)

**Analog:** `notebooks/01_manifold_and_gate.ipynb` §4.0, cell 61 (code) and cell 60 (markdown rationale)

**Structure to copy verbatim** (cell 61, lines 1-21 of the extracted cell):
```python
import warnings

# --- D-09 stage-1 base range and D-11 bounded auto-extend ladder ---
SWEEP_K_RANGE = (5, 8, 10, 15, 20, 30)
K_EXTENSIONS = (40, 50)
K_CEILING = 50  # inclusive: k=50 is tried, k=51 is never tried
K_WARN_ABOVE = 30  # strict: k=30 does not warn, k=31 does

# --- D-10 plateau criterion: three thresholds, fixed before any fit exists ---
PLATEAU_THRESH = {
    "procrustes_disparity_max": 0.10,
    "eig_rel_change_max": 0.15,
    "geodesic_spearman_min": 0.95,
}

GEO_PAIR_COUNT = 100_000
GEO_PAIR_SEED = SEED + 1
MIN_PLATEAU_RUN = 3
STAGE2_MAX_FITS = 4
PLATEAU_TIE_BREAK = "lower"
```
Then a `print("=== Section 4.0: ... ===")` block echoing every constant.

**§6.0 must mirror this shape exactly** (per D-03):
```python
R_MAX_PASS = 0.10
M_MAX_PASS = 0.05
R_MAX_MARGINAL = 0.25
M_MAX_MARGINAL = 0.15

D_SWEEP_MAX = ...      # kneedle sweep ceiling == K eigenvector ceiling (D-05/D-10)
R2_PAIR_COUNT = ...    # point-pair subsample size for Tenenbaum R^2 (Claude's discretion)
R2_PAIR_SEED = SEED + 2  # or similar, following the SEED+1 convention at cell 61 line 18

print("=== Section 6.0: pre-registered gate thresholds ===")
...
```

**Cell-index assertion pattern** — §4.3 (cell 73, not fully quoted above but referenced at
cell 67 line 245 `STAGE2_SWEEP = True  # cell-ordering anchor: must not appear earlier than
PLATEAU_THRESH`) shows the idiom: declare a marker variable in the *consuming* cell whose
name states the ordering invariant, and reference it in an assertion later. Also
`_geo_pair_rng`/`GEO_PAIR_ROWS` at cell 61 lines 36-41 shows the "compute and assert shape
immediately after declaring constants" idiom:
```python
_geo_pair_rng = np.random.default_rng(GEO_PAIR_SEED)
GEO_PAIR_ROWS, GEO_PAIR_COLS = _draw_geo_pairs(_geo_pair_rng, len(LS), GEO_PAIR_COUNT)

assert GEO_PAIR_ROWS.shape == (GEO_PAIR_COUNT,)
assert GEO_PAIR_COLS.shape == (GEO_PAIR_COUNT,)
assert not np.any(GEO_PAIR_ROWS == GEO_PAIR_COLS), "self-pairs present in GEO_PAIR_ROWS/COLS"
```
Copy this idiom for the R²-curve's point-pair draw in §6.

---

### §6.x Spectrum computation (double-centring + split eigensolve)

**Analog:** §5.2 k* fit cell (cell 79) for the "define compute fn, time it, print peak-size
expectations before running" shape; no prior cell in the notebook does dense linear algebra
on a 10,000×10,000 array, so the numerical recipe itself is not present in-repo — pull it
from `PITFALLS.md` Pitfall 3 (named explicitly in CONTEXT.md's canonical refs) rather than
from a codebase analog.

**Shape to copy from cell 79** (the "print expected memory before computing, define a
`_fit_x()` closure, time it, assert on the result" idiom):
```python
_expected_dist_matrix_bytes_kstar = 10_000 * 10_000 * 8
print(f"Expected dist_matrix_ size at k*: ...")
...

_kstar_fit_seconds = {}


def _fit_kstar():
    t0 = time.perf_counter()
    model = Isomap(...)
    model.fit(LS)
    _kstar_fit_seconds["fit_seconds"] = time.perf_counter() - t0
    return model


isomap_kstar = joblib_cache(f"isomap_{fit_key}", ANALYSIS_CFG, _fit_kstar)

print("\n=== Section 5.2: k* fit result ===")
print("dist_matrix_.shape:", isomap_kstar.dist_matrix_.shape)
assert isomap_kstar.dist_matrix_.shape == (10_000, 10_000), (...)
```
Apply this same "closure + timed compute + assert-on-result + printed summary" shape to
the double-centring cell: a `_compute_spectrum()` closure that (1) loads `dist_matrix_` via
`joblib.load(..., mmap_mode="r")` per D-12, (2) drops the `Isomap` object, (3) does the
in-place mean-form double-centring on the squared-distance array, (4) runs
`scipy.linalg.eigvalsh(B)` then `scipy.linalg.eigh(B, subset_by_index=[n-K, n-1])`, (5)
prints peak RSS alongside timing (mirroring the `fit_seconds` print at cell 79 lines
79-91, which already has the "print fit_seconds if fresh, else note it came from cache"
branching to copy for the "compute vs. cache hit" distinction).

**Memory-expectation-printed-before-compute pattern** — also cell 67 lines 233-243 (stage-2
sweep) shows printing expected array sizes and noting *transient peak* separately from
*resident* size, which is exactly D-12's ask ("prints peak RSS alongside the timing").

---

### §6.x Spectrum npz cache

**Analog:** §4.2 stage-2 sweep cell (cell 67), specifically the `npz_cache` call shape

**Core pattern** (cell 67, lines 231-292 — reuse `npz_cache` from `pu_manifold`; the cfg dict
composed at lines 250-253 and the compute closure returning a dict of arrays at lines
256-288 is the template):
```python
from pu_manifold import cache_path, npz_cache

for k in STAGE2_K:
    fit_cfg = dict(ANALYSIS_CFG)
    fit_cfg["n_neighbors"] = k
    key_k = config_key(fit_cfg)
    stem = f"sweep_k{k}_{key_k}"

    def _fit_k(k=k):
        ...
        return {
            "embedding": model.embedding_.astype(np.float32),
            "eigenvalues_truncated": np.asarray(model.kernel_pca_.eigenvalues_, dtype=np.float64),
            ...
        }

    record = npz_cache(stem, fit_cfg, _fit_k)
```
For §6 (D-11), the cfg dict is `dict(fit_cfg=fit_key, K=K_CEILING)` (or equivalent — must
include `fit_key` **plus** the `K` ceiling per D-11), the stem is
`f"spectrum_{fit_key}_{key}"` (or similar), and the compute closure returns
`{"eigvals_all": ..., "eigvecs_top_k": ..., "residual_tenenbaum": ..., "residual_eigen": ...}`.
**Do not write a parallel caching path** — `pu_manifold.cache.npz_cache` is the only
sanctioned mechanism (explicit in CONTEXT.md `<code_context>` Integration Points).

---

### §6.x Point-pair subsampling for Tenenbaum R²

**Analog:** §4.0 `_draw_geo_pairs` (cell 61, lines 24-41)

**Core pattern to copy** (self-pair rejection loop + shape assertions immediately after draw):
```python
def _draw_geo_pairs(rng, n_rows_total, count):
    """Draw `count` off-diagonal (row, col) index pairs, self-pairs rejected and redrawn."""
    rows = rng.integers(0, n_rows_total, size=count)
    cols = rng.integers(0, n_rows_total, size=count)
    self_pairs = rows == cols
    while np.any(self_pairs):
        n_bad = int(self_pairs.sum())
        cols[self_pairs] = rng.integers(0, n_rows_total, size=n_bad)
        self_pairs = rows == cols
    return rows, cols


_geo_pair_rng = np.random.default_rng(GEO_PAIR_SEED)
GEO_PAIR_ROWS, GEO_PAIR_COLS = _draw_geo_pairs(_geo_pair_rng, len(LS), GEO_PAIR_COUNT)

assert GEO_PAIR_ROWS.shape == (GEO_PAIR_COUNT,)
assert GEO_PAIR_COLS.shape == (GEO_PAIR_COUNT,)
assert not np.any(GEO_PAIR_ROWS == GEO_PAIR_COLS), "self-pairs present in GEO_PAIR_ROWS/COLS"
```
This `_draw_geo_pairs` function is directly reusable (same signature: rng, n_rows_total,
count) — the planner should consider **calling it again** with a new seed rather than
reimplementing, since `LS` and the row-index space are unchanged in §6. Note `GEO_PAIR_ROWS`/
`GEO_PAIR_COLS` from §4.0 already exist keyed to `GEO_PAIR_SEED = SEED + 1`; §6 needs its own
draw (different seed, and D-06 says the R² pair count is a separate pre-registered constant),
so a fresh `_draw_geo_pairs(rng, 10_000, R2_PAIR_COUNT)` call with a distinct seed (document
the seed choice — e.g. `SEED + 2`) is the closest-fit approach, not a new pairing scheme.

---

### §6.x Kneedle elbow branch — halt-if-`d`-exceeds-18 (D-08)

**Analog:** §4.1 connectivity-scan ceiling `for/else` halt (cell 63, lines 110-139)

**Core `for/else` documented-halt pattern to copy exactly:**
```python
if not CONNECTED_K:
    print("\nNo base-range k is connected -- entering the bounded auto-extend ladder "
          f"K_EXTENSIONS = {K_EXTENSIONS} (K_CEILING = {K_CEILING}, inclusive).")
    for k in K_EXTENSIONS:
        row = _scan_k(k)
        CONNECTIVITY_SCAN.append(row)
        print(f"  extended k={k}: ...")
        if row["n_components"] == 1:
            CONNECTED_K = [k]
            break
    else:
        assert False, (
            "k-NN graph connectivity scan reached K_CEILING=50 (inclusive) and no k in "
            f"SWEEP_K_RANGE + K_EXTENSIONS = {SWEEP_K_RANGE + K_EXTENSIONS} yields a "
            "single connected component. A documented halt here is a legitimate, "
            "complete result -- ... Three remediation options: "
            "(1) ...; (2) ...; (3) ..."
        )
```
For D-08, the shape is an `if elbow <= 18: freeze d = elbow ... else: assert False, "<halt
message with the elbow value, required n_components, cost estimate, and the exact constant
to change>"` — same "documented halt with all remediation options enumerated in the
assertion message" idiom named in CONTEXT.md's `<specifics>` and inherited directly from
this cell. Note the companion markdown cell 64 ("The D-11 tension, recorded rather than
silently resolved...") is the pattern for stating *why* the branch exists and what tradeoff
it encodes — write an equivalent markdown cell before D-08's branch.

**A second, shorter halt example** exists at cell 66 (`_stage2_k_selection` /
`STAGE2_MAX_FITS` assert, lines 213-222) — a plain `assert len(...) >= 3, "<message with
remediation>"` without the `for/else`, useful if D-08's branch turns out simpler than a
loop (it is a single `if/else`, not a ladder, so this simpler assert-with-message shape may
be the more literal fit — prefer this over the `for/else` scaffolding since D-08 has only
two branches, not a ladder of retries).

---

### §6.x Verdict computation + `gate_verdict_{fit_key}.json` write

**Analog:** §5.3 Phase 1→Phase 2 handoff build (cell 84, full cell) + `json_cache` from
`pu_manifold.cache`

**Core pattern to copy** (build a flat dict from already-computed notebook state, write via
`json_cache`, then print a formatted key/value dump and assert on the required-keys
contract):
```python
_phase1_handoff_built = {
    "phase": 1,
    "fit_key": fit_key,
    ...
    "flags": {
        "short_circuit_risk": bool(SHORT_CIRCUIT_RISK),
        "k_auto_extended": bool(K_AUTO_EXTENDED),
        "n_components_no_headroom": True,
    },
    "notes_for_phase2": _notes_for_phase2,
}

PHASE1_HANDOFF = json_cache(
    f"phase1_handoff_{fit_key}", ANALYSIS_CFG, lambda: _phase1_handoff_built
)

_phase1_handoff_path = cache_path(f"phase1_handoff_{fit_key}", "json")
print(f"phase1_handoff written to: {_phase1_handoff_path.name}\n")

print("=== Section 5.3: Phase 1 -> Phase 2 handoff (phase1_handoff_{fit_key}.json) ===")
for _top_key, _top_val in PHASE1_HANDOFF.items():
    if isinstance(_top_val, dict):
        print(f"| {_top_key}")
        for _sub_key, _sub_val in _top_val.items():
            print(f"|   {_sub_key:24s} = {_sub_val}")
    elif isinstance(_top_val, list):
        print(f"| {_top_key}")
        for _i, _item in enumerate(_top_val):
            print(f"|   [{_i}] {_item}")
    else:
        print(f"| {_top_key:24s} = {_top_val}")

assert set(PHASE1_HANDOFF["flags"].keys()) == {...}
assert len(PHASE1_HANDOFF["notes_for_phase2"]) >= 5
```
For `gate_verdict_{fit_key}.json` (D-13/D-16), build the dict with: `verdict`, `r`, `m`, the
three thresholds **verbatim** (`R_MAX_PASS`, `M_MAX_PASS`, `R_MAX_MARGINAL`, `M_MAX_MARGINAL`),
elbow value, elbow criterion name, frozen `d`, the d-axis sweep range, `K` ceiling, `fit_key`,
`k_star`, the Phase 1 flags (copied straight from `PHASE1_HANDOFF["flags"]` per D-04 — spectral
only, flags never move the verdict), timestamp, library versions, and — on FAIL — the
enumerated remediation list embedded in the JSON itself (not just the halt message), per D-16.
Key the `json_cache` cfg dict on `{"fit_key": fit_key, "K": K_CEILING_OR_SWEEP_MAX, ...thresholds}`
so the manifest mechanism binds the verdict to the exact thresholds it was judged under (same
binding argument D-11/D-13 make for the spectrum npz).

**Load the Phase 1 handoff first** (this is a new read pattern §6 needs that no prior cell
demonstrates, since §5.3 only *writes* json_cache, never reads one back for its own
consumption elsewhere in this notebook) — use the same `json.loads(path.read_text())` shape
`_manifest_matches` uses internally (`pu_manifold/cache.py` lines 138-148), or simply
`json.load(open(cache_path(f"phase1_handoff_{fit_key}", "json")))`.

---

### §6.x D-09 mean-form vs literal-`J`-form equivalence guard

**Analog:** §1.7 alignment negative control (cell 37-40) — the notebook's one existing
"deliberate small self-test, not scaffolding" cell, and the one place the notebook already
departs from its no-`try`/`except` convention for a stated reason (relevant if the
equivalence guard needs any exception handling, though `np.testing.assert_allclose` needs
none).

**Pattern:** build a small random matrix (CONTEXT.md's `<specifics>` says 50×50), compute
`B` two ways, and assert. Keep it as visible, stated code — mirroring the way §1.7's cell
37-40 states in prose *why* the negative-control test exists and what it demonstrates,
immediately before the assertion:
```python
_rng_equiv = np.random.default_rng(<some pre-registered seed>)
_D2_test = _rng_equiv.random((50, 50)) ** 2
_D2_test = (_D2_test + _D2_test.T) / 2  # symmetric

# Mean-form (D-09's optimisation)
_row_mean = _D2_test.mean(axis=1, keepdims=True)
_col_mean = _D2_test.mean(axis=0, keepdims=True)
_grand_mean = _D2_test.mean()
_B_mean_form = -0.5 * (_D2_test - _row_mean - _col_mean + _grand_mean)

# Literal J-form (PITFALLS.md's baseline recipe)
_n_test = _D2_test.shape[0]
_J = np.eye(_n_test) - np.ones((_n_test, _n_test)) / _n_test
_B_j_form = -0.5 * _J @ _D2_test @ _J

np.testing.assert_allclose(_B_mean_form, _B_j_form, atol=<tolerance>, rtol=<tolerance>)
print("D-09 equivalence guard: mean-form and literal-J-form B agree to floating tolerance.")
```
Tolerance and matrix size are Claude's-discretion items per CONTEXT.md — `atol=1e-10` is a
reasonable default for float64 on a 50×50 matrix; state the choice explicitly in a markdown
cell, mirroring how every other pre-registered constant in this notebook is justified in
prose immediately above or below its code cell.

---

## Shared Patterns

### Cache contract (config-hash keyed, gitignored, manifest-verified)
**Source:** `notebooks/pu_manifold/cache.py` — `config_key`, `cache_path`, `npz_cache`,
`json_cache`, `joblib_cache`
**Apply to:** the spectrum npz cache and the `gate_verdict.json` write. Do not add any
parallel caching path; import directly (`from pu_manifold import cache_path, npz_cache,
json_cache, config_key`) exactly as §4.2 (cell 67) and §5.3 (cell 84) already do.
```python
from pu_manifold import cache_path, npz_cache
```

### Pre-registration + cell-index-assertion idiom
**Source:** §4.0 (cells 60-61), extended at §4.3's `STAGE2_SWEEP` ordering marker
**Apply to:** §6.0's threshold cell — every `r`/`m`/elbow-sweep/K/pair-count constant
declared once, printed, and referenced (never re-declared) by every later §6 cell.

### Documented halt with enumerated remediation
**Source:** §4.1's `K_CEILING` `for/else` (cell 63, lines 128-139) and §4.2's
`STAGE2_MAX_FITS` assert (cell 66, lines 213-219)
**Apply to:** D-08's elbow-exceeds-18 halt and SPEC-07's FAIL-path halt message — both must
read as an `assert False, "<message covering: what was observed, the cost of each fix, and
the exact constant to edit>"`.

### Self-contained JSON artifact carrying its own decision inputs
**Source:** §5.3's `phase1_handoff_{fit_key}.json` build (cell 84) — thresholds copied
verbatim into the artifact (`"thresholds": dict(PLATEAU_THRESH)`), flags copied as booleans,
notes as a list of prose strings.
**Apply to:** `gate_verdict_{fit_key}.json` (D-16) — same shape, same "print a formatted dump
of every top-level key after writing" closing block, same "assert the required-keys contract
before declaring the section done" closing assert.

### Timed-compute-with-cache-hit-branch idiom
**Source:** §5.2 (cells 79-80) — `_fit_kstar` closure, `fit_seconds` dict populated only on
a real compute, `if fit_seconds is not None: ... else: print("... already cached")`.
**Apply to:** the spectrum double-centring + eigensolve cell, and the RSS/timing print D-12
asks for.

## No Analog Found

| File / Section | Role | Data Flow | Reason |
|---|---|---|---|
| Numerical double-centring + `scipy.linalg.eigvalsh`/`eigh` split-eigensolve recipe | transform | batch | No prior cell in the notebook performs dense eigendecomposition on a 10,000×10,000 array; use `PITFALLS.md` Pitfall 3's recipe as the primary source instead of an in-repo analog (D-09/D-10 already specify the exact recipe in CONTEXT.md, so this is a spec-to-implement rather than a missing pattern) |
| Kneedle / maximum-curvature elbow-finder implementation | transform | batch | No existing elbow-detection code in `src/effdim/`, `notebooks/pu_manifold/`, or the notebook; D-05 requires a from-scratch deterministic implementation (normalize both axes, max distance from chord) — implement directly, no analog to copy |
| Phase 3 notebook's first-cell inline gate enforcement (D-14) | controller | request-response | Notebook 03 does not exist yet; out of this phase's scope, named only as a forward interface contract in D-14/D-16 |

## Metadata

**Analog search scope:** `notebooks/01_manifold_and_gate.ipynb` (all 90 cells scanned by
header/content), `notebooks/pu_manifold/cache.py` (full file read), `notebooks/pu_manifold/
tests/test_pu_manifold.py` (header scanned for testing convention), `notebooks/.cache/
phase1_handoff_43cf438bc944c509.json` (read for concrete field shapes). `src/effdim/` was
not searched further — CONTEXT.md's `<code_context>` states explicitly it is "not called by
this phase."
**Files scanned:** 1 notebook (90 cells), 1 cache module, 1 test file, 1 handoff JSON artifact
**Pattern extraction date:** 2026-07-31
