# Phase 9: Curvature-Conditioned Label Decodability (Physics Replication) - Pattern Map

**Mapped:** 2026-09-02
**Files analyzed:** 8 (two new modules, two new runners, two new test files, one reporting
notebook, one freeze/preregistration doc)
**Analogs found:** 8 / 8 (one is a role-match, not exact)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|---------------|
| `notebooks/pu_manifold/physics_labels.py` | service/utility (data loader + statistical proof) | file-I/O + transform | `notebooks/pu_manifold/subsample.py` | role-match — seeded row-alignment-safe loader shape, no exact prior "external label join" analog |
| `notebooks/pu_manifold/physics_curvature_probe.py` (frozen constants, verdict rule, OOF wrapper, positive control, freeze guard) | utility/config (pure numpy/scipy statistic module) | transform/batch | `notebooks/pu_manifold/crossmodal_curvature.py` | exact — frozen-constants block, `D_SWEEP`, `split_indices`, `plant_positive_control`, `assert_preregistered` all same shape |
| `notebooks/pu_manifold/tests/test_physics_labels.py` | test | file-I/O (mocked) + batch | `notebooks/pu_manifold/tests/test_density_stratified_null.py` | exact — freeze-guard sweep + git-ancestry idiom |
| `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` | test | batch (in-memory) | `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` (+ `test_cka_import_purity.py` for the import-purity check) | exact |
| `notebooks/diagnostics/09_row_alignment_proof_run.py` | runner/CLI script | batch / file-I/O (parquet read, JSONL append) | `notebooks/diagnostics/07_crossmodal_curvature_run.py` | exact — `--mode`, `_strict_ancestor_or_exit`, `append_record_row` shape; new since D9-05..08 has no prior runner precedent for a row-shift proof |
| `notebooks/diagnostics/09_physics_curvature_run.py` | runner/CLI script | batch / file-I/O (npz read, JSONL append) | `notebooks/diagnostics/08_cka_alignment_run.py` and `notebooks/diagnostics/07_crossmodal_curvature_run.py` | exact |
| `notebooks/09_physics_replication_report.ipynb` + `09-FINDINGS.md` | reporting notebook/doc | request-response (read record, print verdict) | Phase 7/8 reporting notebooks (`notebooks/07_crossmodal_curvature_check.ipynb`-class, `08-FINDINGS.md`) | role-match — not opened this pass (budget), same caution 08-PATTERNS.md gave |
| `.planning/phases/09-.../09-PREREGISTRATION.md` | config/doc | request-response | `.planning/phases/08-.../08-PREREGISTRATION-AMENDMENT-01.md` and Phase 7's freeze-commit pattern | exact |

## Pattern Assignments

### `notebooks/pu_manifold/physics_labels.py` (service/utility, file-I/O + transform)

**Analog:** `notebooks/pu_manifold/subsample.py` (role-match — this is a NEW kind of loader,
joining two separate HF datasets by row-index convention rather than reading one dataset's
paired columns, so no exact "external catalog join" precedent exists in-repo).

**Module docstring / seeded determinism framing to copy** (`subsample.py` lines 1-8):
```python
"""Seeded, row-alignment-safe subsampling of ``UniverseTBD/pu-embeddings``.

No object_id exists in this dataset -- row order is the only join between the paired
columns, so both are read off ONE sorted seeded index array in a single indexing pass;
two independent selections would silently break alignment. ``assert_alignment`` is the
runtime proof (structural check + permuted-null z-score). ...
"""
```
`physics_labels.py`'s docstring should make the analogous claim explicit for D9-05/06: no
shared id exists between `UniverseTBD/pu-embeddings` (`physics_vit_base_test`) and
`Smith42/galaxies@v2.0` either — the join is a row-index CONVENTION, proved statistically
(D9-06's shifted-row check), not assumed. State the revision requirement
(`revision="v2.0"`, never the default `main`) as a load-bearing assertion, mirroring how
`subsample.py` asserts `EXPECTED_N_TOTAL` to "catch a silently changed upstream file":
```python
# Source: notebooks/pu_manifold/subsample.py lines 20-24
EXPECTED_N_TOTAL = 101_725
# load_subsample asserts the loaded config reports exactly this many rows (T-01-02
# mitigation: catches a silently changed upstream file).
```
Phase 9's loader needs the equivalent for `EXPECTED_N_PHYSICS_ROWS = 86_471` on BOTH the
embeddings side and the labels side, plus an explicit revision-pin assertion
(`assert dataset_info.sha == known_v2_ref` or a schema-column check for `mag_r_desi`) since
the Common Pitfalls section of RESEARCH.md found the default revision silently lacks every
label column.

**Deterministic seeded draw to reuse directly** (`subsample.py` lines 37-56, `draw_row_indices`):
```python
def draw_row_indices(n_total: int, n_rows: int, seed: int) -> np.ndarray:
    """Deterministic sorted duplicate-free sample (DATA-03). ..."""
    if n_rows < 2:
        raise ValueError(...)
    ...
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, n_rows, replace=False))
```
D9-03's "512 anchors, seeded uniform draw ... from the AE holdout rows only" (D9-04) should
call this exact function with `n_total = len(holdout_idx)`, then index into `holdout_idx` —
do not reimplement a seeded draw from scratch; this function already exists and is tested.

**Column-projected parquet read** (RESEARCH.md Code Examples, verified live):
```python
import pyarrow.parquet as pq
url = "hf://datasets/Smith42/galaxies@v2.0/data/test-00000-of-00016.parquet"
table = pq.read_table(url, columns=["dr8_id", "mag_r_desi", "photo_z",
                                     "smooth-or-featured_smooth_fraction", "mass_med_photoz"])
```
Never call `load_dataset("Smith42/galaxies", split="test")` without `revision="v2.0"` —
the default revision has no label columns at all (RESEARCH.md Pitfall 1).

**Sentinel-value masking pattern to mirror** (RESEARCH.md Security Domain, citing the
colleague's own `load_catalog_label`):
```python
y[y == -99.0] = np.nan   # apply before any statistic, mirrored per-label
```

---

### `notebooks/pu_manifold/physics_curvature_probe.py` (utility/config, transform/batch)

**Analog:** `notebooks/pu_manifold/crossmodal_curvature.py` — same shape: frozen constants
block, `split_indices`, `plant_positive_control`, `assert_preregistered`, `VERDICT_RULE`.

**Frozen constants + fresh-redeclaration discipline** (`density_stratified_null.py` lines
30-39, the exact discipline this new module must also follow relative to `crossmodal_curvature.py`):
```python
"""**This module adds; it does not edit.** ``notebooks/pu_manifold/crossmodal_curvature.py``
(Phase 7, sealed by D7-05) is never imported for a gating VALUE here -- every constant this
module needs is re-declared as a fresh top-level literal, even where the value is identical to
Phase 7's own ...

**The constants below are FROZEN.** They are committed in this file, in this commit, before any
[Phase 9] number exists anywhere in the tree. ...
"""
```
`physics_curvature_probe.py` must declare its OWN `D_SWEEP = (16, 20, 25, 32)` (D9-12
forbids editing `crossmodal_curvature.D_SWEEP = (20, 25, 32)`), its own `ALPHA_RIDGE = 100.0`,
`K_NEIGHBOURS = 2048`, `N_ANCHORS = 512`, `SHIFT_SET`, `ALIGNMENT_MARGIN`, seeds, etc. — never
importing a gating value from Phase 7/8 modules. It MAY import pure functions
(`split_indices`, `plant_positive_control`'s *mechanism*, `linear_probe.fit_probe`/`predict_probe`,
`cross_split_curvature.partial_spearman`, `decoder_curvature.plain_decoder_curvature`).

**`split_indices` — reuse verbatim for the AE holdout split (D9-04)** (`crossmodal_curvature.py`
line 392):
```python
def split_indices(n: int, split_seed: int, holdout_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    ...
```
Call with Phase 7's own `SPLIT_SEED = 20260813`, `HOLDOUT_FRACTION = 0.2` values IF Phase 9
intends to match Phase 7's exact AE splits (re-declared as fresh Phase-9-owned constants per
the discipline above, even if numerically identical) — `round(86_471 * 0.2) = 17_294` holdout
rows, from which the 512 anchors (D9-03) are drawn via `draw_row_indices`.

**Positive-control bisection mechanism to adapt, NOT the null it calls** (`crossmodal_curvature.py`
lines 542-582, `plant_positive_control` docstring — the mechanism transfers, the validation
null does not, per RESEARCH.md Pitfall 5):
```python
"""... bisect a candidate ``slope`` over 40 iterations on the bracket ``[0.0, 2.0]`` against the
achieved ``scipy.stats.spearmanr(h_real, planted)`` ... For each target, the final planted array
is run through :func:`two_tailed_permutation_null` using the SAME N_PERMUTATIONS,
PERMUTATION_SEED and NULL_QUANTILE_PER_TAIL the headline test uses ...
Raises ``ValueError`` naming ``h_real`` before any search happens -- guard first ..."""
```
D9-14's adaptation: keep the rank-transform-then-bisect-a-slope mechanism and the
guard-first discipline (raise on constant/non-finite `h_real` before any search), but
retarget the achieved statistic at the **3-control partial** (`cross_split_curvature.
partial_spearman(planted, local_r2, controls=Z)`) and validate through the **Freedman–Lane**
null (Pattern below), never `two_tailed_permutation_null`.

**Assert-preregistered guard shape to copy** (`crossmodal_curvature.py` lines 284-364, and
`08-PATTERNS.md`'s citation of `linear_probe.py` lines 249-383 for the per-constant-message
idiom):
```python
if "D_SWEEP" in g and g["D_SWEEP"]:
    d_sweep = g["D_SWEEP"]
    if not (isinstance(d_sweep, (tuple, list)) and all(isinstance(x, int) and x > 0 for x in d_sweep)):
        missing.append("D_SWEEP (contains a non-positive or non-int entry)")
```
One check per constant, in declared order, `RuntimeError` on first failure, message prefixed
`"assert_preregistered: <CONST>=<value> ..."`. `SEED_HANDLING_RULE` (D9-17's never-pool rule)
must be guarded by exact string equality, not truthiness — copy `linear_probe.py`'s
`if SEED_HANDLING_RULE != "no_pooling_per_seed_verdicts": raise RuntimeError(...)` idiom
verbatim in spirit.

**5-fold OOF ridge wrapper — the one genuinely new function, no exact analog exists**
(`linear_probe.py` lines 412-469, `fit_probe`/`predict_probe` signatures to wrap, and
RESEARCH.md Pitfall 4's warning that calling `fit_probe` once on the full set is NOT OOF):
```python
def fit_probe(X_train, Y_train, alpha_grid, alpha_per_target, fit_intercept) -> Dict[str, Any]:
    """Wraps sklearn.linear_model.RidgeCV(...). Never hand-rolls a CV loop or a
    least-squares solver."""
def predict_probe(fit: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """fit["estimator"].predict(X)"""
```
Write a new `oof_ridge_predict(X, y, alpha, n_folds, fold_seed)` in `physics_curvature_probe.py`
that explicitly `KFold(n_splits=5, shuffle=True, random_state=fold_seed)`s, calls
`linear_probe.fit_probe(X[train_idx], y[train_idx].reshape(-1,1), alpha_grid=(100.0,),
alpha_per_target=False, fit_intercept=True)` per fold and `linear_probe.predict_probe(fit,
X[test_idx])`, assembling one full-length OOF array — never a single whole-dataset
`fit_probe` call presented as "OOF" (RESEARCH.md's named failure mode).

**`||H_tan||` decomposition — copy verbatim** (`08_radial_curvature_decomposition_run.py`
lines 114-127, already cited exactly in RESEARCH.md Pattern 3):
```python
def decompose(H_vec, image):
    img_norm = np.linalg.norm(image, axis=1)
    u = image / img_norm[:, None]
    H_rad = np.einsum("ij,ij->i", H_vec, u)
    H_tan = H_vec - H_rad[:, None] * u
    return H_rad, np.linalg.norm(H_tan, axis=1), img_norm
```
Requires keeping `plain_decoder_curvature`'s `H_vec` key (not just the norm) and
`model.decode(z)` — signature confirmed at `decoder_curvature.py` line 161
(`plain_decoder_curvature(model, z) -> Dict[str, Any]`, `H = tr_g(II)` trace convention).

**3-control partial Spearman — reuse directly, never reimplement** (`cross_split_curvature.py`
lines 232-262, already cited in RESEARCH.md Pattern 2):
```python
def partial_spearman(x, y, controls=None):
    rx, ry = rankdata(x), rankdata(y)
    if controls is None:
        return float(np.corrcoef(rx, ry)[0, 1])
    Z = np.column_stack([rankdata(controls[:, j]) for j in range(controls.shape[1])])
    A = np.column_stack([np.ones(len(rx)), Z])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])
```
Call with `controls = np.column_stack([log_knn_radius, local_label_variance,
local_evaluation_count])` for D9-09's exact statistic.

**Freedman–Lane permutation — port the colleague's own function, don't re-derive it**
(`origin/curvature-experiments:.../inference.py` lines 58-73, verified via `git show`):
```python
def freedman_lane_y(y, Z, rng):
    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    yr = rankdata(y[m]); Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    fit = A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    resid = yr - fit
    y2 = y.copy(); y2[m] = fit + rng.permutation(resid)
    return y2
```
FWER across `d`: `max_d |controlled_rho_permuted(d)|` per draw vs. observed max — his
`permutation_curves`'s `tmax`/`p_global` logic, same file lines 76-130.

**07.1-style stratified null — reuse `density_strata` for binning, write a new loop**
(`density_stratified_null.py` lines 466-482, cited exactly in RESEARCH.md Pattern 5):
```python
def density_strata(density, n_strata):
    order = np.argsort(density, kind="stable")
    n = order.shape[0]; bin_size = n // n_strata
    strata = np.empty(n, dtype=int)
    for i in range(n_strata):
        lo, hi = i * bin_size, (i + 1) * bin_size if i < n_strata - 1 else n
        strata[order[lo:hi]] = i
    return strata
```
Do NOT edit `density_stratified_null.py` to generalize its single-control
`stratified_partial_null` — write a new Phase-9-owned function that bins on
`log_knn_radius` rank via this imported `density_strata`, then permutes within-stratum and
calls `cross_split_curvature.partial_spearman(h_perm, y, controls=Z_full)` inside the loop.

---

### `notebooks/pu_manifold/tests/test_physics_labels.py` (test, file-I/O mocked + batch)

**Analog:** `notebooks/pu_manifold/tests/test_density_stratified_null.py` — freeze-guard
sweep + git-ancestry idiom, same shape as 08-PATTERNS.md already documented:
```python
FREEZE_COMMIT_SHA = "676866657676a36abb639782fa10ecb3061fd688"  # Phase 9 gets its own

def _freeze_commit_is_strict_ancestor_of_head() -> bool:
    is_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE_COMMIT_SHA, "HEAD"], ...)
    if is_ancestor.returncode != 0:
        return False
    count_result = subprocess.run(["git", "rev-list", "--count", f"{FREEZE_COMMIT_SHA}..HEAD"], ...)
    return int(count_result.stdout.strip()) >= 1
```
For the row-alignment proof (D9-06/07), test with a SYNTHETIC small embedding+label pair
(never a live HF download inside a unit test) where the true offset is known by construction,
and assert `R^2(shift 0)` is the unique maximum by more than the frozen margin, and a second
case where NO shift passes and the SEARCH branch (D9-08) is exercised and correctly reports
"found, not assumed."

---

### `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` (test, batch in-memory)

**Analog 1 (freeze-guard/verdict-rule shape):** `notebooks/pu_manifold/tests/test_crossmodal_curvature.py`
— malformed-constant parametrized sweep idiom already established project-wide
(`08-PATTERNS.md` lines 237-250, copy the same `_REQUIRED_CONSTANTS` sweep and
`monkeypatch.setattr` mutation idiom):
```python
@pytest.mark.parametrize("name", pcp._REQUIRED_CONSTANTS)
def test_assert_preregistered_rejects_unset_constant(name):
    ...
```

**Analog 2 (import-purity, already invented in Phase 8 — copy directly, don't re-invent):**
`notebooks/pu_manifold/tests/test_cka_import_purity.py` — this test file already exists and
solves exactly the D8-23-class problem D9-18 also requires ("no sealed module mutated on
import"). Add `physics_curvature_probe` and `physics_labels` to its `SEALED_MODULES` tuple
(or write a Phase-9-scoped copy) rather than reinventing the subprocess-per-import-order
mechanism:
```python
SEALED_MODULES: Tuple[str, ...] = (
    "mknn", "cae", "decoder_curvature", "curvature_probe", "cross_split_curvature",
    "linear_probe", "pointcloud_probe", "crossmodal_curvature", "density_stratified_null",
)
```
This is a **stronger analog than 08-PATTERNS.md had available** — Phase 8 had to invent this
pattern from scratch; Phase 9 does not.

---

### `notebooks/diagnostics/09_row_alignment_proof_run.py` (runner, batch/file-I/O)

**Analog:** `notebooks/diagnostics/07_crossmodal_curvature_run.py` — runner shape,
`_strict_ancestor_or_exit`, `--mode` dispatch, `append_record_row`. No exact prior runner
computed a "row-shift proof," so this is new logic inside a familiar shell.

**Strict-ancestor gate — copy nearly verbatim** (cited exactly in 08-PATTERNS.md from
`07.1_density_stratified_null_run.py` lines 106-160; same function exists in
`07_crossmodal_curvature_run.py`):
```python
def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    if not freeze_commit:
        print("ERROR (D9-18): this mode requires --freeze-commit ...", file=sys.stderr)
        sys.exit(1)
    ...
    is_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", freeze_commit, "HEAD"], ...)
    count = int(subprocess.run(["git", "rev-list", "--count", f"{freeze_commit}..HEAD"], ...).stdout.strip())
    if is_ancestor.returncode != 0 or count < 1:
        print("ERROR (D9-18): --freeze-commit is not a STRICT git ancestor of HEAD.", file=sys.stderr)
        sys.exit(1)
```

**JSONL append with raw-numpy guard — copy verbatim** (already cited exactly in
08-PATTERNS.md from `07.1_density_stratified_null_run.py` lines 188-203):
```python
def append_record_row(row: Dict[str, Any], record_path: Path) -> None:
    for key, value in row.items():
        if isinstance(value, (np.ndarray, np.generic)):
            raise TypeError(f"append_record_row: row[{key!r}] is a raw numpy value ...")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
```
This is the exact defect precedent Phase 9's runner must not repeat.

**`--mode` docstring dispatch shape to copy** (07.1 runner lines 1-20, cited in 08-PATTERNS.md):
```python
"""Phase 9 row-alignment proof runner. `--mode smoke` is the tracer (small synthetic
shift/offset case, no HF download). `--mode proof` runs D9-06/07 for real: OOF R²(shift 0)
vs. every shift in the frozen SHIFT_SET. `--mode search` runs D9-08's SEARCH branch when
shift 0 fails.
"""
```

---

### `notebooks/diagnostics/09_physics_curvature_run.py` (runner, batch/file-I/O)

**Analog:** `notebooks/diagnostics/08_cka_alignment_run.py` (the strongest, most load-bearing
analog per 08-PATTERNS.md's own assessment of its own analog, `07.1_density_stratified_null_run.py`).
Same freeze-gate, record-path resolution, argparse shape:
```python
def resolve_record_path(record_path_arg: Optional[str]) -> Path:
    if record_path_arg is None:
        return cache.cache_path(RECORD_STEM, "jsonl")
    candidate = Path(record_path_arg)
    cache._assert_inside_cache(candidate)
    return candidate
```
**Reusing another runner by file path, not package import** (08-PATTERNS.md lines 376-386,
`07.1_density_stratified_null_run.py` lines 89-93) — if Phase 9's runner needs anything from
`08_radial_curvature_decomposition_run.py` (the `decompose()` function), prefer importing it
as `from pu_manifold import ...` if the logic has been moved into `physics_curvature_probe.py`
(the cleaner path per D9-11's phase-9-owned-module discipline); otherwise load by
`importlib.util` path, never `import notebooks.diagnostics.08_radial_curvature_decomposition_run`
as a package member (`diagnostics/` is a plain directory, not a package).

**Package imports, absolute `pu_manifold.*` style** (08-PATTERNS.md lines 388-408):
```python
from pu_manifold import cache            # noqa: E402
from pu_manifold import cross_split_curvature   # noqa: E402
from pu_manifold import linear_probe     # noqa: E402
from pu_manifold import decoder_curvature  # noqa: E402
from pu_manifold import physics_labels as pl   # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402
```

**`argparse` shape** (08-PATTERNS.md lines 414-425): `--mode {smoke,dsweep,positive-control,
shuffled-label,seeds,selfcheck}`, `--record-path`, `--threads`, `--freeze-commit` (help text
naming D9-18), `--smoke-permutations`.

---

### `notebooks/09_physics_replication_report.ipynb` + `09-FINDINGS.md` (reporting notebook/doc)

**Analog:** not opened this pass (5-analog budget already spent on higher-value machinery,
same call 08-PATTERNS.md made). Closest precedent by name/shape: Phase 7/8's own reporting
notebooks and `08-FINDINGS.md`'s D8-21 caveat-bearing-verdict pattern, plus 07.1's
`SURVIVES AT SUBSET OF d` per-`d` independent-reporting vocabulary (D9-10 explicitly adopts
this). **The planner should have the implementing agent open one of Phase 7/8's actual
committed reporting notebooks directly** for cell-count and verdict-sentence structure before
writing Phase 9's — treat as role-match, not exact-match, pending that read.

## Shared Patterns

### Freeze-before-any-number (`assert_preregistered` + git-ancestry proof)
**Source:** `notebooks/pu_manifold/linear_probe.py` lines 249-383 (guard shape) and
`notebooks/diagnostics/07_crossmodal_curvature_run.py` (`_strict_ancestor_or_exit`, same
function also in `08_cka_alignment_run.py`).
**Apply to:** `physics_curvature_probe.py` and `physics_labels.py` (both need
`assert_preregistered()`), `09_row_alignment_proof_run.py` and `09_physics_curvature_run.py`
(both need the CLI-level ancestry check before any write). This is the fourth-plus phase in
this milestone to use this exact pattern.

### Fresh-redeclaration discipline across a freeze boundary
**Source:** `notebooks/pu_manifold/density_stratified_null.py` lines 30-39 (module docstring).
**Apply to:** `physics_curvature_probe.py` — never import a gating constant (`D_SWEEP`,
`alpha`, seeds) from `crossmodal_curvature.py`; only pure functions cross the boundary.

### Non-gating diagnostics (D7-03 shape)
**Source:** `notebooks/pu_manifold/crossmodal_curvature.py` lines 149-165, 250-265
(`DENSITY_SIGN_CONVENTION`, `DIAGNOSTICS_ARE_NON_GATING = True`).
**Apply to:** `||H||` beside `||H_tan||` (D9-11), raw `rho` beside the controlled partial
(D9-09), secondary labels `photo_z`/`smooth_fraction`/`stellar_mass` (D9-16) — each needs its
own `..._IS_NON_GATING = True`-style constant plus a prose rule string.

### Do-not-pool-seeds, unanimous 3-of-3 (`05-03-DECISION.md`)
**Source:** `notebooks/pu_manifold/linear_probe.py` — `SEED_HANDLING_RULE` (exact-equality
guarded), `combine_seed_verdicts` (lines 831-887, three-HOLDS/zero-HOLDS/split branching).
**Apply to:** D9-17's two-wave seed design — Wave B's per-`d` combination must raise on
anything other than exactly 3 seed entries and never average or upgrade a 1-or-2-of-3 split.

### JSONL record writing with a raw-numpy guard
**Source:** `notebooks/diagnostics/07.1_density_stratified_null_run.py` lines 188-203
(`append_record_row`), reused verbatim in `08_cka_alignment_run.py`.
**Apply to:** both new Phase 9 runners' every write path.

### Import-purity regression test (already invented, do not re-invent)
**Source:** `notebooks/pu_manifold/tests/test_cka_import_purity.py` (full file, ~40+ lines
read; subprocess-per-import-order mechanism, `SEALED_MODULES` tuple, planted-mutation
detection test).
**Apply to:** Phase 9's `physics_labels`/`physics_curvature_probe` — extend `SEALED_MODULES`
or write a Phase-9-scoped sibling test file reusing the same subprocess/snapshot mechanism.

### Freeze-commit ancestry test
**Source:** `notebooks/pu_manifold/tests/test_density_stratified_null.py` lines 33-73.
**Apply to:** `test_physics_labels.py` and `test_physics_curvature_probe.py` — same
`_freeze_commit_exists`/`_freeze_commit_is_strict_ancestor_of_head` helper pair, new
`FREEZE_COMMIT_SHA` once Phase 9's freeze commit lands.

### Sentinel-value / finiteness guard-first convention
**Source:** `notebooks/pu_manifold/linear_probe.py` (`fit_probe` raises on non-2D/non-finite
input), `notebooks/pu_manifold/crossmodal_curvature.py` (`plant_positive_control` raises on
non-finite/constant `h_real` before any search).
**Apply to:** `physics_labels.py`'s label loading (`-99.0` sentinel masking, `mag_r_desi`
missingness), every new statistic function in `physics_curvature_probe.py` — guard first,
never silently propagate a NaN/sentinel into a fit.

## No Analog Found

None — the import-purity test that had no analog in Phase 8 (08-PATTERNS.md's own "No Analog
Found" entry) has since been built (`test_cka_import_purity.py`) and now serves as this
phase's exact analog for the same requirement (D9-18's "no sealed module mutated on import").
The reporting notebook/`09-FINDINGS.md` role-match (not opened this pass) is the only
lower-confidence assignment; flagged above, not listed here since a same-milestone precedent
exists by name even though it was not read line-by-line this session.

## Metadata

**Analog search scope:** `notebooks/pu_manifold/*.py`, `notebooks/pu_manifold/tests/*.py`,
`notebooks/diagnostics/*.py`, `origin/curvature-experiments` (via `git show`, for the
Freedman–Lane / controls machinery only — read-only reference, never imported).
**Files scanned (read or grepped):** `subsample.py` (60 lines read directly),
`crossmodal_curvature.py` (`plant_positive_control` docstring + `D_SWEEP`/`split_indices`
grep, ~80 lines), `linear_probe.py` (`fit_probe`/`predict_probe` signatures, ~60 lines),
`decoder_curvature.py` (`plain_decoder_curvature` docstring, ~20 lines),
`test_cka_import_purity.py` (header + `SEALED_MODULES`, ~40 lines),
`08-PATTERNS.md` (full file, 519 lines, reused as the direct structural template for this
document per the task's explicit instruction to match its format), plus the code excerpts
already verified and quoted in `09-RESEARCH.md` (`cross_split_curvature.partial_spearman`,
`08_radial_curvature_decomposition_run.py`'s `decompose()`, the colleague's
`freedman_lane_y`, `density_stratified_null.density_strata`) which were not independently
re-read here to avoid duplicating already-verified excerpts.
**Pattern extraction date:** 2026-09-02
