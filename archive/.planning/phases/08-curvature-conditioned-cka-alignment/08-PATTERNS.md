# Phase 8: Curvature-Conditioned CKA Alignment - Pattern Map

**Mapped:** 2026-08-27
**Files analyzed:** 5 (new module, freeze constants block, unit tests, runner script, reporting notebook)
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|---------------|
| `notebooks/pu_manifold/cka.py` (estimator: HSIC/CKA, Gram builders, tertile split, stratified null) | utility/service (pure numpy statistic module) | transform/batch | `notebooks/pu_manifold/density_stratified_null.py` | exact — statistic + stratified-permutation-null + frozen-constants-block shape |
| `cka.py`'s frozen constants block (kernels, sigma multipliers, `S` grid, tertile rule, `assert_preregistered`) | config | request-response (validated once at call time) | `notebooks/pu_manifold/linear_probe.py` (freeze machinery) + `notebooks/pu_manifold/crossmodal_curvature.py` (D7-03 non-gating pattern) | exact |
| `notebooks/pu_manifold/tests/test_cka.py` | test | batch (in-memory assertions) | `notebooks/pu_manifold/tests/test_density_stratified_null.py` | exact — freeze-guard sweep + git-ancestry proof idiom |
| `notebooks/diagnostics/08_cka_alignment_run.py` | runner/CLI script | batch / file-I/O (npz read, JSONL append) | `notebooks/diagnostics/07.1_density_stratified_null_run.py` | exact — `--mode` dispatch, `_strict_ancestor_or_exit`, `append_record_row` |
| `notebooks/08_cka_alignment_check.ipynb` + `08-FINDINGS.md` | reporting notebook / doc | request-response (read record, print verdict) | Phase 7 / 07.1 reporting notebooks (not read in full this pass — cell-structure precedent only; see note below) | role-match |

## Pattern Assignments

### `notebooks/pu_manifold/cka.py` (utility, transform/batch)

**Analog:** `notebooks/pu_manifold/density_stratified_null.py` (813 lines) — same shape: frozen
constants block at top, a stratification/split helper, a permutation-null function operating on
precomputed matrices/arrays, no file I/O anywhere in the module.

**Module docstring / freeze-provenance pattern** (`density_stratified_null.py` lines 1-45):
```python
"""Phase 07.1 pre-registration: 07.1's own gating constants block, its guard, and the two
independent verdict rules ...

**This module adds; it does not edit.** ``notebooks/pu_manifold/crossmodal_curvature.py``
(Phase 7, sealed by D7-05) is never imported for a gating VALUE here -- every constant this
module needs is re-declared as a fresh top-level literal, even where the value is identical to
Phase 7's own ...

**The constants below are FROZEN.** They are committed in this file, in this commit, before any
07.1 number exists anywhere in the tree. A later edit to any of them after a 07.1 number exists
is a pre-registration BREACH: the only remedy is a fresh freeze and a fresh run, never a silent
fix (mirrors D7-06's discipline, applied here to 07.1's own constants).
"""
```
`cka.py` must copy this docstring shape verbatim in spirit: re-declare every constant it needs
(kernel names, `sigma` multipliers, `S` grid, permutation count/seed, tertile rule) as **fresh
top-level literals**, never importing a gating value across the freeze boundary from
`density_stratified_null.py` or `crossmodal_curvature.py` — only pure functions
(`density_strata`, `local_density_weights`) may be imported.

**Constants-block idiom** (`density_stratified_null.py` lines 61-95):
```python
D_SWEEP = (20, 25, 32)
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 20260825
NULL_QUANTILE_PER_TAIL = 0.975

N_STRATA_HEADLINE = 20
STRATA_GRID = (10, 20, 50)
STRATIFICATION_RULE = (
    "Strata are equal-count quantile bins on density RANK, not equal-width bins in log-density "
    "(... every stratum identical permutation entropy). Stratum assignment is by "
    "np.argsort(density) POSITION: exactly-tied densities are separated by index order ..."
)
SENSITIVITY_GRID_RULE = (
    "STRATA_GRID = (10, 20, 50) is a grid of THRESHOLDS, not of point estimates ... "
    "It may qualify a reading but never overturn or escalate it ..."
)
```
Phase 8's own analogous constants (`KERNELS = ("linear", "rbf")`, `SIGMA_MULTIPLIERS = (0.5, 1.0,
2.0)`, `S_GRID`, `N_TERTILES = 3`, `TERTILE_STATISTIC_RULE`, `SENSITIVITY_GRID_RULE` for `S`)
should follow this exact "value + prose-rule string naming what it may/may not do" pairing.

**`density_strata` signature to reuse directly** (imported, not re-implemented):
```python
# density_stratified_null.density_strata(density, n_strata) -- ready-made equal-count
# rank-quantile binning; D8-06's within-stratum tertile split is built ON TOP of this,
# never a reimplementation of it.
```

**Stratified-permutation-null shape to copy** (structural precedent for D8-11's null,
`density_stratified_null.py`'s `stratified_partial_null`-style per-stratum
`rng.permutation(idx)` loop — the RESEARCH.md code example already adapts this correctly):
```python
def stratified_tertile_label_null(h, strata, K_full, L_full, n_resamples, seed):
    rng = np.random.default_rng(seed)
    strat_indices = [np.where(strata == s)[0] for s in np.unique(strata)]
    null_by_kernel = {name: np.empty(n_resamples) for name in K_full}
    for b in range(n_resamples):
        h_perm = h.copy()
        for idx in strat_indices:
            h_perm[idx] = h[rng.permutation(idx)]
        tertiles = tertile_split_within_strata(h_perm, strata)
        for name in K_full:
            c3 = cka_on_subset(K_full[name], L_full[name], tertiles[2])
            c1 = cka_on_subset(K_full[name], L_full[name], tertiles[0])
            null_by_kernel[name][b] = c3 - c1
    return null_by_kernel
```
**Do not** reach for `mknn.permutation_null(permutation_type="pairings")` — that nulls global
alignment (a question Phase 7 already answered), not the curvature-conditioning link D8-11
requires. This is named explicitly in RESEARCH.md Pitfall 2 and confirmed by reading
`crossmodal_curvature.py`'s own D7-07 scope note.

**Core CKA/HSIC math** — RESEARCH.md's `unbiased_hsic`/`cka`/`rbf_gram`/`cka_on_subset` code
blocks are the primary source (Song et al. 2012 formula, Gram-once/index-many pattern); no
closer in-repo analog exists since this codebase has never computed CKA/HSIC before. Key
invariant carried from research: **raw, zero-diagonal Gram matrices only — never double-center
(`H K H`) before the unbiased-HSIC correction.**

---

### `cka.py`'s freeze machinery (`assert_preregistered`, non-gating diagnostics)

**Analog 1 — freeze guard shape:** `notebooks/pu_manifold/linear_probe.py` lines 249-383
(`assert_preregistered`). Pattern: one check per constant, in order, raising `RuntimeError` on
the FIRST failure, each message prefixed `"assert_preregistered: <CONST>=<value> ..."` naming
the offending constant and what it must equal/satisfy:
```python
def assert_preregistered() -> None:
    if not isinstance(VERDICT_RULE, str) or not VERDICT_RULE.strip():
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} is empty or not a string."
        )
    if "N_BUCKETS" not in VERDICT_RULE:
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} does not name N_BUCKETS."
        )
    ...
    if SEED_HANDLING_RULE != "no_pooling_per_seed_verdicts":
        raise RuntimeError(
            f"assert_preregistered: SEED_HANDLING_RULE={SEED_HANDLING_RULE!r} does not equal "
            '"no_pooling_per_seed_verdicts" -- the ratified no-pooling decision '
            "(05-03-DECISION.md)."
        )
```
Note the **equality check, not truthiness**, on rule-string constants that encode a ratified
decision (`SEED_HANDLING_RULE != "no_pooling_per_seed_verdicts"`) — D8-15's seed rule (3-of-3
unanimous, never pooled) must be guarded the same way in `cka.py`, by exact string equality, so
a future edit that reintroduces pooling fails loudly.

**Analog 2 — seed-combination helper:** `linear_probe.py` lines 831-887
(`combine_seed_verdicts`) — maps a dict of exactly-three per-seed verdicts onto one of three
terminal phase-level outcomes by counting `"HOLDS"`:
```python
def combine_seed_verdicts(per_seed_verdicts: Dict[int, str], rule: str) -> Dict[str, Any]:
    if not isinstance(rule, str) or not rule.strip():
        raise RuntimeError(
            "combine_seed_verdicts: rule is empty; cannot run before the pre-registration "
            "freeze."
        )
    if not isinstance(per_seed_verdicts, dict) or len(per_seed_verdicts) != 3:
        raise ValueError(
            "combine_seed_verdicts: per_seed_verdicts must hold exactly three seeds, got "
            f"{...}."
        )
    # three HOLDS -> "HOLDS IN ALL THREE SEEDS"
    # zero HOLDS  -> "NO DETECTABLE RELATIONSHIP IN ANY SEED"
    # one or two  -> "SPLIT ACROSS SEEDS"  (terminal, non-supportive, NOT a near-miss)
```
D8-15 inherits this verbatim in shape: Phase 8's `combine_seed_verdicts`-equivalent for the
`d=25` seed axis must raise on anything other than exactly 3 seed entries and must never average
or upgrade a 1-or-2-of-3 split by majority vote.

**Analog 3 — D7-03 non-gating-diagnostic pattern:** `crossmodal_curvature.py` lines 149-165,
250-265 (`DENSITY_SIGN_CONVENTION`, `DIAGNOSTICS_ARE_NON_GATING = True`):
```python
DENSITY_SIGN_CONVENTION = (
    "curvature_probe.local_density_weights returns the per-point INVERSE density w, "
    "mean-normalized to 1. The reported density statistic is taken on 1.0 / w, a RELATIVE "
    "density, matching Phase 4's own printed convention (region_partition_mknn_run.py "
    "REGN-01) so the two phases' density-vs-curvature numbers are comparable rather than "
    "sign-flipped against each other. ..."
)
...
"""Density and hubness (D7-03: spearman(1.0 / w, ||H||) under DENSITY_SIGN_CONVENTION,
mknn.hubness_skewness) are reported alongside every verdict above and gate NONE of it
(DIAGNOSTICS_ARE_NON_GATING = True)."""
```
D8-01 (RBF non-gating), D8-04 (sigma ladder non-gating), D8-10 (middle tertile non-gating), and
D8-12 (`d=32` non-gating) should each copy this `DIAGNOSTICS_ARE_NON_GATING`-style boolean +
prose-rule-string pairing rather than inventing a new non-gating idiom.

**Analog 4 — the exact scope constant Phase 8 supersedes** (never edit; supersede by decision):
`crossmodal_curvature.py` lines 109-111:
```python
ALIGNMENT_METRIC = "mknn"
"""D7-07: CKA is out of scope and not implemented anywhere in this codebase. This constant is
carried on every record row so the exclusion is a positive, checkable fact, not only prose."""
```
Confirmed present at the exact location RESEARCH.md/CONTEXT.md cite. Phase 8's plan must record
the supersession of this scope decision in a `<superseded_decision>` block and must NOT touch
this file.

---

### `notebooks/pu_manifold/tests/test_cka.py` (test, batch)

**Analog:** `notebooks/pu_manifold/tests/test_density_stratified_null.py` (984 lines).

**Header / freeze-ancestry docstring pattern** (lines 1-14):
```python
"""Freeze-guard, verdict-rule, and locally-declared-constant tests for 07.1's own
pre-registration module.

... exercises pure, in-memory constants and the two verdict functions only. No PU data is
loaded, nothing is trained, nothing is read from ``notebooks/.cache/``.

Load-bearing tests: the malformed-constant sweep over ``_REQUIRED_CONSTANTS`` (a constant added
later without a guard entry must fail this suite), the git-ancestry proof
(``test_freeze_commit_is_a_strict_ancestor_of_head``), and the two verdict functions' exact-key
and structural-non-gating checks (D-14/D-15/D-16).
"""
```

**Freeze-commit ancestry test idiom** (lines 33-73):
```python
FREEZE_COMMIT_SHA = "676866657676a36abb639782fa10ecb3061fd688"

def _freeze_commit_exists() -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{FREEZE_COMMIT_SHA}^{{commit}}"],
        cwd=_repo_root(), capture_output=True,
    )
    return result.returncode == 0

def _freeze_commit_is_strict_ancestor_of_head() -> bool:
    ...
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT_SHA, "HEAD"], cwd=_repo_root(),
    )
    if is_ancestor.returncode != 0:
        return False
    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{FREEZE_COMMIT_SHA}..HEAD"],
        cwd=_repo_root(), capture_output=True, text=True, check=True,
    )
    return int(count_result.stdout.strip()) >= 1
```

**Malformed-constant parametrized sweep idiom** (lines 78-88):
```python
def test_assert_preregistered_passes_when_all_constants_set():
    dsn.assert_preregistered()

@pytest.mark.parametrize("name", dsn._REQUIRED_CONSTANTS)
def test_assert_preregistered_rejects_unset_constant(name):
    """Setting a required constant to None, deleting it, blanking a string constant, or
    emptying a tuple constant all make assert_preregistered() raise RuntimeError naming that
    constant ..."""
```
`test_cka.py` must define its own `_REQUIRED_CONSTANTS` tuple in `cka.py` and parametrize the
same rejection sweep — this is the mechanism that prevents "a constant added later without a
guard entry" from silently passing.

**monkeypatch idiom for constant mutation in tests** (`test_linear_probe.py` lines 251-283):
```python
def test_assert_preregistered_raises_when_absent(monkeypatch):
    monkeypatch.setattr(lp, "VERDICT_RULE", "a rule that omits the bucket-count constant name")
    ...
    monkeypatch.setattr(lp, "N_BUCKETS", 3)
    monkeypatch.setattr(lp, "TRAIN_FRACTION", 0.8)
    monkeypatch.setattr(lp, "SPLIT_SEED", 20260824)
```
Use `monkeypatch.setattr(cka, "<CONST>", ...)` the same way to exercise each guard branch
without ever mutating the frozen module's real constants for other tests.

**No import-purity / no-sealed-mutation test exists yet.** A repo-wide grep for
`mutat|monkeypatch.*sealed|module-level state` across `notebooks/pu_manifold/tests/` found only
ordinary `monkeypatch.setattr` usages that patch a module's OWN constants for test isolation
(`test_cae.py`, `test_linear_probe.py`) — none of these assert that *importing* a new module
leaves `mknn`, `cae`, `decoder_curvature`, `curvature_probe`, `cross_split_curvature`,
`linear_probe`, `pointcloud_probe`, `crossmodal_curvature`, or `density_stratified_null`
unmutated. **D8-23's cross-cutting constraint has no existing test precedent in this codebase —
Phase 8 will write the first one, and the planner must budget a task for inventing this pattern
from scratch** (e.g., snapshot each sealed module's `vars(module)` before and after `import cka`,
assert equality).

---

### `notebooks/diagnostics/08_cka_alignment_run.py` (runner, batch/file-I/O)

**Analog:** `notebooks/diagnostics/07.1_density_stratified_null_run.py` (1258+ lines) — this is
the strongest and most load-bearing analog per the phase guidance.

**Docstring / `--mode` dispatch shape** (lines 1-20):
```python
"""Phase 07.1 density-stratified null runner. `--mode smoke` (07.1-03) is the tracer: reload the
... `--mode positive-control` (07.1-04) and `--mode null` (07.1-04) build the stratified null's
own positive control and the D7.1-01 verdict. `--mode seeds` (07.1-05) answers D7.1-02 ...
(or `--mode selfcheck`) runs pure in-memory known-answer checks plus a frozen-artifact existence
check.

Usage:
    python notebooks/diagnostics/07.1_density_stratified_null_run.py --mode smoke --record-path notebooks/.cache/07.1_scratch_tracer.jsonl
    python notebooks/diagnostics/07.1_density_stratified_null_run.py --mode seeds --freeze-commit <sha>
"""
```
Phase 8's runner should have `--mode {sweep,positive-control,negative-control}` (per phase
guidance) or `--mode {sweep,seeds,positive-control,negative-control,selfcheck}` following this
exact docstring-with-usage-examples convention.

**Freeze-commit strict-ancestor gate — copy this function nearly verbatim** (lines 106-160):
```python
FREEZE_COMMIT_SHA = "676866657676a36abb639782fa10ecb3061fd688"  # (Phase 8 will have its own)

def _git_rev_parse(rev: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", rev], cwd=str(NOTEBOOK_ROOT.parent),
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()

def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    """... exits 1 naming D-08 unless `freeze_commit` resolves to EXACTLY this module's
    hardcoded `FREEZE_COMMIT_SHA` ... AND is BOTH an ancestor of HEAD (`git merge-base
    --is-ancestor`) AND a STRICT one (`git rev-list --count <freeze>..HEAD >= 1`)."""
    if not freeze_commit:
        print("ERROR (D-08): this mode requires --freeze-commit naming the frozen commit's "
              "SHA. Refusing to compute a 07.1 number without a strict-ancestor proof.",
              file=sys.stderr)
        sys.exit(1)
    try:
        resolved_commit = _git_rev_parse(freeze_commit)
    except subprocess.CalledProcessError:
        resolved_commit = None
    if resolved_commit != FREEZE_COMMIT_SHA:
        print(f"ERROR (D-08): --freeze-commit {freeze_commit} (resolves to {resolved_commit}) "
              f"does not equal the known freeze commit FREEZE_COMMIT_SHA={FREEZE_COMMIT_SHA}. "
              "Refusing to stamp a 07.1 number with the wrong preregistration_commit ...",
              file=sys.stderr)
        sys.exit(1)
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze_commit, "HEAD"], cwd=str(NOTEBOOK_ROOT.parent),
    )
    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{freeze_commit}..HEAD"],
        cwd=str(NOTEBOOK_ROOT.parent), capture_output=True, text=True,
    )
    count = int(count_result.stdout.strip()) if count_result.returncode == 0 and count_result.stdout.strip().isdigit() else -1
    if is_ancestor.returncode != 0 or count < 1:
        print(f"ERROR (D-08): --freeze-commit {freeze_commit} is not a STRICT git ancestor of "
              f"HEAD. ... PREREGISTRATION_FREEZE_RULE: no number may be produced at or before "
              "the freeze commit itself.", file=sys.stderr)
        sys.exit(1)
```
D8-22 requires the exact same discipline: Phase 8's runner must refuse (exit 1) to write any
record row without `--freeze-commit` resolving to its own hardcoded `FREEZE_COMMIT_SHA` and
being a strict git ancestor of HEAD.

**Record-path resolution and traversal guard** (lines 176-185):
```python
def resolve_record_path(record_path_arg: Optional[str]) -> Path:
    if record_path_arg is None:
        return cache.cache_path(dsn.RECORD_STEM, "jsonl")
    candidate = Path(record_path_arg)
    cache._assert_inside_cache(candidate)
    return candidate
```

**JSONL append with a raw-numpy guard — copy verbatim** (lines 188-203):
```python
def append_record_row(row: Dict[str, Any], record_path: Path) -> None:
    """Write one flat JSON-serializable dict per line. Every value must already be a plain
    Python scalar, list or string -- never a raw numpy array or numpy scalar."""
    for key, value in row.items():
        if isinstance(value, (np.ndarray, np.generic)):
            raise TypeError(
                f"append_record_row: row[{key!r}] is a raw numpy value ({type(value)!r}); "
                "serialize it to a plain Python scalar/list before appending."
            )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
```
This is the exact defect precedent (`fix(06): serialize numpy arrays in the Phase 6 record`)
Phase 8's runner must not repeat — every `float(...)`/`.tolist()` cast must happen before this
call.

**Reusing another runner's function by file path, not package import** (lines 89-93):
```python
# Phase 7's own sealed runner, loaded by file path (never imported as a package member --
# notebooks/diagnostics/ is a plain directory, not a pu_manifold package member, and Phase 7's
# runner has its own module-level thread-env-var writes that would collide with this runner's
# own _THREADS setup above if imported at module scope).
RUNNER_07_PATH = DIAGNOSTICS_ROOT / "07_crossmodal_curvature_run.py"
```
If Phase 8's runner needs anything from `07_crossmodal_curvature_run.py` (e.g. a shared
npz-loading helper), load it via `importlib.util` by path, exactly as 07.1 did — never
`import` it as a package member.

**Package imports (module-level, `noqa: E402` after sys.path manipulation)** (lines 55-79):
```python
import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional, Tuple  # noqa: E402
...
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pu_manifold import cache  # noqa: E402
from pu_manifold import cross_split_curvature  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402
from pu_manifold import curvature_probe  # noqa: E402
from pu_manifold import density_stratified_null as dsn  # noqa: E402
```
Phase 8's runner imports `from pu_manifold import cka`, `from pu_manifold import
density_stratified_null as dsn`, `from pu_manifold import curvature_probe` — absolute
`pu_manifold.*` imports (package added to `sys.path` earlier in the file), matching this exact
convention.

**`argparse` shape** (lines 1203-1229):
```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", ...)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke-permutations", type=int, default=100)
    p.add_argument("--freeze-commit", ...,
        help="STRICT git ancestor of HEAD (D-08). --mode smoke does not use this flag.")
```

---

### `notebooks/08_cka_alignment_check.ipynb` + `08-FINDINGS.md` (reporting notebook / doc)

**Analog:** not read in this pass (out of budget for a 5-analog cap that already covers the
higher-value machinery); the closest precedent by name is 07.1's own reporting notebook and
`07.1-FINDINGS.md` (per-`d` table reported unconditionally, D-15 cited directly in D8-21).
**Planner should have the implementing agent open `notebooks/07.1_*.ipynb` (or whichever
07.1/07 notebook exists) directly for cell-count and verdict-sentence-composition structure**
before writing Phase 8's — this file was not itself opened here, only referenced by
`08-CONTEXT.md`'s own citation of 07.1's D-15 precedent. Treat as a **role-match, not an
exact-match analog**, until that read happens.

## Shared Patterns

### Freeze-before-any-number (`assert_preregistered` + git-ancestry proof)
**Source:** `notebooks/pu_manifold/linear_probe.py` lines 249-383 (guard) and
`notebooks/diagnostics/07.1_density_stratified_null_run.py` lines 106-160 (`_strict_ancestor_or_exit`).
**Apply to:** `cka.py` (the guard function) and `08_cka_alignment_run.py` (the CLI-level
ancestry check before any write). Every production `--mode` must call both.

### Non-gating diagnostics (D7-03 shape)
**Source:** `notebooks/pu_manifold/crossmodal_curvature.py` lines 149-165, 250-265.
**Apply to:** RBF CKA (D8-01), sigma ladder (D8-04), middle tertile (D8-10), `d=32` (D8-12),
validation ladder (D8-20) — each needs its own `..._IS_NON_GATING = True`-style constant plus a
prose rule string, never a bare comment.

### Do-not-pool-seeds, unanimous 3-of-3 (`05-03-DECISION.md`)
**Source:** `notebooks/pu_manifold/linear_probe.py` — `SEED_HANDLING_RULE` (equality-checked
string), `combine_seed_verdicts` (lines 831-887).
**Apply to:** Phase 8's `d=25` seed-axis verdict combination (D8-15). Copy the exact-equality
guard on the rule string and the "exactly 3, HOLDS-counted, 1-or-2 is terminal
SPLIT-ACROSS-SEEDS" branching.

### JSONL record writing with a raw-numpy guard
**Source:** `notebooks/diagnostics/07.1_density_stratified_null_run.py` lines 188-203
(`append_record_row`).
**Apply to:** `08_cka_alignment_run.py`'s every write path (sweep, positive-control,
negative-control modes all append rows).

### Freeze-commit ancestry test
**Source:** `notebooks/pu_manifold/tests/test_density_stratified_null.py` lines 33-73.
**Apply to:** `test_cka.py` — same `_freeze_commit_exists` / `_freeze_commit_is_strict_ancestor_of_head`
helper pair, new `FREEZE_COMMIT_SHA` once Phase 8's freeze commit lands.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Import-purity / no-sealed-module-mutation test (D8-23) | test | event-driven (import side-effect check) | Repo-wide grep of `notebooks/pu_manifold/tests/` found no test asserting that importing one module leaves another module's `vars()` unmutated — every existing `monkeypatch.setattr` usage patches a module's own constants for test isolation, not a cross-module import-purity check. Phase 8 must invent this pattern (e.g., snapshot `vars(module)` for each of the 9 named sealed modules before/after `import pu_manifold.cka`, assert equality) — no in-repo shape to copy. |

## Metadata

**Analog search scope:** `notebooks/pu_manifold/*.py`, `notebooks/pu_manifold/tests/*.py`,
`notebooks/diagnostics/*.py`, `notebooks/pu_manifold/__init__.py`.
**Files scanned (read or greped):** `density_stratified_null.py` (partial, ~150 lines read
directly + structure grepped), `linear_probe.py` (freeze/verdict sections read directly, ~250
lines), `crossmodal_curvature.py` (D7-03/D7-07 sections read directly, ~60 lines),
`__init__.py` (read in full, 62 lines), `07.1_density_stratified_null_run.py` (~150 lines read
directly across two passes), `test_density_stratified_null.py` (~90 lines read directly),
`test_linear_probe.py`, `test_cae.py` (grepped for monkeypatch idiom).
**Pattern extraction date:** 2026-08-27

### Confirmed structural facts for the planner

- `notebooks/pu_manifold/__init__.py` exists and re-exports named symbols explicitly from
  `.cache` and `.subsample` only — it deliberately does **not** import `mknn`, `curvature`,
  `linear_probe`, `crossmodal_curvature`, or `density_stratified_null` at package level (the
  docstring states this is to avoid forcing a torch import cost). **Recommendation for
  `cka.py`:** since it is pure numpy with no torch dependency, it MAY be added to
  `__init__.py`'s explicit export list, but is not required to be — either choice is consistent
  with existing convention; the planner should decide based on whether the runner/tests prefer
  `from pu_manifold import cka` (works either way, since `cka.py` is a submodule regardless of
  `__init__.py` re-export) or `from pu_manifold import CKA_SOME_CONSTANT` (requires the
  re-export).
- Sibling modules import each other with **relative-package absolute** style:
  `from pu_manifold import density_stratified_null as dsn`, `from pu_manifold import
  crossmodal_curvature as cc`, `from pu_manifold import curvature_probe` — always `from
  pu_manifold import <module>`, never `from .density_stratified_null import ...` inside
  `notebooks/pu_manifold/*.py` itself (that internal-relative style is only used by
  `__init__.py`). Test files and runner scripts both do `sys.path.insert(0, str(Path(__file__)
  ...))` then `from pu_manifold import <module>`.
- Naming convention confirmed exactly as RESEARCH.md recommends: one-word lowercase module files
  (`mknn.py`, `cae.py`, `curvature_probe.py`, `density_stratified_null.py`) — `cka.py` sits
  beside these without translation. Test files are `test_<module>.py` one-to-one
  (`test_density_stratified_null.py`, `test_linear_probe.py`) — `test_cka.py` follows directly.
  Runner scripts are `<phase>_<topic>_run.py` (`07.1_density_stratified_null_run.py`,
  `07_crossmodal_curvature_run.py`) — `08_cka_alignment_run.py` matches. Notebooks are
  `<phase>_<topic>.ipynb` / `<phase>_<topic>_check.ipynb` in `notebooks/` root (not under
  `pu_manifold/`).
- No import-time mutation test precedent exists (see "No Analog Found" above) — flagged
  explicitly per phase guidance.
