---
phase: 07-curvature-conditioned-crossmodal-alignment
reviewed: 2026-08-26T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - notebooks/pu_manifold/crossmodal_curvature.py
  - notebooks/diagnostics/07_crossmodal_curvature_run.py
  - notebooks/pu_manifold/tests/test_crossmodal_curvature.py
  - notebooks/07_crossmodal_curvature_check.ipynb
findings:
  critical: 3
  warning: 4
  info: 1
  total: 8
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-08-26
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

The frozen pre-registration block in `crossmodal_curvature.py` (everything above the plan
07-02 import group, roughly line 379) was read for context per the review brief but not
flagged for its intentional freeze-pattern properties (late imports, `noqa: E402`, apparent
duplication between constants and their docstrings) — those are the pre-registration design,
not defects. The additive compute functions below the freeze (`split_indices`,
`per_point_mknn`, `two_tailed_permutation_null`, `apply_verdict`, `plant_positive_control`,
`smallest_cleared_target`, `density_diagnostics`) are statistically sound: they compose the
sealed `mknn`/`curvature_probe`/`cross_split_curvature` functions correctly, the bisection in
`plant_positive_control` degrades honestly (an unreachable target still reports its true,
unmet `achieved_rho` rather than a fabricated one), and the 157-test suite genuinely pins
behavior — no tautological assertions were found, ties and degenerate inputs are exercised,
and the regression test for the closed one-sided-permutation defect calls the sealed function
directly rather than through a wrapper.

The real defects are concentrated in `07_crossmodal_curvature_run.py`, the runner script,
which is the one file in scope carrying no automated test coverage of its own. Three findings
there are classified Critical because they defeat mechanisms this phase's own design treats as
load-bearing: (1) `--mode dsweep`'s `--freeze-commit` argument is validated only for being
*some* strict git ancestor of HEAD, never checked against the module's own hardcoded
`FREEZE_COMMIT_SHA` constant, so a wrong-but-valid SHA passes silently and the recorded
provenance would misrepresent what was actually frozen; (2) `--mode positive-control` performs
no strict-ancestor check at all before writing rows to the same frozen record `--mode dsweep`
protects, an asymmetry with no counterpart guard; (3) the reduced-scale-run safety guard
(`--max-epochs` / `--smoke-rows` must be paired with `--record-path`, refusing to let a test
run land in the sealed production record) and the `--threads` cap are all implemented by
scanning raw `sys.argv` for an exact token match, which silently fails — with no error, no
warning — for the standard `--flag=value` CLI syntax, reverting to defaults instead of the
value the caller actually passed. All three are demonstrated by direct reading of the control
flow (and, for the argv issue, by a standalone repro) rather than inferred.

The one warning-level correctness gap inside `crossmodal_curvature.py` itself is
`density_diagnostics`, which — unlike every sibling compute function in the module — has no
non-finite/constant-input guard, so a degenerate field would silently write `NaN` into the
JSONL record rather than raising. The remaining findings are quality/consistency issues: a
duplicated distinct-value-count implementation inside `run_smoke`, and a reporting
inconsistency where the notebook's positive-control section still states the pre-amendment,
imprecise "detects as small as 0.05" framing that the human reviewer specifically corrected in
`07-FINDINGS.md`'s Sec 2 — the notebook (committed with outputs, a primary reporting artifact)
was never updated to match.

## Critical Issues

### CR-01: `--mode dsweep`'s `--freeze-commit` is validated as *an* ancestor, never as *the* freeze commit

**File:** `notebooks/diagnostics/07_crossmodal_curvature_run.py:63, 380-419, 443, 455`
**Issue:** The module defines `FREEZE_COMMIT_SHA = "f032745f6450068c63763993d39fa112fd36bb8c"`
at line 63 and correctly uses it (hardcoded, not caller-supplied) in `run_positive_control`
(line 343). But `run_dsweep` takes `--freeze-commit` as an arbitrary CLI string and passes it
straight into `_strict_ancestor_or_exit`, which only proves the supplied SHA is *some* strict
git ancestor of HEAD (`git merge-base --is-ancestor` + `git rev-list --count >= 1`). It never
compares the supplied value against `FREEZE_COMMIT_SHA`. Any commit that happens to precede
HEAD in history — a typo'd SHA, an unrelated earlier commit, a commit from before the freeze
existed — passes this check and gets stamped as `preregistration_commit` on every sweep row
written to the sealed record (line 455, `_git_rev_parse(args.freeze_commit)`). This defeats
the entire stated purpose of `PREREGISTRATION_FREEZE_RULE` and D7-06: the mechanism is
supposed to prove a PU number was computed under the specific frozen constants block, not
merely under *some* earlier commit. The docstring for `--freeze-commit` even says "read from
07-01-SUMMARY.md, not re-derived from git log" — i.e., the design already expects a human to
copy-paste the correct value by hand, with nothing in code catching a wrong-but-plausible one.
**Fix:**
```python
def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    if not freeze_commit:
        ...
    if freeze_commit != FREEZE_COMMIT_SHA and _git_rev_parse(freeze_commit) != FREEZE_COMMIT_SHA:
        print(
            f"ERROR (D7-06): --freeze-commit {freeze_commit} does not resolve to the "
            f"known freeze commit {FREEZE_COMMIT_SHA}. Refusing to stamp a PU number with "
            "the wrong preregistration_commit.",
            file=sys.stderr,
        )
        sys.exit(1)
    # ... existing strict-ancestor check unchanged
```
Simplest fix: default `--freeze-commit` to `FREEZE_COMMIT_SHA` and drop the CLI override
entirely, since the constant already lives in this file and `run_positive_control` already
uses it that way.

### CR-02: `--mode positive-control` writes to the frozen record with no strict-ancestor check at all

**File:** `notebooks/diagnostics/07_crossmodal_curvature_run.py:287-377`
**Issue:** `run_dsweep` guards every write to the sealed record behind
`_strict_ancestor_or_exit` (line 443). `run_positive_control` also writes rows to the same
sealed record (`append_record_row(row, record_path)` at line 362, once per positive-control
target) but never calls `_strict_ancestor_or_exit` or any equivalent check anywhere in its
body. It calls `cc.assert_preregistered()` (line 305), which only checks that the constants
themselves are well-formed — it says nothing about which git commit is currently checked out.
Nothing stops `--mode positive-control` from being run from a detached HEAD, a stale branch,
or any commit state, and appending rows to `notebooks/.cache/07_crossmodal_curvature.jsonl`
that claim a `run_commit` inconsistent with the rest of the sealed record. This is not merely
stylistic asymmetry with `run_dsweep` — it is a missing instance of the exact T-07-06
mitigation the phase's own threat model requires for every code path that writes a PU number.
**Fix:** Call the same strict-ancestor gate `run_dsweep` uses before the first
`append_record_row` call:
```python
def run_positive_control(args: argparse.Namespace) -> str:
    cc.assert_preregistered()
    _strict_ancestor_or_exit(FREEZE_COMMIT_SHA)
    ...
```

### CR-03: `--max-epochs`, `--smoke-rows`, and `--threads` are silently ignored when passed with `--flag=value` syntax, bypassing the sealed-record safety guard

**File:** `notebooks/diagnostics/07_crossmodal_curvature_run.py:19-27, 445-453, 472-473`
**Issue:** `is_scratch = ("--smoke-rows" in sys.argv) or ("--max-epochs" in sys.argv)` (line
445) and `max_epochs = args.max_epochs if "--max-epochs" in sys.argv else cc.MAX_EPOCHS` (line
472) test for the exact string `"--max-epochs"` as a standalone token in `sys.argv`. argparse
supports (and many users default to) the `--flag=value` form, e.g. `--max-epochs=5`. When
that form is used, the token in `sys.argv` is `"--max-epochs=5"`, not `"--max-epochs"`, so the
membership check is `False` even though `args.max_epochs` was correctly parsed to `5` by
argparse. Confirmed directly:
```
$ .venv/bin/python -c "
import sys; sys.argv=['p','--max-epochs=5']
print('--max-epochs' in sys.argv)             # False
import argparse; p=argparse.ArgumentParser(); p.add_argument('--max-epochs', type=int, default=None)
print(p.parse_args().max_epochs)               # 5
"
```
Two consequences, both silent (no error, no warning):
1. The value the caller explicitly passed is discarded — `max_epochs` falls back to
   `cc.MAX_EPOCHS` (600) and `n_rows_override` falls back to `None` (full dataset) — so a run
   intended as a quick, reduced-scale exercise instead runs at full production scale.
2. Because `is_scratch` is also computed from the same broken check, the guard at
   lines 446-453 ("`--smoke-rows`/`--max-epochs` MUST be paired with `--record-path`, refusing
   to let a reduced-scale run default onto the frozen record") never fires. If both
   `--max-epochs=N` and `--smoke-rows=N` are passed with `=` syntax and `--record-path` is
   omitted, the run silently proceeds to append full-scale sweep rows to the real,
   already-sealed `notebooks/.cache/07_crossmodal_curvature.jsonl` — the exact outcome this
   guard exists to prevent, just reached by a route the guard doesn't recognize. The same
   token-matching bug is present for `--threads` at lines 19-27 (parsed before argparse even
   runs, for the stated reason that the thread cap must precede the torch import); a
   `--threads=N` invocation silently keeps the default of 8 rather than raising or honoring
   the requested value. `argparse`'s own `--threads` definition (line 738) is consequently
   dead — `args.threads` is never read anywhere in the file; only the module-level `_THREADS`
   (populated by the broken raw scan) is used.
**Fix:** Use the argparse-parsed values directly instead of re-scanning `sys.argv`:
```python
is_scratch = args.smoke_rows is not None or args.max_epochs is not None
...
max_epochs = args.max_epochs if args.max_epochs is not None else cc.MAX_EPOCHS
n_rows_override = args.smoke_rows  # already None by default
```
(requires changing `--smoke-rows`'s default from `800` to `None` and moving its default value
resolution into `run_smoke` only, since `run_dsweep` needs to distinguish "not passed" from
"explicitly 800"). For `--threads`, parse it with `argparse.parse_known_args()` on a minimal
pre-parser before the thread-cap block, or accept both `--threads N` and `--threads=N` forms
explicitly rather than a raw substring/token scan.

## Warnings

### WR-01: `density_diagnostics` has no input guards, unlike every sibling compute function

**File:** `notebooks/pu_manifold/crossmodal_curvature.py:649-733`
**Issue:** `plant_positive_control` (lines 584-593) explicitly guards against a non-finite
`h_real`, a constant `h_real` (`np.ptp(h) == 0`), and too few rows, before doing anything else
— and its docstring calls a silently-degenerate control "the single worst failure available
here." `curvature_probe.permutation_null`, which `two_tailed_permutation_null` composes,
guards the same way. `density_diagnostics` has no equivalent guard on `h`, `m`, or the
computed `density` array. If `h_arr` or `m_arr` is constant or contains a non-finite value
(e.g., an upstream field-computation edge case), `spearmanr(density, h_arr).statistic` and
`cross_split_curvature.partial_spearman(...)` will silently return `NaN` rather than raising.
`append_record_row`'s numpy-type guard (line 205 of the runner) does not catch this, because
`float(nan)` is a plain Python `float`, not a numpy type — the check only rejects raw numpy
arrays/scalars. A `NaN` would therefore be written straight into the JSONL record via
`json.dumps`, which (with Python's default `allow_nan=True`) emits the non-standard token
`NaN`, silently breaking any strict JSON consumer of this record.
**Fix:** Add the same guard pattern used elsewhere in this module:
```python
if not np.all(np.isfinite(h_arr)) or not np.all(np.isfinite(m_arr)):
    raise ValueError("density_diagnostics: h or m contains a non-finite value.")
if np.ptp(h_arr) == 0 or np.ptp(m_arr) == 0:
    raise ValueError("density_diagnostics: h or m is constant.")
```

### WR-02: `run_smoke` reimplements the relative-precision distinct-value count instead of reusing `_distinct_value_count`

**File:** `notebooks/diagnostics/07_crossmodal_curvature_run.py:246-247` (vs. `179-185`)
**Issue:** The runner already defines `_distinct_value_count(arr)` (line 179), a thin wrapper
around `crossmodal_curvature._relative_precision_distinct_count`, explicitly documented as
"reused, not reimplemented" to avoid the exact class of bug `05-02-SUMMARY.md` retracted
(5,301/9,852 reported vs. 4/3 true distinct values). `run_dsweep` correctly calls this helper
(line 511). `run_smoke`, four lines away in the same file, instead inlines its own version:
```python
scale = max(float(np.max(np.abs(mknn_arr))), 1e-300)
n_distinct = int(np.unique(np.round(mknn_arr / scale, 12)).shape[0])
```
This is a second, independent implementation of the exact pattern the module's own docstrings
warn against duplicating — the two versions currently agree numerically, but nothing enforces
that they continue to, and the whole point of centralizing this logic in
`_relative_precision_distinct_count` was to prevent a second, silently-diverging convention.
**Fix:** Replace lines 246-247 with a call to the existing helper:
```python
n_distinct = _distinct_value_count(mknn_arr)
```

### WR-03: The notebook still states the pre-amendment, imprecise positive-control detection-floor claim that `07-FINDINGS.md` was corrected to walk back

**File:** `notebooks/07_crossmodal_curvature_check.ipynb` (Cell 9, code)
**Issue:** Per 07-05-SUMMARY.md, the human checkpoint on this plan specifically flagged and
corrected the claim "recovers a planted effect as small as `rho=0.05`" in `07-FINDINGS.md`
Sec 2, because `smallest_cleared_target=0.05` is only the smallest grid point that happened to
clear, not the actual detection floor — the real null-band edge sits at ≈0.0205, and the true
floor is unresolved somewhere in the interval 0.021-0.05 (the grid has no point between 0.02
and 0.05). That amendment (`git diff df8502f^..df8502f`) touched only `07-FINDINGS.md` — `git
diff --stat` for that commit shows exactly one file changed. The notebook's Cell 9 still prints:
> "The permutation test has power to detect a planted relationship as small as
> `{smallest_cleared}` on PU's own realized `d=20` `||H||` dynamic range."
with `smallest_cleared = 0.05` — the identical overclaim the human reviewer asked to have
softened in the sibling document. The cell does append a note that d=32's magnitude sits
"closer to the power boundary," but never states the corrected framing (0.05 is not the floor;
the floor is unresolved in 0.021-0.05). Since both the notebook and `07-FINDINGS.md` are
committed, human-facing reporting artifacts for the same phase, and CLAUDE.md requires
notebooks to be "executed end to end" and committed with outputs, a reader who consults only
the notebook receives a materially different, more overstated claim than a reader of the
findings document — for the exact number the human checkpoint singled out as needing
correction.
**Fix:** Re-run Cell 9 with prose matching the amended `07-FINDINGS.md` Sec 2 framing (state
the ≈0.0205 null-band edge, the unresolved 0.021-0.05 interval, and that 0.05 is the smallest
*grid point that cleared*, not the floor), then re-commit the notebook with the updated output.

### WR-04: The runner script's CLI-argument and provenance-gate logic has no automated test coverage

**File:** `notebooks/diagnostics/07_crossmodal_curvature_run.py` (whole file);
`notebooks/pu_manifold/tests/test_crossmodal_curvature.py` (absence)
**Issue:** All 157 tests in `test_crossmodal_curvature.py` exercise `crossmodal_curvature.py`'s
pure compute functions; none imports or exercises `07_crossmodal_curvature_run.py` at all (not
even `_strict_ancestor_or_exit`, the `is_scratch` detection, or `resolve_record_path`'s
containment behavior via a fake `--record-path`). Per the plan summaries, the runner's
behavior was verified only by manual/procedural runs at specific checkpoints. CR-01 through
CR-03 above are all runner-only defects, in exactly the code this suite does not touch — the
`=`-syntax argv bug in particular (CR-03) is precisely the kind of defect a
`subprocess`-invocation or `argparse`-level unit test (e.g., asserting `is_scratch` is `True`
for `--max-epochs=5`) would have caught before it reached a real `--mode dsweep` invocation.
**Fix:** Add a small test module (e.g. `test_crossmodal_curvature_run.py`) that imports the
runner's pure helper functions (`_strict_ancestor_or_exit` is process-exiting and harder to
unit test directly, but `resolve_record_path`, and a refactored argv-parsing helper per CR-03's
fix, are straightforward to test with `monkeypatch.setattr(sys, "argv", [...])`).

## Info

### IN-01: `assert_preregistered`'s malformed-constant sweep has no boundary case for an empty dict

**File:** `notebooks/pu_manifold/crossmodal_curvature.py:300-334`
**Issue:** The malformed-value checks in `assert_preregistered` cover `None`, absent, blank
string, and empty tuple/list (lines 318-323), and the 118-test freeze guard parameterizes over
every one of those cases for every `_REQUIRED_CONSTANTS` entry. `TRAIN_CFG` (line 91) is a
`dict`, the only dict-typed entry in `_REQUIRED_CONSTANTS`, and an empty dict (`{}`) would pass
`assert_preregistered` silently (it is not `None`, not a string, not a tuple/list) even though
an empty `TRAIN_CFG` would be just as malformed as an empty `MKNN_K_GRID`. This is a minor gap
relative to the function's own stated discipline of catching every "unset, malformed, or
absent" pre-registered constant; low impact since `TRAIN_CFG` is a frozen literal that will
never actually be edited to `{}` in the committed source, but worth closing for completeness.
**Fix:**
```python
elif isinstance(value, dict) and len(value) == 0:
    missing.append(f"{name} (empty dict)")
```

---

_Reviewed: 2026-08-26_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
