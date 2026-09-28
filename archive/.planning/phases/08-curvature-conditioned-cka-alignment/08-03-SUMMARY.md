---
phase: 08-curvature-conditioned-cka-alignment
plan: 03
subsystem: testing
tags: [numpy, scipy, cka, subprocess, git-ancestry, resource, pytest]

# Dependency graph
requires:
  - phase: 08-01
    provides: "cka.py's estimator surface (linear_gram, rbf_gram, median_pairwise_distance) and
      the 37-constant freeze-guard shell (assert_preregistered, still all UNSET after 08-02)"
provides:
  - "notebooks/pu_manifold/tests/test_cka_import_purity.py -- the first test in this repository
    asserting cross-module import purity: SEALED_MODULES (9-tuple), _snapshot_module_state,
    _import_in_order, and three tests proving D8-23 across four subprocess-isolated import
    orders plus a planted-mutation proof that the comparison can fail"
  - "notebooks/diagnostics/08_cka_alignment_run.py's production data layer: _repo_root,
    _git_rev_parse, _strict_ancestor_or_exit (D8-22's freeze-ancestry gate, FREEZE_COMMIT_SHA
    still empty), load_pu_pair, load_frozen_fields (halt-not-regenerate on any missing Phase
    7/07.1 npz), append_record_row/resolve_record_path (raw-numpy guard, traversal guard)"
  - "--mode sigma: build_gram_matrices (Gram-matrix-once, no subset index accepted) and
    run_sigma -- measures the two D8-03 global RBF bandwidths over all 10,000 points and proves
    the eight-Gram-matrix build with measured wallclock and peak RSS, computing no CKA value,
    subset or verdict"
affects: [08-04, 08-05, 08-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Subprocess-isolated import-order testing: each order runs in its own
      `subprocess.run([sys.executable, '-c', script])` with a snapshot function's own source
      embedded via `inspect.getsource`, so `sys.modules` caching in a single process cannot
      mask a mutation -- no in-repo precedent existed for this pattern before this plan"
    - "Textual fingerprinting of `vars(module)` for cross-process comparison: `repr()` for
      immutables, shape/dtype/sha256 for numpy arrays, SORTED element reprs for
      sets/frozensets (CPython's per-process string hash randomization makes a bare
      frozenset repr non-reproducible across independent interpreters), qualified names for
      functions/classes/modules"
    - "Freeze-ancestry gate copied nearly verbatim from 07.1's runner (D-08 renamed to D8-22
      throughout): FREEZE_COMMIT_SHA=\"\" makes every --freeze-commit value fail exact-equality
      until 08-04, closing the gate loudly rather than silently"
    - "on_build instrumentation callback on build_gram_matrices: keeps the pure eight-matrix
      contract while giving run_sigma per-kernel timing/printing without duplicating the
      build loop"

key-files:
  created:
    - notebooks/pu_manifold/tests/test_cka_import_purity.py
  modified:
    - notebooks/diagnostics/08_cka_alignment_run.py

key-decisions:
  - "SEALED_MODULES comparison target is a CKA-FREE baseline (the nine modules imported in a
    subprocess that never imports cka at all), not a literal per-module before/after snapshot
    within one interleaved sequence. The plan's literal wording ('vars(m) ... before and after
    import pu_manifold.cka') does not extend cleanly to an order where some modules are not
    yet imported at the moment cka is: 'before' is undefined for a module that does not exist
    yet. The invariant that actually matters -- a sealed module's final state is independent of
    whether/when cka was ever imported in the same process -- is preserved and is what every
    test asserts. Documented in the test file's own docstrings."
  - "The fourth named order ('cka alone, followed by the nine') is implemented as cka first
    then the nine SEALED_MODULES in REVERSED order, rather than literally duplicating the
    'cka first' order. A literal reading of the plan's four bullet points produces only three
    distinct import sequences (order 1 and order 4 describe the same sequence in different
    words); reversing the nine for order 4 satisfies 'at least four DISTINCT import orders'
    while still satisfying 'a fresh interpreter that has imported only cka' followed by the
    nine sealed modules."
  - "GRAM_DTYPE (float32) and SIGMA_MULTIPLIERS ((0.5, 1.0, 2.0)) are runner-local literals
    (SIGMA_GRAM_DTYPE, SIGMA_MULTIPLIERS), not read from cka.GRAM_DTYPE / cka.SIGMA_MULTIPLIERS
    -- both remain UNSET in cka.py until 08-04's freeze. This mirrors 08-01/08-02's own
    discipline of never reading a gating constant across the freeze boundary before it exists."

patterns-established:
  - "Availability-based (not just post-hoc RSS-based) memory fallback trigger:
    _available_memory_mb() via os.sysconf(SC_AVPHYS_PAGES/SC_PAGE_SIZE) is checked BEFORE
    building anything, so the one-at-a-time disk-cache fallback (T-08-07) can actually prevent
    an OOM rather than only detect one after the fact."

requirements-completed: [D8-03, D8-04, D8-07, D8-14, D8-23, D8-24]

coverage:
  - id: D1
    description: "D8-23's cross-cutting import-purity constraint has a real regression test,
      asserting across four distinct import orders, each isolated in its own subprocess"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka_import_purity.py#test_import_cka_does_not_mutate_sealed_modules"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka_import_purity.py#test_import_purity_holds_under_every_order"
        status: pass
    human_judgment: false
  - id: D2
    description: "The import-purity comparison is proven able to fail (a planted mutation is
      detected), so a passing suite is informative rather than vacuous"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_cka_import_purity.py#test_snapshot_detects_a_planted_mutation"
        status: pass
    human_judgment: false
  - id: D3
    description: "The freeze-ancestry gate (D8-22) refuses every --freeze-commit value while
      FREEZE_COMMIT_SHA is unset, for all three production modes, and refuses a missing
      --freeze-commit outright"
    verification:
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode sweep --freeze-commit HEAD (exit 1, names D8-22 and empty FREEZE_COMMIT_SHA)"
        status: pass
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode sweep (exit 1, names --freeze-commit requirement)"
        status: pass
    human_judgment: false
  - id: D4
    description: "load_frozen_fields loads all six frozen curvature fields read-only, each
      shape (10000,), and halts naming the missing path on a missing frozen npz -- no
      compute-if-missing branch"
    verification:
      - kind: integration
        ref: "load_frozen_fields() happy path: returns 6 keys, all shape (10000,)"
        status: pass
      - kind: integration
        ref: "07.1_seed_fields_d25.npz renamed away -> load_frozen_fields exits 1 naming the absent path; file renamed back"
        status: pass
    human_judgment: false
  - id: D5
    description: "append_record_row's raw-numpy TypeError guard and resolve_record_path's
      traversal guard both fire correctly"
    verification:
      - kind: unit
        ref: "append_record_row({'a': np.float64(1.0)}, ...) raises TypeError naming 'a'"
        status: pass
      - kind: unit
        ref: "resolve_record_path('/etc/passwd') raises ValueError rather than returning a path"
        status: pass
    human_judgment: false
  - id: D6
    description: "--mode sigma measures the two D8-03 global RBF bandwidths over all 10,000
      points, at full precision, and proves the Gram-matrix-once architecture with measured
      wallclock and peak RSS; computes no CKA value and constructs no subset"
    verification:
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode sigma --record-path notebooks/.cache/08_scratch_sigma.jsonl (exit 0; two sigma lines at full precision; 8 gram_build lines; 1 peak_rss_mb line; 8 JSONL rows)"
        status: pass
      - kind: other
        ref: "inspect.getsource(run_sigma) contains no cka_on_subset/tertile_split_within_strata/density_strata; build_gram_matrices signature has no idx parameter"
        status: pass
    human_judgment: false
  - id: D7
    description: "No sealed module edited, src/effdim/ untouched, no torch import, only the
      two files_modified touched since the plan's base commit"
    verification:
      - kind: other
        ref: "git diff --name-only bb34575..HEAD -- notebooks/ src/ -> exactly test_cka_import_purity.py and 08_cka_alignment_run.py"
        status: pass
      - kind: other
        ref: "grep -n torch notebooks/diagnostics/08_cka_alignment_run.py (empty)"
        status: pass
      - kind: other
        ref: "cka.assert_preregistered() still raises RuntimeError on KERNELS; len(cka._REQUIRED_CONSTANTS) == 37"
        status: pass
    human_judgment: false
  - id: D8
    description: "Full notebooks/pu_manifold/tests/ suite stays green: 742 baseline + 6 new
      import-purity tests, no regressions"
    verification:
      - kind: integration
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q -> 748 passed, 1 skipped"
        status: pass
    human_judgment: false

duration: ~1h5m (three tasks, each committed atomically; includes one ~3.5min --mode sigma run and two ~5-6min full-suite runs)
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 03: Import-Purity Test, Runner Data Layer, and D8-03 Sigma Measurement Summary

**Invented this repository's first cross-module import-purity regression test (D8-23, four
subprocess-isolated orders), built the runner's frozen-field/record-guard/freeze-ancestor data
layer, and measured the two D8-03 pre-registration inputs — sigma_hsc=0.6420152563705613,
sigma_legacysurvey=0.5696337821442163 — with the eight-Gram-matrix-once build proven at
measured wallclock (~1.4s linear, ~25.6-25.9s per RBF rung) and peak RSS 5182.74 MB.**

## Performance

- **Duration:** ~1h5m across three atomically-committed tasks
- **Completed:** 2026-08-28T02:14:26Z (Task 3 commit; UTC)
- **Tasks:** 3/3
- **Files modified:** 2 (1 new, 1 modified)

## Accomplishments

- **Task 1 (D8-23):** `test_cka_import_purity.py` snapshots `vars()` of all nine sealed
  `pu_manifold` modules (`mknn`, `cae`, `decoder_curvature`, `curvature_probe`,
  `cross_split_curvature`, `linear_probe`, `pointcloud_probe`, `crossmodal_curvature`,
  `density_stratified_null`) against a cka-free baseline across four distinct import orders
  (cka first; cka last; cka after `crossmodal_curvature` but before `density_stratified_null`;
  cka alone followed by the nine in reversed order), each isolated in its own subprocess via
  `subprocess.run([sys.executable, "-c", script])` so `sys.modules` caching cannot mask a
  mutation. `_snapshot_module_state`'s own source is embedded into each subprocess script via
  `inspect.getsource`, so the fingerprint logic lives in exactly one place. A dedicated test
  plants a new attribute on `crossmodal_curvature` mid-subprocess and confirms the snapshot
  comparison detects it — proving the check can fail before trusting that it never does. 6
  tests, all pass, confirmed stable across three independent full runs (ruling out
  hash-randomization flakiness, see Deviations).
- **Task 2 (D8-14/D8-22 shell):** the runner's production data layer —
  `_repo_root`/`_git_rev_parse`/`_strict_ancestor_or_exit` (D8-22's freeze-ancestry gate, D-08
  renamed to D8-22 throughout, copied from `07.1_density_stratified_null_run.py`;
  `FREEZE_COMMIT_SHA = ""` makes every `--freeze-commit` value fail until 08-04),
  `load_pu_pair` (mirrors 07.1's own copy verbatim), `load_frozen_fields` (halt-not-regenerate
  on any missing Phase 7/07.1 npz or key, no compute branch), and `resolve_record_path` now
  defaulting to `cache.cache_path(RECORD_STEM, "jsonl")` (`RECORD_STEM` also still empty).
  `sweep`/`positive-control`/`negative-control` now call the ancestor gate first and exit 1
  today (previously a flat exit 2).
- **Task 3 (D8-03/D8-04):** `build_gram_matrices` builds all eight Gram matrices exactly once
  per modality (`linear`, `rbf_0.5sigma`, `rbf_sigma`, `rbf_2sigma`), accepting no subset index
  so no call site can manufacture a per-subset bandwidth. `run_sigma` measured, on this
  machine, over the resolved `subsample_20260729_a79b3460b838fd0a.npz` (10000, 768) pair:
  **`sigma_hsc = 0.6420152563705613`**, **`sigma_legacysurvey = 0.5696337821442163`** — the two
  numbers 08-04 freezes as `SIGMA_HSC` / `SIGMA_LEGACYSURVEY`. Gram builds: linear ~1.4s per
  modality, each RBF rung ~25.6-25.9s per modality (8 builds total, ~3m27s wallclock for the
  whole `--mode sigma` run). **Peak RSS: 5182.74 MB** — under the 6 GB `FALLBACK_THRESHOLD_MB`,
  so the one-at-a-time disk-cache fallback was **not** taken (available memory was measured at
  25999 MB via `os.sysconf`, far above the threshold). 8 JSONL rows written to
  `notebooks/.cache/08_scratch_sigma.jsonl`, each carrying `sigma_median_pairwise`,
  `gram_kernel`, `gram_build_s`, `gram_dtype=float32`, `peak_rss_mb`, and `disk_fallback_taken`.
  No CKA value, subset, or verdict computed anywhere — confirmed by source inspection
  (`cka_on_subset`, `tertile_split_within_strata`, `density_strata` all absent from
  `run_sigma`'s source) and by `build_gram_matrices`'s signature carrying no `idx` parameter.
- Full `notebooks/pu_manifold/tests/` suite: **748 passed, 1 skipped** (baseline before this
  plan: 742 passed, 1 skipped; +6 new import-purity tests, confirmed stable across two
  independent full runs after Task 1 and after Task 3).
- `cka.assert_preregistered()` still raises `RuntimeError` on `KERNELS` (first in declaration
  order); `len(cka._REQUIRED_CONSTANTS) == 37`, unchanged from 08-02's close.
- `git diff --name-only bb34575..HEAD -- notebooks/ src/` lists exactly the two
  `files_modified` this plan owns; `src/effdim/` untouched; no `torch` import anywhere in the
  runner.

## Task Commits

Each task was committed atomically:

1. **Task 1: D8-23 import-purity test — nine sealed modules, four import orders, no in-repo
   precedent** - `4c42023` (test)
2. **Task 2: Runner production data layer — frozen-field loading, record guards,
   freeze-ancestor gate** - `517cca5` (feat)
3. **Task 3: `--mode sigma` — the two frozen global RBF bandwidths and the
   Gram-matrix-once proof** - `a3a6059` (feat)

**Plan metadata:** commit pending (this SUMMARY + STATE.md + ROADMAP.md)

## Files Created/Modified

- `notebooks/pu_manifold/tests/test_cka_import_purity.py` - New. `SEALED_MODULES` (9-tuple),
  `_snapshot_module_state`, `_import_in_order`, and 6 tests across 3 named functions asserting
  D8-23 across 4 subprocess-isolated import orders plus a planted-mutation proof.
- `notebooks/diagnostics/08_cka_alignment_run.py` - +`_repo_root`, `_git_rev_parse`,
  `_strict_ancestor_or_exit`, `load_pu_pair`, `load_frozen_fields`, `FROZEN_FIELD_KEYS`,
  `build_gram_matrices`, `run_sigma`, `_available_memory_mb`, `_peak_rss_mb`,
  `_rbf_kernel_name`; `resolve_record_path` gained a `RECORD_STEM`-keyed default branch;
  `FREEZE_COMMIT_SHA`/`RECORD_STEM` module constants added (both `""`); `main()` dispatch
  updated so `sweep`/`positive-control`/`negative-control` call the ancestor gate first and
  `sigma` is now dispatched to `run_sigma`.

## Decisions Made

- **Import-purity comparison target is a cka-free baseline, not a literal per-order
  before/after pair** (see `key-decisions` in frontmatter) — the plan's literal wording does
  not extend cleanly to an interleaved order where some sealed modules are not yet imported at
  the moment `cka` is. The invariant every test actually asserts — a sealed module's final
  `vars()` state never depends on whether or when `cka` was imported in the same process — is
  the property D8-23 cares about, and is what all four orders (plus the baseline import
  itself, all nine with no `cka` at all) are compared against.
- **Order 4 ("cka alone, followed by the nine") implemented as cka-first-then-reversed-nine**,
  not a literal duplicate of order 1 ("cka first") — see `key-decisions`. This produces four
  genuinely distinct import sequences rather than three distinct sequences counted twice.
- **`SIGMA_GRAM_DTYPE`/`SIGMA_MULTIPLIERS` are runner-local literals**, not read from
  `cka.GRAM_DTYPE`/`cka.SIGMA_MULTIPLIERS` (both still UNSET) — never crossing the freeze
  boundary before 08-04, matching 08-01/08-02's own discipline.
- **Killed a duplicate, stray `pytest` process during execution** (PID 1970182, a leftover
  from an earlier background-job attempt that outlived its originating shell) that was
  competing for CPU against the tracked test run and stalling its progress at 9% for several
  minutes; killing it let the tracked run (`bj24wczrj`) complete normally in ~5m41s. Not a
  deviation from the plan — an execution-environment cleanup, noted for the record.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Frozenset repr is not reproducible across independent Python interpreters — false-positive import-purity failure**
- **Found during:** Task 1, first pytest run of the new suite
- **Issue:** `decoder_curvature.ZERO_SECOND_DERIVATIVE_ACTIVATIONS` is a `frozenset` of
  activation-function names. `_snapshot_module_state`'s initial implementation used a bare
  `repr(value)` for all non-array/non-callable values, including frozensets. CPython's string
  hash is randomized per process by default (no fixed `PYTHONHASHSEED`), so a
  frozenset-of-strings' iteration order — and therefore its `repr()` — differs between two
  independent subprocess launches even when the frozenset's *contents* are byte-identical and
  nothing mutated it. This produced 5 of 6 tests failing on the very first run, all with the
  identical false-positive diff on this one constant, in a module `cka.py` never even touches.
- **Fix:** Added a `set`/`frozenset` branch to `_snapshot_module_state` that fingerprints as
  `f"{type(value).__name__}:{sorted(repr(x) for x in value)}"` — sorting the element reprs
  removes the hash-randomization-induced ordering difference while still detecting a genuine
  content change (an added/removed/renamed element still changes the sorted list).
- **Files modified:** `notebooks/pu_manifold/tests/test_cka_import_purity.py`
- **Verification:** Re-ran the full test file 3 independent times after the fix (6/6 passing
  each time) specifically to rule out this exact class of flakiness recurring under a
  different random hash seed per run — not just a single passing run trusted at face value.
- **Committed in:** `4c42023` (Task 1 commit; fixed before committing, not a follow-up)

---

**Total deviations:** 1 auto-fixed (Rule 1 — a genuine bug in the first draft of a
brand-new-to-this-repo test pattern, caught and fixed before committing, with extra
verification specifically targeting recurrence under hash-seed variation).
**Impact on plan:** None on scope. The fix strengthens the snapshot comparison (it now
correctly treats content-identical sets as equal regardless of process-local iteration order)
rather than weakening it.

## Known Stubs

None. `--mode sigma`'s one-at-a-time disk-cache fallback branch (`_build_and_release` inside
`run_sigma`) is implemented but **not exercised** on this machine (measured available memory
25999 MB, far above the 6 GB `FALLBACK_THRESHOLD_MB` trigger) — this is not a stub, it is a
defensive code path documented as untested-in-this-environment. Recorded here for visibility:
if a future run on a memory-constrained machine takes this branch, its first real exercise
should be verified directly rather than assumed correct from code review alone.

## Issues Encountered

A stray duplicate `pytest` process (from an earlier background-job invocation whose owning
shell had already exited) was found running concurrently with the tracked full-suite run,
saturating CPU and stalling visible progress at 9% for several minutes. Diagnosed via `ps aux`
and `lsof` on the tracked run's output file (confirming which PID was the tracked one), then
killed the stray process; the tracked run completed normally afterward. No test result was
affected — the stray process was reading the same immutable source tree, not writing anything
contested.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `notebooks/.cache/08_scratch_sigma.jsonl` carries the full provenance record for 08-04's
  freeze: 8 rows, each with `sigma_median_pairwise`, `gram_build_s`, `gram_dtype`,
  `peak_rss_mb`, `subsample_file`, and a `timestamp`. **08-04's checkpoint reads
  `sigma_hsc = 0.6420152563705613` and `sigma_legacysurvey = 0.5696337821442163` directly from
  this SUMMARY (verbatim, full precision) to ratify `SIGMA_HSC`/`SIGMA_LEGACYSURVEY`.**
- `load_frozen_fields()` is verified against the real, currently-cached Phase 7/07.1 npz files
  on this machine — 08-05's sweep/positive-control/negative-control can call it directly.
- `_strict_ancestor_or_exit` is wired and tested to refuse every `--freeze-commit` value while
  `FREEZE_COMMIT_SHA` is empty; 08-04 only needs to set the literal once the freeze commit
  exists, no other runner change required for the gate itself to start working.
- `build_gram_matrices` is the exact function 08-05's sweep/null/controls will call to get
  their Gram-matrix-once dictionaries (its signature is stable: no subset index, `on_build`
  optional).
- All 37 gating constants remain UNSET, as required — 08-04 is still the single commit that
  may fill them. No blockers. `src/effdim/` confirmed untouched; only the two `files_modified`
  changed since the plan's base commit (`bb34575`).

---
*Phase: 08-curvature-conditioned-cka-alignment*
*Completed: 2026-08-28*

## Self-Check: PASSED

All created/modified files confirmed present on disk; all three task commits (`4c42023`,
`517cca5`, `a3a6059`) confirmed in `git log`.
