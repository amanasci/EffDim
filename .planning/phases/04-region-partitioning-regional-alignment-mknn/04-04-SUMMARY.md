---
phase: 04-region-partitioning-regional-alignment-mknn
plan: 04
subsystem: manifold-curvature
tags: [curvature, region-partitioning, density-confound, diametrical-clustering, pre-registration, mknn]

# Dependency graph
requires:
  - phase: 04-region-partitioning-regional-alignment-mknn
    provides: "plan 04-02's density-corrected k-freeze (K_FROZEN=500, rule_fired=false,
      notebooks/.cache/04_k_freeze.json); plan 04-03's region_partition.py (diametrical
      sign split, all frozen constants, assert_preregistered()) and 04-PREREGISTRATION.md"
provides:
  - "notebooks/diagnostics/region_partition_mknn_run.py: run_partition and --mode
    partition -- the density-corrected PU field at K_FROZEN=500/FIELD_D=20/K_DENSITY=30,
    REGN-01's ambient density, the D4-09 sign split, REGN-05's region counts, REGN-06's
    frozen artifact, REGN-02's density correlations, and the region-level Mann-Whitney
    comparison -- all before any regional MKNN number exists"
  - "notebooks/.cache/04_region_partition.npz (gitignored): v, labels, keep_idx,
    excluded_idx, h_norm, signed_projection, eigval_spectrum -- region_0=6256,
    region_1=3244, excluded=500, n_zero_projection=0"
  - "notebooks/.cache/04_density_diagnostics.json (gitignored): both REGN-02 Spearman
    correlations, the region-level Mann-Whitney comparison, eigval_top/eigval_spectrum_top5,
    mean_unit_norm, h_spread"
  - "notebooks/04_region_partition_mknn.ipynb sections 3-6 executed: the k-freeze table,
    the density confound reported before the split is trusted, the frozen split, and the
    PRE-REGISTRATION cell (index 28 of 30, precedes every cell a later plan will append)"
affects: [04-05, 04-06]

tech-stack:
  added: []
  patterns:
    - "run_partition takes every frozen parameter as a required keyword argument with no
      default, so --smoke can exercise the identical code path at reduced size/d/k without
      touching region_partition.py's own module-level constants, and the full run always
      names K_FROZEN/FIELD_D/K_DENSITY explicitly at the call site (D-07)"
    - "smoke mode skips both cache writes (npz and json) so a reduced-size smoke pass can
      never collide with the real frozen artifact's config manifest -- mirrors --mode
      global --smoke's own 'writes nothing' convention"
    - "the notebook recomputes REGN-01's density (local_density_weights, ~1.8s) rather than
      caching it per-point, since only its summary percentiles are persisted -- the
      recompute is a single k-NN query, not the heavy per-point estimator loop, so it stays
      inside CLAUDE.md's re-execution budget"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/region_partition_mknn_run.py
    - notebooks/04_region_partition_mknn.ipynb

key-decisions:
  - "The density confound is reported as this plan's headline result, not a footnote, per
    explicit coordinator direction: density vs the signed projection onto v is rho=+0.8208
    (n=9500, p~0) while density vs ||H|| is rho=-0.0273 (n=9500, p=0.0078) -- the
    pre-registered split axis is very nearly a density axis, and the confound is specific
    to the DIRECTION the partition uses, not to curvature magnitude. Under D4-14 this is
    the whole density battery (no partial regression, no density-matched null, no
    centroid-distance control, no density-matched stratification), so any regional MKNN
    difference plan 04-05 produces cannot be attributed to curvature rather than density by
    anything in this phase."
  - "Added eigval_spectrum to the frozen npz and eigval_spectrum_top5 to the density
    diagnostics json (Rule 1 fix, mid-plan): the plan's own Section 5 text requires the top
    five eigenvalues of the unit-H covariance, and the first field run did not persist them
    (only eigval_top). Fixed in code and the ~1446-1676s field computation was re-run rather
    than hand-patching the cached scientific artifact with values read off a stdout log --
    every reported number reproduced exactly across both full runs."
  - "The frozen npz is NOT byte-stable across runs (319,658 vs 326,064 bytes) even though
    every reported quantity is bit-for-bit identical. Investigated cheaply rather than
    re-running a third time: the delta (6,406 bytes) is fully accounted for by the added
    eigval_spectrum array (768 float64 = 6,144 bytes) plus per-entry zip overhead between
    the two runs -- an intentional code change, not non-determinism in the field estimator.
    This plan claims reproducing quantities, never a bit-identical artifact."
  - "Verified the cache manifest's changed-constant-raises guard directly and cheaply
    (mutate one key in the stored cfg, confirm cache.npz_cache raises ValueError before
    invoking compute_fn, confirm the unchanged cfg still loads) rather than inferring it
    from the full-field rerun, per explicit coordinator instruction. Also confirmed
    separately: an UNCHANGED cfg does not short-circuit run_partition's own heavy
    centroid_mean_curvature call -- the manifest guard's job is 'never silently return a
    stale artifact for a changed cfg', not 'skip recomputation for an unchanged one', and
    run_partition follows the sibling pu_curvature_rankability_run.py's own no-skip
    precedent."
  - "Fixed a notebook bug (Rule 1) found during execution: Section 6's git-log provenance
    call used a repo-root-relative pathspec, but nbconvert executes with cwd = the
    notebook's own directory, so the commit hash/timestamp printed empty on first
    execution. Resolved by calling git rev-parse --show-toplevel explicitly before the
    pathspec lookup; the notebook was re-executed end to end to pick up the fix."

requirements-completed: [REGN-01, REGN-02, REGN-05, REGN-06]

coverage:
  - id: D1
    description: "REGN-06: the frozen partition artifact (v, labels, keep_idx,
      excluded_idx, h_norm, signed_projection, plus eigval_spectrum) written via
      cache.npz_cache before any regional MKNN number exists; region membership counts
      (region_0=6256, region_1=3244, excluded=500) sum exactly to 10000"
    requirement: REGN-06
    verification:
      - kind: integration
        ref: "notebooks/diagnostics/region_partition_mknn_run.py --mode partition, plus
          the plan's own <verify> assertion script (key presence, labels/keep_idx length
          match, bincount+excluded sums to 10000, diagnostics dict key/value checks) run
          directly against the real artifacts"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -q (376 passed, 1 skipped, unaffected -- no
          sealed module touched)"
        status: pass
    human_judgment: false
  - id: D2
    description: "REGN-01: ambient 768-d local density (rho = 1/w, w mean-normalized to 1)
      computed for all 10,000 points and shown -- p05/p50/p95, spread, and a histogram --
      before the split is trusted"
    requirement: REGN-01
    verification:
      - kind: other
        ref: "notebook Section 4 (cells 16-17), 04_density_diagnostics.json's
          rho_p05/rho_p50/rho_p95/rho_p95_over_p05 fields, inspected directly"
        status: pass
    human_judgment: true
    rationale: "04-VALIDATION.md's Manual-Only table: this is a reporting obligation ('rho
      is shown') that no mechanical check can confirm beyond the cell having run --
      pre-committing a pass/fail threshold here would itself be a forking path."
  - id: D3
    description: "REGN-02: both Spearman correlations (density vs ||H||, density vs the
      signed projection onto v) reported with rho, p-value and n on the surviving
      non-excluded points, plus the region-level median/IQR/Mann-Whitney comparison --
      density vs projection rho=+0.8208 (n=9500) against density vs ||H|| rho=-0.0273
      (n=9500); region medians differ ~5,735x (Mann-Whitney p=0)"
    requirement: REGN-02
    verification:
      - kind: other
        ref: "notebook Section 4 (cells 17, 19) and Section 5 (cell 26);
          04_density_diagnostics.json's spearman_density_vs_hnorm,
          spearman_density_vs_projection, mannwhitneyu_statistic/pvalue,
          median/iqr_density_region_0/1 fields"
        status: pass
    human_judgment: true
    rationale: "04-VALIDATION.md's Manual-Only table: the deliverable is a reported
      statistic and its interpretation (whether the confound is the dominant explanation on
      the table), not a pass/fail threshold -- pre-committing a threshold here would itself
      be a forking path."
  - id: D4
    description: "REGN-05: both region counts, the excluded count, and the
      exact-zero-projection count printed and asserted to sum to 10000; both regions
      (6256, 3244) clear MIN_REGION_N=500, so no cell is undefined on size grounds"
    requirement: REGN-05
    verification:
      - kind: integration
        ref: "notebook Section 5 (cells 21-22); the plan's own <verify> assertion script's
          bincount-sums-to-10000 check, run directly"
        status: pass
    human_judgment: false
  - id: D5
    description: "Notebook sections 3-6 executed end to end, no reordering/editing/deletion
      of sections 0-2's cell source; the PRE-REGISTERED cell (assert_preregistered() +
      VERDICT_RULE + commit provenance) is at index 28 of 30, preceding every cell any
      later plan will append -- the Ordering constraint made visible in the notebook itself"
    verification:
      - kind: integration
        ref: "jupyter nbconvert --to notebook --execute --inplace, plus the plan's own
          post-execution assertion script (every code cell has execution_count,
          pre-registration cell index check, VERDICT_RULE/mannwhitneyu/spearman/r_over_R
          text presence) run directly; git diff confirmed cells 0-10 source byte-identical"
        status: pass
    human_judgment: false

duration: ~1h20min this session (dominated by two full-field density-corrected
  centroid_mean_curvature computations at n=10000, k=500, d=20: 1675.9s then 1446.1s after
  the eigval_spectrum fix, ~52 min combined background compute)
completed: 2026-08-24
status: complete
---

# Phase 4 Plan 4: PU field at frozen k, sign split, density confound reported before it is trusted Summary

**The density confound is this plan's headline result, not a footnote: density vs the
signed projection onto the frozen split axis `v` measures `rho=+0.8208` (n=9500) while
density vs curvature magnitude `||H||` is essentially nil at `rho=-0.0273` — the
pre-registered split axis is very nearly a density axis, the confound is specific to the
DIRECTION the partition uses rather than to curvature magnitude, and under D4-14's declined
controls no regional MKNN difference plan 04-05 produces can be attributed to curvature
rather than density by anything in this phase.**

## Performance

- **Duration:** ~1h20min this session's active work, dominated by two full-field
  `centroid_mean_curvature` computations at `n=10000, k=500, d=20, density_correct=True,
  k_density=30`: 1675.9s (first run) and 1446.1s (verification rerun after the
  `eigval_spectrum` fix), ~52 min combined background compute
- **Tasks:** 2/2 complete
- **Files modified:** 2 (both tracked; two gitignored `.cache` artifacts also produced)

## Accomplishments

- `run_partition` and `--mode partition` added to `region_partition_mknn_run.py`: computes
  the density-corrected PU field on the `legacysurvey` column at the frozen `K_FROZEN=500`,
  `FIELD_D=20`, `K_DENSITY=30` (every parameter named explicitly at the call site, per
  D-07), REGN-01's ambient 768-d local density, the D4-09 diametrical sign split, REGN-05's
  region counts, REGN-06's frozen artifact, REGN-02's density Spearman correlations, and the
  region-level Mann-Whitney comparison — printed in the Ordering constraint's own sequence,
  with no regional MKNN number computed anywhere.
- **The measured split:** `region_0=6256` (62.6%), `region_1=3244` (32.4%), `excluded=500`
  (5.0%), `n_zero_projection=0` — sums exactly to 10,000. Both regions clear
  `MIN_REGION_N=500`; no cell is undefined on size grounds for this split.
- **The measured density confound:** `spearman(density, signed_projection) = +0.8208`
  (n=9500, p≈0) against `spearman(density, ||H||) = -0.0273` (n=9500, p=0.0078). Region-level
  density medians differ by a factor of ~5,735 (`3.76e10` vs `6.56e6`, Mann-Whitney U
  statistic=18844954.0, p=0.0). This is not softened into "a confound to bear in mind": at
  `rho=+0.82` it is the dominant explanation available for any regional MKNN result 04-05
  produces, and D4-14 declined every control (partial regression, density-matched null,
  centroid-distance control, density-matched stratification) that could have separated it
  from curvature.
- **`mean_unit_norm=0.294748`**: the mean-centered and uncentered second-moment covariance
  forms do NOT coincide at this magnitude, so `COVARIANCE_FORM="mean_centered"` (04-03's
  flagged question) is a live choice that affected the axis `v`, not a settled formality.
- **Eigenvalue separation is weak**: top eigenvalue `0.0316` against second `0.0202`
  (ratio 1.57) — `v` is reported as the chosen split axis and is explicitly NOT presented as
  a well-separated principal axis of the unit-`H` covariance.
- **`K_FROZEN=500` provenance restated once more**: a compute-budget ceiling
  (`rule_fired=false`), never described as converged or plateaued anywhere in this plan's
  own text.
- **D4-05 spread comparison, carried in the phase's own words**: PU's measured `||H||`
  spread at k=500 is 3.94x (p95/p05), far nearer the runner's own unrankable
  `quadratic_bowl` calibration (1.4x) than the rankable `cubic`/`ridge` calibration
  (28.2x/34.3x). This gates nothing — direction is a unit vector and does not consume the
  magnitude spread.
- **Codimension caveat, carried in the phase's own words**: every fixture the
  direction-partition decision (D4-01) rests on is codimension 1, where
  `H = H_scalar * n_hat`; PU's codimension is roughly 748 (`d~20` inside `D=768`). A cosine
  near 1.000 on those fixtures demonstrates recovery of a surface's normal orientation, a
  tangent-space problem known to converge well — not resolution of `H`'s direction inside a
  748-wide normal space. That gap is unmeasured on PU and unclosed by anything in this
  milestone.
- Notebook sections 3-6 appended and executed end to end: the k-freeze table (Section 3),
  the density confound reported before the split is trusted (Section 4, including the
  headline-finding cell above), the frozen split with region-level density comparison
  (Section 5), and the PRE-REGISTRATION cell (Section 6, index 28 of 30) — every code cell
  has a non-null `execution_count`, and cells 0-10 (sections 0-2) are source-byte-identical
  to before this plan's work.
- `notebooks/.cache/04_region_partition_mknn.jsonl` still holds exactly its 6 `global` rows
  from plan 04-01 — confirmed directly, no `region != "global"` row exists anywhere.

## Task Commits

Each task was committed atomically:

1. **Task 1: Field at the frozen k, the sign split, and the density confound** -
   `464b906` (feat)
2. **Task 2: Notebook sections 3-6 — field, density, partition, and the pre-registration
   cell** - `fc150a4` (feat)

**Plan metadata:** committed as part of this SUMMARY's own docs commit.

## Files Created/Modified

- `notebooks/diagnostics/region_partition_mknn_run.py` — `run_partition`, `_spearman_report`,
  `--mode partition` branch (full run + `--smoke`)
- `notebooks/04_region_partition_mknn.ipynb` — sections 3-6 appended and executed
- `notebooks/.cache/04_region_partition.npz` (gitignored) — the REGN-06 frozen artifact
- `notebooks/.cache/04_density_diagnostics.json` (gitignored) — the REGN-01/REGN-02
  diagnostics

## Decisions Made

- **The density confound is the plan's headline result, not a diagnostic footnote** — see
  frontmatter `key-decisions` and the Accomplishments section above for the precise numbers
  and framing. This was an explicit coordinator direction given the measured `rho=+0.8208`.
- **`eigval_spectrum` added to the frozen artifact mid-plan (Rule 1 fix)** — the plan's own
  Section 5 text requires the top five eigenvalues of the unit-`H` covariance, which the
  first field run did not persist (only `eigval_top`). Fixed in `run_partition` (added
  `eigval_spectrum` to the npz, `eigval_spectrum_top5` to the json) and the full
  `--mode partition` run was repeated (1446.1s) rather than hand-editing a cached scientific
  artifact with values transcribed from a stdout log. Every previously-reported quantity
  reproduced exactly.
- **The frozen npz's byte size differs across the two runs (319,658 vs 326,064 bytes) —
  investigated and explained, not a live concern.** The 6,406-byte delta is fully accounted
  for by the added `eigval_spectrum` array (768 float64 = 6,144 bytes of array data plus
  per-entry zip overhead) between the two runs — an intentional code change, not
  non-determinism in the field estimator. SUMMARY claims reproducing quantities across
  independent runs, never a bit-identical artifact.
- **Verified the manifest's changed-constant-raises guard directly rather than inferring it
  from the rerun.** Mutating one key (`K_FROZEN`) in the stored cfg and calling
  `cache.npz_cache` with a `compute_fn` that raises if ever invoked confirmed the `ValueError`
  fires immediately, before any recomputation — and confirmed separately that the unchanged
  cfg loads without recomputation. Also noted explicitly: an *unchanged* cfg does not make
  `run_partition` itself skip its own heavy field computation (the manifest guard only
  gates the persisted-artifact write/read, matching `pu_curvature_rankability_run.py`'s own
  no-skip precedent) — the plan's line 218 acceptance criterion ("re-running with any
  changed constant raises") is about the changed-constant path specifically, and that is
  what was verified.
- **Section 6's git-log provenance bug (Rule 1 fix).** `git log -- <pathspec>` resolves the
  pathspec relative to CWD; `nbconvert` executes with `cwd = notebooks/`, not the repo root,
  so the commit hash/timestamp printed empty on first execution. Fixed by resolving
  `git rev-parse --show-toplevel` first; the notebook was re-executed end to end to pick up
  the fix (all 30 cells, not just the patched one, per the plan's own re-execution
  requirement).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing `eigval_spectrum` in the frozen artifact**
- **Found during:** Task 2 (building notebook Section 5, which requires the top five
  eigenvalues of the unit-`H` covariance)
- **Issue:** `run_partition`'s first version computed `eigval_spectrum` in memory (printed
  its top 5 to stdout) but never persisted it to either cached artifact, so the notebook
  could not render it without recomputing the heavy field.
- **Fix:** Added `eigval_spectrum` (full 768-length array) to the frozen npz and
  `eigval_spectrum_top5` to the density diagnostics json; deleted the stale cached artifacts
  and re-ran `--mode partition` in full.
- **Files modified:** `notebooks/diagnostics/region_partition_mknn_run.py`
- **Verification:** Every previously-reported quantity (region counts, both Spearman
  correlations, Mann-Whitney statistic, `mean_unit_norm`, `eigval_top`) reproduced exactly
  across the two full runs; the new `eigval_spectrum_top5` values matched the first run's
  stdout log exactly.
- **Committed in:** `464b906` (Task 1 commit, since the fix landed before Task 1 was
  committed)

**2. [Rule 1 - Bug] Section 6's git-log provenance printed empty**
- **Found during:** Task 2, first notebook execution
- **Issue:** `git log -1 --format=%h %cI -- <repo-root-relative path>` returned empty
  because `nbconvert` executes with `cwd = notebooks/`, and the pathspec resolved relative
  to that directory rather than the repo root — `git log --` fails silently (exit 0, no
  output) rather than raising.
- **Fix:** Resolve `git rev-parse --show-toplevel` explicitly before building the pathspec.
- **Files modified:** `notebooks/04_region_partition_mknn.ipynb` (Section 6 code cell)
- **Verification:** Re-executed the notebook end to end; the cell now prints
  `04-PREREGISTRATION.md committed at: 0305c77 2026-08-24T07:40:26-04:00`.
- **Committed in:** `fc150a4` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs found and fixed during Task 2, before
either task's commit).
**Impact on plan:** Both fixes were necessary for the plan's own acceptance criteria
(persisting the eigenvalue spectrum Section 5 requires; the commit provenance Section 6
requires) and were caught and corrected before either task's commit. No scope creep — no
regional MKNN number was computed at any point.

## Issues Encountered

- **A duplicate background launch of the full field computation** occurred early in Task 1
  (two independent `python` processes targeting the same `04_region_partition.npz`, 20
  seconds apart, both still in the field computation phase). The coordinator killed the
  later duplicate before either wrote the artifact, so the npz came from a single clean
  process. Confirmed no stray processes were running before any subsequent background
  launch in this plan.
- **The frozen npz's byte size is not stable across otherwise-identical runs** — see
  Decisions Made above. Investigated and fully explained (an intentional code change, not
  non-determinism); not re-investigated further per explicit instruction not to spend
  additional wall-clock on it.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 04-05 inherits a frozen split with a documented, dominant density confound.** The
  region-level density medians differ by ~5,735x, and `spearman(density, signed_projection)
  = +0.8208` is the strongest correlation measured anywhere in this plan. Any regional MKNN
  difference 04-05 measures must be read against this number, not despite it — D4-14
  declined every control that could separate curvature from density, so 04-05 cannot resolve
  this ambiguity on its own; it can only report a regional MKNN result alongside the
  confound this plan established.
- **`--mode regional`'s guard (from plan 04-03) now passes its artifact-existence check** —
  `notebooks/.cache/04_region_partition.npz` exists and matches the pre-registered cfg.
  Plan 04-05 is the first plan permitted to compute a regional MKNN cell.
- **Both regions clear `MIN_REGION_N=500`** (6256, 3244) — no cell is undefined on size
  grounds; 04-05 does not need to special-case an undersized region for this split.
- **`eigval_spectrum` is now an extra key in the frozen npz** beyond REGN-06's required six
  (`v`, `labels`, `keep_idx`, `excluded_idx`, `h_norm`, `signed_projection`) — any later
  plan reading this artifact should not assume exactly six keys.
- No blockers for 04-05. The three accepted gaps restated across this phase (unvalidated
  field, unclosed codimension gap, reported-not-controlled density confound) are unchanged
  and remain independently tracked for 04-06's phase record.

---
*Phase: 04-region-partitioning-regional-alignment-mknn*
*Completed: 2026-08-24*

## Self-Check: PASSED
- FOUND: notebooks/diagnostics/region_partition_mknn_run.py
- FOUND: notebooks/04_region_partition_mknn.ipynb
- FOUND: .planning/phases/04-region-partitioning-regional-alignment-mknn/04-04-SUMMARY.md
- FOUND: notebooks/.cache/04_region_partition.npz
- FOUND: notebooks/.cache/04_density_diagnostics.json
- FOUND: commit 464b906
- FOUND: commit fc150a4
- FOUND: commit 7afb73a
