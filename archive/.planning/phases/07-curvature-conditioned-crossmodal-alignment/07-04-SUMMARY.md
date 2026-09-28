---
phase: 07-curvature-conditioned-crossmodal-alignment
plan: 04
subsystem: research-instrumentation
tags: [curvature, mknn, spearman, permutation-test, positive-control, density-confound, d-sweep]

# Dependency graph
requires:
  - phase: 07-curvature-conditioned-crossmodal-alignment
    provides: "crossmodal_curvature.py's frozen constants block, freeze commit f032745f6450068c63763993d39fa112fd36bb8c (07-01); --mode dsweep as one serial in-process loop (07-04 Task 1, a453736); plant_positive_control / density_diagnostics / --mode positive-control (07-03)"
provides:
  - "The phase's headline numbers: three-d sweep (d=20,25,32) of spearman(||H||, MKNN) with two-tailed permutation clearance, density/hubness diagnostics, and the positive control at PU's own realized d=20 dynamic range, all written to notebooks/.cache/07_crossmodal_curvature.jsonl"
  - "The mechanically applied verdict: ASSOCIATION DETECTED, computed by crossmodal_curvature.apply_verdict on the three sweep rows' clears_either booleans and the positive control's smallest_cleared_target, never chosen after reading the numbers"
affects: [07-05-plan]

tech-stack:
  added: []
  patterns:
    - "Verdict row appended via direct invocation of the frozen production functions (cc.apply_verdict, the runner's append_record_row/resolve_record_path/_git_rev_parse) from an inline script rather than a new CLI mode -- the plan's Task 3 <files> tag names only the two gitignored data artifacts, not the runner .py, so no new tracked code was warranted for this step."
    - "The 'commit strictly descended from the freeze' proof is satisfied by the record's own stamped run_commit (a453736, the Task 1 code commit) rather than by committing the data artifacts themselves -- notebooks/.cache/ is gitignored per CLAUDE.md's explicit milestone-artifact convention, matching every prior 07-0x plan's practice of never committing the jsonl/npz to git."

key-files:
  created: []
  modified:
    - notebooks/.cache/07_crossmodal_curvature.jsonl
    - notebooks/.cache/07_crossmodal_curvature_fields.npz

key-decisions:
  - "Verdict computed strictly mechanically: per_d_results = {20: True, 25: True, 32: True} (all three d's clears_either read directly off the sweep rows) and positive_control_cleared_at = 0.05 (the smallest POSITIVE_CONTROL_TARGET_RHOS entry whose recorded clears_either was True) passed to crossmodal_curvature.apply_verdict, which returned ASSOCIATION DETECTED. No number was adjusted after being seen."
  - "The ratified --threads 8 does NOT appear on the three sweep rows (Task 1's schema has no threads field) -- recorded instead on the verdict row (threads: 8) and here in the SUMMARY, per the plan's explicit instruction not to retroactively edit already-written rows."
  - "The density confound is reported, not resolved: partial_rho_density_controlled collapses ~78% of partial_rho_raw at d=20 (-0.1122 -> -0.0242) and ~49% at d=25 (-0.1279 -> -0.0658); at d=20 and d=32 the density-controlled partial sits only marginally above its own d's null threshold (0.0242 vs 0.0206 at d=20; 0.0217 vs 0.0187 at d=32) rather than clearly clear of it. DIAGNOSTICS_ARE_NON_GATING is honored exactly: this does not change the verdict, and is stated plainly rather than buried."

requirements-completed: [D7-01, D7-02, D7-03, D7-04, D7-06]

coverage:
  - id: D1
    description: "The real three-d sweep (d=20,25,32), one serial in-process run under --threads 8, ~2h07m wallclock, writing three sweep rows with the headline Spearman, both tail thresholds, var_explained, cond(g), ||H|| distribution, sensitivity grid, and density/hubness diagnostics at each d"
    requirement: "D7-01"
    verification:
      - kind: other
        ref: "notebooks/.cache/07_crossmodal_curvature.jsonl -- 3 sweep rows with d=20,25,32 in D_SWEEP order, all keys present, preregistration_commit=f032745..., run_commit=a453736... on every row"
        status: pass
      - kind: unit
        ref: "notebooks/pu_manifold/tests/ -q -- 573 passed, 1 skipped"
        status: pass
    human_judgment: false
  - id: D2
    description: "The D7-02 positive control run against PU's own realized d=20 ||H|| field (read from the npz's bare h_norm key), recovering smallest_cleared_target=0.05"
    requirement: "D7-02"
    verification:
      - kind: other
        ref: "notebooks/.cache/07_crossmodal_curvature.jsonl -- 4 positive_control rows, one per POSITIVE_CONTROL_TARGET_RHOS entry (0.02 not cleared, 0.05/0.10/0.20 cleared, all achieved_rho within 0.0001 of target)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Verdict applied mechanically via crossmodal_curvature.apply_verdict and appended as exactly one verdict row: ASSOCIATION DETECTED"
    requirement: "D7-06"
    verification:
      - kind: other
        ref: "notebooks/.cache/07_crossmodal_curvature.jsonl -- exactly 1 row with row_kind=verdict, verdict in VERDICT_VALUES, per_d_clears_either={20:true,25:true,32:true}, positive_control_cleared_at=0.05"
        status: pass
      - kind: other
        ref: "git merge-base --is-ancestor f032745...  a453736... (exit 0); git rev-list --count f032745...a453736... = 10 (>=1)"
        status: pass
      - kind: other
        ref: "git diff --stat f032745..HEAD -- notebooks/pu_manifold/crossmodal_curvature.py -- 365 insertions, 0 deletions (no constant block line changed since freeze)"
        status: pass
    human_judgment: false

duration: ~2h50min (Task 1 authoring + Task 2 checkpoint wait + ~2h07m real sweep + positive control/verdict/full pytest wrap-up)
completed: 2026-08-26
status: complete
---

# Phase 7 Plan 4: The Real d-Sweep, Positive Control, and Mechanically Applied Verdict Summary

**The pre-registered three-d sweep (d=20,25,32) of PU curvature vs. MKNN comes back ASSOCIATION DETECTED — all three d clear the negative tail on `spearman(||H||, MKNN)` (rho -0.11 to -0.02), licensed by a positive control that recovers targets down to 0.05 on PU's own realized d=20 dynamic range — but a density partial that collapses 49-78% of the raw correlation, with the d=20 and d=32 density-controlled residuals sitting only marginally above their own null thresholds, means the finding is not clean of the same density confound Phase 4 already recorded.**

## Performance

- **Duration:** ~2h50min total across this plan (Task 1 authoring in a prior session; Task 2's blocking checkpoint approved `--threads 8`, executor-launched; Task 3's real sweep ran ~2h07m wallclock, 09:35 -> ~11:42 local; positive control, mechanical verdict, full pytest, and this SUMMARY followed in one continuous wrap-up session)
- **Completed:** 2026-08-26
- **Tasks:** 3 (Task 1: `a453736`; Task 2: human-approved checkpoint, no code commit; Task 3: data-only, no code commit — see Deviations)
- **Files modified:** 2 (both gitignored data artifacts: `notebooks/.cache/07_crossmodal_curvature.jsonl`, `notebooks/.cache/07_crossmodal_curvature_fields.npz`)

## Accomplishments

- **The real three-d sweep ran to completion, one serial in-process loop, `--threads 8`.** Total measured wallclock ~7,643s (~2h07m): fit 338.8s/314.0s/424.1s and field 1,773.7s/2,139.7s/2,652.7s at d=20/25/32 respectively. Every row carries `preregistration_commit=f032745f6450068c63763993d39fa112fd36bb8c` and `run_commit=a4537369be204b784d026ac36c6bfc7b14ea483d`.
- **Headline statistic, all three d:**

  | d | observed rho | direction | neg-tail thresh | clears neg | pos-tail thresh | clears pos | clears_either |
  |---|---|---|---|---|---|---|---|
  | 20 | -0.11218 | negative | 0.020624 | True | 0.020983 | False | **True** |
  | 25 | -0.12789 | negative | 0.019683 | True | 0.018896 | False | **True** |
  | 32 | -0.02373 | negative | 0.018728 | True | 0.018613 | False | **True** |

  All three clear on the negative tail — the direction the research hypothesis predicted.
- **Fit quality and conditioning, all three d.** Holdout var_explained: 0.98194 (d=20), 0.98432 (d=25), 0.98647 (d=32) — high, but per the reporting constraint this is NOT read as licensing a precise curvature-fidelity claim; `INSTRUMENT_FIDELITY_RANGE` (`+0.53` to `+0.99`) is quoted as a range, never a point estimate, because reconstruction quality did not predict fidelity when both were measured together (99.70% recon scored `+0.5253`, 99.88% scored `+0.9745`). `cond(g)` median: 15.73 (d=20), 17.98 (d=25), 15.54 (d=32).
- **`||H|| ` spread, order of magnitude only (never a precise quantity, per the instrument's measured est/true ratio swinging 0.665-1.626 non-monotonically).** All three d's `||H||` distributions sit in the same order of magnitude — tens (roughly 10^1) — with a narrow p95/p05 spread within each d (e.g. d=20 median 37.19, p05 30.91, p95 43.50): consistent with the phase's prior finding that PU's realized dynamic range is narrow (~1.5x), not the ~20x Phase 6's retired selfcheck assumed.
- **Sensitivity grid (point-estimate only, non-gating), all three d**, agrees in sign and rough magnitude with the headline k=20 value at every neighboring k in `MKNN_K_GRID`: d=20 {k=5: -0.0938, k=10: -0.1077, k=50: -0.1292}; d=25 {k=5: -0.0846, k=10: -0.1078, k=50: -0.1530}; d=32 {k=5: -0.0189, k=10: -0.0223, k=50: -0.0401}.
- **The `HEADLINE_K=20` MKNN array's distinct-value count is 15** (computed once, before the d-loop, at relative precision — never raw float equality). Full `mknn_n_distinct_by_k` (identical across d, since MKNN depends only on the frozen embeddings and k): `{5: 5, 10: 9, 20: 15, 50: 36}`.
- **Density and hubness diagnostics — reported prominently, gating nothing (D7-03).** Computed once (density is d-independent) and re-combined with each d's field:
  - `spearman_density_vs_mknn = -0.2121` (constant across d).
  - `density_ratio_p95_p05 = 5.98e7` — a striking diagnostic oddity, flagged here as a data-quality note rather than interpreted further; density p05=6.07e4, p50=2.29e9, p95=3.63e12.
  - `hubness_skewness_a = 1.0486`, `hubness_skewness_b = 1.1880` (both columns, constant across d).
  - `chance_floor = 0.002` (at n=10,000, k=20 — an order of magnitude below what the paper's n=101,725 regime would show; per the plan's prohibition, this number is NOT extrapolated to that regime).
  - **The density partial, per d** — `spearman_density_vs_h` / `partial_rho_raw` / `partial_rho_density_controlled`:
    - d=20: 0.4281 / -0.11218 / **-0.02419** — the density-controlled partial collapses to ~22% of the raw value (~78% of the association explained by density), and sits only 17% above its own d's null threshold (0.0242 vs 0.0206) rather than clearly clear of it.
    - d=25: 0.3150 / -0.12789 / **-0.06583** — collapses to ~51% of raw (~49% explained by density); this one stays clearly above its threshold (0.0658 vs 0.0197, over 3x).
    - d=32: 0.0118 / -0.02373 / **-0.02172** — collapses only slightly (~92% of raw retained), but the raw value itself is already the weakest of the three, and the density-controlled residual again sits only 16% above its own threshold (0.0217 vs 0.0187).
  - **This mirrors Phase 4's own recorded finding** — its HOLDS verdict was 0.82 correlated with density and mostly a region-size artifact. `DENSITY_DIAGNOSTICS_ARE_NON_GATING` per D7-03's pre-registration, so none of this changes the verdict below — but it means the ASSOCIATION DETECTED verdict should not be read as a curvature-alignment result independent of density.
- **The positive control (D7-02), planted on PU's own realized d=20 `||H||` field** (the npz's bare `h_norm` key, verified byte-identical to `h_norm_20`), at `HEADLINE_K=20`, `POSITIVE_CONTROL_SEED=20260825`:

  | target rho | achieved rho | clears_either | direction |
  |---|---|---|---|
  | 0.02 | 0.02004 | False | neither |
  | 0.05 | 0.05003 | **True** | positive |
  | 0.10 | 0.10004 | **True** | positive |
  | 0.20 | 0.20004 | **True** | positive |

  `smallest_cleared_target = 0.05`. The test has power to detect a planted relationship as small as 0.05 on PU's own realized field — the observed magnitudes at d=20/25 (0.112, 0.128) comfortably exceed this floor; d=32's observed magnitude (0.024) sits close to the control's own un-cleared 0.02 target, closer to the power boundary than the other two d.
- **Verdict, applied mechanically by `crossmodal_curvature.apply_verdict`, never chosen after reading the numbers: `ASSOCIATION DETECTED`.** All three d's `clears_either` were `True`, so `apply_verdict({20: True, 25: True, 32: True}, 0.05)` returns the "all clear" branch. Appended as exactly one `row_kind=verdict` row carrying the returned string, the three per-d booleans, the positive control's target grid and cleared value, `VERDICT_RULE`'s first line for traceability, `threads: 8`, and both stamped SHAs.
- **Full regression: `notebooks/pu_manifold/tests/ -q` — 573 passed, 1 skipped**, unchanged since 07-03, run after the sweep completed (safe: the machine was free).
- **Ancestry re-verified against the record's own stamped SHAs, not ambient git state:** `git merge-base --is-ancestor f032745...  a453736...` exits 0; `git rev-list --count f032745..a453736 = 10` (>= 1, closing the self-ancestor gap `--is-ancestor` alone would miss).
- **No constant inside the frozen block changed:** `git diff --stat f032745..HEAD -- crossmodal_curvature.py` shows 365 insertions, 0 deletions across the whole phase to date — zero deletions since the freeze means no existing line (including any constants-block line) was altered.

## Task Commits

1. **Task 1: Implement `--mode dsweep` as one serial in-process loop** — `a453736` (feat) [prior session]
2. **Task 2: Gate the two-hour serial run** — human-approved checkpoint (`--threads 8`, executor-launches); no code commit
3. **Task 3: Run the real sweep, the positive control, and apply the verdict** — no code commit; all outputs are gitignored data artifacts (`notebooks/.cache/`), consistent with every prior 07-0x plan (see Deviations)

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `notebooks/.cache/07_crossmodal_curvature.jsonl` — the frozen record: 3 sweep rows (d=20,25,32), 4 positive-control rows, 1 verdict row. Gitignored per `CLAUDE.md`'s milestone-artifact convention; not tracked in git.
- `notebooks/.cache/07_crossmodal_curvature_fields.npz` — per-d `||H||` and `cond(g)` arrays (`h_norm_20`/`cond_g_20`, `_25`, `_32`, plus a bare `h_norm`/`cond_g` alias to the d=20 field that `--mode positive-control` reads). Gitignored, not tracked.

## Decisions Made

See `key-decisions` in frontmatter above. In short: the verdict was computed strictly by reading `clears_either` off the three already-written sweep rows and `smallest_cleared_target` off the four already-written positive-control rows, passing both directly to `crossmodal_curvature.apply_verdict` with no intermediate judgment call; the ratified `--threads 8` was recorded on the new verdict row and here rather than retroactively edited into the sweep rows; and the density confound is reported plainly rather than allowed to imply a clean result.

## Deviations from Plan

**1. [Clarification, not a functional deviation] "Commit the record; prove the commit is strictly descended from f032745" is satisfied via the record's own stamped `run_commit` (the Task 1 code commit `a453736`), not by a new git commit of the jsonl/npz themselves.** `notebooks/.cache/` is gitignored per `CLAUDE.md`'s explicit statement that "Milestone artifacts live in the gitignored `notebooks/.cache/`" — confirmed by `git check-ignore -v` on both files, and matching every prior 07-0x plan's practice (07-02 and 07-03 both used and cleaned up scratch/synthetic record paths under the same directory without ever adding them to git). The plan's own acceptance criteria for this exact check ("re-checked against the record's own stamped SHAs") describe precisely the verification performed: `git merge-base --is-ancestor f032745 a453736` and `git rev-list --count f032745..a453736 >= 1`, both against the SHA already stamped on every row. No git-tracked file needed a new commit for this task.

**2. [Minor, administrative] The verdict row was appended via a one-off inline invocation of the existing production functions (`crossmodal_curvature.apply_verdict`, and the runner's `append_record_row`/`resolve_record_path`/`_git_rev_parse`) rather than through a new `--mode verdict` CLI mode.** The plan's Task 3 `<files>` tag names only the two data artifacts, not `07_crossmodal_curvature_run.py`, and its `<action>` describes "read... call... append" without specifying a new CLI surface. Per `CLAUDE.md`'s "keep things simple first" directive, no new code path was added for a one-time mechanical step that reuses frozen functions verbatim. The exact invocation is reproducible from this SUMMARY's description (read `clears_either` per d, read `clears_either`/`target_rho` per positive-control row, call `apply_verdict`, append one row) against the record already on disk.

No other deviations — the sweep, the positive control, and the verdict all ran exactly per the plan's `<action>` and satisfy every stated `<acceptance_criteria>`.

## Issues Encountered

None. The sweep ran uninterrupted to completion (no `--resume` restart was needed); the positive control, verdict application, and full test suite all completed on the first attempt.

## Known Stubs

None. All three sweep rows, all four positive-control rows, and the verdict row carry real measured PU numbers — no synthetic surrogate, no placeholder.

## Threat Flags

None beyond the plan's own pre-registered threat model. `T-07-02` (frozen-constant tampering) is confirmed mitigated: `assert_preregistered()` ran first in every mode invoked, and the zero-deletions diff against the freeze proves no constant was touched. `T-07-03` (record repudiation) is confirmed mitigated: every one of the 8 rows in the finished record — 3 sweep, 4 positive-control, 1 verdict — carries both `preregistration_commit` and `run_commit`, matching the value independently re-derived from git. `T-07-05` (a degenerate positive control licensing a null) did not fire: the positive control cleared at 3 of 4 targets on PU's own field, and the verdict path taken (`ASSOCIATION DETECTED`) does not depend on the D7-02 override branch at all.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

Plan 07-05 (the reporting notebook and `07-FINDINGS.md`) may now begin. The frozen record (`notebooks/.cache/07_crossmodal_curvature.jsonl`, 8 rows) and the fields npz are complete and verified: 3 sweep rows in `D_SWEEP` order, 4 positive-control rows, exactly 1 verdict row, every required key present, verdict `ASSOCIATION DETECTED` a member of `VERDICT_VALUES`, ancestry proven against the record's own stamped SHAs. The density confound (49-78% collapse under partial correlation, d=20/d=32 sitting only marginally above their own null thresholds) is the single most important caveat for 07-05 to carry forward prominently — this phase's ASSOCIATION DETECTED result is not independent of density in the way a clean finding would be, and 07-05 should state that as plainly as this SUMMARY does, not soften it. `d=32`'s much weaker raw association (-0.0237, barely above its own threshold and close to the positive control's un-cleared 0.02 floor) is the second caveat: the sweep is not monotonically strong across `d`, and 07-05 should report the per-d spread rather than only the headline verdict string.

---
*Phase: 07-curvature-conditioned-crossmodal-alignment*
*Completed: 2026-08-26*

## Self-Check: PASSED

- FOUND: `notebooks/.cache/07_crossmodal_curvature.jsonl`
- FOUND: `notebooks/.cache/07_crossmodal_curvature_fields.npz`
- FOUND: `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-04-SUMMARY.md`
- FOUND commit `a453736` in `git log --oneline --all`
