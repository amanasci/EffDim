---
phase: 07-curvature-conditioned-crossmodal-alignment
verified: 2026-08-26T00:00:00Z
resolved: 2026-08-26T00:00:00Z
status: complete
score: 10/10 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Re-run notebook Cell 9 (positive-control read-out) with prose matching the amended 07-FINDINGS.md Sec 2 framing, then re-commit the notebook with updated outputs."
    expected: "Cell 9's printed text states the ≈0.0205 null-band edge and the unresolved 0.021-0.05 detection-floor interval, and stops saying the test 'has power to detect a planted relationship as small as 0.05' as if 0.05 were the floor."
    why_human: "This is an editorial correctness judgment about whether the notebook's prose now matches the findings document's corrected framing -- not a fact a script can check, since both wordings are syntactically valid and only one is accurate."
    resolution: "RESOLVED, commit 11242a7. Cells 9 AND 13 reworded (the review flagged Cell 9 only; Cell 13 carried the identical overclaim and was fixed alongside it) to derive the ≈0.0205 null-band edge, the 0.00047 margin, and the unresolved 0.021-0.05 interval directly from the frozen record's own 0.02 positive-control row, matching 07-FINDINGS.md Sec 2 exactly. Notebook re-executed end to end (11s); the Cell-5 guard still confirms the recomputed per-point MKNN reproduces the record's own d=20 rho (-0.1121807159) before plotting; verdict and every other cell's output are byte-identical to the pre-fix commit."
  - test: "Decide whether the three REVIEW.md Critical findings (CR-01/CR-02/CR-03, all runner-only guard-strength gaps around --freeze-commit verification, positive-control ancestry checking, and --flag=value argv parsing) need to be fixed now or tracked as follow-up debt before this phase is treated as fully closed."
    expected: "A recorded decision (fix now, defer with a tracked issue, or accept as-is) — the task brief states the actual production run used correct values and these gaps did not corrupt this phase's own numbers, but they remain live latent defects in code that will be reused by future runs against the same sealed record."
    why_human: "Whether latent guard-strength gaps in already-reviewed code block phase completion is a project-management judgment call the task brief explicitly reserves for a human ('already recorded, do not re-derive... but do assess whether they should block'); this verifier's role is to confirm the findings are real (done, independently reproduced below) and surface the decision, not make it."
    resolution: "DECIDED: fix now. RESOLVED, commit c92260f (fix) + 1570ec0 (07-REVIEW.md correction/status notes). CR-01: --freeze-commit must now resolve to exactly FREEZE_COMMIT_SHA, not merely be some strict ancestor, in addition to the existing strict-ancestor check. CR-02: --mode positive-control now calls the same _strict_ancestor_or_exit(FREEZE_COMMIT_SHA) gate before writing any row. CR-03: --threads/--smoke-rows/--max-epochs now accept both `--flag value` and `--flag=value` argv syntax via a new _flag_value_from_argv helper, used both in the pre-torch-import thread-cap block and in run_dsweep. 15 focused guard tests added (notebooks/pu_manifold/tests/test_crossmodal_curvature_run.py) -- the runner had zero coverage before (WR-04). The exact reproducibility invocation (--freeze-commit f032745... --threads 8 --resume) was independently re-verified to still pass the hardened gate without triggering a sweep. Full suite: 588 passed, 1 skipped (573 baseline + 15 new)."
---

# Phase 7: Curvature-Conditioned Crossmodal Alignment Verification Report

## Resolution Update (2026-08-26)

Both `human_needed` items above are resolved (see each item's `resolution:` field for the
specific commits). This supersedes Truth 9, the notebook artifact row, and the runner artifact
row's caveat below, all of which were accurate as of the original 2026-08-26 verification pass
and are left unedited as the historical record of what was found:

- **Truth 9 (notebook/findings disagreement, WR-03):** RESOLVED, commit `11242a7`. Both
  Cell 9 (the one the review flagged) and Cell 13 (which carried the identical overclaim and
  was found during the fix) now derive the ≈0.0205 null-band edge, the 0.00047 margin, and the
  unresolved 0.021-0.05 interval from the frozen record's own 0.02 positive-control row,
  matching `07-FINDINGS.md` Sec 2's corrected framing. Notebook re-executed end to end (11s);
  the Cell 5 guard still confirms the recomputed per-point MKNN reproduces the record's own
  d=20 rho (`-0.1121807159`) before plotting; the verdict and every other cell's output are
  byte-identical to the pre-fix commit (`git diff --stat` for the fix touches only cells 9 and
  13's source/outputs plus execution timestamps).
- **CR-01/CR-02/CR-03 (runner guard-strength gaps):** Decided fix-now. RESOLVED, commits
  `c92260f` (fix + 15 new focused tests) and `1570ec0` (07-REVIEW.md CR-03 severity-framing
  correction + Status: Fixed notes on all three). `--freeze-commit` now requires equality with
  `FREEZE_COMMIT_SHA` in addition to strict ancestry; `--mode positive-control` now runs the
  same gate before writing any row; `--threads`/`--smoke-rows`/`--max-epochs` now accept both
  `--flag value` and `--flag=value` argv forms. The exact reproducibility invocation
  (`--freeze-commit f032745f6450068c63763993d39fa112fd36bb8c --threads 8 --resume`) was
  independently re-verified (parsed via `build_arg_parser()` and passed through
  `_strict_ancestor_or_exit` directly, without triggering a sweep) to still succeed.

All hard constraints reconfirmed unchanged after the fix: `crossmodal_curvature.py` still
365 insertions / 0 deletions against the freeze commit; the seven sealed modules and
`src/effdim/` still show an empty diff; the frozen record and fields npz are untouched
(gitignored, `git status --short` empty); the verdict is still `ASSOCIATION DETECTED`
verbatim. Full suite: 588 passed, 1 skipped (573 baseline + 15 new guard tests), exit 0.

**Score is now 10/10** (Truth 9 flips to VERIFIED under the resolution above; nothing else in
the original 9/10 changed).

**Phase Goal:** Answer the milestone's actual research question — does the curvature of the PU
embedding manifold explain the weak crossmodal convergence reported by the Platonic Universe paper
(arXiv:2509.19453)? Measure `spearman(||H||_i, MKNN_i)` over all 10,000 points, using a curvature
field from a validated instrument, with a positive control establishing power and density/hubness
reported as diagnostics.

**Verified:** 2026-08-26
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The phase answers the research question, not just produces numbers | ✓ VERIFIED | `07-FINDINGS.md` opens with a one-paragraph answer stating verdict `ASSOCIATION DETECTED` verbatim, qualified by "this is not a clean result," the density-collapse caveat, and which `d` carries the signal. Answer + qualifications co-located, not buried. |
| 2 | D7-02's positive control is planted at PU's own realized dynamic range, not a synthetic surrogate or Phase 6's rejected `rng.random(n)` selfcheck | ✓ VERIFIED | `plant_positive_control`'s docstring states the requirement explicitly and names the rejected Phase 6 selfcheck. The npz's bare `h_norm` key used by `--mode positive-control` is byte-identical (`np.array_equal` confirmed) to `h_norm_20`, PU's own real d=20 field written by the sweep. |
| 3 | D7-03's "gating nothing" property is structural, not promised | ✓ VERIFIED | `inspect.signature(cc.apply_verdict)` independently confirmed: exactly two parameters (`per_d_results`, `positive_control_cleared_at`), neither naming density. Density cannot reach the verdict function even if a caller tried. |
| 4 | The verdict was applied mechanically from `VERDICT_RULE`, nothing adjusted after seeing the numbers | ✓ VERIFIED | The frozen record's own numbers were independently re-read (`notebooks/.cache/07_crossmodal_curvature.jsonl`) and match `07-04-SUMMARY.md`/`07-FINDINGS.md` to full float precision on every field checked (observed_rho, thresholds, density partials, positive-control achieved values). `apply_verdict({20:True,25:True,32:True}, 0.05)` independently traced through source returns the "all clear" branch = `ASSOCIATION DETECTED`, matching the recorded verdict row. |
| 5 | Every frozen constant is byte-identical to the freeze commit | ✓ VERIFIED | `git diff --stat f032745..HEAD -- crossmodal_curvature.py` = 365 insertions, 0 deletions (independently re-run). `git show --stat f032745` confirms the freeze commit itself is exactly one file, 368 insertions, no compute imports. |
| 6 | No sealed module or `src/effdim/` was touched | ✓ VERIFIED | `git diff --stat f032745..HEAD -- mknn.py linear_probe.py pointcloud_probe.py cae.py decoder_curvature.py curvature_probe.py cross_split_curvature.py src/effdim/` independently re-run: empty output. `git status --porcelain` on those same paths: clean. |
| 7 | The freeze commit is a strict ancestor of the run commit (not merely `--is-ancestor`, which a self-ancestor commit would also pass) | ✓ VERIFIED | Independently re-run: `git merge-base --is-ancestor f032745 HEAD` exits 0; `git rev-list --count f032745..HEAD` = 16 (>= 1). Record's own stamped SHAs (`f032745`/`a453736`) checked the same way in `07-FINDINGS.md` Sec 7: count = 10 between freeze and run commit. |
| 8 | The reporting notebook reads the frozen record and recomputes nothing | ✓ VERIFIED | 14 cells, 7 code cells, all carry non-empty outputs (independently re-checked). `grep -E "train_plain_ae|plain_decoder_curvature|permutation_null"` against the notebook returns nothing. |
| 9 | The notebook and findings, as committed reporting artifacts, state a consistent claim about the positive control's detection power | ✗ FAILED | `07-FINDINGS.md` Sec 2 was human-amended (commit `df8502f`) to correct the claim "recovers a planted effect as small as 0.05" — the actual null-band edge is ≈0.0205 and the true floor is unresolved in 0.021-0.05. `git diff --stat` for that amendment touches only `07-FINDINGS.md`. The notebook's Cell 9 (independently inspected) still prints the pre-amendment claim verbatim: "The permutation test has power to detect a planted relationship as small as 0.05." Two committed, human-facing artifacts for the same phase now disagree on a specific number a reader would check. |
| 10 | Test suite is green, no debt markers in touched files | ✓ VERIFIED | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` independently re-run: 573 passed, 1 skipped, exit 0. `grep -E "TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER"` across `crossmodal_curvature.py`, the runner, the test file, and `07-FINDINGS.md`: no matches. |

**Score:** 9/10 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `notebooks/pu_manifold/crossmodal_curvature.py` | Frozen constants + compute functions | ✓ VERIFIED | 365 lines added since freeze, 0 removed; 157 tests exercise every added compute function |
| `notebooks/diagnostics/07_crossmodal_curvature_run.py` | Runner: smoke/dsweep/positive-control modes | ✓ VERIFIED (with caveats) | Runs, wired to `crossmodal_curvature`. Contains the three guard-strength gaps documented in `07-REVIEW.md` (CR-01/CR-02/CR-03), independently reproduced below — see Anti-Patterns. |
| `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` | 157 tests | ✓ VERIFIED | 157 passed in ~30s, independently re-run |
| `notebooks/07_crossmodal_curvature_check.ipynb` | Reporting notebook, executed with outputs | ⚠️ ORPHANED CLAIM | Executed with outputs (confirmed), but carries a stale claim not updated after the sibling findings document's human-approved correction (Truth 9 above) |
| `07-FINDINGS.md` | The answer + limitations | ✓ VERIFIED | States verdict, power evidence, instrument range, limitations, non-claims, provenance; amended in place per human checkpoint |
| `notebooks/.cache/07_crossmodal_curvature.jsonl` | Frozen record, 8 rows | ✓ VERIFIED | Confirmed present locally (gitignored per CLAUDE.md, as expected — not a gap): 3 sweep + 4 positive-control + 1 verdict row, every number matching the SUMMARY/FINDINGS documents exactly |
| `notebooks/.cache/07_crossmodal_curvature_fields.npz` | Per-d ||H||/cond(g) arrays | ✓ VERIFIED | Confirmed present; `h_norm` bare key byte-identical to `h_norm_20` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `07_crossmodal_curvature_run.py --mode dsweep` | `crossmodal_curvature.assert_preregistered()` | called first in every mode | ✓ WIRED | Confirmed at source: `cc.assert_preregistered()` runs before ancestry check and before data load |
| `--mode dsweep` | strict-ancestor proof | `_strict_ancestor_or_exit(args.freeze_commit)` | ⚠️ PARTIAL (guard-strength gap) | Runs, but validates the supplied SHA is *some* ancestor, never checks it equals `FREEZE_COMMIT_SHA` (CR-01, independently reproduced — see Anti-Patterns). The actual production run used the correct SHA (confirmed: record's `preregistration_commit` = `f032745...` exactly), so this run's output is not affected, but the guard itself is bypassable. |
| `--mode positive-control` | strict-ancestor proof | (none) | ✗ NOT_WIRED (guard-strength gap) | Independently confirmed: `run_positive_control` calls only `cc.assert_preregistered()`; no call to `_strict_ancestor_or_exit` exists anywhere in its body (CR-02). |
| `plant_positive_control` | PU's own d=20 field | `h_norm` npz key | ✓ WIRED | Confirmed byte-identical to `h_norm_20` |
| `density_diagnostics` output | `apply_verdict` | (must be impossible) | ✓ VERIFIED NON-LINK | `apply_verdict`'s 2-parameter signature independently confirmed to exclude density |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full notebook test suite passes | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` | 573 passed, 1 skipped, exit 0 | ✓ PASS |
| Focused crossmodal_curvature test file | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py -q` | 157 passed in ~30s | ✓ PASS |
| Frozen constants unchanged | `git diff --stat f032745..HEAD -- crossmodal_curvature.py` | 365 insertions(+), 0 deletions(-) | ✓ PASS |
| Sealed modules untouched | `git diff --stat f032745..HEAD -- <7 sealed files> src/effdim/` | empty | ✓ PASS |
| Strict-ancestor proof (independently re-derived) | `git merge-base --is-ancestor f032745 HEAD && git rev-list --count f032745..HEAD` | exit 0, count=16 | ✓ PASS |
| `apply_verdict` signature is 2-parameter | `inspect.signature(cc.apply_verdict)` | `(per_d_results: Dict[int, bool], positive_control_cleared_at: Any) -> str` | ✓ PASS |
| `h_norm` == `h_norm_20` in fields npz | `np.array_equal(z['h_norm'], z['h_norm_20'])` | True | ✓ PASS |
| Record numbers match report numbers | manual field-by-field diff of jsonl vs. FINDINGS/SUMMARY tables | exact match (observed_rho, thresholds, partials, positive-control achieved values) | ✓ PASS |
| `--freeze-commit` bypass (CR-01) reproduced | source read of `run_dsweep`/`_strict_ancestor_or_exit` | confirmed: `args.freeze_commit` never compared to `FREEZE_COMMIT_SHA` | ✓ REPRODUCED (defect confirmed real) |
| Positive-control missing ancestry check (CR-02) reproduced | source read of `run_positive_control` body | confirmed: no `_strict_ancestor_or_exit` call present | ✓ REPRODUCED (defect confirmed real) |
| Notebook stale-claim (WR-03) reproduced | grep + cell inspection of `07_crossmodal_curvature_check.ipynb` | Cell 9 states pre-amendment "as small as 0.05" framing | ✓ REPRODUCED (defect confirmed real) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `notebooks/diagnostics/07_crossmodal_curvature_run.py` | 443, 455 | `--freeze-commit` validated only as *an* ancestor, never compared against the hardcoded `FREEZE_COMMIT_SHA` | Warning (already recorded in `07-REVIEW.md` CR-01; production run's actual value was correct, independently confirmed) | A future wrong-but-plausible SHA would pass silently and mis-stamp `preregistration_commit` on new sweep rows |
| `notebooks/diagnostics/07_crossmodal_curvature_run.py` | 305-343 | `run_positive_control` performs no strict-ancestor check at all before writing to the sealed record | Warning (already recorded, CR-02; production run's stamped SHA independently confirmed correct) | Asymmetric with `run_dsweep`'s guard; a future run from a detached/stale commit could append inconsistent-provenance rows |
| `notebooks/diagnostics/07_crossmodal_curvature_run.py` | 19-27, 445-453, 472-473 | `--flag=value` argv syntax silently bypasses the thread-cap and the reduced-scale-run safety guard (raw `sys.argv` token scan, not `args.<flag>`) | Warning (already recorded, CR-03; the actual run used space-separated flags, independently confirmed via the record's stamped `threads: 8` and full-scale row counts) | A future `--max-epochs=N` (no space) invocation could silently write full-scale rows over the sealed record with no error |
| `notebooks/07_crossmodal_curvature_check.ipynb` | Cell 9 | Notebook still states the pre-amendment "detects as small as 0.05" framing that `07-FINDINGS.md` Sec 2 was corrected to walk back | **Blocker for a clean pass** (not previously flagged as blocking; `07-REVIEW.md` records it as WR-03/warning-level, but per this phase's own CLAUDE.md requirement that notebooks be "executed end to end" and committed as a primary reporting artifact, an uncorrected overclaim in a committed output is a completion gap, not merely a style note) | A reader who consults only the notebook (rather than `07-FINDINGS.md`) receives a materially overstated detection-power claim for the exact number the human checkpoint singled out as needing correction |

The runner-level guard-strength gaps (CR-01/CR-02/CR-03) were pre-recorded in `07-REVIEW.md` and independently reproduced here; per the task brief, they concern future-run robustness, not this phase's own already-produced result — the actual production run's stamped provenance (`f032745`.../`a453736`..., `threads: 8`, correct-scale row counts) was independently verified correct. They are carried forward as a WARNING requiring a human decision on whether to fix now or track as follow-up debt, not as a BLOCKER on this phase's scientific conclusion.

### Requirements Coverage

Phase-scoped decision IDs D7-01 through D7-07 (from `07-CONTEXT.md` §3), not `REQUIREMENTS.md`
entries — a `not_found` against `REQUIREMENTS.md` is expected, confirmed (`grep -n "D7-0" .planning/REQUIREMENTS.md` → no matches, and no "Phase 7" heading exists there either — this is not an ORPHANED requirement, the CONTEXT.md decision scheme is the intended source per the task brief).

| Decision | Description | Status | Evidence |
|---|---|---|---|
| D7-01 | Curvature field, 3-d sweep | ✓ SATISFIED | Sweep rows for d=20,25,32 in record, matching FINDINGS |
| D7-02 | Positive control at PU's own dynamic range | ✓ SATISFIED | `h_norm`/`h_norm_20` identity confirmed; docstring rejects Phase 6 selfcheck |
| D7-03 | Density/hubness, reported and gating nothing | ✓ SATISFIED | `apply_verdict` signature structurally excludes density (independently confirmed) |
| D7-04 | Per-point, not per-region MKNN | ✓ SATISFIED | 15 distinct values at k=20 across 10,000 points, matches record |
| D7-05 | Additive only | ✓ SATISFIED | Zero diff on all 7 sealed modules + `src/effdim/` |
| D7-06 | Freeze before compute | ✓ SATISFIED | Freeze commit is constants-only (368 insertions, no compute imports); strict-ancestor proof independently re-verified |
| D7-07 | CKA out of scope | ✓ SATISFIED | `ALIGNMENT_METRIC="mknn"` frozen constant on every record row; no CKA implementation found in repo |

### Gaps Summary

The phase's scientific deliverable is sound and independently reproduced end to end: the freeze
proof holds, the sealed modules are untouched, the positive control is genuinely planted on PU's
own dynamic range, `apply_verdict` is structurally non-gating on density, every number in
`07-FINDINGS.md` traces exactly to the frozen record, and the record itself was not rewritten
post-hoc (the sweep rows' full field-by-field content matches the SUMMARY to full float
precision). `07-FINDINGS.md` states an actual answer (`ASSOCIATION DETECTED`) with its
qualifications intact, not a bare number.

One gap needs a human decision before this phase is a clean pass: the committed reporting
notebook (`notebooks/07_crossmodal_curvature_check.ipynb`) was not updated when
`07-FINDINGS.md` was human-amended to correct an overstated detection-power claim, so the two
committed, human-facing artifacts for this phase currently disagree on a specific number
(0.05-as-floor vs. the corrected ≈0.0205 null-band-edge / 0.021-0.05 unresolved-interval
framing). This is a small, mechanical fix (re-run one cell's prose, re-commit with outputs) but
it was not verified as done — the task brief's own artifact list and `07-REVIEW.md`'s WR-03
finding both point at this specific unresolved state.

Separately, `07-REVIEW.md`'s three Critical findings (CR-01/CR-02/CR-03) were independently
reproduced by direct source reading and are real, live gaps in the runner's guard code. They did
not corrupt this phase's actual result — the production run's stamped provenance is independently
confirmed correct — but they remain unfixed. Whether to fix them now or carry them as tracked
follow-up debt is a human call the task brief explicitly reserves rather than delegates to this
verifier.

**Both gaps above are now resolved** — see "Resolution Update (2026-08-26)" near the top of
this document for the specific commits, the re-verified hard constraints, and the full-suite
result. This phase is a clean 10/10 pass.
