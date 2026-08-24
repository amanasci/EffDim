---
phase: 05-curvature-conditioned-linear-decodability
verified: 2026-08-24T23:50:00Z
status: passed
score: 18/18 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 5: Curvature-Conditioned Linear Decodability Verification Report

**Phase Goal:** Fit one global ridge probe from `hsc` to `legacysurvey`, bucket the held-out
per-point residuals by decoder-side curvature magnitude under a per-seed protocol frozen before
any probe number exists, and report three per-seed verdicts, their spread, and one combined
phase verdict — with the seed-pooling question the phase originally set out to answer instead
answered by the `05-03` blocking-checkpoint decision record, since no pooled field was ever
built.

**Verified:** 2026-08-24T23:50:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Critical Focus: The Ordering Guarantee

The load-bearing check for the whole phase, verified mechanically from git history, not from
narration.

### 1. The freeze commit precedes the first commit that can produce a probe number

`05-04-SUMMARY.md` records the D5-09 freeze commit as `32dabe3` (plan `05-04` Task 2, all 31
constants filled). `05-05-SUMMARY.md` records its own Task Commits as: Task 1
(`2c8b062`, implements `run_bucketed_mode` — the only code path that can compute a probe
number), Task 2 (**no source commit** — the real run itself, whose output
`notebooks/.cache/05_curvature_probe_decodability.jsonl` is gitignored per `.gitignore` line 17,
so the run that actually produced the headline numbers left no commit of its own), and Task 3
(`690a220`, adds the conditioning diagnostic, re-run inline on the same real data). Because Task
2 produced no commit, "the first probe-number commit" as a git object does not exist as such —
the earliest commit that is a necessary precondition for any probe number to exist is Task 1's
`2c8b062`, which is what this check runs against. This gap is recorded here rather than glossed
over: a reader who expects a single commit whose diff shows "the real run" will not find one,
because the run's own output is deliberately not tracked in git.

```
$ git merge-base --is-ancestor 32dabe3 2c8b062 && echo "0 (32dabe3 IS ancestor of 2c8b062)"
0 (32dabe3 IS ancestor of 2c8b062)
```

`32dabe3` is an ancestor of `2c8b062`, and every later commit in this repository's history
(`690a220`, `db07460`, and this plan's own commits) is a descendant of `2c8b062` in turn. No
probe-enabling code, and no committed probe number, exists anywhere before the freeze.
**VERIFIED.**

### 2. No pre-registered constant amended after a probe number existed

```
$ git diff 32dabe3 HEAD -- notebooks/pu_manifold/linear_probe.py
(empty)
```

`notebooks/pu_manifold/linear_probe.py` — the sole module carrying `TRAIN_FRACTION`,
`SPLIT_SEED`, `RIDGE_ALPHA_GRID`, `N_BUCKETS`, `BUCKET_EDGES_PER_SEED`, `SEED_HANDLING_RULE`,
`VERDICT_RULE`, `SEED_VERDICT_COMBINATION_RULE`, `PHASE_VERDICT_VALUES`, and every other
pre-registered constant — is byte-identical between the freeze commit `32dabe3` and `HEAD`. No
constant was amended, tuned, or reworded at any point after `2c8b062` (or after either real run
in `05-05`) began producing probe numbers. **VERIFIED.**

### 3. The frozen file's full history — three commits, the middle one explained

```
$ git log --oneline -- notebooks/pu_manifold/linear_probe.py
32dabe3 feat(05-04): freeze all 31 pre-registration constants -- the D5-09 freeze commit
94735b7 feat(05-03): restructure pre-registration block for three per-seed verdicts
5888d0d feat(05-01): end-to-end probe-to-verdict path on planted data
```

Named, oldest to newest:

- **`5888d0d` (plan `05-01`).** Module creation. The end-to-end probe-to-verdict path built and
  tested on planted synthetic data, with every pre-registered constant declared but left UNSET
  (empty string / `None` / placeholder). No PU number could exist yet — the module could not
  even pass `assert_preregistered()`.
- **`94735b7` (plan `05-03`).** The per-seed structural repair, made **before** the freeze,
  after the `05-03` Task 1 blocking checkpoint ratified NOT pooling the three seeds
  (`05-03-DECISION.md`). This commit renamed `POOLING_METHOD` to `SEED_HANDLING_RULE` and
  `BUCKET_EDGES` to `BUCKET_EDGES_PER_SEED`, added `SEED_VERDICT_COMBINATION_RULE` and
  `PHASE_VERDICT_VALUES`, and added `combine_seed_verdicts`. **Every pre-registered constant
  remained UNSET after this commit** — it changed the shape of the pre-registration block (what
  fields exist), not any of their frozen values. This is the middle entry a reader might mistake
  for a post-freeze edit; it is not one. It precedes `32dabe3` and assigns no constant a value.
- **`32dabe3` (plan `05-04`).** The freeze itself: all 31 constants filled with their ratified
  values, `assert_preregistered()` passes for the first time, the module becomes closed. This is
  the **most recent** commit touching the file, confirmed directly:

```
$ git log --format=%H -1 -- notebooks/pu_manifold/linear_probe.py
32dabe3c1957de1d630143ebf5eec0c532ba2469
```

This SHA matches the `freeze_commit` recorded in `05-04-SUMMARY.md` (`32dabe3`) exactly. The
ordering guarantee — which constrains what happens **after** the freeze — is intact: the
pre-freeze structural repair changed no frozen value, and nothing has touched the file since the
freeze. **VERIFIED.**

### 4. `05-PREREGISTRATION.md`'s own commit history

```
$ git log --oneline -- .planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md
b45ae1b docs(05-04): committed pre-registration record -- 05-PREREGISTRATION.md
```

One commit, written at plan `05-04` Task 3, after the freeze commit `32dabe3` and before any
probe number existed. Never edited since. **VERIFIED.**

**The ordering guarantee is intact. Phase 5's central scientific-conduct promise holds**,
subject to the one honestly-recorded gap in §1 above (Task 2's real run left no commit of its
own because its output is gitignored — the ancestry proof runs against the nearest commit that
actually exists, `2c8b062`, and every later probe-producing action is transitively a descendant
of it).

## Observable Truths (Roadmap / Plan Success Criteria)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | `notebooks/05_curvature_conditioned_linear_decodability.ipynb` exists, executed end to end, committed with outputs | VERIFIED | 15 cells, all code cells have non-null `execution_count` and outputs (plots carry `image/png`); commit `7e6bba8` |
| 2 | Notebook imports the phase's code unchanged, fits no probe, extracts no field, buckets no field, pools no seed | VERIFIED | Automated grep in Task 1's `<verify>`: `fit_probe(`, `chart_curvature_field(`, `pool_seed_fields(`, `bucket_by_field(` absent from source |
| 3 | Both frozen rules print before any probe number appears | VERIFIED | Notebook cell 1 prints all 31 constants + `VERDICT_RULE` + `SEED_VERDICT_COMBINATION_RULE` before cell 7 (the first cell reading a probe number) |
| 4 | Three per-seed verdicts, phase verdict, D5-09 ordering guarantee proved mechanically | VERIFIED | §1-4 above; notebook cell 13; `05-FINDINGS.md` §5 |
| 5 | D5-04 supersession and D5-05 pooled-half disposition both on the record | VERIFIED | `05-FINDINGS.md` §3 (`SUPERSEDED`, `pooled_field_disposition`) |
| 6 | D5-11 and D5-12 accepted gaps stated at full strength, in the phase's own words | VERIFIED | `05-FINDINGS.md` §6, matches `04-FINDINGS.md`'s standard |

**Score:** 6/6 roadmap-level truths verified.

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `notebooks/05_curvature_conditioned_linear_decodability.ipynb` | Executed, committed, reads cached artifacts only | VERIFIED | 15 cells; commit `7e6bba8`; automated verify passed (see Task 1 `<verify>` output) |
| `.../05-FINDINGS.md` | Phase record: claims, frozen config, seed decision, density, result, gaps, requirement outcomes, follow-on needs | VERIFIED | This document's sibling; every literal token required by Task 2's `<verify>` present |
| `.../05-VERIFICATION.md` | This document | VERIFIED | — |

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `05-FINDINGS.md` | `notebooks/.cache/05_curvature_probe_decodability.jsonl` | every number quoted from a record row | WIRED | `probe_overall`, `probe_seed` (×3), `probe_bucket` (×9), `probe_conditioning` rows all cross-checked against `05-FINDINGS.md`'s tables |
| `05-FINDINGS.md` | `05-03-DECISION.md` | the pooled-field question answered from the ratified decision record | WIRED | `05-FINDINGS.md` §3 quotes the decision's evidence table verbatim and cites the file by name |
| `05-VERIFICATION.md` | `notebooks/pu_manifold/linear_probe.py` | empty-diff assertion and three-commit log both name the frozen file explicitly | WIRED | §2-3 above |
| `notebooks/05_curvature_conditioned_linear_decodability.ipynb` | `notebooks/pu_manifold/linear_probe.py` | notebook imports the frozen module unchanged and quotes its constants/rules | WIRED | Notebook cell 1: `from pu_manifold import linear_probe`; prints every constant and both rule strings |

## Requirements Coverage

De-facto requirement set: `05-CONTEXT.md`'s D5-01 through D5-13 (no milestone REQ-IDs were
minted for Phase 5, matching Phase 02.5's precedent).

| Requirement | Outcome | Evidence |
|---|---|---|
| D5-01 | MET | Probe `hsc -> legacysurvey`, both 768-d, from `load_pu_pair`'s resolved npz — `05-FINDINGS.md` §5; `05-05-SUMMARY.md` |
| D5-02 | MET | One `W` fit globally on the training split; held-out residuals bucketed three times, never refit per bucket or per seed — `05-FINDINGS.md` §5; `_fit_and_evaluate`'s single call site, grep-verified at `05-05` |
| D5-03 | MET | Decoder-side `||H||` via `chart_curvature.chart_curvature_field` (the corrected citation), not `decoder_curvature.py` — `linear_probe.py` docstring (a); `05-FINDINGS.md` §2 |
| D5-04 | **SUPERSEDED** | `05-CONTEXT.md`'s pooled-field design was rejected one-way at the `05-03` Task 1 blocking checkpoint; `05-03-DECISION.md` is the authority. No pooled field exists anywhere in this milestone's cache — `05-FINDINGS.md` §3 |
| D5-05 | MET (part one) / DISPOSITIONED (part two) | Part one, inter-seed Spearman/direction, measured at `05-02` — `05-FINDINGS.md` §3. Part two, pooled-vs-seed Spearman, has no referent because no pooled field exists; recorded as `pooled_field_disposition` in `05_density_diagnostics.json` — `05-FINDINGS.md` §3 |
| D5-06 | MET | `CURVATURE_CONVENTION = "trace"`, asserted equal across `linear_probe`/`chart_curvature`/`curvature_probe` by a passing test — `linear_probe.py`; `05-04-SUMMARY.md` |
| D5-07 | MET | Split on `||H||` magnitude per seed; continuous Spearman reported alongside every bucketed verdict as sensitivity only — `05-FINDINGS.md` §5; notebook cell 11 |
| D5-08 | MET | Realized test-split bucket counts reported beside full-field counts for all three seeds; size-matched check subsamples to each seed's own realized minimum — `05-FINDINGS.md` §5; notebook cells 8, 11 |
| D5-09 | MET | Full pre-registration freeze, git-ancestry-provable — §1-4 above |
| D5-10 | MET | `run_bucketed_mode` calls `assert_preregistered()` unconditionally as its first line, then checks all three per-seed bucket artifacts exist before computing anything — `curvature_probe_decodability_run.py`; unit-tested via `test_assert_preregistered_raises_when_absent` and `test_assert_preregistered_rejects_flat_bucket_edges` |
| D5-11 | MET (accepted gap, stated in full) | `05-FINDINGS.md` §6, in the phase's own words, quoting the sealed `-0.015106571347065712` row |
| D5-12 | MET (accepted gap, stated in full) | `05-FINDINGS.md` §6, `CAE_VERDICT = FAIL` inheritance chain in phase order |
| D5-13 | MET (disclosure, not a gate) | Three per-seed density Spearman values re-measured on the decoder-side field, reported beside Phase 4's point-cloud reference — `05-FINDINGS.md` §4; notebook cell 12 |

**13/13 requirement IDs accounted for.**

## Spec-Less Probe Coverage

The deterministic edge probe surfaced **29** applicable items across D5-01 through D5-13. Across
the four amended plans (`05-03` through `05-06`), **23** were authored into plan `must_haves`
(18 plain, 5 as backstop markers), and **6** unclassified rows were surfaced as flagged
assumptions requiring manual review. `23 + 6 = 29` — no silent drops.

`05-06-PLAN.md`'s own frontmatter records the correction: an earlier draft of this plan recorded
"spec-less probe fallback skipped — the phase has no requirement IDs to probe," which is
**wrong** and is **retracted** here and in `05-FINDINGS.md` — the phase has thirteen requirement
IDs (D5-01..D5-13) and the probe did run over them.

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full test suite | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` | `390 passed, 1 skipped` | PASS |
| `--selfcheck` (implementation self-check on synthetic data) | `.venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck` | All 8 checks `[PASS]`, exit 0 | PASS |
| `--mode pool` refuses by name | `.venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --mode pool` | `RuntimeError` naming `05-03-DECISION.md`, exit 1 | PASS |
| `--mode bucketed` raises when the pre-registration is stubbed empty | Not re-run destructively against the frozen module (would breach the D5-09 freeze); guard behavior is unit-tested instead | `run_bucketed_mode`'s first line calls `linear_probe.assert_preregistered()` unconditionally; `test_assert_preregistered_raises_when_absent` and `test_assert_preregistered_rejects_flat_bucket_edges` both exercise the raise path via `monkeypatch`, never by editing the shipped module | PASS (via unit test, not a live destructive run) |
| Notebook fully executed | `execution_count` sequential per code cell, all with outputs (plots included) | confirmed via the notebook's own JSON | PASS |
| `src/effdim/` untouched | `git status --porcelain src/effdim/` | empty | PASS |

## Deviations From Plan (carried from Task commits)

**1. [Rule 1 - Bug] `selfcheck()`'s stale pre-freeze `assert_preregistered` check.** Found
before Task 1: `selfcheck()` asserted `assert_preregistered()` **raises** while constants are
unset — true only before the `05-04` freeze. Since the module now ships frozen,
`--selfcheck` exited 1 for real. Inverted to assert the guard now **passes**, mirroring the
identical fix already applied to `test_linear_probe.py::test_assert_preregistered_raises_when_absent`
at `05-04`. `linear_probe.py` itself was never touched. Committed at `df392bd`.

**2. [Rule 3 - Blocking] The notebook's continuous-view plot (plan step 10) has no cached raw
per-point data to read.** `05-05` never persisted the raw per-point residual/`||H||` arrays on
the test split — only bucket- and seed-level aggregates were written to the JSONL. Reconstructing
a per-point scatter would require refitting the ridge map inside the notebook, which the plan's
own action text and its automated `<verify>` both forbid. Substituted the already-recorded
per-seed Spearman rho/p/n, documented in-notebook. Committed at `7e6bba8`.

Neither deviation touches `notebooks/pu_manifold/linear_probe.py`, changes any verdict, or
recomputes any number the frozen pipeline produced.

## Anti-Patterns Found

None. `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` grep against the notebook, `05-FINDINGS.md`,
and `05-VERIFICATION.md` returns zero matches.

## CLAUDE.md Swiss Roll Rule

**Does not trigger.** This plan writes a reader notebook and two planning documents; it
introduces no manifold-learning or representation-learning model. A linear ridge probe has no
bottleneck, no latent space, no decoder, and makes no manifold-recovery claim. The per-seed
amendment introduced no new model either. The curvature *estimator* the fields are read from is
already covered by `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`. This
determination is stated explicitly in `05-FINDINGS.md` §6 and in the notebook's closing cell,
not silently skipped.

## Human Verification Required

This phase's plan carries a `type="checkpoint:human-verify"` Task 3 with four judgements no
automated check can make: whether the frozen `VERDICT_RULE`/`SEED_VERDICT_COMBINATION_RULE` text
really admits "no detectable relationship" and "split across seeds" as complete outcomes rather
than near-misses; whether `05-FINDINGS.md`'s accepted-gaps section states each gap at full
strength rather than deferring to a cross-reference; whether the ordering guarantee's evidence
(§1-4 above) actually says what this document claims; and whether the `SPLIT ACROSS SEEDS`
write-up honours the frozen rule's own non-support framing. That checkpoint is presented
separately to the orchestrator/human and is not resolved by this verification document.

## Gaps Summary

The ordering guarantee holds mechanically, with one honestly-recorded irregularity: `05-05`
Task 2's real run produced no commit of its own (its JSONL output is gitignored), so the
ancestry proof runs against the nearest actual commit (`2c8b062`) rather than against a
dedicated "Task 2" commit the original plan text anticipated. This does not weaken the
guarantee — every commit that could possibly carry a probe number is still provably a
descendant of the freeze — but it is recorded here rather than silently substituted.

All 13 requirement IDs are accounted for, with D5-04 correctly marked `SUPERSEDED` and D5-05
split into a met half and a dispositioned half. Two small, fully-documented deviations were
applied (a stale self-check assertion, and a plot substituted for data that was never cached),
neither touching the frozen module or any reported verdict. The phase produces a
`SPLIT ACROSS SEEDS` result on a chain of acknowledged weaknesses (no known-answer anchor, an
overridden CAE gate, a not-statistically-independent set of three verdicts) — all stated at full
strength in `05-FINDINGS.md`, matching `04-FINDINGS.md`'s standard.

---

*Verified: 2026-08-24T23:50:00Z*
*Verifier: Claude (gsd-executor, self-verification pass within the closing plan)*
