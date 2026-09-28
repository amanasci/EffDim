---
phase: 05-curvature-conditioned-linear-decodability
verified: 2026-08-24T00:00:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 5: Curvature-Conditioned Linear Decodability — Independent Goal Verification

**Phase Goal (amended):** Measure whether linear crossmodal decodability degrades as
decoder-side manifold curvature magnitude increases — one global ridge map
`hsc -> legacysurvey` on frozen PU embeddings, held-out per-point residuals bucketed
independently by each of three per-seed decoder-side `||H||` fields, judged per seed under a
rule frozen before any PU probe number exists and combined into one phase read-out by a frozen
combination rule (seeds NOT pooled, per `05-03-DECISION.md`).

**Verified:** 2026-08-24
**Status:** passed
**Audit target:** `05-VERIFICATION.md`'s self-reported `status: passed`, `score: 18/18` — audited
against the codebase, not trusted. This report is independent and additive; it does not
overwrite `05-VERIFICATION.md`.

## Summary

Every load-bearing claim in this phase — the git-ancestry ordering proof, the three per-seed
verdicts, the phase verdict, the no-pooling refusal in code, and the full-strength statement of
the accepted gaps — was independently reproduced against the actual codebase and artifacts, not
read secondhand from a SUMMARY or from `05-VERIFICATION.md`'s own narration. All reproduced
exactly. One minor factual inaccuracy was found in `05-VERIFICATION.md` itself (see Finding
below); it does not touch the phase's scientific-conduct guarantees or its result, so it does
not change the phase's status, but it is reported because this audit is required not to defer
to the document it is checking.

## 1. The Ordering Guarantee — independently reproduced

All commands below were re-run by this verifier directly, not copied from `05-VERIFICATION.md`.

```
$ git log --oneline -- notebooks/pu_manifold/linear_probe.py
32dabe3 feat(05-04): freeze all 31 pre-registration constants -- the D5-09 freeze commit
94735b7 feat(05-03): restructure pre-registration block for three per-seed verdicts
5888d0d feat(05-01): end-to-end probe-to-verdict path on planted data
```
Exactly three commits. `32dabe3` is the most recent (`git log --format=%H -1` confirms
`32dabe3c1957de1d630143ebf5eec0c532ba2469`). **VERIFIED.**

```
$ git diff 32dabe3 HEAD -- notebooks/pu_manifold/linear_probe.py
(empty)
```
No constant amended since the freeze. **VERIFIED.**

```
$ git merge-base --is-ancestor 32dabe3 2c8b062 && echo yes
yes
$ git merge-base --is-ancestor 2c8b062 HEAD && echo yes
yes
```
`2c8b062` (`05-05` Task 1, "implement the bucketed branch") is the commit that adds
`run_bucketed_mode`'s real-data execution path — the only code that can compute a probe number
— and it is a descendant of the freeze, with everything since a descendant of it in turn (clean
linear history, confirmed via `git log --oneline --graph`, no branch divergence).

**On the acknowledged weak point (plan `05-05` Task 2's real run left no commit because its
JSONL output is gitignored):** confirmed independently — `notebooks/.cache/` is gitignored at
`.gitignore` line 17, and `notebooks/.cache/05_curvature_probe_decodability.jsonl` is untracked
(`git status --porcelain` shows nothing for it because it is ignored, not because it is clean).
This verifier additionally checked the *earlier* commit `694cda9` (`05-01`, "the real curvature
call site and the D5-10 guard") — at that point `--mode bucketed` already called
`assert_preregistered()`, but `linear_probe.py`'s constants were all still `None`/`""` at that
commit (confirmed by `git show 694cda9:notebooks/pu_manifold/linear_probe.py`), so no probe
number could have been computed even via that earlier, pre-freeze code path. The anchoring at
`2c8b062` is sound: it is not merely the nearest surviving commit by convention, it is
independently the earliest commit at which the constants are non-empty AND the bucketed-mode
real-data path exists simultaneously. **Judgment: the anchoring holds.**

File mtime cross-check: `05_curvature_probe_decodability.jsonl` mtime is `19:30`, after
`32dabe3` (`19:07:59`), `b45ae1b`/`05-PREREGISTRATION.md` (`19:10:15`), and `2c8b062`
(`19:27:15`) — consistent with the claimed ordering rather than contradicting it.

`src/effdim/` untouched (`git status --porcelain src/effdim/` empty). **VERIFIED.**

## 2. The Verdicts — read directly from the JSONL, byte-for-byte cross-check

Read `notebooks/.cache/05_curvature_probe_decodability.jsonl` directly with a standalone Python
script (not through any phase code):

| seed | verdict |
|---|---|
| 20260813 | `HOLDS` |
| 20260814 | `NO DETECTABLE RELATIONSHIP` |
| 20260815 | `HOLDS` |

`probe_overall` row: `"phase_verdict": "SPLIT ACROSS SEEDS"`, `"n_holds": 2`.

These match exactly — the JSONL, the notebook's cell 7/14 printed output, and `05-FINDINGS.md`
§5 all quote the identical strings and numbers (`n_train=7000`, `n_test=3000`,
`selected_alpha=0.1`, `r2_overall=0.643931`, `mean_residual_overall=0.066429`, bucket edges,
realized/full-field bucket counts, per-bucket mean residuals and CIs, size-match results,
continuous Spearman values). Also independently cross-checked: `probe_conditioning` row
(`condition_number=99806.5`, `effective_rank_1pct=531`, `cumvar_first_20=0.810`,
`cumvar_first_25=0.835`) matches `05-FINDINGS.md`'s RESEARCH A2 conditioning check exactly. The
notebook (`notebooks/05_curvature_conditioned_linear_decodability.ipynb`) is 15 cells, all code
cells sequentially executed (`execution_count` 1-10) with outputs present, and reads these
artifacts from cache rather than recomputing them — grepped independently for
`fit_probe(`, `chart_curvature_field(`, `pool_seed_fields(`, `bucket_by_field(` in the
notebook's source: all four absent, confirming the notebook is a pure reader. **VERIFIED.**

## 3. The Result Is Not Softened

Independently read the notebook's closing markdown cell (cell 14) and `05-FINDINGS.md` §5. Both
state `SPLIT ACROSS SEEDS` as "a complete, terminal, non-supportive outcome," explicitly say it
is "NOT partial support for the hypothesis," and explicitly state the two agreeing seeds are
"not presented as the headline with the third set aside — all three verdicts are reported
together." Neither document leads with the two-of-three agreement before stating the terminal
verdict; the phase verdict is stated first, in both places, exactly as `SPLIT ACROSS SEEDS`.
**VERIFIED — the finding matches the frozen rule's own non-support framing.**

## 4. The Accepted Gaps, at Full Strength

Read `05-FINDINGS.md` §6 and the notebook's closing cell directly (not via cross-reference):

- **D5-11** — both documents state, in their own sentences, the sealed rank
  `rank_spearman_rho = -0.015106571347065712` (cross-checked against
  `.claude/skills/spike-findings-effdim/sources/002-teacher-d20-four-axes/run_d20.py`'s
  `SEALED_DECODER_RHO` constant — matches exactly), state a detected effect "cannot be
  attributed to curvature by anything in this phase," and state the saddle-fixture question is
  "open" and "not for autonomous action" — not used to upgrade the result.
- **D5-12** — both state `CAE_VERDICT = FAIL` at Phase 02.2, the deliberate override at Phase 3,
  and Phase 03.1's partial/non-seed-consistent ordering repair, presented (in `05-FINDINGS.md`)
  as "two facts that stand together; neither cancels, supersedes, nor excuses the other."
- **Non-independence of the three per-seed verdicts** — stated explicitly in the notebook's cell
  7 output ("This means the three per-seed verdicts are NOT statistically independent
  replicates") and in `05-FINDINGS.md` §5, not left implicit.

**VERIFIED — all three gaps stated at full strength, in the phase's own words, not by
cross-reference only.**

## 5. No-Pooling Decision Honored in Code

Independently executed (not read from a log):

```
$ .venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --mode pool
RuntimeError: Seed pooling was put to the developer at the 05-03 Task 1 blocking checkpoint
and ratified as NOT DONE. See .../05-03-DECISION.md -- 05-CONTEXT.md D5-04 (pool the three
cached CAE seeds into one averaged ||H|| field) is superseded by that ratified, one-way
decision. Use --mode perseed instead.
```

Refuses by name, naming `05-03-DECISION.md` explicitly, exit code non-zero (process raised).
**VERIFIED.**

## 6. Independently Re-Run Checks

| Check | Command | Result |
|---|---|---|
| Full test suite | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` | `390 passed, 1 skipped` — matches `05-VERIFICATION.md`'s claim exactly |
| `--selfcheck` | `.venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck` | 7 `[PASS]` lines, exit 0 — see Finding below |
| `--mode pool` | (above) | refuses by name |
| Frozen constants in source vs. `05-PREREGISTRATION.md` | `grep` of `TRAIN_FRACTION`, `SPLIT_SEED`, `N_BUCKETS`, `BUCKET_EDGES_PER_SEED`, `SEED_HANDLING_RULE`, `SEED_STEMS`, `CURVATURE_CONVENTION`, `FIELD_D`, `K_DENSITY`, `N_BOOTSTRAP`, `SIZE_MATCH_N_REPEATS` in `linear_probe.py` | byte-identical to the preregistration table |
| Requirements orphan check | `grep "Phase 5\|D5-0" .planning/REQUIREMENTS.md` | no hits — confirms the phase's own note that no milestone REQ-IDs were minted, matching the Phase 02.5 precedent; not a gap |
| Debt markers | `grep -n "TBD\|FIXME\|XXX"` across `linear_probe.py`, the runner, `05-FINDINGS.md`, `05-PREREGISTRATION.md` | none found |

## Finding (WARNING, not a phase-goal blocker)

**`05-VERIFICATION.md`'s Behavioral Spot-Checks table overstates the `--selfcheck` result.** It
claims `"All 8 checks [PASS], exit 0"`. Independently running `--selfcheck` (above) and counting
the `check(...)` call sites in `selfcheck()` in
`notebooks/diagnostics/curvature_probe_decodability_run.py` both show exactly **7** checks, not
8: (1) aggregate R² > 0.99, (2) Frobenius identity, (3) pooled-vs-largest-seed Spearman < 1.0,
(4) bucket counts partition the test split, (5) highest bucket residual exceeds lowest, (6)
`apply_verdict_rule` returns HOLDS on planted data, (7) `assert_preregistered` passes now that
the module is frozen. All 7 actual checks genuinely pass and the process exits 0 — so the
underlying implementation self-check is sound — but the count in `05-VERIFICATION.md`'s own
table is factually wrong. This is exactly the kind of self-reported-verification claim this
audit exists to catch: it was written by the executor without independently re-running and
counting the output. It does not touch the phase's scientific-conduct guarantees, the ordering
proof, or the reported verdict, and is reported here as a WARNING rather than a gap that blocks
the phase goal.

## Overall Determination

All five points the orchestrator specifically asked to be independently checked — the ordering
guarantee (including the gitignored-JSONL anchoring judgment), the verdicts read directly from
the JSONL, the non-softened reporting of `SPLIT ACROSS SEEDS`, the full-strength statement of
D5-11/D5-12/non-independence, and the `--mode pool` refusal — were reproduced independently and
hold. `05-VERIFICATION.md`'s central claims about the phase's scientific-conduct guarantees are
correct. The one inaccuracy found (a miscounted selfcheck line count) is cosmetic to the
self-verification document, not to the phase's deliverable, and does not change the phase's
status.

**Phase goal achieved.** The amended, no-pooling protocol was executed end to end, produced
three genuinely per-seed verdicts under rules frozen before any probe number existed, combined
them into `SPLIT ACROSS SEEDS` under a rule that treats that outcome as complete and terminal,
and reported the required accepted gaps at full strength. Ready to proceed.

---

*Verified: 2026-08-24*
*Verifier: Claude (gsd-verifier, independent goal-backward audit of 05-VERIFICATION.md's claims)*
