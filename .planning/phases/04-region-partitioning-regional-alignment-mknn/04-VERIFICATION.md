---
phase: 04-region-partitioning-regional-alignment-mknn
verified: 2026-08-24T13:49:41Z
status: passed
score: 14/14 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 4: Region Partitioning & Regional Alignment (MKNN) Verification Report

**Phase Goal:** With all upstream hyperparameters frozen from Phases 1-3, points are
pre-specified into density-checked high/low curvature regions and crossmodal MKNN alignment
compared between them against region-specific permutation nulls and bootstrap CIs, under a
verdict rule pre-registered before any regional number exists (the ROADMAP's
garden-of-forking-paths Ordering constraint).

**Verified:** 2026-08-24T13:49:41Z
**Status:** passed
**Re-verification:** No — initial verification

## Critical Focus: The Ordering Guarantee

This is the load-bearing check for the whole phase. Verified mechanically from git history,
not from SUMMARY claims.

### 1. Pre-registration commit precedes every regional-number commit

```
$ git merge-base --is-ancestor 0305c77 647d01d && echo "0305c77 IS ancestor of 647d01d"
0305c77 IS ancestor of 647d01d
```

`git log --all --oneline -- notebooks/pu_manifold/region_partition.py` returns exactly two
commits: `e1106b4` (the sign-split helper, zero pre-registered constants, zero PU region
labels) and `0305c77` (the pre-registration constants block, `04-PREREGISTRATION.md`, and the
runner's `assert_preregistered()` guard). The first commit computing any regional cell is
`647d01d` (`04-05`'s eight-cell grid), which is a descendant of `0305c77`. No regional number
existed in the repo before the freeze commit. **VERIFIED.**

### 2. No pre-registered constant amended after a regional number existed

```
$ git log --all --follow --oneline --reverse -- notebooks/pu_manifold/region_partition.py
e1106b4 feat(04-03): diametrical sign-split partition helper with known-answer test
0305c77 docs(04-03): freeze phase-4 pre-registration before any regional MKNN number exists
$ git diff 0305c77 HEAD -- notebooks/pu_manifold/region_partition.py
(empty)
```

`region_partition.py` — the sole module carrying `MIN_NORM_PERCENTILE`, `MIN_REGION_N`,
`MKNN_K_GRID`, `HEADLINE_K`, `NULL_QUANTILE`, `CONFIDENCE_LEVEL`, `N_PERMUTATIONS`,
`N_BOOTSTRAP`, `K_FROZEN`, and `VERDICT_RULE` — was never touched by any commit after `0305c77`
(the freeze commit itself). The working tree at HEAD is byte-identical to the tree at `0305c77`
for this file. No constant was amended, tuned, or reworded at any point after regional numbers
began appearing in `647d01d` and later. **VERIFIED.**

### 3. Verdict applied mechanically from the committed rule

`VERDICT_RULE` (read from `region_partition.py` at HEAD, identical to `0305c77`): HOLDS iff (a)
the two regions' 95% bootstrap CIs at that `k` are disjoint, AND (b) the higher-scoring region's
score strictly exceeds its own 99th-percentile region-scoped permutation-null threshold.

Checked directly against all eight rows of `notebooks/.cache/04_region_partition_mknn.jsonl`:

| k | region 0 CI | region 1 CI | disjoint? | region 1 score vs own null_threshold | clears? |
|---|---|---|---|---|---|
| 5 | [0.03942, 0.04393] | [0.09383, 0.10382] | yes | 0.09889 > 0.002713 | yes |
| 10 | [0.05655, 0.06052] | [0.12762, 0.13647] | yes | 0.13212 > 0.004470 | yes |
| 20 | [0.07946, 0.08346] | [0.16977, 0.17842] | yes | 0.17408 > 0.008323 | yes |
| 50 | [0.12282, 0.12697] | [0.23817, 0.24729] | yes | 0.24258 > 0.019760 | yes |

Both conditions hold at every `k` including the pre-registered `HEADLINE_K=20`. HOLDS is the
correct mechanical output of the committed rule applied to the committed data — not a restated
or reinterpreted rule. **VERIFIED.**

### 4. The three accepted gaps and the region-size artifact, stated at full strength in 04-FINDINGS.md

Confirmed present in the document's own words (not by reference to a decision ID):
- Density confound: `spearman(density, ||H||) = -0.0273` (n=9500, p=0.0078) vs
  `spearman(density, signed_projection) = +0.8208` (n=9500, p≈0); region medians `3.7642e10`
  vs `6.5641e6` (~5,735x); "no partial regression, no density-matched null... run."
- No known-answer anchor: codimension-1 fixtures (`saddle`, `bowl`, `aniso`, `cubic`, `sine`,
  `ridge`) explicitly named as measuring surface-normal orientation, not direction resolution
  at PU's codimension ~748 (`d~20` inside `D=768`); `make_ridge_graph_control` /
  `make_multinormal_ridge_control` explicitly stated as not run (D4-10).
- `K_FROZEN=500`: table reproduced verbatim (`median_R_H` 0.0279→0.3436, `rule_fired=false`,
  the 0.0516→0.0583 rise at the last step called out explicitly); "never described as
  converged, plateaued, or settled anywhere in this phase's record."
- Region-size artifact: raw-score gaps 137.6%/125.9%/113.6%/94.3% at k=5/10/20/50 vs
  ratio-over-chance gaps 23.2%/17.2%/10.6%/0.6% — independently recomputed from the JSONL in
  this verification and matched to within rounding.

**VERIFIED.**

**The ordering guarantee is intact. Phase 4's central scientific-conduct promise holds.**

## Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Local sample-density measure per point in the ambient 768-d embedding space, correlation with curvature shown before the split is trusted (REGN-01, REGN-02) | VERIFIED | `04-FINDINGS.md` §4; `04_density_diagnostics.json`; both Spearman correlations present with n and p |
| 2 | Points partitioned by a data-derived direction criterion (diametrical sign split), pre-specified and frozen before any regional MKNN number (REGN-03..05) | VERIFIED | `region_partition.py`; `0305c77` precedes `647d01d`; ARI=1.0 known-answer test passes (7/7 in `test_region_partition.py`) |
| 3 | MKNN score matches the origin paper's formula; global crossmodal number reproduced and compared against the paper's published range (MKNN-01, MKNN-02) | VERIFIED | `mknn.py`'s `mknn_score` = `k^-1 * mean(|N_k(z1) ∩ N_k(z2)|)`; global JSONL rows present at every k with paper-range comparison stated with the n mismatch disclosed |
| 4 | Per-region MKNN score for both regions, each with its own permutation null (never global) and bootstrap CI (MKNN-03..05) | VERIFIED | JSONL: 8/8 regional rows have `null_scope: "region"`, `null_n == n_region`, `ci_low`/`ci_high` populated |
| 5 | High-vs-low result across k=5,10,20,50 shown with explicit verdict, hubness caveat stated alongside (MKNN-06..08) | VERIFIED | Eight-cell table in `04-FINDINGS.md` §5; `VERDICT_RULE` mechanically applied (see Critical Focus §3); hubness skewness 0.966–1.494 printed per cell |

**Score:** 5/5 roadmap success criteria verified, 0 present-but-behavior-unverified.

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `notebooks/pu_manifold/mknn.py` | MKNN score, region-scoped null, bootstrap CI, hubness | VERIFIED | 180 lines; `def mknn_score(`, `permutation_null`, `bootstrap_ci`, `hubness_skewness`, `chance_floor` all present and substantive |
| `notebooks/pu_manifold/region_partition.py` | Diametrical sign split, pre-registered constants, `VERDICT_RULE` | VERIFIED | 91 lines added at `0305c77`; `assert_preregistered()` guard present; never modified after freeze |
| `notebooks/diagnostics/region_partition_mknn_run.py` | Two-column PU loader, global + regional MKNN passes | VERIFIED | 941 lines; `--mode regional` calls `assert_preregistered()` before computing |
| `notebooks/pu_manifold/tests/test_region_partition.py` | Known-answer round-trip and boundary tests | VERIFIED | 157 lines, 7/7 tests pass on direct run (`python -m pytest ... -q` → `7 passed in 1.13s`) |
| `.planning/.../04-PREREGISTRATION.md` | Committed, timestamped pre-registration record | VERIFIED | Committed at `0305c77`; every constant matches `region_partition.py` verbatim |
| `notebooks/04_region_partition_mknn.ipynb` | Executed notebook, all sections | VERIFIED | 25/25 code cells have output, execution counts sequential (1..25); key content strings (`0.990`, `0.469`, `codimension`, `748`, `no known-answer anchor`, `swiss roll`, `2509.19453`, `radovanovic`, `dhillon`, `density-matched`, `ratio-over-chance`, `5,735`) all present |
| `.../04-FINDINGS.md` | Phase record: results, three accepted gaps, region-size artifact | VERIFIED | See Critical Focus §4 |
| `.../COVERAGE.md` | No-external-API declaration | VERIFIED | One-line reason present, accepted at seal-time gate |

## Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `region_partition_mknn_run.py` | `mknn.py` | `from pu_manifold import mknn` | WIRED | Confirmed by JSONL output existing with mknn fields |
| `region_partition_mknn_run.py` | `region_partition.py` | frozen constants + `assert_preregistered()` guard | WIRED | Regional grid could not exist without this guard passing |
| `04-PREREGISTRATION.md` | `region_partition.py` | verbatim constant values | WIRED | All 13 constants cross-checked verbatim between the two files |
| `04-FINDINGS.md` | `04_region_partition_mknn.jsonl` | quoted numbers traceable to cache | WIRED | All eight regional rows and all four global rows independently recomputed and matched to FINDINGS.md's tables |
| `REQUIREMENTS.md` | `04-FINDINGS.md` | coverage rows point at evidence | WIRED | §6 of FINDINGS.md maps every REGN/MKNN ID to a findings section |

## Requirements Coverage

All 14 requirement IDs declared across the 6 plans (`MKNN-01,02,05,08` in 04-01; `REGN-04` in
04-02; `REGN-03,04`, `MKNN-07` in 04-03; `REGN-01,02,05,06` in 04-04; `MKNN-03..08` in 04-05;
`REGN-02`, `MKNN-07,08` in 04-06) union to exactly REGN-01..06 and MKNN-01..08 — matching
REQUIREMENTS.md's Phase 4 mapping (14 requirements) with no orphans and no gaps.

| Requirement | Status | Evidence |
|---|---|---|
| REGN-01 | SATISFIED | Ambient 768-d density shown for all 10,000 points |
| REGN-02 | SATISFIED (reported, not controlled — stated as such) | Both Spearman correlations, region-level Mann-Whitney |
| REGN-03 | SATISFIED (validated only at codimension-1, gap stated) | ARI=1.0 known-answer test; codimension gap disclosed |
| REGN-04 | SATISFIED | Ordering verified mechanically (Critical Focus §1-2) |
| REGN-05 | SATISFIED | Region counts sum to 10,000 exactly |
| REGN-06 | SATISFIED | `04_region_partition.npz` freezes v/labels/keep_idx/excluded_idx/eigval_spectrum before any regional MKNN number |
| MKNN-01 | SATISFIED | Formula verified against `mknn.py` source |
| MKNN-02 | SATISFIED (n mismatch disclosed) | Global JSONL rows; paper comparison with ratio-over-chance |
| MKNN-03 | SATISFIED | 8-cell regional grid |
| MKNN-04 | SATISFIED | `null_scope: "region"` on every regional row |
| MKNN-05 | SATISFIED | CI present on every row |
| MKNN-06 | SATISFIED | Table across k=5,10,20,50 |
| MKNN-07 | SATISFIED | Verdict rule applied mechanically, HOLDS, qualified honestly |
| MKNN-08 | SATISFIED | Hubness skewness printed beside every result |

Re-minted IDs (REGN-01/03/04, REGN-06 added) are documented with rationale in
REQUIREMENTS.md's "Phase 4 Requirement Re-Mint" section, consistent with `ROADMAP.md`'s Phase 4
success-criteria strikethrough/supersession record.

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| region_partition known-answer test suite | `python -m pytest notebooks/pu_manifold/tests/test_region_partition.py -q` | `7 passed in 1.13s` | PASS |
| `src/effdim/` untouched during this milestone | `git diff --stat bc60f38..HEAD -- src/effdim/` | empty | PASS |
| Notebook fully executed | cell execution_count sequential 1..25, all with output | confirmed | PASS |
| VERDICT_RULE mechanical application | JSONL CI-disjointness + null-exceedance recomputed by hand at all 4 k | matches FINDINGS.md exactly | PASS |
| Region-size artifact arithmetic | raw-gap and ratio-over-chance-gap recomputed from JSONL at all 4 k | matches FINDINGS.md table to rounding | PASS |

## Anti-Patterns Found

None. `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` grep against `mknn.py`,
`region_partition.py`, and `region_partition_mknn_run.py` returns zero matches.

## CLAUDE.md Swiss Roll Rule (D4-12)

Plan `04-06` argues the standing Swiss roll rule does not apply to this phase's two
deliverables: the diametrical sign split reads an already-computed vector field and emits two
labels — no learned mapping, no latent space, no reconstruction, no manifold-recovery claim —
and MKNN is a metric on two given embeddings, also with no learned mapping. The curvature
estimator that produces the field is already covered by
`notebooks/02.5_swiss_roll_curvature_probe_check.ipynb`; the sign split has its own
known-answer coverage (`test_region_partition.py`'s two-antipodal-cone fixture, ARI=1.0),
which is a stricter test of the split logic than a Swiss roll would be. This reasoning is
stated explicitly in `04-FINDINGS.md` §9.5 / the notebook's own section 9.5, not silently
omitted. This is a reasonable, non-evasive reading of CLAUDE.md's rule (which is scoped to
"models that map data to a lower-dimensional representation and back, or that claim to recover
manifold structure") — a diametrical clustering label assignment on an existing field and a
pairwise alignment metric are neither. **Accepted as consistent with the rule's own scope.**

## Human Verification Required

None. All must-haves resolve mechanically from git history, source, cache artifacts, and the
executed notebook. The checkpoint ratification (orchestrator answering the Task 2 blocking
checkpoint on the user's standing authorization while asleep) is itself mechanically checkable —
the pre-registered values match the plan's own `ratify-recommended` option text verbatim, with
no amendment — and that check has been performed above.

## Gaps Summary

None. The ordering guarantee holds mechanically (verified from git ancestry and diff, not from
narration). The verdict rule was applied mechanically and correctly. All 14 requirement IDs are
accounted for and honestly qualified. The three accepted gaps (unvalidated field, unclosed
codimension gap, uncontrolled density confound) and the fourth mechanical qualification
(region-size artifact) are all stated at full strength, in the phase's own words, in
`04-FINDINGS.md`, and independently reproduced from the underlying JSONL in this verification.
The phase produced a confounded, heavily-qualified result while keeping its central
scientific-conduct guarantee — the pre-registration ordering — fully intact. That is what this
phase set out to do, and it did it.

---

*Verified: 2026-08-24T13:49:41Z*
*Verifier: Claude (gsd-verifier)*
