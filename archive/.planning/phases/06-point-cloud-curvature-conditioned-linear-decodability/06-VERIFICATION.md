---
phase: 06-point-cloud-curvature-conditioned-linear-decodability
verified: 2026-08-24
status: passed
score: 9/9 checks verified
executed: autonomously, at the developer's standing instruction of 2026-08-24
---

# Phase 6 Verification

**Scope note.** `status: passed` means the phase's *conduct* was verified — the freeze held, the
protocol inheritance is exact, the machine has discriminating power, and every reported number
was reproduced from the artifact rather than read from a summary. It does **not** mean the
hypothesis was supported. The verdict is `NO DETECTABLE RELATIONSHIP`.

**Awaiting developer review.** This phase was planned and executed without a human checkpoint.
Section 4 lists the two judgment calls made autonomously that a reviewer should confirm or
overturn.

## 1. The ordering guarantee — reproduced

```
$ git log --oneline -- notebooks/pu_manifold/pointcloud_probe.py
62dc10a fix(06): amendment 01 -- R2_MULTIOUTPUT was dropped from the Phase 6 freeze
c11218c feat(06): freeze all Phase 6 pre-registration constants -- the D6-05 freeze commit
```
Exactly two commits. Both ancestors of `HEAD`, as are the runner (`37d1ba8`) and the
serialization fix (`a3883d6`). **VERIFIED.**

```
$ git diff 62dc10a HEAD -- notebooks/pu_manifold/pointcloud_probe.py \
                            notebooks/diagnostics/pointcloud_probe_decodability_run.py
(empty)
```
No constant or rule amended after amendment 01. **VERIFIED.**

Record mtime `22:56:46`; amendment commit `62dc10a` at `22:56:13`. The authoritative number was
produced **33 seconds after** the last commit that could have changed it. **VERIFIED.**

## 2. Isolation — reproduced

- `git status --porcelain src/effdim/` → 0 lines. `src/effdim/` untouched, per CLAUDE.md's v1.1
  rule. **VERIFIED.**
- `git log --oneline c11218c..HEAD -- notebooks/pu_manifold/linear_probe.py` → 0 commits. Phase
  5's sealed module was imported, never edited (D6-04). **VERIFIED.**
- No existing notebook or runner was deleted or rewritten; Phase 6 adds three files.
  **VERIFIED.**

## 3. The result — read directly from the artifact

Read from `notebooks/.cache/06_pointcloud_probe_decodability.jsonl` (2 rows; the **second** is
authoritative, distinguishable by `inheritance.…R2_MULTIOUTPUT == "variance_weighted"`):

| check | value | status |
|---|---|---|
| verdict | `NO DETECTABLE RELATIONSHIP` | in `VERDICT_VALUES` ✓ |
| criteria | `ci_disjoint=False`, `residual_higher=True`, `sign_stable=True` | consistent with the verdict ✓ |
| `mean_residual_overall` | `0.06642936194948156` | **byte-identical to Phase 5's** ✓ |
| `r2_overall` | `0.643931` | matches Phase 5's `0.6439307736500615` ✓ |
| `selected_alpha` | `0.1` | matches Phase 5 ✓ |
| bucket CI overlap | `0.000914` | recomputed from the stored CIs ✓ |
| cross-estimator (D6-06) | `−0.0875`, `+0.0487`, `−0.1177` | all `|rho| < 0.12`, signs inconsistent ✓ |

The byte-identical residual mean is the load-bearing check: it establishes that Phase 6 scored the
same 3,000 held-out numbers Phase 5 did, which is the sole condition under which the two verdicts
are comparable.

## 4. Judgment calls made autonomously — for developer review

1. **Amendment 01 was applied and the phase re-run, rather than halting.** `R2_MULTIOUTPUT` was
   omitted from the first freeze. The correction was not a free choice — the value is fixed by
   Phase 5's sealed constant — and `apply_verdict_rule` never reads `r2`, so no verdict could
   move. The superseded record was kept. **A reviewer may prefer that a freeze defect halt the
   phase for a checkpoint instead.**
2. **The criterion-(c) rule-text discrepancy was reported, not fixed.** `VERDICT_RULE`'s text says
   criterion (c) requires CIs disjoint in ≥ half the repeats; the implementation tests only that
   all repeats share the sign. Measured scope: immaterial in Phase 5 (all three seeds recorded
   `ci_disjoint_fraction = 1.0`) and immaterial in Phase 6 (criterion (a) already failed).
   Reconciling the text touches Phase 5's **sealed** `VERDICT_RULE`, which is not for autonomous
   action. **This is the item most in need of a decision.**

## 5. Self-check

`--selfcheck` on planted data, touching no PU row: a field that genuinely drives residual returns
`HOLDS`; the same field shuffled returns `NO DETECTABLE RELATIONSHIP`. The machine has
discriminating power in both directions. Re-run after the serialization fix and after amendment
01; passes in all cases. **VERIFIED.**

## 6. Tests

26 Phase 6 tests pass, including `test_no_phase_5_scalar_constant_is_silently_dropped`, which
enumerates every scalar constant in `linear_probe` and requires each to be inherited with an equal
value or explicitly excluded — the check that would have caught amendment 01's defect. Full suite
before the phase: **414 passed, 1 skipped**. **VERIFIED.**

## 7. What is NOT verified

- That either curvature field measures true curvature (G6-01; no ground truth for PU exists).
- That `K_FROZEN = 500` is a principled choice (G6-03; `rule_fired: false`).
- That the point-cloud instrument is better than the decoder instrument. D6-06 establishes only
  that they disagree.
