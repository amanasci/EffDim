# Phase 6 Findings — Point-Cloud Curvature-Conditioned Linear Decodability

**Date:** 2026-08-24. **Milestone:** v1.1 PU Manifold Curvature.
**Executed autonomously** at the developer's standing instruction of 2026-08-24.

**One-line outcome.** With the ridge map, the split, the residual metric and the verdict criteria
held byte-identical to Phase 5 and **only the curvature field changed** — from the three CAE
decoder-side `||H||` fields to Phase 4's sealed density-corrected `centroid_mean_curvature` field
at `K_FROZEN = 500` — the verdict is **`NO DETECTABLE RELATIONSHIP`**. It fails on criterion (a)
alone: the highest and lowest tertile's 95% bootstrap CIs **overlap by 0.000914**. Criteria (b)
and (c) both hold. **The answer changed with the instrument**, and the two instruments' fields
are measured here to be near-orthogonal.

---

## 1. What this phase claims and what it deliberately does not

**Claim.** Under `VERDICT_RULE`, frozen in committed source at `c11218c` before any Phase 6
number existed, held-out per-point residual from a single global ridge map between the two PU
modalities does **not** differ detectably between the highest and lowest tertile of the Phase 4
point-cloud curvature field.

**What this phase deliberately does not claim, at full strength:**

- **That either field measures true curvature.** The point-cloud field is validated on PU by
  split-half reliability alone (G6-01), which cannot detect a bias both halves share — measured
  directly on the Swiss roll at `R_H = 0.990` alongside `rho = 0.469` against the true answer.
- **That Phase 6 is right and Phase 5 wrong, or the reverse.** A disagreement localizes to the
  instrument. Nothing here adjudicates which instrument is correct (G6-04).
- **That `NO DETECTABLE RELATIONSHIP` means no relationship exists.** It is a terminal outcome of
  a rule with a specific power, on 3,000 held-out points, at one `k`, in three buckets.
- **That `K_FROZEN = 500` is a principled `k`.** `04_k_freeze.json` records `rule_fired: false` —
  it is the largest `k` the budget ran, not one the freeze rule selected (G6-03).

**This document reopens, softens, recomputes and reinterprets no sealed verdict** — not Phase 2's
`GATE_VERDICT = FAIL`, not Phase 02.2's `CAE_VERDICT = FAIL`, not Phase 4's `HOLDS`, and not
Phase 5's `SPLIT ACROSS SEEDS`.

---

## 2. The protocol inheritance is exact, and that is checkable

Phase 6's whole design rests on changing one thing. Two independent confirmations:

| quantity | Phase 5 | Phase 6 |
|---|---|---|
| `mean_residual_overall` | `0.06642936194948156` | `0.06642936194948156` |
| `r2_overall` | `0.6439307736500615` | `0.643931` |
| `selected_alpha` | `0.1` | `0.1` |
| `n_train` / `n_test` | 7000 / 3000 | 7000 / 3000 |

The residual mean is **byte-identical**: the two phases score literally the same 3,000 held-out
numbers. `test_no_phase_5_scalar_constant_is_silently_dropped` enumerates every scalar constant in
`linear_probe` and requires each to be inherited-with-equal-value or explicitly excluded; it
passes.

**One defect was found and fixed en route.** The first freeze omitted `R2_MULTIOUTPUT` and the
runner passed `"uniform_average"` instead of Phase 5's frozen `"variance_weighted"`, so the first
run reported `r2 = 0.605806`. Recorded in full, with what it could and could not have moved, in
`06-PREREGISTRATION-AMENDMENT-01.md`. It never reached the residuals or the verdict —
`apply_verdict_rule` contains zero references to `r2`. The superseded record remains as the first
row of the JSONL rather than being deleted.

---

## 3. The result

```
VERDICT: NO DETECTABLE RELATIONSHIP
criteria: ci_disjoint=False   residual_higher_at_high_curvature=True   size_match_sign_stable=True
```

Field: Phase 4 `h_norm`, `n = 10,000`, median `13.3327`, p05 `7.5846`, p95 `29.9105`,
`p95/p05 = 3.944`. Tertile edges `11.216966296798418` and `16.002525781781117`.

| bucket | mean residual | 95% CI | n (test) |
|---|---|---|---|
| 0 (lowest `‖H‖`) | 0.065026 | [0.062365, 0.067909] | 1000 |
| 1 | 0.064105 | [0.061263, 0.066966] | 1006 |
| 2 (highest `‖H‖`) | 0.070193 | [0.066995, 0.073620] | 994 |

**Why it failed.** Only criterion (a). The low bucket's CI reaches `0.067909`; the high bucket's
starts at `0.066995` — an **overlap of 0.000914**. The high-minus-low difference is `+0.005167`,
7.95% of the low bucket's mean, in the hypothesized direction, and the size-matched re-check finds
that sign in **200 of 200** repeats (`median_diff = +0.005160`).

**The pattern is not monotone.** Bucket 1 (`0.064105`) sits *below* bucket 0 (`0.065026`). Only
the top tertile is elevated. A monotone degradation with curvature is not what these three numbers
show, and that is worth more than the verdict string.

**Sensitivity, not a gate.** `spearman(field, residual)` on the test split is `+0.0461`
(`p = 0.0115`, `n = 3000`). Positive and nominally significant, and it neither establishes nor
overturns the verdict.

---

## 4. D6-06 — cross-estimator agreement, closing D4-08

Recommended at the Phase 3 close and declined twice (D4-03, D4-08). Free here, because both
phases index the same sealed subsample. Spearman between Phase 4's point-cloud field and each
Phase 5 decoder field, over all 10,000 rows:

| Phase 5 seed | Spearman rho | p |
|---|---|---|
| 20260813 | **−0.0875** | 1.90e−18 |
| 20260814 | **+0.0487** | 1.11e−06 |
| 20260815 | **−0.1177** | 3.59e−32 |

**The two instruments do not agree.** All three correlations are below `|0.12|`, and the signs are
inconsistent — one positive, two negative. The p-values are tiny only because `n = 10,000`; an
`|rho|` of 0.05–0.12 is a near-null association however small its p-value.

**D4-08 is closed with a negative answer.** This does not tell you which instrument is right. It
does mean the decoder-side and point-cloud fields are not two measurements of one quantity, so a
verdict from either cannot be read as corroborating the other.

**D6-07.** PU has no ground-truth `H`, so no direction axis against truth exists here. Spike 003's
rule — a rank gain without a direction gain is not curvature recovery — cannot be applied to PU,
and that absence is stated rather than passed over.

**D6-08, density.** `spearman(density, h_norm) = +0.0300` (`p = 0.0027`, `n = 10,000`), computed
from `curvature_probe.local_density_weights` at `K_DENSITY = 30`. Near-null. Disclosure only.

---

## 5. Phase 5 and Phase 6 side by side

| | Phase 5 | Phase 6 |
|---|---|---|
| field | 3 CAE decoder-side `‖H‖` | Phase 4 point-cloud, `k = 500` |
| requires training | yes (3 checkpoints, ~2.6 h each) | **no** |
| model seeds | 3 | **none** |
| effective distinct field levels / 10,000 | 10,000 / **4** / **3** | **9,750** |
| verdict | `SPLIT ACROSS SEEDS` (2 of 3 HOLDS) | `NO DETECTABLE RELATIONSHIP` |
| headline effect (size-matched median diff) | `+0.0413` (seed 13), `+0.0096` (seed 15) | `+0.0052` |

Two of the three fields Phase 5 bucketed 10,000 points by take **three or four distinct values**
(`05_curvature_buckets_seed*.npz`, `effective_distinct_levels` `[4,4,4]` and `[3,3,3]`). The
point-cloud field takes 9,750.

**Neither verdict upgrades or downgrades the other.** Phase 5's `SPLIT ACROSS SEEDS` stands
exactly as sealed. What Phase 6 adds is that the outcome is **instrument-dependent**, and that the
instruments are near-orthogonal on this data.

---

## 6. A rule-text discrepancy found, and its scope measured in both directions

`VERDICT_RULE`'s criterion (c) — inherited verbatim in shape from Phase 5 — reads "the sign
survives ... with CIs disjoint in at least half of `SIZE_MATCH_N_REPEATS = 200` repeats."

**The implementation tests something else.** `apply_verdict_rule` reads
`size_match["sign_stable"]`, and `size_matched_check` computes
`sign_stable = bool(np.all(diffs > 0) or np.all(diffs < 0))` — *all repeats share the sign*.
`ci_disjoint_fraction` is computed, returned, and **never consulted by the verdict**. The
function's own docstring describes `sign_stable` correctly; it is the rule *text* that
misdescribes it.

**Measured scope, both phases:**

- **Phase 5 — immaterial.** All three seeds recorded `ci_disjoint_fraction = 1.0` (200 of 200).
  Both readings of criterion (c) give the same answer, so Phase 5's sealed verdicts are unaffected
  under either reading.
- **Phase 6 — the readings diverge but the verdict does not.** `ci_disjoint_fraction = 0.0`
  alongside `sign_stable = True`. Criterion (c) passes as implemented and would fail as written —
  but criterion (a) already failed, so the verdict is `NO DETECTABLE RELATIONSHIP` either way.

**So the discrepancy changes no verdict anywhere in the record.** It is reported because it is
real, because it would matter in a future run where (a) passes and the disjoint fraction is low,
and because the rule text is what a reader quotes. **Reconciling the text with the code touches
Phase 5's sealed `VERDICT_RULE` and is therefore not for autonomous action.**

---

## 7. Artifacts

| path | what |
|---|---|
| `notebooks/pu_manifold/pointcloud_probe.py` | frozen constants, `assert_preregistered`, `VERDICT_RULE` |
| `notebooks/pu_manifold/tests/test_pointcloud_probe.py` | 26 tests incl. the exhaustive inheritance check |
| `notebooks/diagnostics/pointcloud_probe_decodability_run.py` | runner, `--selfcheck` and `--mode bucketed` |
| `notebooks/.cache/06_pointcloud_probe_decodability.jsonl` | 2 rows; **the second is authoritative** |
| `notebooks/.cache/06_probe_selfcheck.jsonl` | planted-data selfcheck records |
| `06-PREREGISTRATION.md`, `06-PREREGISTRATION-AMENDMENT-01.md` | the freeze and its one amendment |

**Ordering proof.** Freeze `c11218c` → runner `37d1ba8` → serialization fix `a3883d6` → amendment
`62dc10a` → the authoritative run. Every commit precedes the run that produced the number.
