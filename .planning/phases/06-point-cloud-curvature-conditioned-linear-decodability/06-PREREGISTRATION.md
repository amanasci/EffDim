# Phase 6 Pre-Registration — Point-Cloud Curvature-Conditioned Linear Decodability

**Date:** 2026-08-24. **Milestone:** v1.1 PU Manifold Curvature.

**This document is the human-readable record of constants frozen in committed source
(`notebooks/pu_manifold/pointcloud_probe.py`) before any Phase 6 probe number existed.** The
source is authoritative; this file restates it so the freeze is readable without running Python.
Git ancestry is the proof of ordering, exactly as Phase 5 established it: the freeze commit
precedes the commit that first gives the runner a real-data execution path, which precedes the
run.

**Nothing here reopens, softens, recomputes or reinterprets any sealed verdict** — not Phase 2's
`GATE_VERDICT = FAIL`, not Phase 02.2's `CAE_VERDICT = FAIL`, not Phase 3's field, not Phase 4's
`HOLDS`, and not Phase 5's `SPLIT ACROSS SEEDS`.

---

## 1. The question

Does linear crossmodal decodability degrade as **point-cloud-estimated** manifold curvature
magnitude increases?

One global ridge map `hsc -> legacysurvey` on the frozen PU embeddings; held-out per-point
squared-L2 residuals bucketed into tertiles by the Phase 4 sealed `centroid_mean_curvature`
field; judged once under `VERDICT_RULE`.

## 2. What changed from Phase 5, exhaustively

**One thing: the curvature field.** Everything else is Phase 5's own frozen value, inherited by
explicit re-declaration and checked mechanically by
`test_inherited_constants_match_phase_5_exactly` (16 constants, `==`, no tolerance).

| | Phase 5 | Phase 6 |
|---|---|---|
| `CURVATURE_SOURCE_FUNCTION` | `chart_curvature.chart_curvature_field` | `curvature_probe.centroid_mean_curvature` |
| requires a trained model | yes — three CAE checkpoints | **no** |
| model seeds | 3 (`20260813/14/15`) | **none** — deterministic given `k` |
| number of verdicts | 3 per-seed + 1 combined | **1** |
| `SPLIT ACROSS SEEDS` reachable | yes (and realized) | **no** |
| effective distinct field levels / 10,000 | 10,000 / **4** / **3** | **9,750** |

That last row is measured, not asserted: `05_curvature_buckets_seed*.npz` records
`effective_distinct_levels` of `[10000, 9904, 1173]`, `[4, 4, 4]` and `[3, 3, 3]` for the three
seeds. Two of the three fields Phase 5 bucketed 10,000 points by take **three or four values**.

## 3. The frozen constants

### Inherited from Phase 5 unchanged (D6-02, D6-04)

| Constant | Value |
|---|---|
| `TRAIN_FRACTION` | `0.7` |
| `SPLIT_SEED` | `20260824` |
| `SPLIT_RULE` | one permutation of `np.arange(10000)`; first 7,000 train, last 3,000 test; both sorted ascending; NOT stratified |
| `RIDGE_ALPHA_GRID` | `(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)` |
| `RIDGE_SELECTION_RULE` | `RidgeCV` generalized leave-one-out CV on the training split alone |
| `ALPHA_PER_TARGET` | `False` |
| `FIT_INTERCEPT` | `True` |
| `EMBEDDING_PREPROCESSING` | `"raw_as_cached"` — both modalities already L2-normalized upstream |
| `RESIDUAL_METRIC` | `"squared_l2_per_point"` |
| `N_BUCKETS` | `3` |
| `BUCKET_RULE` | tertiles over all 10,000 rows; edges applied to test rows, never recomputed on them |
| `N_BOOTSTRAP` | `1000` |
| `BOOTSTRAP_SEED` | `20260824` |
| `CONFIDENCE_LEVEL` | `0.95` |
| `SIZE_MATCH_RULE` | subsample to realized test-split bucket counts; sign stable if CIs disjoint in ≥ half the repeats |
| `SIZE_MATCH_N_REPEATS` | `200` |
| `SIZE_MATCH_SEED` | `20260824` |
| `K_DENSITY` | `30` |
| `FIELD_D` | `20` |
| `CURVATURE_CONVENTION` | `"trace"` |

### Phase 6's own (D6-01, D6-03)

| Constant | Value |
|---|---|
| `CURVATURE_SOURCE` | `"phase_4_sealed_point_cloud_field"` |
| `CURVATURE_SOURCE_ARTIFACT` | `notebooks/.cache/04_region_partition.npz` |
| `CURVATURE_SOURCE_KEY` | `"h_norm"` |
| `CURVATURE_SOURCE_FUNCTION` | `curvature_probe.centroid_mean_curvature` |
| `CURVATURE_DENSITY_CORRECTED` | `True` |
| `K_FROZEN` | `500` |
| `SEED_HANDLING_RULE` | `"single_field_no_seeds"` |
| `CROSS_ESTIMATOR_DISCLOSURE_SEEDS` | `(20260813, 20260814, 20260815)` — disclosure only |
| `VERDICT_VALUES` | `("HOLDS", "NO DETECTABLE RELATIONSHIP")` |

**The field is read, never recomputed** (D6-01). Recomputing it would silently re-tune `k` and
make Phase 4's freeze meaningless.

## 4. `VERDICT_RULE`

The verdict is **HOLDS** if and only if all three of:

- **(a)** the highest and lowest tertile's 0.95 percentile bootstrap CIs on mean per-point
  squared-L2 residual are **disjoint**;
- **(b)** the highest bucket's mean residual **strictly exceeds** the lowest bucket's;
- **(c)** the sign survives the size-matched re-check with CIs disjoint in **at least half** of
  200 repeats.

Otherwise **NO DETECTABLE RELATIONSHIP** — a complete, valid, terminal outcome. Never escalated
by the continuous Spearman statistic, never re-decided by trying a different `N_BUCKETS` or a
different `k`.

**These are the same three criteria Phase 5 applied per seed.** Holding the decision rule fixed
while changing only the field is what makes the two phases comparable at all.

## 5. Disclosures — reported beside the verdict, gating nothing

- **D6-06 / D4-08, cross-estimator agreement.** Spearman rank correlation between the Phase 4
  point-cloud field and each of the three Phase 5 decoder fields, over the same 10,000 rows.
  Recommended at the Phase 3 close and declined twice (D4-03, D4-08); free here because both
  phases index the same sealed subsample. **Not a gate.**
- **D6-08, density confound.** `spearman(density, h_norm)` at `K_DENSITY = 30`. **Not a gate.**
- **D6-07, the direction axis.** Spike 003's standing rule is that a rank gain arriving without a
  direction gain is not curvature recovery. PU has no ground-truth `H`, so **no direction axis
  against truth exists here**; that absence is stated rather than passed over.

## 6. Accepted gaps, at full strength

- **G6-01.** The field is validated on PU by split-half reliability alone, which cannot detect a
  bias both halves share — measured on the Swiss roll at `R_H = 0.990` alongside `rho = 0.469`
  against true curvature (`04-FINDINGS.md` Gap 1). No ground truth for PU exists, so no amount of
  further split-half measurement closes it. **Phase 6 inherits this and closes nothing.**
- **G6-02.** Magnitude ordering is the weaker functional at `d = 20`; spike 003 measured rank
  saturating at `rho ~ 0.41-0.65` while direction reached cosine `0.77-0.92`. Phase 6 buckets by
  magnitude.
- **G6-03.** `K_FROZEN = 500` is the largest `k` actually run, not one the freeze rule selected —
  `04_k_freeze.json` records `rule_fired: false`. At `k = 500` on 10,000 rows a neighbourhood is
  5% of the cloud; whether that is still local on PU is unmeasured.
- **G6-04.** A disagreement with Phase 5 localizes to the instrument. It does **not** establish
  which instrument is correct, does not validate either field, and does not show that either
  measures true curvature.
