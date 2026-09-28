# Phase 6 Context — Point-Cloud Curvature-Conditioned Linear Decodability

**Created:** 2026-08-24. **Milestone:** v1.1 PU Manifold Curvature.
**Added autonomously** at the developer's standing instruction of 2026-08-24 ("create a phase 6
where we do centroid mean curvature to estimate on the raw point cloud and then redo phase 5
experiment of looking at probe performance that way"). No sealed verdict is reopened, softened,
recomputed or reinterpreted by this phase.

## 1. What this phase is

Phase 5 with **exactly one thing changed**: the field the residuals are bucketed by.

| held identical to Phase 5 | changed |
|---|---|
| `hsc -> legacysurvey` ridge map, one global fit | the curvature field |
| `TRAIN_FRACTION = 0.7`, `SPLIT_SEED = 20260824`, same `SPLIT_RULE` | |
| `RIDGE_ALPHA_GRID`, `RIDGE_SELECTION_RULE`, `ALPHA_PER_TARGET`, `FIT_INTERCEPT` | |
| `EMBEDDING_PREPROCESSING = "raw_as_cached"`, `RESIDUAL_METRIC = "squared_l2_per_point"` | |
| `N_BUCKETS = 3`, tertile `BUCKET_RULE` | edges recomputed on the new field |
| bootstrap CI, size-match, and verdict machinery in `linear_probe.py` | one verdict, not three |

Because the split, the map and the residuals are **byte-for-byte the same 3,000 held-out
per-point residuals Phase 5 scored**, Phase 5 and Phase 6 differ in the instrument and in
nothing else. That is the entire design.

## 2. Locked decisions

- **D6-01. The field is Phase 4's, not a new one.** `centroid_mean_curvature`, density-corrected,
  at `K_FROZEN = 500`, `FIELD_D = 20`, `K_DENSITY = 30` — already computed and sealed as `h_norm`
  in `notebooks/.cache/04_region_partition.npz`, shape `(10000,)`, over the same frozen subsample.
  **Phase 6 reads that array and does not recompute it.** Re-deriving it would silently re-tune
  `k` and make Phase 4's freeze meaningless.

- **D6-02. Inherit Phase 5's split rather than re-drawing it.** Call
  `linear_probe.train_test_split_indices(10000, TRAIN_FRACTION, SPLIT_SEED)` with Phase 5's own
  frozen values. Any re-draw would make the two phases incomparable and would forfeit the free
  cross-estimator reading in D6-06.

- **D6-03. One field, therefore one verdict.** `SEED_VERDICT_COMBINATION_RULE` and
  `PHASE_VERDICT_VALUES` do not apply and are not carried over. The point-cloud estimator is
  deterministic given `k` — it has no seeds — so `SPLIT ACROSS SEEDS` is not a reachable outcome
  and must not appear anywhere in this phase's artifacts. `VERDICT_RULE` is restated for a single
  verdict rather than reused verbatim.

- **D6-04. Additive only.** `notebooks/pu_manifold/linear_probe.py` is **sealed** — Phase 5's 31
  frozen constants live there and its git history is the ordering proof for Phase 5's result. It
  is imported, never edited. Phase 6's constants live in a new module,
  `notebooks/pu_manifold/pointcloud_probe.py`. `src/effdim/` is untouched.

- **D6-05. Freeze before any number.** Every Phase 6 constant, and the full `VERDICT_RULE`, is
  committed in source **before** the runner is capable of producing a Phase 6 probe number, with
  `assert_preregistered()` refusing to run while any constant is unset. Git ancestry is the proof,
  exactly as Phase 5 established it.

- **D6-06. Report cross-estimator agreement, and gate nothing on it.** Because both phases bucket
  the same 3,000 residuals, the Spearman rank correlation between Phase 4's `h_norm` and each of
  the three Phase 5 decoder fields is computable at zero extra cost and closes D4-08, which was
  recommended at the Phase 3 close and declined twice (D4-03, D4-08). It is a **disclosure**, not
  a gate: no Phase 6 verdict may be upgraded or downgraded by it.

- **D6-07. The direction axis travels with every rank statistic.** Spike 003's standing rule — a
  rank gain arriving without a direction gain is not curvature recovery. Where a direction axis
  exists it is reported beside the rank statistic; where it does not (PU has no ground-truth `H`),
  its absence is stated rather than passed over.

- **D6-08. The density confound is disclosed per Phase 5's D5-13 precedent.**
  `spearman(density, h_norm)` is reported alongside the verdict. Not a gate.

## 3. Inherited gaps, carried at full strength

- **G6-01 — the field is validated only by split-half reliability, which cannot detect a bias both
  halves share.** `04-FINDINGS.md` Gap 1 states this and measures it directly on the Swiss roll:
  `R_H = 0.990` (near-perfect two-half agreement) alongside `rho = 0.469` against true curvature.
  Reliability and correctness came apart on the one fixture where both could be checked. There is
  no ground truth for PU, so this gap cannot be closed by more of the same measurement. **Phase 6
  inherits it verbatim and does not close it.**

- **G6-02 — magnitude ordering is the weaker functional at `d = 20`.** Spike 003 measured rank
  saturating at `rho ~ 0.41-0.65` and direction at cosine `0.77-0.92` on fixtures with known
  answers, with magnitude ~50x attenuated. Phase 6, like Phase 4 and Phase 5, buckets by
  magnitude — the weaker of the two.

- **G6-03 — `K_FROZEN = 500` is the largest `k` run, not a `k` the freeze rule selected.**
  `04_k_freeze.json` records `rule_fired: false` — `median_R_H` never reached `0.5` with a gain
  below `0.03` anywhere in the grid `[30, 60, 120, 231, 350, 500]`, so the frozen value is a
  budget boundary. At `k = 500` on 10,000 rows a neighbourhood is 5% of the cloud, and whether
  that is still "local" on the PU manifold is unmeasured.

- **G6-04 — what Phase 6 does NOT do.** It does not validate the point-cloud field, does not
  establish that either field measures true curvature, does not adjudicate which estimator is
  correct, and does not reopen `CAE_VERDICT = FAIL` or Phase 5's `SPLIT ACROSS SEEDS`. A
  disagreement between the two phases localizes to the instrument; it does not tell you which
  instrument is right.

## 4. Source artifacts this phase reads

| path | what |
|---|---|
| `notebooks/.cache/04_region_partition.npz` | `h_norm` `(10000,)` — D6-01's field |
| `notebooks/.cache/04_region_partition.meta.json` | `K_FROZEN`, `K_DENSITY`, `FIELD_D`, subsample path |
| `notebooks/.cache/subsample_*.npz` | `hsc` and `legacysurvey` columns, same loader as Phase 5 |
| `notebooks/.cache/05_curvature_buckets_seed*.npz` | Phase 5's three decoder fields, for D6-06 only |
| `notebooks/pu_manifold/linear_probe.py` | imported unchanged for split/fit/residual/CI/verdict |
