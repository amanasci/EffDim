---
status: pre-registered
phase: 02-eigenspectrum-audit-validity-gate
created: 2026-07-31
trigger: 02-FINDINGS.md §8 — the model is the one variable never varied
---

# Cross-Model Sweep — Pre-Registration

**Written and committed before any model other than `dinov3_vitb16` is fitted.** A 34-way
comparison run without a rule fixed in advance is the single easiest place in this project to
manufacture a result: with 34 numbers, some will look special by chance, and any of them can be
narrated after the fact. This document fixes the model set, the held-constant parameters, the
statistics, the primary question, and every secondary analysis, before any of them are seen.

## 1. What triggered this

Phase 2 measured `GATE_VERDICT = FAIL` on `legacysurvey_dinov3_vitb16` (`r = 0.052419`,
`m = 0.412071`). Four experiments established the result is not numerical, not an
implementation bug, not a single short-circuit edge, not kNN hop inflation, not L2
normalization, not an absence of manifold structure, not specific to the Legacy Survey column,
and not a property of the particular 10,000 objects drawn (`m` reproduces to 0.03% on a ~90%
disjoint sample).

**One variable was never varied: the model.** Every fit used DINOv3 ViT-B/16. `02-FINDINGS.md`
§8 names this as the largest open question.

## 2. Question

For the ~34 vision foundation models whose embeddings are published in
`UniverseTBD/pu-embeddings` (the embedding set accompanying *The Platonic Universe*,
arXiv:2509.19453):

**Is classical MDS an invalid description of the Isomap geodesic geometry across the board, or
is the DINOv3 ViT-B/16 result model-specific?**

## 3. Pre-registered design

### 3.1 Model set — fixed now, not extensible after seeing results

Every `legacysurvey_*` config in the dataset. Enumerated from
`get_dataset_config_names("UniverseTBD/pu-embeddings")` at pre-registration time:

```
astropt_015M, astropt_095M, astropt_850M
clip_base, clip_large
convnext_nano, convnext_tiny, convnext_base, convnext_large
dino_small, dino_base, dino_large, dino_giant
dinov3_vits16, dinov3_vits16plus, dinov3_vitb16, dinov3_vitl16,
  dinov3_vith16plus, dinov3_vit7b16
ijepa_giant, ijepa_huge
llava_15_7b, llava_15_13b
paligemma_3b, paligemma_10b, paligemma_28b
vit_base, vit_large, vit_huge
vit-mae_base, vit-mae_large, vit-mae_huge
vjepa_large, vjepa_giant, vjepa_huge
```

**Every model in this list is fitted and reported, regardless of outcome.** No model may be
dropped for being inconvenient, anomalous, expensive, or "not really comparable." A model that
fails to fit for a technical reason is reported as a technical failure with its error, never
silently omitted.

**No model outside this list may be added** without a new committed pre-registration. In
particular, the `desi_*`, `jwst_*`, `jwst_gio_*`, and `physics_*_test` survey families are
deliberately excluded here — varying survey *and* model at once would confound them.

### 3.2 What is held constant

Only the model varies. Everything else is pinned to the Phase 2 configuration:

```
survey column   = legacysurvey            (the hsc column is not read)
row_indices     = the exact 10,000 indices from seed 20260729
n_rows          = 10000
normalize       = True  (L2)
n_neighbors     = 15
n_components    = 18
eigen_solver    = "dense"
dtype           = float64 end to end
```

**Using identical `row_indices` across all models is deliberate**: it holds the object
population exactly constant, so a difference between two models cannot be a difference in which
galaxies were drawn. Experiment 4 (§7.3 of `02-FINDINGS.md`) established that the choice of draw
does not matter (`m` within 0.03% on a ~90% disjoint sample), so nothing is lost by fixing it.

**Ambient dimension is NOT held constant and cannot be.** It is a property of the model
(384 for `dino_small`, larger for bigger models). This is a structural confound, declared here
rather than discovered later, and §5.2 states the analysis that addresses it.

### 3.3 Statistics and thresholds — copied verbatim, unchanged

```
r = |lambda_min_neg| / lambda_max_pos
m = sum|lambda_neg| / sum|lambda|

R_MAX_PASS = 0.10   M_MAX_PASS = 0.05
R_MAX_MARGINAL = 0.25   M_MAX_MARGINAL = 0.15
```

Strict less-than at every boundary; verdict is the worse of the two. **These are not revisable
by this document or anything it produces.** The full 10,000-value spectrum is computed by hand
double-centring, exactly as Experiment 1 did; `n_components` does not enter `r` or `m`.

Recorded per model alongside `r`/`m`: ambient dimension, `n_positive`, `n_negative`,
`lambda_max_pos`, `lambda_min_neg`, float64 noise floor, graph connectivity
(`n_components == 1`), TwoNN and local-PCA intrinsic dimension, fit and eigensolve wall-clock.

## 4. Primary rule — fixed before any fit runs

**Rule A — across the board.** If every model in §3.1 returns FAIL, classical MDS is an invalid
description of Isomap geodesic geometry for this object population across all tested
architectures. The Phase 2 verdict is not a DINOv3 artifact, and any Phase 3 respec must be
curvature-native regardless of which encoder is chosen.

**Rule B — model-dependent.** If any model returns PASS or MARGINAL, the result is not universal.
That model is reported with its `r`, `m`, ambient dimension, architecture family, and intrinsic
dimension, and becomes a candidate encoder for a flat-embedding pipeline. Adopting it is **not**
automatic — it would require re-running Phase 1's connectivity and plateau-stability selection
under that model, and a documented amendment.

**Rule C — reporting obligation, binds under both A and B.** The complete 34-row table is
reported whatever it shows. No model is dropped, no subset is highlighted as "the interesting
ones" without the full table adjacent to it.

## 5. Secondary analyses — declared in advance

These are stated now so that finding one of them is a result rather than a story. All are
descriptive; none carries a pass/fail threshold, and none can override the primary rule.

1. **`m` vs ambient dimension.** Spearman correlation across models. This is the §3.2 confound:
   if `m` tracks ambient dimension, "all these models see curved geometry" is the wrong reading
   and "wide embeddings produce non-Euclidean graph geodesics" is the right one.
2. **`m` vs model capacity.** Within a family where size is ordered (`dino` small→giant,
   `dinov3` vits16→vit7b16, `convnext` nano→large, `vit` base→huge, `astropt` 015M→850M,
   `paligemma` 3b→28b). The source paper reports representational alignment increasing with
   capacity; whether `m` moves with capacity is a separate question worth recording.
3. **`m` by architecture family.** Whether astronomy-specific (`astropt`) differs from
   general-purpose vision, and whether contrastive (`clip`), self-distillation (`dino`,
   `dinov3`), masked-autoencoding (`vit-mae`), joint-embedding-predictive (`ijepa`, `vjepa`),
   and VLM (`llava`, `paligemma`) backbones separate.
4. **Intrinsic dimension vs `m`.** Whether models whose embeddings carry higher measured
   intrinsic dimension show higher negative mass.

## 6. Prohibitions

- MUST NOT revise the four thresholds, or add, drop, or reweight a gate statistic.
- MUST NOT add or remove a model from §3.1 without a new committed pre-registration.
- MUST NOT report a subset without the full table.
- MUST NOT adopt a passing model as a new project default; see Rule B.
- MUST NOT treat a universal FAIL as a disappointing outcome. Under Rule A the finding is
  *stronger* and more general than the single-model result it replaces.
- MUST NOT run the secondary analyses of §5 in place of the primary rule, or lead with a
  secondary correlation because the primary result is uniform.

## 7. Expected cost

~34 Isomap fits at n=10,000 plus ~34 dense 10,000x10,000 eigensolves, roughly 2 minutes each,
≈ 70 minutes of compute. Transfer is ~20 GB: only the `legacysurvey` column of each parquet is
read (the paired `hsc` column is skipped, halving the 39.9 GB total). Each model's embeddings
are released before the next is fetched, so peak disk stays small.

## 8. Outcome

To be appended below after all models complete. Empty at pre-registration time by design.
