---
status: pre-registered
phase: 02-eigenspectrum-audit-validity-gate
created: 2026-07-31
trigger: 02-FINDINGS.md §8 — the model is the one variable never varied
---

# Cross-Model Sweep — Pre-Registration

**Committed before any model other than `dinov3_vitb16` is fitted.** With 35 numbers some
will look special by chance; this fixes the model set, held-constants, statistics, primary
question, and every secondary analysis in advance.

## 1. Trigger

Phase 2: FAIL on `legacysurvey_dinov3_vitb16` (r=0.052419, m=0.412071). Ruled out: numerics,
bugs, short-circuit edges, kNN hop inflation, L2 normalization, absence of manifold
structure, column specificity, draw specificity (m within 0.03% on ~90% disjoint sample).
The model was never varied.

## 2. Question

Across the 35 vision foundation models in `UniverseTBD/pu-embeddings` (arXiv:2509.19453):
is classical MDS invalid across the board, or is DINOv3 ViT-B/16 special?

## 3. Design

**§3.1 Model set — fixed, not extensible.** Every `legacysurvey_*` config:

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

Every model fitted and reported regardless of outcome; technical failures reported with
their error, never omitted. No model added without a new pre-registration. `desi_*`,
`jwst_*`, `physics_*_test` excluded — varying survey and model at once would confound.

**§3.2 Held constant** (only the model varies): legacysurvey column, the exact 10,000
row_indices from seed 20260729, normalize True, n_neighbors 15, n_components 18, dense
solver, float64. Identical row_indices hold the object population constant (justified by
§7.3's 0.03% disjoint-sample result). **Ambient dimension cannot be held constant** — a
declared structural confound, addressed by secondary analysis 1.

**§3.3 Statistics/thresholds, verbatim, not revisable:** r, m as in Phase 2;
R_MAX_PASS=0.10, M_MAX_PASS=0.05, R_MAX_MARGINAL=0.25, M_MAX_MARGINAL=0.15; strict
less-than; worse-of-two. Full 10,000-value hand-double-centred spectrum. Recorded per model:
ambient dim, n_positive/negative, lambda_max_pos, lambda_min_neg, noise floor, connectivity,
TwoNN + local-PCA intrinsic dimension, wall-clock.

## 4. Primary rule — fixed before any fit

- **Rule A — across the board:** every model FAILs → classical MDS invalid for this
  population across all tested architectures; any Phase 3 respec must be curvature-native
  regardless of encoder.
- **Rule B — model-dependent:** any PASS/MARGINAL → not universal; that model reported with
  r/m/ambient/family/intrinsic-dim and becomes a *candidate* encoder — adoption requires
  re-running Phase 1's selection under it plus a documented amendment.
- **Rule C — reporting obligation:** the complete 35-row table is reported whatever it
  shows; no subset without the full table adjacent.

## 5. Secondary analyses — declared in advance, descriptive only

1. m vs ambient dimension (Spearman) — the §3.2 confound test.
2. m vs capacity within ordered families (dino, dinov3, convnext, vit, astropt, paligemma).
3. m by architecture family (astronomy-specific vs contrastive vs self-distillation vs MAE
   vs JEPA vs VLM).
4. Intrinsic dimension vs m.

## 6. Prohibitions

No threshold revision; no model set changes without new pre-registration; no subset
reporting; no adopting a passing model as default (Rule B); no treating universal FAIL as
disappointing (Rule A is the stronger finding); no leading with a secondary correlation
because the primary result is uniform.

## 7. Expected cost

35 fits + 35 dense eigensolves ≈ 2 min each ≈ 70 min compute; ~20 GB transfer (legacysurvey column only; skipping hsc halves the 39.9 GB total); embeddings released between models.

## 8. Outcome

To be appended after all models complete. Empty at pre-registration time by design.
