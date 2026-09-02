# Relational Geometry Size Scaling Report

Methodology aligned with arXiv:2509.19453 (within-family adjacent size
comparisons + exact binomial test against p=0.5).

Primary comparison mode: **paper_same_size_cross_survey**
(Legacy Survey ↔ HSC, same architecture size on both sides).

## Config
```json
{
  "phase": "analyze",
  "families": [
    "convnext",
    "astropt",
    "dinov2",
    "vit",
    "ijepa"
  ],
  "sae_cache": "outputs/universetbd_shared_basis_mknn_ks/size_scaling",
  "bsf_cache": "outputs/universetbd_shared_basis_mknn_ks/size_scaling_bsf",
  "comparison_mode": "paper_same_size_cross_survey",
  "method_map": {
    "dense": [
      "dense_cosine",
      "paper_full_catalog"
    ],
    "sae": [
      "shared_best_basis_idf",
      "heldout_query_full_gallery"
    ],
    "bsf": [
      "shared_best_basis_cosine",
      "heldout_query_full_gallery"
    ]
  },
  "cache_ks": [
    10,
    20,
    50
  ],
  "note": "Primary k=10; cache has k\u2208{10,20,50}. k=5/100 require recompute."
}
```

## Answers

### 1. Does dense mKNN size-scaling reproduce?
8/11 positive (frac=0.727); one-sided p=0.1133, two-sided p=0.2266; mean Δ=+0.00217

### 2–3. Adjacent increases and binomial tests (dense k=10)
8/11 positive (frac=0.727); one-sided p=0.1133, two-sided p=0.2266; mean Δ=+0.00217

### 4. Strongest / weakest families
- Strongest first→last: ijepa (Δ=+0.00738)
- Weakest first→last: dinov2 (Δ=+0.00107)

### 5. k=10 vs k=50/100
- dense k=10: 8/11 positive (frac=0.727); one-sided p=0.1133, two-sided p=0.2266; mean Δ=+0.00217
- dense k=50: 8/11 positive (frac=0.727); one-sided p=0.1133, two-sided p=0.2266; mean Δ=+0.00325
- dense k=100: _no rows for dense k=100_

### 6. Shared SAE vs dense scaling
- SAE k=10: 7/11 positive (frac=0.636); one-sided p=0.2744, two-sided p=0.5488; mean Δ=+0.00146

### 7. Shared BSF vs dense scaling
- BSF k=10: 9/11 positive (frac=0.818); one-sided p=0.03271, two-sided p=0.06543; mean Δ=+0.00190

### 8. Does SAE/BSF lift itself scale?
- bsf lift@k10 vs logP Spearman=+0.144 (p=0.594); mean lift=+0.00708
- sae lift@k10 vs logP Spearman=-0.043 (p=0.875); mean lift=+0.00520

### 9–15. Unpaired / oracle / recoverability
_Filled when `--phase oracle` / `--phase unpaired` complete._

### 16–17. Reference / leave-one-family-out
Primary ladders use paper same-size cross-survey references (not pooled).
Leave-one-family-out (dense mKNN@10):
- drop astropt: 6/9 pos, one-sided p=0.2539
- drop convnext: 6/8 pos, one-sided p=0.1445
- drop dinov2: 6/8 pos, one-sided p=0.1445
- drop ijepa: 7/10 pos, one-sided p=0.1719
- drop vit: 7/9 pos, one-sided p=0.08984

### 18. Curve shape
See family Spearman / first→last deltas; many ladders are mild/plateau-ish.

### 19. vs arXiv:2509.19453
Paper reported crossmodal 28/33 positive steps (p≈3e-5). Compare our dense
binomial counts above on the official Legacy↔HSC ladders.

### 20. Strongest defensible statement
_Update after oracle/unpaired phases; Phase-1: dense adjacent signs + SAE/BSF lifts._

## Tables
- `model_size_manifest.csv`
- `dense/family_scaling_dense.csv`
- `sae/family_scaling_sae.csv`
- `bsf/family_scaling_bsf.csv`
- `combined/adjacent_size_differences.csv`
- `combined/binomial_scaling_tests.csv`
- `combined/family_correlations.csv`
- `combined/representation_lift_scaling.csv`
- `combined/leave_one_family_out.csv`

## Oracle / unpaired update

- Oracle mKNN@10 adjacent: 9/11 (one-sided p=0.03271)

## Oracle / unpaired update

- Oracle mKNN@10 adjacent: 9/11 (one-sided p=0.03271)
- unpaired_dense mKNN@10 adjacent: 2/6 (one-sided p=0.8906)
- unpaired_sae_shared mKNN@10 adjacent: 4/6 (one-sided p=0.3438)
- Recoverability mknn vs logP Spearman=-0.533
- Recoverability cka vs logP Spearman=-0.018
