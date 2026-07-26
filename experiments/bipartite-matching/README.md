# Bipartite matching — structure of the SAE shared-basis map W

How are two vision models' SAE feature dictionaries related? The Ridge affine
shared-basis map `C_dst ≈ C_src · W + b` (see `../SAE-shared-basis/`) carries
the best cross-model mKNN we have (~0.22 ViT↔DINO). This experiment
characterizes **W itself** and shows the alignment is carried by a
**high-rank but row-sparse bipartite feature graph** — not a permutation,
not an unstructured rotation.

All numbers below: Physics galaxies (Smith42), TopK SAEs `F2048_k64_seed0`,
n=16384 rows (seed 0), 70/30 train/test, mKNN k=10 with IDF-cosine on the
held-out 30%. Run on the platonic-universe GPU host, 2026-07-26.

## Findings

### 1. Not a permutation (`run_sparsity_rank.py`)

One partner per source feature (Hungarian one-to-one or greedy row-argmax —
they agree on only ~40% of rows) scores mKNN **0.13**: double the
random-matching control (0.06), so feature correspondences are real, but it
only ties dense cosine (0.132) and loses to unmapped SAE+IDF (0.172).

### 2. Row-sparse: a few dozen partners per feature suffice

DINO→ViT (ViT→DINO analogous; see `artifacts/sparse_W.json`):

| partners kept per row | % of W entries | mKNN |
|---:|---:|---:|
| 1 | 0.06% | 0.131 |
| 16 | 0.9% | 0.181 |
| 64 | 3.6% | 0.200 |
| full | 100% | 0.212 |

### 3. …but high-rank: no low-rank shortcut

Stable rank of W ≈ 700–800. Truncated SVD (`artifacts/lowrank_W.json`):
rank 32 → 0.081 (below dense cosine), rank 256 → 0.194, rank 512 → 0.210,
full → 0.212. The per-feature bundles collectively span hundreds of
independent directions. *High-rank + row-sparse* is the signature.

### 4. Soft shared concept clusters (`run_cluster_bipartite.py`)

Spectral co-clustering of the top-64 |W| graph (1479 live DINO × 1754 live
ViT features, 32 blocks), validated on held-out galaxies:

- Cluster pairs **co-fire across models**: corr(src-side activation,
  dst-side activation) = 0.6–0.95 for nearly every cluster — each block is a
  concept both models represent independently.
- Physics alignment is **morphology-first**: strongest clusters track
  `smooth_fraction` (r ≈ ±0.3–0.42), weak `mag_r` clusters (~0.25), nothing
  clean for `photo_z` / `stellar_mass`.
- Only *softly* modular: the 32 diagonal blocks hold ~30% of edge mass;
  the most self-contained blocks are physics-neutral.

Full per-cluster data (incl. feature indices for visual inspection):
`artifacts/cluster_W.json`.

### 5. Universal across model pairs (`run_transfer_pairs.py`)

All 10 pairs of {ViT-B, ViT-L, DINOv3, CLIP-B, ConvNeXt-B}
(`artifacts/transfer_pairs.json`):

| pair | dense | SAE IDF | shared best | top64 | top1 |
|---|---:|---:|---:|---:|---:|
| dinov3–vit_large | 0.180 | 0.191 | **0.247** | 0.234 | 0.136 |
| vit_base–vit_large | 0.152 | 0.192 | **0.232** | 0.220 | 0.142 |
| clip–dinov3 | 0.148 | 0.168 | **0.226** | 0.211 | 0.133 |
| dinov3–vit_base | 0.132 | 0.172 | **0.221** | 0.211 | 0.131 |
| convnext–dinov3 | 0.146 | 0.169 | **0.218** | 0.206 | 0.141 |
| convnext–vit_large | 0.139 | 0.159 | **0.200** | 0.190 | 0.108 |
| clip–vit_large | 0.120 | 0.140 | **0.195** | 0.185 | 0.107 |
| clip–convnext | 0.113 | 0.140 | **0.188** | 0.177 | 0.114 |
| convnext–vit_base | 0.107 | 0.144 | **0.186** | 0.177 | 0.108 |
| clip–vit_base | 0.098 | 0.134 | **0.183** | 0.172 | 0.103 |

Ordering (shared > IDF > dense), row-sparsity (top-64 ≈ full), and high rank
(stable rank 580–920) replicate on **every** pair. Pair ranking tracks
training-objective similarity (self-supervised / same-family align best).

## Files

| file | role |
|---|---|
| `_shared.py` | data loading, SAE encoding, IDF/mKNN, `RidgeMap`, top-k / rank-truncation helpers |
| `run_transfer_pairs.py` | all-pair transfer matrix (baselines + shared basis + W structure) |
| `run_sparsity_rank.py` | one pair: SVD rank sweep + top-k / Hungarian / random-1-to-1 sweep |
| `run_cluster_bipartite.py` | one pair: spectral co-clustering + cross-model / physics correlations |
| `artifacts/` | result JSONs from the 2026-07-26 runs (see below) |

`artifacts/`: `sparse_W.json` (sparsification, both directions ViT↔DINO),
`lowrank_W.json` (rank sweep, both directions), `cluster_W.json`
(32 clusters, DINO→ViT), `transfer_pairs.json` (10-pair matrix).

## Running

Requires the platonic-universe tree (`PLATONIC_ROOT` or `--platonic-root`)
with `data_hf/physics/*.parquet` and trained SAEs under
`outputs/sae/<stem>/<col>/F2048_k64_seed0` (train with
`experiments/sae/train_sae.py --feature-dim 2048 --k 64 --epochs 300
--batch-size 256 --lr 1e-3 --patience 30 --seed 0 --test-size 0.2`).
Dependencies as in `../SAE-shared-basis/requirements.txt`; `sae_model.py` is
imported from the sibling package.

```bash
# full pair matrix
python experiments/bipartite-matching/run_transfer_pairs.py --max-n 16384

# W structure for one pair
python experiments/bipartite-matching/run_sparsity_rank.py --src dinov3 --dst vit_base
python experiments/bipartite-matching/run_cluster_bipartite.py --src dinov3 --dst vit_base
```

## Caveats / next

- Single seed (SAE + split + subsample); headline numbers need error bars.
- n=16384 of 86471; full-n verification pending.
- No CKA / linear-stitching baselines yet.
- Cluster interpretability: pull galaxy images for top co-firing clusters
  (feature indices are in `cluster_W.json`).
