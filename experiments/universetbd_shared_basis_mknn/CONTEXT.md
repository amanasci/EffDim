# CONTEXT — UniverseTBD SAE shared-basis mKNN

Focused package for the multi-survey / multi-k evaluation of **Ridge affine SAE
shared-basis + IDF** against the Platonic Universe dense MKNN baseline
([arXiv:2509.19453](https://arxiv.org/pdf/2509.19453) Table 2).

Branch: `sae-shared-basis`.  
Canonical run tag: `full_paper_protocol`  
(`$PLATONIC_ROOT/outputs/universetbd_shared_basis_mknn_ks/full_paper_protocol/`).

---

## 1. Goal

Take the SAE shared-basis method that worked on Physics ViT↔DINOv3 and ask:

1. Does it still beat **paper-protocol dense** ambient cosine MKNN on
   UniverseTBD cross-survey / cross-model pairs?
2. How does the gap scale with neighbourhood size \(k \in \{10,20,50,100\}\)?

No new representation learning: reuse existing TopK SAE checkpoints
(`F≈2048`, typical TopK \(k_{\mathrm{SAE}}\in\{19..22\}\) or 128 for cosmosweb).

---

## 2. Internals

### 2.1 Pair catalog (`compatible_pairs.yaml`)

34 named pairs across surveys:

| survey | # pairs | Notes |
|---|---:|---|
| physics | 10 | Smith42/galaxies cross-model |
| jwst | 8 | HSC↔JWST cross-survey + same-survey cross-model |
| desi | 5 | HSC↔DESI + cross-model |
| legacy | 7 | HSC↔Legacy + cross-model |
| cosmosweb | 4 | high-SNR HSC↔JWST |

Kinds:

- **cross_survey** — same encoder, two surveys (paper Table 2 style)
- **cross_model** — two encoders, same objects

Embeddings: row-aligned parquets under `data_hf/`. Cap usually `n≤16384`
(JWST uses full \(N\approx 1496\)).

### 2.2 Shared-basis map

For codes \(C_1, C_2 \in \mathbb{R}^{N\times F}\) from each side’s TopK SAE:

1. Train/test split (`test_size=0.2`, fixed seed).
2. Fit Ridge both ways on **train only** (standardized codes):

\[
C_{\text{basis}} \approx C_{\text{other}} W + b
\]

3. Fit IDF weights on train activations only:

\[
\mathrm{idf}_j = \log\frac{N_{\mathrm{train}}}{1 + \#\{\text{rows with }c_j>0\}}
\]

4. Apply \(W,b\) and IDF to **all** rows; score as below.

Core implementation: `sae_affine_basis_mknn_gpu.py`
(`fit_affine_express_in_basis`, `encode`, `idf_np`, `knn_cos`, `mknn`).

### 2.3 Evaluation protocol (important)

Earlier drafts averaged MKNN **only among the test subset** (kNN gallery =
test). That **inflates** absolute dense scores vs the paper and is **not**
the baseline we test against.

Current protocol:

| Method | Neighbour search | Averaged over |
|---|---|---|
| `dense_cosine` | Full catalog | All \(N\) objects (**paper**) |
| `dense_cosine_heldout` | Full catalog | Test queries only |
| SAE / shared / IDF | Full catalog | Test queries only (maps fit on train) |
| `dense_cosine_test_subset` | Test subset only | Test (diagnostic; inflated) |

**Primary lift:** \(\Delta =\) `shared_best_basis_idf` − `dense_cosine_heldout`  
(same query set / gallery rule).

**Absolute paper baseline:** `dense_cosine` (compare to Table 2 on overlapping pairs).

Do **not** score train-mapped codes with full-catalog averages that include
train rows as queries — that leaks and can push JWST shared MKNN into the
0.6 range unrealistically.

---

## 3. Key findings (`full_paper_protocol`, 34/34 pairs)

### 3.1 Aggregate (mean over pairs)

| k | paper dense | heldout dense | SAE IDF | shared best IDF | Δ(shared−heldout) |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.106 | 0.107 | 0.096 | **0.127** | **+0.020** |
| 20 | 0.121 | 0.122 | 0.115 | **0.154** | **+0.032** |
| 50 | 0.152 | 0.153 | 0.151 | **0.199** | **+0.046** |
| 100 | 0.187 | 0.187 | 0.187 | **0.241** | **+0.054** |

Lift **increases with \(k\)**.

Relative to the older test-subset-only full run, absolute scores dropped
(protocol correction), but **relative lift is larger** (e.g. k=10: +19% vs
old +7%).

### 3.2 Where the gain lives

At k=10 (heldout dense vs shared):

| slice | dense | shared | Δ |
|---|---:|---:|---:|
| **cross_survey** | 0.064 | **0.102** | **+0.038 (+60%)** |
| cross_model | 0.146 | 0.149 | +0.003 |

By survey (k=10): JWST strong (+0.087); physics / cosmosweb / legacy modest
positive; **DESI mean negative** (shared below dense) — pulls the global mean
down.

Example paper-style pairs (k=10):

| pair | paper dense (ours) | shared best |
|---|---:|---:|
| `jwst_cross_vit` | 0.112 | **0.224** |
| `legacy_cross_vit` | 0.028 | 0.041 |

### 3.3 Dense vs published Table 2

Our paper-protocol dense is much closer than test-subset scoring, but still
above published fractions on overlapping pairs (local `data_hf/` parquets ≠
necessarily identical official HF dumps; Legacy uses n=16384 not full ~100k):

| pair | ours dense k=10 | paper | Δ |
|---|---:|---:|---:|
| `jwst_cross_vit` | 0.112 | 0.068 | +0.044 |
| `jwst_cross_convnext` | 0.113 | 0.072 | +0.041 |
| `legacy_cross_vit` | 0.028 | 0.009 | +0.019 |
| `legacy_cross_dino` | 0.026 | 0.016 | +0.010 |

Treat `dense_cosine` as the **in-experiment baseline** matched to the paper
*protocol*; residual level gaps are data/N, not a return to test-subset inflation.

---

## 4. Interpretation

- Shared basis is a **cross-survey alignment** tool more than a cross-model
  one: ambient cosine already competes when two encoders see the same survey.
- SAE IDF alone does not beat dense on the aggregate under this protocol;
  the affine map into a shared SAE feature space does.
- Reporting should lead with **cross-survey** (and/or paper Table 2 pairs)
  if the claim is “beat Platonic Universe dense MKNN.”

---

## 5. Related history (out of scope of this folder)

The broader geometry → SAE path (Tyagi charts, Lasso maps, eigenbasis
controls, Physics-only Ridge breakthrough ~0.22 vs 0.13 dense under
test-split mKNN) lives in the older
`experiments/SAE-shared-basis/` notes. This folder only ships the
UniverseTBD multi-k paper-protocol evaluation.
