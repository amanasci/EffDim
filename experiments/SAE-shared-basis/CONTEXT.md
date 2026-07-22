# CONTEXT — Geometry / SAE shared-basis discussion

Detailed handoff summary of the research conversation (Jul 19–23, 2026) that led to the `sae-shared-basis` branch and `experiments/SAE-shared-basis/` package.

**Primary pair throughout:** Physics galaxies, L2-normalized **ViT-B ↔ DINOv3 ViT-B** embeddings (row-aligned parquets from the Platonic Universe pipeline on [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies)). Secondary pilots used JWST / Legacy / DESI where noted.

**Success metric for cross-model structure:** mutual k-nearest-neighbour overlap (**mKNN**, typically \(k{=}10\)).

**Current git focus:** branch `sae-shared-basis` (worktree), package under `experiments/SAE-shared-basis/` — see also `README.md`.

---

## 1. Starting goal

Estimate **local geometry** of foundation-model embedding clouds (tangent / metric / extrinsic curvature), then test whether that geometry improves **cross-model neighbour agreement** (mKNN) relative to ambient cosine.

Early library work: Tyagi-style bootstrap PCA in `src/effdim/curvature.py` — local PCA about a **reference point** \(p\) (displacements \(y_j = x_j - p\)), **not** empirical-mean centering ([Tyagi, Vural & Frossard, arXiv:1208.1065](https://arxiv.org/abs/1208.1065)). Projector bootstrap variance \(\tilde V_d\) used as a local “curvature / chart instability” proxy.

---

## 2. Curvature vs physics probes

**Setup:** ViT-base Physics embeddings; Euclidean radii \(\approx 0.33/0.38/0.45\); bootstrap PCA (\(d{=}10\) or \(20/40\)); local CV linear probes on Smith42 labels (`mag_r_desi`, `photo_z`, `smooth_fraction`, `sfr`, …).

**Finding:** Higher local projector variance ↔ **worse** local probe \(R^2\), especially photometry (`mag_r_desi` Spearman often \(\sim -0.4\) to \(-0.6\)). So the curvature proxy tracks something real about local linear predictability.

**Does not imply:** that low-curvature charts give a better *cross-model* metric than ambient cosine.

---

## 3. Local metric / II / chart programmes (mostly negative for mKNN)

Many constructions were tried; pattern repeated:

| Idea | Outcome for ViT↔DINO mKNN |
|---|---|
| Global PCA subspaces | Beat random planes; **lose to full ambient cosine** on Physics n=16k (full \(\approx 0.088\); PCA-40 \(\approx 0.076\)) |
| Local Tyagi / induced metric | Beats random; loses to full |
| Graph + second fundamental form (II) | Weak; II barely tracks bootstrap variance on JWST |
| Local isotropy / multi-anchor SPD metrics | Implemented + tested; no clear mKNN win over ambient |
| Low-curvature sphere merges / ID filters | Fragmented CCs; not a usable global metric |
| CC sphere geodesic in PCA flat | Ambient already on unit sphere → degenerate if done ambiently; low-dim sphere fit not competitive |
| Isomap / ANN paths | Failed vs dense/PCA |

**Working picture of the cloud:** roughly a **~10D linear core + thick soft shell** (local PCA charts vary smoothly across spheres; PCA-95 planes median \(d\sim 87\) but still smooth). Ambient cosine already uses the informative directions; aggressive low-rank projection throws signal away.

JWST pilots (small \(N\)) showed higher absolute mKNN (e.g. full cosine \(\sim 0.26\) ViT↔DINO) but the same ranking: full ≥ global PCA ≥ Tyagi ≫ random.

---

## 4. SAE as a code space (modest then large gains)

TopK SAE (\(F{=}2048\), \(k{=}64\)) on Physics embeddings:

- Raw SAE-code cosine > dense slightly; **SAE × IDF** better still (test-split mKNN order-of-magnitude: dense \(\sim 0.13\), codes \(\sim 0.16\), IDF \(\sim 0.17\)).
- **IDF** = inverse document frequency on feature activation frequency: rare features upweighted before cosine.
- Learned diagonal / decoder-Jacobian pullback metrics did **not** beat IDF.

### Breakthrough: affine shared basis

Fit Ridge (standardized codes):

\[
C_{\text{basis}} \approx C_{\text{other}} W + b
\]

Evaluate mKNN between **true** basis codes and the **mapped** other codes in that same feature space.

**Physics n=16k test (reproduced after review fixes):**

| method | mKNN |
|---|---:|
| dense cosine | 0.132 |
| SAE IDF | 0.172 |
| shared ViT basis + IDF | 0.209 |
| **shared DINO basis + IDF** | **0.220** |

Dense Ridge map wins; **Lasso / L1** maps underperform (val-selected best shared \(\sim 0.13\)).

**Eigenbasis follow-ups (A/B):** SVD charts of \(W_{\mathrm{std}}\) beat shuffle/randn controls; full FISTA/Ridge maps remain near \(\sim 0.21\)–\(0.22\). Protocol C (local balls) showed local affine alignment is largely **illusory** if fit and evaluate on the same points (matched ≈ random) — hence later holdout fixes.

**Local affine “interpolation” pitfall:** Ridge can align any two neighbourhoods; random-ball controls are mandatory. Same lesson for local VAE/AE charts.

---

## 5. Nonlinear charts, VAEs, pullback metrics

| Probe | Result |
|---|---|
| Quadratic residual in knn balls | Balls look flat by construction |
| Quadratic in Euclidean radius balls | Linear degrades with \(r\); quadratic helps only slightly at large \(r\) |
| Local chart smoothness (dense + SAE PCA) | High Spearman ~0.9 — charts vary smoothly |
| Euclidean β-VAE | Loses badly to PCA on recon; mKNN collapses |
| Plain AE (\(\beta{=}0\)) | Slight recon edge at \(d{=}10\); then PCA catches up; mKNN not competitive with dense/PCA-64 |
| AE pullback \(G=J^\top J\) | Nearly constant / Euclidean-like (logvol CV tiny) |

**Acosta et al. (2212.10414)** topological VAEs need a **named latent topology** (\(S^1\), \(T^2\), …). That prior was **not** justified by the data (see §6). Continuous decoder-per-sphere remains a valid *method* for a pullback metric, but evidence said charts are too linear for \(G\) to be interesting (local AE logvol CV \(\approx 0\)).

---

## 6. Topology (TDA / paths)

**Rips PH (ripser):** Dense activations show real H1 mass (stronger than SAE codes/binary). But **no clean single loop** — top1/top2 persistence ratios \(\sim 1.0\)–\(1.3\) (best per-cluster \(\sim 1.7\)). Soft multi-modality; one large-scale connected component.

**Path / fundamental cycles** on kNN graph: many weak cycles; clean ones not highly cluster-pure; **plane diversity** higher across soft clusters than within — consistent with **homology (if any) living in different subspaces**, which explains messy ambient PH.

**Implication:** Cannot honestly hardwire a hyperspherical/toroidal VAE prior. Path sampling supports multi-subspace weak 1-cycles, not one template manifold.

---

## 7. SAE thrashing vs physics probes

Hypersphere SAE support churn (Jaccard distance of TopK supports, entropy, code_std, …) vs local probe \(R^2\):

**Mostly null** for mean \(R^2\). Some property-specific signals (e.g. activation entropy vs `mag_r_desi` negative at larger radius). Not a clean general rule that “thrashing ⇒ bad probes.”

---

## 8. What worked vs what did not (executive)

**Worked**

1. Tyagi bootstrap projector variance as a **local instability / probeability** correlate.
2. SAE codes + **IDF** as a better ambient than dense cosine (modest).
3. **Ridge affine map into a shared SAE basis** + IDF — clearest cross-model mKNN gain (~0.22).
4. Singular charts of the fitted \(W\) beating shuffle/randn (structure in the map, not noise).

**Did not work (for cross-model mKNN or clean topology)**

1. Replacing ambient cosine by local low-rank / Tyagi / II / isotropy metrics.
2. Forcing topological VAE priors from PH.
3. Global/local AE pullbacks as interesting non-Euclidean metrics.
4. Lasso shared-basis maps (weaker than Ridge and than dense).
5. Treating local Ridge alignment without random-ball / holdout controls.

**Standing geometric hypothesis:** embeddings behave like a **shared soft low-dimensional linear structure** (visible in SAE features and affine code maps), not like a single curved manifold with clean nontrivial \(\pi_1\) in ambient space.

---

## 9. Package state (`experiments/SAE-shared-basis/`)

| File | Role |
|---|---|
| `README.md` | How to run; data links; system requirements |
| `CONTEXT.md` | This document |
| `run_shared_basis.py` | CLI: `list` / `doctor` / `run` over `datasets.yaml` |
| `datasets.yaml` | Named cross-matched pairs |
| `sae_affine_basis_mknn_gpu.py` | **Ridge** shared basis (primary) |
| `sae_affine_lasso_basis_mknn_gpu.py` | L1 / Lasso maps |
| `sae_lasso_eigenbasis_mknn_gpu.py` | Protocols A/B/C (singular charts, low-rank \(W\), local balls) |
| `_common.py` | Alignment checks, positive TopK Jaccard, standardized singular charts, `PLATONIC_ROOT` |
| `sae/sae_model.py` | Vendored TopKSAE |
| `requirements.txt` | Runtime deps |

**Review fixes applied (and smoke + full suite re-run):**

- A/C: apply SVD of \(W_{\mathrm{std}}\) in **standardized** space (not raw codes/embeddings).
- C: fit local Ridge on 70% of ball, evaluate mKNN on held-out 30%.
- Lasso active-set: **positive** TopK, not \(|\hat y|\).
- Hyperparameters / “Best”: select on **validation**, report test.
- Ridge ref JSON only if meta matches; no silent wrong comparison.
- Non-square \(W\): skip symmetrized eig.
- Parquet pairs: equal length required (optional `--allow-truncate`).

**Full suite mKNN (Physics ViT↔DINO, n=16384, post-fix):** Ridge shared DINO basis+IDF **0.220**; Lasso val-selected ~**0.128**; eigenbasis FISTA/Ridge full maps ~**0.21–0.22**; A true charts ~**0.21** at val-selected \(r{=}256\) ≫ controls.

---

## 10. Glossary (as used here)

- **mKNN:** fraction of shared kNN among paired points across two representations.
- **IDF:** \(\log((n+1)/(\mathrm{df}+1))+1\) per SAE feature from train activation frequency; multiply into codes before cosine.
- **Jaccard (at k):** overlap of active feature sets \(|A\cap B|/|A\cup B|\) after positive TopK of predictions.
- **Tyagi PCA:** local PCA about reference \(p\), uncentered second moment of \(x_j-p\).
- **Shared basis:** express model B’s SAE codes in model A’s feature coordinates via affine \(W,b\).

---

## 11. Natural next questions (open)

1. Does Ridge shared-basis transfer beyond Physics ViT↔DINO (CLIP, ConvNeXt, JWST, Legacy) once SAEs exist for both sides?
2. Is the gain mostly **feature alignment** (permutation/basis matching) vs needing a dense \(W\)?
3. Can singular charts of \(W\) define a cheaper metric than the full map without collapsing mKNN?
4. Chart **glueing** / LTSA-style alignment — still untested as a global continuous structure (local generative charts alone look flat).
5. Keep treating topological VAEs as blocked until a clean template homology appears in a subspace-localized analysis.

---

## 12. Branch / provenance

- Discussion spanned EffDim branches including geometry WIP; **current deliverable branch:** `sae-shared-basis`.
- Data/compute historically on a GPU host with a platonic-universe tree (`data_hf/`, `outputs/sae/`); configure via `PLATONIC_ROOT` (see README — no machine-specific paths).
- Upstream data: [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies); embedding export tooling: [UniverseTBD/platonic-universe](https://github.com/UniverseTBD/platonic-universe).
