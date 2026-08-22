# Alignment-control audit

Frozen baseline copy: host `outputs/paper_alignment_controls/frozen_baseline/mknn_by_size.parquet` (original size-scaling file untouched).
New namespace: `outputs/paper_alignment_controls/` on the Ubuntu host; local mirrors under `paper_working/`.

## Protocol

- Same 16 official Legacy↔HSC rungs, n=16384, `train_test_split(..., test_size=0.2, random_state=0)` → 13107 / 3277.
- Mapping direction fixed: Legacy → HSC (col2 → col1). No eval-set direction shopping.
- Ridge: `StandardScaler` on X and Y, `Ridge(alpha=1.0, fit_intercept=True)` — same recipe as the SAE/BSF shared-basis maps.
- Dense: l2-normalised cosine, no map.
- Dense+Ridge: Ridge on **raw** dense embeddings, then cosine mKNN.
- SAE+Ridge: TopK SAE codes, same Ridge, train-only IDF on the HSC dictionary (`shared_side1_basis_idf`).
- BSF+Ridge: full signed BSF codes, same Ridge, cosine (`shared_side1_basis_cosine`).
- Evaluation gallery = the 3277 test IDs only (self-excluded). Chance = 10/3276 ≈ 0.00305.
- Do not compare these numbers to the old full-gallery scores without labelling the gallery change.

## Q1. Does supervised Dense+Ridge account for the SAE/BSF lift?

**Yes, and then some.**

Mean lifts over dense (test-only gallery, k=10):

| Method | Mean mKNN | Mean lift vs dense |
| --- | ---: | ---: |
| Dense | 0.0218 | — |
| SAE+Ridge | 0.0331 | +0.0113 |
| BSF+Ridge | 0.0354 | +0.0136 |
| Dense+Ridge | 0.0415 | **+0.0197** |

Dense+Ridge is positive on **16/16** rungs and larger than the SAE/BSF lifts. The original “sparse lift” is smaller than an equally supervised dense alignment.

## Q2. Do SAE/BSF retain positive lift over an equally supervised dense alignment?

**No, not systematically.**

Residual \(S_R = M_R - M_{\mathrm{dense+Ridge}}\):

- SAE: mean **−0.0085**, 1/16 positive (only I-JEPA huge). Family-clustered bootstrap 95% CI **[−0.0120, −0.0043]**.
- BSF: mean **−0.0062**, 4/16 positive. Family-clustered bootstrap 95% CI **[−0.0104, −0.0007]**.

Both CIs lie below zero. Sparse/block codes do **not** add correspondence beyond Dense+Ridge on this protocol.

## Q3. Does the result survive a gallery containing only unseen mapping-test objects?

**Yes — these numbers are that gallery.** Train objects cannot appear as neighbours. Chance is 10/3276, not 10/16383. The Dense+Ridge dominance is not an artefact of retrieving mapped training objects.

(The old paper scores used held-out queries / full gallery, so they are a different functional. The qualitative SAE/BSF-over-dense lift existed there too; the missing control is Dense+Ridge, not the gallery change alone.)

## Q4. Does correct Legacy↔HSC correspondence matter to the fitted map?

**Yes.** B=20 correspondence shuffles (permute train pairing, refit, evaluate on true test pairs / test-only gallery). Chance-neighbour floor is ~0.003; this null is higher because a map is still fit.

| Method | Mean real mKNN | Mean shuffle-refit | Real > all 20 nulls |
| --- | ---: | ---: | ---: |
| Dense+Ridge | 0.0415 | 0.0047 | 16/16 (p=1/21) |
| SAE+Ridge | 0.0331 | 0.0112 | 16/16 |
| BSF+Ridge | 0.0354 | 0.0047 | 16/16 |

The maps are using true object correspondence, not just matching marginals. That does **not** restore a sparse-representation advantage: Dense+Ridge still has the highest real score.

## Q5. Is there a consistent positive representation×scale interaction after Dense+Ridge?

**No.**

Adjacent \(D^{\mathrm{struct}}_{R} = \Delta M_R - \Delta M_{\mathrm{dense+Ridge}}\) (11 steps):

- SAE: mean **−0.0021**, 4/11 positive
- BSF: mean **−0.0029**, 6/11 positive

Family slopes of mKNN vs \(\log_{10}P\) (I-JEPA is two-point; do not overread):

| Family | n | β dense | β Dense+Ridge | β SAE | β BSF | Δβ SAE | Δβ BSF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astropt | 3 | +0.0050 | +0.0101 | +0.0064 | −0.0010 | −0.0037 | −0.0111 |
| convnext | 4 | +0.0028 | +0.0057 | +0.0029 | +0.0063 | −0.0028 | +0.0007 |
| dinov2 | 4 | +0.0000 | +0.0032 | +0.0039 | +0.0074 | +0.0007 | +0.0042 |
| vit | 3 | +0.0130 | +0.0232 | +0.0160 | +0.0146 | −0.0072 | −0.0086 |
| ijepa | 2 | +0.0662 | +0.0806 | +0.0351 | +0.0140 | −0.0455 | −0.0666 |

After controlling for supervised alignment, sparse structure does **not** consistently steepen the size relationship. Language: **no consistent positive representation × scale interaction**.

## Q6. Does a generic PCA/Ridge bottleneck explain the residual lift?

There is no positive residual to explain. PCA+Ridge (ranks 32/64/128/256, rank chosen on an inner **train** split, never on test mKNN) selected **256** on every rung. Mean test mKNN@10 = **0.0496**, above Dense+Ridge (0.0415) and well above SAE/BSF.

So a generic low-dimensional bottleneck plus Ridge is at least as strong as — and here stronger than — SAE/BSF. Do **not** call PCA rank 256 “dimension-matched” to SAE TopK 18–23.

Ill-conditioned Ridge warnings appeared on a few full-dim dense maps (DINOv2 giant, ViT huge, I-JEPA). That is another reason not to treat full-dim Dense+Ridge as a unique optimum; the qualitative ranking is unchanged.

## Decision (do not edit the paper yet)

### Outcome 1

> Dense+Ridge explains the lift; sparse representation attribution does not survive.

SAE/BSF still beat **unmapped** dense, but that gap is smaller than the gap from fitting the same supervised cross-survey map on dense embeddings.

### Scaling

> After controlling for supervised alignment, representation×scale interaction is heterogeneous / near zero (if anything slightly negative). Full write-up: `paper_working/post_control_scaling_report.md`. SAE Δβ family-bootstrap 95% CI [−0.029, −0.0016]; BSF [−0.042, +0.0004]. No consistent positive representation×scale interaction.

### What this does to the paper if we later revise

The comparison the current draft reports is confounded: SAE/BSF include a fitted Legacy→HSC map; dense does not. Under a matched alignment control, the story is **learned cross-survey alignment vs raw dense geometry**, not sparse representation structure. The unpaired DualEncoder was not rerun.

Manuscript not edited in this pass.
