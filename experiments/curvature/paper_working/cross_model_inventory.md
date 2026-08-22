# Cross-model inventory

Audit of **completed** same-object, different-model mKNN experiments.
No new runs. Missing fields are left blank rather than inferred.

Primary question for the paper: is there a same-survey analogue of the official Legacy↔HSC **size ladders** with dense / SAE / BSF on a matched holdout?

**Short answer:** no. Existing cross-model results are mostly **same-scale, cross-architecture** pairs on Physics galaxies, plus two unofficial Legacy ViT↔DINOv3 pairs. They support a **level** comparison, not a probe × scale interaction.

---

## A. Official UniverseTBD size ladders (current paper)

| Field | Value |
|---|---|
| Config | `experiments/universetbd_shared_basis_mknn/official_legacy_pairs.yaml` |
| Kind | **cross_survey only** (every rung is Legacy↔HSC, same architecture size) |
| Completed outputs | `outputs/universetbd_shared_basis_mknn_ks/size_scaling{,_bsf}/` |
| Same-survey adjacent-size pairs (nano↔tiny, …) | **not present** |

These embeddings exist as separate per-size parquets, but no completed mKNN job pairs two sizes on one survey.

---

## B. Physics holdout20 — strongest existing cross-model suite

Host: `outputs/sae_shared_basis/pipeline_isomap_{sae,vanilla_bsf}_shared_mknn_physics_holdout20/`
Local summary: `experiments/SAE-shared-basis/artifacts/physics_holdout20_bsf_vs_sae/`

| Field | Recorded value |
|---|---|
| Survey / modality | Physics / Smith42 galaxies (single imaging domain) |
| Object IDs / n | n=16384 subsample; n_train=13107; n_test=3277 |
| Train/test | test_size=0.2, seed=0; Ridge on train |
| k | **10** (`config.mknn_k`; `results.json` meta `k: 10`) |
| mKNN split field | `split: test` (not named `heldout_query_full_gallery`) |
| Mapping | `shared_best_cosine` (max of two directions) |
| SAE | TopK, F=2048; per-pair `sae_k` in eval dirname (k17–k22) |
| BSF | vanilla, n_blocks=512, block_dim=4 |
| Held out? | Yes for the Ridge map (train 80%). Direction is **shared_best**, i.e. eval-set selection |
| Pairs | 10, all `kind: cross_model` |

Pairs (model A ↔ model B), from `compatible_pairs.yaml` columns:

| pair | model A | model B | within-family size? |
|---|---|---|---|
| physics_vit_dino | vit_base | dinov3_vitb16 | no |
| physics_vit_clip | vit_base | clip_base | no |
| physics_vit_convnext | vit_base | convnext_base | no |
| physics_vit_vitlarge | vit_base | vit_large | **yes (only one)** |
| physics_dino_clip | dinov3_vitb16 | clip_base | no |
| physics_dino_convnext | dinov3_vitb16 | convnext_base | no |
| physics_dino_vitlarge | dinov3_vitb16 | vit_large | no |
| physics_clip_convnext | clip_base | convnext_base | no |
| physics_clip_vitlarge | clip_base | vit_large | no |
| physics_convnext_vitlarge | convnext_base | vit_large | no |

Parameter counts: **not stored** in the comparison artifacts. Do not copy official-ladder P from a different embedding release.

Dense / SAE / BSF scores: `paper_working/cross_model_results.csv` (physics_holdout20 rows).

---

## C. Unofficial Legacy holdout20 — ViT↔DINOv3 only

Local: `experiments/SAE-shared-basis/artifacts/legacy_holdout20_bsf_vs_sae/`
Host evals: `pipeline_isomap_*_legacy_holdout20`

| Field | Recorded value |
|---|---|
| Pairs | `legacy_hsc_vit_dinov3` (HSC columns); `legacy_ls_vit_dinov3` (Legacy Survey columns) |
| Models | vit_base ↔ dinov3_vitb16 (not the official DINOv2 size ladder) |
| n / split / k | 16384 / 0.2 / 10 (same pipeline as B) |
| Mapping | shared_best_cosine |
| Size ladder? | no |

---

## D. Holdout50 mixed-survey comparison (sensitivity, different split)

Host: `outputs/sae_shared_basis/holdout50_bsf_sae_comparison/comparison.json`
Protocol string: test_size=0.5; vanilla BSF + sae_init BSF + TopK SAE; surveys physics, desi, legacy, cosmosweb.

Additional cross-model pairs (not in the local holdout20 artifacts):

| pair | survey | notes |
|---|---|---|
| cosmosweb_hsc_vit_dino | cosmosweb | vit↔dinov3 on HSC |
| cosmosweb_jwst_vit_dino | cosmosweb | vit↔dinov3 on JWST |
| desi_hsc_vit_dino | desi | vit↔dino on HSC columns of DESI files |
| desi_desi_vit_dino | desi | **dense mKNN = 1.0** in this file — treat as invalid / unusable |

Physics 10 pairs are repeated here at test_size=0.5 (scores differ; not interchangeable with holdout20).

k for holdout50: not copied into `comparison.json`. Pipeline default in sibling SAE jobs is mknn_k=10, but **not verified in that comparison file**.

---

## E. Configured but not a completed size-ladder analogue

`experiments/SAE-shared-basis/compatible_pairs.yaml` also defines JWST/DESI/CosmosWeb vit↔dino same-survey pairs. Completed scores for some of these exist only in holdout50 (D), not as official size rungs.

JWST paper-table2 (`artifacts/paper_table2_official/`) is **cross-survey dense only**.

---

## F. Explicitly absent

- ConvNeXt nano↔tiny↔base↔large on one survey
- DINOv2 small↔base↔large↔giant on one survey
- ViT base↔large↔huge beyond the single Physics vit_base↔vit_large pair
- Matched side1 (no shared_best) cross-model scores
- Probe × scale interaction \(D_R\) for adjacent same-family pairs
