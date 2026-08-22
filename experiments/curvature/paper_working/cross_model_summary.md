# Cross-model summary

Now in the manuscript as **Appendix A** (level only). Not a scale result.

## 1. Do clean cross-model outputs exist?

Yes, for **level**, not for **scale**.

The only complete dense / SAE / BSF suite on a matched holdout with k=10 and test_size=0.2 is **Physics galaxies**, 10 same-object model pairs (`physics_holdout20`). Local numbers: `paper_working/cross_model_results.csv`.

Unofficial Legacy adds two ViT-B↔DINOv3 pairs (HSC and Legacy Survey), same protocol.

Official Legacy↔HSC size ladders in the paper are **cross-survey, same architecture**. Nobody has run the complementary **same-survey, adjacent-size** ladders.

## 2. Are dense / SAE / BSF directly comparable?

**Within Physics holdout20: yes.** Same n=16384, same 80/20 split, same k=10, same pairs. SAE and BSF both use `shared_best_cosine` (eval-set direction max). That is a protocol mismatch with the current paper (predefined side1), but SAE and BSF are comparable to each other and to dense inside this suite.

mKNN rows are labeled `split: test`. Gallery is not recorded as `heldout_query_full_gallery`.

## 3. Is there enough ladder coverage to talk about scale?

**No.**

- 9/10 Physics pairs are cross-architecture at roughly similar “base/large” capacity, not a size ladder.
- The only within-family size pair is `physics_vit_vitlarge` (vit_base ↔ vit_large). One transition cannot support Spearman or \(D_R\).
- There is no ConvNeXt or DINOv2 adjacent-size same-survey chain.

A probe × scale interaction would be forced and awkward here. Lift-vs-size is not identified.

## 4. What the numbers show

### Physics (10 pairs) — Case: dense already high, structured probes still add a lot

| | dense | SAE | BSF |
|---|---:|---:|---:|
| mean mKNN@10 | 0.153 | 0.196 | 0.221 |
| mean lift vs dense | — | **+0.043** | **+0.068** |
| pairs with \(L_R>0\) | — | 10/10 | 10/10 |

Range of dense: 0.113 (ViT–CLIP) to 0.201 (DINOv3–ViT-L).
Range of \(L_{\mathrm{SAE}}\): +0.030 to +0.054.
Range of \(L_{\mathrm{BSF}}\): +0.055 to +0.086.

Single size pair: ViT-B↔ViT-L dense 0.173, SAE 0.215 (\(L{+}0.042\)), BSF 0.249 (\(L{+}0.075\)) — in line with the other nine pairs, not a scale trend.

This is **not** Case B (“SAE/BSF add little”). Dense cross-model alignment on Physics is an order of magnitude above official Legacy↔HSC dense (~0.01), and structured maps still raise it.

### Unofficial Legacy ViT↔DINOv3 — different regime

HSC: dense 0.0043, SAE 0.0053, BSF 0.0048.
Legacy Survey: dense 0.0038, SAE 0.0050, BSF 0.0041.

Near-chance dense; tiny lifts. Not a size ladder. Do not pool with Physics.

### Holdout50 extras

CosmosWeb / DESI vit↔dino exist at test_size=0.5. `desi_desi_vit_dino` has dense mKNN **1.0** — unusable. Do not mix with holdout20.

## 5. Match to the current paper?

| Paper claim | Cross-model analogue |
|---|---|
| Structured probes raise **level** | **Yes** on Physics (all 10 pairs). Different dataset and `shared_best`. |
| Extra alignment does **not** grow with size | **Not testable** with existing pairs. |
| Weak size dependence is not just domain shift | Suggestive for the *level* half only. Cannot confirm the interaction half. |

## 6. Include in the current paper?

**Yes — Appendix A, as a level check only.**

Not a main result and not a scaling panel: wrong pairing structure (cross-architecture, not adjacent size); Physics not official Legacy ladders; `shared_best`; one within-family step.

Figure: `paper/figures/cross_model_physics.png`.
