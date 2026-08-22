# Claim provenance

Headline quantitative claims for the Legacy↔HSC workshop draft.
Host: `/home/angus/platonic-universe/outputs/…`
Local mirrors: `paper_working/data/`, `paper/figures/`.

External claims: `citation_inventory.md`. Novelty: `novelty_audit.md`. Levels: `claim_hierarchy.md`.
Final cleanup audit: `final_audit.md`.

---

## Probe × scale interaction (central result)

**Definition:** \(D_R=\Delta M_R-\Delta M_{\mathrm{dense}}=L_R(P_{j+1})-L_R(P_j)\) on the matched holdout.
**Source scores:** `paper_working/data/sae_k10.csv` (`dense_cosine_heldout`, `shared_side1_basis_idf`) and `bsf_k10.csv` (`shared_side1_basis_cosine`). Exact stored mKNN; not PDF-rounded.
**Artifact:** `paper_working/probe_scale_interaction.csv` (11 rows, \(k{=}10\)).

**SAE \(D\) @k=10:** mean \(-5.166\times 10^{-4}\), median \(+4.882\times 10^{-4}\), min \(-4.577\times 10^{-3}\), max \(+1.221\times 10^{-3}\), IQR \([-1.755,+0.946]\times 10^{-3}\). Signs \(6\) / \(5\). Sign test two-sided \(p{=}1\).
**BSF \(D\) @k=10:** mean \(-1.11\times 10^{-5}\), median \(+1.22\times 10^{-4}\), min \(-6.347\times 10^{-3}\), max \(+5.249\times 10^{-3}\), IQR \([-1.343,+1.709]\times 10^{-3}\). Signs \(6\) / \(5\). Sign test two-sided \(p{=}1\).
**Mean \(|D|\):** SAE \(0.00151\); BSF \(0.00217\). Mean \(|\Delta M_{\mathrm{dense}}|{=}0.00255\).

Family-block bootstrap 95% CI for mean \(D\): SAE \([-0.00145,+0.00015]\); BSF \([-0.00185,+0.00160]\) (both include 0). Not used as a headline.

k=50 matched side1 interaction was not written this pass (host parquet exists; not required for the primary table).

---

## Setup

**Claim:** Official Legacy↔HSC ladders; five families; n=16384; primary k=10.  
**Source:** `relational_geometry_size_scaling/model_size_manifest.csv`; `config_analyze.json`.

**Claim:** Parameter counts (M): AstroPT 15/95/850; ConvNeXt 15/28/89/198; DINOv2 22/86/300/1100; ViT 86/307/632; IJEPA 630/1000.  
**Source:** YAML `approx_params_m`.

**Claim:** SAE/BSF mapping direction is predefined side1 (Legacy mapped into HSC / col1 basis).  
**Source:** `official_legacy_pairs.yaml` (`col1=*_hsc`, `col2=*_legacysurvey`); methods `shared_side1_basis_idf` / `shared_side1_basis_cosine`.

---

## Dense (paper-style PRH baseline)

**Claim:** Dense mKNN@10 ≈ 0.0072–0.0178 (mean ≈ 0.0115).  
**Source:** `family_scaling_dense.csv`, k=10, `dense_cosine` / `paper_full_catalog`.

**Claim:** Adjacent signs 8/11, \(p_1=0.113\), mean Δ=+0.00217, at both k=10 and k=50.  
**Source:** `binomial_scaling_tests.csv`, k=10/50, `scope==all_families`.

**Claim:** Chance ≈ k/(n−1) ≈ 6.1×10⁻⁴ for k=10, n=16384.  
**Source:** analytic (Gröger).

**Claim:** First→last dense Δ: I-JEPA +0.00738; DINOv2 +0.00107.  
**Source:** `family_correlations.csv`.

**Claim:** Leave-one-family-out dense@10 stays 6–7 / 8–10.  
**Source:** `leave_one_family_out.csv`.

This protocol is **only** the UniverseTBD replication. Do not put it in the same sign table as SAE/BSF.

---

## Matched held-out dense / SAE / BSF (representation-controlled)

Same query IDs, gallery, k, self-exclusion, object subset, survey/model pairing.  
\(n_{\mathrm{test}}=3277\). Protocol `heldout_query_full_gallery`.

**Artifacts:**
- Host: `universetbd_shared_basis_mknn_ks/size_scaling/mknn_by_size.parquet` (dense + SAE)
- Host: `universetbd_shared_basis_mknn_ks/size_scaling_bsf/mknn_by_size.parquet` (BSF)
- Local: `paper_working/data/sae_k10.csv`, `bsf_k10.csv`
- Local: `paper_working/data/dense_heldout_k10.csv`
- Local: `paper_working/data/sae_side1_matched_lifts.csv`, `bsf_side1_matched_lifts.csv`
- Local: `paper_working/data/matched_heldout_k10_panel.csv`

### Held-out dense mKNN@10

**Claim:** Range 0.00659–0.01758 (mean 0.01098).  
**Claim:** Adjacent signs **9/11**, \(p_1=0.0327\), two-sided 0.065, mean Δ=+0.00191.  
**Claim:** First→last: AstroPT +0.00421; ConvNeXt +0.00320; DINOv2 −0.00055; I-JEPA +0.00684; ViT +0.00735.  
**Claim:** k=20 also 9/11 (\(p_1=0.033\)); k=50 is 8/11 (\(p_1=0.113\)).  
**Source:** `dense_cosine_heldout` in the parquets / `sae_k10.csv`; host binomial 2026-08-16.

### SAE (predefined side1 IDF)

**Claim:** Method `shared_side1_basis_idf` (not `shared_best_basis_idf`).  
**Claim:** Adjacent signs 7/11, \(p_1=0.274\), mean Δ=+0.00140 (k=10); k=20 7/11; k=50 6/11.  
**Claim:** vs held-out dense @k=10: mean lift **+0.00562** (min +0.00253, max +0.01269); 16/16 positive.  
**Claim:** Lift vs log₁₀P Spearman **+0.113** (t≈0.42, df=14, p≈0.68).  
**Claim (examples @k=10):** ConvNeXt nano 0.00925 → 0.01492; I-JEPA giant 0.01758 → 0.02737.  
**Source:** `sae_side1_matched_lifts.csv`.

Direction shopping removed: previously `shared_best == side1` on 12/16 rungs; mean (best−side1)=\(6.7\times 10^{-5}\). We now report side1 only.

### BSF (predefined side1 cosine)

**Claim:** Method `shared_side1_basis_cosine` (equals `shared_best` on all 16 rungs).  
**Claim:** Adjacent signs **9/11**, \(p_1=0.033\), two-sided 0.065, mean Δ=+0.00190 at k=10, 20, and 50.  
**Claim:** vs held-out dense @k=10: mean lift **+0.00756** (min +0.00171, max +0.01678); 16/16 positive.  
**Claim:** Lift vs log₁₀P Spearman **+0.127** (t≈0.48, df=14, p≈0.64).  
**Claim (examples @k=10):** ConvNeXt nano 0.00925 → 0.01608; I-JEPA giant 0.01758 → 0.03207.  
**Source:** `bsf_side1_matched_lifts.csv`.

**Do not use** `representation_lift_scaling.csv` for lifts (that file subtracts paper-style dense).

---

## SAE checkpoint convention

**Final convention:** report the existing ladder; do not silently mix undocumented configs.

| Field | Status |
|---|---|
| Dictionary width F | **homogeneous**: 2048 on all 16 rungs |
| Seed | **homogeneous**: 0 on all 16 rungs |
| IDF | train-only IDF, both surveys of a pair |
| TopK k | **heterogeneous**: k∈{18,19,20,21,22,23}; same tag on both surveys of a pair |
| Direction | predefined side1 (HSC basis) |

Per-rung tags (both surveys):  
ConvNeXt nano/tiny/base `F2048_k21_seed0`; large `F2048_k20_seed0`.  
AstroPT 15m `F2048_k22_seed0`; 95m/850m `F2048_k19_seed0`.  
DINOv2 small `k20`; base `k22`; large `k21`; giant `k22`.  
ViT base `k20`; large/huge `k23`.  
I-JEPA huge/giant `k18`.  

**Source:** `sae_k10.csv` columns `sae1_tag`, `sae2_tag`.  
Homogenizing TopK would require re-encoding; not done for this draft.

---

## Legacy unpaired size-ladder analysis

**Source:** `paper_working/data/unpaired/family_scaling_unpaired.csv`  
(`relational_geometry_size_scaling/unpaired/`).

**Claim:** Families ConvNeXt + DINOv2; Z=256; hidden=512; 80 epochs; 2 seeds; n=16384; disjoint A/B train 5500/5500; eval 2884.  
**Source:** script defaults + `build_split`; CSV `Z`, `hidden`, `n`.

**Claim:** Losses recon + RBF-MMD + cycle + Gram (weights 1,1,1,1).  
**Source:** `run_unpaired_universal_geometry.py`.

**Claim:** mKNN implementation slices `neighbors[:, :k]`; all unpaired mKNN ∈ [0.006, 0.104] ⊂ [0,1].  
**Source:** `mknn_score`; CSV min/max.

**Claim:** unpaired_dense mKNN@10 — see `result_inventory.md`. Adjacent signs **2/6**. No binomial.  
**Claim:** CKA unpaired_dense: ConvNeXt 0.208 / 0.264 / 0.318 / 0.171; DINOv2 0.330 / 0.304 / 0.287 / 0.248.  
**Claim:** top-1 identity ≲ 0.0024. Chance mKNN@10 on eval ≈ 10/2883 ≈ 0.0035.

**Do not cite** Physics `smoke_mknn_fixed` numbers in this paper.

---

## Appendix A: Physics cross-model level (not scale)

**Source:** `paper_working/cross_model_results.csv` (`suite==physics_holdout20`); original `experiments/SAE-shared-basis/artifacts/physics_holdout20_bsf_vs_sae/comparison.json`.

**Claim:** 10 same-object pairs; \(n{=}16384\); \(n_{\mathrm{test}}{=}3277\); \(k{=}10\); mapping `shared_best_cosine`.
**Claim:** dense mean 0.153 (range 0.113--0.201); SAE mean 0.196, lift +0.043 (range +0.030--+0.054); BSF mean 0.221, lift +0.068 (range +0.055--+0.086); 10/10 \(L_R>0\).
**Claim:** only within-family size pair is ViT-B↔ViT-L (dense 0.173, SAE 0.215, BSF 0.249).
**Claim:** unofficial Legacy ViT-B↔DINOv3 (HSC/LS) dense ≈0.004, lifts ≲0.0013; not pooled.

Do not estimate \(D_R\) or Spearman vs \(\log P\) from these pairs.
