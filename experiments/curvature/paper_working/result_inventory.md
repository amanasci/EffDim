# Result inventory — Legacy↔HSC workshop paper

Audited: 2026-08-16. Central object: **probe × scale interaction** on the matched holdout.

## Conditions

| # | Method | Role |
|---|---|---|
| 1 | Dense | Ambient baseline; matched holdout for \(L_R\) and \(D_R\) |
| 2 | Shared SAE | Structured paired probe (predefined HSC-basis direction) |
| 3 | Shared BSF | Structured paired probe |
| 4 | Unpaired DualEncoder | Qualitative triangulation only (ConvNeXt, DINOv2) |

---

## Probe × scale interaction (\(k{=}10\), matched holdout)

Artifact: `paper_working/probe_scale_interaction.csv`.
Sources: `data/sae_k10.csv`, `data/bsf_k10.csv` (side1; `dense_cosine_heldout`).

| Probe | mean \(D\) | median \(D\) | min | max | \(D{>}0\) | \(D{<}0\) |
|---|---|---|---|---|---|---|
| SAE | \(-5.2\times 10^{-4}\) | \(+4.9\times 10^{-4}\) | \(-4.6\times 10^{-3}\) | \(+1.2\times 10^{-3}\) | 6 | 5 |
| BSF | \(-1.1\times 10^{-5}\) | \(+1.2\times 10^{-4}\) | \(-6.3\times 10^{-3}\) | \(+5.2\times 10^{-3}\) | 6 | 5 |

No systematic positive interaction. Sign test \(p{=}1\).

Matched lifts @k=10: SAE \(+0.00562\) (16/16 +); BSF \(+0.00756\) (16/16 +).
Spearman \(L\) vs \(\log_{10}P\): \(+0.11\) / \(+0.13\).

Raw adjacent signs (demoted): held-out dense 9/11; SAE 7/11; BSF 9/11.
Paper-style dense (replication only): 8/11, \(p_1{=}0.113\).

---

## Unpaired (shape only)

ConvNeXt mKNN@10: 0.01574 → 0.01153 → 0.02263 → 0.01212.
DINOv2: 0.02202 → 0.01895 → 0.01963 → 0.01857.
Adjacent 2/6. Z=256, hidden=512, 80 epochs, 2 seeds, disjoint 5500/5500, eval 2884.

---

## Out of scope

Physics / decoder-metric / tangent / CKA appendix / chance-adjusted \(M_{\mathrm{adj}}\) tables / mixing paper-style dense into \(D_R\).
