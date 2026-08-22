# Figure inventory

## Main paper

| Fig | File | Job |
|---|---|---|
| 1 | `paper/figures/legacy_small_multiples_k10.png` | **Magnitude.** Matched held-out dense / SAE side1 / BSF. Regenerated for NeurIPS 5.5in column (`plot_probe_scale_figures.py`). |
| 2 | `paper/figures/probe_scale_interaction.png` | **Interaction.** (a) \(L_R\) vs \(\log_{10}P\); (b) 11 adjacent \(D_R\). |

## Appendix

| Fig / table | File | Job |
|---|---|---|
| Fig 3 | `paper/figures/unpaired_legacy_mknn_vs_size.png` | Unpaired DualEncoder (Appendix B). |
| Fig 4 | `paper/figures/cross_model_physics.png` | Physics cross-model level (Appendix C). Regen: `plot_cross_model.py`. |
| Table 1 | `paper_working/probe_scale_interaction.csv` | All 11 \(D_R\) steps. |
| Table 2 | `paper_working/cross_model_results.csv` | Physics holdout20 mKNN@10. |

## Regeneration

`paper_working/plot_probe_scale_figures.py` rebuilds Figs 1–2.
`paper_working/plot_cross_model.py` rebuilds Fig 4.
