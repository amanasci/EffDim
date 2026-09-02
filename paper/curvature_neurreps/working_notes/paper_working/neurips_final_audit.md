# NeurIPS 2026 final audit (compressed 4-page body)

## Counts

| Item | Value |
|---|---|
| main body pages | **1–4** (last main-text is Conclusion, p.4) |
| reference start page | **4** (continues onto p.5; does not count toward the 4-page body) |
| appendix start page | **5** (Appendix A begins after the remaining references on p.5) |
| total PDF pages | **7** |
| NeurIPS style file used | `paper/neurips_2026.sty` (official 2026-01-29; zip 2026-06-23) |
| package option | `dblblindworkshop` (anonymous) |
| workshop title command | `\workshoptitle{Machine Learning and the Physical Sciences}` — change if the target workshop differs |

PDF: `paper/main.pdf`

## Main figures

| Fig | File | Page |
|---|---|---|
| 1 | `figures/legacy_small_multiples_k10.png` | 3 |
| 2 | `figures/probe_scale_interaction.png` | 3 |

No other main-text figures. No main-text tables.

## Appendix figures

| Fig | File | Page |
|---|---|---|
| 3 | `figures/unpaired_legacy_mknn_vs_size.png` | 6 (App. B) |
| 4 | `figures/cross_model_physics.png` | 7 (App. C) |

Appendix tables: Table 1 = 11-step \(D_R\) (App. A); Table 2 = Physics cross-model mKNN (App. C).

## Checks

- Main argument does **not** spill onto page 5 (page 5 is remaining bibliography + Appendix A).
- Anonymous: style file replaces authors with “Anonymous Author(s)”. Acknowledgments section removed (`ack` environment would also be hidden).
- No custom geometry, font-size, `\baselinestretch`, or negative `\vspace` hacks. References use `\small` as permitted by the NeurIPS template.
- Citations resolve; no undefined references in the compile log.
- Appendix labels `\ref{app:details}`, `\ref{app:unpaired}`, `\ref{app:cross-model}` resolve from the main text.
- Headline numbers unchanged vs `claim_provenance.md` / `probe_scale_interaction.csv`.
- Official main-track `checklist.tex` is **not** included. That checklist is a main-track desk-reject item; this is a 4-page workshop paper. Add it after the appendix if the workshop CFP requires it.

## Warnings

- Both main figures land on page 3 (`[ht]` / `[t]`); Results prose is on page 2. Readable, but Figure 1 is not adjacent to §3.1. Not forced with `\vspace` hacks.
- Long Jha URL in the bibliography produces an underfull `\hbox` (allowed reference wrapping).
- Footer on anonymous copies is the generic NeurIPS 2026 submission line (style file; workshop name appears in camera-ready via `\workshoptitle`).
