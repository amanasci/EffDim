# NeurIPS 2026 page audit (before content compression)

Style: official `neurips_2026.sty` from
`https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`
(retrieved 2026-08-20). Option: `dblblindworkshop` (anonymous workshop submission).
`\workshoptitle{Machine Learning and the Physical Sciences}` — change if the target workshop differs.

Source of this compile: `paper/main_neurips_baseline.tex` (current scientific draft, preamble only swapped).
PDF: `paper/main_neurips_baseline.pdf`.

## Counts

| Item | Page |
|---|---|
| Initial NeurIPS main-text page count | **5** (conclusion and unpaired figure land on p.5) |
| References start page | **5** (continues through p.6) |
| Appendix start page | **7** |
| Total PDF pages | **8** |

Target: pages 1–4 main manuscript; references after page 4; appendix after references.

## Section / float map (untrimmed)

| Content | Page |
|---|---|
| Title, abstract, Introduction | 1 |
| Setup (ladders, mKNN, SAE/BSF) | 2 |
| Setup unpaired methods; Results §3.1; **Figure 1** | 3 |
| §3.2; **Figure 2**; start of unpaired + discussion | 4 |
| **Figure 3** (unpaired); Conclusion; start of references | 5 |
| References continue | 6 |
| Appendix A (Physics); **Figure 4** | 7 |
| Cross-model table | 8 |

## Bottlenecks

1. **Figure 3 in the main body** — unpaired DualEncoder plot forces main text onto page 5. Move to appendix.
2. **Long Introduction** — repeated “scale ⇏ shared representation” formulations.
3. **Setup length** — four subsections; SAE TopK / IDF / BSF block details; full DualEncoder protocol.
4. **Interaction statistics** — mean, median, range, signs, Spearman, and raw 9/11–7/11–9/11 discussion.
5. **Figure 1 aspect ratio** — generated at 11.2×2.55. NeurIPS text width is 5.5in, so a naive `\linewidth` shrink makes tick labels too small. Must regenerate with a taller aspect, not shrink the PDF.

## Number check vs provenance

No inconsistencies found between `main.tex` headline numbers and `claim_provenance.md` / `probe_scale_interaction.csv` for the quantities used in the compressed draft (SAE/BSF mean lifts, \(D\) means, 6/11, Spearman +0.11/+0.13, unpaired 0.012–0.023 vs 0.0035, paper-style 8/11 \(p{=}0.113\)).
