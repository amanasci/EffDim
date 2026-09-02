# Final audit (probe × scale narrowing)

Date: 2026-08-16.

## Interaction result (\(k{=}10\), matched holdout)

Artifact: `paper_working/probe_scale_interaction.csv`.

| | SAE \(D\) | BSF \(D\) |
|---|---|---|
| mean | \(-5.17\times 10^{-4}\) | \(-1.11\times 10^{-5}\) |
| median | \(+4.88\times 10^{-4}\) | \(+1.22\times 10^{-4}\) |
| signs | 6+ / 5− | 6+ / 5− |
| sign test | \(p{=}1\) | \(p{=}1\) |

No systematic positive probe × scale interaction. Paper wording matches the data (not “probe-invariant”).

## Paper structure

Abstract / Intro / Setup+Probes / Results 3.1–3.3 / merged Discussion+Limitations / Conclusion.
No Related Work section. No CKA appendix. No \(M_{\mathrm{adj}}\) subsection.
Fig 1 = level; Fig 2 = lift + \(D_R\); Fig 3 = unpaired triangulation.

## Compile

- `pdflatex` + `bibtex` + `pdflatex` ×2: success
- Pages: **6**
- 15/15 cite keys resolve; Cohen and Kornblith removed with the cut subsections
- No overfull boxes; no unused bib entries
- Anonymous authors retained
