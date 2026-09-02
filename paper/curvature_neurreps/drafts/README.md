# Workshop paper workspace

## Layout

```text
paper/
  main.tex              # NeurIPS 2026 workshop draft (4-page body)
  neurips_2026.sty      # official 2026 style
  references.bib
  figures/
paper_working/
  neurips_page_audit.md
  neurips_final_audit.md
  claim_provenance.md
  ...
```

## Format

Official NeurIPS 2026 style, option `dblblindworkshop`.
Main body ≤ 4 pages; references and appendix do not count.

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Change `\workshoptitle{...}` if the target workshop is not ML4PS.

## Core message (locked)

Shared SAE/BSF bases substantially raise Legacy↔HSC correspondence; the extra correspondence does not systematically grow with model size. Scale does not necessarily imply a more shared representation.

## Method roles

1. Shared-basis SAE — paired feature-level probe (main)
2. Shared BSF — paired block/subspace probe (main)
3. Unpaired DualEncoder — triangulation (main paragraph; figure in Appendix B)
4. Physics cross-model mKNN — level check only (Appendix C)

Physics decoder-metric / tangent geometry remain **out of this paper**.

## Citations

`references.bib` is verified against publisher/arXiv/DOI pages (see `citation_inventory.md`).
No repository artifacts are cited as literature.
