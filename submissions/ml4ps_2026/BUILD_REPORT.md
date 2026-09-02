# BUILD_REPORT

## Command

From `submissions/ml4ps_2026/`:

```bash
python3 figures/make_fig2.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Figure 2 is regenerated from frozen CSV/JSON under `paper/curvature_neurreps/audit_outputs/` (read-only). Figure 1 is the existing vector PDF in `figures/fig1_global.pdf` (copy of the NeurReps confirmatory curvature–error figure).

## Output

- PDF: `submissions/ml4ps_2026/main.pdf`
- Last successful compile: 2026-08-31 (`latexmk`; 4 pages, letter 612×792 pt)
- Body: pages 1–4; **References begin on page 4** after the generative-AI paragraph. No appendix.
- `neurips_2026.sty`: unmodified (diff-identical to `paper/neurips_2026.sty`). Footer override is only in `main.tex`.

## Page map (after reflow)

| Page | Content |
|---|---|
| 1 | Title, anonymous authors, abstract, introduction, required footer |
| 2 | Method; Results through the start of the anisotropic-prior paragraph |
| 3 | Figure 1 (confirmatory global); Figure 2 (quadratic mechanism); remainder of geometry/alignment results |
| 4 | Null calibration and LPA secondary note; Discussion; Reproducibility; Generative AI use; References (5 entries) |

## Layout checks

| Check | Result |
|---|---|
| Overfull boxes | none in `main.log` |
| Undefined citations / references | none |
| Missing glyphs | none observed |
| Footer | exact ML4PS string, page 1 only |
| Template geometry / fonts / spacing | not altered |
| Line numbers | NeurIPS 2026 default (submission), left margin |
| Column layout | NeurIPS default single-column article (not `[final]`) |
| PDF Author / Creator / Producer | empty |

Pages were rasterized with `pdftoppm -png -r 140` and inspected. Both figures are on page 3 at `0.90\linewidth`; axis labels and the \(A_B=2.43\) / 94% annotations remain readable. No overlapping floats. Figure 2 appears with the quadratic results rather than after the discussion.

## Intentionally omitted

- Method schematic (no accurate existing vector asset).
- Page-preview PNGs are not stored in the submission tree.
- Anonymous code URL.
- Body numerical table (truncation fractions are in prose and Fig. 2d; repeating them in a table would duplicate).

## Residual notes

- Fig. 2b axes are controlled rank residuals, not raw \(K_H\) or \(\Delta_Q\).
- Underfull `\vbox` on page 1 is the usual NeurIPS title-block warning, not a margin violation.
- Two figures share page 3; they are smaller than a single-figure page but still above unreadability at preprint width.
