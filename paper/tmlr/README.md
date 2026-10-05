# TMLR draft: "What a Linear Probe Reads on a Curved Representation"

**Status: first full draft, 2026-10-05.** Written from existing results only ("option 2": cheap fixes, no new
experiments). Not reviewed by a human. The ML4PS workshop paper in `paper/latex/` is untouched.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (no latexmk on this machine). 22 pages
(main text pages 1-13, references 13-14, appendices 15-22). No errors, no undefined references or citations.

## Files

- `main.tex`, `references.bib` (ML4PS bib plus 16 verified entries, each with a comment naming how it was checked)
- `tmlr.sty`, `tmlr.bst`, `math_commands.tex`, `fancyhdr.sty`: official style, fetched with curl from
  github.com/JmlrOrg/tmlr-style-file on 2026-10-05; anonymous submission mode.
- `scripts/make_tables.py` -> `tables/*.tex`. Every table file starts with a `% SOURCE:` header.
- `scripts/prop_check.py`: toy check of Lemma 1 / Proposition 1, copied unchanged from the outline review.
- `figures/make_figures.py` -> `fig_ii_spectrum`, `fig_normal_readout`, `fig_diag_scatter`.
  `fig1_intervention.pdf` and `figF_mean_curvature.pdf` are copies from `paper/latex/figures/`.

Regenerate: `.venv/bin/python paper/tmlr/scripts/make_tables.py` and `.venv/bin/python paper/tmlr/figures/make_figures.py`.

## Sources by section

| section | sources |
|---|---|
| Abstract, 1 Introduction | numbers repeated from Sections 4-8 |
| 2 Related work | Liu+23 Thm 2.4 read from arXiv 2307.02478v2; novelty report `docs/novelty/2026-10-03-framing1-novelty.md` for the must-cite list |
| 3 Setup | ML4PS `paper/latex/main.tex` Sec. 3; Sigma-metric and its derivation from the outline review (f) |
| 4 Instrument | `results/tensor-fidelity/REPORT.md`; `scaling/records/*__main_xfit.jsonl` (checks, var_explained); `results/scaling/tab_scaling_robust.tex`; notebook 02.6 outputs; trace validation numbers from ML4PS App. E |
| 5 How the embeddings bend | `results/ii-rank/*.json`; runner `12_ii_rank_run.py` docstring for the pre-stated rule |
| 6 Normal component | outline review corrected proposition (a)-(g); `notebooks/.cache/09_physics_normal_scaling_*.npz`; `scaling/arrays/*__cf.npz`; `results/scaling/tab_scaling_cf.tex`; `review-robustness/REPORT.md` (t* at alpha*) |
| 7 Residual curvature | `scaling/records/*__main_xfit.jsonl`, `*__robust.jsonl`; `review-robustness/REPORT.md` (held-out, extended controls, thinned) |
| 8 QM9 | `results/qm9/QM9_REPORT.md`; `qm9/records/*__robust.jsonl`; `qm9/arrays/*__cf.npz` |
| 9 Discussion, planned experiments | outline review (C1-C4, I1-I11, N0-N3'), test-1 design review, sanity review |
| App. A proofs | outline review derivation, re-derived while writing |
| App. B | tensor-fidelity REPORT (paper-scale table, copied by hand) |
| App. C-E | generated tables (see SOURCE headers) and the scaling report tables |
| App. F | ML4PS App. E (copied, not re-derived) |

## Discrepancies between reviews and records (records followed; each also noted in a LaTeX comment)

- The sanity review says every real-data mismatch table uses `hess_mismatch_emp`. The `xfit` rows of the
  records hold only `hess_mismatch_dec`, `hess_label` and `align_cos_tan`, so the cross-fitted "mismatch"
  columns of the ML4PS appendix (and `tab_scaling_xfit.tex`) are the decoder mismatch. The draft labels them dec.
- RR REPORT "Limitation" says the label-Hessian magnitude is "about 2x too large"; the TF relerr is unsigned. Draft
  says 29-118% relative error, sign undetermined.
- Outline review "75-85%" of normal-readout variance off-model; from the same arrays 1 - median(q^2/p^2) gives 74-85%.

## Numbers not traced to a record in this draft

- Trace validation of the decoder (cosine 0.999, ratio 1.00, rank 0.94), the resampled-surface partials
  (-0.29 to +0.26), and all of App. F Table 11 come from the ML4PS paper text, not re-checked against records.
- "Each galaxy lies in about twelve neighbourhoods" (ML4PS App. D).

## Open TODOs

1. Swiss roll: notebook 02.6 (plain AE, d = 2) FAILS the curvature check against a raw-point comparator. Run or
   cite a Swiss roll check of the exact sphere-projected configuration used here, and reconcile (CLAUDE.md).
2. State multiplicity handling (Bonferroni over 20 galaxy pairs) once, with corrected counts.
3. Trace the ML4PS-only numbers above to their records.
4. Human check of every added citation (novelty report asks for this); decide which Platonic Universe version to
   cite (`duraphe2025platonic` has 4 authors / NeurIPS ML4PS; the ICML 2026 MechInterp page lists 11).
5. Cite encoders not yet in the bib (ViT, CLIP, ConvNeXt) and the label catalogue.
6. Fig. 3 (`fig1_intervention.pdf`) is the unchanged workshop figure; its legend wording ("bending toward") should
   be redrawn to match the instrument-check reading.
7. Planned experiments (Sec. 9.1), none run: decoder-prior nulls, decoder-free II, d sweep, S-variant ridge sweep
   to plain OLS with <p,q> stored, geometry-isolating null (II from another anchor / Haar-rotated frame),
   multi-direction fixture, signed magnitude ratio, neighbourhood-residual baseline, QM9 geometry, the revised
   test-1 design and a non-learned embedding.
8. Remove the red `\todo` markers before submission.
