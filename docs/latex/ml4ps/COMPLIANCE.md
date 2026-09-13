# ML4PS 2026 compliance note

Target: 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026), NeurIPS 2026 workshop, initial submission
Official sources:
- https://ml4physicalsciences.github.io/2026/ (call for papers)
- https://ml4physicalsciences.github.io/2026/guidelines.html (submission guidelines)
- https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip (official style package)
Checked: 2026-09-10

Rules recorded from the guidelines page:
- Page limit: 4 pages, excluding references. Appendices discouraged; reviewers need not read them.
- Template: NeurIPS 2026 template mandatory. "Papers submitted without using this template will be desk rejected." No other modifications to the template.
- Footer must read exactly: "Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute."
- Review: double-blind (single-blind optional for the Evaluations & Benchmarks track). Outside that track, fully anonymized: names, code links, text, figures.
- Checklist: not required for workshop submissions.
- Deadline: Saturday 12 September 2026, 23:59 AoE. Notification 10 October 2026. Workshop 11 December 2026.
- Submission via OpenReview: https://openreview.net/group?id=ML4PS/2026/Workshop
- Tracks named on the call: Research, Evaluations & Datasets, Perspectives.
- Reviewing criteria: novelty, correctness, relevance, potential impact.

How this project meets them:
- `neurips_2026.sty` is the file from the official zip, byte-unchanged. `checklist.tex` and `neurips_2026.tex` from the zip are kept alongside for reference and are not included in the build.
- The footer string is set by redefining `\@noticestring` in the preamble of `main.tex`, not by editing the style file.
- Submission mode (no `final` option): author block anonymized, line numbers on.
- No author names, affiliations, repository links, or identifying paths appear in `main.tex`.
- Main text ends before the References heading on or before page 4 (verify with the page-count check below after every edit).
- Track: Research (assumed; confirm at submission).

Page-count check after compiling:
    pdftotext main.pdf - | grep -n "^References"   # and confirm which page it lands on
    pdfinfo main.pdf | grep Pages

Page-count status (2026-09-13, after folding the cross-encoder evidence and the relative-II pilot in):
- `check.sh` builds with XeLaTeX + Liberation Serif (Times metrics) and passes when the References
  heading lands on page 4 or main text ends on page 4. Current: PASS, References begin on page 4 with
  11 reference lines on that page (about 11 lines of margin); bibliography runs onto page 5, which the
  4-pages-excluding-references rule allows (`preview_times_metric.pdf`). MUST still be confirmed on
  Overleaf with the official style (Adobe Times). First cuts if it does not fit: Figure 1 to 0.40
  textwidth; the Validation paragraph's noise clause; the Known-surface paragraph's last sentence.
- Table 1 (alignment numbers) was folded into the Section 3 text on 2026-09-13; every number is the
  same as the former table (records: 07_crossmodal_curvature.jsonl, 07.1_density_stratified_null.jsonl,
  08_* records as before).
- Table 2 and Figure 1(b) read `notebooks/.cache/09_physics_probe_facing_split.jsonl` (sha256
  09e869ff…7be65); Figure 1(a) reads `09_fixture_probe_facing.jsonl` and `09_fixture_probe_decodability*.jsonl`;
  the sphere-term identification of panel (a) is from `09_fixture_probe_facing_split.jsonl`.
- "Across encoders" paragraph: the five density-controlled associations (−0.24, 0.00, +0.27, +0.23,
  −0.19; column C_R2 vs local OOF R² of the mag_r probe; ViT-B, DINOv3, CLIP-B, ConvNeXt-B, ViT-L; d=16,
  k=2048, 512 anchors) are copied from `outputs/geometry/curvature_program_synthesis/CURVATURE_PROGRAM_SUMMARY.md`
  §6 on branch `origin/curvature-experiments` at `dabe5e2` (mirrored in
  `experiments/curvature_program/EXPERIMENT_REGISTRY.md`). The "136-coefficient fit failed a split-half
  reliability gate" limitation is that branch's `physics_cross_model_hessian_mismatch` decision
  (`label_hessian_unreliable`, gate cosine 0.20). The estimator is written up as "the quadratic-chart
  estimator" per the no-personal-language rule.
- Relative-II pilot sentence (Discussion): `notebooks/.cache/08_relative_ii_d20.jsonl` (sha256
  48b3322d…9652f), runner `08_relative_ii_run.py`, pod run 2026-09-12 (d=20, 2,048 anchors, multi-scale
  control values: II_rel +0.036/+0.048 n.s. at k=20/50, II_rel_loc −0.176/−0.239, II_rel_emp +0.131/+0.134,
  tan_resid median 0.439, alignment R² holdout 0.636). d=25 was not run to completion.
- Eq. (2) corrected 2026-09-12 (missing `c s² tr Δ` cross term); Eq. (3) is now inline in Section 4.

- Appendices (2026-09-13, after the References; allowed but "reviewers need not read"): A = decoder ablations
  (records `09_physics_probe_facing_split_seed1.jsonl`, `_seed2.jsonl`, `_w400.jsonl`, pod runs 2026-09-13,
  d=16, stored in `notebooks/.cache/`); B = cross-fitted Hessian and weak-ridge sensitivity (`_xfit.jsonl`,
  `_alpha1.jsonl`); C = relative-II pilot rows (`08_relative_ii_d20.jsonl`, `_d25_seed0.jsonl`, `_d20_seed1.jsonl`).
  All three tables are generated by the scratch script `appendix_gen.py` and spliced between the
  `% BEGIN/END APPENDIX AUTOGEN` markers in `main.tex`; never edit the numbers by hand.
- Page gate after this pass: main text ends on page 4 with essentially zero margin (References start on
  page 5). Overleaf/Times confirmation is mandatory; cut list unchanged.

Open items before submission:
- Author affiliation and names are only needed for the camera-ready `final` build.
- Seven uncited bibliography entries still carry a `note` field asking for author lists (harmless with unsrtnat, which prints cited entries only); `lee2023curvature` and `acosta2022templates` were verified against arXiv 2026-09-12.
- Confirm the track choice on OpenReview.
