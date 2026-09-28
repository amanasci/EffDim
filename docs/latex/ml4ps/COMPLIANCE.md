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
- Deadline: Saturday 19 September 2026, 23:59 AoE (extended by one week from 12 September; page re-checked 2026-09-13). Notification 10 October 2026. Workshop 11 December 2026.
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

Page-count status (2026-09-18, Table 1 trimmed): Table 1 now carries only the mismatch and alignment partials (four
numeric columns); the shape and sphere columns (both d) moved into Table F beside ‖H^S‖, Appendix F retitled "the shape and
sphere terms, and mean curvature" with the sphere-term and shape-term sentences from the Galaxies paragraph. Galaxies
paragraph rewritten around Hess_M y estimation, the mismatch/alignment result, and a one-sentence pointer for the two
terms. Appendix A intro now says "Tables 1 and F" (generator updated). `check.sh` PASS; References begin on page 4 with
5 reference lines there.

Page-count status (2026-09-18, Section 3 restructure): Section 3 is now a run-in "Notation." paragraph plus three numbered
facts (readout gradient/Hessian with K := <w_N,II>; the sphere split K = K_S + K_sph as a display equation with underbraces;
the residual expansion), followed by the helps-iff condition, the shape-term condition with R, and the density remark.
Section 4's intervention paragraph lost its definitions; the related-work parenthetical became its own paragraph
"Relation to prior work." Appendix E now refers the tensor-level condition to Section 3 (generator updated). `check.sh`
PASS; References begin on page 4 with only 4 reference lines there (tight; first cuts if Overleaf overflows: the
"Relation to prior work" citations list, the Validation numbers, Figure 1 height).

Page-count status (2026-09-18, headline figure): Figure 1 is now `figures/fig1_intervention.pdf` from
`figures/make_fig_intervention.py` (Appendix E records `09_physics_normal_scaling_*.npz`, variants `S_model` and
`random_qmatched`): median over anchors of the change in local R² vs t in {-1..2}, one panel per label, one blue line per
run, grey dashed random control. Medians at t=±1 reproduce Appendix E's ΔR² columns exactly. The shape/sphere single-panel
plot (`fig1_probe_facing.pdf`) is still generated but no longer included. `check.sh` PASS; References begin on page 4
with 10 reference lines there.

Page-count status (2026-09-18, line-by-line review with the user): abstract mean-curvature sentence cut; Milnor 1963 and
Chern & Lashof 1957 cited for the height-function Hessian after Eq. (1); mean curvature moved to appendix-only (new
hand-written Appendix F after the AUTOGEN markers: H definition, radial split, validation-surface definition and
numbers, known-surface resampling, ‖H^S‖ table, cross-encoder trace signs, closing sentence). Main text keeps only the
II estimate, a one-sentence validation pointer, and Eq. (1) without the Laplacian identity. Figure 1 is now the
single-panel galaxies plot (shape and sphere terms) at 0.42 textwidth from `make_fig1.py`; the former two-panel figure
is `figures/figF_mean_curvature.pdf` in Appendix F. Table 1 lost its two ‖H^S‖ columns (they are Table F). `check.sh`
PASS; References begin on page 4.

Page-count status (2026-09-18, second reframe pass, fresh-context review + colleague feedback: PRH / Platonic Universe /
cross-survey MKNN application REMOVED entirely (former Section 5, contribution (iv), abstract clause, H_raw/H_rad macros,
Appendix C relative-II pilot; `appendix_gen.py` keeps the Appendix C code behind `INCLUDE_RELATIVE_II = False`); title
shortened to "Linear Probes on Curved Latent Spaces"; abstract and Introduction lead with per-object reliability of
probe-based inference and which quantities an embedding delivers linearly; Discussion stellar-mass sentence scoped to
"cross-anchor variation not explained by the mismatch, within-anchor bending still helps"; Limitations "one survey pair"
replaced; Figure 1 legend labels shortened (right column was clipped in the PDF); Appendix E now `\ref{sec:real}` instead
of a hard-coded "Section 5"). Data citation: embeddings from the Platonic Universe release (`UniverseTBD/pu-embeddings`),
labels from the AstroPT galaxy set, cited as data sources only. `check.sh` PASS; References begin on page 4 with 14
reference lines there.

Page-count status (2026-09-18, reframe: PRH dropped as the frame; new title, abstract without numbers, introduction on
probes and reliability, cross-survey result moved to a short application section after the galaxies results, the
two-embedding alignment expansion reduced to an Appendix C pointer, Discussion on the mismatch tensor as a per-object
diagnostic; Figure 1 back to 0.22 textwidth): `check.sh` PASS; Figure 1 settled at 0.20 textwidth; References begin on page 4 with 11 reference lines there.

Page-count status (2026-09-16, freeze pass: "under first-order tangent matching" in abstract and Discussion, |K|_g^2, g-subscripts in known-surface paragraph and Table 1, k=30 density definition restored, stellar-mass sentence scoped; sixteen trims, Figure 1 at 0.17 textwidth): `check.sh` PASS; References begin on page 4 with 13 reference lines there.

Page-count status (2026-09-16, geometry-audit follow-up: Appendix C chart-dependence justification, g-subscripts on every tensor norm, w_S in Eq. (split) and Table 1, "large majority" for the help result, stellar-mass sentence scoped to mismatch/alignment; nine trims, Figure 1 at 0.19 textwidth): `check.sh` PASS; References begin on page 4 with 14 reference lines there.

Page-count status (2026-09-16, geometry-audit pass: immersion condition, unnormalized-trace convention, radial
component of the Euclidean mean-curvature vector, H_tan as non-radial mean-curvature component, Section 3 'subtracting
the radial projection of H_raw', metric inner product defined once, Monge coordinates as estimator, 'within the
quadratic surrogate', sign reversal of the probe-facing shape quadratic, Appendix C relative-II qualified as an
extrinsic diagnostic, abstract 'does not in general determine either'; p-bounds in Appendix E now true upper bounds
(4e-3, 3e-5); sixteen compressions, Figure 1 at 0.20 textwidth): `check.sh` PASS; References begin on page 4 with 12
reference lines on that page.

Page-count status (2026-09-16, second review-fix pass: exact finite-sample criterion with centred q_c, q-matched
random control, encoder naming made explicit (supervised ImageNet-21k ViT-B for the readout set vs DINOv3 ViT-B/16 for
the cross-survey set), within-anchor vs cross-anchor sentence, abstract PRH sentence, "dominated by", connection
mismatch retained in the Discussion; Probe/Result paragraphs merged, Figure 1 at 0.22 textwidth, ~15 compressions):
`check.sh` PASS; References begin on page 4 with 12 reference lines on that page.

Page-count status (2026-09-15, review-fix pass: intervention paragraph rewritten as a counterfactual local
second-order surrogate with shape-flat baseline, condition 2<R,K_S> > |K_S|^2, matched random control, and
"consistent with" wording for t*>1; Section 4 residual wording, Galaxies infinitesimal-vs-finite sentence,
Discussion agreement wording; fifteen compressions): `check.sh` PASS; References begin on page 4 with 14
reference lines on that page.

Page-count status (2026-09-15, citations pass: eight verified references added for the novelty review — Liu+23,
Chung+16, Slatton+26, Psenka+24, Kaufman&Azencot 23, Cheng&Wu 13, Bangachev+26, Acosta+23 published record —
cited in Section 2, the Section 5 intervention paragraph and the Discussion; paid for by cutting the Discussion
secondary-terms sentence, the Section 4 intuition sentence, the trace-statistic sentence, and Figure 1 at 0.25
textwidth): `check.sh` PASS; References begin on page 4 with 14 reference lines on that page.

Page-count status (2026-09-15, earlier: manuscript restructured so the headline is "a linear readout gains
where the manifold bends toward the label and loses where it bends away" — new title, abstract sentence,
contribution (iv), Section 5 heading and a dedicated paragraph, Discussion opening; ~20 compressions elsewhere
paid for it): `check.sh` PASS; References begin on page 4 with 14 reference lines on that page.

Page-count status (2026-09-15, earlier: Appendix E and the intervention sentence in Section 5 and the abstract added;
six further cuts incl. the two decoder-curvature related-work citations and the CKA-strata clause): `check.sh` PASS;
References begin on page 4 with 13 reference lines on that page.

Page-count status (2026-09-15, after Appendix D, the cross-encoder sentence and the Limitations sentence on
the known-surface demo; superseded above): `check.sh` PASS; References begin on page 4 with 12 reference lines on that page
(Times metric); CM fallback spills 12 lines. Fourteen trims were made to pay for the two additions
(Figure 1a cross-reference sentence, Section 4 summary opening, "In sum" sentence, Known-surface last
sentence, pilot numbers, noise clause, in-sphere ratio clause, density formula, and shorter phrasings).

Page-count status (2026-09-13, after folding the cross-encoder evidence and the relative-II pilot in; superseded above):
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
  All three tables are generated by the script `docs/latex/ml4ps/appendix_gen.py` (run from the repo root) and spliced between the
  `% BEGIN/END APPENDIX AUTOGEN` markers in `main.tex`; never edit the numbers by hand.
- Page gate after this pass: main text ends on page 4 with essentially zero margin (References start on
  page 5). Overleaf/Times confirmation is mandatory; cut list unchanged.

Citation check (2026-09-18): all 17 cited entries re-verified. Every DOI resolved on CrossRef and matched title,
authors, venue, volume, pages. arXiv API checked for published versions of the six preprints: Lee & Park 2023 has a
PMLR record (221:505-518), entry updated; Chou et al. 2026 arXiv id 2603.01879 added (ICLR 2026 per arXiv comment);
Jurewicz et al. article number 6424 and issue 1 added; AstroPT arXiv DOI added (no journal version exists as of
today); Platonic Universe, Slatton et al. and Alain & Bengio remain preprints. Fourteen uncited entries left in the
file (harmless under unsrtnat).

Open items before submission:
- Author affiliation and names are only needed for the camera-ready `final` build.
- Seven uncited bibliography entries still carry a `note` field asking for author lists (harmless with unsrtnat, which prints cited entries only); `lee2023curvature` and `acosta2022templates` were verified against arXiv 2026-09-12.
- Confirm the track choice on OpenReview.
