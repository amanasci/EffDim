# CHANGELOG_FROM_EXISTING_MANUSCRIPT

## Source manuscript (preserved, not edited)

The ML4PS text was written from the existing NeurReps extended abstract

- `submissions/neurreps_2026/main.tex`  
  title: *Rank- and Scale-Conditioned Curvature Is Associated with Local Linear Decodability in Vision Representations*

and its LPA-revision sibling `submissions/neurreps_2026_lpa_revision/`, **without modifying those trees**. Frozen experiment output trees were not rewritten. Figure 1 is a copy of the NeurReps confirmatory curvature–error figure (`figure2_curvature_probe.pdf`), not a replacement of the NeurReps submission files.

Bibliography entries were copied from `submissions/neurreps_2026/references.bib` (already verified for that venue) and re-checked; no new citations were invented.

## Venue and template

NeurReps used the JMLR/PMLR workshop class (`jmlr`, `mlabstract`) with a watermark and an appendix. ML4PS requires the official NeurIPS 2026 LaTeX style, \(\le 4\) body pages, **no appendix**, and the exact footer

`Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute.`

Only `\@noticestring` is redefined in `main.tex`. `neurips_2026.sty` is byte-identical to the repository copy and is not otherwise patched. The paper does not claim that ML4PS is an official NeurIPS workshop.

## Framing change

NeurReps asked whether **rank-conditioned sphere-normal curvature co-varies with local linear-probe \(R^2\)**, and labelled the result `claim_supported_but_scale_dependent`. The scientific payload was an audit of a curvature–decodability correlation, with scale dependence as a first-class caveat and LPA as an exploratory appendix.

ML4PS asks a physical-science question: **how photometric information is geometrically organized in an astronomical foundation-model representation, and when local decoding needs second-order structure.** The spine is now

1. curvature \(\leftrightarrow\) global linear error at frozen \(d=16\), \(k=2048\);
2. held-out quadratic label structure \(\Delta_Q\) and its curvature association;
3. Hessian alignment with sphere-normal bending, interpreted as an **anisotropic prior** (geometry-regularized quadratic decoding).

Title used: **How Representation Curvature Organizes Photometric Decoding in Astronomical Foundation Models**  
(“physical-property decoding” is not used; the target is apparent \(r\)-band magnitude.)

## Claims removed or weakened relative to NeurReps

- No scale-invariance or “tracking across \(k\)” claim. Intermediate-\(k\) \(n=128\) numbers, \(k=512\) reliability failure, and FWER tables are dropped for space; the paper states the result is **rank- and bandwidth-conditioned** and quotes \(d=12\) / \(d=20\) only as a compact caveat.
- No “evaluated chart range as geometric adequacy,” eigengap language, or reconstruction-vs-rank figure (NeurReps Fig. 1).
- No internal target identifiers (`mag_r_desi_local_oof_r2`) in the prose.
- No DESI spectroscopic / excluded-label results.
- No causal language.
- LPA is not an appendix. Only the secondary observation that \(\rho(K_H,\Delta\mathrm{MSE}_{G\to P})\) **rises** after conditioning on \(\Delta_Q\) remains, plus the statement that patch probes are not better on average.
- Mechanical decision labels (`claim_supported_but_scale_dependent`, `quadratic_chart_link_unresolved`) are omitted.

## How QLCA and the audit were incorporated

Frozen QLCA (`physics_quadratic_label_chart_alignment`) supplies the two preregistered primaries: median \(\Delta_Q\) and \(\rho_{\mathrm{ctl}}(K_H^{\mathrm{cross}},\Delta_Q)\), Holm, and the observed \(A_B\) / Hessian cosine.

The audit (`physics_quadratic_label_chart_alignment_audit`) supplies the scientifically load-bearing **reinterpretation**:

- \(B^S_{\mathrm{flat}}\) is algebraically rank 136 at every anchor;
- energy ranks \(r_{90},r_{95},r_{99}\) are 71, 90, 119;
- the original 48-mode BS map is an **implementation cap**, not an energy-rank constraint;
- truncated geometry-only maps still retain most UQ gain (median of per-anchor ratios);
- Haar / isotropic / matched-anchor nulls put \(A_B\) in an orientation-sensitive regime;
- real-design nested-CV shuffles (\(n=192\), \(\Delta_Q\approx-0.0004\)) calibrate the actual estimator.

The original synthetic \(|\Delta_Q|\) gate with fixed \(\alpha_Q=100\) is mentioned in one sentence as a miscalibrated gate, not as evidence against the nested result.

## Why “geometry-regularized quadratic decoding”

A low-dimensional chart-constraint claim would require the reachable Hessian family to be a proper subspace of the 136 quadratic coefficients. The audit shows algebraic full rank and a hard cap at 48 modes. Full-rank ridge on the coefficient \(c\) is equivalent to unrestricted quadratic regression with penalty \(\gamma^\top(B^{S\top}B^S)^+\gamma\). Predictive concentration onto leading bending modes is real; geometric low-rank concentration is not. The paper therefore says geometry supplies an **informative anisotropic prior**, not a genuinely low-dimensional quadratic function class.

## Figures

- No new method schematic: no existing vector asset defined tangent coordinates, \(B^S\), and \(\hat y_{UQ}\) with enough fidelity.
- At most two body figures: confirmatory global error (Fig. 1) and quadratic mechanism (Fig. 2).
- No numerical table in the body (the truncation fractions live in prose + Fig. 2d).
