# Novelty review (2026-09-15): has "a linear readout gains where the manifold bends toward the label and loses where it bends away" been reported before?

**Status:** literature review, single AI reviewer, abstracts read and one full text; every citation below
must be opened and verified by a human before it is relied on. Not pre-registered. Search date 2026-09-15.
Raw records: `search_records.json` (all queries, result rows, ids). Flow: `flow.png`.

## Question
Does prior work report, on learned representations, that a linear probe's local accuracy is higher where
the embedding manifold's probe-facing bending ⟨w_N, II⟩ is aligned with the label's own Hessian and lower
where it is anti-aligned — in particular an anchor-level intervention showing the mirror-image bending
hurts relative to a flat readout?

Facets searched: (A) probe/readout accuracy vs manifold curvature; (B) second-order expansion of a linear
map on a submanifold / Hess_M(w·x) = ⟨w_N, II⟩; (C) decoder-based pointwise second fundamental form;
(D) Platonic Representation Hypothesis and geometry; (E) direct phrasings (label Hessian, probe-facing,
relative second fundamental form).

## Search log
| source | queries | records | note |
|---|---|---|---|
| OpenAlex, keyword (full-text) search | 8 | 160 | noisy; only 5 on-topic |
| OpenAlex, title-restricted, ≥2015, by citations | 8 | 84 | |
| OpenAlex, citing sets | 2 works | 4 | Liu+23 has 0 citations in OpenAlex; "Platonic Universe" not indexed under that title |
| Semantic Scholar | 6 (2 failed on 429) | 44 | rate-limited |
| Web search (Google-style) | 12 | ≈100 links | 8 phrase queries + 4 targeted follow-ups |
| arXiv API | 17 + 9 retries | 0 | HTTP 429 throughout the session; **gap: arXiv full-text search not done** |
Inclusion: any work relating extrinsic curvature / second fundamental form of a data or representation
manifold to linear regression, linear probes, linear readouts, or cross-model alignment. Exclusion:
curvature in physics/biology senses, graph or point-cloud curvature methods without a readout, intrinsic
dimension without curvature, PRH work without geometry.

## Screened works (abstract level), by facet

**Closest antecedent (facet A/B).**
- **Liu, He & Tsai 2023**, "Linear Regression on Manifold Structured Data: the Impact of Extrinsic
  Geometry on Solutions", TAG-ML at ICML 2023, PMLR 221:557–576 (arXiv 2307.02478). Full text read.
  Theorem 2.4 (hypersurface): the optimal normal-direction weight of least squares is
  ∂g/∂ν + ½ Σκᵢ ∂²g/∂xᵢ² / Σκᵢ² + O(L²) — the label's second derivatives contracted with the principal
  curvatures over their squares. Theorem 2.5 (codimension k): the regression is ill-posed when the
  submanifold is flat in any normal direction. Framing is uniqueness, blow-up as κ→0, and OOD stability;
  experiments on synthetic curves and a bent MNIST digit-2 set. **They do not** state that bending toward
  the label helps and away hurts, do not intervene on the readout, do not measure on learned embeddings,
  and do not have the sphere/shape split or a per-anchor test. Our t* = ⟨e0,q⟩/⟨q,q⟩ is the empirical,
  per-anchor, general-codimension analogue of their Theorem 2.4 ratio. **Must cite.**
- **Cheng & Wu 2012/2013**, "Local Linear Regression on Manifolds and its Geometric Interpretation"
  (arXiv 1201.0327; JASA). Nonparametric local linear regression on tangent-plane estimates with bias
  analysis including curvature; classical statistics antecedent for "curvature enters the bias of a local
  linear fit". No readout-alignment statement. Optional cite.

**Manifold-capacity line (facet A).** Chung, Lee & Sompolinsky 2016 (Phys. Rev. E 93, 060301, "Linear
readout of object manifolds"), Chung et al. 2018 (PRX), Cohen et al. 2020 (Nat. Commun.), and
**Slatton, Chou & Chung 2026** ("Linear Readout of Neural Manifolds with Continuous Variables",
arXiv 2603.10956): linear decodability/regression capacity as a function of manifold dimension, radius
and correlation structure. No extrinsic curvature, no label Hessian, no alignment term. Different
question (capacity of a population of manifolds) from ours (where on one manifold a fixed readout fails).

**Curvature as an obstacle to linear methods (facet A).** Psenka et al. 2023/JMLR 2024 ("Representation
Learning via Manifold Flattening and Reconstruction"): removes extrinsic curvature so linear methods work.
Kaufman & Azencot 2023 ("Data Representations' Study of Latent Image Manifolds"): layer-wise curvature
profile, curvature gap correlates with generalization. Kaufman Sirot & Azencot 2025 (ICML, "Curvature
Enhanced Data Augmentation for Regression"): second-order manifold model for augmentation. These treat
curvature as a scalar to reduce or exploit; none conditions on alignment with the label's curvature, which
is what makes our sign flip. Our result is the refinement: curvature helps or hurts a linear readout
according to the sign of ⟨Hess_M y, ⟨w_N, II⟩⟩, not its magnitude.

**Decoder-based curvature instruments (facet C).** Acosta et al. 2023 (CVPRW, "Quantifying Extrinsic
Curvature in Neural Manifolds", arXiv 2212.10414): topological-VAE decoder → metric and extrinsic
curvature. Couéraud, Sunkara & Schütte 2025 (arXiv 2508.20413): conformal regularization of decoders,
scalar curvature. Chen, Latifi Jebelli & Rockmore 2025 (arXiv 2511.02873): point-cloud curvature
estimators, bias in high d. None connects the instrument to a readout. Instrument precedent for Section 2;
Acosta was cited in an earlier draft and cut for space.

**PRH and geometry (facet D).** Duraphe et al. 2025 ("The Platonic Universe", arXiv 2509.19453): the
motivating study. Bangachev, Bresler & Polyanskiy 2026 ("Representation Alignment Rests on Linear
Structure", arXiv 2605.28870): alignment from the Linear Representation Hypothesis; no curvature. Sarkar
2026 (arXiv 2604.08579): spectral/functional-map diagnostic; no curvature. NeurIPS 2025
"An Information-Geometric View of the Platonic Hypothesis": Bayesian posterior concentration; no
submanifold curvature. Wurgaft et al. 2026 ("Manifold Steering", arXiv 2605.05115): intervening along
on-manifold paths for behaviour control — an intervention on activations, not on a readout's normal
component, and no curvature/label term.

**Separability measures (facet A).** Wei, Qi & Shen 2026 (arXiv 2606.08721): affine separability
diagnostic, no curvature. "Neural Feature Geometry Evolves as Discrete Ricci Flow" (arXiv 2509.22362):
graph Ricci curvature of features, classification; not extrinsic II vs label.

**Classical identity (facet B).** Hess_M(w·x) = ⟨w_N, II⟩ is textbook (Lee 2018, cited). The residual
expansion with the sphere/shape split and the relative-II alignment term were not found elsewhere; the
"relative second fundamental form" phrase returned only pure differential geometry.

## Verdict
1. **Not replicated before.** No work found reports, on learned embeddings, that a linear readout gains
   local accuracy where the manifold bends toward the label's curvature and loses it where it bends away,
   nor the per-anchor intervention with a mirror-image arm and a random-direction null, nor the
   cross-encoder replication.
2. **Theoretical antecedent exists and must be cited:** Liu, He & Tsai 2023 derived the optimal normal
   weight of least squares on a curved manifold in terms of label second derivatives and curvature; they
   read it as a uniqueness/stability condition. Our contribution relative to it: the residual expansion
   and improvement condition 2⟨Hess_M y,K⟩ > ‖K‖², the sphere/shape split on unit-normalized
   embeddings, measurement on five foundation-model embeddings, the per-anchor intervention (t = ±1,
   random null, t* > 1 undershoot), and the PRH framing.
3. **Adjacent framings to distinguish in one sentence:** manifold capacity (dimension/radius, no
   alignment) and manifold flattening (curvature as obstacle). Our sign-conditioned result explains when
   each is right.

## Limitations of this review
arXiv API unavailable (429) for the whole session, so arXiv full-text search was not run; Semantic Scholar
partially rate-limited; no Google Scholar; single AI reviewer; only one full text read; venue and page
metadata verified from the PMLR page for Liu+23 only, other records from abstract pages. A human should
re-run the arXiv queries in `search_records.json` (keys `A1…F1`) once the API is reachable.

## Action taken in the manuscript
One sentence with `\citep{liu2023linear}` in the Section 5 intervention paragraph; bib entry from the
PMLR record.
