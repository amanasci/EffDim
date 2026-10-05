# Novelty review, framing 1 (2026-10-03): "the probe-selected bend points toward the label's own curvature"

Status: single AI reviewer acting as a skeptical ICML area chair. Search date 2026-10-03. Builds on the
2026-09-15 review (`archive/.planning/phases/09-.../novelty-review-2026-09-15/`) and its 2026-09-17 addendum;
it does not repeat that review's screening of manifold capacity, flattening, PRH, etc. except where the
verdict changes. Raw records are in `2026-10-03-framing1-search-records.json` next to this file, which also
holds the source of the small simulation used in Section 6. Every citation should be opened by a human before
it goes into the paper.

## 1. Question

Framing 1 makes this the headline: for a fitted linear probe w on a curved embedding manifold M, the probe's
restriction to M has intrinsic Hessian K = <w_N, II>, and **the bend the probe selects points toward the
label's own curvature**. The main evidence is a per-anchor counterfactual on a local second-order surrogate
that scales the decoder-estimated in-sphere shape term q = (1/2)<w_S, II^S>(u,u) by t. At t = 1 local R^2
beats shape-flat (t = 0) at 93-99% of anchors (galaxies, 10 encoders including a DINOv3 21M-6.7B ladder)
against about 5-35% for a random normal direction of matched quadratic amplitude; t* > 0 (in the current
draft t* is 1.5-3.6, so t* > 1); the direction of the effect carries over to QM9 (32/32 encoder-label pairs,
8 SMILES encoders), with a smaller effect.

Two questions. (a) Has this been shown before, in whole or in part? (b) Even if nobody has written it down,
would an ICML reviewer call it known or trivial, and does the evidence answer that?

## 2. Search log

| source | queries | records returned | notes |
|---|---|---|---|
| arXiv API (export.arxiv.org, https) | 75 (sets X1-X25, Y1-Y30, Z1-Z20) | about 700 rows; see JSON | **Worked this time** (the 09-15 gap). Some Z-set queries hit 429 or timeouts and were retried; the final status of each one is in the JSON |
| arXiv id_list lookups | 6 batches | 25 abstracts | used to check web-search hits and pull metadata (dates, comments, journal refs) |
| Semantic Scholar graph API | 14 keyword queries + 1 citation query | 7 hits from the 3 keyword queries that returned; 3 citing papers of Liu+23 | 11 of the 14 keyword queries failed with HTTP 429 even with 15 s backoff. The citation query for arXiv:2307.02478 worked |
| OpenAlex | 12 title/abstract searches (from 2022) + 3 citing-set queries + 7 id lookups | 400+ rows, mostly noise (Zenodo) | Liu+23 has 0 citing works in OpenAlex; Cheng & Wu 2013 has 34 since 2022 (all statistics, none about probes); Acosta+23 has 6 |
| Web search (Google-style) | 17 | about 150 links | phrasings aimed at scoops ("linear probe" + "normal component" + Hessian; curvature aligned with the task helps readout; second-order probing; astro/chem FM geometry; decoder II estimation; per-sample reliability) |
| Full texts opened | 4 | | Liu, He & Tsai 2023 (PDF, Sections 2.1-2.3 and Thm 2.4 read closely); Yocum et al. 2026 (PDF, read for theorems and any mention of curvature); Slatton, Chou & Chung 2026 (HTML, checked for curvature/Hessian terms); the EffDim draft `paper/latex/main.tex` (Sections 3-5, Appendix D) |
| Local computation | 1 simulation (Section 6) | | checks whether the intervention's headline numbers come out under labels that have no special relation to curvature |

Inclusion criteria: work that (i) relates a linear readout's behaviour on a curved data or representation
manifold to curvature, II or the Hessian of the target, (ii) intervenes on a readout's second-order or
curvature term, (iii) studies curved or nonlinear feature manifolds against the linear representation
hypothesis and says something about linear probes, (iv) estimates II or curvature of learned representations
with a decoder or a generative model (2023-2026), (v) studies probe reliability or geometry of foundation-model
embeddings in astronomy or chemistry, (vi) local or kernel regression bias from curvature, or follow-ups of
Liu+23.

## 3. Screened works

Threat levels: **scoops** (the claim is already shown), **partial overlap** (a main ingredient or the theory
is already there), **must-cite adjacent** (a reviewer will ask about it, but it does not make the same claim),
**optional**, **irrelevant**. "Read" means what I actually looked at.

| # | work (verified id / venue) | what it shows | relation to framing 1 | threat | read |
|---|---|---|---|---|---|
| 1 | Liu, He & Tsai 2023, "Linear Regression on Manifold Structured Data: the Impact of Extrinsic Geometry on Solutions", TAG-ML @ ICML 2023, PMLR 221; arXiv 2307.02478 | Local least squares on a curved hypersurface: normal weight w_y* = dg/dy + (1/2) sum_i kappa_i d2g/dx_i^2 / sum_i kappa_i^2 + O(L^2) (Thm 2.4); ill-posed if flat in a normal direction (Thm 2.5) | Thm 2.4 says the LS-fitted normal weight is the L2 projection coefficient of the label's Hessian onto the curvature. So the LS probe's K is the best approximation of Hess y within the span of the curvature, which is "the probe-selected bend points toward the label's curvature", with t* = 1 and improvement over flat guaranteed (Pythagoras). Missing are the codimension-k case, learned embeddings, a global probe, the intervention | **partial overlap, close to a theoretical scoop of the headline** | full text |
| 2 | Cheng & Wu 2013, JASA 108(504):1421-1434 (doi 10.1080/01621459.2013.827984, via OpenAlex); arXiv 1201.0327 | local linear regression on manifolds; curvature enters the bias | classical reason why local error grows with the label's (and manifold's) second-order terms; your cross-anchor mismatch reducing to roughly the label-Hessian norm is the bias term in this literature | must-cite adjacent | abstract (09-15 review) |
| 3 | Jurewicz et al. 2024, Nat Commun 15 (doi 10.1038/s41467-024-49568-4) | value is coded on a curved vmPFC manifold; a linear readout makes systematic, predictable errors | curvature causes linear-decoder error, no II/label-Hessian alignment | must-cite adjacent (already cited) | abstract + search snippets |
| 4 | Canatar, Bordelon & Pehlevan 2021, "Spectral bias and task-model alignment...", Nat Commun 12:2914 (doi 10.1038/s41467-021-23103-1, via OpenAlex); arXiv 2006.13198 | how well a kernel/linear readout generalizes depends on how well the target aligns with the representation's eigenfunctions | the general principle "a readout does well where the representation is aligned with the target"; a reviewer can say framing 1 is a local, second-order instance of task-model alignment | must-cite adjacent | abstract |
| 5 | Yocum, Allen, Olshausen & Russell 2026, "Neural Manifold Geometry Encodes Feature Fields", NeurIPS Workshop on Symmetry and Geometry in Neural Representations, PMLR 282:770-791 | proves the geometry of the embedding of a feature's domain fully determines which functions are linearly representable (RKHS/Mercer argument) | the same idea one level up: which targets a linear probe can read off is fixed by the embedding geometry. No curvature, no II, no local error | must-cite adjacent | full text (theorems; grep for curvature/Hessian found none) |
| 6 | Gurnee, Ameisen, Kauvar, Tarng, Pearce, Olah & Batson 2026, "When Models Manipulate Manifolds: The Geometry of a Counting Task", arXiv 2601.04480 (Transformer Circuits 2025) | character counts sit on curved 1-D manifolds; attention heads twist them so that a linear boundary reads off the decision; curvature is put together by many heads | curvature that a linear readout uses, built by the model, in LLMs. Qualitative and mechanistic, with no II or label Hessian | must-cite adjacent | abstract |
| 7 | Hindupur, Orgad, Fel, Ba et al. 2026, "When Models Don't Manipulate Manifolds: The Geometry of a Comparison Task", arXiv 2609.37680 | Qwen2.5-7B mostly uses linear representations of numbers even though curved geometry is present | counterpoint: curvature present but not used by the readout | must-cite adjacent | abstract |
| 8 | Engels, Michaud, Liao, Gurnee et al. 2024, "Not All Language Model Features Are One-Dimensionally Linear", arXiv 2405.14860 (ICLR 2025) | irreducible multi-dimensional (circular) features | standard citation for curved features vs the LRH; says nothing about how a probe interacts with curvature | optional / adjacent | abstract |
| 9 | Modell, Rubin-Delanchy & Whiteley 2025, arXiv 2505.18235; Modell 2026, "Probing for Representation Manifolds in Superposition", arXiv 2605.18537 | features as manifolds; a "manifold probe" that generalizes linear regression probes | manifold-aware probing; no curvature-target alignment | must-cite adjacent (2605.18537) | abstract |
| 10 | Sarfati, Bigelow, Wurgaft et al. 2026, "The Shape of Beliefs...", arXiv 2602.02315; Wurgaft et al. 2026, "Manifold Steering...", arXiv 2605.05115 | posteriors on curved manifolds; linear steering leaves the manifold; linear field probing | interventions on activations along curved paths, not on a readout's normal component | adjacent | abstract |
| 11 | Dooms et al. 2026, "Bilinear autoencoders find interpretable manifolds", arXiv 2605.08891; Kim 2026, "Non-linear Interventions on LLMs", arXiv 2605.14749 | quadratic latents and nonlinear interventions for curved features | "quadratic vs linear readout" in the interpretability sense; no II | optional | abstract |
| 12 | Zaher et al. 2026, "The Geometric Wall", arXiv 2605.09887 | layerwise manifold curvature and intrinsic dimension predict SAE scaling | curvature as an obstacle to linear dictionaries | optional | abstract |
| 13 | Hai & Li 2026, "When and Why Do Linear Bias Probes Fail?", arXiv 2609.22337 | theory of linear-probe failure on a manifold of curvature kappa (curvature ceiling on chordal separation) | curvature limits a linear probe, as a scalar; no label Hessian, no alignment | adjacent | abstract |
| 14 | King, Fedorenko & Hosseini 2026, "Representational Curvature Modulates Behavioral Uncertainty in LLMs", arXiv 2604.23985 | trajectory curvature correlates with next-token entropy; perturbations aligned with the trajectory change entropy, misaligned ones do not | an "aligned vs misaligned curvature intervention with a control", but curvature of a token trajectory, not of a manifold seen by a readout | adjacent (design echo) | abstract |
| 15 | Rahman, Barrett & Last 2026, "Characterizing AlphaEarth Embedding Geometry...", arXiv 2604.18715 | Earth-observation FM embeddings: tangent spaces rotate, linear-probe concept directions rotate across the manifold, local geometry predicts retrieval coherence | the closest "physical-science FM embedding geometry + linear probes" study; first-order only, no II or label Hessian | must-cite adjacent | abstract |
| 16 | Duraphe et al. 2025/2026, "The Platonic Universe", arXiv 2509.19453 (NeurIPS 2025 ML4PS; ICML 2026 Workshop on Mechanistic Interpretability poster, 11 authors on the ICML page) | linear probes and local (MKNN) geometry on astro FM embeddings; local alignment tracks physics performance | data source and motivation; no curvature | already cited (update author list/venue) | abstract + ICML page |
| 17 | Tame-Narvaez, Ciprijanovic & Trivedi 2026, "Beyond Point Estimates: Benchmarking UQ on the AION-1 Astronomical Foundation Model", arXiv 2606.07771 (PAI 2026) | conformal and locally valid UQ on frozen astro FM embeddings, per-galaxy intervals | a direct competitor for any "local reliability diagnostic" claim; a reviewer will ask why curvature beats locally adaptive conformal intervals | must-cite adjacent (if the reliability angle is kept) | abstract |
| 18 | Steier 2026, "Information Routing in Atomistic Foundation Models", arXiv 2603.03155 | ridge-probe linear accessibility on QM9 across 10 models; nonlinear probes mislead | QM9 probing methodology; no curvature | optional (chemistry section) | abstract |
| 19 | Aldeghi et al. 2022, "Roughness of molecular property landscapes and its impact on modellability" (ROGI), arXiv 2207.09250; J. Chem. Inf. Model. 62(19):4660-4671 (doi 10.1021/acs.jcim.2c00903, via OpenAlex) | a label-roughness index predicts ML regression error | the chemistry version of "local label nonlinearity predicts error", which is what your mismatch diagnostic turns out to track | must-cite adjacent (QM9 section) | abstract |
| 20 | Elabid et al. 2026, "Molecules Meet Language", arXiv 2605.06303 | linear probes on a molecular VAE latent; some properties have global directions, others only local gradients | molecular probe reliability is local; no curvature | optional | abstract |
| 21 | Acosta et al. 2023 (CVPRW; arXiv 2212.10414); Lee & Park 2023 (TAG-ML, PMLR 221; arXiv 2309.10237); Couéraud, Sunkara & Schütte 2025 (arXiv 2508.20413); Bouss et al. 2025 (arXiv 2506.12187, now Phys. Rev. per OpenAlex) | decoder or flow based extrinsic curvature of learned manifolds | instrument precedents for the decoder II; none is tied to a readout | must-cite adjacent for the instrument, not threats to framing 1 | abstract |
| 22 | Kaufman & Azencot 2025, "Curvature Enhanced Data Augmentation for Regression", ICML 2025, PMLR 267:29321-29344 (from PMLR/mlanthology listing in search results) | second-order manifold model for sampling in regression | a second-order local manifold model used for regression, no readout analysis | adjacent | abstract |
| 23 | Slatton, Chou & Chung 2026, arXiv 2603.10956 | regression capacity from dimension, radius, correlations | confirmed: no curvature, II or label Hessian | adjacent (already cited) | HTML |
| 24 | Liao, Maggioni & Vigogna 2021 (arXiv 2101.05119); "Convergence of Hessian estimator from random samples on a manifold with boundary" (Pure Appl. Anal. 7:807, 2025) | multiscale local polynomial regression on unknown manifolds; Hessian estimation from samples | relevant to how you estimate Hess_M y, not to the claim | optional | abstract / title |
| 25 | Zhang & Mueller 2026 (arXiv 2608.06809); Levada 2026 (2606.06329, 2605.04274, 2608.15313); Bedratyuk 2026 (2605.01073) | second-order or shape-operator diagnostics for DR embeddings or point clouds | curvature estimation without a readout | irrelevant / optional | abstract |
| 26 | Liu+23 citing works (S2: 2609.05744, 2403.20200, 2402.03021) | point-cloud functional manifolds; ridge on non-iid data; multirate GD | none extends Thm 2.4 to learned embeddings or probes | irrelevant | titles + abstracts |

Nothing in arXiv, OpenAlex, the partial Semantic Scholar results or web search reports the specific
per-anchor intervention (scale a fitted probe's in-sphere shape quadratic by t, compare to a random normal
direction) or the claim stated as "the probe's bend points toward the label's curvature" on learned
embeddings. The 09-15 conclusion "no direct precedent" survives the arXiv full-text search that it could not
run.

## 4. Close works in detail

**Liu, He & Tsai 2023 (full text).** In Sections 2.1-2.2 they fit least squares locally around a point of a
curve (x, kappa x^2) and of a hypersurface (x', sum kappa_i x_i^2) with symmetric sampling. The normal
coefficient comes out as the label's normal derivative plus (1/2) sum kappa_i d2g/dx_i^2 / sum kappa_i^2.
Restricted to M, the probe's Hessian is then w_y* kappa, which has the label's Hessian projected onto the
curvature (plus a contamination term from dg/dy). They read this as a uniqueness and blow-up result (w_y*
explodes as kappa -> 0) and an out-of-distribution warning. They do not say "the bend helps", but the
formula is exactly the least-squares statement that the fitted normal weight picks the curvature multiple
that best matches the label's Hessian. Under their local model your t* equals 1 by construction, and the
improvement condition 2<H,K> > ||K||^2 holds automatically, because K is an orthogonal projection of H.
Your draft already describes t* as "the empirical, per-anchor, general-codimension analogue of their
Theorem 2.4 ratio" (09-15 review). That is right, and it is also the problem for framing 1. A reviewer who
knows this paper will read the headline as Thm 2.4 measured on real data. What Liu+23 do not have is the
global-probe setting (one w fit on all data and evaluated anchor by anchor), codimension k > 1 with an
estimated II, the sphere/shape split, foundation-model embeddings, and the intervention.

**Yocum et al. 2026 and Canatar et al. 2021.** Both make the general point that what a linear readout can
represent or learn is set by how the target lines up with the representation's geometry (domain embedding in
Yocum, kernel eigenfunctions in Canatar). Neither involves curvature. They matter because a reviewer can
place framing 1 as "the second-order, local version of task-model alignment", which shrinks the headline.

**Gurnee et al. 2026 and Hindupur et al. 2026.** Together these are the interpretability literature's current
answer to "do models use curved feature manifolds for linear readout": sometimes yes (counting/linebreaking,
the model twists curved manifolds so a linear boundary works), sometimes no (number comparison stays mostly
linear despite curvature). Neither measures II or a label Hessian, and both are about LLM circuits, not
frozen-embedding probes. They give framing 1 a context to engage with. Your result, if it went beyond least
squares (Section 6), would be a quantitative, model-agnostic test of the same question.

**Rahman et al. 2026 (AlphaEarth) and Tame-Narvaez et al. 2026 (AION-1 UQ).** The nearest physical-science
FM-embedding papers. AlphaEarth reports first-order geometry (tangent rotation, probe directions rotating
across the manifold). AION-1 UQ gives per-galaxy, locally adaptive conformal intervals on frozen embeddings.
If the paper keeps any "local reliability diagnostic" language, it must compare to locally adaptive conformal
or kNN-residual baselines, which currently it does not.

**Aldeghi et al. 2022 (ROGI) and Cheng & Wu 2013.** The cross-anchor mismatch statistic ends up close to the
label-Hessian norm in rank. A reviewer will then say it measures local label nonlinearity, which is known to
predict the error of a linear (or any smooth) fit, in statistics (Cheng & Wu bias) and in chemistry (ROGI).
This does not hit the intervention headline directly, but it removes the cross-anchor result as independent
support for it.

## 5. Verdict on novelty

1. **Not scooped as stated.** No prior work runs the per-anchor second-order intervention on a fitted probe
   or reports, on learned embeddings, that the probe's normal component bends toward the label's curvature.
   The arXiv search closes the main gap of the 09-15 review without finding a precedent.
2. **The theory is essentially there.** Liu+23 Thm 2.4 gives the least-squares normal weight as the label
   Hessian projected on the curvature. In their local setting, "bends toward the label, t* = 1, improvement
   over flat" is a corollary. The extension to a global probe and codimension k is new but routine.
3. **The empirical headline is close to implied by least squares** (Section 6). With the current controls it
   does not show anything about the encoders or the physics beyond "the label is linearly decodable and the
   probe is a (ridge) least-squares fit". My simulation reproduces the paper's headline numbers with labels
   that have no special relation to curvature.
4. **ICML main track:** as the headline, framing 1 is not novel enough. A competent reviewer would call it a
   restatement of least squares (or of Liu+23) with an uninformative null. The work's real assets (a
   validated decoder II on frozen embeddings with an exact sphere/shape split, run at scale over 10 encoders
   plus QM9) are measurement contributions. They would carry a workshop paper (ML4PS, NeurReps, TAG-ML) or a
   TMLR paper. They need one of the results in Section 8 to carry an ICML main-track paper.

## 6. Strongest reviewer objections, and whether the evidence answers them

### Objection A (the strongest): "t = 1 is the probe itself; of course putting back part of a fitted predictor beats deleting it. The random direction is the wrong null."

The algebra, in the draft's own notation (Appendix D). At an anchor with neighbours i, the shape-flat base is
b_i = (w_T + w_rad)·x_i, and q_i = (1/2)K_S(u_i,u_i). The actual probe is yhat_i = b_i + w_S·x_i, and
w_S·x_i = w_S·x_0 + q_i + e_i, where e_i holds third-order terms and decoder error. Let r_i = y_i - yhat_i be the
real local residual of the probe and write _c for centring. Then the shape-flat residual is

  r0 = r_c + q_c + e_c,

so with the exact SSE(t) = ||r0 - t q_c||^2,

- t = 1 beats t = 0  iff  ||q_c||^2 + 2<r_c + e_c, q_c> > 0,
- t = -1 is worse than t = 0  iff  3||q_c||^2 + 2<r_c + e_c, q_c> > 0,
- t* = 1 + <r_c + e_c, q_c> / ||q_c||^2.

Both "help" and "hurt" hold unless the fitted probe's own local residual is strongly anti-correlated with its
own quadratic, with correlation below -(1/2)||q_c||/||r_c|| (help) or -(3/2)||q_c||/||r_c|| (hurt). Least
squares pushes the other way. For a ridge fit, X_c^T r = alpha w, so <X_c w_S, r> = alpha ||w_S||^2 >= 0: on
the whole data set the residual is uncorrelated (OLS) or positively correlated (ridge) with the very function
w_S·x whose local quadratic part is q. If that holds roughly in each neighbourhood, help and hurt follow. With
k = 2048 neighbours the sampling noise in <r_c, q_c> is about sigma ||q_c||, so even a small ||q_c|| clears the
bar. **t* > 1 is the ridge signature.** Positive correlation of residual with w_S·x is what shrinkage
produces, and the draft's own Table caption says so ("globally ridge-regularized probe under-using a locally
beneficial term").

The random arm is not comparable. Its criterion is 2<r_c + q_c + e_c, q_v> > ||q_c||^2. That asks a random
quadratic to fit a residual it was never fit to, so it fails at more than half the anchors whatever the
geometry. The decoder arm puts back something that was deleted from a fitted predictor. The random arm adds
something unfitted. The 93-99% vs 5-35% gap is built into the design.

**Simulation (source in the JSON).** I used a 3-D manifold in R^60 (random Fourier embedding, exact II from
analytic derivatives), 30,000 points, 300 anchors with k = 400, a global ridge probe, and the draft's exact
intervention and matched-amplitude random null, plus a cross-label control (K from a probe fit on another,
independent label, rescaled to the same ||q_c||). The labels are random smooth functions of the intrinsic
coordinates ("generic"), higher-frequency ones ("rough"), and pure noise. None has any designed relation to
the curvature.

| label | ridge alpha / mean eigenvalue of X_c^T X_c | label noise sd | global R^2 | t=1 helps | t=-1 hurts | median t* | random direction helps | other label's K helps |
|---|---|---|---|---|---|---|---|---|
| generic | ~0 | 0 | 0.99-1.00 | 0.92-1.00 | 1.00 | 1.03-1.04 | 0.15-0.18 | 0.13-0.45 |
| generic | ~0 | 0.5 | 0.79-0.80 | 0.81-0.92 | 0.99-1.00 | 0.98-1.06 | 0.18-0.24 | 0.15-0.43 |
| generic | 1 | 0 | 0.66-0.87 | 0.77-0.85 | 0.83-0.92 | 1.97-2.56 | 0.35-0.42 | 0.41-0.62 |
| generic | 1 | 0.5 | 0.53-0.69 | 0.76-0.80 | 0.81-0.90 | 2.04-2.79 | 0.34-0.43 | 0.43-0.64 |
| generic | 10 | 0 | 0.34-0.63 | 0.76-0.80 | 0.80-0.85 | 4.3-7.5 | 0.41-0.50 | 0.56-0.63 |
| rough | ~0 | 0 | 0.58-0.81 | 0.60-0.81 | 0.83-0.98 | 0.96-1.09 | 0.28-0.35 | 0.18-0.41 |
| rough | 1 | 0 | 0.22-0.41 | 0.56-0.60 | 0.63-0.68 | 1.7-5.2 | 0.44-0.48 | 0.41-0.47 |
| noise | ~0 | (label is noise) | 0.00 | 0.54-0.59 | 0.63-0.69 | ~1 | 0.38-0.51 | 0.40-0.46 |
| noise | 1 | (label is noise) | 0.00 | 0.47-0.52 | 0.48-0.52 | unstable | 0.46-0.53 | 0.45-0.54 |

(Ranges over three label seeds. The draft's alpha = 100 on unit-normalized D = 768 embeddings with N of about
70k training rows works out to roughly 1-2 times the mean eigenvalue, if the centred total variance is
0.3-0.6. That is the alpha/mean-eigenvalue = 1 row.)

What the simulation says: for any label the probe can decode, t = 1 helps at 76-100% of anchors and t = -1
hurts at 80-100%, while the matched random direction helps at 15-45%. And t* comes out around 2-3 at the
draft's level of shrinkage. All of this happens with no designed curvature-label relationship. The draft's
numbers (79-99% / 97-100% vs 22-36%, t* = 1.5-3.6) sit inside this null band. The effect goes away only when
the probe decodes nothing (noise label, about 50%). So "help at t = 1" measures that the label is linearly
decodable and that the probe was fit by least squares. It does not show that the encoder bends toward the
label. The 32/32 transfer to QM9 "in direction" is what this null predicts for any decodable label. The
smaller effect there is what the null predicts if the QM9 probes decode less well (check the global R^2 per
encoder-label pair), so it is not evidence of a weaker curvature mechanism.

Caveats. The simulation is Euclidean (no sphere split), uses exact II rather than a decoder, and d = 3 rather
than 16-20. Decoder noise would lower the help rates, which makes 93-99% on real data look, if anything,
like a sign that the decoder q tracks w_S·x well. That is a validation of the instrument, not of a
curvature-label mechanism.

**Does the current evidence answer Objection A? No.** The random-direction control rules out "any quadratic
of this size helps". It does not rule out "the probe's own quadratic helps because the probe is a fit". The
draft's own sentence "Under the quadratic Gaussian model, this is the signature of bending toward the
target's remaining intrinsic curvature" is true. But under least squares that signature is expected, not
discovered.

### Objection B: "Liu+23 already derived this."

Partly fair (Section 4). The defence is that Liu+23 is local LS on a hypersurface with an exact quadratic
model, while yours is a global ridge probe, codimension around 750, a decoder-estimated II and real
embeddings. That defends the measurement, not the headline. The headline has to state something Liu+23's
formula does not imply.

### Objection C: "The cross-anchor mismatch is just the label's local nonlinearity."

The draft finds mismatch close to the label-Hessian norm in rank. So the cross-anchor result says "where the
label curves more, a linear probe is locally worse", which is local-regression bias (Cheng & Wu) or ROGI.
K's own contribution to the cross-anchor result has to be shown by partialling out ||Hess_M y||. Then report
whether the alignment term (cos_g(Hess y, K)) predicts local R^2 beyond ||Hess y||, local label variance and
scale. The draft has an alignment association (+0.24 to +0.43), but the label-Hessian direction is only
weakly recovered (split-half tensor cosine 0.19-0.35). That makes the alignment estimate noisy, and it is
open to the same LS-implied reading.

### Objection D: "The intervention changes a surrogate, not the model or the manifold."

The draft admits this. It matters less than A, but together with A it means the counterfactual is about the
probe's own Taylor expansion.

## 7. Is t* > 0, or help at t = 1, guaranteed?

Not exactly guaranteed, but close to it in practice:

- **Local LS (Liu+23 setting):** guaranteed. t* = 1, and the improvement condition holds by orthogonal
  projection.
- **Global OLS:** t* = 1 + <r_c, q_c>/||q_c||^2, and the global normal equations make the residual
  uncorrelated with w_S·x summed over the data set. Neighbourhoods can break this locally (the probe's normal
  component at anchor A was chosen partly to fit first-order structure elsewhere, where w_S is tangent). So
  failures are possible, and a label with zero intrinsic Hessian everywhere would make t = 1 hurt. But for
  smooth decodable labels the simulation gives help at 80-100% and t* around 1.
- **Global ridge:** <X_c w_S, r> = alpha ||w_S||^2 > 0 gives a systematic push toward t* > 1. t* > 0 is
  therefore nearly automatic, and t* > 1 is expected from shrinkage alone.
- **When it is not automatic:** when the label is not decodable (noise, about 50%), or when the label's
  Hessian is anti-aligned with the probe's curvature. That anti-alignment would be the interesting
  geometric finding, and the current analysis treats it as noise (the 1-21% of anchors where t = 1 fails).

**Is the random-direction null the right control?** No. It answers "is the specific orientation of K special
compared with an arbitrary normal orientation?" The answer is yes, trivially, because K is the fitted one.
Better nulls are listed next.

## 8. Recommendations

Keep the measurement contribution, and change the headline unless one of the following comes out positive.

1. **Calibrated "fitted-but-geometry-free" null on the real manifold (necessary).** Build synthetic labels on
   the same embedding manifold, for example smooth random functions of the decoder latent z, or of a
   low-dimensional nonlinear map of the embeddings. Match each real label's global R^2, local variance and
   ||Hess_M y|| distribution. Fit the same ridge probe and run the identical intervention. If real labels'
   help fraction, t* and alignment lie inside the synthetic band, framing 1 is a least-squares fact and
   should not be the headline. If physical labels exceed it, that excess is the novel result: "foundation-model
   embeddings bend toward physical quantities more than toward equally decodable random quantities".
2. **Encoder-level curvature-label alignment that does not use the fitted probe.** For each anchor compute
   the best possible curvature fit, max over normal directions v of the fraction of the (held-out) label
   Hessian explained by <v, II> (a "curvature capacity" for the label). Compare across encoders, against
   random-init and shuffled-pixel encoders of matched decodability, and against label permutations within
   neighbourhoods. This says something about the representation, not the probe.
3. **Cross-fitted, cross-label and cross-encoder K.** Use K from a probe fit (a) on data outside a ball
   around the anchor, (b) on a different label, (c) on a different encoder mapped by Procrustes, or (d) on a
   different survey. If a K that was not fit to this label in this neighbourhood still helps (my simulation
   gives about 13-64% for an unrelated label), that is evidence of shared geometry. Physically correlated
   labels (magnitude vs redshift) need care.
4. **Predictive payoff.** Show that the curvature term improves held-out per-object prediction or
   calibration. Candidates: a local curvature-corrected readout using t* estimated out-of-fold from labelled
   neighbours, or an error predictor that adds mismatch to kNN label variance. Compare against ridge,
   local-linear, matched-parameter quadratic probes and locally adaptive conformal intervals (AION-1 UQ
   paper). A per-object reliability gain over these baselines would be a clear ICML-style contribution.
5. **Report the decomposition t* = 1 + <r_c,q_c>/||q_c||^2 and an alpha sweep.** Show t* against alpha,
   including near-OLS on a PCA-truncated basis. If t* -> 1 as alpha -> 0, say so plainly and present t* > 1
   as shrinkage. If t* stays well above 1 at alpha -> 0, the global probe really under-uses the curvature
   locally. That is a real, if modest, finding.
6. **Look at the anchors where t = 1 hurts.** Under the null they are noise. If they cluster in a physically
   interpretable region (blends, edge-on discs, high-z), "the bend points away from the label here" is a
   more interesting and less trivial claim than the majority result.
7. **Citations to add:** Canatar+21 (task-model alignment), Yocum+26 (geometry fixes linear
   representability), Gurnee+26 and Hindupur+26 (curved manifolds and linear readout in LMs), Rahman+26
   (AlphaEarth geometry and probes), Tame-Narvaez+26 (if local reliability is claimed), Aldeghi+22 ROGI and
   Steier+26 (QM9 section), Modell 2026 (manifold probe). Rephrase the Liu+23 sentence so it says their
   Thm 2.4 already gives the LS normal weight as the label-Hessian/curvature projection, and say what is
   added.
8. **Bibliography housekeeping:** `duraphe2025platonic` lists 4 authors and the NeurIPS 2025 ML4PS
   workshop. The ICML 2026 MechInterp workshop page lists 11 authors (Borrell, Dillmann, Duraphe, Eris,
   Khederlarian, Kumar, Marraffini, Smith, Sourav, Di Tella, Wu). Check which version you cite.

## 9. Limitations of this review

- Semantic Scholar keyword search mostly failed (11/14 queries returned HTTP 429), so S2's relevance ranking
  and its citation graph beyond Liu+23 were not used. There was no Google Scholar access, so the "cited by"
  list for Liu+23 comes from S2 (3 papers) and OpenAlex (0).
- arXiv API search is keyword-in-abstract/title. A paper that makes the same point in its body without these
  words would be missed. Several Z-set queries needed retries, and their final status is recorded in the
  JSON.
- Most works were screened at abstract level. Full texts read: Liu+23, Yocum+26 (theorems), Slatton+26
  (checked for curvature terms), and the EffDim draft. The venue for Kaufman & Azencot 2025 comes from search-result
  listings (PMLR/mlanthology), not from opening the publisher page. Cheng & Wu, Canatar and Aldeghi were
  checked against OpenAlex records.
- The simulation is a toy (Euclidean, exact II, d = 3, k = 400, 300 anchors, 3 seeds). It shows the
  intervention's statistics can come from least squares alone. It does not show that this is what happens in
  the galaxy data. Recommendation 1 is the real test.
- Single AI reviewer, no human cross-check. Treat threat levels as a starting point.
