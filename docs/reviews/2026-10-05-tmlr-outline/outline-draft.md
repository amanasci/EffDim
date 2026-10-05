# Extended paper (TMLR): outline draft, design section 1

Working title: "What a Linear Probe Sees on a Curved Representation" (alt: keep "Linear Probes on Curved Latent Spaces").
Venue: TMLR (double-blind, no hard page limit; aim ~12-14 pages main text + appendix). Lives in a new folder paper/tmlr/, copied from paper/latex/; the ML4PS paper in paper/latex/ stays as submitted.

## Positioning (the honest frame, after the novelty and sanity reviews)
A linear probe on a curved embedding manifold is not linear on the manifold: its intrinsic Hessian is <w_N, II> (classical). We make three things precise and measure them at scale:
(i) a least-squares probe's bend points toward the label's curvature as a consequence of least squares, with an explicit deviation term for a global, regularised probe (theory, extending Liu, He & Tsai 2023 to arbitrary codimension and a global probe);
(ii) how learned representations actually bend: a validated pointwise II on frozen foundation-model embeddings, which turns out to bend in a few dominant directions (effective rank 22-42 of 136 at d = 16 across ten encoders);
(iii) what the curvature mismatch does and does not tell you about local probe reliability (it tracks local accuracy for magnitude and redshift under dependence-aware inference, but is close to the label-Hessian norm in rank, so its added value over label curvature is small).
We do not claim that encoders bend toward physical quantities beyond what least squares implies; we say why that question is hard to test and what a test would need (test-1 review).

## Sections
1. Introduction. Probes as instruments in science; global R^2 hides local reliability; curvature is the second-order story. Contributions (i)-(iii). Explicit scope statement of what is not claimed.
2. Background and related work. Liu+23 (Thm 2.4), manifold capacity (Chung et al.), flattening (Psenka), curved features in LMs and linear representability (Gurnee+26, Hindupur+26, Yocum+26, Modell 2026), extrinsic curvature instruments (Acosta+23), local regression bias (Cheng & Wu), Jurewicz+24, Canatar+21. Verify every citation (novelty report lists them).
3. Theory.
   3.1 Identity Hess_M(w.x) = <w_N, II>; sphere split K = K_S + K_sph on unit-normalised embeddings.
   3.2 Residual expansion: local squared error = c^2 + s^2||delta||^2 + (1/2)s^4||Hess_M y - K||_g^2 + ...; improvement condition 2<Hess_M y, K> > ||K||^2.
   3.3 Proposition (bend toward the label): finite-sample surrogate SSE(t) = ||r0 - t q_c||^2; t* = 1 + <r_c + e_c, q_c>/||q_c||^2 where r is the probe's own local residual; local least squares gives t* = 1 exactly; a global OLS probe gives t* near 1 with a deviation from local misfit; ridge adds a positive shift (shrinkage). Corollaries: help at t = 1 iff t* > 1/2, hurt at t = -1 iff t* > -1/2 (so "hurt" is not independent evidence); the random-direction null tests orientation, not geometry.
   3.4 Reachable Hessians: the probe-reachable set at an anchor is the row space of the flattened II; its spectrum sets the cost of bending toward a given Hessian.
4. Instrument: decoder-based pointwise II.
   4.1 Plain AE decoder, sphere-projected; forward-mode geometry for large D.
   4.2 Validation: trace (original) and full tensor (tensor fidelity), with the known limits: the in-sphere fixture's normal bending is one-dimensional; the label-Hessian magnitude is biased (about 1.3-2x at paper scale); direction recovered. NEW WORK: extend the fixture to r > 1 bending directions so the probe contraction's direction depends on w.
   4.3 Decoder ablations (seeds, width) for the II spectrum.
5. How learned representations bend.
   5.1 II spectrum across ten galaxy encoders (test 0): effective rank, k90/k99, condition numbers vs random; DINOv3 21M-6.7B ladder shows no size trend. NEW WORK: QM9 (geometry pruned; recompute on the pod GPU), decoder seeds/width for ViT-B (decoder-prior check).
   5.2 Shape vs sphere terms; mean curvature fails as a scalar summary (existing Appendix E).
6. The probe's bend in practice: confirming the theory.
   6.1 Counterfactual on galaxies, 10 encoders: help/random-help, t*.
   6.2 NEW WORK: ridge sweep, t* vs alpha down to near-OLS (PCA-truncated if needed): t* -> 1 as alpha -> 0 is the theory's prediction.
   6.3 QM9 (8 encoders): direction holds 32/32, help > 0.5 in 3-4/8, surrogate fidelity per anchor near zero on molecules (Spearman -0.30 to +0.23) and why (third-order terms / decoder fit; tokenizer-merged molecules).
7. Curvature mismatch as a diagnostic, and its limits.
   7.1 Mismatch vs local R^2 (multi-scale controls), cross-fit, block bootstrap: magnitude and redshift robust (19-20/20 pairs), morphology and stellar mass not.
   7.2 Mismatch vs the label-Hessian norm alone (differ by <= 0.05 in 36/40 cells); extended controls; held-out Delta R^2 (median +0.001 to +0.15 at alpha 100; +0.015 to +0.26 tuned).
   7.3 Validation-tuned probe.
8. Discussion and limitations: what would show encoders bend toward physics beyond least squares (the test-1 design and its pitfalls); surrogate vs probe; one galaxy sample; label-Hessian noise (split-half 0.19-0.35); estimator biases (tilted tangent leaks II-shaped error).
Appendices: full tables from tensor-fidelity, review-robustness, scaling, QM9 reports; QM9 pipeline (tokenizer, d choice incl. tle == mle disclosure, duplicates); proofs.

## New work implied (each would be its own small sub-project)
N1 ridge sweep (t* vs alpha) on stored geometry; CPU; galaxy 10 encoders, QM9 if geometry recomputed.
N2 multi-direction tensor-fidelity fixture.
N3 II spectrum for QM9 (recompute geometry, GPU) and for ViT-B decoder seeds/width.
N4 proposition proof and a numerical check of the t* decomposition on real anchors.

## Open questions for review
1. Is the positioning honest and still worth a TMLR paper (contribution enough)?
2. Is the proposition correct as stated (global OLS / ridge cases), and is it more than Liu+23?
3. Is anything in sections 5-7 overstated given the sanity review?
4. What is missing that a TMLR reviewer would demand?
