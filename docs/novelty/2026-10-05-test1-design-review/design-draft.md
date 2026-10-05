# Test 1 draft design: probe-free curvature capacity

## Question
Does the manifold's dominant bending point toward the curvature of physical labels more than toward the curvature of comparable non-physical labels on the same manifold? Asked without using a probe fit to the label being tested, so a "yes" is a statement about the encoder's representation, not about least squares.

## Context (established)
- Test 0 (curvature-experiment/results/ii-rank/): at d = 16 the in-sphere II at each of 512 anchors, as a map Sym^2(T) (m = 136) -> normal space, has effective rank 22-42 (random Gaussian matrix 114-134), 90% of its squared spectrum in 29-50 directions, condition number 40-140. Ten galaxy encoders.
- A linear probe's Hessian on M is <w_N, II>, so the probe-reachable Hessians at an anchor are the row space of the flattened II (M, D x m). Reaching weak directions costs 40-140x more probe weight.
- The paper's label-Hessian estimate: least-squares quadratic fit of the label over the anchor's 2,048 neighbours in tangent-projected coordinates; split-half tensor cosine only 0.19-0.35 on real labels (noisy direction), norm well ranked.

## Measure
Per anchor a, per label:
1. h = label Hessian in orthonormal tangent coordinates (latent-coordinate fit converted with J = QR), flattened Frobenius-preserving to R^136. Estimated twice on disjoint random halves of the neighbours (h_A, h_B), as in the paper's cross-fit.
2. V_k = top-k right singular vectors of the in-sphere II matrix (from test 0's SVD), k in {16, 32, 64}.
3. Noise-robust captured fraction: F_k = <P_k h_A, P_k h_B> / <h_A, h_B>, P_k = V_k V_k^T. Independent estimation noise in the two halves averages out of the numerator and denominator in expectation; ratio taken over the median across anchors (or as sum-over-anchors ratio) for stability.
4. Cost-weighted variant: ridge capacity with lambda = c * s_max^2: fitted fraction of h reachable by min_v ||M^T v - h||^2 + lambda ||v||^2.

## Nulls and controls
- N0 isotropic: E[F_k] = k / m for a random direction in Sym^2.
- N1 anchor shuffle: pair label Hessians at anchor a with II at anchor b (random permutation; or nearest-in-radius matched). Tests whether any alignment is local/specific rather than generic structure shared by all anchors (e.g., trace-heavy Hessians meeting trace-heavy II).
- N2 matched synthetic labels on the same manifold and same decoder (the key control). Circularity at the label level: any linear function y = u.x has Hess_M y = <u_N, II> exactly, so linear decodability alone drives h into II's dominant directions (a random normal vector v gives a Hessian with energy distributed like s^2). Synthetic labels y_syn = alpha (u.x) + beta f(z) + noise, u a random ambient unit vector, f a smooth random function of the decoder latent (random Fourier features), with alpha, beta, noise chosen per physical label to match its global linear OOF R^2 (alpha=100 ridge), its median local label variance, and its median ||Hess_M y||. Several draws (e.g. 20) per physical label give a null band.
- Decoder-prior confound: the decoder's smoothness may itself concentrate II. Comparing physical vs synthetic labels on the SAME decoder geometry cancels that prior. Optional: repeat on the seed-1/seed-2 and width-400 decoders that exist for ViT-B (Appendix A of the paper).

## Statistic and decision rule (to be fixed before running)
Excess E_k = F_k(physical) - median F_k(synthetic band), per encoder and label; 95% block-bootstrap CI over 32 neighbourhood-overlap blocks (as in review-robustness).
Claim "the representation bends toward physical label curvature beyond decodability" for a label if E_32 > 0 with CI excluding 0 in >= 8 of 10 encoders. Pre-stated labels of interest: mag_r and photo_z (where the paper's effects are strong); smooth_fraction and stellar_mass reported, no claim.
Also report F_k against N0 and N1 for every label.

## Compute
CPU on the pod (geometry npz in /mnt/ssd-cluster/EffDim/sweep-out/geometry/<enc>/, embeddings and labels from setup_pod's hf parquets). 10 encoders x 4 labels x (1 physical + 20 synthetic) quadratic fits at 512 anchors x 2 halves of 1,024 points with 153 regressors. Estimated < 1 h per encoder on 4 threads; the 7B encoder (D = 4096) dominated by loading.

## Open questions for review
1. Is N2's matching (linear R^2, local variance, Hessian norm) the right set, and is a random ambient direction u the right "linear part" (vs. matching the physical probe's normal-component norm)?
2. Is F_k with split-half cross-products a sound noise-robust estimator, given split-half cosines of 0.2-0.35?
3. Does the comparison still contain a hidden circularity (e.g. physical labels are more linearly decodable along high-variance directions, which are also the strongly bent ones)?
4. Is there a cleaner test that answers the same question?
