# Review of the Test 1 draft design (probe-free curvature capacity)

Reviewer stance: skeptical, read-only. Scratch simulations are in this folder (`sim_estimator.py`, `sim_leak.py`, `sim_leak2.py`, `sim_trace_noise.py`). Spectral numbers below are recomputed from `curvature-experiment/results/ii-rank/*_spectra.npz`. No pod access was used. The local machine has no `.cache` (no geometry npz, embeddings or labels), so nothing here touches real label Hessians.

## Verdict

Do not run the test as designed. The estimator itself is sound. A split-half ratio of sums is unbiased in expectation, and the simulation confirms this at split-half cosines of 0.22 to 0.28. The comparison it feeds is not sound. The key control, N2, is miscalibrated in a direction the data cannot tell you about. A linear function of the ambient coordinates has a Hessian that lies exactly in II's row space, weighted by s_j² times the readout's loading on each bent normal direction. From the test-0 spectra, a random ambient direction gives F_32 of 0.82 to 0.94. A readout whose loading tracks curvature-induced variance gives 0.99 to 1.00. N0 gives 0.235. So the synthetic band is set mainly by two things N2 does not match. One is the share of the label Hessian that comes from the linear part. The other is where that linear part sits among the normal directions. Neither has anything to do with whether a label is physical. A gap of ±0.15 can be produced or hidden this way, more than twice the toy minimum detectable effect of about 0.06. On top of that, f(z) cannot be computed from what the pod stores, because only anchor latents were ever saved. Cross-fitting removes noise, not bias, and one of the estimator's biases is itself II-shaped. The 8-of-10-encoders rule gives no protection against a misspecified null, because every encoder shares that misspecification. The question is the right one, though, and a revised test can answer it. Hold the label's own near-OLS linear part fixed, randomise only the residual with surrogates on the data's kNN graph, use a k-free spectrum-weighted statistic, and calibrate the decision rule on surrogate-as-physical runs before looking at the physical labels. If that test passes, it supports a narrower claim than the paper's headline. It says the label's curvature beyond its global linear part lies in the manifold's dominant bending subspace more than smooth random residuals do. It does not rescue the t = 1 vs t = 0 counterfactual, which stays a least-squares fact.

## Background facts used throughout

- Per anchor, M is the in-sphere II flattened to D x m (m = 136), with singular values s_j and right singular vectors r_j (`12_ii_rank_run.py:83-90`). For any ambient vector v, Hess_M(v·x) = M^T v_N + (sphere term), exactly. So a linear label's flattened Hessian has coordinates s_j (l_j·v) on r_j, and its captured fraction is F_k = Σ_{j≤k} s_j²(l_j·v)² / Σ_j s_j²(l_j·v)².
- Pooled over the 512 anchors, from the test-0 spectra:

| encoder | random direction v (s²-weighted), k = 16 / 32 / 64 | v loading ∝ s_j (s⁴-weighted), k = 16 / 32 / 64 |
|---|---|---|
| vit_base | 0.67 / 0.83 / 0.95 | 0.96 / 0.99 / 1.00 |
| vit_large | 0.69 / 0.83 / 0.95 | 0.97 / 0.99 / 1.00 |
| convnext_base | 0.67 / 0.82 / 0.94 | 0.96 / 0.99 / 1.00 |
| clip_base | 0.74 / 0.88 / 0.97 | 0.97 / 1.00 / 1.00 |
| dinov3 S / S+ / B / L / H+ / 7B | 0.67-0.89 / 0.83-0.94 / 0.95-0.98 | 0.95-1.00 / 0.99-1.00 / 1.00 |
| N0 (k/m) | 0.118 / 0.235 / 0.471 | |

- Every galaxy point lies in about twelve neighbourhoods (main.tex:531), and a maximal low-overlap anchor set has 19 to 21 members. The effective sample per encoder is tens, not 512.

## Critical issues

**C1. N2's linear part (random ambient unit u) is the wrong reference, and the composition of the label Hessian is not matched.**
For y_syn = α u·x + β f + noise, the α-part has F_32 ≈ 0.82-0.94 (table above), whatever the encoder does. The f-part has some other F. So F(y_syn) is about w·0.85 + (1-w)·F_f, where w is the linear share of the Hessian energy. Matching global R², median local variance and median ||Hess|| does not pin w. A physical label is decodable through high-variance directions, where curvature-induced variance lives. Any smooth function of position on the data cloud is too, because a curved manifold bends into the span it occupies elsewhere. Its normal gradient therefore loads on the bent directions far more than a random u does, since a random u puts most of its mass in the roughly 700 directions where the decoder manifold does not extend. That alone moves F_32 from 0.83 to about 0.99, a "physical" excess of up to +0.16 with no physics in it. The opposite failure is just as easy. If the synthetic labels end up with a larger linear share than the physical ones, they sit near the ceiling and the physical labels lose. A random u also has low variance, so ridge at α = 100 recovers u·x poorly. To match OOF R², the draft would then have to inflate α, which inflates the linear part's Hessian norm. That is a third uncontrolled coupling.
*Fix.* Give every synthetic label the physical label's own linear part. Take near-OLS β, y = β·x + r, and set y_syn = β·x + r_syn, randomising only the residual (revised design, step 2). Then the linear channel is identical by construction, global R² matches exactly, and the comparison is "beyond decodability" in the literal sense. Use near-OLS and not ridge (see I7).

**C2. The f(z) component cannot be built from stored artifacts, and even if it could, it shares the decoder's frame.**
The f(z) term needs the decoder latent for every one of the 2,048 neighbours of every anchor. The original geometry runner saves `z_anchor` for the 512 anchors only (`09_physics_probe_facing_run.py:285-286`). The sweep geometry the test would read, `geometry/<enc>/09_probe_facing_geometry_d16_seed0.npz` (`sweep/jobs.py:106`), comes from the split runner's refit path, which saves `anchor_idx, J, Hess, image` and no latents at all (`09_physics_probe_facing_split_run.py:349-350`). No model weights are saved either. So f(z) would mean refitting ten decoders and checking that each refit reproduces the stored geometry. There is a second problem even if the latents existed. Converted to orthonormal coordinates with J = QR, a Hessian defined in z coordinates becomes R^{-T} H_z R^{-1}, and II_on is II(R^{-1}·, R^{-1}·) (`12_ii_rank_run.py:83-85`). Both get amplified along the same short tangent directions. So an f(z) built to be isotropic in z will line up with II partly through the chart, not through the label.
*Fix.* Build surrogates intrinsically on the data, as heat-diffused white noise on a kNN graph of X, re-orthogonalised to linear functions of x (revised design, step 2). Do not use random Fourier features in PCA coordinates as the primary null. Their normal gradient lives in whichever top-q PC subspace you choose, so F would depend on q, an arbitrary knob.

**C3. Cross-fitting removes independent noise, not the estimator's shared errors, and one shared error is II-shaped and scales with the label's gradient.**
The tangent-projected quadratic fit (`09_physics_probe_facing_split_run.py:135-161`, u = dx J g^{-1} at :146) is exact when the decoder tangent is the data's tangent. In the toy, a label with zero intrinsic Hessian comes out at zero to machine precision (`sim_leak2.py`, eps = 0 rows). Tilt the decoder tangent into the normal space by angle ε and the fit returns h_bias = <v, II> with |v| ∝ |∇y|·ε. Its captured fraction is F_k = 0.95, which is exactly the isotropic-linear value 0.945 for that toy. The bias grows linearly in ε (|bias| 0.017, 0.034, 0.070 at ε = 0.05, 0.1, 0.2 with |∇y| = 1). Off-manifold scatter combined with ambient kNN selection gives a second shared bias. It is large (|bias| 0.15-0.22 per unit gradient at a scatter comparable to the ball radius) and not II-shaped (F ≈ k/m), so it pulls F toward N0. Both halves see both biases, so <h_A, h_B> contains ||h + b||² and F mixes curvature with gradient-scaled artefact. Labels with a large gradient-to-curvature ratio are pulled toward the bias's F, and physical and synthetic labels differ in that ratio unless it is matched. With real decoders at 0.966-0.985 variance explained, a tangent error of a few degrees is plausible, so this has to be measured, not assumed small.
*Fix.* (a) Placebo labels y = v·x with v in the top-16 local-PCA directions of each neighbourhood. Their true Hessian is about 0, so ||ĥ||/|v| and F(ĥ) calibrate the tilt leak on real data. (b) Report principal angles between the decoder J and the local-PCA tangent at each anchor. (c) In the revised design, physical and surrogate labels share β·x and so share most of the gradient. Also match residual gradient norms explicitly.

**C4. The decision rule is uncalibrated, and its multi-encoder clause does not protect against the main risk.**
All ten encoders share the same 86,471 galaxies, the same labels and the same anchor indices (sanity review I12). The main failure mode here is a misspecified null (C1-C3), and that misspecification is common to every encoder. If it produces a spurious excess, it produces one in all ten, so "≥ 8 of 10" is about as stringent as a single test. The CI is a block bootstrap over anchors for the physical label only. It leaves out the uncertainty of the median over 20 synthetic draws, and percentile resolution at 20 draws is 1/21. No size or power is stated. In the toy (`sim_estimator.py`, 32 clustered blocks, heavy-tailed ||h||) the SD of a difference between two labels' F_32 is about 0.023, giving an MDE of about 0.06. That is optimistic, because real anchor weights are concentrated (see I4). Meanwhile, the excess that can come from the residual part of the Hessian is diluted by (1 - w).
*Fix.* Before any physical-label number is looked at, run the full pipeline with surrogates standing in for the physical label (pseudo-physical runs, at least 20 per encoder). Check that the rule rejects at 5% or less, and read the SD and MDE off the same runs. Use a paired bootstrap that resamples anchors jointly for the physical label and all surrogates, recomputing the band in every replicate. Pre-state a minimum effect size. Report the encoder count as robustness to the embedding, not as replication.

## Important issues

**I1. F_k mostly measures which tangent directions the label varies along.** The Swiss roll shows it in one line. There II has rank 1, bending only along the roll direction s (ds⊗ds), so for any label g(s, h), F_1 = g_ss² / (g_ss² + 2g_sh² + g_hh²). Any function of s scores 1 and any function of the flat height scores 0, whether or not anything "bends toward" anything. For single-index-like labels h ≈ g''∇y∇yᵀ + g'Hess(s). So F is high whenever the label's gradient lies along strongly bent tangent directions. Magnitude and redshift are dominant factors of image variation, and the decoder spends its curvature on dominant directions of variation. Random f in N2 has random gradient orientation. *Fix.* Report F_k(ĝĝᵀ), with ĝ the label's local gradient from the other half, for physical labels and surrogates. Add a robustness version that projects h and V_k off the d-dimensional subspace {ĝ ⊙ e_i}. If the excess survives only with the gradient-aligned part included, the claim reads as "the manifold bends along the direction the label varies", which is weaker and has to be worded that way.

**I2. The trace direction is contaminated three ways.** (a) On the sphere, an ambient-smooth label's Hessian includes -(∇y·x̂)g, which is -(β·x0)·I for its linear part. V_k comes from II^S, which has the radial part removed (`12_ii_rank_run.py:86-89`). That puts the identity term outside the subspace by construction, with a size set by how the readout meets the embedding cone's mean direction. (b) In d = 16, kNN points sit near the ball's shell, so the regressor |u|²/2 is nearly collinear with the intercept. For a uniform-ball design the trace direction carries 8.2 times the average noise variance per direction (`sim_trace_noise.py`; a Gaussian design gives 1.0). (c) Ambient kNN selection |u|² + |II(u,u)|²/4 ≤ r² bends the shell in an II-dependent way, which makes the trace coefficient selection-sensitive. *Fix.* Make the traceless analysis primary, projecting both h and M onto Sym²₀ (m - 1 = 135), and report the trace share separately. Alternatively, run on the full II_on with the radial row kept and show that both versions agree.

**I3. N1 (anchor shuffle) is not frame-invariant.** The orthonormal frame at each anchor is the Q of J = QR, tied to the decoder's latent axis order. Frobenius products are O(d)-invariant only when h and II sit in the same frame at the same anchor. Pairing h_a with V_k(b) compares coordinates in two unrelated bases, so the result depends on the QR convention (polar or QR, axis permutation). *Fix.* Replace it with a Haar tangent-rotation null, h → Q h Qᵀ with Q uniform in O(d) at the same anchor. That keeps h's eigenvalues and trace and randomises only orientation relative to II, which is the "trace-heavy meets trace-heavy" control the draft was after.

**I4. Estimator details.**
- Use the ratio of sums. In the toy it is unbiased (truth 0.413, estimate 0.413 to 0.414) at every linear share. Median-of-ratios is biased low by 0.02-0.05, targets a different estimand, and 7-10% of per-anchor denominators are negative at split-half cosines of 0.22-0.28. The naive single-half F is biased toward k/m (0.34 vs 0.41).
- Gate on reliability. With no signal, the ratio of sums swings between -1.5 and +1.5 (`sim_leak.py`, L0 rows). Require the lower 95% block-bootstrap bound of the pooled split-half cosine Σ<h_A,h_B> / Σ||h_A||·||h_B|| to be above about 0.1, or report "not estimable". This will probably bite for stellar mass.
- The ratio of sums weights anchors by ||h_a||². With lognormal norms (log-sd 0.8) the effective anchor count is about exp(-2.56)·512 ≈ 40 before any clustering. If physical labels curve most where II is most concentrated, F_phys rises through weighting alone. The paper found |Hess y| tracking local accuracy in rank (sanity review C2), so its anchor profile is not random. Match the across-anchor profile of the signal norm, or add an equal-weight variant that divides each anchor's numerator and denominator by an independent scale, such as local label variance.
- Average the cross-products over several random half-splits (for example 5). Each split stays unbiased and the variance drops.

**I5. A hard cut at k is unstable and adds multiplicity.** The spectrum is smooth (vit_base s_32/s_1 ≈ 0.21, k90 at 45-49), so decoder noise reorders directions at the cut. At k = 32 the linear channel already sits at 0.82-0.94, which leaves little headroom. *Fix.* Make a k-free statistic primary, G = m·<h_A, MᵀM h_B> / (tr(MᵀM)·<h_A,h_B>), pooled as a ratio of sums. It equals 1 for isotropic h and Σs⁴·m / (Σs²)² for an isotropic linear label. It needs no k, uses the same cross-fit cancellation, and does not depend on singular vectors at near-degenerate cuts. Keep F_16/32/64 as descriptive output.

**I6. II's multi-direction structure is unvalidated, and its stability across decoders is unknown.** The tensor-fidelity fixture has a one-dimensional in-sphere normal space (sanity review I5), so nothing has checked that the decoder recovers the right-singular subspace V_k when there are many normal directions. Test 0 saved singular values only (`12_ii_rank_run.py:152`), with no cross-seed subspace comparison. *Fix (gate).* For ViT-B, compare seed 0/1/2 and the width-400 geometry. Report the per-anchor subspace overlap ||V_k⁽¹⁾ᵀV_k⁽²⁾||²_F / k against the random value k/m, and the agreement of G for a fixed label across decoders. If the overlap at k = 32 is close to k/m, the test is about the decoder and should not run.

**I7. Ridge vs OLS in the linear part.** If the decomposition uses the α = 100 ridge β, then r = y - β_ridge·x still contains (β_OLS - β_ridge)·x. That is a linear function, so its Hessian is pure <·, II>. It would add II-shaped curvature to the physical residual but not to an orthogonalised surrogate residual. It is the ridge signature again (novelty review Section 7). *Fix.* Use near-OLS β for both, either an OLS fit on a PCA basis keeping 99.9% of the variance or ridge with α ≤ 1e-3 times the mean eigenvalue. Orthogonalise r_syn with the same projector.

**I8. Matching and nuisance details in N2 (they carry over to any surrogate).**
- Match the signal Hessian norm (the cross-product <h_A,h_B>), not ||ĥ||. The estimated norm is noise-inflated, and the noise share differs between labels.
- Copy each label's missingness mask to its surrogates. photo_z is 92.6% populated and stellar mass 91.9% (`physics_labels.py:113-119`). Missingness is not at random and it changes the local design.
- Physical residuals are heavy-tailed (photo_z catastrophic outliers), while Gaussian surrogates are not. Build the surrogate's fine-scale part by permuting the physical residual's own fine-scale remainder, and use the same robust fit (for example Huber) for both.

**I9. "A statement about the encoder's representation" needs a non-learned baseline.** If a random-init ViT or pixel-PCA embedding of the same images shows the same excess, the result is about galaxy-image statistics. Brightness and size dominate pixel variance. The galaxy embeddings come from the Platonic Universe release and `sweep/embed.py` only handles SMILES, so this probably needs images and a GPU. If it cannot be done, word the claim as being about "the embedding manifold" and not about learning.

**I10. Label selection.** mag_r and photo_z were picked because the paper's effects are strong there, and they are physically correlated. Treat them as one family. If the catalogue has further physical columns (colours, sizes, other Galaxy Zoo fractions), pre-state one or two as held-out confirmation labels.

## Minor issues

- **M1.** CLAUDE.md requires a Swiss roll check for a new curvature measure. It is cheap here and illustrates I1 directly. Expected values are F_1 ≈ 1 for f(t) and for the ambient x-coordinate, and ≈ 0 for f(height) with an undefined denominator for a flat label. Note that the roll's II has rank 1, so it does not test multi-direction recovery (I6).
- **M2.** V_k has to be recomputed from the geometry npz, because test 0 kept singular values only. Assert that the stored `anchor_idx` equals `pcp.anchor_indices(...)`, as `09_physics_probe_facing_split_run.py:359` does.
- **M3.** In the ridge-capacity variant, the constant c in λ = c·s_max² is unspecified, which is a forking path. Drop the variant, or fix c now. G covers the same idea without a tuning constant.
- **M4.** Use at least 40 surrogates per label, so that a 95th-percentile claim has resolution.
- **M5.** Report d = 20 for ViT-B, where that geometry exists, as a sensitivity check.
- **M6.** Always print F/G for K = Hess(β·x) (noise-free), for the surrogate band and for N0 side by side, so the reader can see the ceiling the linear channel sets.
- **M7.** Present N0 as a scale reference only. Any decodable label clears it.
- **M8.** Compute. The fits are cheap if one QR per anchor-half is shared across all targets. 512 × 2 × S fits of 1024 × 153, solved for about 160 targets at once, take minutes. The kNN graph for the surrogates (86k points, k = 10) is the most expensive new step, roughly 10 minutes at D = 4096 on 4 threads, less after a PCA to 256. The draft's "< 1 h per encoder" holds for the revised design.

## Revised design sketch

**Gates.** All of these run blind to the physical labels' F/G.
1. Swiss roll notebook (M1).
2. Decoder stability (I6). For ViT-B seeds 0/1/2 and width 400, report the V_32 overlap and the agreement of G. Stop if the overlap is close to k/m.
3. Leak placebo (C3a) and the J-vs-local-PCA principal angles (C3b).
4. Pseudo-physical calibration (C4). Run 20+ surrogate-as-physical passes per encoder through the whole pipeline. Record size, SD and MDE, then fix the effect threshold.

**Main test, per encoder and label.**
1. Near-OLS β on all finite rows. Set r = y - β·x.
2. Surrogates. Build a kNN graph (k = 10) on X. Set r_syn = A·exp(-τL)ξ, with ξ white noise, plus the permuted fine-scale remainder of r. Choose (τ, A) so that local residual variance at the 2,048 scale and the median and IQR of the split-half cross-product Hessian norm match r's. Orthogonalise r_syn to the linear functions of x with the same OLS projector, and copy the missingness mask. Set y_syn = β·x + r_syn. Draw 40 per label.
3. Hessians. Use `local_quadratics` unchanged (`09_physics_probe_facing_split_run.py:135-161`) on 5 random half-splits, converted to the orthonormal frame with J = QR, for y, every y_syn, β·x (giving K, noise-free) and r.
4. Statistics. G as primary, F_16/32/64 descriptive, both in-sphere and traceless. Ratio of sums over anchors, averaged over splits. Signed cross term c = Σ<h_r,A, K> / Σ||K||², where c ≈ -1 means the label is flatter than its readout and c > 0 means it bends with it. Gradient diagnostics F(ĝĝᵀ) and the gradient-orthogonal G (I1).
5. Nulls. Surrogate band (primary), Haar tangent rotation (orientation vs spectral shape), N0 (scale only). Drop the anchor shuffle.
6. Inference. A 32-block paired bootstrap (`11_review_robustness_run.py:122-136` blocks), resampling anchors jointly for the physical label and all surrogates. In each replicate compute E = G_phys - median G_syn and the physical label's percentile within the band. A label passes in an encoder if the lower bound of E's 97.5% interval (Bonferroni over mag_r and photo_z) is above 0, E is at least the calibrated threshold, and the physical label sits above the band's 95th percentile. Report the count over the 10 encoders as robustness.
7. Optional secondary, for the linear channel on its own. Compare F of K = <β_N, II> against β rotated randomly within the embedding's PCA variance bands, which keeps the variance profile. It is noise-free and asks whether the readout direction points along bent normals more than equal-variance directions do.

## What would and would not support the claim

**Supports** "physical-label curvature lies in the manifold's dominant bending beyond decodability" if all of the following hold.
- The physical G is above the surrogate band for mag_r and photo_z in most encoders, with the calibrated rule.
- It survives the traceless and Haar-rotation controls.
- It holds across decoder seeds.
- The leak placebo is small relative to the physical signal norm.
- Ideally it is weaker in a non-learned baseline.
If the excess holds only with the gradient-aligned part included, the supported claim shrinks to "the manifold bends most along the directions in which the label varies".

**Does not support it.**
- An excess only against N0, or only against random-ambient-u labels. Both are expected for any decodable label.
- An excess carried by the trace component.
- An excess that shows up only with median-of-ratios or at a single k.
- An excess driven by a few high-||h|| anchors.
- Any F/G where the pooled split-half cosine is not clearly above 0.
- A physical G inside the surrogate band. That means the label's curvature is what its linear readout and the manifold's bending produce together, and the paper's headline should be framed as a least-squares and decodability fact.
- c ≈ -1. That means the label is intrinsically flatter than its readout, which runs against "bending toward".

**Scope.** A positive result supports a new, narrower claim. It does not make "t = 1 beats t = 0" non-trivial. That comparison stays implied by least squares (novelty review Sections 6-7), and the paper should present it as a check of the instrument.
