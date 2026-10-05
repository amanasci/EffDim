# Review of the TMLR outline draft ("What a Linear Probe Sees on a Curved Representation")

Reviewer stance: TMLR action editor plus a line-by-line check of the mathematics. Read-only on the repo. Sources read in full: the outline, `paper/latex/main.tex`, the four result reports, the ten `ii-rank/*.json`, the sanity review, the Test-1 design review, and the runners `09_physics_normal_scaling_run.py`, `09_physics_probe_facing_split_run.py` (geometry, `local_quadratics`, `split_columns`), `12_ii_rank_run.py`, and the probe-fitting parts of `11_review_robustness_run.py`. I read only the opening of the novelty report (`docs/novelty/2026-10-03-framing1-novelty.md`). A tool-permission denial stopped the rest, so what I say below about Liu, He & Tsai 2023 must be checked against that report and the paper itself.

New numbers in this review come from two places. (a) The per-anchor counterfactual arrays already in `notebooks/.cache/09_physics_normal_scaling_*_d{16,20}.npz` (variants `S` and `S_model`, alpha = 100). (b) A toy check of the proposition, `prop_check.py`, in this folder.

## Verdict

The outline is much more honest than the ML4PS paper, and the bones of a TMLR paper are here. TMLR's bar is that the claims are correct and supported and that some audience cares, and a careful measurement paper on probe geometry clears that bar. But the outline still has four problems that would sink it with a careful reviewer. The fixes are cheap compared with the work already done. (1) The Section 3.3 proposition is not correct for the surrogate as implemented. "Local LS gives t* = 1 exactly" holds for the data-side variant `S`, not for the published decoder variant `S_model`. "Global OLS gives t* near 1" fails per anchor; in a toy with near-OLS, the per-anchor t* is 0.10 while the pooled t* is exactly 1. The ridge sign claim holds only pooled over all rows (or per anchor for a local ridge at the tensor level). So N1 as planned ("t* → 1 as alpha → 0") tests something the theory does not predict for the plotted quantity. (2) The published counterfactual mostly measures how well the decoder reproduces the probe's own normal readout. On the stored arrays the decoder quadratic has only 0.38-0.51 of the amplitude of the probe's data-side in-sphere readout, so about 75-85% of what the probe's normal component reads on a patch is not second-order curvature at d = 16. Section 6 cannot be titled "confirming the theory". (3) The II-spectrum result, which is now the paper's main new empirical claim, is stated more strongly than the numbers allow. II has full rank at every anchor (condition number 41-140), 90% of the squared spectrum needs 29-50 of 136 directions, and the runner's own rule, fixed before any number was printed, calls 8 of 10 encoders "partial". Nothing yet separates the decoder's prior from the representation. (4) The Section 7 "curvature mismatch" is `hess_mismatch_emp`, which contains no second fundamental form. In rank it is almost the label-Hessian norm, and the neighbourhood-residual baseline was never run. With these fixed, and with decoder-prior nulls and a decoder-free II estimate added, I would expect a TMLR accept as a measurement-and-clarification paper. The theory is a modest extension of Liu+23, not a headline, and the paper should say so.

## Critical issues

**C1. Proposition 3.3 is wrong for the implemented surrogate, and N1 inherits the error.**
The published counterfactual is variant `S_model` (Appendix D; RR Concern 5; `09_physics_normal_scaling_run.py:93-97`). The base readout is the data-side `X(w_T + w_rad)`, and the scaled term is the decoder quadratic `q = ½<w_S, II^S>(u,u)`, not the data-side `p = X w_S`. Then t* = <r_c + p, q_c>/‖q_c‖². "t* = 1 exactly under local least squares" is true only for variant `S` (q replaced by p). For `S_model` it is t* = 1 + <e, q_c − r_c>/‖q_c‖² with e = p_c − q_c, and that is not 1 unless e ⊥ (q_c − r_c). In the toy, local OLS gives t*_S = 1.000000 and t*_model = 7.06. "Global OLS gives t* near 1" is false per anchor. The global normal equations constrain only the pooled sum (toy: per-anchor t*_S = 0.10, pooled = 1.0000). "Ridge adds a positive shift" holds pooled and for the `S` variant: pooled t*_S = 1 + α‖w_S‖²/‖X̃w_S‖², which the toy matches to 4 dp. Per anchor it can take either sign (stored arrays: t*_S > 1 at only 69-100% of anchors). For `S_model`, t* also contains the scale-free fidelity slope β(p|q) = <p,q_c>/‖q_c‖², which can exceed 1 with no shrinkage at all.
*Fix.* Replace 3.3 with the corrected proposition below. Store <p,q>, ‖p‖² and ‖q‖² per anchor (one line in `scaling_at_anchor`). Re-target N1 to the pooled and per-anchor t*_S, which has an exact prediction, and report β(p|q) and the residual term of t*_model separately. Drop "PCA-truncated if needed". Truncation deletes the low-variance directions where w_S lives, so it changes the object being measured. OLS at D ≤ 4096 with n = 86k (or 131k for QM9) is directly feasible.

**C2. The published counterfactual is mainly a decoder-fidelity check, and most of the normal readout is not curvature.**
From the stored arrays (alpha = 100, 6 runs × 4 labels), median ‖q_c‖/‖p_c‖ is 0.38-0.51. If q were the exact second-order part of p, the remainder (off-manifold reconstruction residual along w_S, third-order terms, and the chart-origin offset, since u is centred at the data anchor and II at the decoder image) would carry 75-85% of the probe's in-sphere normal readout variance. This agrees with sanity-review I9 (the surrogate recovers 18-30% of the data-side gain). By the decomposition t*_model = β(p|q) + <r_c,q_c>/‖q_c‖², "help > random help" mostly says that cos(p, q) is clearly positive for the fitted w_S and near 0 for a random v (which is not part of w). That is a useful real-data validation of the instrument. It is not evidence about labels, and the outline's 6.1 and 6.2 present it as confirming the theory. The appendix caption's reading of t* > 1 as "ridge under-using a locally beneficial term" is confounded with ‖p‖/‖q‖ ≈ 2-2.6.
*Fix.* Retitle Section 6 along the lines of "What the probe's normal component reads, and what scaling it can show". Report three things. First, the share of the normal readout the decoder's second-order term explains (‖q‖/‖p‖, cos(p,q)) and how it moves with d. Second, the `S` variant together with the LS identity. Third, a null that does isolate the geometry: keep w_S and use II from another anchor, or Haar-rotate the tangent frame, or use the decoder of a different encoder. That last null should fail where the seed-1/2 decoders pass. State in the introduction that at d = 16 most of what a probe's normal component reads is off-model variation.

**C3. The II-spectrum claim (contribution ii, Section 5) is overstated and not separated from the decoder prior.**
From the ten JSONs, median effective rank is 22.3-42.4, participation ratio 9.9-22.1, k90 29-50, k99 78-104 of m = 136, and condition number 41-140 (Gaussian reference 1.4-3.9). The spectrum decays smoothly (s₁₆/s₁ = 0.22-0.34, s₃₂/s₁ = 0.13-0.22). II has full rank at every anchor, so every Hessian is reachable at a cost. The runner's pre-stated rule (k90 ≤ m/4 means "low") gives "partial" for 8 encoders and "low" only for the two 384-wide DINOv3 S/S+. "Bends in a few dominant directions" is therefore not what the data say. The Gaussian D×m matrix is a weak null, because any smooth SiLU MLP produces a decaying II spectrum. The tensor-fidelity fixture has a one-dimensional in-sphere normal space, so nothing validates spectrum recovery. Singular vectors were not saved, so stability across seeds is unknown (Test-1 review I6). m = d(d+1)/2 is fixed by the chosen d.
*Fix.* Reword to "a graded spectrum: about a quarter to a third of the 136 directions hold 90% of the squared second fundamental form, far from an isotropic map but not low rank". Make the following gates for any spectrum claim (detail in N3′ below). (a) Decoder-prior nulls: a random-init decoder; an AE fitted to a covariance-matched Gaussian cloud; an AE fitted to column-permuted embeddings. (b) A decoder-free II from local quadratic regression of the ambient coordinates on the tangent chart. Its noise floor flattens the spectrum where the decoder prior concentrates it, so the two bracket the truth. (c) A fixture with a known multi-direction spectrum (flat and decaying). (d) Seed and width subspace overlap against k/m. (e) A sweep over d.

**C4. The Section 7 diagnostic is not a curvature quantity, and its baselines are missing.**
Every real-data mismatch column is `hess_mismatch_emp` = ‖Hess y − Hess(p̂)‖_g, where Hess(p̂) is the local quadratic fit of the probe's predictions (`09_physics_probe_facing_split_run.py:192`). It is the quadratic part of the probe residual in the decoder chart, and II enters only through J and g. The decoder column `hess_mismatch_dec` is the one the synthetic test validated, and it is not tabulated. In rank, the label-Hessian norm alone matches the mismatch within 0.05 in 36 of 40 galaxy cells. The neighbourhood-residual baseline the ML4PS reviewer asked for was never run, and it would predict local R² from the same neighbours almost exactly. A diagnostic that needs labelled neighbours is useless in practice unless it beats the neighbours' own residual variance.
*Fix.* Retitle Section 7 along the lines of "Local residual curvature and local accuracy". Tabulate the emp and dec columns side by side. Add the cross-fitted residual-variance baseline (half-A residual variance → half-B local R²) and |Hess y| as baselines with block-bootstrap CIs. Claim explanation ("local error has a curvature component of size X"), not prediction.

## Important issues

**I1. Section 3.4, "reachable Hessians", is vacuous per anchor.** With D ≫ m and II of full rank, the reachable set is all of Sym². The binding constraints are the ridge penalty, which acts as spectral shrinkage (part (g) of the proposition below), and the fact that one w serves every anchor. *Fix.* Replace 3.4 with the local-ridge spectral formula K = V diag(s_j²/(s_j² + τ)) Vᵀ h, τ = 2α/s⁴. This makes "the spectrum sets the cost" exact. Add one paragraph on the global constraint (the per-anchor w_N is the projection of one w onto anchor-dependent normal spaces). Either prove something or state it as the open problem.

**I2. The residual expansion (3.2) and its improvement condition hold only for a Gaussian patch.** kNN patches are closer to uniform balls. For a spherically symmetric patch with E u₁⁴ = 3λs⁴, Cov(½A(u,u), ½B(u,u)) = (s⁴/4)[2λ<A,B>_F + (λ−1) trA trB]. For a uniform d-ball, λ = (d+2)/(d+4) = 0.9 at d = 16, so the trace direction is down-weighted about 9× relative to traceless directions. That matches the Test-1 review's simulated 8.2× noise inflation on the trace direction. *Fix.* State 3.2 and the proposition in this Σ-metric. Note that alignment cosines computed in the plain g-metric weight the trace differently from the error they are meant to explain.

**I3. The QM9 claims in 6.3 are overstated and incomplete.** "Direction holds 32/32" is help > random help, which (C2) is a fidelity test, and the per-anchor surrogate–probe Spearman on molecules is −0.30 to +0.23. The data-side `S` helps at 0.88-1.00 everywhere (sanity C3), so the transfer failure is surrogate fidelity, not "size depends on encoder". Further results are left out. With the tuned probe, help falls to 0.12-0.60, t* to 0.11-0.60, and the ChemFM alpha/cv mismatch turns positive with CIs spanning 0. ChemFM 3B mu is significantly positive (+0.11, p = 0.015). At alpha = 100 the ChemFM probes are badly underfit (OOF R² 0.27-0.63 vs 0.54-0.90 tuned). The suggested causes ("third-order terms / decoder fit; tokenizer-merged molecules") are untested. *Fix.* Move QM9 into its own short section or an appendix titled as a transfer test with a failure. Lead with the mismatch result (bootstrap: gap/alpha/cv 8/8, mu 5/8, at alpha = 100), report the tuned results and the ChemFM reversal, and either test the causes (‖q‖/‖p‖ on molecules; drop the duplicate rows) or call them conjectures.

**I4. Inference wording in 7.1.** Use block-bootstrap counts only. Add the thinned-anchor results (n ≈ 30; mismatch mag/z 5/10 at alpha = 100, 6/10 at α*, five encoders), and say that the encoder counts measure robustness to the embedding, not independent replications (same galaxies, anchors and labels). Report alignment's weaker bootstrap counts (mag 4/10, z 5/10 at alpha = 100).

**I5. One number in 7.2 is wrong.** The held-out ΔR² medians at alpha = 100 run from −0.006 (vit_large smooth_fraction) to +0.147, positive in 19/20, with p05 > 0 in 9/20. The outline's "+0.001 to +0.15" is wrong. The tuned range +0.015 to +0.258 is right. Also report that adding |Hess y| as a control makes 13/20 partials grow at α*, which is the suppression pattern, not over-control.

**I6. The label-Hessian magnitude bias direction is not established (4.2).** relerr is unsigned. For `lin` at paper scale (relerr 0.88, cos 0.978), ‖est‖/‖true‖ solves λ² − 1.956λ + 0.226 = 0, giving λ = 1.83 or 0.12. "About 1.3-2× too large" picks one root without evidence. *Fix.* Compute the signed norm ratio from the TF records (cheap), or say "magnitude error 29-118%".

**I7. The tuned probe (7.3, N1) changes the geometry being tested.** α* sits on the grid edge in 5/20 galaxy pairs and 6/8 QM9 encoders, the near-OLS w_N was never validated (tensor fidelity ran at alpha = 100), and the robustness battery is not cross-fitted. *Fix.* For 7.3, either cross-fit at α* or present it as descriptive. For N1, validate w_N against the fixture at small α.

**I8. Liu+23 must be stated exactly.** The delta has to be explicit: arbitrary codimension, the Σ-metric for non-Gaussian patches, spectral ridge shrinkage, and the finite-sample identity for the counterfactual. The last is algebra. Present the theory as a Proposition plus a Lemma that make the measurements interpretable, not as contribution (i) in the title position. Check the novelty report's Sections 6-7, which I could not read, for anything Liu+23 already covers.

**I9. The paper needs a d axis.** d = 16 and d = 20 were chosen, and no galaxy ID estimate is reported. Both the spectrum (m grows as d²) and ‖q‖/‖p‖ (more tangent directions move variance out of w_N) depend on d. *Fix.* Sweep d ∈ {8, 12, 16, 20, 24} for ViT-B at least, and report the normalised spectrum metrics (erank/m, k90/m) and ‖q‖/‖p‖ against d, with a galaxy ID estimate for reference. Known caveat: the d = 20 spike findings (`Skill spike-findings-effdim`) flagged dead ends, so check those before choosing the grid.

**I10. The decoder-free K cross-check (main.tex 214-216, median cosine 0.55-0.65) belongs in the main text.** It is the only decoder-free number on real data. Extend it to the full II (C3b).

**I11. "How learned representations bend" needs a non-learned baseline or a narrower title.** Without an untrained-encoder or pixel-PCA embedding of the same images, the spectrum is a property of "the embedding manifold", not of learning (Test-1 review I9). If GPU access to the images is not possible, retitle 5 as "How foundation-model embeddings bend".

## Minor issues

- M1. DINOv3 ladder: erank by size is 22.3, 28.3, 29.1, 29.5, 42.4, 26.2 (Spearman with params ≈ +0.43, n = 6), and D changes from 384 to 4096. Say "no monotone trend; size confounded with D", not "no size trend".
- M2. The Gaussian reference depends on D (erank 114-134), so report erank/m against each encoder's own reference.
- M3. In 3.3, call the corollaries a design property ("hurt" follows from help, since help ⊂ hurt), and report random-direction hurt (64-95%) next to model hurt.
- M4. The title is fine. Avoid "bend toward the label" anywhere it suggests intent. Under local LS t*_S = 1 even for a pure-noise label, which is overfitting, not bending toward physics.
- M5. Cite the plain-AE Swiss roll check (`02.6_swiss_roll_plainae_curvature_check.ipynb`) in the instrument appendix, and give the new multi-direction fixture its own small notebook per CLAUDE.md.
- M6. The tensor-fidelity paper-scale cells are single-seed, and the small fixture uses 64 anchors. Say so in the table caption.
- M7. Define every column (emp, dec, pf_tan, align_cos_tan) in one notation table in the main text.
- M8. Related work should add the classical curvature estimators from samples (local polynomial fits, Aamari & Levrard 2019 rates) and the probing-methodology line (control tasks, Hewitt & Liang 2019; Belinkov 2022 survey), since the paper speaks to probe users. Verify each citation.
- M9. Multiplicity: a Bonferroni correction over the 20 pairs barely changes the counts, so state it once.
- M10. The ‖q‖/‖p‖ numbers above come from the stored alpha = 100 arrays. Recompute them in the new runner and do not quote mine.

## Corrected proposition (with derivation)

**Setting.** At an anchor with k neighbours, let C = I − 11ᵀ/k centre vectors over the neighbourhood. The global affine probe ŷ = w·x + b₀ has residual r = y − ŷ. At the anchor split w orthogonally as w = w_T + w_rad + w_S (tangent, radial, in-sphere normal; the projectors are orthogonal on the unit sphere because Jᵀx̂ = 0). Define

- p = C X w_S, the data-side in-sphere readout on the neighbours;
- q = C q̃ with q̃_i = ½ A_S(u_i,u_i), A_S = <w_S, II^S>, u_i = g⁻¹Jᵀ(x_i − x₀), the decoder's second-order term;
- e = p − q, the part of the normal readout the second-order model misses;
- r_c = C r.

The surrogate family is ŷ_t = X(w_T + w_rad) + t s + c, with c refit, where s = p (variant S) or s = q (variant S_model, the published one).

**(a) Exact finite-sample identity.** The centred residual of ŷ_t is r_c + p − t s, so SSE(t) = ‖r_c + p − t s‖² and

  t*_s = <r_c + p, s>/‖s‖².

Hence
- t*_S = 1 + <r_c, p>/‖p‖²;
- t*_model = <p,q>/‖q‖² + <r_c,q>/‖q‖² = β(p|q) + ρ_r, equivalently 1 + <r_c + e, q>/‖q‖².

β(p|q) = cos(p,q)·‖p‖/‖q‖ is the regression slope of the probe's own normal readout on the decoder quadratic. It does not change when w_S is rescaled, so it measures fidelity, not shrinkage.

**(b) Help and hurt.** SSE(t) − SSE(0) = ‖s‖²(t² − 2t t*). So t = 1 beats t = 0 iff t* > ½, t = −1 is worse than t = 0 iff t* > −½, and ΔR²(1) = (2t* − 1)‖s‖²/SST. Help implies hurt.

**(c) Local least squares.** If the probe is refit by LS with intercept on the neighbourhood (k > D + 1), then r_c ⊥ C X v for every v. So t*_S = 1 exactly, for every label, including pure noise. For the decoder variant, <r_c, q> = <r_c, p − e> = −<r_c, e>, and so

  t*_model = 1 + <e, q − r_c>/‖q‖²,

which equals 1 iff e ⊥ (q − r_c). The outline's "local LS gives t* = 1 exactly" holds for S only. (Toy: 1.000000 vs 7.06.) If D ≥ k − 1, local OLS interpolates (r = 0) and t*_S = 1 trivially.

**(d) Global OLS and ridge.** The in-sample ridge normal equations with intercept read X̃ᵀr = αw (X̃ globally centred; α = 0 is OLS). For any fixed direction v,

  Σ_i r_i (x_i − x̄)·v = α w·v.

For any partition of the rows into patches P this splits as

  Σ_P <C_P r, C_P X v> = α w·v − Σ_P n_P r̄_P (x̄_P − x̄)·v

(expand both sides and use Σ_i r_i = 0, which the intercept guarantees). The last term is the between-patch covariance of patch-mean residual and patch-mean readout, that is, the global probe's local bias. With v = w_S (an orthogonal projection of w, so w·v = ‖w_S‖² ≥ 0), the pooled counterfactual that scales w_S on all rows has

  t*_pooled = 1 + α‖w_S‖²/‖X̃w_S‖² ≥ 1, → 1 as α → 0.

Per anchor nothing is guaranteed. t*_S − 1 = <r_c,p>/‖p‖² is that patch's share of the pooled identity plus the between-patch term, and ‖p‖² is small (normal directions carry little within-patch variance), which amplifies any local misfit. (Toy, near-OLS: per-anchor t*_S = 0.10, pooled 1.0000. Toy pooled values match the formula to 4 dp at α = 1 and 100. Stored galaxy arrays at α = 100: median t*_S 1.34-2.76, t*_S > 1 at 69-100% of anchors.) Because w_S depends on the anchor, the patch decomposition holds for one fixed v at a time.

**(e) What ridge does to the published t*.** In t*_model = β(p|q) + ρ_r, α acts on ρ_r (which scales as 1/‖w_S‖) and on the direction of w_S. It does not act on the size of β. On galaxies ‖p‖/‖q‖ ≈ 2-2.6, so t*_model > 1 can arise from β alone, and "t* → 1 as α → 0" is not a prediction for the published quantity.

**(f) Population tensor statement (the genuine extension of Liu+23).** Let u be spherically symmetric with E u_i² = s² and E u₁⁴ = 3λs⁴. Define

  <A,B>_Σ := Cov(½A(u,u), ½B(u,u)) = (s⁴/4)[2λ<A,B>_F + (λ−1) trA trB].

λ = 1 for a Gaussian, giving (s⁴/2)<A,B>_F as in main.tex Eq. 3. λ = (d+2)/(d+4) for a uniform d-ball. Suppose the manifold and label are exactly second order on the patch, x(u) = x₀ + Ju + ½II(u,u) and y = c + bᵀu + ½H(u,u), with u in an orthonormal frame. Odd moments vanish, so linear and quadratic terms decouple, and local LS over (w, c) gives Jᵀw = b and

  K* = <w_N, II> = Π_R(H)

in <·,·>_Σ, where R = {<v, II> : v ⊥ T_xM}, the row space of the flattened II. Then <H − K*, K*>_Σ = 0, so scaling K* gives t* = 1 exactly, and the improvement condition 2<H,K*>_Σ − ‖K*‖²_Σ = ‖K*‖²_Σ ≥ 0 always holds. With the radial term held fixed, the same holds for the shape part with R_S and H − K_sph. In codimension 1, R = span(A) and this gives the normal weight <H,A>_Σ/‖A‖²_Σ (compare Liu+23 Thm 2.4). If II has full rank m, as measured at every galaxy anchor, then R = Sym² and K* = H. Per-anchor expressivity is then trivial.

**(g) Local ridge (makes 3.4 precise).** Let M be the D×m flattened II, with SVD M = Σ_j s_j l_j r_jᵀ, and h = flat(H), Gaussian case. Minimising (s⁴/2)‖h − Mᵀv‖² + α‖v‖² gives

  Kflat = Σ_j f_j <h, r_j> r_j, f_j = s_j²/(s_j² + τ), τ = 2α/s⁴.

So t*_tensor = Σ f_j c_j² / Σ f_j² c_j² ≥ 1 (c_j = <h, r_j>), with equality iff there is no shrinkage. Help (t* > ½) always holds. The II spectrum therefore decides which components of the label Hessian a regularised probe can bend toward. τ = 2α/s⁴ is large for small patches, which is why the ridge effect is strong in normal directions. This is the per-anchor version of the ridge sign claim. The global probe is not a local ridge, so (d) is what holds for the global probe.

**Recommended statement for the paper.**
- Lemma = (a) + (b).
- Proposition 1 = (c) + (d): what local and global LS force on the counterfactual, with the explicit per-anchor deviation and pooled ridge shift.
- Proposition 2 = (f) + (g): population projection and spectral shrinkage, in the Σ-metric.
- Remark = (e).

The proofs fit in an appendix page. The toy check is `prop_check.py` in this folder.

## The four new-work items: right ones, right size?

| item | verdict | change |
|---|---|---|
| N1 ridge sweep | Right idea, wrong target | Sweep t*_S (pooled and per anchor; prediction (d)) and the β/ρ_r split of t*_model. Plain OLS, no PCA truncation. CPU, about 1 day once `scaling_at_anchor` stores <p,q>. Galaxy 10 encoders. QM9 only alongside N3′. |
| N2 multi-direction fixture | Essential, but under-specified | The fixture must have r > 1 normal bending directions with a known spectrum (one flat, one decaying), so it validates erank/k90 recovery, subspace recovery and w-dependent contraction. Add label noise (split-half cosine), the signed magnitude ratio (I6), and a Swiss-roll-style notebook. About 3-5 days. |
| N3 QM9 spectrum + ViT-B seeds/width | Wrong priority | Replace with N3′, the decoder-prior and decoder-free gates (C3): random-init decoder; AE on a covariance-matched Gaussian; AE on column-permuted embeddings; local-polynomial II; seed/width subspace overlap against k/m (save singular vectors); d sweep (I9). Mostly CPU on stored geometry plus a few refits, about 1 week. QM9 spectrum is second priority (GPU). |
| N4 proof + numerical check | Fine; fold into N0 | — |
| **N0 (new, cheap, do first)** | Missing | Store <p,q> and ‖e‖ per anchor; a mismatched-II null (II from another anchor, or a Haar tangent rotation); the neighbourhood-residual baseline and |Hess y| baseline with bootstrap CIs; the dec mismatch column next to emp; the signed TF magnitude ratio. CPU, 1-2 days, all from stored geometry and records. |
| Optional | Strong if feasible | Untrained-encoder or pixel-PCA embedding of the same galaxies (needs images and GPU), so the paper can say "learned". Without it, narrow the Section 5 title (I11). |

## Revised outline (recommended structure)

1. **Introduction.** Probes as instruments, and global R² hiding local error. Contributions in this order. (i) A validated pointwise II instrument for high-D embeddings, with its limits. (ii) The measured II spectrum of ten foundation-model embeddings: graded, not low rank, set against decoder-prior and decoder-free baselines. (iii) What a probe's normal component reads: mostly off-model variation at d = 16, and a counterfactual whose outcome least squares largely fixes in advance (Lemma and Propositions). (iv) Local residual curvature explains part of local error for magnitude and redshift, adds little beyond label curvature, and does not beat neighbourhood residuals as a predictor. An explicit "not claimed" paragraph.
2. **Background and related work.** Add sample-based curvature estimation and probing methodology (M8). State Liu+23 Thm 2.4 exactly.
3. **Setup and classical facts.** Identity, sphere split, residual expansion in the Σ-metric (I2). One page.
4. **Instrument.** Decoder II; validation on the trace, the full tensor, and the new multi-direction fixture (N2); decoder-free cross-check (I10, C3b); known biases. Main text has one summary table; full tables go to the appendix.
5. **How the embeddings bend.** Spectrum across ten encoders with null bands (random-init, Gaussian-matched, column-permuted, local-polynomial), seed/width subspace stability, d sweep, DINOv3 ladder (descriptive). Shape vs sphere and the mean-curvature failure, briefly, with details in the appendix.
6. **What the probe's normal component reads, and what scaling it can show.** Lemma and Propositions (proofs in the appendix). ‖q‖/‖p‖ and cos(p,q) per encoder and d. Variant S with the ridge sweep confirming (d). Variant S_model as an instrument check with the mismatched-II null. One figure: t*_S against α, pooled and per anchor, with the β/ρ_r split.
7. **Local residual curvature and local accuracy.** emp and dec columns, block-bootstrap CIs, thinned anchors, |Hess y| and residual baselines, held-out ΔR². Tuned probe as descriptive.
8. **Transfer to molecules (QM9).** Mismatch transfers for 3 of 4 properties. The counterfactual does not, because surrogate fidelity is near 0. Tuned-probe weakening and the ChemFM reversal. Pipeline caveats.
9. **Discussion and limitations.** What a test of "encoders bend toward physics beyond least squares" would need (the revised Test-1 design). Off-model variance. One galaxy sample. Label-Hessian noise. Estimator biases (tilted-tangent leak).

**Appendices.** Proofs and toy check; full tensor-fidelity tables; per-encoder robustness, scaling and QM9 tables; permutation p-values (relegated); QM9 pipeline (tokenizer, d choice including the tle == mle disclosure, duplicates); the mean-curvature appendix (existing E); notebooks list.

Main text holds about 5 figures and 3 tables: instrument validation, spectrum with null bands, the t*_S ridge sweep, ‖q‖/‖p‖ against d, the diagnostic table with baselines, and a QM9 summary. Everything that is a per-encoder × per-label grid goes to the appendix.
