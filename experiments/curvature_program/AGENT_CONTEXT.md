# Curvature programme — agent context

Read this before proposing a new geometry run. Frozen numbers live in
`COMPLETE.json` / `decision.json` / tables, not in this file or any manuscript.

Broader manuscript onboarding (do not overwrite): repository-root `CONTEXT.md`.
Scoped agent rules: `experiments/geometry/AGENTS.md`.

## Source-of-truth hierarchy

1. `COMPLETE.json`
2. `decision.json`
3. `summary.json`
4. machine-readable CSV / parquet / JSON tables
5. `REPORT.md` and `METHODS.md`
6. manuscript prose
7. historical notes or conversational summaries

A manuscript must never override the corresponding machine-readable results.
An experiment without `COMPLETE.json` is incomplete unless its protocol uses a
different completion marker (the NeurReps submission-validation tree uses
`decision.json`).

## 1. Local quadratic estimator Q

For an anchor \(x_0\), tangent frame \(J\), and local coordinate \(u\),

\[
f(u)=x_0+Ju+\tfrac12 Q(u,u).
\]

Decompose

\[
Q=Q_T+Q_R+B^S,
\]

where

- \(Q_T=P_T Q\) is **manifold-tangential** coordinate acceleration;
- \(Q_R=-g\otimes x_0\) is **sphere-radial** curvature forced by unit normalization;
- \(B^S=P_{N,S}Q\) is **sphere-normal residual** bending.

Mean-curvature vector (averaged convention used by production Q tables):

\[
H^S=\frac1d\,g^{ab}B^S_{ab}.
\]

Split-half trace statistic:

\[
K_H^{\mathrm{cross}}=\langle H_A^S,H_B^S\rangle.
\]

It is **signed**, **not clamped**, an approximately noise-debiased cross estimate,
**trace-only**, finite-patch and \(k\)-dependent. It is **not** intrinsic curvature
and is **not** interchangeable with full \(B^S\) energy.

Full directional statistic:

\[
K_{\mathrm{dir}}^{\mathrm{cross}}
=
\frac{
2\langle B_A^S,B_B^S\rangle_F
+
\langle\operatorname{tr}B_A^S,\operatorname{tr}B_B^S\rangle
}{d(d+2)}.
\]

Quadratic features use the Frobenius-preserving \(\sqrt{2}\) convention
(\(\varphi_{aa}=\tfrac12 u_a^2\), \(\varphi_{ab}=u_a u_b/\sqrt{2}\)).
Never average the two halves and then square. Never clamp a negative cross estimate.

## 2. Decoder estimator D

Raw decoder:

\[
J=DF,\qquad g=J^\top J,\qquad
II^E=(I-P_T)D^2F,\qquad
H^E=g^{ab}II^E_{ab}.
\]

Normalized decoder \(\widetilde F=F/\|F\|\). For a manifold inside the unit sphere,

\[
II^E=II^S-g\otimes x,\qquad H^E=H^S-dx
\]

under the **unnormalized** (not \(1/d\)) trace convention.

Distinguish carefully:

| Name | Object |
|---|---|
| D-full / raw | full Euclidean curvature of the raw decoder image |
| D-normalized-full | full Euclidean curvature after differentiating through normalization |
| D-residual | sphere-normal residual curvature \(II^S\) |
| Q | finite-patch quadratic curvature |

These are **not** estimates of one identical scalar law.

## 3. Probe-aligned curvature

For an affine probe \(\hat y=b_0+w^\top x\),

\[
w_N=(I-P_T)w,\qquad
B_w{}_{ab}=\langle w_N,II_{ab}\rangle.
\]

Identity:

\[
\operatorname{Hess}_{\mathcal M}(w^\top x)=\langle w_N,II\rangle.
\]

For normalized embeddings

\[
B_w
=
\underbrace{\langle w_N,II^S\rangle}_{\text{shape}}
-
\underbrace{(w^\top x)\,g}_{\text{sphere}}.
\]

The sphere component satisfies

\[
\|B_w^R\|_g=\sqrt{d}\,|w^\top x|=\sqrt{d}\,|\hat y-b_0|.
\]

That sphere term is **algebraically coupled to the probe prediction**. It is a
diagnostic, not evidence of representation-specific bending.

For Q, the task-aligned squared-norm estimate must remain split-half:

\[
E_{Q,w}^{S,\mathrm{cross}}
=
\big\langle
\langle w_N,B_{Q,A}^S\rangle,
\langle w_N,B_{Q,B}^S\rangle
\big\rangle_g.
\]

Never average the halves and then square. Never clamp negative cross estimates.
When a statistic contains \(w\), use leakage-safe train-only weights.

## 4. Label-Hessian mismatch

Label intrinsic Hessian \(H_y=\operatorname{Hess}_{\mathcal M} y\).
Probe-induced Hessian \(B_w=\langle w_N,II\rangle\). Mismatch \(\Delta=H_y-B_w\).

Local error expansion (approximate):

\[
\mathbb E_{\mathrm{patch}}(y-\hat y)^2
=
\bar c^2+s^2\|\delta\|^2
+\tfrac12 s^4\|\Delta\|_F^2
+O(s^4\|\delta\|)+O(s^6).
\]

Consequences:

- probe-aligned curvature alone has **no universal marginal sign**;
- error depends on the **mismatch** between label curvature and probe-induced curvature;
- a full \(d=16\) label Hessian needs **136** quadratic coefficients;
- the cross-model label-Hessian experiment **failed its frozen reliability gate**;
- a positive mismatch-size association **cannot** be interpreted mechanistically when
  \(H_y\) is unreliable and alignment is null.

Use train-only labels for \(H_y\). Distinguish a **null** Hessian from an
**unstable** Hessian.

## 5. Terminology

Use: manifold-tangential; sphere-radial; sphere-normal residual;
pointwise decoder curvature; finite-patch quadratic curvature;
probe-aligned second fundamental form; label-Hessian mismatch.

Avoid “tangential curvature” unless the tangent space is named.
Do not write a generic `curvature()` that hides the estimator.

Named functions in `geometry.curvature`:

- `decoder_full_euclidean_second_fundamental_form`
- `decoder_sphere_residual_second_fundamental_form`
- `quadratic_sphere_residual_second_fundamental_form`
- `probe_aligned_second_fundamental_form`

Also distinguish global \(R^2\), global MSE, patch performance, and adaptation gain.
They are not interchangeable outcomes.

## Artifact-backed result synthesis

1. **ViT-B Q vs global probes exists at frozen \(d=16\), \(k=2048\).**
   `rho16(KHcross, MSE) = 0.2270478922763529`
   (`physics_curvature_probe_submission_validation`, label
   `claim_supported_but_scale_dependent`).
   Controlled `rho_ctl(KHcross, R_G^2) = -0.2404841119636992`.

2. **Higher ViT-B `KHcross` corresponds to worse global-probe performance and
   greater relative benefit from local adaptation; patches remain worse on average.**
   `rho_ctl(KHcross, Δ_adapt) = +0.15334238492921803`;
   mean `Δ_adapt = -0.1011990469212172`
   (`physics_local_probe_adaptation`, label
   `curvature_predicts_local_direction_adaptation`).

3. **Those associations survive geometry resampling** (32+32 refits, original
   sign retained, intervals exclude 0). Label
   `q_global_and_adaptation_associations_geometry_robust`.
   This is a perturbation of the same representation population, not a new survey.

4. **The result is rank- and neighbourhood-bandwidth-conditioned**
   (`scale_magnitude_varies=true`; label `claim_supported_but_scale_dependent`).

5. **Cross-model Q does not support one universal joint global-penalty / local-adaptation law.**
   Label `representation_specific_effect`. Only `vit_base` has both sides.
   DINOv3 / CLIP / ConvNeXt flip the global sign; ViT-L has adaptation without the global penalty.

6. **Full \(K_{\mathrm{dir}}\) and trace-only \(K_H\) differ.**
   Aniso share \(0.943\). ViT-B `ρ(K_dir, Δ_adapt) = -0.1594762627172955`
   vs `ρ(K_H, Δ_adapt) = +0.15334238492921803`.
   Label `full_curvature_partial_cross_model_replication`.
   Most residual bending is trace-free; the original effect is tied to the trace statistic.

7. **Local ViT-B labels have held-out quadratic structure, but fitted \(B^S\) is
   full rank (136) in quadratic-feature space.**
   `median Δ_Q = 0.020581617601622228`. Parent label
   `quadratic_chart_link_unresolved`. Audit interpretation
   `geometry_regularized_quadratic_decoding` (COMPLETE only; does not replace the parent label).

8. **Pointwise D-residual is useful on matched analytic fixtures
   (`F4` clean \(ρ=0.8193223443223443\)) but does not reproduce the ViT-B Q story.**
   `ρ_ctl(C_H, R_G^2) = +0.02970575697233952` (null).
   Label `pointwise_residual_probe_relation_unresolved`.
   Dual-estimator historical gates: `neither_estimator_validated`
   (residual internal gates passed; D-full and Q tensor gates failed).

9. **Leakage-safe task-aligned Q predicts held-out global-probe MSE in aggregate on ViT-B.**
   `P2 = +0.11614541311898466`, CI \([0.062027437327914936, 0.15559769877589985]\),
   Holm \(p=0.00019998000199980003\).
   Positive: `photo_z +0.1740391038955837`, `smooth_fraction +0.20223387735892243`,
   `stellar_mass +0.12517032092216843`. Opposite: `mag_r_desi -0.03686164970073586`.
   Label `quadratic_task_aligned_effect_only`.

10. **Matching decoder task-aligned statistic is null.**
    `P1 = -0.0016449827122028832`, Holm fail. Same tree / label as (9).

11. **Cross-model Hessian mismatch: positive mismatch-size aggregate, failed reliability gate, null alignment.**
    `P1 = +0.10804301439243084` (Holm pass); `P2 = -0.011993127617845984` (fail).
    **Decision label remains `label_hessian_unreliable`.** Do not reinterpret.

12. **None of the above warrants** a causal claim; a universal curvature law;
    a claim of intrinsic manifold curvature; a claim that local probes outperform
    globally on average; a claim that all labels behave identically; or a claim
    of cross-model replication of the joint ViT-B Q law.

Later related trees (do not collapse into the Q law):

- `q_task_aligned_replicates_across_models`: `R1 = +0.05794641314866791`; CLIP negative.
- `cross_model_pointwise_patch_degradation_only`: `H1 = -0.20682288626818188`; adaptation H2 null.

## Safety

Never modify a completed output tree. Create a new experiment/output path.
Stop when a reliability or identity gate fails. Write `COMPLETE.json` only after
required stages finish. Never edit a manuscript automatically.
Do not launch new broad sweeps without explicit authorization.

## Proposed next experiment (not run)

One-dimensional residual-direction test: fit
\(r_L(u)=a_0+a_1^\top u\) vs
\(r_B(u)=a_0+a_1^\top u+c\,q_w(u)\) with
\(q_w(u)=\tfrac12 u^\top B_w u\) on training-labelled neighbours;
evaluate \(\Delta_B=\mathrm{MSE}(r_L)-\mathrm{MSE}(r_B)\) on untouched neighbours.
This avoids estimating 136 unrestricted \(H_y\) coefficients.
See the human README, section 16.
