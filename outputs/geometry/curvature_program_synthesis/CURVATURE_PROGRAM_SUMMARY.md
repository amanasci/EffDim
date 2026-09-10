# Curvature and Probe Performance: Programme Synthesis

Documentation-only. No new experiments, no model training, no probe or estimator refits, no manuscript edits, and no historical decision labels overwritten. Numerical values below are copied from completed artifacts; trivial consistency checks only.

Canonical host root: `/home/angus/platonic-universe/outputs/geometry/`.  
This worktree: `/home/angus/Documents/Code/PlatonicUniverse/EffDim-worktrees/SAE-shared-basis/outputs/geometry/`.

Shared frozen protocol for ViT-B real-data analyses unless a row says otherwise: \(d=16\), \(k=2048\), \(n=512\) anchors, target `mag_r_desi`, controls `{log_knn_radius, local_label_variance, local_evaluation_count}`.

---

## 1. Executive summary

The programme measured three different geometric objects and then asked whether any of them predicted frozen probe outcomes. Once those objects are named separately, the results are coherent rather than contradictory.

**What is robust.** Austin’s D-full formula is recovered on clean non-spherical fixtures (cubic ρ = 0.939, ridge ρ = 0.990). Differentiating through decoder normalization yields D-residual, which recovers sphere-relative mean curvature on known fixtures (F4 clean ρ(\(H^S\)) = 0.819; intrinsic Scal ρ = 0.862) and remains usable under sampling and noise (combined-stress ρ = 0.737; sampling reliability \(r_{\mathrm{rel}}=0.829\)). Q, as a split-half finite-patch sphere-normal quadratic-bending statistic, carries moderate rank information versus its matched patch oracle T2 (ρ(\(K_H^{\mathrm{cross}},K_{H,\mathrm{T2}}^\star\)) = 0.523; tensor cosine = 0.618). On frozen ViT-B, the controlled associations \(\rho_{\mathrm{ctl}}(K_H,R_G^2)=-0.240\), \(\rho_{\mathrm{ctl}}(K_H,\mathrm{MSE}_G)=+0.227\), \(\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})=+0.153\) keep sign under 32 conditional-support and 32 object-support Q refits. Local unrestricted quadratic label gain is real (median \(\Delta_Q=0.0206\)).

**What failed.** Neither historical instrument met the strict known-answer tensor-recovery gates (`neither_estimator_validated`). Q is not a validated pointwise manifold-curvature estimator. D-full \(\|H^E\|\) is nearly constant on unit-sphere fixtures, so Spearman of that magnitude is not a ranking test. The joint ViT-B Q package (global-probe penalty plus positive relative adaptation) does not reproduce across DINOv3, CLIP, ConvNeXt-B, or ViT-L (`representation_specific_effect`). Full directional Q (\(K_{\mathrm{dir}}^{\mathrm{cross}}\)) does not reproduce the remembered global/local sign pattern (`full_curvature_partial_cross_model_replication`). Fixture-validated D-residual does not reproduce the ViT-B Q story: it is globally null and negatively associated with patch \(R^2\) and adaptation (`pointwise_residual_probe_relation_unresolved`). Patch probes are worse on average (mean \(\Delta_{\mathrm{adapt}}=-0.101\); fraction of anchors with patch \(R^2>\) global \(R^2\) = 0.00195). Quadratic gain does not mediate adaptation. Independent-dataset replication does not exist. Cross-model D-residual has not been run.

**What changed interpretation.** Early language treated “curvature” as one quantity. Later audits split full Euclidean mean curvature, sphere-residual mean curvature, finite-patch \(K_H^{\mathrm{cross}}\), and full directional \(K_{\mathrm{dir}}^{\mathrm{cross}}\). The original ViT-B associations remain as associations of Q, not as recovered pointwise geometry. `neither_estimator_validated` answers a gate, not operational uselessness. `quadratic_chart_link_unresolved` remains the QLCA decision label; `geometry_regularized_quadratic_decoding` is an audit interpretation only.

**What remains unresolved.** What Q captures beyond moderate T2 rank; why Q and D-residual have opposite adaptation signs; whether D-residual probe relations replicate across encoders; whether cross-model Q heterogeneity is geometry-resampling stable; realistic observation/manifold-thickness scale; bandwidth dependence beyond frozen \(d=16,k=2048\).

**Why this is not contradictory.** D-residual estimates pointwise geometry of a globally learned decoder surface. Q estimates mesoscopic quadratic structure of a sampled neighbourhood and is affected by bandwidth, sampling measure, PCA tangents, ridge, and split variance. Those estimands need not correlate (raw \(\rho(C_H,K_H)=-0.284\)) and need not share probe signs. Stable outcome association is not exact geometric calibration. Imperfect tensor recovery is not absence of operational utility.

The seven required statements:

1. **D-full** reproduces clean non-spherical fixture results but its magnitude is poorly discriminative on nearly spherical normalized data.
2. **D-residual** is the strongest validated pointwise geometric instrument.
3. **Q** carries moderate finite-patch rank information but is biased and sampling-dependent as a geometric estimator.
4. The **ViT-B Q associations** with global probe error and relative local adaptation are stable under the completed geometry-resampling audit.
5. **D-residual does not reproduce the ViT-B Q story**: it is globally null and negatively associated with patch performance/adaptation.
6. Existing **cross-model Q** results do not reproduce the joint ViT-B pattern.
7. **Cross-model D-residual remains untested.**

---

## 2. Definitions and estimand map

On a unit-sphere immersion, the second fundamental form decomposes as

\[
II^E = -g\otimes x + B^S.
\]

The sphere-radial piece \(-g\otimes x\) is genuine Euclidean curvature imposed by normalization (\(\|x\|=1\)). \(B^S=(I-\tilde x\tilde x^\top-P_T)D^2F\) is additional bending within the unit sphere. \(\|H^E\|^2=d^2+\|H^S\|^2\) when \(H^E=g^{ab}II^E_{ab}\) and \(H^S=g^{ab}B^S_{ab}\) (unaveraged traces). Low Spearman of \(\|H^E\|\) on a near-sphere is a degenerate rank target, not automatic estimator failure.

| Name | Formula | Pointwise or finite-patch | Full or sphere-residual | Trace or full tensor | Sampling dependence | Validated target | Appropriate scientific wording |
|---|---|---|---|---|---|---|---|
| D-full | \(II_D^E=(I-P_T)D^2F\), \(H_D^E=g^{ab}II^E_{D,ab}\) (raw decode; **no** \(1/d\)) | pointwise | full Euclidean | unaveraged trace of \(II^E\) | decoder fit / init | analytic \(H^E\) on non-spherical fixtures; sphere \(\|H^E\|\) rank degenerate | historical pointwise full Euclidean decoder mean curvature |
| D-residual | \(\widetilde F=F/\|F\|\), \(B_D^S=(I-\tilde x\tilde x^\top-P_T)D^2\widetilde F\), \(H_D^S=g^{ab}B^S_{D,ab}\) | pointwise | sphere-residual | dual-estimator tables: **unaveraged** \(g^{ab}B_{ab}\); ViT-B probe relation \(C_H\): **averaged** \(H^S=(1/d)\mathrm{tr}_g B^S\) | decoder fit / init | analytic \(B^S\), \(H^S\), Scal on F0/F4 | fixture-validated pointwise sphere-residual decoder curvature |
| Q \(K_H^{\mathrm{cross}}\) | \(\langle H_A^S,H_B^S\rangle\) with \(H=(1/d)\mathrm{tr}(\mathrm{whiten} B)\) on split-half Hessians | finite-patch | sphere-normal residual of a local quadratic | trace (mean-curvature energy) | neighbourhood, PCA tangent, ridge, split, sampling measure | sampling-matched patch oracle T2, not pointwise \(B^S\) | split-half finite-patch sphere-normal quadratic-bending statistic (trace) |
| Q \(K_{\mathrm{dir}}^{\mathrm{cross}}\) | \((2\langle B_A,B_B\rangle_F+\langle\mathrm{tr}B_A,\mathrm{tr}B_B\rangle)/(d(d+2))\) | finite-patch | sphere-normal residual | full (mean + traceless) | same as \(K_H\) | T2 tensor / directional oracle; not interchangeable with \(K_H\) | split-half finite-patch sphere-normal quadratic-bending statistic (full directional) |
| Intrinsic Scal | \(d(d-1)+\|\mathrm{tr}_g B^S\|^2-\|B^S\|_g^2\) | pointwise (or patch analogue) | sphere-residual | scalar from full \(B^S\) | as parent estimator | analytic Scal on F4 | Gauss/intrinsic scalar from residual second fundamental form; **not** \(K_H\) or \(K_{\mathrm{dir}}\) |
| \(\Delta_Q\) | held-out unrestricted quadratic gain of the physical label in local chart coordinates | finite-patch (label model) | n/a | n/a | neighbourhood / chart | held-out label risk, not geometry | unrestricted local quadratic label gain; not a curvature estimator |

Do not describe Q as a validated pointwise manifold-curvature estimator. Do not call \(K_H\) or \(K_{\mathrm{dir}}\) intrinsic curvature.

---

## 3. Chronology of experiments

Historical conclusions are left in the order they were frozen. Later distinctions are listed as subsequent reinterpretation, not as if they were known at the time.

| When / tree | Scientific question | Estimator | Data / models | Frozen scale | Main result | Decision label | Subsequent reinterpretation | Source |
|---|---|---|---|---|---|---|---|---|
| NeurReps submission validation | Does Q trace predict OOF probe **error** at \(d=16\)? | Q \(K_H^{\mathrm{cross}}\) | ViT-B, mag_r | \(d=16,k=2048\) | ρ(\(K_H\), MSE) = 0.227; scale-dependent | `claim_supported_but_scale_dependent` | Later work splits MSE vs \(R^2\) vs adaptation and mean vs full Q | `outputs/geometry/physics_curvature_probe_submission_validation/decision.json` |
| Local probe adaptation | Does \(K_H\) predict relative local-vs-global probe **adaptation**? | Q \(K_H^{\mathrm{cross}}\) | ViT-B | same | \(\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})=+0.153\); patches worse on average | `curvature_predicts_local_direction_adaptation` | Exploratory audit: `curvature_predicts_relative_local_adaptation`; still not absolute patch superiority | `outputs/geometry/physics_local_probe_adaptation/` (host) |
| QLCA | Does the label have held-out quadratic structure aligned with Q? | Q + unrestricted/BS quadratic probes | ViT-B | same | median \(\Delta_Q=0.0206\); ρ(\(K_H,\Delta_Q\))=0.111; synthetic shuffle **fails** | `quadratic_chart_link_unresolved` | Audit interpretation `geometry_regularized_quadratic_decoding` (not in `decision.json`) | `outputs/geometry/physics_quadratic_label_chart_alignment/` (host) |
| Cross-model Q | Does the ViT-B \(K_H\) package replicate? | Q \(K_H^{\mathrm{cross}}\) | 5 encoders | same 512 anchors | joint C_G+C_A only on ViT-B | `representation_specific_effect` | Full-Q audit does **not** overwrite this label | `outputs/geometry/physics_cross_model_curvature_local_adaptation/` (host) |
| Component decomposition | Do mean and traceless Q play the same predictive role? | \(K_H\), \(K_{\mathrm{tf}}\), \(K_{\mathrm{dir}}\) | 5 encoders | same | distinct roles; traceless drives Hessian alignment | `distinct_mean_and_traceless_predictive_roles` | Exploratory; does not replace CMCLA | `outputs/geometry/physics_curvature_component_predictive_decomposition/` |
| Full directional Q | Does historical \(K_{\mathrm{dir}}\) recover the Q–probe pattern? | Q \(K_{\mathrm{dir}}^{\mathrm{cross}}\) | 5 encoders | same | partial global MSE signal; adaptation aggregate **flips** | `full_curvature_partial_cross_model_replication` | Historical \(K_H\) tables remain \(K_H\) | `outputs/geometry/physics_cross_model_full_curvature_reconciliation/` (host) |
| AE scale-match | Does a 600-epoch AE \(H\) match Q? | D-like AE vs Q | ViT-B | 600 ep, not 400-epoch raw-decode protocol | ρ(point \(H\), \(K_H\))=0.202; ρ vs \(R^2_G\) n.s. | **no COMPLETE / no decision.json** | Not the D-residual protocol | `outputs/geometry/physics_ae_local_patch_scale_match/summary.json` |
| Known-fixture audit | Do D and Q recover known tensors? | mixed (later identified estimand mismatch) | F0–F4 at D=768 then D=28 lineage | mixed k | mean vs full divergence; both fail gates | `mean_vs_full_curvature_divergence` | Dual-estimator re-scored matched targets | `outputs/geometry/known_curvature_point_patch_fixture_audit/` |
| Instrument failure | Where does Q lose the tensor? | Q ablations Q1–Q5 | F0–F4 | k=2048/512 | exact-frame Q1 works; PCA is the principal loss | `quadratic_tangent_estimation_failure` | Explains operating-characteristic Q bias | `outputs/geometry/known_curvature_instrument_failure_localization/` (COMPLETE on host) |
| Austin reproduction | Is historical D-full the raw unaveraged formula? | D-full | Swiss, cubic, ridge | 400 ep, (250³) SiLU | 3/4 cells match; Swiss ρ discrepancy | `colleague_decoder_results_reproduced` | Confirms D-full ≠ D-residual | `outputs/geometry/pointwise_decoder_curvature_reproduction/` |
| Dual-estimator robustness | Do historical D-full and Q pass matched gates under stress? | D-full, D-residual, Q | F0/F1/F2/F4, D=28 | k=1024, 12 AEs | residual passes F0/F4; D-full and Q fail gates | `neither_estimator_validated` | Operating-characteristics: residual still useful; Q moderately informative | `outputs/geometry/known_curvature_dual_estimator_robustness/` |
| Operating characteristics | Do they retain rank information under stress? | D-full, D-residual, Q | same fixtures | 8 new AEs | see §8.4 | three estimator-specific labels; prior gate **not** overwritten | Separates gates from utility | `outputs/geometry/known_curvature_estimator_operating_characteristics/` |
| D-residual vs probes | Does validated \(C_H\) reproduce the Q–probe story on ViT-B? | D-residual \(C_H=(1/d)\mathrm{tr}_g B^S\) | ViT-B, 3 seeds | d=16, 400 ep | P1 null; P2 negative; P3 opposite | `pointwise_residual_probe_relation_unresolved` | Pointwise ≠ mesoscopic | `outputs/geometry/physics_pointwise_residual_curvature_probe_relation/` |
| Q geometry resampling | Do Q–probe associations survive refitting Q on resampled neighbourhoods? | Q \(K_H^{\mathrm{cross}}\) | ViT-B | 32+32 refits | signs retained; intervals exclude 0 | `q_global_and_adaptation_associations_geometry_robust` | Fixed-dataset robustness, not a new population | `outputs/geometry/physics_q_geometry_resampling_stability/` |

Predecessor data (no scientific COMPLETE): `outputs/geometry/physics_multimodel_graph_prior_quadratic/` (embeddings, neighbourhoods, folds).

---

## 4. Original ViT-B Q results

Authoritative controlled Spearman values, reproduced exactly from `physics_local_probe_adaptation/decision.json` and `parity.json`, and re-checked in later parity files (`physics_q_geometry_resampling_stability/parity.json`, `physics_pointwise_residual_curvature_probe_relation/parity.json`):

\[
\rho_{\mathrm{ctl}}(K_H,R_G^2)=-0.2404841119636992
\]

\[
\rho_{\mathrm{ctl}}(K_H,\mathrm{MSE}_G)=+0.22704789227635297
\]

\[
\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})=+0.15334238492921803
\]

Raw (uncontrolled) companions: \(\rho(K_H,R_G^2)=-0.41243\); \(\rho(K_H,\mathrm{MSE}_G)=+0.35599\); \(\rho(K_H,\Delta_{\mathrm{adapt}})=+0.07591\) (the adaptation association is **stronger after controls**). Sources: `physics_q_geometry_resampling_stability/parity.json`.

**How to read this.** \(R_G^2\) is global OOF coefficient of determination; \(\mathrm{MSE}_G\) is global OOF mean squared error; \(\Delta_{\mathrm{adapt}}\) is relative local-minus-global probe error (positive means the patch probe improves on the global probe). Mean \(\Delta_{\mathrm{adapt}}=-0.1012\) (`physics_pointwise_residual_curvature_probe_relation/correlation_contrasts.csv`); fraction of anchors with patch \(R^2>\) global \(R^2\) = 0.00195. Patch probes are **worse on average**. The positive \(\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})\) is a **relative** benefit: higher \(K_H\) ranks with a smaller patch *disadvantage*, not with patch superiority.

Conditioning: \(d=16\), \(k=2048\), ViT-B, mag_r_desi, 512 anchors. Q is a **finite-patch** statistic.

### Geometry-resampling audit

Source: `outputs/geometry/physics_q_geometry_resampling_stability/`. Label: `q_global_and_adaptation_associations_geometry_robust`. Runtime 1678.9 s. Association parity exact; 8-anchor Q refit parity median abs diff = 0.

| Scheme | Meaning | n | Field reliability (pairwise / rank ICC) |
|---|---|---|---|
| A conditional-support | re-partition the same k=2048 neighbours into split halves | 32 | 0.899 / 0.903 |
| B object-support | keep ~80% of the 16384-object cloud (delete-20% of the **same** representation population) | 32 | 0.809 / 0.817 |

Object-support is **not** an independent new dataset.

| Outcome | Original | A median [q025, q975] | B median [q025, q975] | Sign retention A/B | Interval excludes 0 (expected direction) |
|---|---|---|---|---|---|
| \(\rho_{\mathrm{ctl}}(K_H,R_G^2)\) | −0.2405 | −0.2257 [−0.2595, −0.1821] | −0.2190 [−0.2832, −0.1139] | 1.00 / 1.00 | yes / yes |
| \(\rho_{\mathrm{ctl}}(K_H,\mathrm{MSE}_G)\) | +0.2270 | +0.2163 [0.1732, 0.2455] | +0.2077 [0.0944, 0.2675] | 1.00 / 1.00 | yes / yes |
| \(\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})\) | +0.1533 | +0.1529 [0.1095, 0.1930] | +0.1304 [0.0405, 0.2113] | 1.00 / 1.00 | yes / yes |

Source tables: `association_stability.csv`, `field_reliability.csv`, `decision.json`. Prior labels not overwritten: `quadratic_chart_link_unresolved`, `neither_estimator_validated`, `q_moderately_informative_sampling_dependent_statistic`.

---

## 5. Quadratic label-chart alignment

Source: `outputs/geometry/physics_quadratic_label_chart_alignment/` (host). Decision: `quadratic_chart_link_unresolved`. Runtime 159.5 s.

| Quantity | Value | File |
|---|---|---|
| median held-out unrestricted quadratic gain \(\Delta_Q\) | 0.020582 | `decision.json` |
| \(\Delta_Q\) CI | [0.019673, 0.021616] | `primary_inference.json` |
| \(\rho_{\mathrm{ctl}}(K_H,\Delta_Q)\) | 0.111249 (p_MC = 0.0075) | `decision.json` |
| raw \(\rho(K_H,\Delta_Q)\) | 0.388089 | `primary_inference.json` |
| median \(\Delta_{\mathrm{BS}}\) | 0.019606 | `decision.json` |
| fraction of UQ captured by BS | 0.937637 | `decision.json` |
| A_B median (alignment) | 2.427 vs null 0.978 | `alignment_summary.json` |
| fold cosine γ | 0.924343 | `alignment_summary.json` |
| synthetic shuffle gate | **fail** (`synth_ok: false`) | `synthetic_results.json` |

**Mediation.** \(\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})=0.153\); after controlling \(\Delta_Q\), 0.205 (`secondary_inference.json`). Quadratic gain does **not** mediate the adaptation association (the association does not shrink).

**Audit (not `decision.json`).** `outputs/geometry/physics_quadratic_label_chart_alignment_audit/COMPLETE.json` records `audit_interpretation: geometry_regularized_quadratic_decoding`, with median numerical rank of the BS map 136 (rank fraction 0.353). Energy is concentrated; the label Hessian aligns with high-energy sphere-normal quadratic-bending modes, and that alignment is traceless-driven (`physics_curvature_component_predictive_decomposition`: unique KTF vs \(\Delta_Q\) = 0.320; unique \(K_H\) n.s.).

**Supported interpretation (wording required by this synthesis, not a new decision label):**

> At the frozen chart rank and neighbourhood scale, the physical label exhibits held-out quadratic structure in local chart coordinates. Its predictive importance increases with the Q trace statistic, and the label Hessian preferentially aligns with high-energy sphere-normal quadratic-bending modes.

This is **not** proven curvature-mediated decoding. The frozen decision remains `quadratic_chart_link_unresolved` because the synthetic shuffle calibration did not pass.

---

## 6. Cross-model Q evidence

Source: `outputs/geometry/physics_cross_model_curvature_local_adaptation/per_model_primary.json` and `decision.json`. Label: `representation_specific_effect` (reason: `pattern_confined_to_vit_base`). Runtime 46602 s. Models: vit_base, dinov3, clip_base, convnext_base, vit_large. All five passed reliability gates.

Controlled associations of **\(K_H^{\mathrm{cross}}\)** (trace). Signs: C_G is vs \(\mathrm{MSE}_G\) (positive = more Q-trace, **worse** global squared error); C_R2 vs \(R_G^2\); C_P vs patch error; C_A vs \(\Delta_{\mathrm{adapt}}\).

| Model | C_G (vs MSE_G) | C_R2 (vs \(R_G^2\)) | C_P | C_A (vs \(\Delta_{\mathrm{adapt}}\)) | Mean \(\Delta_{\mathrm{adapt}}\) | Reliability (R_H) |
|---|---|---|---|---|---|---|
| ViT-B | **+0.227** | **−0.240** | +0.175 | **+0.153** | −0.101 | 0.514 |
| DINOv3 | −0.130 | −0.002 | −0.123 | −0.045 | −0.093 | 0.569 |
| CLIP-B | −0.274 | +0.270 | −0.384 | −0.179 | −0.054 | 0.487 |
| ConvNeXt-B | −0.187 | +0.231 | −0.183 | −0.255 | −0.071 | 0.589 |
| ViT-L | −0.038 | −0.190 | −0.149 | **+0.181** | −0.071 | 0.373 |

`models_with_both` (positive global-MSE penalty **and** positive adaptation) = `["vit_base"]` only. ViT-L shows positive adaptation without the global-MSE penalty. DINOv3, CLIP and ConvNeXt-B do not reproduce the ViT-B package. Five-model aggregate C_G_bar = −0.080, C_A_bar = −0.029 (`cross_model_aggregate.json`). All five models have patch probes worse on average.

Only ViT-B has completed Q geometry resampling. The frozen label `representation_specific_effect` applies to **this** cross-model \(K_H\) analysis and was not overwritten by the full-directional audit.

---

## 7. Full directional Q curvature

Source: `outputs/geometry/physics_cross_model_full_curvature_reconciliation/`. Label: `full_curvature_partial_cross_model_replication`. Runtime 9673 s. Prior trace label `representation_specific_effect` left untouched and scoped to \(K_H^{\mathrm{cross}}\) only.

Recovered production formula (`summary.json`):

\[
K_{\mathrm{dir}}^{\mathrm{cross}}
=
\frac{2\langle B_A^S,B_B^S\rangle_F+\langle\mathrm{tr} B_A^S,\mathrm{tr} B_B^S\rangle}{d(d+2)},
\]

equivalently \(\langle H_A,H_B\rangle+[2/(d(d+2))]\langle B0_A,B0_B\rangle_F\) with \(H=(1/d)\mathrm{tr} B^S\) and \(B0=B^S-H\otimes I\).

Most full residual energy is **trace-free**: ViT-B median aniso share of \(K_{\mathrm{dir}}\) = 0.943 (`estimand_comparison_table.csv`); other models ~0.99. Mean and directional Q are not interchangeable.

ViT-B specifically (`central_comparison_table.csv`): \(\rho(K_{\mathrm{dir}},\mathrm{MSE}_G)=+0.217\) vs \(\rho(K_H,\mathrm{MSE}_G)=+0.227\); \(\rho(K_{\mathrm{dir}},\Delta_{\mathrm{adapt}})=\mathbf{-0.159}\) vs \(\rho(K_H,\Delta_{\mathrm{adapt}})=\mathbf{+0.153}\) — **sign flip** on adaptation.

Five-model aggregates (`summary.json`):

| Estimand | C_G_bar | C_P_bar | C_A_bar | C_R2_bar | A_bar |
|---|---|---|---|---|---|
| \(K_{\mathrm{dir}}\) | +0.0546 | +0.0777 | **−0.0732** | −0.0246 | −0.0231 |
| \(K_H\) | −0.0803 | −0.1328 | −0.0288 | +0.0140 | +0.0526 |

The five-model \(K_{\mathrm{dir}}\) aggregate is weak and model-sensitive (C_G_bar CI does not exclude 0). Full directional Q did not reproduce the remembered global-penalty plus positive-adaptation pattern. Do not relabel historical `KHcross` tables as full curvature.

Related exploratory label: `distinct_mean_and_traceless_predictive_roles` in `physics_curvature_component_predictive_decomposition/decision.json`. Unique traceless vs MSE_G_bar = +0.077 (p_MC = 0.0002); unique \(K_H\) n.s.

---

## 8. Known-answer validation

### 8.1 Austin reproduction

Source: `outputs/geometry/pointwise_decoder_curvature_reproduction/`. Primary label: `colleague_decoder_results_reproduced`. Secondary: `reported_quantity_is_full_euclidean_curvature`. Runtime 834.3 s.

Exact recovered D-full: \(H=g^{ab}II_{ab}\) **unnormalized**; \(II=(I-P_T)D^2F\) on **raw** `decode`; \(H\) is **not** divided by \(d\); the decoder is **not** differentiated after normalization (`decision.json`).

| Cell | ρ | Historical ρ | Median cosine | Match |
|---|---|---|---|---|
| R1 Swiss roll | 0.6538 | 0.553 | 0.9995 | ρ **fails** Δ = 0.101 > 0.05; cosine/ratio pass |
| R2 cubic | 0.9393 | 0.942 | 0.9936 | pass |
| R3 ridge D=28 | 0.9898 | 0.987 | 0.9997 | pass |
| R4 ridge D=768 | 0.9869 | 0.988 | 0.9994 | pass |

`n_primary_cells_passed = 3`. The Swiss-roll discrepancy is recorded, not hidden. Radial baseline: `radial_baseline_not_applicable`.

### 8.2 Initial known-fixture audit

Source: `outputs/geometry/known_curvature_point_patch_fixture_audit/`. Label: `mean_vs_full_curvature_divergence`. Runtime 4.23 s (bounded / reused cells).

Original result: decoder ρ(\(H\)) = 0.949 and ρ(\(K_{\mathrm{dir}}\)) = 0.944 with cosine 0.999, but seed Spearman of \(K_{\mathrm{dir}}\) = 0.158 (below 0.5). Q ρ(\(K_{\mathrm{dir}}\), T2) = 0.253. F0 false \(K_{\mathrm{dir}}\) decoder = 8.47 vs Q ~ 0. Q vs pointwise \(H\) ρ = 0.009. `decoder_ok = false`, `quadratic_patch_ok = false`.

**Limitation / later mismatch.** This audit mixed full Euclidean and residual targets and mixed T2/T3 naming with later experiments. Do **not** repeat “the decoder hallucinated curvature on a great sphere” without specifying the target: on F0, **full** \(H^E\) is the genuine \(-d\,x\) of the sphere; **residual** \(B^S\) should be ~0. False residual energy is the hallucination claim; false full \(K_{\mathrm{dir}}\) on F0 is a different (mean-vs-full / scaling) issue. The dual-estimator audit re-scored matched targets.

### 8.3 Dual-estimator matched-target audit

Source: `outputs/geometry/known_curvature_dual_estimator_robustness/`. Label: `neither_estimator_validated`. Runtime 612.2 s. D=28, d=16, k=1024, 12 AEs. D-full: unaveraged raw decode. D-residual: differentiate through \(\widetilde F=F/\|F\|\); tables use **unaveraged** \(\|H^S\|\).

| Check | Value | Source |
|---|---|---|
| D-full F4 clean vector cosine | 0.99940 | `decoder_full_metrics.csv` |
| D-full F4 clean ρ(\(\|H^E\|\)) | −0.0786 (`constant_truth` false but CV tiny) | same |
| D-residual F0 energy fraction | 0.000964 | `decoder_residual_metrics.csv` |
| D-residual F4 clean ρ(\(H^S\)) | 0.8193 | same |
| D-residual F4 Scal ρ | 0.8618 | same |
| D-residual F4 S3+N4 ρ | 0.7369 | same |
| Q vs T2 median tensor cosine | 0.6176 | `quadratic_T2_metrics.csv` |
| T2 vs T3 median tensor cosine | 0.7482 | `quadratic_T3_metrics.csv` |
| Q vs pointwise \(K_H\) ρ | 0.5232 | `quadratic_pointwise_metrics.csv` |

Gates: `decoder_full_clean_ok=false`, `decoder_residual_clean_ok=true`, `decoder_residual_stress_ok=true`, all Q tensor/intrinsic gates false, `sampling_measure_dependence_detected=true`.

`neither_estimator_validated` means the **two historical instruments** (D-full and Q) failed the **strict matched-tensor recovery gates**. It does not mean D-residual is unusable, or that Q has zero rank information.

### 8.4 Operating characteristics

Source: `outputs/geometry/known_curvature_estimator_operating_characteristics/`. Runtime 386.0 s; 8 new AEs. Labels (this audit only): `d_full_useful_on_non_spherical_clean_geometry`, `q_moderately_informative_sampling_dependent_statistic`, `d_residual_useful_but_stress_sensitive`. Prior `neither_estimator_validated` **not** overwritten.

**D-full.** Non-spherical cubic/ridge utility as in §8.1. Sphere \(\|H^E\|\) marked `rank_target_degenerate` (F4 CV = 0.00158). Residualized \(\sqrt{\max(\|H^E\|^2-d^2,0)}\) does not recover \(H^S\) rank (clean ρ = −0.034). Repeat sampling reliability of D-full magnitudes is weak.

**D-residual operating curve (F4 ρ vs analytic \(\|H^S\|\)):** clean 0.819; sparse 0.832; non-uniform dense 0.815; 1% normal noise 0.802; 5% normal 0.731; 5% isotropic 0.762; combined stress 0.737 (retained 0.90). Init reliability 0.90; sampling reliability 0.83; fraction of attenuation ceiling 0.88 (`repeat_reliability.csv`, `reliability_ceiling.csv`).

**Q.** Scalar T2 rank ρ(\(K_H^{\mathrm{cross}},K_{H,\mathrm{T2}}^\star\)) = 0.523 (moderate). Tensor T2 cosine 0.618 (moderate; below the 0.80 gate). T2–T3 cosine 0.748. Combined-stress T2 ρ = 0.070. Sampling \(r_{\mathrm{rel}}(K_H)=0.204\) (weak); \(f_{\mathrm{ceiling}}=0.196\). After residualizing density and radius, clean Q vs T2 ρ = 0.607 — not a pure density proxy.

**Density.** On non-uniform dense F4, D-residual ρ in the sparsest quintile is 0.39 vs 0.83–0.93 in denser quintiles. Q is weak in every quintile (`density_stratified.csv`).

**Noise comparator.** ViT-B decoder reconstruction RMSE / unit signal ≈ 0.213 (`noise_scale_comparison.csv`, from the D-residual physics run `final_recon` MSE ≈ 0.0453). Synthetic 1–5% RMS/\(s_x\) is **below** that scale. **Decoder reconstruction residual is an empirical scale comparator, not observational noise.** Quadratic residual/radius on existing ViT-B FCR parquet: unresolved (no matching columns). Augmentation variation: unresolved.

---

## 9. Real-data D-residual result

Source: `outputs/geometry/physics_pointwise_residual_curvature_probe_relation/`. Label: `pointwise_residual_probe_relation_unresolved`. Runtime 731.5 s. Estimator: pointwise sphere-residual decoder curvature with

\[
C_H=\bigl\|H^S\bigr\|^2,\qquad H^S=\tfrac1d\,g^{ab}B^S_{ab}
\]

through \(\widetilde F=F/\|F\|\) (`METHODS.md`). Historical full \(H^E\) is a control only. Q is not treated as ground truth.

**Three-seed reliability** (`seed_reliability.json`): seeds {0,1,2}, PlainAutoEncoder 250³ SiLU, 400 epochs. Median ρ(\(C_H\)) = 0.881; median cos(\(H^S\)) = 0.878; gates 0.7 / 0.8 **passed**. Consensus = median rank across seeds (no best-seed selection).

| Test | Controlled ρ | 95% CI | p_Holm | Result |
|---|---|---|---|---|
| P1 \(\rho_{\mathrm{ctl}}(C_H,R_G^2)\) | **+0.0297** | [−0.060, +0.115] | 1.0 | n.s. — **not** a positive global relationship |
| P2 \(\rho_{\mathrm{ctl}}(C_H,R_P^2)\) | **−0.2196** | [−0.311, −0.133] | 0.00030 | worse patch \(R^2\) |
| P3 \(\Delta\rho=\rho_{\mathrm{ctl}}(C_H,R_P^2)-\rho_{\mathrm{ctl}}(C_H,R_G^2)\) | **−0.2493** | [−0.312, −0.184] | 1.0 (preregistered **greater**) | contrast significant in magnitude but **opposite** the hypothesized sign |

\(\rho_{\mathrm{ctl}}(C_H,\Delta_{\mathrm{adapt}})=-0.2073\) (`decision.json`). Mean \(\Delta_{\mathrm{adapt}}=-0.1012\).

**Full-versus-residual control** (`full_vs_residual.csv`): residual and full decoder magnitudes both show patch penalties; residual P1 remains null (\(\rho_{\mathrm{ctl}}(C_{H,\mathrm{full}},R_G^2)=-0.018\)). Median cos(\(H^E,H^R\)) = 0.613.

**D-residual versus Q** (`q_comparison.csv`): raw \(\rho(C_H,K_H)=-0.284\); after the standard controls \(\rho_{\mathrm{ctl}}=+0.230\). Conditional on \(C_H\), \(\rho_{\mathrm{ctl}}(K_H,R_G^2)\) remains −0.254. Conditional on \(K_H\), \(\rho_{\mathrm{ctl}}(C_H,R_G^2)=0.095\). The two statistics are not substitutes.

**Scientific result:**

> On frozen ViT-B, fixture-validated pointwise sphere-residual curvature is unrelated to global-probe performance but is associated with worse patch-probe performance and a larger patch disadvantage.

Do not call the nonsignificant global value a positive relationship. The preregistered test of \(\Delta\rho>0\) was **falsified**; the negative contrast is strong but opposite the original directional hypothesis.

---

## 10. Reconciliation

D-residual estimates the **pointwise** geometry of a globally learned decoder surface (jets of \(\widetilde F\)). Q estimates **mesoscopic** quadratic structure of a sampled neighbourhood after PCA tangents, ridge, and split-half cross terms. Q is affected by bandwidth \(k\), sampling measure (T2 vs T3 cosine 0.75), tangent estimation (PCA is the principal fixture loss: `quadratic_tangent_estimation_failure`), ridge, and split variance (repeat \(r_{\mathrm{rel}}(K_H)=0.20\) on fixtures). These quantities need not correlate strongly and need not share probe signs.

Stable outcome association (Q resampling) does not imply exact geometric calibration (Q vs T2 cosine 0.62; vs pointwise weaker). Imperfect tensor recovery (`neither_estimator_validated`) does not imply that a statistic has no operational utility (`q_moderately_informative_sampling_dependent_statistic`, `d_residual_useful_but_stress_sensitive`).

Evidence hierarchy (which level each estimator currently reaches):

```text
Geometric definition
→ known-answer recovery
→ repeat reliability
→ robustness under stress
→ stable real-data association
→ cross-model generality
→ causal mechanism
```

| Estimator | Definition | Known-answer | Repeat reliability | Stress robustness | Stable real-data association | Cross-model generality | Causal mechanism |
|---|---|---|---|---|---|---|---|
| D-full | yes (raw, unaveraged) | non-spherical yes; sphere rank degenerate | weak on sphere magnitudes | vector cosine remains high | untested as residual instrument | untested | no |
| D-residual | yes (normalize-then-differentiate) | F0/F4 yes | strong (seeds 0.88; fixture sampling 0.83) | F4 stress ρ 0.74 | ViT-B: global null, patch negative | **untested** | no |
| Q \(K_H\) | finite-patch statistic | moderate T2 rank; tensor gate fail | weak on fixtures; high field ICC on ViT-B refits | sparse/combined stress collapse | ViT-B global/adaptation **yes** (resampling-stable) | **no** (representation-specific) | no |
| Q \(K_{\mathrm{dir}}\) | full directional patch | not interchangeable with \(K_H\) | moderate on fixtures | not a recovery of the \(K_H\) probe pattern | ViT-B adaptation **sign-flips** vs \(K_H\) | partial / model-sensitive | no |
| \(\Delta_Q\) | label gain, not geometry | n/a | n/a | n/a | held-out quadratic structure on ViT-B | not claimed | does **not** mediate adaptation |

---

## 11. Supported, unsupported and unresolved claims

### Supported

- D-full clean non-spherical reproduction (cubic/ridge) with recovered unaveraged raw-decode formula (`colleague_decoder_results_reproduced`).
- D-full recovers the Euclidean mean-curvature **vector** on sphere fixtures (cosine ≈ 0.999) even when \(\|H^E\|\) rank is degenerate.
- D-residual recovers F0 near-zero residual energy and F4 \(H^S\) / Scal rank; operating curve retains ~90% of clean ρ under combined stress.
- D-residual three-seed reliability on ViT-B (median ρ = 0.881, cos = 0.878).
- Q carries moderate finite-patch rank information versus T2 (\(K_H\) ρ = 0.523; tensor cosine = 0.618).
- Q is sampling-measure dependent (T2–T3 cosine = 0.748) and PCA-tangent limited on fixtures.
- ViT-B \(\rho_{\mathrm{ctl}}(K_H,R_G^2)\), \(\rho_{\mathrm{ctl}}(K_H,\mathrm{MSE}_G)\), \(\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})\) are stable under completed geometry perturbations (32+32).
- Patch probes are worse on average on these frozen outcomes.
- Held-out local quadratic label structure (\(\Delta_Q>0\)) with Q-trace association and high-energy bending-mode alignment.
- Mean versus trace-free bending are distinct (energy and predictive roles).
- Cross-model \(K_H\) joint pattern is confined to ViT-B among the five encoders tested.

### Unsupported

- Q as an accurate pointwise curvature tensor.
- Q associations as a general law across encoders.
- D-full magnitude as an informative sphere-curvature ranking.
- Curvature **causing** probe degradation or readout rotation.
- Patch probes being superior on average.
- Quadratic gain mediating adaptation.
- Independent dataset replication.
- Cross-model D-residual replication (not run).
- Calling the D-residual global association positive.
- Treating object-support delete-20% as a new population.
- Treating decoder reconstruction residual as observational noise.
- Relabeling historical \(K_H\) tables as full directional curvature.

### Unresolved

- What Q primarily captures beyond its moderate T2 relationship.
- Why Q and D-residual have opposite adaptation associations.
- Whether D-residual probe relations replicate across encoders.
- Whether cross-model Q heterogeneity is geometry-resampling stable (only ViT-B resampled).
- Realistic observation / manifold-thickness scale.
- Bandwidth and rank dependence beyond frozen \(d=16,k=2048\).
- Swiss-roll D-full ρ discrepancy versus the historical 0.553 figure.
- QLCA synthetic shuffle failure versus real-data quadratic gain.

---

## 12. Decision-label registry

No single “master” label. Each row answers one question.

| Decision label | Source experiment | What it answers | What it does not answer | Still frozen? |
|---|---|---|---|---|
| `claim_supported_but_scale_dependent` | `physics_curvature_probe_submission_validation` | \(K_H\)–OOF error association at d=16 survives FWER but varies with scale | Adaptation; residual geometry; cross-model | yes |
| `curvature_predicts_local_direction_adaptation` | `physics_local_probe_adaptation` | ViT-B \(\rho_{\mathrm{ctl}}(K_H,\Delta_{\mathrm{adapt}})>0\) | Absolute patch superiority; pointwise geometry; other encoders | yes (supersedes an earlier smoke `local_probe_result_unresolved` in the same tree) |
| `curvature_predicts_relative_local_adaptation` | `physics_local_probe_adaptation_audit` | Exploratory restatement: relative, not absolute | Does not replace the parent decision | audit tree |
| `quadratic_chart_link_unresolved` | `physics_quadratic_label_chart_alignment` | Real-data \(\Delta_Q\) exists but synthetic shuffle failed | Whether Q is pointwise curvature | yes |
| `geometry_regularized_quadratic_decoding` | `physics_quadratic_label_chart_alignment_audit` | Audit reading: anisotropic geometry-derived regularizer | **Not in `decision.json`** | audit interpretation only |
| `representation_specific_effect` | `physics_cross_model_curvature_local_adaptation` | Joint \(K_H\) global+adaptation pattern is ViT-B-specific | D-residual; \(K_{\mathrm{dir}}\); new datasets | yes |
| `distinct_mean_and_traceless_predictive_roles` | `physics_curvature_component_predictive_decomposition` | Mean and traceless Q predict different outcomes | Causal mechanism | yes (exploratory flag true) |
| `full_curvature_partial_cross_model_replication` | `physics_cross_model_full_curvature_reconciliation` | \(K_{\mathrm{dir}}\) global signal is model-sensitive; adaptation aggregate flips | Does not relabel \(K_H\) | yes |
| `mean_vs_full_curvature_divergence` | `known_curvature_point_patch_fixture_audit` | Mean curvature discards traceless bending \(K_{\mathrm{dir}}\) recovers | Later matched-target residual scoring | yes |
| `quadratic_tangent_estimation_failure` | `known_curvature_instrument_failure_localization` | PCA tangent is Q’s principal fixture loss | Real-data probe associations | yes |
| `colleague_decoder_results_reproduced` | `pointwise_decoder_curvature_reproduction` | D-full formula/protocol match on 3/4 cells | Residual geometry; Q | yes |
| `neither_estimator_validated` | `known_curvature_dual_estimator_robustness` | Historical D-full and Q fail strict tensor gates | Operational rank utility of residual or of Q-vs-T2 | yes |
| `d_full_useful_on_non_spherical_clean_geometry` | `known_curvature_estimator_operating_characteristics` | D-full operating utility off the sphere | Sphere rank; real-data probes | yes (does not overwrite `neither_estimator_validated`) |
| `q_moderately_informative_sampling_dependent_statistic` | same | Q has moderate T2 rank, sampling-dependent | Pointwise tensor recovery | yes |
| `d_residual_useful_but_stress_sensitive` | same | D-residual rank survives stress with degradation | Real-data probes | yes |
| `pointwise_residual_probe_relation_unresolved` | `physics_pointwise_residual_curvature_probe_relation` | Validated \(C_H\) is globally null and patch-negative; preregistered contrast fails | Cross-model D-residual | yes |
| `q_global_and_adaptation_associations_geometry_robust` | `physics_q_geometry_resampling_stability` | ViT-B Q–probe associations survive Q refits | Independent datasets; other encoders; exact geometry | yes |

If a label appears only in a report/audit and not `decision.json`, that is marked above.

---

## 13. Recommended manuscript framing

Do not automatically rewrite the manuscript.

**Conservative framing.** Distinguish pointwise decoder geometry from mesoscopic finite-patch quadratic bending. Report estimator validation (D-residual recovers known \(B^S\); Q does not recover the pointwise tensor; D-full recovers non-spherical \(H^E\)). Report scale and sampling dependence (\(d=16,k=2048\); T2≠T3; PCA tangents). Report robust **within-ViT-B** associations of \(K_H^{\mathrm{cross}}\) with global OOF \(R^2\)/MSE and relative adaptation, and their stability under neighbourhood resampling. Report absence of that joint pattern on the other four encoders, and report that fixture-validated D-residual does not reproduce it on ViT-B.

**Stronger but still defensible framing:**

> Pointwise and finite-scale geometric measurements capture distinct aspects of neural representation structure. In ViT-B, mesoscopic sphere-normal quadratic bending predicts global readout error and relative local-adaptation gains, whereas fixture-validated pointwise residual curvature is unrelated to global performance and instead predicts local-probe degradation.

Flag that **“quadratic bending”** is safer than unqualified **“curvature”** for Q. Keep \(R^2\), MSE, and \(\Delta_{\mathrm{adapt}}\) named. Do not claim causation, patch superiority, or cross-model generality.

---

## 14. Next experiment

**Described only. Not launched.**

Proposed: cross-model D-residual on DINOv3, CLIP, ConvNeXt-B, and ViT-L, three decoder seeds, frozen \(d=16\), the same 512 anchors and frozen probe outcomes, synchronized cross-model inference, optional bounded Q resampling on non-ViT-B models.

**Replication** of the ViT-B D-residual result would be: seed-reliable \(C_H\); \(\rho_{\mathrm{ctl}}(C_H,R_G^2)\) consistent with null; \(\rho_{\mathrm{ctl}}(C_H,R_P^2)<0\); \(\rho_{\mathrm{ctl}}(C_H,\Delta_{\mathrm{adapt}})<0\).

**Heterogeneity** would be: some encoders show a global \(C_H\)–\(R_G^2\) penalty or a sign-flipped patch association, with intervals excluding the ViT-B pattern.

**ViT-B specificity** would be: only ViT-B shows the patch-negative D-residual pattern, analogous to `representation_specific_effect` for Q.

This synthesis does not start that run.

---

## 15. Final bottom line

1. **Best validated geometrically:** D-residual (pointwise sphere-residual decoder curvature), on known fixtures and with seed-stable ViT-B fields.
2. **Most robust predictor of the original ViT-B global/adaptation outcomes:** Q \(K_H^{\mathrm{cross}}\) (finite-patch trace statistic), including geometry-resampling stability.
3. **Are those the same quantity?** No. Raw \(\rho(C_H,K_H)=-0.284\); they have opposite adaptation signs.
4. **Does the Q joint pattern generalize across models?** No, not in the five-encoder \(K_H\) table. Cross-model D-residual is untested.
5. **Strongest currently defensible scientific claim:** In frozen ViT-B embeddings, a mesoscopic split-half sphere-normal quadratic-bending statistic is associated with worse global OOF \(R^2\) / higher MSE and with relative local-adaptation gains, and those associations survive resampling of the geometry used to estimate it; a fixture-validated pointwise residual curvature of a learned decoder is a different estimand, globally null for \(R_G^2\), and associated with worse patch \(R^2\). Neither result is a causal law of curvature, a validated pointwise Q tensor, or a cross-encoder regularity.

---

## Artifact index (this synthesis)

- This document: `outputs/geometry/curvature_program_synthesis/CURVATURE_PROGRAM_SUMMARY.md`
- Claim matrix: `outputs/geometry/curvature_program_synthesis/CLAIM_MATRIX.csv`
- Source manifest: `outputs/geometry/curvature_program_synthesis/SOURCE_MANIFEST.json`
- Experiment context: `experiments/geometry/curvature_program_synthesis/CONTEXT.md`
