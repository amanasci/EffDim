# CLAIM_SOURCE_MAP

Every numerical claim in `main.tex` is traced to a frozen artifact. Paths are relative to the repository root. Values are transcribed from those files, not from conversation notes. Rounding in the manuscript is stated explicitly.

**Evidence classes**

- **Confirmatory:** frozen curvature–probe submission-validation at \(n=512\), \(k=2048\); preregistered QLCA primary tests.
- **Secondary:** QLCA secondary tables (constrained maps, LPA parity recovery, partial correlation).
- **Post-hoc audit:** `physics_quadratic_label_chart_alignment_audit` rank, truncation, Haar/isotropic/matched nulls, real-design nested shuffles.

Host provenance (not copied into this submission tree):  
`~/platonic-universe/outputs/geometry/physics_curvature_probe_submission_validation/`,  
`~/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment/`,  
`~/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment_audit/`.  
Laptop mirrors used here: `paper/curvature_neurreps/audit_outputs/`.

Target identity (typed, not resurrected): local out-of-fold coefficient of determination of a global ridge probe of **apparent \(r\)-band magnitude**, field `mag_r_desi_local_oof_r2`. Catalog vector `mag_r_desi_catalog_value` is never substituted. This is an astronomical photometric observable, not a fundamental physical property and not a spectroscopic DESI label.

---

## Configuration and target

| Manuscript wording | Artifact | Field | Value | Class |
|---|---|---|---|---|
| \(n=512\) hash-stable anchors | `…/quadratic_label_chart_alignment/primary_inference.json` | `n` | 512 | confirmatory |
| \(k=2048\) neighbours | `…/submission_validation/parity_report.json` | `k` | 2048 | confirmatory |
| frozen chart rank \(d=16\) | `…/submission_validation/predeclared_chart_positions.json` | `positions.middle` | 16 | confirmatory |
| five-fold ridge \(\alpha=100\) | NeurReps methods / QLCA `CONFIG.json` (probe reused, not refit) | protocol | \(\alpha=100\), 5-fold OOF | confirmatory |
| photometric \(r\)-band magnitude; local OOF \(R_G^2\) | `…/submission_validation/parity_report.json` | `target_id` | `mag_r_desi_local_oof_r2` | confirmatory |
| \(q=136\) quadratic coefficients at \(d=16\) | `…/quadratic_label_chart_alignment_audit/rank_audit_summary.json` | `q` | 136 \(=d(d+1)/2\) packed symmetric Hessian terms | post-hoc audit |
| \(B=10^4\) permutations | `…/quadratic_label_chart_alignment/primary_inference.json` | `n_perm` | 10000 | confirmatory |
| \(B=2000\) bootstraps | same | `n_boot` | 2000 | confirmatory |

Note on \(q\): `rank_audit_summary.json` records `q=136`. That is the dimension of the packed symmetric Hessian (quadratic block), \(d(d+1)/2=136\) at \(d=16\), matching the manuscript.

---

## 1. Global linear decoding (confirmatory)

| Manuscript wording | Artifact | Field | Exact value | Paper rounding | Class |
|---|---|---|---|---|---|
| \(\rho_{\mathrm{ctl}}(K_H^{\mathrm{cross}},R_G^2)\approx-0.240\) | `…/submission_validation/metric_associations.csv` row `d=16`, `target_id=mag_r_desi_local_oof_r2`, `slice_mode=full`, `analysis=confirmatory` | `controlled` | −0.2404841119636992 | −0.240 | confirmatory |
| same, QLCA parity recovery | `…/quadratic_label_chart_alignment/parity.json` | `rho_r2_G.controlled` | −0.2404841119636992 | −0.240 | confirmatory (recovered) |
| \(\rho_{\mathrm{ctl}}(K_H^{\mathrm{cross}},\mathrm{MSE}_G)\approx+0.227\) | `…/submission_validation/decision.json` | `error_rhos_d16.mag_r_desi_oof_mse` | 0.2270478922763529 | +0.227 | confirmatory |
| same, QLCA parity | `…/quadratic_label_chart_alignment/parity.json` | `rho_mse_G.controlled` | 0.22704789227635297 | +0.227 | confirmatory (recovered) |
| controlled \(\rho\) with local SST \(\approx-0.025\) | `…/submission_validation/decision.json` | `denom_rhos_d16.mag_r_desi_local_sst` | −0.0245571844475716 | −0.025 | confirmatory |
| \(+0.143\) at \(d=12\) | `…/submission_validation/metric_associations.csv` | `controlled` at `d=12`, same target/slice | 0.14299000211716503 | +0.143 | confirmatory (rank-condition, not the scientific \(d=16\) claim) |
| \(-0.233\) at \(d=20\) | same CSV | `controlled` at `d=20` | −0.23332526879413143 | −0.233 | confirmatory (rank-condition) |
| Fig. 1 \(n=128\) scale markers “not power-matched” | `…/submission_validation/CLAIMS.md` / NeurReps figure provenance | scale subset | \(n=128\) hash-selected | qualitative | confirmatory caption caveat |

---

## 2. Held-out quadratic structure (confirmatory primary)

| Manuscript wording | Artifact | Field | Exact value | Paper rounding | Class |
|---|---|---|---|---|---|
| median \(\Delta_Q\approx+0.021\) | `…/quadratic_label_chart_alignment/primary_inference.json` | `median_delta_Q` | 0.020581617601622228 | +0.021 | confirmatory |
| bootstrap 95% CI \([0.0197,0.0216]\) | same | `delta_Q_ci_lo`, `delta_Q_ci_hi` | 0.019672638372278062, 0.021616092760041075 | three-decimal-friendly 4 d.p. | confirmatory |
| positive at all 512 anchors | same | `frac_positive_delta_Q` | 1.0 | all 512 | confirmatory |
| \(p_{\mathrm{MC}}<1/2001\) for median \(\Delta_Q\) | same | `p_mc_median_delta_Q`, `n_boot` | 0.0004997501249375312 \(=1/(2000+1)\) | \(<1/2001\) | confirmatory |
| \(\rho_{\mathrm{ctl}}(K_H^{\mathrm{cross}},\Delta_Q)\approx+0.111\) | same | `rho_KH_delta_Q` | 0.111248619551161 | +0.111 | confirmatory |
| \(p_{\mathrm{MC}}\approx0.0075\) | same | `rho_KH_delta_Q_p_mc`, `n_perm` | 0.007499250074992501 (\(B=10^4\)) | 0.0075 | confirmatory |
| both primary tests pass Holm | same | `holm_both_pass`; `holm.median_delta_Q`; `holm.rho_KH_delta_Q` | true; 0.0009995002498750624; 0.007499250074992501 | pass | confirmatory |

---

## 3. Geometry-regularized / truncated recovery (secondary + audit)

Aggregation for “fraction of UQ gain”: **median of per-anchor ratios** \(\Delta_{B^S}/\Delta_Q\) (clipped in the original QLCA table), **not** a ratio of medians. Check: median \(\Delta_{BS}/\)median \(\Delta_Q\) \(=0.019606/0.020582\approx0.953\), whereas `frac_UQ_captured_by_BS` \(=0.9376\).

| Manuscript wording | Artifact | Field | Exact value | Paper rounding | Class |
|---|---|---|---|---|---|
| algebraically full rank 136 at every anchor | `…/quadratic_label_chart_alignment_audit/rank_audit_summary.json` | `min_numerical_rank`, `max_numerical_rank`, `frac_algebraic_full_136` | 136, 136, 1.0 | 136 everywhere | post-hoc audit |
| median \(r_{90}=71\) | same | `median_r90` | 71.0 | 71 | post-hoc audit |
| median \(r_{95}=90\) | same | `median_r95` | 90.0 | 90 | post-hoc audit |
| median \(r_{99}=119\) | same | `median_r99` | 119.0 | 119 | post-hoc audit |
| original cap retained 48 modes | same | `median_r_original`, `frac_original_at_cap48`, `constraint_class` | 48.0; 1.0; `implementation_cap_below_energy_rank` | 48, hard cap | post-hoc audit |
| leading 48 modes retain \(\approx94\%\) | `…/quadratic_label_chart_alignment/secondary_inference.json` and audit `truncated_bs_summary.json` `original_rule.median_frac_UQ` | median per-anchor ratio | 0.9376366120634971 | \(\approx94\%\) | secondary / audit |
| 90% energy \(\approx98\%\) | `truncated_bs_summary.json` `e90.median_frac_UQ` | 0.9773080168140899 | \(\approx98\%\) | post-hoc audit |
| 95% energy \(\approx99\%\) | `e95.median_frac_UQ` | 0.9896957045769678 | \(\approx99\%\) | post-hoc audit |
| 99% energy \(\approx100\%\) | `e99.median_frac_UQ` | 1.0008661546926496 | \(\approx100\%\) (slightly above 1) | post-hoc audit |
| median \(\Delta_{BS}\approx0.0196\) (not printed as a headline) | `secondary_inference.json` | `median_delta_BS` | 0.01960613735261897 | used in Fig. 2a boxplot | secondary |
| ridge-on-\(c\) \(\equiv\) UQ with \(\gamma^\top(B^\top B)^+\gamma\) | `regularizer_equivalence.json` | `note`, `full_rank`, real-tensor `rank` | algebraic rank 136; formula match | qualitative | post-hoc audit |

---

## 4. Hessian–curvature alignment (QLCA secondary + audit nulls)

Original QLCA isotropic-style null in `alignment_summary.json` is `A_B_null_median=0.9783494151098924`. The manuscript reports the **estimator-matched audit nulls** (Haar / isotropic / matched-anchor), which sit near 0.99.

| Manuscript wording | Artifact | Field | Exact value | Paper rounding | Class |
|---|---|---|---|---|---|
| \(A_B\approx2.43\) | `…/quadratic_label_chart_alignment/alignment_summary.json` and audit `alignment_tests.json` `haar_all.observed_median` | 2.4271836244410787 | 2.43 | secondary (observed), audit (nulls) |
| Haar null \(\approx0.99\) | `alignment_tests.json` `haar_all.null_median` | 0.9860801603749605 | near 0.99 | post-hoc audit |
| isotropic null \(\approx0.99\) | `isotropic_all.null_median` | 0.9860959584005413 | near 0.99 | post-hoc audit |
| matched-anchor null \(\approx0.99\) | `matched_anchor_spectrum.null_median` | 0.9857431669474271 | near 0.99 | post-hoc audit |
| \(p_{\mathrm{MC}}<1/2001\), \(B=2000\) | `haar_all.p_mc`, `n_null` | 0.0004997501249375312, 2000 | \(<1/2001\) | post-hoc audit |
| foldwise Hessian cosine \(\approx0.92\) | `alignment_summary.json` | `gamma_fold_cosine_median` | 0.9243428097633845 | 0.92 | secondary |
| all 512 Hessians pass stability | `alignment_summary.json` / `alignment_tests.json` | `frac_stable` | 1.0 | all 512 | secondary / audit |
| stability threshold (not printed) | `alignment_tests.json` | `stability_threshold` | 0.5 | — | audit metadata |
| A/B split both exceed Haar | `alignment_tests.json` `split_half.both_exceed_haar` | true; medians 2.04636 and 2.06513 | qualitative | post-hoc audit |
| split-half Spearman \(\approx0.82\) | `split_half.spearman_A_B` | 0.8187268613886314 | 0.82 | post-hoc audit |

---

## 5. Null calibration (audit; do not use original synthetic as counterevidence)

| Manuscript wording | Artifact | Field | Exact value | Paper rounding | Class |
|---|---|---|---|---|---|
| 192 real-design label shuffles | `…/quadratic_label_chart_alignment_audit/shuffle_cause.json` | `real_nested_battery.n` | 192 | 192 | post-hoc audit |
| shuffled \(\Delta_Q\approx-0.0004\) | same | `real_nested_battery.median_delta_Q` | −0.0003797641555049469 | −0.0004 | post-hoc audit |
| no false-positive quadratic gain; well calibrated | same | `shuffle_no_positive_gain`, `shuffle_well_calibrated` | true, true | qualitative | post-hoc audit |
| original fixed-\(\alpha_Q=100\) synthetic (mentioned only as a miscalibrated gate) | same | `original_shuffle_dQ` | −7.561259848029579 | not used as evidence against the result | audit diagnostic |
| UQ grid has no \(\alpha_Q=\infty\) | same | `uq_contains_L`, `quad_grid` | false; `[0.1,…,10000]` | one-sentence caveat | audit diagnostic |

Internal mechanical labels `quadratic_chart_link_unresolved` and `geometry_regularized_quadratic_decoding` are **not** in the manuscript.

---

## 6. Local adaptation is distinct (secondary)

| Manuscript wording | Artifact | Field | Exact value | Paper rounding | Class |
|---|---|---|---|---|---|
| \(\rho_{\mathrm{ctl}}(K_H,\Delta\mathrm{MSE}_{G\to P})\approx+0.153\) | `parity.json` `rho_dMSE_GP.controlled` and `secondary_inference.json` `rho_KH_dMSE_GP` | 0.15334238492921803 | +0.153 | secondary (QLCA recovery of LPA association) |
| conditioning on \(\Delta_Q\) raises it to \(\approx0.205\) | `secondary_inference.json` | `rho_KH_dMSE_GP_adj_deltaQ` | 0.20518047401609046 | 0.205 | secondary |
| patch probes not better on average | `…/local_probe_adaptation_audit/MANUSCRIPT_RECOMMENDATION.md` | recommendation text | negative mean patch advantage | qualitative, secondary | LPA audit |

This is **not** a causal mediation analysis. The partial correlation **rises**, so \(\Delta_Q\) does not explain the adaptation association.

---

## Figures

| Panel | Source | Class |
|---|---|---|
| Fig. 1 | vector copy of NeurReps `figure2_curvature_probe.pdf`, from frozen submission-validation rank curve + \(d=16\) MSE scatter | confirmatory |
| Fig. 2a | `anchor_risks.csv` columns `delta_Q`, `delta_BS` | confirmatory / secondary |
| Fig. 2b | same CSV, rank residuals of `K_H_cross` vs `delta_Q` with controls `log_knn_radius`, `local_label_variance`, `local_evaluation_count` | confirmatory |
| Fig. 2c | `anchor_risks.csv` `A_B`; vertical lines from `alignment_tests.json` | audit |
| Fig. 2d | `truncated_bs_summary.json` `median_r` vs `median_frac_UQ` | audit |
