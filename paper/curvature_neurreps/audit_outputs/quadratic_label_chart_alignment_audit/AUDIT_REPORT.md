# AUDIT REPORT — quadratic label chart alignment

Original decision label (unchanged): `quadratic_chart_link_unresolved`
Audit interpretation: `geometry_regularized_quadratic_decoding`
Runtime: 8273.2s  smoke=False
Outputs: `/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment_audit`

## Phase 0 parity (reproduced from frozen tables)
- median Δ_Q = 0.020582  (expected ≈ +0.021)
- ρ_ctl(K_H, Δ_Q) = 0.111249  (expected ≈ +0.111)
- Δ_BS = 0.019606;  BS capture = 0.9376
- Δ_FQ = 0.019695
- A_B = 2.4272; isotropic null = 0.9783; γ cosine = 0.9243
- previous ρ: -0.240, 0.227, 0.153
- partial ρ after Δ_Q: 0.205  (rose; not mediation)
- synthetic shuffle Δ_Q = -7.561

## Phase 1 rank
- ambient dim = 768; normal dim = 751; q = 136
- median numerical rank = 136.0
- median r_95 = 90.0; r_99 = 119.0
- median rank actually used (frozen rule) = 48.0  (fraction 0.35294117647058826)
- constraint class: `implementation_cap_below_energy_rank`
- BS is genuinely constrained: False

Do not read the original 94% UQ-capture figure as a low-dimensional constraint unless retained rank is materially below 136.

## Phase 2 geometry-only truncated BS
{
  "e90": {
    "median_r": 71.0,
    "median_delta": 0.020408841711712783,
    "median_frac_UQ": 0.9773080168140899,
    "median_edf": 44.39622206343948,
    "median_f_reachable": 0.7915033739035195,
    "rho_KH": 0.16469756011032147
  },
  "e95": {
    "median_r": 90.0,
    "median_delta": 0.02070250525617774,
    "median_frac_UQ": 0.9896957045769678,
    "median_edf": 46.6935130833772,
    "median_f_reachable": 0.8639254567038048,
    "rho_KH": 0.16536078352273378
  },
  "e99": {
    "median_r": 119.0,
    "median_delta": 0.020836647398419963,
    "median_frac_UQ": 1.0008661546926496,
    "median_edf": 48.20551351935775,
    "median_f_reachable": 0.9561691129180693,
    "rho_KH": 0.16418919015575467
  },
  "original_rule": {
    "median_r": 48.0,
    "median_delta": 0.019606137352618916,
    "median_frac_UQ": 0.9376366120634965,
    "median_edf": 39.93230349743439,
    "median_f_reachable": 0.6781184870035892,
    "rho_KH": 0.1566640452443895
  }
}

## Phase 3 alignment nulls
- Haar p_MC = 0.0004997501249375312; observed median = 2.4271836244410787
- isotropic p_MC = 0.0004997501249375312
- matched-anchor (secondary) p_MC = 0.0004997501249375312
- split-half Spearman(A,B) = 0.8187268613886314
- both A and B exceed Haar: True
- stable subset p_MC = 0.0004997501249375312

## Phase 4 shuffle
- cause: The original synthetic UQ path uses a finite quadratic penalty (α_Q=100) and cannot omit the quadratic block. Shuffled labels therefore let extra quadratic capacity overfit noise (train MSE drops, held-out MSE rises), producing large negative Δ_Q. This is null miscalibration, not a false-positive quadratic recovery. False-positive safety uses the one-sided test Δ_Q>0, not |Δ_Q|.
- synthetic false-positive safety: True
- synthetic null calibration: False
- real nested-CV false-positive safety: True
- UQ contains L: False

## Phase 5 v2
v2 rerun: True
reasons: ['nested_cv_selects_max_quadratic_penalty_on_null_or_real_shuffle']
A v2 rerun is required only if fitted predictions change. Absence of α_Q=∞ is a real hyperparameter-family defect for null calibration, but original ViT-B Δ_Q is positive at every anchor, so adding the nested-null candidate would not replace UQ with L on the scientific data. The original |Δ_Q| synthetic gate is a semantic error, not an estimator error.

## Figures
- `/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment_audit/figures/fig1_paired_gains.pdf`
- `/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment_audit/figures/fig2_KH_vs_deltaQ.pdf`
- `/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment_audit/figures/fig3_BS_vs_UQ_alignment.pdf`
- `/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment_audit/figures/figS_singular_spectrum.pdf`
