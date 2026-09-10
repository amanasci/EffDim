# Report — curvature-component predictive decomposition

Exploratory (not prospectively preregistered). Decision **`distinct_mean_and_traceless_predictive_roles`**
(mean_predicts_error_or_gain_and_tf_accounts_for_hessian_or_tq). Runtime 61.2 min. Tests 11/11
parity=True.

## Which component carries decodability?

$B^S$ is the sphere-normal curvature tensor. $K_H$ and $K_{\mathrm{TF}}$ quantify
distinct organizations of that tensor. $K_H$ is not the complete curvature;
$K_{\mathrm{dir}}$ is not the unique mathematically correct scalar.

### Unique associations with global OOF error (primary)

Equal-weight mean unique $\rho_{\mathrm{ctl}}(K_H,\mathrm{MSE}_G\mid K_{\mathrm{TF}})$
= -0.031 CI [-0.07301826188195358, 0.012382396616804977] $p_{\mathrm{Holm}}$=0.1185.

Equal-weight mean unique $\rho_{\mathrm{ctl}}(K_{\mathrm{TF}},\mathrm{MSE}_G\mid K_H)$
= 0.077 CI [0.017068504363465454, 0.13243907705817906] $p_{\mathrm{Holm}}$=0.0004.

Sign counts: $K_H$ positive in 2 / 5 encoders;
$K_{\mathrm{TF}}$ positive in 3 / 5.

These are **absolute probe-error** associations, not local-adaptation gains
and not quadratic label gains.

### Organization

High total curvature is typically greater traceless bending: see
`organization_table.csv`. Cross-estimated $K_{\mathrm{TF}}$ share of $K_{\mathrm{dir}}$
remains ~90%+, consistent with the completed reconciliation.

### ViT-B quadratic gain and Hessian

Median $\|\Gamma_H\|_F^2/\|\Gamma\|_F^2$=0.013; traceless=0.987. Median $\Delta_{IQ}$=-0.0000, $\Delta_{TQ}$=0.0201, $\Delta_{UQ2}$=0.0206. Alignment driver: traceless. BSTF explains BS: True.

Do not infer causality or mediation.

## Decision flags

```
{'unique_KH_mseg': False, 'unique_KTF_mseg': True, 'unique_KH_vitb': True, 'unique_KTF_vitb': True, 'unique_KH_delta_Q': False, 'unique_KTF_delta_Q': True, 'loo_flip_KH': np.True_, 'loo_flip_KTF': np.False_, 'n_pos_KH': 2, 'n_pos_KTF': 3, 'hessian_isotropic': False, 'hessian_traceless': True, 'align_mean': False, 'align_tf': True, 'align_interaction': False, 'iq_explains_uq2': False, 'tq_explains_uq2': True, 'bstf_explains_bs': True}
```
