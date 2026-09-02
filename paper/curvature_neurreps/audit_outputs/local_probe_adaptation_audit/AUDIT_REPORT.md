# AUDIT REPORT

## Label

`curvature_predicts_relative_local_adaptation`

## Parity

ok=True; mean ΔMSE_G→P=-0.1012

## Primary ρ(K_H, ΔMSE_G→P)

+0.1533 CI [0.07462574054758203, 0.2405865572160708] p_MC=0.0005999

## Paired Δρ(MSE_G - MSE_P)

{'name': 'delta_rho_MSE_GP', 'y_a': 'mse_G', 'y_b': 'mse_P', 'rho_a': 0.22704789227635297, 'rho_b': 0.17477573070804867, 'delta_rho': 0.052272161568304304, 'ci95_lo': -0.00018859676990487097, 'ci95_hi': 0.1123187176190151, 'p_boot_positive': 0.9746, 'p_mc_two_sided': 0.5523447655234477}

## Alignment controls

       model      raw  controlled   n    p_raw    p_ctl
  A_baseline 0.098699    0.152437 384 0.053297 0.002745
   B_plus_AH 0.098699    0.152437 384 0.053297 0.002745
   C_plus_AB 0.098699    0.152437 384 0.053297 0.002745
D_plus_AH_AB 0.098699    0.152437 384 0.053297 0.002745

## Shuffle

{'obs_rho_ctl': 0.20586316608679728, 'n_audit': 128, 'n_perm': 200, 'p_mc': 0.11442786069651742, 'null_mean': -0.01692386925471525, 'pass': False, 'inconclusive': False}

## Manuscript

include_as_exploratory_appendix

Runtime 22974.7s smoke=False
