# SAE Curvature ↔ Physics Probe Error

- n_max=16384, probes=independent (38), k_curv=50, SAE F=2048 k=64

## Aggregated Spearman |ρ| (mean across models)

| Metric | Mean |ρ| | Mean |ρ| (Dense Q1) | Mean AUC |
|---|---:|---:|---:|
| reconstruction_error | 0.175 | 0.077 | 0.611 |
| atom_turnover_rate | 0.119 | 0.019 | 0.570 |
| local_code_rank | 0.043 | 0.024 | 0.520 |
| code_gradient_norm | 0.020 | 0.005 | 0.517 |
| active_set_jaccard_var | 0.018 | 0.009 | 0.507 |

## vit_base (n_test=4916)

| Metric | ρ | ρ (Dense Q1) | p-val | AUC |
|---|---:|---:|---:|---:|
| active_set_jaccard_var | -0.020 | -0.014 | 1.68e-01 | 0.521 |
| code_gradient_norm | +0.009 | +0.006 | 5.13e-01 | 0.520 |
| local_code_rank | +0.040 | +0.017 | 5.26e-03 | 0.520 |
| atom_turnover_rate | +0.130 | +0.026 | 5.11e-20 | 0.583 |
| reconstruction_error | +0.180 | +0.077 | 5.28e-37 | 0.626 |

## dinov3_vitb16 (n_test=4916)

| Metric | ρ | ρ (Dense Q1) | p-val | AUC |
|---|---:|---:|---:|---:|
| active_set_jaccard_var | -0.015 | -0.004 | 2.78e-01 | 0.493 |
| code_gradient_norm | +0.031 | +0.004 | 2.74e-02 | 0.513 |
| local_code_rank | +0.046 | +0.031 | 1.16e-03 | 0.520 |
| atom_turnover_rate | +0.107 | +0.011 | 4.58e-14 | 0.557 |
| reconstruction_error | +0.170 | +0.077 | 3.66e-33 | 0.597 |
