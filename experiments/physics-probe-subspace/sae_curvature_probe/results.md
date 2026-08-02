# SAE Curvature ↔ Physics Probe Error

- n_max=16384, probes=independent (38), k_curv=50, SAE F=2048 k=64

## Aggregated Spearman |ρ| (mean across models)

| Metric | Mean |ρ| | Mean AUC |
|---|---:|---:|
| reconstruction_error | 0.175 | 0.611 |
| atom_turnover_rate | 0.119 | 0.570 |
| local_code_rank | 0.043 | 0.520 |
| code_gradient_norm | 0.020 | 0.517 |
| active_set_jaccard_var | 0.018 | 0.507 |

## vit_base (n_test=4916)

| Metric | ρ | p-val | AUC |
|---|---:|---:|---:|
| active_set_jaccard_var | -0.020 | 1.69e-01 | 0.521 |
| code_gradient_norm | +0.009 | 5.12e-01 | 0.520 |
| local_code_rank | +0.040 | 5.27e-03 | 0.520 |
| atom_turnover_rate | +0.130 | 5.08e-20 | 0.583 |
| reconstruction_error | +0.180 | 5.28e-37 | 0.626 |

## dinov3_vitb16 (n_test=4916)

| Metric | ρ | p-val | AUC |
|---|---:|---:|---:|
| active_set_jaccard_var | -0.016 | 2.75e-01 | 0.492 |
| code_gradient_norm | +0.031 | 2.75e-02 | 0.513 |
| local_code_rank | +0.046 | 1.17e-03 | 0.520 |
| atom_turnover_rate | +0.107 | 4.59e-14 | 0.557 |
| reconstruction_error | +0.170 | 3.66e-33 | 0.597 |
