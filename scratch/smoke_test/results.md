# SAE Curvature ↔ Physics Probe Error

- n_max=100, probes=independent (38), k_curv=5, SAE F=64 k=4

## Aggregated Spearman |ρ| (mean across models)

| Metric | Mean |ρ| | Mean |ρ| (Dense Q1) | Mean AUC |
|---|---:|---:|---:|
| reconstruction_error | 0.230 | N/A | 0.598 |
| active_set_jaccard_var | 0.122 | N/A | 0.550 |
| code_gradient_norm | 0.079 | N/A | 0.536 |
| local_code_rank | 0.079 | N/A | 0.564 |
| atom_turnover_rate | 0.071 | N/A | 0.561 |

## vit_base (n_test=30)

| Metric | ρ | ρ (Dense Q1) | p-val | AUC |
|---|---:|---:|---:|---:|
| active_set_jaccard_var | +0.202 | N/A | 2.84e-01 | 0.598 |
| code_gradient_norm | +0.078 | N/A | 6.83e-01 | 0.529 |
| local_code_rank | +0.112 | N/A | 5.54e-01 | 0.547 |
| atom_turnover_rate | +0.104 | N/A | 5.85e-01 | 0.536 |
| reconstruction_error | +0.453 | N/A | 1.19e-02 | 0.698 |

## dinov3_vitb16 (n_test=30)

| Metric | ρ | ρ (Dense Q1) | p-val | AUC |
|---|---:|---:|---:|---:|
| active_set_jaccard_var | -0.041 | N/A | 8.30e-01 | 0.502 |
| code_gradient_norm | +0.079 | N/A | 6.77e-01 | 0.542 |
| local_code_rank | -0.045 | N/A | 8.14e-01 | 0.582 |
| atom_turnover_rate | +0.038 | N/A | 8.41e-01 | 0.587 |
| reconstruction_error | +0.006 | N/A | 9.75e-01 | 0.498 |
