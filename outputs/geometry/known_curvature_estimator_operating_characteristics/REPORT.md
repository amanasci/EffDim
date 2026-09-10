# Report — operating characteristics of D-full and Q

## Runtime

- actual_s: 386.0059542655945
- wall_s: 2700.0
- new_decoder_fits: 8
- reused_cells: ['F0_S0_N0', 'F0_S3_N4', 'F1_S0_N0', 'F2_S0_N0', 'F2_S3_N4', 'F4_S0_N0', 'F4_S0_N1', 'F4_S0_N2', 'F4_S0_N3', 'F4_S1_N0', 'F4_S2_N0', 'F4_S3_N4']
- new_cells: ['F4_S0_N0_drawA_seed0', 'F4_S0_N0_drawA_seed1', 'F4_S0_N0_drawB_seed0', 'F4_S0_N0_drawB_seed1', 'F4_S3_N4_drawA_seed0', 'F4_S3_N4_drawA_seed1', 'F4_S3_N4_drawB_seed0', 'F4_S3_N4_drawB_seed1']
- stopped_before_cap: True

## Parity

- ok: True
- details: `{'ok': True, 'reproduction': {'d_full_cubic_rho': 0.9392535779861432, 'd_full_cubic_cosine': 0.9935515167974374, 'd_full_ridge_rho': 0.9897837249033488, 'd_full_ridge_cosine': 0.999738235498892, 'cubic_ok': True, 'ridge_ok': True}, 'per_anchor': {'ok': True, 'got': {'d_res_f4_clean_rho': 0.8193223443223443, 'd_res_f4_scal_rho': 0.8617673992673993, 'd_res_f4_combined_stress_rho': 0.7369047619047618, 'q_f4_t2_tensor_cos': 0.6176322563326069, 't2_t3_tensor_cos': 0.7481855145541307, 'd_res_f0_energy_frac': 0.0009638843433637, 'd_full_f4_cosine': 0.9993950499913692}, 'checks': {'d_res_f4_clean_rho': {'got': 0.8193223443223443, 'expect': 0.8193223443223443, 'ok': True, 'tol': 0.02}, 'd_res_f4_scal_rho': {'got': 0.8617673992673993, 'expect': 0.8617673992673993, 'ok': True, 'tol': 0.02}, 'd_res_f4_combined_stress_rho': {'got': 0.7369047619047618, 'expect': 0.7369047619047618, 'ok': True, 'tol': 0.02}, 'q_f4_t2_tensor_cos': {'got': 0.6176322563326069, 'expect': 0.6176322563326069, 'ok': True, 'tol': 0.02}, 't2_t3_tensor_cos': {'got': 0.7481855145541307, 'expect': 0.7481855145541307, 'ok': True, 'tol': 0.02}, 'd_res_f0_energy_frac': {'got': 0.0009638843433637, 'expect': 0.0009638843433637421, 'ok': True, 'tol': 0.0005}, 'parquet_vs_csv_dres_rho': {'got': 0.8193223443223443, 'csv': 0.8193223443223443, 'ok': True}, 'parquet_vs_csv_qt2_cos': {'got': 0.6176322563326069, 'csv': 0.6176322563326069, 'ok': True}, 'parquet_vs_csv_t2t3': {'got': 0.7481855145541307, 'csv': 0.7481855145541307, 'ok': True}}}, 'prior_dual_label': 'neither_estimator_validated'}`

## Estimator labels (this audit only)

- D-full: **d_full_useful_on_non_spherical_clean_geometry**
- Q: **q_moderately_informative_sampling_dependent_statistic**
- D-residual diagnostic: **d_residual_useful_but_stress_sensitive**
- Prior exact-recovery label **neither_estimator_validated** was not overwritten.

Exact-recovery validation asked whether D-full and Q recovered the matched tensor at a frozen gate. Operating characteristics ask whether the same frozen estimators still rank, discriminate, and repeat under sampling and noise, even when those gates fail. The two questions are not substitutes.

## Headline operating facts


- D-full cubic/ridge reuse: ρ(R2)=0.939, ρ(R3)=0.990
- D-full F4 clean vector cosine=0.999; sphere rank degenerate=True
- D-full residualized F4 clean ρ=-0.03427419595949699
- D-residual F4 clean ρ=0.819; stress ρ=0.737; Scal ρ=0.862
- Q T2 scalar K_H ρ=0.5232142857142856; T2 tensor cos=0.618; T2–T3 cos=0.748
- Repeat reliability / ceiling: see repeat_reliability.csv and reliability_ceiling.csv
- Sparsest density quintile: D-full residualized ρ=0.211; D-residual ρ=0.390; Q K_H vs T2 ρ=0.066; D-full residualized ρ=0.431; D-residual ρ=0.582; Q K_H vs T2 ρ=0.220
- Labels: D-full `d_full_useful_on_non_spherical_clean_geometry`; Q `q_moderately_informative_sampling_dependent_statistic`; D-residual `d_residual_useful_but_stress_sensitive`
- Prior exact-recovery label unchanged: `neither_estimator_validated`
- Runtime 386.0059542655945 s; new AEs 8


## Dimensions

- d_full_clean_vector_recovery: **strong**
- d_full_rank_informative: **degenerate_target**
- d_residual_clean_rank_recovery: **strong**
- d_residual_stress_rank_recovery: **moderate**
- q_t2_scalar_rank_recovery: **moderate**
- q_t2_tensor_recovery: **moderate**
- q_t3_geometric_rank_recovery: **moderate**
- q_pointwise_rank_recovery: **moderate**
- q_sampling_measure_dependence: **strong**
- d_sampling_reliability: **strong**
- q_sampling_reliability: **weak**
- d_fraction_of_ceiling: **strong**
- q_fraction_of_ceiling: **weak**
