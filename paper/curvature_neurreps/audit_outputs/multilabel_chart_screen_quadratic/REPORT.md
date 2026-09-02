# Multi-label frozen-chart screen

Charts, neighbourhoods, and global OOF probes are reused. Geometry is not refit.
Targets are physics-table labels with a proven `sample_id` join.
DESI spectroscopic / DESI imaging labels are excluded.

## Excluded

- `sfr`: underpowered (≈45 labelled anchors; n<64 rule)
- `desi_spec_z`: object-level identity join unproven
- `desi_mag_r`: object-level identity join unproven

## Results

- **apparent r-band magnitude** (`mag_r_desi`, photometric_magnitude, n=512): ρ_ctl(K_H, R²_G)=-0.240, ρ_ctl(K_H, MSE_G)=0.227
  Δ_Q median=0.0206, ρ_ctl(K_H, Δ_Q)=0.111, A_B=2.427
- **photometric redshift** (`photo_z`, photometric_redshift, n=512): ρ_ctl(K_H, R²_G)=-0.047, ρ_ctl(K_H, MSE_G)=0.032
  Δ_Q median=0.0003, ρ_ctl(K_H, Δ_Q)=-0.111, A_B=2.030
- **smooth fraction** (`smooth_fraction`, morphology, n=512): ρ_ctl(K_H, R²_G)=-0.007, ρ_ctl(K_H, MSE_G)=0.025
  Δ_Q median=0.0018, ρ_ctl(K_H, Δ_Q)=0.081, A_B=1.960
- **catalog stellar-mass proxy** (`stellar_mass`, stellar_population_proxy, n=512): ρ_ctl(K_H, R²_G)=-0.231, ρ_ctl(K_H, MSE_G)=0.124
  Δ_Q median=-0.1148, ρ_ctl(K_H, Δ_Q)=0.127, A_B=1.127

mag_r_desi parity ok=True (expected ρ_R²=-0.24, ρ_MSE=0.227).
Runtime 53770.9s smoke=False.

These secondary labels are a screen, not a replacement for the frozen r-band confirmatory analysis.
Catalog vectors are never used as the global-decoding outcome; only local OOF probe metrics are.
