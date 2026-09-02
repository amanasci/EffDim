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
- **photometric redshift** (`photo_z`, photometric_redshift, n=512): ρ_ctl(K_H, R²_G)=-0.047, ρ_ctl(K_H, MSE_G)=0.032
- **smooth fraction** (`smooth_fraction`, morphology, n=512): ρ_ctl(K_H, R²_G)=-0.007, ρ_ctl(K_H, MSE_G)=0.025
- **catalog stellar-mass proxy** (`stellar_mass`, stellar_population_proxy, n=512): ρ_ctl(K_H, R²_G)=-0.231, ρ_ctl(K_H, MSE_G)=0.124

mag_r_desi parity ok=True (expected ρ_R²=-0.24, ρ_MSE=0.227).
Runtime 146.9s smoke=False.

These secondary labels are a screen, not a replacement for the frozen r-band confirmatory analysis.
Catalog vectors are never used as the global-decoding outcome; only local OOF probe metrics are.
