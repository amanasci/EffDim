# Review robustness (concerns 2-5)

All numbers recomputed from the published decoder geometry (sha256-verified). Mismatch = `hess_mismatch_emp`, alignment = `align_cos_tan`, partial Spearman with the published multi-scale controls unless stated.

- clip_base: guard: refit PASS -- reproduced at alpha = 100: 8 split cells and 36 counterfactual values, max |diff| 0 (split), 1.8e-15 (counterfactual); tolerance 0.02
- convnext_base: guard: refit PASS -- reproduced at alpha = 100: 8 split cells and 36 counterfactual values, max |diff| 0 (split), 2.2e-15 (counterfactual); tolerance 0.02
- dinov3_vitb16: guard: refit PASS -- reproduced at alpha = 100: 8 split cells and 36 counterfactual values, max |diff| 0 (split), 4.4e-16 (counterfactual); tolerance 0.02
- vit_base: guard: exact PASS -- reproduced at alpha = 100: 8 split cells and 36 counterfactual values, max |diff| 0 (split), 5.6e-17 (counterfactual); tolerance 1e-06
- vit_large: guard: refit PASS -- reproduced at alpha = 100: 8 split cells and 36 counterfactual values, max |diff| 0 (split), 1.8e-15 (counterfactual); tolerance 0.02

## Concern 2: validation-tuned probe

alpha* is chosen by RidgeCV (leave-one-out) on all rows and sets the global probe w (and so w_N, the geometry columns and the counterfactual); the local R2 outcome uses out-of-fold predictions whose alpha is chosen inside each outer fold (fold alphas column). The two can differ. `grid edge` marks alpha* at the end of the grid (0.001..1e+04): the optimum is then not interior and the tuned w is the least regularised the grid allows. p-values are anchor-level permutation p's; Concern 4 gives dependence-aware intervals, which are the ones to quote.

| encoder | label | alpha* | fold alphas | OOF R2 @100 | OOF R2 @alpha* | mismatch @100 (p) | mismatch @alpha* (p) | align @100 (p) | align @alpha* (p) | help/hurt @100 | help/hurt @alpha* | random-null help @100/@alpha* | median t* @100/@alpha* | thinned sign-test p(help) @100/@alpha* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit_base | mag_r | 0.1 | 0.1 x5 | 0.516 | 0.666 | -0.387 (<0.001) | -0.485 (<0.001) | +0.353 (<0.001) | +0.341 (<0.001) | 0.97/1.00 | 0.96/1.00 | 0.26/0.07 | 2.30/1.12 | <0.001/<0.001 |
| vit_base | photo_z | 0.001 (grid edge) | 0.1 x5 | 0.508 | 0.667 | -0.449 (<0.001) | -0.569 (<0.001) | +0.258 (<0.001) | +0.257 (<0.001) | 0.99/0.99 | 0.99/1.00 | 0.33/0.11 | 3.02/1.36 | <0.001/<0.001 |
| vit_base | smooth_fraction | 0.001 (grid edge) | 0.1/0.1/0.001/0.1/0.1 | 0.534 | 0.675 | -0.086 (0.058) | -0.225 (<0.001) | +0.186 (<0.001) | +0.282 (<0.001) | 0.90/1.00 | 0.98/1.00 | 0.24/0.08 | 1.74/1.14 | <0.001/<0.001 |
| vit_base | stellar_mass | 0.1 | 0.1 x5 | 0.477 | 0.640 | -0.040 (0.376) | -0.132 (0.005) | +0.008 (0.856) | +0.116 (0.012) | 0.96/0.99 | 0.96/1.00 | 0.29/0.09 | 2.62/1.19 | <0.001/<0.001 |
| dinov3_vitb16 | mag_r | 0.0316 | 0.0316 x5 | 0.568 | 0.710 | -0.574 (<0.001) | -0.720 (<0.001) | +0.532 (<0.001) | +0.530 (<0.001) | 0.92/0.99 | 0.96/1.00 | 0.29/0.06 | 2.05/0.97 | <0.001/<0.001 |
| dinov3_vitb16 | photo_z | 0.0316 | 0.0316 x5 | 0.560 | 0.684 | -0.615 (<0.001) | -0.656 (<0.001) | +0.327 (<0.001) | +0.392 (<0.001) | 0.96/0.99 | 0.97/1.00 | 0.29/0.06 | 2.55/1.08 | <0.001/<0.001 |
| dinov3_vitb16 | smooth_fraction | 0.1 | 0.1 x5 | 0.588 | 0.727 | -0.327 (<0.001) | -0.433 (<0.001) | -0.020 (0.671) | +0.225 (<0.001) | 0.93/1.00 | 0.99/1.00 | 0.22/0.07 | 1.52/1.04 | <0.001/<0.001 |
| dinov3_vitb16 | stellar_mass | 0.0316 | 0.0316 x5 | 0.544 | 0.656 | +0.031 (0.490) | -0.104 (0.023) | -0.047 (0.288) | +0.076 (0.078) | 0.95/1.00 | 0.98/1.00 | 0.26/0.06 | 2.07/1.03 | <0.001/<0.001 |
| clip_base | mag_r | 0.01 | 0.01/0.0316/0.0316/0.0316/0.0316 | 0.499 | 0.702 | -0.337 (<0.001) | -0.674 (<0.001) | +0.325 (<0.001) | +0.546 (<0.001) | 0.95/0.99 | 0.96/1.00 | 0.30/0.07 | 2.16/0.96 | 0.004/<0.001 |
| clip_base | photo_z | 0.001 (grid edge) | 0.001 x5 | 0.493 | 0.668 | -0.449 (<0.001) | -0.429 (<0.001) | +0.064 (0.159) | +0.112 (0.012) | 0.99/1.00 | 0.99/1.00 | 0.36/0.11 | 3.44/1.17 | <0.001/<0.001 |
| clip_base | smooth_fraction | 0.001 (grid edge) | 0.001 x5 | 0.489 | 0.647 | -0.250 (<0.001) | -0.414 (<0.001) | +0.235 (<0.001) | +0.272 (<0.001) | 0.92/0.99 | 0.93/1.00 | 0.25/0.07 | 1.73/1.00 | <0.001/<0.001 |
| clip_base | stellar_mass | 0.001 (grid edge) | 0.00316/0.00316/0.00316/0.00316/0.001 | 0.502 | 0.659 | -0.124 (0.007) | -0.154 (0.001) | -0.007 (0.870) | -0.073 (0.083) | 0.97/1.00 | 0.95/1.00 | 0.35/0.11 | 2.85/0.98 | <0.001/<0.001 |
| convnext_base | mag_r | 0.1 | 0.1 x5 | 0.515 | 0.688 | -0.325 (<0.001) | -0.391 (<0.001) | +0.211 (<0.001) | +0.310 (<0.001) | 0.92/1.00 | 0.98/1.00 | 0.26/0.05 | 2.03/1.08 | <0.001/<0.001 |
| convnext_base | photo_z | 0.0316 | 0.0316 x5 | 0.493 | 0.674 | -0.482 (<0.001) | -0.503 (<0.001) | +0.216 (<0.001) | +0.503 (<0.001) | 0.99/1.00 | 0.98/1.00 | 0.28/0.09 | 2.89/1.25 | <0.001/<0.001 |
| convnext_base | smooth_fraction | 0.1 | 0.1 x5 | 0.557 | 0.713 | -0.200 (<0.001) | -0.225 (<0.001) | +0.037 (0.412) | +0.258 (<0.001) | 0.88/0.98 | 0.96/1.00 | 0.23/0.06 | 1.56/1.09 | <0.001/<0.001 |
| convnext_base | stellar_mass | 0.1 | 0.1 x5 | 0.494 | 0.656 | -0.056 (0.204) | -0.222 (<0.001) | +0.067 (0.130) | +0.331 (<0.001) | 0.99/1.00 | 0.99/1.00 | 0.29/0.07 | 3.08/1.27 | <0.001/<0.001 |
| vit_large | mag_r | 0.0316 | 0.1 x5 | 0.518 | 0.701 | -0.439 (<0.001) | -0.666 (<0.001) | +0.130 (0.004) | +0.337 (<0.001) | 0.96/0.99 | 0.97/1.00 | 0.32/0.07 | 2.57/1.07 | <0.001/<0.001 |
| vit_large | photo_z | 0.0316 | 0.0316 x5 | 0.512 | 0.695 | -0.462 (<0.001) | -0.578 (<0.001) | +0.136 (0.001) | +0.403 (<0.001) | 0.97/0.99 | 0.99/1.00 | 0.34/0.11 | 3.20/1.25 | <0.001/<0.001 |
| vit_large | smooth_fraction | 0.0316 | 0.0316 x5 | 0.545 | 0.718 | -0.007 (0.872) | -0.290 (<0.001) | +0.137 (0.005) | +0.382 (<0.001) | 0.79/0.97 | 0.93/1.00 | 0.25/0.06 | 1.62/1.06 | <0.001/<0.001 |
| vit_large | stellar_mass | 0.1 | 0.1 x5 | 0.491 | 0.670 | -0.018 (0.684) | -0.266 (<0.001) | -0.025 (0.578) | +0.154 (<0.001) | 0.98/1.00 | 0.99/1.00 | 0.30/0.09 | 2.94/1.21 | <0.001/<0.001 |

## Concern 3: added value beyond target difficulty (alpha=100)

Extended controls = published controls + label-Hessian norm + local label roughness (1 - the label's linear R2 in the decoder's tangent chart; it shares J and g^-1 with the Hessian estimate, so it over-controls rather than under-controls). With the label-Hessian norm as a control the mismatch partial measures the part of the mismatch the label's own curvature does not explain; a sign change from the published-controls column signals suppression and should be read with care. Held-out: out-of-sample Delta R2 of local R2 from adding mismatch and alignment to the extended controls, fit on half the overlap blocks, scored on the rest (20 splits; ranks).

| encoder | label | mismatch published ctl | mismatch extended ctl (p) | align extended ctl (p) | held-out Delta R2 median [p05, p95] | splits > 0 |
|---|---|---|---|---|---|---|
| vit_base | mag_r | -0.387 | -0.487 (<0.001) | +0.380 (<0.001) | +0.042 [+0.009, +0.095] | 0.95 |
| vit_base | photo_z | -0.449 | -0.460 (<0.001) | +0.186 (<0.001) | +0.093 [+0.017, +0.142] | 1.00 |
| vit_base | smooth_fraction | -0.086 | -0.214 (<0.001) | +0.212 (<0.001) | +0.010 [+0.000, +0.025] | 0.95 |
| vit_base | stellar_mass | -0.040 | -0.156 (0.002) | -0.010 (0.843) | +0.013 [-0.011, +0.024] | 0.70 |
| dinov3_vitb16 | mag_r | -0.574 | -0.442 (<0.001) | +0.334 (<0.001) | +0.036 [+0.017, +0.073] | 0.95 |
| dinov3_vitb16 | photo_z | -0.615 | -0.392 (<0.001) | +0.264 (<0.001) | +0.044 [+0.017, +0.090] | 1.00 |
| dinov3_vitb16 | smooth_fraction | -0.327 | -0.198 (<0.001) | +0.048 (0.295) | +0.011 [-0.007, +0.024] | 0.80 |
| dinov3_vitb16 | stellar_mass | +0.031 | -0.263 (<0.001) | +0.166 (0.002) | +0.008 [-0.033, +0.021] | 0.60 |
| clip_base | mag_r | -0.337 | -0.332 (<0.001) | +0.147 (<0.001) | +0.024 [-0.012, +0.046] | 0.85 |
| clip_base | photo_z | -0.449 | -0.401 (<0.001) | +0.220 (<0.001) | +0.079 [+0.000, +0.128] | 0.95 |
| clip_base | smooth_fraction | -0.250 | -0.250 (<0.001) | +0.193 (<0.001) | +0.030 [+0.007, +0.057] | 1.00 |
| clip_base | stellar_mass | -0.124 | -0.121 (0.007) | +0.133 (0.004) | +0.006 [-0.026, +0.023] | 0.60 |
| convnext_base | mag_r | -0.325 | -0.316 (<0.001) | +0.129 (0.006) | +0.014 [-0.008, +0.033] | 0.80 |
| convnext_base | photo_z | -0.482 | -0.343 (<0.001) | +0.108 (0.015) | +0.051 [-0.003, +0.114] | 0.90 |
| convnext_base | smooth_fraction | -0.200 | -0.147 (0.001) | +0.116 (0.012) | +0.001 [-0.038, +0.006] | 0.65 |
| convnext_base | stellar_mass | -0.056 | -0.320 (<0.001) | +0.172 (0.001) | +0.030 [-0.033, +0.064] | 0.85 |
| vit_large | mag_r | -0.439 | -0.231 (<0.001) | +0.077 (0.107) | +0.014 [+0.001, +0.032] | 0.95 |
| vit_large | photo_z | -0.462 | -0.426 (<0.001) | +0.197 (<0.001) | +0.147 [+0.109, +0.283] | 1.00 |
| vit_large | smooth_fraction | -0.007 | -0.082 (0.061) | +0.053 (0.256) | -0.006 [-0.037, +0.003] | 0.20 |
| vit_large | stellar_mass | -0.018 | -0.096 (0.038) | +0.171 (<0.001) | +0.013 [-0.020, +0.042] | 0.70 |

## Concern 3: added value beyond target difficulty (alpha*)

Extended controls = published controls + label-Hessian norm + local label roughness (1 - the label's linear R2 in the decoder's tangent chart; it shares J and g^-1 with the Hessian estimate, so it over-controls rather than under-controls). With the label-Hessian norm as a control the mismatch partial measures the part of the mismatch the label's own curvature does not explain; a sign change from the published-controls column signals suppression and should be read with care. Held-out: out-of-sample Delta R2 of local R2 from adding mismatch and alignment to the extended controls, fit on half the overlap blocks, scored on the rest (20 splits; ranks).

| encoder | label | mismatch published ctl | mismatch extended ctl (p) | align extended ctl (p) | held-out Delta R2 median [p05, p95] | splits > 0 |
|---|---|---|---|---|---|---|
| vit_base | mag_r | -0.485 | -0.625 (<0.001) | +0.224 (<0.001) | +0.117 [+0.044, +0.202] | 1.00 |
| vit_base | photo_z | -0.569 | -0.367 (<0.001) | +0.093 (0.035) | +0.015 [+0.008, +0.037] | 0.95 |
| vit_base | smooth_fraction | -0.225 | -0.447 (<0.001) | +0.332 (<0.001) | +0.042 [+0.004, +0.075] | 1.00 |
| vit_base | stellar_mass | -0.132 | -0.442 (<0.001) | +0.115 (0.009) | +0.066 [+0.034, +0.096] | 1.00 |
| dinov3_vitb16 | mag_r | -0.720 | -0.715 (<0.001) | +0.286 (<0.001) | +0.099 [+0.064, +0.146] | 1.00 |
| dinov3_vitb16 | photo_z | -0.656 | -0.486 (<0.001) | +0.316 (<0.001) | +0.043 [+0.027, +0.065] | 1.00 |
| dinov3_vitb16 | smooth_fraction | -0.433 | -0.318 (<0.001) | +0.321 (<0.001) | +0.019 [-0.019, +0.060] | 0.80 |
| dinov3_vitb16 | stellar_mass | -0.104 | -0.460 (<0.001) | +0.253 (<0.001) | +0.039 [+0.016, +0.082] | 1.00 |
| clip_base | mag_r | -0.674 | -0.737 (<0.001) | +0.471 (<0.001) | +0.156 [+0.041, +0.273] | 1.00 |
| clip_base | photo_z | -0.429 | -0.379 (<0.001) | +0.057 (0.194) | +0.030 [+0.009, +0.053] | 1.00 |
| clip_base | smooth_fraction | -0.414 | -0.498 (<0.001) | +0.357 (<0.001) | +0.074 [+0.029, +0.116] | 1.00 |
| clip_base | stellar_mass | -0.154 | -0.470 (<0.001) | +0.165 (<0.001) | +0.044 [+0.030, +0.067] | 1.00 |
| convnext_base | mag_r | -0.391 | -0.503 (<0.001) | +0.201 (<0.001) | +0.061 [+0.013, +0.121] | 1.00 |
| convnext_base | photo_z | -0.503 | -0.553 (<0.001) | +0.424 (<0.001) | +0.258 [+0.111, +0.493] | 1.00 |
| convnext_base | smooth_fraction | -0.225 | -0.429 (<0.001) | +0.379 (<0.001) | +0.051 [+0.002, +0.111] | 0.95 |
| convnext_base | stellar_mass | -0.222 | -0.512 (<0.001) | +0.451 (<0.001) | +0.213 [+0.053, +0.349] | 1.00 |
| vit_large | mag_r | -0.666 | -0.653 (<0.001) | +0.218 (<0.001) | +0.129 [+0.029, +0.243] | 1.00 |
| vit_large | photo_z | -0.578 | -0.478 (<0.001) | +0.232 (<0.001) | +0.059 [+0.032, +0.100] | 1.00 |
| vit_large | smooth_fraction | -0.290 | -0.312 (<0.001) | +0.360 (<0.001) | +0.028 [-0.047, +0.121] | 0.75 |
| vit_large | stellar_mass | -0.266 | -0.347 (<0.001) | +0.144 (0.003) | +0.031 [+0.009, +0.053] | 1.00 |

## Concern 4: dependence across anchors

Cluster bootstrap: 32 overlap blocks (average linkage on 1 - neighbourhood overlap), 2000 resamples of whole blocks, 95% percentile interval; excludes-0 at 16 and 64 blocks as sensitivity. Adjacent blocks still share boundary points, so these intervals are more honest than anchor-level permutation, not exact. Thinned: anchors with pairwise overlap <= 0.10 (low power).

| encoder | label | probe | column | partial | 95% CI (32 blocks) | excl. 0 at 16/32/64 | thinned partial (p, n) |
|---|---|---|---|---|---|---|---|
| vit_base | mag_r | alpha=100 | hess_mismatch_emp | -0.387 | [-0.556, -0.164] | y/y/y | -0.558 (0.003, 31) |
| vit_base | mag_r | alpha=100 | align_cos_tan | +0.353 | [+0.179, +0.490] | y/y/y | +0.574 (0.003, 31) |
| vit_base | mag_r | alpha* | hess_mismatch_emp | -0.485 | [-0.669, -0.244] | y/y/y | -0.526 (0.008, 31) |
| vit_base | mag_r | alpha* | align_cos_tan | +0.341 | [+0.149, +0.503] | y/y/y | +0.592 (0.004, 31) |
| vit_base | photo_z | alpha=100 | hess_mismatch_emp | -0.449 | [-0.569, -0.224] | y/y/y | -0.478 (0.015, 31) |
| vit_base | photo_z | alpha=100 | align_cos_tan | +0.258 | [+0.128, +0.336] | y/y/y | +0.583 (0.004, 31) |
| vit_base | photo_z | alpha* | hess_mismatch_emp | -0.569 | [-0.664, -0.375] | y/y/y | -0.608 (0.001, 31) |
| vit_base | photo_z | alpha* | align_cos_tan | +0.257 | [+0.086, +0.365] | y/y/y | +0.694 (<0.001, 31) |
| vit_base | smooth_fraction | alpha=100 | hess_mismatch_emp | -0.086 | [-0.268, +0.161] | n/n/n | -0.344 (0.078, 31) |
| vit_base | smooth_fraction | alpha=100 | align_cos_tan | +0.186 | [+0.006, +0.331] | y/y/y | +0.564 (0.003, 31) |
| vit_base | smooth_fraction | alpha* | hess_mismatch_emp | -0.225 | [-0.382, +0.091] | n/n/n | -0.286 (0.159, 31) |
| vit_base | smooth_fraction | alpha* | align_cos_tan | +0.282 | [-0.057, +0.482] | n/n/n | +0.566 (0.003, 31) |
| vit_base | stellar_mass | alpha=100 | hess_mismatch_emp | -0.040 | [-0.178, +0.148] | n/n/n | -0.103 (0.607, 31) |
| vit_base | stellar_mass | alpha=100 | align_cos_tan | +0.008 | [-0.159, +0.154] | n/n/n | -0.001 (0.998, 31) |
| vit_base | stellar_mass | alpha* | hess_mismatch_emp | -0.132 | [-0.274, +0.077] | n/n/n | -0.240 (0.242, 31) |
| vit_base | stellar_mass | alpha* | align_cos_tan | +0.116 | [-0.095, +0.279] | n/n/n | +0.281 (0.164, 31) |
| dinov3_vitb16 | mag_r | alpha=100 | hess_mismatch_emp | -0.574 | [-0.747, -0.358] | y/y/y | -0.621 (0.001, 32) |
| dinov3_vitb16 | mag_r | alpha=100 | align_cos_tan | +0.532 | [+0.271, +0.675] | y/y/y | +0.516 (0.004, 32) |
| dinov3_vitb16 | mag_r | alpha* | hess_mismatch_emp | -0.720 | [-0.840, -0.529] | y/y/y | -0.721 (<0.001, 32) |
| dinov3_vitb16 | mag_r | alpha* | align_cos_tan | +0.530 | [+0.246, +0.720] | y/y/y | +0.638 (0.001, 32) |
| dinov3_vitb16 | photo_z | alpha=100 | hess_mismatch_emp | -0.615 | [-0.719, -0.442] | y/y/y | -0.278 (0.184, 32) |
| dinov3_vitb16 | photo_z | alpha=100 | align_cos_tan | +0.327 | [+0.098, +0.471] | y/y/y | +0.317 (0.122, 32) |
| dinov3_vitb16 | photo_z | alpha* | hess_mismatch_emp | -0.656 | [-0.750, -0.501] | y/y/y | -0.379 (0.063, 32) |
| dinov3_vitb16 | photo_z | alpha* | align_cos_tan | +0.392 | [+0.119, +0.524] | y/y/y | +0.506 (0.011, 32) |
| dinov3_vitb16 | smooth_fraction | alpha=100 | hess_mismatch_emp | -0.327 | [-0.550, +0.013] | n/n/y | -0.136 (0.506, 32) |
| dinov3_vitb16 | smooth_fraction | alpha=100 | align_cos_tan | -0.020 | [-0.147, +0.121] | n/n/n | -0.076 (0.713, 32) |
| dinov3_vitb16 | smooth_fraction | alpha* | hess_mismatch_emp | -0.433 | [-0.649, -0.160] | y/y/y | -0.513 (0.008, 32) |
| dinov3_vitb16 | smooth_fraction | alpha* | align_cos_tan | +0.225 | [+0.013, +0.340] | n/y/y | +0.050 (0.826, 32) |
| dinov3_vitb16 | stellar_mass | alpha=100 | hess_mismatch_emp | +0.031 | [-0.243, +0.174] | n/n/n | -0.156 (0.445, 32) |
| dinov3_vitb16 | stellar_mass | alpha=100 | align_cos_tan | -0.047 | [-0.212, +0.085] | n/n/n | +0.104 (0.612, 32) |
| dinov3_vitb16 | stellar_mass | alpha* | hess_mismatch_emp | -0.104 | [-0.343, +0.071] | n/n/n | -0.160 (0.434, 32) |
| dinov3_vitb16 | stellar_mass | alpha* | align_cos_tan | +0.076 | [-0.227, +0.237] | n/n/n | +0.335 (0.103, 32) |
| clip_base | mag_r | alpha=100 | hess_mismatch_emp | -0.337 | [-0.615, -0.062] | y/y/y | -0.385 (0.071, 30) |
| clip_base | mag_r | alpha=100 | align_cos_tan | +0.325 | [+0.052, +0.539] | y/y/y | +0.395 (0.045, 30) |
| clip_base | mag_r | alpha* | hess_mismatch_emp | -0.674 | [-0.853, -0.434] | y/y/y | -0.665 (<0.001, 30) |
| clip_base | mag_r | alpha* | align_cos_tan | +0.546 | [+0.318, +0.683] | y/y/y | +0.622 (0.001, 30) |
| clip_base | photo_z | alpha=100 | hess_mismatch_emp | -0.449 | [-0.554, -0.305] | y/y/y | -0.382 (0.074, 30) |
| clip_base | photo_z | alpha=100 | align_cos_tan | +0.064 | [-0.037, +0.262] | n/n/n | +0.387 (0.077, 30) |
| clip_base | photo_z | alpha* | hess_mismatch_emp | -0.429 | [-0.558, -0.240] | y/y/y | -0.222 (0.315, 30) |
| clip_base | photo_z | alpha* | align_cos_tan | +0.112 | [-0.034, +0.296] | n/n/n | +0.311 (0.148, 30) |
| clip_base | smooth_fraction | alpha=100 | hess_mismatch_emp | -0.250 | [-0.504, +0.038] | n/n/n | -0.301 (0.150, 30) |
| clip_base | smooth_fraction | alpha=100 | align_cos_tan | +0.235 | [+0.056, +0.368] | y/y/y | +0.093 (0.660, 30) |
| clip_base | smooth_fraction | alpha* | hess_mismatch_emp | -0.414 | [-0.653, -0.004] | y/y/y | -0.505 (0.007, 30) |
| clip_base | smooth_fraction | alpha* | align_cos_tan | +0.272 | [-0.018, +0.422] | n/n/y | +0.287 (0.155, 30) |
| clip_base | stellar_mass | alpha=100 | hess_mismatch_emp | -0.124 | [-0.295, +0.025] | n/n/n | -0.126 (0.566, 30) |
| clip_base | stellar_mass | alpha=100 | align_cos_tan | -0.007 | [-0.136, +0.103] | n/n/n | -0.459 (0.026, 30) |
| clip_base | stellar_mass | alpha* | hess_mismatch_emp | -0.154 | [-0.362, +0.072] | n/n/n | -0.024 (0.909, 30) |
| clip_base | stellar_mass | alpha* | align_cos_tan | -0.073 | [-0.226, +0.080] | n/n/n | -0.066 (0.765, 30) |
| convnext_base | mag_r | alpha=100 | hess_mismatch_emp | -0.325 | [-0.566, +0.020] | y/n/n | -0.186 (0.367, 34) |
| convnext_base | mag_r | alpha=100 | align_cos_tan | +0.211 | [+0.013, +0.344] | y/y/y | +0.060 (0.767, 34) |
| convnext_base | mag_r | alpha* | hess_mismatch_emp | -0.391 | [-0.640, -0.119] | y/y/y | -0.173 (0.364, 34) |
| convnext_base | mag_r | alpha* | align_cos_tan | +0.310 | [-0.006, +0.551] | n/n/y | -0.148 (0.454, 34) |
| convnext_base | photo_z | alpha=100 | hess_mismatch_emp | -0.482 | [-0.613, -0.205] | y/y/y | -0.397 (0.038, 34) |
| convnext_base | photo_z | alpha=100 | align_cos_tan | +0.216 | [+0.062, +0.315] | y/y/y | +0.075 (0.711, 34) |
| convnext_base | photo_z | alpha* | hess_mismatch_emp | -0.503 | [-0.641, -0.168] | y/y/y | -0.295 (0.137, 34) |
| convnext_base | photo_z | alpha* | align_cos_tan | +0.503 | [+0.298, +0.587] | y/y/y | +0.462 (0.018, 34) |
| convnext_base | smooth_fraction | alpha=100 | hess_mismatch_emp | -0.200 | [-0.433, +0.139] | n/n/n | +0.018 (0.921, 34) |
| convnext_base | smooth_fraction | alpha=100 | align_cos_tan | +0.037 | [-0.146, +0.164] | n/n/n | -0.149 (0.438, 34) |
| convnext_base | smooth_fraction | alpha* | hess_mismatch_emp | -0.225 | [-0.508, +0.239] | n/n/n | -0.094 (0.638, 34) |
| convnext_base | smooth_fraction | alpha* | align_cos_tan | +0.258 | [-0.008, +0.403] | n/n/y | -0.075 (0.707, 34) |
| convnext_base | stellar_mass | alpha=100 | hess_mismatch_emp | -0.056 | [-0.290, +0.142] | n/n/n | -0.133 (0.522, 34) |
| convnext_base | stellar_mass | alpha=100 | align_cos_tan | +0.067 | [-0.076, +0.207] | n/n/n | -0.060 (0.771, 34) |
| convnext_base | stellar_mass | alpha* | hess_mismatch_emp | -0.222 | [-0.407, -0.010] | n/y/y | -0.413 (0.029, 34) |
| convnext_base | stellar_mass | alpha* | align_cos_tan | +0.331 | [+0.123, +0.480] | y/y/y | +0.207 (0.298, 34) |
| vit_large | mag_r | alpha=100 | hess_mismatch_emp | -0.439 | [-0.538, -0.219] | y/y/y | -0.268 (0.220, 30) |
| vit_large | mag_r | alpha=100 | align_cos_tan | +0.130 | [-0.091, +0.336] | n/n/n | +0.302 (0.158, 30) |
| vit_large | mag_r | alpha* | hess_mismatch_emp | -0.666 | [-0.817, -0.447] | y/y/y | -0.714 (<0.001, 30) |
| vit_large | mag_r | alpha* | align_cos_tan | +0.337 | [+0.075, +0.557] | y/y/y | +0.464 (0.022, 30) |
| vit_large | photo_z | alpha=100 | hess_mismatch_emp | -0.462 | [-0.563, -0.294] | y/y/y | -0.423 (0.041, 30) |
| vit_large | photo_z | alpha=100 | align_cos_tan | +0.136 | [-0.037, +0.271] | n/n/n | +0.276 (0.171, 30) |
| vit_large | photo_z | alpha* | hess_mismatch_emp | -0.578 | [-0.675, -0.375] | y/y/y | -0.452 (0.027, 30) |
| vit_large | photo_z | alpha* | align_cos_tan | +0.403 | [+0.214, +0.498] | y/y/y | +0.448 (0.031, 30) |
| vit_large | smooth_fraction | alpha=100 | hess_mismatch_emp | -0.007 | [-0.223, +0.208] | n/n/n | -0.276 (0.211, 30) |
| vit_large | smooth_fraction | alpha=100 | align_cos_tan | +0.137 | [-0.109, +0.338] | n/n/n | +0.018 (0.933, 30) |
| vit_large | smooth_fraction | alpha* | hess_mismatch_emp | -0.290 | [-0.528, +0.006] | n/n/n | -0.295 (0.191, 30) |
| vit_large | smooth_fraction | alpha* | align_cos_tan | +0.382 | [-0.043, +0.608] | n/n/n | +0.269 (0.212, 30) |
| vit_large | stellar_mass | alpha=100 | hess_mismatch_emp | -0.018 | [-0.218, +0.142] | n/n/n | +0.323 (0.145, 30) |
| vit_large | stellar_mass | alpha=100 | align_cos_tan | -0.025 | [-0.151, +0.094] | n/n/n | -0.407 (0.059, 30) |
| vit_large | stellar_mass | alpha* | hess_mismatch_emp | -0.266 | [-0.430, -0.094] | y/y/y | -0.111 (0.611, 30) |
| vit_large | stellar_mass | alpha* | align_cos_tan | +0.154 | [-0.063, +0.300] | n/n/n | -0.115 (0.597, 30) |

## Concern 5: surrogate definition and fidelity

Which baseline each result uses: every counterfactual number in the manuscript -- the Section 4 figure (make_fig_intervention.py) and the appendix counterfactual and sign-test tables (help/hurt, t*, sign tests) -- uses variant `S_model`: the data readout's tangent-plus-radial part w_T.x + w_rad.x is kept exactly on the neighbours (the exact ambient baseline), and only the scaled term is the decoder's second-order in-sphere term (1/2)<w_S, II^S>(u,u). Variant `S` scales the data-side in-sphere readout w_S.x instead, so at t = 1 it is exactly the global probe's own readout (with a local intercept). The median absolute difference below is therefore |local R2(surrogate) - local R2(probe)| at t = 1, per anchor; the probe here is the in-sample global ridge, not the out-of-fold probe whose local R2 is the outcome in Concerns 2-4.

| encoder | label | probe | Spearman(S_model, S) | median abs diff | n |
|---|---|---|---|---|---|
| vit_base | mag_r | alpha=100 | +0.494 | 0.0929 | 512 |
| vit_base | mag_r | alpha* | +0.509 | 0.2276 | 512 |
| vit_base | photo_z | alpha=100 | +0.657 | 0.0775 | 512 |
| vit_base | photo_z | alpha* | +0.461 | 0.1794 | 512 |
| vit_base | smooth_fraction | alpha=100 | +0.714 | 0.0636 | 512 |
| vit_base | smooth_fraction | alpha* | +0.577 | 0.1679 | 512 |
| vit_base | stellar_mass | alpha=100 | +0.608 | 0.0878 | 512 |
| vit_base | stellar_mass | alpha* | +0.299 | 0.1876 | 512 |
| dinov3_vitb16 | mag_r | alpha=100 | +0.628 | 0.0610 | 512 |
| dinov3_vitb16 | mag_r | alpha* | +0.584 | 0.1663 | 512 |
| dinov3_vitb16 | photo_z | alpha=100 | +0.817 | 0.0433 | 512 |
| dinov3_vitb16 | photo_z | alpha* | +0.398 | 0.1233 | 512 |
| dinov3_vitb16 | smooth_fraction | alpha=100 | +0.577 | 0.0889 | 512 |
| dinov3_vitb16 | smooth_fraction | alpha* | +0.663 | 0.1891 | 512 |
| dinov3_vitb16 | stellar_mass | alpha=100 | +0.747 | 0.0444 | 512 |
| dinov3_vitb16 | stellar_mass | alpha* | +0.404 | 0.1119 | 512 |
| clip_base | mag_r | alpha=100 | +0.396 | 0.0780 | 512 |
| clip_base | mag_r | alpha* | +0.659 | 0.2619 | 512 |
| clip_base | photo_z | alpha=100 | +0.754 | 0.0436 | 512 |
| clip_base | photo_z | alpha* | +0.545 | 0.1272 | 512 |
| clip_base | smooth_fraction | alpha=100 | +0.644 | 0.0353 | 512 |
| clip_base | smooth_fraction | alpha* | +0.638 | 0.1356 | 512 |
| clip_base | stellar_mass | alpha=100 | +0.747 | 0.0496 | 512 |
| clip_base | stellar_mass | alpha* | +0.523 | 0.0989 | 512 |
| convnext_base | mag_r | alpha=100 | +0.475 | 0.0644 | 512 |
| convnext_base | mag_r | alpha* | +0.559 | 0.2786 | 512 |
| convnext_base | photo_z | alpha=100 | +0.794 | 0.0587 | 512 |
| convnext_base | photo_z | alpha* | +0.432 | 0.2226 | 512 |
| convnext_base | smooth_fraction | alpha=100 | +0.722 | 0.0517 | 512 |
| convnext_base | smooth_fraction | alpha* | +0.637 | 0.2032 | 512 |
| convnext_base | stellar_mass | alpha=100 | +0.804 | 0.0502 | 512 |
| convnext_base | stellar_mass | alpha* | +0.618 | 0.1608 | 512 |
| vit_large | mag_r | alpha=100 | +0.354 | 0.0816 | 512 |
| vit_large | mag_r | alpha* | +0.517 | 0.2332 | 512 |
| vit_large | photo_z | alpha=100 | +0.679 | 0.0588 | 512 |
| vit_large | photo_z | alpha* | +0.162 | 0.1786 | 512 |
| vit_large | smooth_fraction | alpha=100 | +0.763 | 0.0479 | 512 |
| vit_large | smooth_fraction | alpha* | +0.563 | 0.1686 | 512 |
| vit_large | stellar_mass | alpha=100 | +0.785 | 0.0529 | 512 |
| vit_large | stellar_mass | alpha* | +0.433 | 0.1558 | 512 |

## Limitation

On a known manifold (results/tensor-fidelity) the label-Hessian estimate has the right direction but its magnitude is about 2x too large at paper scale (d=16, n=86,471). The partials above are rank-based and unaffected; any claim about mismatch magnitudes would be.
