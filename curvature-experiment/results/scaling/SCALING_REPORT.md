10 of 10 encoders have records
10 of 10 encoders complete all 6 jobs

## (a) Mismatch partial negative and significant (main_xfit record)

- mag_r: 10 of 10
- photo_z: 10 of 10
- smooth_fraction: 8 of 10
- stellar_mass: 6 of 10

## (b) Alignment partial sign and significance (main_xfit record)

Paper's claim per label (main.tex): mag_r and photo_z positive and significant; stellar_mass non-significant; smooth_fraction no stated claim ("less consistent"). Exceptions are encoders that do not match their label's claim.

- mag_r: 0 negative-significant / 8 positive-significant / 2 non-significant (of 10); paper claim: positive-significant
- photo_z: 0 negative-significant / 7 positive-significant / 3 non-significant (of 10); paper claim: positive-significant
- smooth_fraction: 0 negative-significant / 7 positive-significant / 3 non-significant (of 10); no paper claim
- stellar_mass: 1 negative-significant / 1 positive-significant / 8 non-significant (of 10); paper claim: non-significant

## (c) Counterfactual: model-normal help exceeds random help, and help > 0.5

- mag_r: help > random help in 10 of 10; help > 0.5 in 10 of 10
- photo_z: help > random help in 10 of 10; help > 0.5 in 10 of 10
- smooth_fraction: help > random help in 10 of 10; help > 0.5 in 10 of 10
- stellar_mass: help > random help in 10 of 10; help > 0.5 in 10 of 10

## (c') Counterfactual: sign reversal hurts (model-normal hurt > 0.5 and > random hurt)

- mag_r: hurt > 0.5 and hurt > random hurt in 10 of 10
- photo_z: hurt > 0.5 and hurt > random hurt in 10 of 10
- smooth_fraction: hurt > 0.5 and hurt > random hurt in 10 of 10
- stellar_mass: hurt > 0.5 and hurt > random hurt in 10 of 10

## (d) Thinned-anchor sign test p_help < 0.05

- mag_r: 10 of 10
- photo_z: 10 of 10
- smooth_fraction: 10 of 10
- stellar_mass: 10 of 10

## Encoders that break a claim

- (a) mismatch negative and significant, mag_r: none
- (a) mismatch negative and significant, photo_z: none
- (a) mismatch negative and significant, smooth_fraction: vit_base, vit_large
- (a) mismatch negative and significant, stellar_mass: convnext_base, dinov3_vitb16, vit_base, vit_large
- (b) alignment positive and significant, mag_r: dinov3_vitl16, dinov3_vith16plus
- (b) alignment positive and significant, photo_z: clip_base, dinov3_vits16, dinov3_vith16plus
- (b) alignment non-significant, stellar_mass: dinov3_vits16plus, dinov3_vith16plus
- (c) help > random help, mag_r: none
- (c) help > 0.5, mag_r: none
- (c) help > random help, photo_z: none
- (c) help > 0.5, photo_z: none
- (c) help > random help, smooth_fraction: none
- (c) help > 0.5, smooth_fraction: none
- (c) help > random help, stellar_mass: none
- (c) help > 0.5, stellar_mass: none
- (c') hurt > 0.5 and hurt > random hurt, mag_r: none
- (c') hurt > 0.5 and hurt > random hurt, photo_z: none
- (c') hurt > 0.5 and hurt > random hurt, smooth_fraction: none
- (c') hurt > 0.5 and hurt > random hurt, stellar_mass: none
- (d) sign test p_help < 0.05, mag_r: none
- (d) sign test p_help < 0.05, photo_z: none
- (d) sign test p_help < 0.05, smooth_fraction: none
- (d) sign test p_help < 0.05, stellar_mass: none

## Reproduction guard (robust job)

- clip_base: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- convnext_base: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- dinov3_vits16: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- dinov3_vits16plus: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- dinov3_vitb16: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- dinov3_vitl16: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- dinov3_vith16plus: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- dinov3_vit7b16: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- vit_base: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- vit_large: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- not run: none

## Stale encoders

- none

## Published five: sweep GPU versus published CPU

Per label and column, the sweep's main_xfit multiscale partial against the published record. A cell agrees when the sign and significance at 0.05 match and |GPU - CPU| is at most the tolerance; tolerance = ViT-B's published seed spread (seeds 0, 1, 2; the only CPU seed spread that exists), used for all five encoders. Borderline: p in (0.01, 0.1) on either side, counted separately. -- = no published value.

| encoder | label | column | published | sweep | result |
|---|---|---|---|---|---|
| clip_base | mag_r | hess_mismatch_emp | -0.34 | -0.34 | agree |
| clip_base | mag_r | align_cos_tan | +0.32 | +0.33 | agree |
| clip_base | photo_z | hess_mismatch_emp | -0.45 | -0.45 | agree |
| clip_base | photo_z | align_cos_tan | +0.06 | +0.06 | agree |
| clip_base | smooth_fraction | hess_mismatch_emp | -0.25 | -0.25 | agree |
| clip_base | smooth_fraction | align_cos_tan | +0.24 | +0.24 | agree |
| clip_base | stellar_mass | hess_mismatch_emp | -0.12 | -0.13 | agree |
| clip_base | stellar_mass | align_cos_tan | -0.01 | -0.01 | agree |
| convnext_base | mag_r | hess_mismatch_emp | -0.32 | -0.32 | agree |
| convnext_base | mag_r | align_cos_tan | +0.21 | +0.21 | agree |
| convnext_base | photo_z | hess_mismatch_emp | -0.48 | -0.48 | agree |
| convnext_base | photo_z | align_cos_tan | +0.22 | +0.22 | agree |
| convnext_base | smooth_fraction | hess_mismatch_emp | -0.20 | -0.20 | agree |
| convnext_base | smooth_fraction | align_cos_tan | +0.04 | +0.04 | agree |
| convnext_base | stellar_mass | hess_mismatch_emp | -0.06 | -0.06 | agree |
| convnext_base | stellar_mass | align_cos_tan | +0.07 | +0.07 | agree |
| dinov3_vitb16 | mag_r | hess_mismatch_emp | -0.57 | -0.57 | agree |
| dinov3_vitb16 | mag_r | align_cos_tan | +0.53 | +0.53 | agree |
| dinov3_vitb16 | photo_z | hess_mismatch_emp | -0.61 | -0.61 | agree |
| dinov3_vitb16 | photo_z | align_cos_tan | +0.33 | +0.33 | agree |
| dinov3_vitb16 | smooth_fraction | hess_mismatch_emp | -0.33 | -0.33 | agree |
| dinov3_vitb16 | smooth_fraction | align_cos_tan | -0.02 | -0.02 | agree |
| dinov3_vitb16 | stellar_mass | hess_mismatch_emp | +0.03 | +0.03 | agree |
| dinov3_vitb16 | stellar_mass | align_cos_tan | -0.05 | -0.05 | agree |
| vit_base | mag_r | hess_mismatch_emp | -0.39 | -0.39 | agree |
| vit_base | mag_r | align_cos_tan | +0.35 | +0.35 | agree |
| vit_base | photo_z | hess_mismatch_emp | -0.45 | -0.45 | agree |
| vit_base | photo_z | align_cos_tan | +0.26 | +0.26 | agree |
| vit_base | smooth_fraction | hess_mismatch_emp | -0.09 | -0.09 | borderline |
| vit_base | smooth_fraction | align_cos_tan | +0.19 | +0.19 | agree |
| vit_base | stellar_mass | hess_mismatch_emp | -0.04 | -0.04 | agree |
| vit_base | stellar_mass | align_cos_tan | +0.01 | +0.01 | agree |
| vit_large | mag_r | hess_mismatch_emp | -0.44 | -0.44 | agree |
| vit_large | mag_r | align_cos_tan | +0.13 | +0.13 | agree |
| vit_large | photo_z | hess_mismatch_emp | -0.46 | -0.46 | agree |
| vit_large | photo_z | align_cos_tan | +0.14 | +0.14 | agree |
| vit_large | smooth_fraction | hess_mismatch_emp | -0.01 | -0.01 | agree |
| vit_large | smooth_fraction | align_cos_tan | +0.14 | +0.14 | agree |
| vit_large | stellar_mass | hess_mismatch_emp | -0.02 | -0.02 | agree |
| vit_large | stellar_mass | align_cos_tan | -0.03 | -0.02 | agree |

Counts: agree 39, borderline 1

OOF R2 identity: clip_base max |diff| 1.1e-16
OOF R2 identity: convnext_base max |diff| 1.1e-16
OOF R2 identity: dinov3_vitb16 max |diff| 1.1e-16
OOF R2 identity: vit_base max |diff| 1.1e-16
OOF R2 identity: vit_large max |diff| 1.1e-16

Counterfactual (yes/no agreement; there is no CPU seed spread for it):

| encoder | label | help>0.5 pub/sweep | hurt>0.5 pub/sweep | sign test p<0.05 pub/sweep | agree |
|---|---|---|---|---|---|
| clip_base | mag_r | y/y | y/y | y/y | agree |
| clip_base | photo_z | y/y | y/y | y/y | agree |
| clip_base | smooth_fraction | y/y | y/y | y/y | agree |
| clip_base | stellar_mass | y/y | y/y | y/y | agree |
| convnext_base | mag_r | y/y | y/y | y/y | agree |
| convnext_base | photo_z | y/y | y/y | y/y | agree |
| convnext_base | smooth_fraction | y/y | y/y | y/y | agree |
| convnext_base | stellar_mass | y/y | y/y | y/y | agree |
| dinov3_vitb16 | mag_r | y/y | y/y | y/y | agree |
| dinov3_vitb16 | photo_z | y/y | y/y | y/y | agree |
| dinov3_vitb16 | smooth_fraction | y/y | y/y | y/y | agree |
| dinov3_vitb16 | stellar_mass | y/y | y/y | y/y | agree |
| vit_base | mag_r | y/y | y/y | y/y | agree |
| vit_base | photo_z | y/y | y/y | y/y | agree |
| vit_base | smooth_fraction | y/y | y/y | y/y | agree |
| vit_base | stellar_mass | y/y | y/y | y/y | agree |
| vit_large | mag_r | y/y | y/y | y/y | agree |
| vit_large | photo_z | y/y | y/y | y/y | agree |
| vit_large | smooth_fraction | y/y | y/y | y/y | agree |
| vit_large | stellar_mass | y/y | y/y | y/y | agree |

## DINOv3 size ladder

One family, one training recipe. Supports: sign and significance of the partials and the counterfactual pattern across sizes. Does not support 'the effect scales with size' (D changes with size at fixed d = 16; n = 6). Spearman with log params is descriptive.

| encoder | params | label | mismatch (seeds min..max) | alignment (seeds min..max) | help/hurt | mismatch @alpha* |
|---|---|---|---|---|---|---|
| dinov3_vits16 | 21,596,544 | mag_r | -0.62 (-0.62..-0.62) | +0.21 (+0.19..+0.21) | 0.92/0.99 | -0.57 |
| dinov3_vits16 | 21,596,544 | photo_z | -0.63 (-0.64..-0.62) | -0.01 (-0.01..+0.06) | 0.93/1.00 | -0.69 |
| dinov3_vits16 | 21,596,544 | smooth_fraction | -0.13 (-0.13..-0.12) | +0.31 (+0.29..+0.32) | 0.85/0.99 | -0.41 |
| dinov3_vits16 | 21,596,544 | stellar_mass | -0.23 (-0.24..-0.23) | +0.01 (+0.01..+0.06) | 0.83/0.96 | -0.33 |
| dinov3_vits16plus | 28,692,864 | mag_r | -0.59 (-0.60..-0.58) | +0.18 (+0.18..+0.26) | 0.89/0.99 | -0.58 |
| dinov3_vits16plus | 28,692,864 | photo_z | -0.54 (-0.54..-0.53) | +0.12 (+0.12..+0.18) | 0.94/0.99 | -0.60 |
| dinov3_vits16plus | 28,692,864 | smooth_fraction | -0.16 (-0.16..-0.12) | +0.25 (+0.18..+0.29) | 0.84/0.99 | -0.35 |
| dinov3_vits16plus | 28,692,864 | stellar_mass | -0.13 (-0.15..-0.13) | +0.12 (+0.08..+0.14) | 0.89/0.99 | -0.27 |
| dinov3_vitb16 | 85,660,416 | mag_r | -0.57 (-0.57..-0.56) | +0.53 (+0.53..+0.56) | 0.92/0.99 | -0.72 |
| dinov3_vitb16 | 85,660,416 | photo_z | -0.61 (-0.62..-0.61) | +0.33 (+0.27..+0.33) | 0.96/0.99 | -0.66 |
| dinov3_vitb16 | 85,660,416 | smooth_fraction | -0.33 (-0.34..-0.33) | -0.02 (-0.07..+0.05) | 0.93/1.00 | -0.43 |
| dinov3_vitb16 | 85,660,416 | stellar_mass | +0.03 (+0.02..+0.04) | -0.05 (-0.14..-0.05) | 0.95/1.00 | -0.10 |
| dinov3_vitl16 | 303,129,600 | mag_r | -0.55 (-0.56..-0.52) | +0.01 (+0.01..+0.13) | 0.97/1.00 | -0.66 |
| dinov3_vitl16 | 303,129,600 | photo_z | -0.49 (-0.50..-0.48) | +0.18 (+0.16..+0.22) | 0.94/0.98 | -0.51 |
| dinov3_vitl16 | 303,129,600 | smooth_fraction | -0.41 (-0.44..-0.41) | +0.10 (+0.01..+0.10) | 0.91/1.00 | -0.38 |
| dinov3_vitl16 | 303,129,600 | stellar_mass | -0.27 (-0.27..-0.24) | -0.05 (-0.05..+0.06) | 0.95/0.99 | -0.34 |
| dinov3_vith16plus | 840,592,640 | mag_r | -0.57 (-0.58..-0.57) | -0.08 (-0.08..+0.06) | 0.98/1.00 | -0.70 |
| dinov3_vith16plus | 840,592,640 | photo_z | -0.33 (-0.34..-0.33) | +0.05 (+0.05..+0.17) | 0.98/0.99 | -0.31 |
| dinov3_vith16plus | 840,592,640 | smooth_fraction | -0.32 (-0.32..-0.30) | +0.05 (+0.05..+0.13) | 0.95/0.99 | -0.43 |
| dinov3_vith16plus | 840,592,640 | stellar_mass | -0.27 (-0.29..-0.27) | -0.13 (-0.13..-0.05) | 0.98/0.99 | -0.34 |
| dinov3_vit7b16 | 6,716,035,072 | mag_r | -0.65 (-0.66..-0.65) | +0.15 (+0.10..+0.15) | 0.97/1.00 | -0.80 |
| dinov3_vit7b16 | 6,716,035,072 | photo_z | -0.37 (-0.42..-0.37) | +0.17 (+0.02..+0.17) | 0.97/1.00 | -0.38 |
| dinov3_vit7b16 | 6,716,035,072 | smooth_fraction | -0.36 (-0.39..-0.36) | +0.10 (+0.07..+0.11) | 0.94/0.99 | -0.64 |
| dinov3_vit7b16 | 6,716,035,072 | stellar_mass | -0.37 (-0.37..-0.33) | -0.05 (-0.05..+0.02) | 0.90/1.00 | -0.47 |

Spearman with log params (descriptive):
- mag_r hess_mismatch_emp: +0.09 (n=6)
- mag_r align_cos_tan: -0.66 (n=6)
- photo_z hess_mismatch_emp: +0.89 (n=6)
- photo_z align_cos_tan: +0.31 (n=6)
- smooth_fraction hess_mismatch_emp: -0.71 (n=6)
- smooth_fraction align_cos_tan: -0.49 (n=6)
- stellar_mass hess_mismatch_emp: -0.71 (n=6)
- stellar_mass align_cos_tan: -0.60 (n=6)
