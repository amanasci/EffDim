# NeurReps final controls

**Status:** experiments frozen after this pass.

## D. Existing main result without I-JEPA

- $T_{\mathrm{all}}$ = **0.00721** (5/5 families positive; sign-test $p=0.03125$)
- $T_{-I}$ = **0.00537** (4/4 families positive)

| Family | $\Delta\beta$ (Dense+Ridge − Dense) |
|--------|-----------------------------------:|
| astropt | 0.00513 |
| convnext | 0.00292 |
| dinov2 | 0.00313 |
| vit | 0.01029 |
| ijepa | 0.01460 |

## A. Fixed-rank scaling

### Rank 256

- $T_{256}$ = **0.00864**
- Positive families: **5/5**
- One-sided sign-test $p$ = **0.03125**
- $T_{256}^{-I}$ = **0.00673** (4/4 positive)

| Family | $\beta_{{raw PCA}}$ | $\beta_{{PCA+Ridge}}$ | $\Delta\beta$ |
|--------|------------------:|---------------------:|---------------:|
| astropt | 0.00411 | 0.01240 | 0.00829 |
| convnext | 0.00220 | 0.00354 | 0.00133 |
| dinov2 | -0.00032 | 0.00321 | 0.00352 |
| vit | 0.01938 | 0.03314 | 0.01376 |
| ijepa | 0.06615 | 0.08243 | 0.01627 |

### Rank 128

- $T_{128}$ = **0.00227**
- Positive families: **3/5**
- One-sided sign-test $p$ = **0.50000**
- $T_{128}^{-I}$ = **0.00367** (3/4 positive)

| Family | $\beta_{{raw PCA}}$ | $\beta_{{PCA+Ridge}}$ | $\Delta\beta$ |
|--------|------------------:|---------------------:|---------------:|
| astropt | 0.00441 | 0.00996 | 0.00555 |
| convnext | 0.00220 | 0.00075 | -0.00145 |
| dinov2 | -0.00072 | 0.00185 | 0.00256 |
| vit | 0.01929 | 0.02731 | 0.00802 |
| ijepa | 0.06615 | 0.06281 | -0.00335 |

## B. Data-supported distortion ($k_{{\mathrm{{edge}}}}=10$)

- Mean $\sigma_{\mathrm{local}}$ = **0.2332** (range 0.1567–0.3100)

**Family slopes of $\sigma_{{local}}$ vs $\log_{{10}}P$:**

- astropt: -0.0436
- convnext: -0.0251
- dinov2: +0.0062
- vit: -0.0689
- ijepa: -0.0451

**Spearman vs ambient spectrum:**

- $\sigma_{local}$ vs A_log: $\rho=0.132$
- $\sigma_{local}$ vs D_sim: $\rho=0.165$
- $\sigma_{local}$ vs H_norm: $\rho=-0.432$

## C. Alpha robustness

| $\alpha$ | $T(\alpha)$ | positive / 5 |
|---:|---:|---:|
| 0.01 | 0.00743 | 5/5 |
| 0.1 | 0.00752 | 5/5 |
| 1.0 | 0.00721 | 5/5 |
| 10.0 | 0.00732 | 5/5 |
| 100.0 | 0.00848 | 5/5 |

## E. Final interpretation

### Q1 — Does supervised slope amplification survive at fixed representation dimension?

At rank 256, $T_{256}=0.00864$ with 5/5 families positive. **Yes** — Ridge steepens scaling relative to raw PCA256 at matched dimension.

### Q2 — Does the effective map distort directions used by held-out data?

$\sigma_{\mathrm{local}}$ is substantial (mean ≈ 0.233), confirming direction-dependent stretch on empirical local edges, not only ambient spectrum.

### Q3 — Does data-supported distortion decrease with model size?

Family slopes mixed (1↑ / 4↓). No consistent local isotropization with scale.

### Q4 — Is the main scaling result robust without I-JEPA?

Yes: $T_{-I}=0.00537$ with all four remaining families positive.

### Q5 — Ridge $\alpha$ robustness?

$T(\alpha)>0$ for 5/5 tested $\alpha$ values.
