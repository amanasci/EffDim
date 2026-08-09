# Is the sparse fringe more curved than dense regions?

- n_max=16384, test_size=0.3, seed=0
- K ladder=200,300,400, density proxy d_k at K=50
- p_quad=3, m_norm=5, n_perm=16, min_k_factor=6

Q1 = densest quartile, Q4 = sparsest. `rank_biserial` compares Q4 vs Q1: positive means the sparsest quartile has larger values. `rho` is Spearman against d_k (larger d_k = sparser).

## 1. Fixed-radius feasibility gate

Why fixed-k neighbourhoods were used rather than fixed-radius ones.

| model | k_t | median d_k Q1 | median d_k Q4 | ratio | predicted Q4 neighbours at Q1 eps | verdict |
|---|---:|---:|---:|---:|---:|---|
| vit_base | 22 | 0.3517 | 0.5718 | 1.626 | 0.0011 | epsilon-ball infeasible |
| dinov3_vitb16 | 21 | 0.3178 | 0.4939 | 1.554 | 0.0048 | epsilon-ball infeasible |

## 2. Methodology validation (synthetic ground truth)

Manifolds with known curvature and deliberately non-uniform density. A usable metric must read ~1.0 with rho ~ 0 on the flat manifolds, and must still detect the sphere. rho far from zero on a FLAT manifold means the metric is measuring neighbourhood radius, not geometry.

| control | true kappa | K | metric | median | rho(d_k, metric) |
|---|---:|---:|---|---:|---:|
| synthetic flat | 0.00 | 200 | kappa_ratio | 1.0622 | -0.109 |
| synthetic flat | 0.00 | 200 | kappa_jet | 0.1891 | -0.996 |
| synthetic flat | 0.00 | 200 | rf_k | 0.1884 | -0.999 |
| synthetic flat | 0.00 | 200 | kappa_naive_ratio | 2.3339 | -0.973 |
| synthetic flat | 0.00 | 300 | kappa_ratio | 1.0422 | -0.098 |
| synthetic flat | 0.00 | 300 | kappa_jet | 0.1481 | -0.996 |
| synthetic flat | 0.00 | 300 | rf_k | 0.2225 | -0.999 |
| synthetic flat | 0.00 | 300 | kappa_naive_ratio | 3.8542 | -0.972 |
| synthetic flat | 0.00 | 400 | kappa_ratio | 1.0303 | -0.067 |
| synthetic flat | 0.00 | 400 | kappa_jet | 0.1252 | -0.997 |
| synthetic flat | 0.00 | 400 | rf_k | 0.2458 | -0.999 |
| synthetic flat | 0.00 | 400 | kappa_naive_ratio | 5.2810 | -0.973 |
| synthetic sphere | 0.30 | 200 | kappa_ratio | 1.2891 | -0.065 |
| synthetic sphere | 0.30 | 200 | kappa_jet | 0.0800 | -0.129 |
| synthetic sphere | 0.30 | 200 | rf_k | 0.0328 | -0.525 |
| synthetic sphere | 0.30 | 200 | kappa_naive_ratio | 0.1170 | -0.849 |
| synthetic sphere | 0.30 | 300 | kappa_ratio | 1.3861 | -0.106 |
| synthetic sphere | 0.30 | 300 | kappa_jet | 0.0741 | -0.160 |
| synthetic sphere | 0.30 | 300 | rf_k | 0.0340 | -0.572 |
| synthetic sphere | 0.30 | 300 | kappa_naive_ratio | 0.1163 | -0.845 |
| synthetic sphere | 0.30 | 400 | kappa_ratio | 1.4945 | -0.092 |
| synthetic sphere | 0.30 | 400 | kappa_jet | 0.0720 | -0.157 |
| synthetic sphere | 0.30 | 400 | rf_k | 0.0346 | -0.594 |
| synthetic sphere | 0.30 | 400 | kappa_naive_ratio | 0.1156 | -0.829 |

## 3. Headline — vit_base, K=200

| series | Q1 | Q2 | Q3 | Q4 | rank_biserial Q4-Q1 | rho(d_k) | rho 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **Real** | 1.5802 | 1.6427 | 1.7325 | 1.9243 | +0.590 | +0.397 | [+0.371, +0.420] |
| Flat surrogate (gauss) | 1.1159 | 1.1526 | 1.1584 | 1.1109 | -0.029 | -0.012 | [-0.040, +0.015] |
| Flat surrogate (shuffle) | 1.1719 | 1.1977 | 1.2285 | 1.2499 | +0.216 | +0.142 | [+0.112, +0.169] |
| _method bias floor (synthetic flat)_ | 1.1228 | 1.0661 | 1.0583 | 1.0569 | -0.186 | -0.109 | [-0.141, -0.079] |

### 4. Diagnostics and negative controls — vit_base, K=200

These are reported for completeness. Section 2 shows they carry a large density trend on manifolds that are exactly flat, so they cannot support a conclusion in either direction.

| metric | series | Q1 | Q2 | Q3 | Q4 | rank_biserial | rho(d_k) | rho 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kappa_jet | real | 1.7866 | 1.6218 | 1.6039 | 1.6460 | -0.260 | -0.162 | [-0.191, -0.135] |
| kappa_jet | null:gauss | 1.5216 | 1.2598 | 1.0781 | 0.7993 | -0.945 | -0.764 | [-0.777, -0.750] |
| kappa_jet | null:shuffle | 1.4845 | 1.2875 | 1.1972 | 1.0774 | -0.659 | -0.460 | [-0.482, -0.437] |
| kappa_null | real | 1.1452 | 1.0132 | 0.9459 | 0.8620 | -0.852 | -0.615 | [-0.632, -0.597] |
| kappa_null | null:gauss | 1.3738 | 1.1079 | 0.9305 | 0.7257 | -0.976 | -0.851 | [-0.860, -0.840] |
| kappa_null | null:shuffle | 1.2725 | 1.0627 | 0.9711 | 0.8439 | -0.831 | -0.625 | [-0.643, -0.606] |
| rf_k | real | 0.2128 | 0.2175 | 0.2238 | 0.2344 | +0.429 | +0.278 | [+0.252, +0.303] |
| rf_k | null:gauss | 0.5101 | 0.4602 | 0.4193 | 0.3617 | -0.996 | -0.904 | [-0.910, -0.897] |
| rf_k | null:shuffle | 0.3301 | 0.3009 | 0.2859 | 0.2634 | -0.835 | -0.633 | [-0.651, -0.614] |
| kappa_naive_ratio | real | 4.2195 | 3.8385 | 3.6096 | 3.2770 | -0.877 | -0.643 | [-0.659, -0.624] |
| kappa_naive_ratio | null:gauss | 8.6134 | 7.0725 | 6.0358 | 4.8377 | -0.988 | -0.869 | [-0.877, -0.861] |
| kappa_naive_ratio | null:shuffle | 5.4139 | 4.6371 | 4.2357 | 3.7622 | -0.840 | -0.632 | [-0.650, -0.614] |
| kappa_slope | real | 0.7553 | 0.7202 | 0.6862 | 0.6051 | -0.516 | -0.328 | [-0.354, -0.304] |
| kappa_slope | null:gauss | -0.3922 | -0.3003 | -0.2533 | -0.1798 | +0.964 | +0.787 | [+0.775, +0.799] |
| kappa_slope | null:shuffle | -0.2140 | -0.1134 | -0.0583 | -0.0051 | +0.615 | +0.417 | [+0.393, +0.440] |
| noise_floor | real | 0.1062 | 0.1171 | 0.1279 | 0.1523 | +0.974 | +0.797 | [+0.787, +0.808] |
| noise_floor | null:gauss | 0.2619 | 0.2619 | 0.2622 | 0.2622 | +0.081 | +0.067 | [+0.040, +0.095] |
| noise_floor | null:shuffle | 0.1913 | 0.1929 | 0.1943 | 0.1975 | +0.369 | +0.244 | [+0.220, +0.274] |
| R_med | real | 0.2786 | 0.3117 | 0.3423 | 0.3863 | +0.987 | +0.857 | [+0.848, +0.866] |
| R_med | null:gauss | 0.3476 | 0.3662 | 0.3830 | 0.4106 | +0.991 | +0.878 | [+0.870, +0.885] |
| R_med | null:shuffle | 0.3198 | 0.3424 | 0.3582 | 0.3789 | +0.863 | +0.659 | [+0.642, +0.676] |

## 3. Headline — vit_base, K=300

| series | Q1 | Q2 | Q3 | Q4 | rank_biserial Q4-Q1 | rho(d_k) | rho 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **Real** | 1.8601 | 1.9280 | 2.0123 | 2.2601 | +0.540 | +0.361 | [+0.337, +0.384] |
| Flat surrogate (gauss) | 1.1609 | 1.1874 | 1.1448 | 1.0738 | -0.260 | -0.176 | [-0.203, -0.149] |
| Flat surrogate (shuffle) | 1.2606 | 1.3088 | 1.3299 | 1.3592 | +0.240 | +0.166 | [+0.140, +0.195] |
| _method bias floor (synthetic flat)_ | 1.0900 | 1.0452 | 1.0393 | 1.0376 | -0.168 | -0.098 | [-0.132, -0.065] |

### 4. Diagnostics and negative controls — vit_base, K=300

These are reported for completeness. Section 2 shows they carry a large density trend on manifolds that are exactly flat, so they cannot support a conclusion in either direction.

| metric | series | Q1 | Q2 | Q3 | Q4 | rank_biserial | rho(d_k) | rho 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kappa_jet | real | 1.5478 | 1.4087 | 1.3953 | 1.4650 | -0.176 | -0.107 | [-0.132, -0.080] |
| kappa_jet | null:gauss | 1.0230 | 0.8589 | 0.7109 | 0.5168 | -0.935 | -0.749 | [-0.761, -0.734] |
| kappa_jet | null:shuffle | 1.1633 | 1.0113 | 0.9666 | 0.8693 | -0.596 | -0.405 | [-0.427, -0.379] |
| kappa_null | real | 0.8413 | 0.7521 | 0.7087 | 0.6485 | -0.840 | -0.596 | [-0.616, -0.577] |
| kappa_null | null:gauss | 0.8969 | 0.7311 | 0.6257 | 0.4843 | -0.950 | -0.812 | [-0.824, -0.799] |
| kappa_null | null:shuffle | 0.9235 | 0.7780 | 0.7140 | 0.6214 | -0.807 | -0.601 | [-0.619, -0.581] |
| rf_k | real | 0.2214 | 0.2270 | 0.2340 | 0.2457 | +0.458 | +0.298 | [+0.272, +0.321] |
| rf_k | null:gauss | 0.5126 | 0.4623 | 0.4204 | 0.3604 | -0.983 | -0.869 | [-0.877, -0.859] |
| rf_k | null:shuffle | 0.3410 | 0.3096 | 0.2938 | 0.2706 | -0.833 | -0.627 | [-0.646, -0.608] |
| kappa_naive_ratio | real | 4.1722 | 3.8361 | 3.6198 | 3.3106 | -0.851 | -0.614 | [-0.631, -0.596] |
| kappa_naive_ratio | null:gauss | 8.4983 | 6.9836 | 5.9285 | 4.6997 | -0.969 | -0.831 | [-0.842, -0.820] |
| kappa_naive_ratio | null:shuffle | 5.4388 | 4.6141 | 4.2112 | 3.6971 | -0.828 | -0.618 | [-0.635, -0.600] |
| kappa_slope | real | 0.8557 | 0.8275 | 0.8088 | 0.7381 | -0.403 | -0.245 | [-0.268, -0.218] |
| kappa_slope | null:gauss | -0.2474 | -0.1922 | -0.1635 | -0.1179 | +0.923 | +0.731 | [+0.716, +0.745] |
| kappa_slope | null:shuffle | -0.2793 | -0.1515 | -0.0966 | -0.0236 | +0.678 | +0.479 | [+0.455, +0.501] |
| noise_floor | real | 0.1088 | 0.1193 | 0.1293 | 0.1505 | +0.950 | +0.748 | [+0.736, +0.761] |
| noise_floor | null:gauss | 0.2645 | 0.2647 | 0.2651 | 0.2650 | +0.172 | +0.135 | [+0.109, +0.162] |
| noise_floor | null:shuffle | 0.2038 | 0.2040 | 0.2048 | 0.2068 | +0.232 | +0.148 | [+0.123, +0.176] |
| R_med | real | 0.2906 | 0.3207 | 0.3508 | 0.3951 | +0.970 | +0.820 | [+0.809, +0.832] |
| R_med | null:gauss | 0.3556 | 0.3740 | 0.3921 | 0.4208 | +0.972 | +0.838 | [+0.827, +0.848] |
| R_med | null:shuffle | 0.3308 | 0.3525 | 0.3685 | 0.3924 | +0.842 | +0.633 | [+0.614, +0.651] |

## 3. Headline — vit_base, K=400

| series | Q1 | Q2 | Q3 | Q4 | rank_biserial Q4-Q1 | rho(d_k) | rho 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **Real** | 2.1500 | 2.2039 | 2.3019 | 2.5535 | +0.494 | +0.326 | [+0.298, +0.350] |
| Flat surrogate (gauss) | 1.2117 | 1.1867 | 1.1211 | 1.0518 | -0.487 | -0.328 | [-0.350, -0.304] |
| Flat surrogate (shuffle) | 1.3813 | 1.4217 | 1.4521 | 1.4884 | +0.238 | +0.161 | [+0.138, +0.188] |
| _method bias floor (synthetic flat)_ | 1.0497 | 1.0386 | 1.0183 | 1.0219 | -0.091 | -0.067 | [-0.098, -0.039] |

### 4. Diagnostics and negative controls — vit_base, K=400

These are reported for completeness. Section 2 shows they carry a large density trend on manifolds that are exactly flat, so they cannot support a conclusion in either direction.

| metric | series | Q1 | Q2 | Q3 | Q4 | rank_biserial | rho(d_k) | rho 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kappa_jet | real | 1.4580 | 1.3193 | 1.3050 | 1.3743 | -0.158 | -0.099 | [-0.126, -0.070] |
| kappa_jet | null:gauss | 0.7958 | 0.6482 | 0.5188 | 0.3861 | -0.945 | -0.753 | [-0.765, -0.741] |
| kappa_jet | null:shuffle | 1.0176 | 0.9048 | 0.8525 | 0.7770 | -0.554 | -0.370 | [-0.392, -0.344] |
| kappa_null | real | 0.6820 | 0.6156 | 0.5808 | 0.5387 | -0.823 | -0.579 | [-0.597, -0.558] |
| kappa_null | null:gauss | 0.6628 | 0.5502 | 0.4706 | 0.3670 | -0.936 | -0.790 | [-0.803, -0.776] |
| kappa_null | null:shuffle | 0.7410 | 0.6375 | 0.5838 | 0.5143 | -0.798 | -0.588 | [-0.606, -0.570] |
| rf_k | real | 0.2256 | 0.2313 | 0.2383 | 0.2499 | +0.480 | +0.313 | [+0.287, +0.338] |
| rf_k | null:gauss | 0.5069 | 0.4587 | 0.4162 | 0.3541 | -0.967 | -0.842 | [-0.853, -0.830] |
| rf_k | null:shuffle | 0.3447 | 0.3127 | 0.2964 | 0.2728 | -0.827 | -0.616 | [-0.634, -0.598] |
| kappa_naive_ratio | real | 4.1000 | 3.8054 | 3.6025 | 3.3050 | -0.834 | -0.596 | [-0.613, -0.578] |
| kappa_naive_ratio | null:gauss | 8.2036 | 6.8393 | 5.7939 | 4.5686 | -0.949 | -0.802 | [-0.816, -0.789] |
| kappa_naive_ratio | null:shuffle | 5.3516 | 4.5601 | 4.1528 | 3.6143 | -0.815 | -0.601 | [-0.618, -0.583] |
| kappa_slope | real | 0.8839 | 0.8673 | 0.8642 | 0.8047 | -0.288 | -0.170 | [-0.195, -0.142] |
| kappa_slope | null:gauss | -0.1786 | -0.1383 | -0.1209 | -0.0896 | +0.898 | +0.694 | [+0.678, +0.711] |
| kappa_slope | null:shuffle | -0.2993 | -0.1754 | -0.1116 | -0.0329 | +0.699 | +0.504 | [+0.481, +0.524] |
| noise_floor | real | 0.1112 | 0.1211 | 0.1314 | 0.1494 | +0.915 | +0.700 | [+0.687, +0.714] |
| noise_floor | null:gauss | 0.2660 | 0.2661 | 0.2665 | 0.2665 | +0.175 | +0.138 | [+0.111, +0.168] |
| noise_floor | null:shuffle | 0.2108 | 0.2107 | 0.2116 | 0.2127 | +0.162 | +0.098 | [+0.071, +0.127] |
| R_med | real | 0.3002 | 0.3278 | 0.3576 | 0.4017 | +0.959 | +0.800 | [+0.787, +0.812] |
| R_med | null:gauss | 0.3622 | 0.3797 | 0.3978 | 0.4277 | +0.952 | +0.807 | [+0.794, +0.820] |
| R_med | null:shuffle | 0.3392 | 0.3599 | 0.3750 | 0.4014 | +0.826 | +0.615 | [+0.596, +0.632] |
| kappa_ms | real | 0.0138 | 0.0146 | 0.0145 | 0.0159 | +0.265 | +0.175 | [+0.144, +0.203] |
| kappa_ms | null:gauss | -0.0051 | -0.0034 | -0.0039 | -0.0023 | +0.195 | +0.142 | [+0.114, +0.169] |
| kappa_ms | null:shuffle | 0.0145 | 0.0136 | 0.0128 | 0.0122 | -0.164 | -0.102 | [-0.131, -0.074] |

## 3. Headline — dinov3_vitb16, K=200

| series | Q1 | Q2 | Q3 | Q4 | rank_biserial Q4-Q1 | rho(d_k) | rho 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **Real** | 1.6357 | 1.5384 | 1.6156 | 1.7947 | +0.296 | +0.196 | [+0.167, +0.222] |
| Flat surrogate (gauss) | 1.1408 | 1.1286 | 1.1331 | 1.0991 | -0.120 | -0.071 | [-0.100, -0.042] |
| Flat surrogate (shuffle) | 1.1870 | 1.1961 | 1.2293 | 1.2896 | +0.290 | +0.198 | [+0.173, +0.224] |
| _method bias floor (synthetic flat)_ | 1.1228 | 1.0661 | 1.0583 | 1.0569 | -0.186 | -0.109 | [-0.141, -0.079] |

### 4. Diagnostics and negative controls — dinov3_vitb16, K=200

These are reported for completeness. Section 2 shows they carry a large density trend on manifolds that are exactly flat, so they cannot support a conclusion in either direction.

| metric | series | Q1 | Q2 | Q3 | Q4 | rank_biserial | rho(d_k) | rho 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kappa_jet | real | 1.7354 | 1.6795 | 1.6470 | 1.6668 | -0.114 | -0.076 | [-0.104, -0.049] |
| kappa_jet | null:gauss | 1.5010 | 1.2448 | 1.0675 | 0.8287 | -0.955 | -0.781 | [-0.793, -0.768] |
| kappa_jet | null:shuffle | 1.6362 | 1.4361 | 1.3335 | 1.2293 | -0.656 | -0.464 | [-0.488, -0.439] |
| kappa_null | real | 1.0521 | 1.0869 | 1.0252 | 0.9281 | -0.458 | -0.301 | [-0.326, -0.274] |
| kappa_null | null:gauss | 1.3250 | 1.1060 | 0.9423 | 0.7642 | -0.990 | -0.883 | [-0.890, -0.876] |
| kappa_null | null:shuffle | 1.3644 | 1.1970 | 1.0720 | 0.9513 | -0.830 | -0.646 | [-0.665, -0.627] |
| rf_k | real | 0.1778 | 0.2035 | 0.2080 | 0.2111 | +0.361 | +0.235 | [+0.208, +0.266] |
| rf_k | null:gauss | 0.4824 | 0.4291 | 0.3877 | 0.3403 | -0.993 | -0.921 | [-0.927, -0.915] |
| rf_k | null:shuffle | 0.3146 | 0.2867 | 0.2665 | 0.2464 | -0.846 | -0.668 | [-0.686, -0.650] |
| kappa_naive_ratio | real | 3.8391 | 3.8230 | 3.6576 | 3.3321 | -0.563 | -0.390 | [-0.413, -0.364] |
| kappa_naive_ratio | null:gauss | 8.9932 | 7.2294 | 6.1083 | 5.0408 | -0.978 | -0.879 | [-0.889, -0.868] |
| kappa_naive_ratio | null:shuffle | 5.7245 | 4.8877 | 4.3419 | 3.8802 | -0.835 | -0.659 | [-0.680, -0.639] |
| kappa_slope | real | 0.5583 | 0.4448 | 0.5084 | 0.5697 | +0.046 | +0.061 | [+0.031, +0.092] |
| kappa_slope | null:gauss | -0.3779 | -0.2898 | -0.2350 | -0.1937 | +0.936 | +0.745 | [+0.732, +0.758] |
| kappa_slope | null:shuffle | -0.1104 | -0.1659 | -0.0961 | 0.0030 | +0.242 | +0.183 | [+0.153, +0.213] |
| noise_floor | real | 0.0974 | 0.1182 | 0.1246 | 0.1330 | +0.845 | +0.629 | [+0.610, +0.647] |
| noise_floor | null:gauss | 0.2242 | 0.2247 | 0.2249 | 0.2255 | +0.345 | +0.244 | [+0.215, +0.273] |
| noise_floor | null:shuffle | 0.1675 | 0.1721 | 0.1736 | 0.1724 | +0.348 | +0.247 | [+0.219, +0.274] |
| R_med | real | 0.2631 | 0.2913 | 0.3136 | 0.3450 | +0.962 | +0.858 | [+0.847, +0.869] |
| R_med | null:gauss | 0.3069 | 0.3264 | 0.3433 | 0.3651 | +0.980 | +0.885 | [+0.875, +0.894] |
| R_med | null:shuffle | 0.2905 | 0.3107 | 0.3278 | 0.3436 | +0.861 | +0.696 | [+0.678, +0.714] |

## 3. Headline — dinov3_vitb16, K=300

| series | Q1 | Q2 | Q3 | Q4 | rank_biserial Q4-Q1 | rho(d_k) | rho 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **Real** | 1.9058 | 1.7888 | 1.8752 | 2.0852 | +0.290 | +0.192 | [+0.166, +0.220] |
| Flat surrogate (gauss) | 1.1720 | 1.1330 | 1.1189 | 1.0776 | -0.303 | -0.206 | [-0.233, -0.179] |
| Flat surrogate (shuffle) | 1.3190 | 1.3018 | 1.3581 | 1.4121 | +0.238 | +0.169 | [+0.140, +0.193] |
| _method bias floor (synthetic flat)_ | 1.0900 | 1.0452 | 1.0393 | 1.0376 | -0.168 | -0.098 | [-0.132, -0.065] |

### 4. Diagnostics and negative controls — dinov3_vitb16, K=300

These are reported for completeness. Section 2 shows they carry a large density trend on manifolds that are exactly flat, so they cannot support a conclusion in either direction.

| metric | series | Q1 | Q2 | Q3 | Q4 | rank_biserial | rho(d_k) | rho 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kappa_jet | real | 1.5027 | 1.4583 | 1.4544 | 1.5073 | +0.029 | +0.021 | [-0.005, +0.048] |
| kappa_jet | null:gauss | 0.9722 | 0.8137 | 0.6902 | 0.5499 | -0.947 | -0.771 | [-0.783, -0.759] |
| kappa_jet | null:shuffle | 1.2982 | 1.1443 | 1.0811 | 1.0195 | -0.584 | -0.402 | [-0.425, -0.377] |
| kappa_null | real | 0.7635 | 0.8047 | 0.7697 | 0.7092 | -0.323 | -0.211 | [-0.237, -0.183] |
| kappa_null | null:gauss | 0.8277 | 0.7182 | 0.6208 | 0.5125 | -0.986 | -0.862 | [-0.870, -0.854] |
| kappa_null | null:shuffle | 0.9735 | 0.8708 | 0.7884 | 0.7053 | -0.807 | -0.621 | [-0.640, -0.601] |
| rf_k | real | 0.1805 | 0.2087 | 0.2148 | 0.2202 | +0.413 | +0.273 | [+0.246, +0.303] |
| rf_k | null:gauss | 0.4710 | 0.4245 | 0.3851 | 0.3397 | -0.993 | -0.911 | [-0.918, -0.905] |
| rf_k | null:shuffle | 0.3185 | 0.2926 | 0.2729 | 0.2536 | -0.845 | -0.666 | [-0.685, -0.648] |
| kappa_naive_ratio | real | 3.6294 | 3.7292 | 3.6111 | 3.3435 | -0.434 | -0.293 | [-0.318, -0.265] |
| kappa_naive_ratio | null:gauss | 8.4554 | 6.9466 | 5.9609 | 4.9621 | -0.974 | -0.866 | [-0.876, -0.854] |
| kappa_naive_ratio | null:shuffle | 5.5549 | 4.7846 | 4.2911 | 3.8901 | -0.831 | -0.653 | [-0.673, -0.634] |
| kappa_slope | real | 0.6128 | 0.5299 | 0.5916 | 0.6816 | +0.175 | +0.153 | [+0.123, +0.184] |
| kappa_slope | null:gauss | -0.2263 | -0.1803 | -0.1457 | -0.1296 | +0.889 | +0.667 | [+0.652, +0.682] |
| kappa_slope | null:shuffle | -0.1290 | -0.1828 | -0.1141 | 0.0002 | +0.231 | +0.190 | [+0.157, +0.219] |
| noise_floor | real | 0.1008 | 0.1210 | 0.1261 | 0.1320 | +0.782 | +0.564 | [+0.542, +0.584] |
| noise_floor | null:gauss | 0.2265 | 0.2269 | 0.2269 | 0.2276 | +0.348 | +0.239 | [+0.214, +0.264] |
| noise_floor | null:shuffle | 0.1769 | 0.1811 | 0.1819 | 0.1798 | +0.240 | +0.169 | [+0.138, +0.205] |
| R_med | real | 0.2792 | 0.3027 | 0.3231 | 0.3520 | +0.964 | +0.848 | [+0.836, +0.858] |
| R_med | null:gauss | 0.3177 | 0.3355 | 0.3513 | 0.3725 | +0.975 | +0.871 | [+0.860, +0.880] |
| R_med | null:shuffle | 0.3039 | 0.3219 | 0.3374 | 0.3522 | +0.856 | +0.685 | [+0.667, +0.704] |

## 3. Headline — dinov3_vitb16, K=400

| series | Q1 | Q2 | Q3 | Q4 | rank_biserial Q4-Q1 | rho(d_k) | rho 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **Real** | 2.1896 | 2.0167 | 2.1414 | 2.3806 | +0.274 | +0.179 | [+0.152, +0.207] |
| Flat surrogate (gauss) | 1.1375 | 1.1158 | 1.0933 | 1.0464 | -0.291 | -0.191 | [-0.218, -0.166] |
| Flat surrogate (shuffle) | 1.4138 | 1.4387 | 1.5099 | 1.5463 | +0.283 | +0.198 | [+0.171, +0.225] |
| _method bias floor (synthetic flat)_ | 1.0497 | 1.0386 | 1.0183 | 1.0219 | -0.091 | -0.067 | [-0.098, -0.039] |

### 4. Diagnostics and negative controls — dinov3_vitb16, K=400

These are reported for completeness. Section 2 shows they carry a large density trend on manifolds that are exactly flat, so they cannot support a conclusion in either direction.

| metric | series | Q1 | Q2 | Q3 | Q4 | rank_biserial | rho(d_k) | rho 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| kappa_jet | real | 1.3958 | 1.3556 | 1.3637 | 1.4272 | +0.095 | +0.063 | [+0.036, +0.089] |
| kappa_jet | null:gauss | 0.6789 | 0.5951 | 0.5099 | 0.4041 | -0.929 | -0.732 | [-0.744, -0.718] |
| kappa_jet | null:shuffle | 1.1103 | 1.0001 | 0.9741 | 0.9205 | -0.482 | -0.321 | [-0.344, -0.296] |
| kappa_null | real | 0.6148 | 0.6531 | 0.6280 | 0.5922 | -0.228 | -0.155 | [-0.181, -0.127] |
| kappa_null | null:gauss | 0.5854 | 0.5281 | 0.4650 | 0.3871 | -0.980 | -0.833 | [-0.842, -0.823] |
| kappa_null | null:shuffle | 0.7681 | 0.7006 | 0.6459 | 0.5853 | -0.781 | -0.590 | [-0.609, -0.569] |
| rf_k | real | 0.1807 | 0.2098 | 0.2171 | 0.2241 | +0.455 | +0.302 | [+0.277, +0.331] |
| rf_k | null:gauss | 0.4532 | 0.4149 | 0.3784 | 0.3356 | -0.995 | -0.902 | [-0.908, -0.895] |
| rf_k | null:shuffle | 0.3168 | 0.2933 | 0.2745 | 0.2559 | -0.844 | -0.664 | [-0.683, -0.647] |
| kappa_naive_ratio | real | 3.4702 | 3.6312 | 3.5502 | 3.3294 | -0.318 | -0.208 | [-0.235, -0.180] |
| kappa_naive_ratio | null:gauss | 7.8308 | 6.6032 | 5.7555 | 4.8423 | -0.970 | -0.855 | [-0.865, -0.844] |
| kappa_naive_ratio | null:shuffle | 5.3068 | 4.6453 | 4.2162 | 3.8265 | -0.830 | -0.648 | [-0.668, -0.629] |
| kappa_slope | real | 0.6346 | 0.5574 | 0.6331 | 0.7269 | +0.281 | +0.220 | [+0.189, +0.249] |
| kappa_slope | null:gauss | -0.1587 | -0.1294 | -0.1046 | -0.0960 | +0.842 | +0.612 | [+0.595, +0.628] |
| kappa_slope | null:shuffle | -0.1316 | -0.1988 | -0.1285 | -0.0122 | +0.234 | +0.195 | [+0.161, +0.222] |
| noise_floor | real | 0.1068 | 0.1232 | 0.1280 | 0.1321 | +0.716 | +0.505 | [+0.482, +0.528] |
| noise_floor | null:gauss | 0.2281 | 0.2280 | 0.2280 | 0.2286 | +0.248 | +0.153 | [+0.126, +0.180] |
| noise_floor | null:shuffle | 0.1831 | 0.1867 | 0.1868 | 0.1846 | +0.136 | +0.096 | [+0.067, +0.131] |
| R_med | real | 0.2976 | 0.3118 | 0.3301 | 0.3572 | +0.974 | +0.840 | [+0.830, +0.850] |
| R_med | null:gauss | 0.3268 | 0.3432 | 0.3579 | 0.3782 | +0.972 | +0.860 | [+0.849, +0.870] |
| R_med | null:shuffle | 0.3169 | 0.3310 | 0.3451 | 0.3586 | +0.850 | +0.671 | [+0.653, +0.689] |
| kappa_ms | real | 0.0057 | 0.0074 | 0.0101 | 0.0123 | +0.677 | +0.485 | [+0.462, +0.510] |
| kappa_ms | null:gauss | -0.0286 | -0.0128 | -0.0092 | -0.0042 | +0.916 | +0.722 | [+0.706, +0.737] |
| kappa_ms | null:shuffle | 0.0021 | 0.0063 | 0.0070 | 0.0089 | +0.471 | +0.331 | [+0.305, +0.359] |

## 5. Scale ladder

See `<model>_scale_ladder.png`: median kappa_ratio against the actual median neighbourhood radius, one line per density quartile. Because each quartile spans its own radius range, overlapping x-values compare curvature at a matched physical scale.


## 6. Connect-back: does curvature explain probe / SAE failure?

`partial rho(curv, target | d_k)` asks whether curvature adds anything beyond raw density; `partial rho(d_k, target | curv)` asks the reverse. `rho(curv, n_valid)` is the label-availability confound.

### vit_base, K=200

| curvature | target | rho | partial rho given d_k | partial rho given n_valid | rho(curv, n_valid) |
|---|---|---:|---:|---:|---:|
| kappa_ratio | mean_residual_good | +0.118 | -0.032 | +0.122 | -0.036 |
| kappa_ratio | redshift_residual | +0.043 | +0.006 | +0.047 | -0.036 |
| kappa_ratio | mean_residual_all | +0.120 | -0.026 | +0.150 | -0.036 |
| kappa_ratio | sae_reconstruction_error | +0.309 | -0.172 | +0.310 | -0.036 |
| kappa_ratio | sae_atom_turnover_rate | -0.172 | -0.357 | -0.170 | -0.036 |
| kappa_jet | mean_residual_good | -0.026 | +0.036 | -0.031 | +0.044 |
| kappa_jet | redshift_residual | -0.044 | -0.026 | -0.041 | +0.044 |
| kappa_jet | mean_residual_all | +0.003 | +0.066 | -0.017 | +0.044 |
| kappa_jet | sae_reconstruction_error | -0.076 | +0.200 | -0.076 | +0.044 |
| kappa_jet | sae_atom_turnover_rate | +0.057 | +0.121 | +0.053 | +0.044 |
| rf_k | mean_residual_good | +0.226 | +0.139 | +0.213 | +0.182 |
| rf_k | redshift_residual | -0.015 | -0.051 | -0.004 | +0.182 |
| rf_k | mean_residual_all | +0.243 | +0.160 | +0.186 | +0.182 |
| rf_k | sae_reconstruction_error | +0.390 | +0.368 | +0.396 | +0.182 |
| rf_k | sae_atom_turnover_rate | +0.573 | +0.529 | +0.569 | +0.182 |
| kappa_naive_ratio | mean_residual_good | -0.169 | +0.095 | -0.182 | +0.114 |
| kappa_naive_ratio | redshift_residual | -0.107 | -0.055 | -0.107 | +0.114 |
| kappa_naive_ratio | mean_residual_all | -0.124 | +0.149 | -0.194 | +0.114 |
| kappa_naive_ratio | sae_reconstruction_error | -0.455 | +0.496 | -0.458 | +0.114 |
| kappa_naive_ratio | sae_atom_turnover_rate | +0.148 | +0.512 | +0.140 | +0.114 |

### vit_base, K=300

| curvature | target | rho | partial rho given d_k | partial rho given n_valid | rho(curv, n_valid) |
|---|---|---:|---:|---:|---:|
| kappa_ratio | mean_residual_good | +0.098 | -0.040 | +0.102 | -0.035 |
| kappa_ratio | redshift_residual | +0.048 | +0.015 | +0.050 | -0.035 |
| kappa_ratio | mean_residual_all | +0.098 | -0.036 | +0.126 | -0.035 |
| kappa_ratio | sae_reconstruction_error | +0.261 | -0.212 | +0.262 | -0.035 |
| kappa_ratio | sae_atom_turnover_rate | -0.242 | -0.417 | -0.240 | -0.035 |
| kappa_jet | mean_residual_good | -0.011 | +0.030 | -0.016 | +0.048 |
| kappa_jet | redshift_residual | -0.041 | -0.027 | -0.037 | +0.048 |
| kappa_jet | mean_residual_all | +0.020 | +0.063 | -0.001 | +0.048 |
| kappa_jet | sae_reconstruction_error | -0.050 | +0.133 | -0.051 | +0.048 |
| kappa_jet | sae_atom_turnover_rate | -0.025 | +0.012 | -0.030 | +0.048 |
| rf_k | mean_residual_good | +0.227 | +0.133 | +0.214 | +0.174 |
| rf_k | redshift_residual | -0.018 | -0.056 | -0.007 | +0.174 |
| rf_k | mean_residual_all | +0.246 | +0.156 | +0.192 | +0.174 |
| rf_k | sae_reconstruction_error | +0.405 | +0.361 | +0.410 | +0.174 |
| rf_k | sae_atom_turnover_rate | +0.559 | +0.510 | +0.555 | +0.174 |
| kappa_naive_ratio | mean_residual_good | -0.157 | +0.093 | -0.171 | +0.113 |
| kappa_naive_ratio | redshift_residual | -0.107 | -0.057 | -0.107 | +0.113 |
| kappa_naive_ratio | mean_residual_all | -0.111 | +0.148 | -0.179 | +0.113 |
| kappa_naive_ratio | sae_reconstruction_error | -0.427 | +0.483 | -0.431 | +0.113 |
| kappa_naive_ratio | sae_atom_turnover_rate | +0.161 | +0.500 | +0.152 | +0.113 |

### vit_base, K=400

| curvature | target | rho | partial rho given d_k | partial rho given n_valid | rho(curv, n_valid) |
|---|---|---:|---:|---:|---:|
| kappa_ratio | mean_residual_good | +0.088 | -0.035 | +0.093 | -0.038 |
| kappa_ratio | redshift_residual | +0.053 | +0.024 | +0.055 | -0.038 |
| kappa_ratio | mean_residual_all | +0.088 | -0.033 | +0.116 | -0.038 |
| kappa_ratio | sae_reconstruction_error | +0.230 | -0.205 | +0.230 | -0.038 |
| kappa_ratio | sae_atom_turnover_rate | -0.257 | -0.414 | -0.254 | -0.038 |
| kappa_jet | mean_residual_good | -0.014 | +0.024 | -0.019 | +0.041 |
| kappa_jet | redshift_residual | -0.042 | -0.030 | -0.039 | +0.041 |
| kappa_jet | mean_residual_all | +0.019 | +0.058 | +0.001 | +0.041 |
| kappa_jet | sae_reconstruction_error | -0.052 | +0.109 | -0.052 | +0.041 |
| kappa_jet | sae_atom_turnover_rate | -0.046 | -0.013 | -0.050 | +0.041 |
| rf_k | mean_residual_good | +0.230 | +0.130 | +0.217 | +0.171 |
| rf_k | redshift_residual | -0.019 | -0.058 | -0.008 | +0.171 |
| rf_k | mean_residual_all | +0.250 | +0.155 | +0.198 | +0.171 |
| rf_k | sae_reconstruction_error | +0.416 | +0.355 | +0.422 | +0.171 |
| rf_k | sae_atom_turnover_rate | +0.549 | +0.495 | +0.544 | +0.171 |
| kappa_naive_ratio | mean_residual_good | -0.151 | +0.091 | -0.164 | +0.112 |
| kappa_naive_ratio | redshift_residual | -0.107 | -0.059 | -0.106 | +0.112 |
| kappa_naive_ratio | mean_residual_all | -0.103 | +0.148 | -0.169 | +0.112 |
| kappa_naive_ratio | sae_reconstruction_error | -0.411 | +0.475 | -0.414 | +0.112 |
| kappa_naive_ratio | sae_atom_turnover_rate | +0.163 | +0.487 | +0.155 | +0.112 |

### dinov3_vitb16, K=200

| curvature | target | rho | partial rho given d_k | partial rho given n_valid | rho(curv, n_valid) |
|---|---|---:|---:|---:|---:|
| kappa_ratio | mean_residual_good | -0.033 | -0.109 | -0.032 | -0.011 |
| kappa_ratio | redshift_residual | -0.020 | -0.023 | -0.014 | -0.011 |
| kappa_ratio | mean_residual_all | -0.002 | -0.079 | +0.003 | -0.011 |
| kappa_ratio | sae_reconstruction_error | +0.091 | -0.170 | +0.092 | -0.011 |
| kappa_ratio | sae_atom_turnover_rate | -0.349 | -0.506 | -0.349 | -0.011 |
| kappa_jet | mean_residual_good | +0.074 | +0.107 | +0.072 | +0.022 |
| kappa_jet | redshift_residual | +0.050 | +0.052 | +0.055 | +0.022 |
| kappa_jet | mean_residual_all | +0.091 | +0.127 | +0.090 | +0.022 |
| kappa_jet | sae_reconstruction_error | +0.065 | +0.275 | +0.065 | +0.022 |
| kappa_jet | sae_atom_turnover_rate | -0.131 | -0.109 | -0.134 | +0.022 |
| rf_k | mean_residual_good | +0.350 | +0.295 | +0.346 | +0.074 |
| rf_k | redshift_residual | +0.076 | +0.073 | +0.090 | +0.074 |
| rf_k | mean_residual_all | +0.351 | +0.294 | +0.354 | +0.074 |
| rf_k | sae_reconstruction_error | +0.448 | +0.517 | +0.448 | +0.074 |
| rf_k | sae_atom_turnover_rate | +0.476 | +0.426 | +0.472 | +0.074 |
| kappa_naive_ratio | mean_residual_good | +0.113 | +0.287 | +0.109 | +0.045 |
| kappa_naive_ratio | redshift_residual | +0.095 | +0.111 | +0.095 | +0.045 |
| kappa_naive_ratio | mean_residual_all | +0.108 | +0.289 | +0.099 | +0.045 |
| kappa_naive_ratio | sae_reconstruction_error | -0.085 | +0.578 | -0.086 | +0.045 |
| kappa_naive_ratio | sae_atom_turnover_rate | +0.112 | +0.359 | +0.108 | +0.045 |

### dinov3_vitb16, K=300

| curvature | target | rho | partial rho given d_k | partial rho given n_valid | rho(curv, n_valid) |
|---|---|---:|---:|---:|---:|
| kappa_ratio | mean_residual_good | -0.082 | -0.161 | -0.082 | -0.001 |
| kappa_ratio | redshift_residual | -0.033 | -0.037 | -0.028 | -0.001 |
| kappa_ratio | mean_residual_all | -0.048 | -0.127 | -0.052 | -0.001 |
| kappa_ratio | sae_reconstruction_error | +0.065 | -0.217 | +0.065 | -0.001 |
| kappa_ratio | sae_atom_turnover_rate | -0.364 | -0.521 | -0.366 | -0.001 |
| kappa_jet | mean_residual_good | +0.044 | +0.039 | +0.042 | +0.025 |
| kappa_jet | redshift_residual | +0.026 | +0.026 | +0.032 | +0.025 |
| kappa_jet | mean_residual_all | +0.072 | +0.069 | +0.067 | +0.025 |
| kappa_jet | sae_reconstruction_error | +0.102 | +0.173 | +0.101 | +0.025 |
| kappa_jet | sae_atom_turnover_rate | -0.159 | -0.191 | -0.162 | +0.025 |
| rf_k | mean_residual_good | +0.357 | +0.291 | +0.353 | +0.077 |
| rf_k | redshift_residual | +0.074 | +0.071 | +0.089 | +0.077 |
| rf_k | mean_residual_all | +0.359 | +0.290 | +0.361 | +0.077 |
| rf_k | sae_reconstruction_error | +0.471 | +0.501 | +0.472 | +0.077 |
| rf_k | sae_atom_turnover_rate | +0.486 | +0.422 | +0.482 | +0.077 |
| kappa_naive_ratio | mean_residual_good | +0.153 | +0.284 | +0.149 | +0.053 |
| kappa_naive_ratio | redshift_residual | +0.092 | +0.102 | +0.094 | +0.053 |
| kappa_naive_ratio | mean_residual_all | +0.152 | +0.288 | +0.143 | +0.053 |
| kappa_naive_ratio | sae_reconstruction_error | -0.001 | +0.555 | -0.002 | +0.053 |
| kappa_naive_ratio | sae_atom_turnover_rate | +0.208 | +0.406 | +0.204 | +0.053 |

### dinov3_vitb16, K=400

| curvature | target | rho | partial rho given d_k | partial rho given n_valid | rho(curv, n_valid) |
|---|---|---:|---:|---:|---:|
| kappa_ratio | mean_residual_good | -0.098 | -0.174 | -0.099 | +0.011 |
| kappa_ratio | redshift_residual | -0.049 | -0.052 | -0.043 | +0.011 |
| kappa_ratio | mean_residual_all | -0.059 | -0.134 | -0.070 | +0.011 |
| kappa_ratio | sae_reconstruction_error | +0.048 | -0.229 | +0.048 | +0.011 |
| kappa_ratio | sae_atom_turnover_rate | -0.384 | -0.537 | -0.387 | +0.011 |
| kappa_jet | mean_residual_good | +0.024 | +0.003 | +0.021 | +0.037 |
| kappa_jet | redshift_residual | +0.004 | +0.003 | +0.011 | +0.037 |
| kappa_jet | mean_residual_all | +0.058 | +0.038 | +0.047 | +0.037 |
| kappa_jet | sae_reconstruction_error | +0.120 | +0.134 | +0.119 | +0.037 |
| kappa_jet | sae_atom_turnover_rate | -0.190 | -0.248 | -0.195 | +0.037 |
| rf_k | mean_residual_good | +0.359 | +0.284 | +0.354 | +0.077 |
| rf_k | redshift_residual | +0.067 | +0.064 | +0.083 | +0.077 |
| rf_k | mean_residual_all | +0.361 | +0.283 | +0.363 | +0.077 |
| rf_k | sae_reconstruction_error | +0.489 | +0.488 | +0.489 | +0.077 |
| rf_k | sae_atom_turnover_rate | +0.496 | +0.421 | +0.492 | +0.077 |
| kappa_naive_ratio | mean_residual_good | +0.176 | +0.270 | +0.171 | +0.062 |
| kappa_naive_ratio | redshift_residual | +0.079 | +0.085 | +0.084 | +0.062 |
| kappa_naive_ratio | mean_residual_all | +0.179 | +0.278 | +0.169 | +0.062 |
| kappa_naive_ratio | sae_reconstruction_error | +0.066 | +0.525 | +0.065 | +0.062 |
| kappa_naive_ratio | sae_atom_turnover_rate | +0.303 | +0.461 | +0.299 | +0.062 |

### 7. Probe health

- 8 of 38 probes have r2_cv > 0.1.
- Legacy target `mean_residual_all`: median 0.303, mean 1.825 (a standardised squared residual should be ~1).
- Cleaned target `mean_residual_good`: median 0.260, mean 0.512.
- Spearman(mean_residual_all, n_valid_probes) = +0.434 — the legacy target partly encodes which galaxies have rare labels.


## Limitations

- The permutation null assumes normal residuals are exchangeable across neighbours once the linear term is removed. Direction-dependent heteroscedastic thickness would misspecify it.
- `p_quad` selects the top-p_quad *local-variance* tangent directions, and which directions those are is itself mildly density-dependent.
- The flat surrogate bounds the artifact under the null hypothesis only. It does not bound residual scale sensitivity under a genuinely curved alternative.

