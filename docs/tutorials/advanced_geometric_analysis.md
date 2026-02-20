# Advanced Geometric Analysis

This tutorial showcases the advanced geometric estimators available in EffDim: DANCo, MiND-MLi, MiND-MLk, ESS, TLE, and GMST.

## Overview of New Estimators

| Estimator | Key Idea | Parameters |
|-----------|----------|------------|
| **DANCo** | Angle concentration between neighbor vectors | `k` (neighbors) |
| **MiND-MLi** | Nearest-neighbor distance distribution | None |
| **MiND-MLk** | k-neighbor distances, median aggregation | `k` (neighbors) |
| **ESS** | Simplex skewness of local neighborhoods | `k` (neighbors) |
| **TLE** | Scale-normalized distance likelihood | `k` (neighbors) |
| **GMST** | MST length scaling with sample size | `geodesic` (bool) |

## Example: Swiss Roll Analysis

The Swiss Roll is a classic benchmark — a 2D manifold embedded in 3D space.

```python
import numpy as np
import effdim
from sklearn.datasets import make_swiss_roll

# Generate Swiss Roll (intrinsic dimension = 2)
X, _ = make_swiss_roll(n_samples=2000, noise=0.01, random_state=42)

results = effdim.compute_dim(X)

print("=== Spectral Estimators ===")
print(f"PCA (95% variance):  {results['pca_explained_variance_95']}")
print(f"Participation Ratio: {results['participation_ratio']:.2f}")

print("\n=== Classic Geometric Estimators ===")
print(f"kNN MLE:   {results['mle_dimensionality']:.2f}")
print(f"Two-NN:    {results['two_nn_dimensionality']:.2f}")

print("\n=== New Geometric Estimators ===")
print(f"DANCo:     {results['danco_dimensionality']:.2f}")
print(f"MiND-MLi:  {results['mind_mli_dimensionality']:.2f}")
print(f"MiND-MLk:  {results['mind_mlk_dimensionality']:.2f}")
print(f"ESS:       {results['ess_dimensionality']:.2f}")
print(f"TLE:       {results['tle_dimensionality']:.2f}")
print(f"GMST:      {results['gmst_dimensionality']:.2f}")
```

## Example: Comparing Estimators on Known Manifolds

```python
import numpy as np
import effdim

# 1D curve in 3D (helix)
t = np.linspace(0, 4 * np.pi, 500)
helix = np.column_stack([np.cos(t), np.sin(t), t / (4 * np.pi)])
results_1d = effdim.compute_dim(helix)

# 2D plane in 5D
np.random.seed(42)
plane_2d = np.random.randn(500, 2)
embedding = np.random.randn(2, 5)
data_5d = plane_2d @ embedding
results_2d = effdim.compute_dim(data_5d)

# 5D Gaussian
data_5d_full = np.random.randn(500, 5)
results_5d = effdim.compute_dim(data_5d_full)

print("Estimator         | 1D Helix | 2D in 5D | 5D Gaussian")
print("-" * 55)
for key in ['mle_dimensionality', 'two_nn_dimensionality',
            'mind_mlk_dimensionality', 'tle_dimensionality']:
    print(f"{key:25s} | {results_1d[key]:7.2f}  | {results_2d[key]:7.2f}  | {results_5d[key]:7.2f}")
```

## Using GMST with Geodesic Distances

The GMST estimator supports geodesic distances, which follow the manifold surface rather than cutting through ambient space. This is especially useful for curved manifolds.

```python
import numpy as np
from effdim.geometry import gmst_dimensionality
from sklearn.datasets import make_swiss_roll

X, _ = make_swiss_roll(n_samples=500, noise=0.01, random_state=42)

# Euclidean mode
dim_euclidean = gmst_dimensionality(X, geodesic=False)
print(f"GMST (Euclidean): {dim_euclidean:.2f}")

# Geodesic mode
dim_geodesic = gmst_dimensionality(X, geodesic=True)
print(f"GMST (Geodesic):  {dim_geodesic:.2f}")
```

## Using Individual Estimators

Each estimator can be called directly for fine-grained control:

```python
import numpy as np
from effdim.geometry import (
    danco_dimensionality,
    mind_mli_dimensionality,
    mind_mlk_dimensionality,
    ess_dimensionality,
    tle_dimensionality,
    gmst_dimensionality,
)

np.random.seed(42)
data = np.random.randn(200, 5)

# DANCo with custom k
print(f"DANCo (k=5):  {danco_dimensionality(data, k=5):.2f}")
print(f"DANCo (k=15): {danco_dimensionality(data, k=15):.2f}")

# MiND variants
print(f"MiND-MLi:     {mind_mli_dimensionality(data):.2f}")
print(f"MiND-MLk:     {mind_mlk_dimensionality(data, k=10):.2f}")

# ESS and TLE
print(f"ESS (k=10):   {ess_dimensionality(data, k=10):.2f}")
print(f"TLE (k=10):   {tle_dimensionality(data, k=10):.2f}")

# GMST
print(f"GMST:         {gmst_dimensionality(data):.2f}")
```

## When to Use Which Estimator

- **DANCo**: Best when angle-based geometric analysis is informative; works well for higher-dimensional data.
- **MiND-MLi**: Quick, parameter-free estimate using only nearest-neighbor distances. Best for quick sanity checks.
- **MiND-MLk**: More robust version of MLE using median aggregation. Use when outlier robustness is important.
- **ESS**: Low-bias estimator based on simplex geometry. Good for manifolds with moderate curvature.
- **TLE**: Scale-invariant due to per-point normalization. Use when data has non-uniform density.
- **GMST**: Graph-based approach. Use geodesic mode for curved manifolds where Euclidean distances are misleading.
