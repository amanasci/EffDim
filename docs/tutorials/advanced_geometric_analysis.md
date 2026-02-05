# Advanced Geometric Analysis

!!! warning "Not Implemented"
    The advanced geometric estimators described in this tutorial (DANCo, MiND, ESS, TLE, GMST) are currently **planned features** and are **not yet implemented** in the `effdim` package. The code examples below are for illustration of the future API.

EffDim provides a suite of advanced geometric estimators for Intrinsic Dimension (ID) estimation. These methods go beyond standard kNN/TwoNN approaches by exploiting additional statistical properties of high-dimensional space, such as the distribution of angles (DANCo) or simplex volumes (ESS).

## Methods Overview

| Method | Full Name | Key Idea |
| --- | --- | --- |
| **DANCo** | Dimensionality from Angle and Norm Concentration | Uses both distances and angles to neighbors. |
| **MiND** | Minimum Neighbor Distance | MLE based on min distances (MLi) or joint k-NN (MLk). |
| **ESS** | Expected Simplex Skewness | Uses the volume of the simplex formed by neighbors. |
| **TLE** | Tight Localities Estimator | Maximizes likelihood on scale-normalized distances. |
| **GMST** | Geodesic Minimum Spanning Tree | Uses the scaling of MST length with sample size. |

## Setup

We will generate two datasets:

1. **High-Dimensional Noise**: 1000 points in 20D (ID should be ~20).
2. **Low-Dimensional Manifold**: A 5D hypercube embedded in 20D space (ID should be ~5).

```python
import numpy as np
import effdim

np.random.seed(42)

# 1. High-Dimensional Noise (ID = 20)
N = 1000
D = 20
noise_data = np.random.randn(N, D)

# 2. 5D Manifold embedded in 20D
# Generate 5D data
latent = np.random.rand(N, 5)
# Embed in 20D with a random rotation
projection = np.linalg.qr(np.random.randn(20, 5))[0]
manifold_data = latent @ projection.T
```

## Running Estimators

### DANCo

DANCo is often more accurate than kNN for high dimensions but is computationally more expensive due to angle computations.

```python
# Noise Data (Expect ~20)
# results = effdim.compute_dim(noise_data)
# d_danco = results['danco']
# print(f"DANCo (Noise): {d_danco:.2f}")

# Manifold Data (Expect ~5)
# results_m = effdim.compute_dim(manifold_data)
# d_danco_m = results_m['danco']
# print(f"DANCo (Manifold): {d_danco_m:.2f}")
```

### MiND (MLi and MLk)

MiND estimators are fast and robust Maximum Likelihood approaches.

```python
# MiND-MLi (Uses 1st NN)
# d_mli = effdim.compute_dim(manifold_data)['mind_mli']
# print(f"MiND-MLi: {d_mli:.2f}")

# MiND-MLk (Uses k NNs)
# d_mlk = effdim.compute_dim(manifold_data)['mind_mlk']
# print(f"MiND-MLk: {d_mlk:.2f}")
```

### ESS (Expected Simplex Skewness)

ESS compares the volume of local simplices to theoretical expectations. It relies on precomputed constants (currently supported for d=1 to 20).

```python
# d_ess = effdim.compute_dim(manifold_data)['ess']
# print(f"ESS: {d_ess:.2f}")
```

### GMST (Geodesic MST)

GMST estimates dimension from the scaling of the Minimum Spanning Tree length. It is particularly useful for detecting the topology of the manifold.

* `mode='euclidean'` (default): Uses Euclidean distances. Good for flat manifolds.
* `mode='geodesic'`: Uses graph geodesic distances. Better for curved manifolds (like Swiss Roll).

```python
# Euclidean MST
# d_gmst = effdim.compute_dim(manifold_data)['gmst']
# print(f"GMST (Euclidean): {d_gmst:.2f}")
```

## Comparative Analysis

You can analyze multiple methods at once to check for consistency.

```python
# report = effdim.compute_dim(manifold_data)
# for method, val in report.items():
#     print(f"{method}: {val:.2f}")
```

## Limitations

* **Computational Cost**: DANCo and GMST (geodesic) can be slow for large $N$ ($>10,000$).
* **Sample Size**: Geometric estimators generally require samples growing exponentially with dimension. For ID > 20, estimates might degrade unless $N$ is very large.
* **ESS Constraints**: ESS in `effdim` currently supports dimensions up to 20 due to precomputed constants.
