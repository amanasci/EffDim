# Geometric Analysis

Geometric estimators calculate the "Intrinsic Dimension" (ID) based on distances between points, rather than variance of global projections. This is crucial for manifolds that are non-linear (e.g., a Swiss Roll).

## The Swiss Roll Problem

A "Swiss Roll" is a 2D plane rolled up in 3D.

* **PCA** will see it as 3D (because variance exists in x, y, z).
* **Geometric ID** should see it as 2D (locally, it's a plane).

```python
import numpy as np
import effdim
from sklearn.datasets import make_swiss_roll

# Generate Swiss Roll
X, _ = make_swiss_roll(n_samples=2000, noise=0.01)

results = effdim.compute_dim(X)

# PCA
print(f"Global PCA Dimension: {results['pca_explained_variance_95']}")
# Likely 3, because the roll occupies 3D volume globally.

# MLE Intrinsic Dimension (kNN-based)
print(f"MLE Intrinsic Dimension: {results['mle_dimensionality']:.2f}")
# Should be close to 2.0

# Two-NN
print(f"Two-NN Intrinsic Dimension: {results['two_nn_dimensionality']:.2f}")
# Should be close to 2.0
```

## When to use Geometric Estimators?

1. **Non-linear manifolds**: Image datasets (digits, faces) often lie on low-dimensional non-linear manifolds.
2. **Manifold Learning**: Checking if your autoencoder latent space has matched the intrinsic dimension of the data.
3. **Local Analysis**: kNN can be computed per-point (though `effdim` currently returns the average).

## Limitations

* **Computational Cost**: Requires computing nearest neighbors, which can be slow for large $N$.

    !!! tip "Performance"
        `effdim` uses a **Rust-accelerated implementation** with parallel nearest neighbor search for 10-50x speedup on large datasets. See [Performance](../performance.md) for benchmarks.

* **Curse of Dimensionality**: In extremely high dimensions, distance concentration can make geometric estimation unstable.
