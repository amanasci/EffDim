# Geometric Analysis

Geometric estimators calculate the "Intrinsic Dimension" (ID) based on distances between points, rather than variance of global projections. This is crucial for manifolds that are non-linear (e.g., a Swiss Roll).

## The Swiss Roll Problem

A "Swiss Roll" is a 2D plane rolled up in 3D.

* **PCA** will see it as 3D (because variance exists in x, y, z).
* **Geometric ID** should see it as 2D (locally, it's a plane).

```python
import numpy as np
import effdim

# Synthetic Swiss-roll-like manifold (2D intrinsic, 3D ambient)
# For bit-identical pytest data, see tests/fixtures/swiss_roll_n1000_noise001_rs42.f64bin
rng = np.random.default_rng(42)
t = 1.5 * np.pi * (1 + 2 * rng.random(2000))
height = 21 * rng.random(2000)
X = np.column_stack([t * np.cos(t), height, t * np.sin(t)])
X = X + 0.01 * rng.standard_normal(X.shape)

# Compute dimensionalities
results = effdim.compute_dim(X)

# PCA
pca_dim = results['pca_explained_variance_95']
print(f"Global PCA Dimension: {pca_dim}")
# Likely 3, because the roll occupies 3D volume globally.

# kNN Intrinsic Dimension (MLE)
knn_dim = results['mle_dimensionality']
print(f"kNN Intrinsic Dimension: {knn_dim:.2f}")
# Should be close to 2.0

# Two-NN
twonn_dim = results['two_nn_dimensionality']
print(f"Two-NN Intrinsic Dimension: {twonn_dim:.2f}")
# Should be close to 2.0
```

## When to use Geometric Estimators?

1. **Non-linear manifolds**: Image datasets (digits, faces) often lie on low-dimensional non-linear manifolds.
2. **Manifold Learning**: Checking if your autoencoder latent space has matched the intrinsic dimension of the data.
3. **Local Analysis**: Using pure geometry approaches can capture local variability better.

## Limitations

* **Computational Cost**: Exact nearest-neighbor search in the Rust core scales with sample size; large $N$ is slower than approximate indexes.
* **Curse of Dimensionality**: In extremely high dimensions, distance concentration can make geometric estimation unstable.
