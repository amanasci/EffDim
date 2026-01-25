# EffDim

**EffDim** is a unified, research-oriented Python library designed to compute "effective dimensionality" (ED) across diverse data modalities.

**NEW:** EffDim now includes a Rust implementation of geometry functions for 10-50x performance improvements on large datasets! See [RUST_BUILD.md](RUST_BUILD.md) for details.

## Installation

```bash
pip install effdim
```

**Prebuilt wheels with Rust acceleration** are available for Linux, macOS, and Windows (Python 3.8-3.12). The Rust implementation provides 10-50x speedup for geometry calculations on large datasets. See [RUST_BUILD.md](RUST_BUILD.md) for more details.

## Usage

```python
import numpy as np
import effdim

data = np.random.randn(100, 50)
results = effdim.compute_dim(data)
print(f"Effective Dimension (PCA): {results['pca_explained_variance_95']}")
```

## Features

- **Modality Agnostic**: Works with raw data, covariance matrices, and pre-computed spectra.
- **Unified Interface**: Simple `compute_dim` function.
- **Extensive Estimators**:
    - **Spectral Methods** (operate on eigenvalues/singular values):
        - PCA Explained Variance (`pca`)
        - Participation Ratio (`participation_ratio`, `pr`)
        - Shannon Effective Dimension (`shannon`, `entropy`)
        - Rényi Effective Dimension (`renyi`)
        - Effective Rank (`effective_rank`, `erank`)
        - Geometric Mean Dimension (`geometric_mean`)
        - Stable Rank (`stable_rank`)
        - Numerical Rank (`numerical_rank`)
        - Cumulative Eigenvalue Ratio (`cumulative_eigenvalue_ratio`)
    - **Geometric Methods** (operate on distance/topology):
        - kNN Intrinsic Dimension (`knn`)
        - Two-NN (`twonn`)
        - DANCo (`danco`)
        - MiND (`mind_mli`, `mind_mlk`)
        - ESS (`ess`)
        - TLE (`tle`)
        - GMST (`gmst`)
