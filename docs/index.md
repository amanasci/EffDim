# Welcome to EffDim

**EffDim** is a unified, research-oriented Python library designed to compute "effective dimensionality" (ED) across diverse data modalities.

It aims to standardize the fragmented landscape of ED metrics found in statistics, physics, information theory, and machine learning into a single, cohesive interface.

!!! success "Performance Enhancement"
    EffDim now includes a **Rust-accelerated implementation** of geometry functions, providing **10-50x speedup** for large datasets! Prebuilt wheels are available for all major platforms.

## Key Features

*   **Modality Agnostic**: Works with raw data, covariance matrices, and pre-computed spectra.
*   **Unified Interface**: Simple `compute` and `analyze` functions.
*   **Extensive Estimators**: PCA, Participation Ratio, Shannon Entropy, and more.
*   **Research Ready**: Accurate implementations of metrics from literature.
*   **High Performance**: Rust-accelerated geometry calculations for large-scale datasets.

## Installation

Install via pip (includes prebuilt Rust extensions):

```bash
pip install effdim
```

Prebuilt wheels with Rust acceleration are available for:

- **Linux** (manylinux, x86_64 & aarch64)
- **macOS** (x86_64 & Apple Silicon ARM64)
- **Windows** (x86_64)
- **Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12

## Quick Start

```python
import numpy as np
import effdim

# Generate random high-dimensional data
data = np.random.randn(100, 50)

# Compute Effective Dimension
results = effdim.compute_dim(data)

# Access results
print(f"Effective Dimension (PCA 95%): {results['pca_explained_variance_95']}")
print(f"Participation Ratio: {results['participation_ratio']}")
```

Explore the [User Guide](tutorials/getting_started.md) for more examples.
