# Welcome to EffDim

**EffDim** is a unified, research-oriented Python library designed to compute "effective dimensionality" (ED) across diverse data modalities.

It aims to standardize the fragmented landscape of ED metrics found in statistics, physics, information theory, and machine learning into a single, cohesive interface.

## Key Features

*   **Modality Agnostic**: Works robustly across different datasets.
*   **Unified Interface**: Simple `compute_dim` function to get all estimates.
*   **Extensive Estimators**: PCA, Participation Ratio, Shannon Entropy, and more.
*   **Research Ready**: Accurate implementations of metrics from literature.

## Installation

Install via pip:

```bash
pip install effdim
```

*(EffDim relies on Faiss for fast kNN approximation under the hood).*

## Quick Start

```python
import numpy as np
import effdim

# Generate random high-dimensional data
data = np.random.randn(100, 50)

# Compute all Effective Dimensions at once
results = effdim.compute_dim(data)

# Extract specific metrics
ed = results['pca_explained_variance_95']
print(f"Effective Dimension (PCA): {ed}")

pr = results['participation_ratio']
print(f"Participation Ratio: {pr}")
```

Explore the [User Guide](tutorials/getting_started.md) for more examples.
