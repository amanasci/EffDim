# Getting Started

This guide will walk you through the basic usage of `effdim`.

## Installation

Ensure `effdim` is installed:

```bash
pip install effdim
```

## Basic Concepts

EffDim revolves around a single main function that computes various effective dimensionalities at once:

*   `effdim.compute_dim(data)`: Calculates a dictionary of dimension metrics.

Data is typically passed as a **N x D** numpy array, where $N$ is the number of samples and $D$ is the number of features.

## Example: Random Noise vs Structured Data

Let's see how effective dimension differs between random noise and structured data.

### 1. Random Noise

High-dimensional random noise should have a high effective dimension because the variance is spread out in all directions.

```python
import numpy as np
import effdim

# 1000 samples, 100 dimensions
noise = np.random.randn(1000, 100)

# Compute dimensionalities
results = effdim.compute_dim(noise)
pr = results['participation_ratio']
print(f"PR of Noise: {pr:.2f}")
# Expected: close to 100 (or slightly less due to finite sampling)
```

### 2. Structured Data (Low Rank)

If we create data that lies on a low-dimensional plane embedded in high-dimensional space, the effective dimension should be low.

```python
# Create 1000 samples with only 5 meaningful dimensions
latent = np.random.randn(1000, 5)
projection = np.random.randn(5, 100)
structured_data = latent @ projection

# Add a tiny bit of noise
structured_data += 0.01 * np.random.randn(1000, 100)

pr = effdim.compute_dim(structured_data)['participation_ratio']
print(f"PR of Structured Data: {pr:.2f}")
# Expected: close to 5
```

## Available Methods

You can check the available methods in the [Theory](../theory.md) section.

**Spectral Methods:**

*   `pca_explained_variance_95`: PCA Explained Variance (with 95% threshold)
*   `participation_ratio`: Participation Ratio
*   `shannon_entropy`: Shannon Effective Dimensionality
*   `renyi_eff_dimensionality_alpha_2` (also 3, 4, 5): Rényi Effective Dimensionality
*   `geometric_mean_eff_dimensionality`: Geometric Mean Dimension

**Geometric Methods:**

*   `mle_dimensionality`: k-Nearest Neighbors (Maximum Likelihood Estimate)
*   `two_nn_dimensionality`: Two-Nearest Neighbors

## analyzing Multiple Metrics

Use `effdim.compute_dim` to get a report with all available estimators at once.

```python
report = effdim.compute_dim(structured_data)
print(report)
# {'pca_explained_variance_95': ..., 'participation_ratio': ..., 'shannon_entropy': ..., 'mle_dimensionality': ..., ...}
```
