# Getting Started

This guide will walk you through the basic usage of `effdim`.

## Installation

Ensure `effdim` is installed:

```bash
pip install effdim
```

!!! success "Performance"
    The installed package includes prebuilt Rust extensions for high-performance geometry calculations. No additional setup needed!

## Basic Concepts

EffDim revolves around a single unified function:

*   `effdim.compute_dim(data)`: Calculates all available dimension metrics and returns a dictionary.

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

# Compute all dimensions
results = effdim.compute_dim(noise)

# Participation Ratio
print(f"PR of Noise: {results['participation_ratio']:.2f}")
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

results = effdim.compute_dim(structured_data)
print(f"PR of Structured Data: {results['participation_ratio']:.2f}")
# Expected: close to 5
```

## Available Methods

You can check the available methods in the [Theory](../theory.md) section.

**Spectral Methods:**

*   `pca_explained_variance_95`: PCA Explained Variance (95% threshold)
*   `participation_ratio`: Participation Ratio
*   `shannon_entropy`: Shannon Entropy of the spectrum
*   `renyi_eff_dimensionality_alpha_{k}`: Rényi Entropy (for k=2,3,4,5)
*   `geometric_mean_eff_dimensionality`: Geometric Mean

**Geometric Methods:**

*   `mle_dimensionality`: Maximum Likelihood Estimator (Levina-Bickel)
*   `two_nn_dimensionality`: Two-Nearest Neighbors
*   `box_counting_dimensionality`: Box Counting

!!! note "Not Implemented"
    Methods such as DANCo, MiND, ESS, TLE, GMST, Stable Rank, and Numerical Rank are not yet implemented in the current version.

## Analyzing Multiple Metrics

`effdim.compute_dim` returns a dictionary with all computed metrics, allowing you to easily compare them.

```python
results = effdim.compute_dim(structured_data)
print(results)
# {'participation_ratio': ..., 'pca_explained_variance_95': ..., ...}
```
