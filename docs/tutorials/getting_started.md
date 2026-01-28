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

EffDim revolves around two main functions:

* `effdim.compute(data, method=...)`: Calculates a single dimension metric.
* `effdim.analyze(data, methods=[...])`: Calculates multiple metrics at once.

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

# Participation Ratio
pr = effdim.compute(noise, method='participation_ratio')
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

pr = effdim.compute(structured_data, method='participation_ratio')
print(f"PR of Structured Data: {pr:.2f}")
# Expected: close to 5
```

## Available Methods

You can check the available methods in the [Theory](../theory.md) section.

**Spectral Methods:**

* `'pca'`: PCA Explained Variance
* `'participation_ratio'` (or `'pr'`)
* `'shannon'` (or `'entropy'`)
* `'effective_rank'` (or `'erank'`): Alias for Shannon Effective Dimension (Trace Norm).
* `'renyi'`
* `'stable_rank'`: Ratio of sum/max eigenvalues.
* `'numerical_rank'`: Count of singular values > epsilon.

**Geometric Methods:**

* `'knn'`: k-Nearest Neighbors
* `'twonn'`: Two-Nearest Neighbors
* `'danco'`: Angle and Norm Concentration
* `'mind_mlk'`: MiND (Maximum Likelihood)
* `'ess'`: Expected Simplex Skewness

## analyzing Multiple Metrics

Use `effdim.analyze` to get a report.

```python
report = effdim.analyze(structured_data, methods=['pr', 'pca', 'shannon', 'danco'])
print(report)
# {'participation_ratio': ..., 'pca': ..., 'shannon': ..., 'danco': ...}
```
