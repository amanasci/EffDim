# EffDim

**EffDim** is a unified, research-oriented Python library designed to compute "effective dimensionality" (ED) across diverse data modalities.

## Installation

```bash
pip install effdim
```

## Usage

```python
import numpy as np
import effdim

data = np.random.randn(100, 50)
ed = effdim.compute(data, method='pca', threshold=0.95)
print(f"Effective Dimension: {ed}")
```

## Features

- **Modality Agnostic**: Works with raw data, covariance matrices, and pre-computed spectra.
- **Unified Interface**: Simple `compute` and `analyze` functions.
- **Extensive Estimators**: PCA, Participation Ratio, Shannon Entropy, and more.
