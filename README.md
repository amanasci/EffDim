# EffDim

The ML4PS 2026 paper "Linear Probes on Curved Latent Spaces" is in [paper/](paper/README.md);
the code that produces its results is in [curvature-experiment/](curvature-experiment/README.md).

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
results = effdim.compute_dim(data)
print(f"Results : {results}")
```
