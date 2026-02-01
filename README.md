# EffDim

**EffDim** is a unified, research-oriented Python library designed to compute
"effective dimensionality" (ED) across diverse data modalities.

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
