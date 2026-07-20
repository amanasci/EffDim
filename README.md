# EffDim

**EffDim** is a unified, research-oriented Python library designed to compute "effective dimensionality" (ED) across diverse data modalities.


## Installation

```bash
pip install effdim
```

End users install from wheels (no Rust toolchain). Contributors building from source need Rust stable + maturin — see **Contributor develop** in [`docs/deployment.md`](docs/deployment.md) (`maturin develop --release`, then `pytest`). Multi-OS wheel publish CI is Phase 5.

## Usage

```python
import numpy as np
import effdim

data = np.random.randn(100, 50)
results = effdim.compute_dim(data)
print(f"Results : {results}")
```