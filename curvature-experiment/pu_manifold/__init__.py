"""
Notebook-scoped helper package for the v1.1 PU Manifold Curvature milestone. Never
installed, never imported from ``src/effdim/`` -- a plain relative import.

- ``cache``      -- config-hash-keyed npz cache helper with sidecar-manifest verification
  and a ``CACHE_DIR`` containment guard.
- ``subsample``  -- the seeded, sorted, duplicate-free row draw (``draw_row_indices``) and
  row L2 normalization (``l2_normalize``).

The torch-dependent modules (``cae``, ``decoder_curvature``, ``chart_curvature``, ...) are NOT
imported here at module level, so importing this package does not require torch.
"""

from .cache import (
    CACHE_DIR,
    KEY_LEN,
    config_key,
    cache_path,
    npz_cache,
)
from .subsample import (
    EXPECTED_N_TOTAL,
    MAX_N_ROWS,
    draw_row_indices,
    l2_normalize,
)

__all__ = [
    "CACHE_DIR",
    "KEY_LEN",
    "config_key",
    "cache_path",
    "npz_cache",
    "EXPECTED_N_TOTAL",
    "MAX_N_ROWS",
    "draw_row_indices",
    "l2_normalize",
]
