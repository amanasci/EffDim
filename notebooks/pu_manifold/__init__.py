"""
Notebook-scoped helper package for the v1.1 PU Manifold Curvature milestone. Never
installed, never imported from ``src/effdim/`` -- a plain relative import.

- ``cache``      -- config-hash-keyed npz/joblib/json cache helpers with sidecar-manifest
  verification and a ``CACHE_DIR`` containment guard.
- ``subsample``  -- seeded, row-alignment-safe subsampling of the paired HSC/Legacy Survey
  embedding columns.
- ``curvature``  (Phase 3 stub) -- ``torch.func`` fundamental-form / mean-curvature helpers.
- ``mknn``       (Phase 4 stub) -- mutual k-NN alignment, permutation null, bootstrap CI.

``curvature`` and ``mknn`` are NOT imported here at module level: Phase 3's module needs
torch, which must not become an import-time requirement for this package.
"""

from .cache import (
    CACHE_DIR,
    KEY_LEN,
    config_key,
    cache_path,
    npz_cache,
    joblib_cache,
    json_cache,
)
from .subsample import (
    N_FEATURES,
    EXPECTED_N_TOTAL,
    MAX_N_ROWS,
    ALIGNMENT_MARGIN_Z,
    ALIGNMENT_N_PERMUTATIONS,
    draw_row_indices,
    l2_normalize,
    row_indices_sha256,
    assert_structural_alignment,
    alignment_smoke_test,
    assert_alignment,
    load_subsample,
)

__all__ = [
    "CACHE_DIR",
    "KEY_LEN",
    "config_key",
    "cache_path",
    "npz_cache",
    "joblib_cache",
    "json_cache",
    "N_FEATURES",
    "EXPECTED_N_TOTAL",
    "MAX_N_ROWS",
    "ALIGNMENT_MARGIN_Z",
    "ALIGNMENT_N_PERMUTATIONS",
    "draw_row_indices",
    "l2_normalize",
    "row_indices_sha256",
    "assert_structural_alignment",
    "alignment_smoke_test",
    "assert_alignment",
    "load_subsample",
]
