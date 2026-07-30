"""
Notebook-scoped helper package for the v1.1 PU Manifold Curvature milestone.

This package is never installed and is never imported from ``src/effdim/`` -- it is a
plain relative import used only by ``notebooks/01_manifold_and_gate.ipynb`` (and the
notebooks Phase 2-4 append to or add alongside it). Two modules are implemented in
Phase 1; two are stubbed per D-02 so the eventual four-module package shape is visible
from the start:

- ``cache``      (Phase 1, implemented) -- config-hash-keyed npz/joblib/json cache helpers
  with sidecar-manifest verification and a ``CACHE_DIR`` containment guard.
- ``subsample``  (Phase 1, implemented) -- seeded, row-alignment-safe subsampling of the
  paired HSC/Legacy Survey embedding columns.
- ``curvature``  (Phase 3 stub) -- ``torch.func`` fundamental-form / mean-curvature helpers.
- ``mknn``       (Phase 4 stub) -- mutual k-NN alignment, permutation null, bootstrap CI.

``curvature`` and ``mknn`` are intentionally NOT imported here at module level: Phase 3's
module will need torch, which must not become an import-time requirement for notebook 01
(or for this package's own test suite).

D-01's three-notebook filenames (``01_manifold_and_gate.ipynb`` and the two Phase 2-4
append/add targets) and the ``§N.M`` section-numbering convention they use are **costly**
to change once written: every cache path, cross-notebook check, and doc reference in this
milestone is written against them.
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
