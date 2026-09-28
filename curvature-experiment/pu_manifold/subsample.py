"""Seeded, row-alignment-safe subsampling of ``UniverseTBD/pu-embeddings``.

No object_id exists in this dataset — row order is the only join between the paired
columns, so both are read off ONE sorted seeded index array in a single indexing pass;
two independent selections would silently break alignment. Only ``draw_row_indices`` and
``l2_normalize`` remain in this file; the PU loader (``load_subsample``) and its runtime
alignment proof (``assert_alignment``: structural check + permuted-null z-score) are archived
verbatim in ``archive/pu_manifold_trimmed/subsample.py``.
"""

from typing import Tuple

import numpy as np


# Row count of the legacysurvey_dinov3_vitb16 config, per PROJECT.md. The archived load_subsample
# asserts the loaded config reports exactly this many rows (T-01-02 mitigation: catches a
# silently changed upstream file).
EXPECTED_N_TOTAL = 101_725

# A dense geodesic distance matrix over the full EXPECTED_N_TOTAL rows would be roughly
# 83 GB (101_725**2 float64 entries). This cap keeps every Isomap fit in this milestone
# tractable on a single machine (T-01-05 mitigation).
MAX_N_ROWS = 20_000


def draw_row_indices(n_total: int, n_rows: int, seed: int) -> np.ndarray:
    """Deterministic sorted duplicate-free sample (DATA-03). Both paired columns must
    be read off this single array in one pass. Raises ValueError on degenerate sizes."""
    if n_rows < 2:
        raise ValueError(f"n_rows must be at least 2, got {n_rows}.")
    if n_rows > MAX_N_ROWS:
        raise ValueError(
            f"n_rows={n_rows} exceeds MAX_N_ROWS={MAX_N_ROWS}. A dense geodesic distance "
            f"matrix over the full {EXPECTED_N_TOTAL} rows would be roughly 83 GB "
            f"({EXPECTED_N_TOTAL}**2 float64 entries); this cap keeps the Isomap fit "
            f"tractable on a single machine."
        )
    if n_total < n_rows:
        raise ValueError(
            f"n_total={n_total} is smaller than n_rows={n_rows}; cannot draw {n_rows} "
            f"rows without replacement from only {n_total}."
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, n_rows, replace=False))


def l2_normalize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """L2-normalize each row of a ``(n_rows, n_features)`` array, returning
    ``(x / norms[:, None], norms)``. Raises ValueError on a zero-norm row."""
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError(
            "l2_normalize received at least one zero-norm row; cannot normalize a "
            "zero vector to the unit sphere."
        )
    return x / norms[:, None], norms
