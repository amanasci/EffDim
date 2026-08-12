"""
notebooks/pu_manifold/confidence_band.py -- D-02: per-cloud, per-metric bootstrap
significance band for a persistence diagram bar.

Phase 02.7 manifold-template-inference-front-end-inserted. A bar is significant iff it
clears a bootstrap-derived confidence band (Fasy, Lecci, Rinaldo, Chazal, Singh,
Wasserman, "Confidence sets for persistence diagrams," Ann. Statist. 42(6):2301-2339
(2014), arXiv:1303.7117 -- algorithm and defaults cross-checked against the R
`TDA::bootstrapDiagram` package's own documented defaults, per 02.7-RESEARCH.md Q2). Not
a fixed fraction of diagram diameter, not a largest-gap cut.

The band is derived PER cloud, PER metric, PER degree: Euclidean and graph-geodesic
filtrations have non-comparable scales, so one hand-set constant would need a different
value for each -- per-metric tuning through the back door, which D-02 exists to rule out.

**`B`, `alpha` and `seed` are REQUIRED arguments with NO default value.** This mirrors
`persistence_probe.cloud_distance_matrix`'s required-`prescale` precedent, and here it is
what makes SC-1's "ratified blind" property structural rather than disciplinary: plan
`02.7-08` ratifies the real values, and no call site in this module can silently inherit
an unratified convention because none is offered.

The bootstrap resample operates on `D`'s own row/column indices (`D[np.ix_(idx, idx)]`),
so it is metric-agnostic by construction and needs no geodesic adapter -- the resample
code is identical whether `D` came from `persistence_probe.cloud_distance_matrix`
(Euclidean) or `geodesic_graph.geodesic_distance_matrix` (geodesic).

**This module never passes a threshold to `ripser`.** 02.7-RESEARCH.md Pitfall 2 verified
that a threshold-truncated diagram can carry multiple infinite-death bars per degree, and
`persistence_probe.finite_pairs` -- which every diagram this module touches has already
passed through -- would silently discard genuine long-lived features it should keep.
Every call into `persistence_probe.persistence_diagram` below relies on that function's
own unbounded default and must stay that way.

Arrays in, dicts and arrays out -- no file I/O, no cache handling.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import persim
except ImportError as _e:  # pragma: no cover -- environment, not logic
    raise ImportError(
        "confidence_band requires persim. It is NOT declared in pyproject.toml -- "
        "CLAUDE.md bars editing it for the whole v1.1 milestone -- so it must already be "
        "installed into this repository's venv by hand:\n\n"
        "    .venv/bin/pip install persim\n\n"
        "This is a known reproducibility gap, inherited from persistence_probe.py's own "
        "module docstring, not resolved here."
    ) from _e

from . import persistence_probe


def bootstrap_band(D: np.ndarray, degree: int, B: int, alpha: float, seed: int) -> Dict[str, Any]:
    """Fasy et al.'s bottleneck bootstrap band for one (cloud, metric, degree). `D` is a
    square symmetric distance matrix -- Euclidean or geodesic, identical code either way.

    `B`, `alpha`, `seed` carry NO default (see module docstring) -- ratifiable constants,
    supplied by the caller every time.

    Returns `{"c_alpha", "band", "boot_distances", "B", "alpha", "seed", "degree"}`.
    `band = 2 * c_alpha`; a bar with life (death - birth) greater than `band` is
    significant at this confidence.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"bootstrap_band: D must be a square 2-d distance matrix, got shape {D.shape!r}"
        )
    n = D.shape[0]

    rng = np.random.default_rng(seed)
    original_dgm = persistence_probe.persistence_diagram(D, maxdim=degree)[degree]

    boot_distances: List[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)  # WITH replacement, same size n
        D_boot = D[np.ix_(idx, idx)]  # pure index resample -- metric-agnostic
        boot_dgm = persistence_probe.persistence_diagram(D_boot, maxdim=degree)[degree]
        boot_distances.append(float(persim.bottleneck(original_dgm, boot_dgm)))

    c_alpha = float(np.quantile(boot_distances, 1.0 - alpha))
    band = 2.0 * c_alpha

    return {
        "c_alpha": c_alpha,
        "band": band,
        "boot_distances": boot_distances,
        "B": B,
        "alpha": alpha,
        "seed": seed,
        "degree": degree,
    }


def significant_bars(dgm: np.ndarray, band: float) -> np.ndarray:
    """The boolean mask of `dgm`'s bars whose life (death - birth) clears `band`."""
    dgm = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    if dgm.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    return (dgm[:, 1] - dgm[:, 0]) > band


def betti_vector(dgms: Sequence[np.ndarray], bands: Sequence[float]) -> Tuple[int, int, int]:
    """The 3-tuple `(beta_0, beta_1, beta_2)` from `dgms` (H0, H1, H2, each already
    `finite_pairs`-filtered) and `bands` (one band per degree, from `bootstrap_band`).

    `beta_0` is the count of significant H0 bars PLUS 1 for the always-present base
    component -- `finite_pairs` has already dropped H0's one true infinite bar, so this
    term restores it. `beta_1`/`beta_2` are the plain significant counts.
    """
    if len(dgms) != 3 or len(bands) != 3:
        raise ValueError(
            f"betti_vector: expected exactly 3 (H0, H1, H2) diagrams and bands, got "
            f"{len(dgms)} diagrams and {len(bands)} bands"
        )
    beta_0 = int(np.sum(significant_bars(dgms[0], bands[0]))) + 1
    beta_1 = int(np.sum(significant_bars(dgms[1], bands[1])))
    beta_2 = int(np.sum(significant_bars(dgms[2], bands[2])))
    return (beta_0, beta_1, beta_2)
