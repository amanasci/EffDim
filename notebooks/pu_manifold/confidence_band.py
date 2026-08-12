"""
notebooks/pu_manifold/confidence_band.py -- D-02: per-cloud, per-metric bootstrap
significance band for a persistence diagram bar, and the Betti-vector reading that
consumes it.

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

**Assumption A-03.** The band is this phase's practical stand-in for a sampling-density
guarantee, not a proof that the recovered homology is the manifold's -- homology recovery
from finite samples needs sampling-density and regularity assumptions an arbitrary cloud
does not guarantee, and a bootstrap band on a finite draw cannot supply them. Every reader
of `betti_vector`'s output should read it as "significant under this band," not as
"provably the manifold's true Betti number."

**D-02's band loop and D-08's independent-draw loop are two different procedures whose
`ripser` costs multiply, not substitute.** D-08 draws several independent point clouds
from the immersion generator and reports how the *decision* varies across draws; D-02's
band is computed *within* one of those draws by resampling that cloud's own points `B`
times. The full cost shape is `(draws) x (1 main call + B bootstrap calls) x (2 metrics) x
(grid cells)` -- no caller should size compute by counting either loop alone.

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


def bands_for_diagram(
    D: np.ndarray, maxdim: int, B: int, alpha: float, seed: int
) -> Dict[int, Dict[str, Any]]:
    """One `bootstrap_band` result per degree `0..maxdim`, keyed by degree, so a caller
    cannot accidentally apply an `H_1` band to an `H_2` diagram (or any other degree
    mismatch). `B`, `alpha`, `seed` carry NO default (see module docstring) and are the
    same triple passed to every degree's `bootstrap_band` call -- the bands differ by
    degree, not by these settings.

    Returns `{degree: bootstrap_band(D, degree, B, alpha, seed) for degree in
    range(maxdim + 1)}`.
    """
    if not (isinstance(maxdim, (int, np.integer)) and maxdim >= 0):
        raise ValueError(f"bands_for_diagram: maxdim must be a non-negative int, got {maxdim!r}")
    return {degree: bootstrap_band(D, degree, B, alpha, seed) for degree in range(maxdim + 1)}


def band_spread(
    D: np.ndarray, degree: int, B: int, alpha: float, seeds: Sequence[int]
) -> Dict[str, Any]:
    """Assumption A-04's instrument: PH seed/draw variance is still unmeasured in this
    project, and this function is how a caller measures it rather than absorbs it. Runs
    `bootstrap_band(D, degree, B, alpha, seed)` once per entry in `seeds` and reports the
    per-seed `c_alpha` values themselves -- never an average or any other single number
    standing in for them.

    Returns `{"c_alpha_values", "seeds", "min", "max", "range"}`. `c_alpha_values[i]`
    corresponds to `seeds[i]`. If the between-seed spread swamps the between-template
    differences this instrument exists to detect, that is a finding about the criterion,
    to be reported rather than silently averaged away.
    """
    if len(seeds) == 0:
        raise ValueError("band_spread: seeds must be non-empty")

    c_alpha_values: List[float] = []
    for seed in seeds:
        result = bootstrap_band(D, degree, B, alpha, seed)
        c_alpha_values.append(result["c_alpha"])

    c_alpha_array = np.asarray(c_alpha_values, dtype=np.float64)
    return {
        "c_alpha_values": c_alpha_values,
        "seeds": list(seeds),
        "min": float(c_alpha_array.min()),
        "max": float(c_alpha_array.max()),
        "range": float(c_alpha_array.max() - c_alpha_array.min()),
    }


def significant_bars(dgm: np.ndarray, band: float) -> np.ndarray:
    """The boolean mask of `dgm`'s bars whose life (death - birth) clears `band`."""
    dgm = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    if dgm.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    return (dgm[:, 1] - dgm[:, 0]) > band


def betti_vector(dgms: Sequence[np.ndarray], bands: Any) -> Tuple[int, int, int]:
    """The 3-tuple `(beta_0, beta_1, beta_2)` from `dgms` (H0, H1, H2, each already
    `finite_pairs`-filtered) and `bands` -- one band per degree, indexed or keyed by
    degree `0..2`. `bands` accepts either a plain sequence of numeric bands (the tracer's
    own `[band_h0, band_h1, band_h2]` shape) or a mapping keyed by degree, such as
    `bands_for_diagram`'s own per-degree `bootstrap_band` result dicts -- each entry's
    `"band"` key is used when the entry is a dict, so both shapes serve unchanged.

    `beta_0` is the count of significant H0 bars PLUS 1 for the always-present base
    component -- `finite_pairs` has already dropped H0's one true infinite bar, so this
    term restores it. `beta_1`/`beta_2` are the plain significant counts under their own
    degree's band.

    Raises `ValueError` naming the missing degree if `bands` supplies no entry for a
    degree present in `dgms`.
    """
    if len(dgms) != 3:
        raise ValueError(f"betti_vector: expected exactly 3 (H0, H1, H2) diagrams, got {len(dgms)}")

    resolved_bands: List[float] = []
    for degree in range(3):
        try:
            entry = bands[degree]
        except (KeyError, IndexError):
            raise ValueError(f"betti_vector: bands is missing a band for degree {degree}") from None
        resolved_bands.append(float(entry["band"]) if isinstance(entry, dict) else float(entry))

    beta_0 = int(np.sum(significant_bars(dgms[0], resolved_bands[0]))) + 1
    beta_1 = int(np.sum(significant_bars(dgms[1], resolved_bands[1])))
    beta_2 = int(np.sum(significant_bars(dgms[2], resolved_bands[2])))
    return (beta_0, beta_1, beta_2)
