"""
notebooks/pu_manifold/tests/test_confidence_band.py -- D-02 known-answer tests for
`notebooks/pu_manifold/confidence_band.py`, the per-cloud, per-metric bootstrap
significance band and the Betti-vector reading that consumes it.

Phase 02.7 manifold-template-inference-front-end-inserted. Pins the band against answers
known independently of the code under test: a circle carries exactly one significant H1
loop, a disc carries none. The disc case is the more important of the two -- D-02's whole
rationale is that a largest-gap cut is undefined precisely when the true Betti number is 0,
and manufactures a cycle there. That is the expected PU case, so a band that cannot return
`beta_1 = 0` is not fit for this phase.

Not collected by the core `effdim` test suite (`pyproject.toml`'s `testpaths = ["tests"]`
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_confidence_band.py -q

Every test here pins a function against an input whose answer is known independently --
same discipline as `test_persistence_probe.py` and `test_geodesic_graph.py`. `B` is kept
in the single digits and `maxdim=1` (never `H_2`) in every test that does not itself need
`beta_2`, so the whole file stays well inside 02.7-VALIDATION.md's 90-second feedback
latency -- the measured `H_2` `ripser` cost wall makes an `H_2` call unsuitable for a unit
test (02.7-RESEARCH.md Pitfall 1; 02.7-01-SUMMARY.md's measured ~7.5s/call at n=300).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pu_manifold import confidence_band as cb
from pu_manifold import geodesic_graph as gg
from pu_manifold import persistence_probe as pp


def _circle_fixture(n: int = 150, seed: int = 20270802) -> np.ndarray:
    """A seeded uniform sample on the unit circle, `(n, 2)`. Known answer: `beta_1 = 1`,
    one loop -- same convention as `test_persistence_probe.py`'s own `_circle_fixture`."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _disc_fixture(n: int = 150, seed: int = 20270803) -> np.ndarray:
    """A seeded uniform sample on the unit disc, `(n, 2)`. Known answer: contractible,
    `beta_1 = 0` -- every H1 bar it carries is sampling noise, same convention as
    `test_persistence_probe.py`'s own `_disc_fixture`."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def _grid_blob_fixture(
    n_side: int = 9, seed: int = 0, jitter: float = 0.05, offset: tuple = (0.0, 0.0)
) -> np.ndarray:
    """A near-uniformly-dense jittered grid, `(n_side**2, 2)`, offset by `offset`. Known
    answer: a single connected component with no internal outlier gaps -- unlike a
    Gaussian blob, whose tail naturally produces a few moderately-isolated points whose
    H0 bar life can clear a bootstrap band computed from a handful of replicates. This
    fixture is deliberately near-uniform so the base-component test isolates `beta_0`'s
    `+1` convention from that unrelated tail-density effect."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, n_side), np.linspace(0.0, 1.0, n_side))
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    points = points + rng.uniform(-jitter, jitter, size=points.shape)
    return points + np.asarray(offset)


def _two_blob_fixture(n_side: int = 6, seed: int = 0, jitter: float = 0.05, sep: float = 20.0) -> np.ndarray:
    """Two `_grid_blob_fixture` blobs separated by `sep`, well beyond either blob's own
    diameter. Known answer: two connected components, so H0 carries exactly one
    significantly long-lived bar (the merge distance) on top of the always-present base
    component -- `beta_0 = 2`."""
    blob_1 = _grid_blob_fixture(n_side=n_side, seed=seed, jitter=jitter, offset=(0.0, 0.0))
    blob_2 = _grid_blob_fixture(n_side=n_side, seed=seed + 1, jitter=jitter, offset=(sep, 0.0))
    return np.vstack([blob_1, blob_2])


def test_band_is_deterministic_under_seed():
    circle = _circle_fixture(n=100)
    D, _ = pp.cloud_distance_matrix(circle, prescale=False)

    a = cb.bootstrap_band(D, 1, B=3, alpha=0.05, seed=5)
    b = cb.bootstrap_band(D, 1, B=3, alpha=0.05, seed=5)

    assert a["c_alpha"] == b["c_alpha"]
    assert abs(a["band"] - 2.0 * a["c_alpha"]) < 1e-15


def test_circle_yields_one_significant_h1():
    circle = _circle_fixture()
    D, _ = pp.cloud_distance_matrix(circle, prescale=False)
    dgm_h1 = pp.persistence_diagram(D, maxdim=1)[1]

    band = cb.bootstrap_band(D, 1, B=5, alpha=0.05, seed=42)
    sig = cb.significant_bars(dgm_h1, band["band"])

    assert int(np.sum(sig)) == 1


def test_disc_yields_no_significant_h1():
    """The more important of the two known-answer H1 tests. D-02's rationale is that a
    largest-gap cut is undefined precisely when the true Betti number is 0 -- there is no
    gap to find, so a gap rule manufactures a cycle. That is the expected PU case, and a
    band that cannot return 0 here is not fit for this phase."""
    disc = _disc_fixture()
    D, _ = pp.cloud_distance_matrix(disc, prescale=False)
    dgm_h1 = pp.persistence_diagram(D, maxdim=1)[1]

    band = cb.bootstrap_band(D, 1, B=5, alpha=0.05, seed=42)
    sig = cb.significant_bars(dgm_h1, band["band"])

    assert int(np.sum(sig)) == 0


def test_euclidean_and_geodesic_bands_differ():
    """Pins D-02's per-metric requirement as a measured property, not an intention: the
    same cloud's Euclidean and graph-geodesic distance matrices yield different `c_alpha`
    values under the identical bootstrap procedure."""
    circle = _circle_fixture()
    D_euc, _ = pp.cloud_distance_matrix(circle, prescale=False)
    D_geo, readout = gg.geodesic_distance_matrix(circle, k=8)
    assert readout["n_components"] == 1  # the geodesic graph must be connected for this test

    band_euclidean = cb.bootstrap_band(D_euc, 1, B=5, alpha=0.05, seed=7)
    band_geodesic = cb.bootstrap_band(D_geo, 1, B=5, alpha=0.05, seed=7)

    assert band_euclidean["c_alpha"] != band_geodesic["c_alpha"]


def test_beta_zero_counts_the_base_component():
    """Pins the `beta_0 = (significant H0 bars) + 1` convention that `finite_pairs`'
    infinite-bar filtering makes necessary: a single connected blob reads `beta_0 == 1`
    (no significant merges, just the base component); two well-separated blobs read
    `beta_0 == 2` (one significant merge bar, plus the base component)."""
    blob = _grid_blob_fixture()
    D_blob, _ = pp.cloud_distance_matrix(blob, prescale=False)
    dgms_blob = pp.persistence_diagram(D_blob, maxdim=2)
    bands_blob = cb.bands_for_diagram(D_blob, maxdim=2, B=5, alpha=0.05, seed=11)
    betti_blob = cb.betti_vector(dgms_blob, bands_blob)
    assert betti_blob[0] == 1

    two_blob = _two_blob_fixture()
    D_two_blob, _ = pp.cloud_distance_matrix(two_blob, prescale=False)
    dgms_two_blob = pp.persistence_diagram(D_two_blob, maxdim=2)
    bands_two_blob = cb.bands_for_diagram(D_two_blob, maxdim=2, B=5, alpha=0.05, seed=11)
    betti_two_blob = cb.betti_vector(dgms_two_blob, bands_two_blob)
    assert betti_two_blob[0] == 2


def test_band_spread_reports_every_seed():
    circle = _circle_fixture(n=100)
    D, _ = pp.cloud_distance_matrix(circle, prescale=False)
    seeds = [1, 2, 3]

    spread = cb.band_spread(D, 1, B=3, alpha=0.05, seeds=seeds)

    assert len(spread["c_alpha_values"]) == len(seeds)
    assert spread["seeds"] == seeds
    assert spread["range"] == spread["max"] - spread["min"]
    # no averaged value stands in for the individual seeds' c_alpha values
    assert "mean" not in spread
    assert "average" not in spread
