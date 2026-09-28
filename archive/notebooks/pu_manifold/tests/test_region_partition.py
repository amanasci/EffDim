"""
Known-answer and boundary tests for ``pu_manifold.region_partition`` (D4-09's diametrical
sign-split partition).

No HuggingFace access, no torch, no fixtures beyond synthetic point clouds generated
in-test. Not collected by the core `effdim` test suite (``pyproject.toml``'s
``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_region_partition.py -q

This is a permitted new test file: D4-18 declines `tests/test_mknn.py` specifically, for
`mknn.py`'s statistical functions; 04-VALIDATION.md's Wave 0 explicitly records a
round-trip test for the partition helper as the one new test file this phase can add
without contradicting it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from pu_manifold import region_partition as rp


# --- Test 1: known two-antipodal-cone answer -----------------------------------------------


def test_region_partition_recovers_known_two_cone_split():
    """n=600 unit vectors in R^12, normalize(s_i * w + 0.15 * noise_i) for a fixed unit
    axis w and a balanced s_i in {+1, -1}, each scaled by a random positive magnitude so
    ||H|| varies. min_norm_percentile=0.0 keeps every point. The recovered labels must
    match the construction's s_i exactly (ARI == 1.0), and the recovered top eigenvector
    must align with w up to sign (|dot(v, w)| > 0.99)."""
    rng = np.random.default_rng(0)
    n, D = 600, 12
    w = rng.normal(size=D)
    w = w / np.linalg.norm(w)
    s = np.tile(np.array([1, -1]), n // 2)
    noise = rng.normal(size=(n, D))
    raw = s[:, None] * w[None, :] + 0.15 * noise
    unit_dirs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    magnitudes = rng.uniform(0.1, 5.0, size=n)
    H = unit_dirs * magnitudes[:, None]

    result = rp.region_partition(H, min_norm_percentile=0.0)

    assert result["labels"].shape[0] == n
    ari = adjusted_rand_score(s, result["labels"])
    assert ari == 1.0
    assert abs(float(np.dot(result["v"], w))) > 0.99


# --- Test 2: inclusive boundary (>=, never >) -----------------------------------------------


def test_region_partition_inclusive_boundary():
    """H whose ||H|| values are 1..21 (chosen so the 25th percentile of an evenly spaced
    integer sequence lands exactly on a data point -- with n=20 points and the default
    linear-interpolation percentile method, the 25th percentile of 1..20 falls at 5.75,
    between two data points, so n=21 is used instead so the percentile lands exactly on
    the value 6.0 with no interpolation). The point whose norm equals the percentile
    value exactly must be KEPT: the comparison is greater-than-or-equal, not
    strictly-greater."""
    direction = np.array([1.0, 0.0, 0.0])
    norms = np.arange(1, 22, dtype=np.float64)  # 1..21
    H = norms[:, None] * direction[None, :]

    result = rp.region_partition(H, min_norm_percentile=25.0)

    floor = np.percentile(norms, 25.0)
    boundary_idx = np.flatnonzero(norms == floor)
    assert boundary_idx.size == 1  # construction sanity: exactly one point at the boundary
    idx = int(boundary_idx[0])
    assert idx in result["keep_idx"]
    assert idx not in result["excluded_idx"]
    assert float(result["floor"]) == float(floor)


# --- Test 3: sign canonicalization and reproducibility --------------------------------------


def test_region_partition_reproducible_and_sign_canonical():
    """Calling region_partition twice on the same input returns identical v and labels;
    canonical_eigvec_sign applied to a vector and to its negation returns the same
    vector."""
    rng = np.random.default_rng(1)
    H = rng.normal(size=(50, 5)) * rng.uniform(0.5, 3.0, size=(50, 1))

    r1 = rp.region_partition(H, min_norm_percentile=0.0)
    r2 = rp.region_partition(H, min_norm_percentile=0.0)
    assert np.array_equal(r1["v"], r2["v"])
    assert np.array_equal(r1["labels"], r2["labels"])

    v = rng.normal(size=6)
    assert np.array_equal(rp.canonical_eigvec_sign(v), rp.canonical_eigvec_sign(-v))


# --- Test 4: counts close, zero-projection assignment ----------------------------------------


def test_region_counts_close_and_zero_projection_goes_to_region_0():
    """region_counts(labels, n_excluded)'s three counts sum exactly to the original point
    count. A point whose signed projection is exactly 0.0 lands in region 0 (proj >= 0)
    and is counted in n_zero_projection.

    Construction: 8 points collinear with a fixed x-axis direction (4 positive, 4
    negative sign, varying magnitude) plus 2 points collinear with the orthogonal y-axis
    (1 positive, 1 negative sign). The x-direction carries strictly more variance (8
    points vs 2), so the top eigenvector v is unambiguously the x-axis; the two y-axis
    points then have proj = unit . v == 0.0 exactly."""
    x_signs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    x_mags = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    H_x = np.stack([x_signs * x_mags, np.zeros(8)], axis=1)
    H_y = np.array([[0.0, 1.5], [0.0, -2.5]])
    H = np.concatenate([H_x, H_y], axis=0)
    n = H.shape[0]

    result = rp.region_partition(H, min_norm_percentile=0.0)
    assert result["n_zero_projection"] == 2
    y_point_indices = [8, 9]
    for i in y_point_indices:
        pos_in_kept = int(np.flatnonzero(result["keep_idx"] == i)[0])
        assert result["proj"][pos_in_kept] == 0.0
        assert result["labels"][pos_in_kept] == 0

    counts = rp.region_counts(
        result["labels"], len(result["excluded_idx"]), result["n_zero_projection"]
    )
    assert counts["n_region_0"] + counts["n_region_1"] + counts["n_excluded"] == n
    assert counts["n_zero_projection"] == 2


# --- Guard tests ------------------------------------------------------------------------------


def test_region_partition_raises_on_one_dimensional_H():
    with pytest.raises(ValueError):
        rp.region_partition(np.zeros(10), min_norm_percentile=0.0)


def test_region_partition_raises_on_non_finite_H():
    H = np.ones((10, 3))
    H[3, 1] = np.nan
    with pytest.raises(ValueError):
        rp.region_partition(H, min_norm_percentile=0.0)


def test_region_partition_raises_on_percentile_out_of_range():
    H = np.random.default_rng(2).normal(size=(10, 3))
    with pytest.raises(ValueError):
        rp.region_partition(H, min_norm_percentile=100.0)
    with pytest.raises(ValueError):
        rp.region_partition(H, min_norm_percentile=-1.0)
