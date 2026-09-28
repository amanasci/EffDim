"""
Fast synthetic-fixture tests for the ``pu_manifold.geometry_probes`` module.

No HuggingFace access, no torch, no fixtures beyond ``tmp_path``/``monkeypatch``. Not
collected by the core `effdim` test suite (``pyproject.toml``'s ``testpaths = ["tests"]``
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_geometry_probes.py -q

Every test here exists to close the Wave 0 gap `02.1-VALIDATION.md` records: every number
`geometry_probes_run.py` will later report comes from a function first proved correct
against a synthetic input whose answer is known independently, not merely plausible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial.distance import pdist, squareform

from pu_manifold import geometry_probes as g


# --- draw_geo_pairs ------------------------------------------------------------------------


def test_draw_geo_pairs_shapes_and_no_self_pairs():
    rng = np.random.default_rng(0)
    rows, cols = g.draw_geo_pairs(rng, n=500, count=10_000)
    assert rows.shape == (10_000,)
    assert cols.shape == (10_000,)
    assert rows.dtype.kind in "iu"
    assert cols.dtype.kind in "iu"
    assert not np.any(rows == cols)


def test_draw_geo_pairs_reproducible_under_fixed_seed():
    a_rows, a_cols = g.draw_geo_pairs(np.random.default_rng(42), n=500, count=5_000)
    b_rows, b_cols = g.draw_geo_pairs(np.random.default_rng(42), n=500, count=5_000)
    assert np.array_equal(a_rows, b_rows)
    assert np.array_equal(a_cols, b_cols)


def test_draw_geo_pairs_different_seeds_give_different_arrays():
    a_rows, a_cols = g.draw_geo_pairs(np.random.default_rng(1), n=500, count=5_000)
    b_rows, b_cols = g.draw_geo_pairs(np.random.default_rng(2), n=500, count=5_000)
    assert not (np.array_equal(a_rows, b_rows) and np.array_equal(a_cols, b_cols))


# --- sampled_delta_hyperbolicity -------------------------------------------------------------


def _weighted_tree_geodesics(n, seed):
    """A random spanning tree over `n` nodes with random positive edge weights; its
    shortest-path metric satisfies the four-point condition exactly (delta == 0)."""
    rng = np.random.default_rng(seed)
    # A random complete-graph weight matrix, then take its MST -- guarantees a tree.
    W = rng.uniform(0.5, 5.0, size=(n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    mst = minimum_spanning_tree(W)
    D = shortest_path(mst, method="D", directed=False)
    return D


def test_sampled_delta_hyperbolicity_on_tree_is_zero():
    D = _weighted_tree_geodesics(300, seed=0)
    out = g.sampled_delta_hyperbolicity(D, n_quadruples=20_000, seed=0)
    assert out["delta_max"] == pytest.approx(0.0, abs=1e-8)


def test_sampled_delta_hyperbolicity_on_flat_euclidean_is_materially_larger():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20))
    D = squareform(pdist(X))
    tree_D = _weighted_tree_geodesics(300, seed=0)

    out_euclidean = g.sampled_delta_hyperbolicity(D, n_quadruples=20_000, seed=0)
    out_tree = g.sampled_delta_hyperbolicity(tree_D, n_quadruples=20_000, seed=0)

    assert out_euclidean["delta_rel_max"] > out_tree["delta_rel_max"]
    assert out_euclidean["delta_rel_max"] > 0.05


def test_sampled_delta_hyperbolicity_reproducible_and_has_required_keys():
    D = _weighted_tree_geodesics(200, seed=3)
    a = g.sampled_delta_hyperbolicity(D, n_quadruples=5_000, seed=7)
    b = g.sampled_delta_hyperbolicity(D, n_quadruples=5_000, seed=7)
    for key in (
        "delta_max", "delta_p95", "delta_mean", "diam_sample",
        "delta_rel_max", "delta_rel_p95", "n_quadruples",
    ):
        assert key in a
        assert a[key] == pytest.approx(b[key])


# --- krein_mass_capture ----------------------------------------------------------------------


def test_krein_mass_capture_nonnegative_spectrum_identical_across_q():
    rng = np.random.default_rng(0)
    eigvals = np.sort(rng.uniform(0.1, 10.0, size=100))[::-1]  # all positive
    p_ladder = (1, 2, 5, 10)
    q_ladder = (0, 1, 3)
    out = g.krein_mass_capture(eigvals, p_ladder, q_ladder)
    for p in p_ladder:
        captures = {row["mass_capture"] for row in out if row["p"] == p}
        assert len(captures) == 1, f"p={p} capture varies across q on an all-positive spectrum"


def test_krein_mass_capture_monotone_and_full_coverage():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.1, 10.0, size=60)
    neg = -rng.uniform(0.1, 10.0, size=40)
    eigvals = np.concatenate([pos, neg])
    p_ladder = (5, 10, 20, 60)
    q_ladder = (0, 5, 10, 40)
    out = g.krein_mass_capture(eigvals, p_ladder, q_ladder)
    by_pq = {(row["p"], row["q"]): row["mass_capture"] for row in out}

    for q in q_ladder:
        vals = [by_pq[(p, q)] for p in p_ladder]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), "not non-decreasing in p"
    for p in p_ladder:
        vals = [by_pq[(p, q)] for q in q_ladder]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), "not non-decreasing in q"

    assert by_pq[(60, 40)] == pytest.approx(1.0, abs=1e-8)


# --- pseudo_euclidean_sq_distances ------------------------------------------------------------


def _nonmetric_graph_sq_distances(n, seed):
    """A random weighted graph's shortest-path metric squared -- generically non-Euclidean,
    so its double-centred matrix carries negative eigenvalues (verified below by construction:
    the full signed eigendecomposition reconstructs it exactly, but the positive-only
    restriction does not, which is only possible if negative eigenvalues are present)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.5, 5.0, size=(n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    # sparsify so shortest paths are genuinely graph-geodesic, not direct edges
    mask = rng.uniform(size=(n, n)) < 0.3
    mask = mask | mask.T
    np.fill_diagonal(mask, True)
    Wg = np.where(mask, W, 0.0)
    D = shortest_path(Wg, method="D", directed=False)
    assert np.isfinite(D).all(), "graph must be connected for a finite geodesic metric"
    return D ** 2


def _double_centre(d2):
    row = d2.mean(axis=1, keepdims=True)
    col = d2.mean(axis=0, keepdims=True)
    tot = d2.mean()
    return -0.5 * (d2 - row - col + tot)


def test_pseudo_euclidean_reconstruction_exact_with_full_signed_eigendecomposition():
    n = 40
    d2 = _nonmetric_graph_sq_distances(n, seed=0)
    B = _double_centre(d2)
    eigvals, eigvecs = np.linalg.eigh(B)
    assert np.any(eigvals < -1e-8), "fixture must carry negative eigenvalues to be a real test"

    rows, cols = np.triu_indices(n, k=1)
    rec = g.pseudo_euclidean_sq_distances(eigvals, eigvecs, rows, cols)
    np.testing.assert_allclose(rec, d2[rows, cols], rtol=1e-8, atol=1e-8)


def test_pseudo_euclidean_positive_only_restriction_biased_upward():
    n = 40
    d2 = _nonmetric_graph_sq_distances(n, seed=0)
    B = _double_centre(d2)
    eigvals, eigvecs = np.linalg.eigh(B)

    pos_mask = eigvals > 0
    rows, cols = np.triu_indices(n, k=1)
    rec_pos_only = g.pseudo_euclidean_sq_distances(
        eigvals[pos_mask], eigvecs[:, pos_mask], rows, cols
    )
    true_d2 = d2[rows, cols]
    assert not np.allclose(rec_pos_only, true_d2, rtol=1e-8, atol=1e-8)
    # dropping the negative block removes a subtracted term -> reconstruction biased upward
    assert np.median(rec_pos_only - true_d2) > 0


# --- distortion_stats --------------------------------------------------------------------------


def test_distortion_stats_zero_on_identical_inputs():
    rng = np.random.default_rng(0)
    d2 = rng.uniform(1.0, 100.0, size=1000)
    out = g.distortion_stats(d2, d2)
    assert out["median_abs_rel"] == pytest.approx(0.0, abs=1e-12)
    assert out["median_signed_rel"] == pytest.approx(0.0, abs=1e-12)
    assert out["p95_abs_rel"] == pytest.approx(0.0, abs=1e-12)


def test_distortion_stats_uniform_inflation():
    rng = np.random.default_rng(0)
    d2_geo = rng.uniform(1.0, 100.0, size=1000)
    d2_rep = d2_geo * 1.10
    out = g.distortion_stats(d2_rep, d2_geo)
    assert out["median_signed_rel"] == pytest.approx(0.10, abs=1e-9)
    assert out["median_abs_rel"] == pytest.approx(0.10, abs=1e-9)


# --- correction_blindness ---------------------------------------------------------------------


def test_correction_blindness_zero_negative_mass_on_twenty_five_arbitrary_spectra():
    rng = np.random.default_rng(7)
    for _ in range(25):
        eigvals = rng.standard_normal(200) * rng.uniform(0.5, 50.0)
        out = g.correction_blindness(eigvals)
        assert out["m_after_clip"] == 0.0
        assert out["m_after_shift"] == 0.0


def test_correction_blindness_m_before_matches_gate_stats_definition():
    rng = np.random.default_rng(1)
    eigvals = rng.standard_normal(500)
    out = g.correction_blindness(eigvals)
    expected_m = np.abs(eigvals[eigvals < 0]).sum() / np.abs(eigvals).sum()
    assert out["m_before"] == pytest.approx(expected_m)


def test_correction_blindness_on_already_psd_and_nonmetric_spectra():
    rng = np.random.default_rng(2)
    psd = rng.uniform(0.1, 10.0, size=100)  # already all-positive
    out_psd = g.correction_blindness(psd)
    assert out_psd["m_before"] == pytest.approx(0.0, abs=1e-12)
    assert out_psd["m_after_clip"] == 0.0
    assert out_psd["m_after_shift"] == 0.0

    nonmetric = rng.standard_normal((100, 100))
    nonmetric = (nonmetric + nonmetric.T) / 2
    ev = np.linalg.eigvalsh(nonmetric)
    out_nonmetric = g.correction_blindness(ev)
    assert out_nonmetric["m_after_clip"] == 0.0
    assert out_nonmetric["m_after_shift"] == 0.0


# --- kneedle_elbow -----------------------------------------------------------------------------


def test_kneedle_elbow_unambiguous_curve():
    x = np.arange(20, dtype=float)
    y = np.concatenate([np.linspace(0, 1, 5), np.full(15, 1.0)])
    idx = g.kneedle_elbow(x, y)
    assert 3 <= idx <= 6


def test_kneedle_elbow_straight_line_ties_to_lower_index():
    x = np.arange(10, dtype=float)
    y = x.copy()
    idx = g.kneedle_elbow(x, y)
    assert idx == 0


def test_kneedle_elbow_invariant_to_affine_rescaling():
    x = np.arange(20, dtype=float)
    y = np.concatenate([np.linspace(0, 1, 5), np.full(15, 1.0)])
    idx_a = g.kneedle_elbow(x, y)
    idx_b = g.kneedle_elbow(x * 1000.0 + 5.0, y * 37.0 - 2.0)
    assert idx_a == idx_b
