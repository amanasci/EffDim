"""
Fast synthetic-fixture tests for the ``pu_manifold.curvature_probe`` module.

No HuggingFace access, no torch, no fixtures beyond synthetic point clouds generated
in-test. Not collected by the core `effdim` test suite (``pyproject.toml``'s
``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_curvature_probe.py -q

Every test here exists to prove a function correct against a synthetic input whose
answer is known independently (a flat plane, a sphere, the Swiss roll's own closed-form
mean curvature), not merely plausible -- same discipline as ``test_geometry_probes.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

from pu_manifold import curvature_probe as cp


# --- Task 1: end-to-end tracer ----------------------------------------------------------


def test_tracer_swiss_roll_end_to_end():
    """One path only: generate a Swiss roll under CLAUDE.md's exact preprocessing,
    estimate its local mean curvature field, and rank it against the closed-form
    analytic answer via Spearman.

    The 0.5 floor here is a TRACER SANITY FLOOR ONLY -- it is NOT the D-01/D-02 gate,
    which is null-calibrated and pre-registered in plan 02.5-06. Do not mistake this
    assertion for that gate.
    """
    X_raw, t = make_swiss_roll(n_samples=3000, noise=0.0, random_state=20260807)
    global_std = X_raw.std()  # single scalar, no axis argument (CLAUDE.md)
    X = (X_raw - X_raw.mean(axis=0)) / global_std

    h_true = cp.swiss_roll_analytic_H_scaled(t, global_std)

    H_est = cp.centroid_mean_curvature(X, k=15, d=2)
    h_est = cp.mean_curvature_norm(H_est)

    rho = cp.spearman_gate_statistic(h_est, h_true)
    assert rho > 0.5  # tracer sanity floor only -- NOT the pre-registered D-01/D-02 gate


# --- Task 2: closed-form known-answer fixtures and the trace-vs-averaged convention -----


def _flat_plane_fixture(n: int, D: int, seed: int):
    """`n` points uniform in `[-1, 1]^2` (true `d=2` tangent plane), embedded into `R^D`
    by padding with exact zeros, then rotated by a fixed random orthogonal `D x D` matrix
    so the tangent plane is not axis-aligned. A flat manifold has zero curvature exactly.

    Returns `(X, mean_nn_dist)` -- the point cloud and its mean nearest-neighbour
    distance, used to build a scale-aware near-zero tolerance.
    """
    rng = np.random.default_rng(seed)
    uv = rng.uniform(-1.0, 1.0, size=(n, 2))
    X = np.zeros((n, D))
    X[:, :2] = uv
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))  # random orthogonal rotation
    X = X @ Q.T
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    dist, _ = nn.kneighbors(X)
    mean_nn_dist = float(dist[:, 1].mean())
    return X, mean_nn_dist


def _sample_sphere(d: int, radius: float, n: int, seed: int) -> np.ndarray:
    """`n` points on the `d`-sphere of the given `radius`, embedded in `R^{d+1}`, via
    normalized Gaussian sampling (a standard exactly-uniform construction) with a fixed
    seed. Under this module's trace convention, `||H|| = d / radius` exactly."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, d + 1))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return radius * pts / norms


def test_centroid_estimator_known_curvature():
    """Two known-answer fixtures whose analytic curvature is exact, not merely
    plausible: a flat plane (`||H|| = 0`) and a sphere (`||H|| = d/R`)."""
    # (a) Flat d=2 plane in R^10: true ||H|| = 0 exactly. Anything above numerical
    # noise (scaled by the fixture's own inverse mean nearest-neighbour distance,
    # since H has units of inverse length) is a bug, not finite-radius bias.
    X_plane, mean_nn_dist = _flat_plane_fixture(n=2000, D=10, seed=7)
    H_plane = cp.centroid_mean_curvature(X_plane, k=30, d=2)
    h_plane_norm = cp.mean_curvature_norm(H_plane)
    scale = 1.0 / mean_nn_dist
    assert np.median(h_plane_norm) < 1e-6 * scale

    # (b) d=2 sphere, R=1.5, in R^3: true ||H|| = d/R = 2/1.5. The 20% band is the
    # O(r^2) finite-radius bias at k=30, not a tuning knob -- see Pattern 1's
    # derivation (relative bias is O(r^2), r set implicitly by k).
    R = 1.5
    X_sphere = _sample_sphere(d=2, radius=R, n=3000, seed=11)
    H_sphere = cp.centroid_mean_curvature(X_sphere, k=30, d=2)
    median_est = np.median(cp.mean_curvature_norm(H_sphere))
    true_H = 2 / R
    assert abs(median_est - true_H) / true_H < 0.20


def test_curvature_convention_is_trace_not_averaged():
    """The OQ-CONV regression guard.

    Spearman is invariant to the trace-vs-averaged factor of `d`, so D-01's gate would
    never catch a silent regression to the averaged convention -- but D-01's non-gating
    median relative error and D-05's estimator-agreement check would both be wrong by
    `d`. This test pins the convention two ways: the Swiss roll closed form at a fixed
    point, and the sphere estimator's scaling with `d` at fixed radius.
    """
    t = np.array([2 * np.pi])
    expected_trace = (4 * np.pi**2 + 2) / (1 + 4 * np.pi**2) ** 1.5
    val = float(cp.swiss_roll_analytic_H(t)[0])
    assert np.isclose(val, expected_trace, rtol=1e-12)
    # A module using the averaged convention (kappa/2) would return exactly half of
    # `expected_trace` here, not strictly more than half.
    assert val > expected_trace / 2

    # A module using the averaged convention would return the same ||H|| for both d=2
    # and d=5 spheres of the same radius; the trace convention scales with d. d=5's
    # sphere needs more points than d=2's at the same k=30 to keep finite-radius bias
    # (O(r^2), and r grows faster with d at fixed n/k) from eating the 2x margin --
    # n is chosen per-d for that reason, not tuned to hit the assertion.
    n_by_d = {2: 3000, 5: 15000}
    medians = {}
    for d in (2, 5):
        X = _sample_sphere(d=d, radius=1.0, n=n_by_d[d], seed=101 + d)
        H = cp.centroid_mean_curvature(X, k=30, d=d)
        medians[d] = np.median(cp.mean_curvature_norm(H))
    assert medians[5] >= 2 * medians[2]


def _numerical_swiss_roll_H(t_vals: np.ndarray, y: float = 0.0, h: float = 1e-3) -> np.ndarray:
    """Central-finite-difference mean curvature (trace convention, d=2) of the exact
    parametric surface `X(t, y) = (t*cos(t), y, t*sin(t))`, computed independently of
    `swiss_roll_analytic_H`'s closed form via the first/second fundamental forms
    (`E, F, G` and `L, M, N`). `H_avg = (E*N - 2*F*M + G*L) / (2*(E*G - F^2))`, then
    `H_trace = 2 * H_avg` -- the explicit conversion from the textbook averaged formula
    to this module's trace convention, which is the whole point of this helper.
    """

    def X(t, y):
        return np.array([t * np.cos(t), y, t * np.sin(t)])

    H_vals = np.zeros_like(t_vals, dtype=np.float64)
    for i, t in enumerate(t_vals):
        Xt = (X(t + h, y) - X(t - h, y)) / (2 * h)
        Xy = (X(t, y + h) - X(t, y - h)) / (2 * h)
        Xtt = (X(t + h, y) - 2 * X(t, y) + X(t - h, y)) / h**2
        Xty = (
            (X(t + h, y + h) - X(t + h, y - h)) - (X(t - h, y + h) - X(t - h, y - h))
        ) / (4 * h**2)
        Xyy = (X(t, y + h) - 2 * X(t, y) + X(t, y - h)) / h**2

        n = np.cross(Xt, Xy)
        n = n / np.linalg.norm(n)

        E = Xt @ Xt
        F = Xt @ Xy
        G = Xy @ Xy
        L = Xtt @ n
        M = Xty @ n
        N = Xyy @ n

        H_avg = (E * N - 2 * F * M + G * L) / (2 * (E * G - F**2))
        H_vals[i] = 2 * H_avg  # d=2 trace convention: H_trace = 2 * H_avg
    return H_vals


def test_swiss_roll_analytic_H_matches_numerical():
    """`swiss_roll_analytic_H` matches an independent central-finite-difference
    computation of the exact parametric surface's mean curvature, to a tight tolerance."""
    t_vals = np.linspace(1.5 * np.pi, 4.5 * np.pi, 25)
    H_numeric = _numerical_swiss_roll_H(t_vals)
    H_analytic = cp.swiss_roll_analytic_H(t_vals)
    assert np.allclose(H_numeric, H_analytic, rtol=1e-4)


def test_local_tangent_basis_shapes_and_orthonormality():
    rng = np.random.default_rng(5)
    d, D, k = 3, 8, 20
    # A neighbourhood whose tangent plane is known exactly: the first d coordinates.
    centered = np.zeros((k, D))
    centered[:, :d] = rng.standard_normal((k, d))

    Vt = cp.local_tangent_basis(centered, d)
    assert Vt.shape == (d, D)
    assert np.allclose(Vt @ Vt.T, np.eye(d), atol=1e-10)

    # Spans the known tangent plane: each of its basis vectors round-trips through
    # projection onto Vt's row space unchanged.
    for i in range(d):
        e_i = np.zeros(D)
        e_i[i] = 1.0
        proj = Vt.T @ (Vt @ e_i)
        assert np.allclose(proj, e_i, atol=1e-10)

    with pytest.raises(ValueError, match=r"d=.*k="):
        cp.local_tangent_basis(centered, d=100)


# --- Task 3: guard the normal projection against the density-leak failure mode ----------


def _skewed_flat_plane_fixture(n: int, D: int, seed: int, power: float = 3.0) -> np.ndarray:
    """`n` points on an exact `d=2` flat plane (true `||H|| = 0` everywhere) embedded in
    `R^D`, but with a deliberately NON-UNIFORM sampling density along one tangent
    direction: `u = sign(z) * |z|^power` for `z ~ Uniform(-1, 1)` bunches points near
    `u = 0` and thins them toward the edges, a pure reparametrization of the SAME flat
    coordinate axis (the manifold never leaves the plane, so its curvature stays exactly
    zero) that gives almost every neighbourhood a nonzero local density gradient -- the
    Pitfall 3 / D-05 scenario where the raw centroid gap is large but is not curvature.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n)
    u = np.sign(z) * np.abs(z) ** power
    v = rng.uniform(-1.0, 1.0, size=n)
    X = np.zeros((n, D))
    X[:, 0] = u
    X[:, 1] = v
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))  # random orthogonal rotation
    return X @ Q.T


def _raw_centroid_gap_norms(X: np.ndarray, k: int) -> np.ndarray:
    """The UNPROJECTED centroid gap norm per point -- `||mean(neighbours) - p||` -- with
    no tangent/normal split. Mirrors `centroid_mean_curvature`'s kNN and gap steps
    exactly, but deliberately omits the normal projection, so it can be compared against
    the real estimator's (projected) output."""
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)
    gaps = np.zeros(n)
    for i in range(n):
        neigh = X[idx[i, 1:]]
        centered = neigh - X[i]
        gaps[i] = np.linalg.norm(centered.mean(axis=0))
    return gaps


def test_tangential_perturbation_does_not_leak_into_H():
    """The Pitfall 3 / D-05 regression guard -- the sharpest single test in this plan.

    A purely tangential density asymmetry must not be reportable as curvature. This is
    what distinguishes a curvature estimator from a density meter, and it is expected to
    FAIL if the `gap - Vt.T @ (Vt @ gap)` line in `centroid_mean_curvature` is ever
    reduced to a bare `gap` -- demonstrated below, then reverted.
    """
    X = _skewed_flat_plane_fixture(n=3000, D=5, seed=13)

    # (a) the perturbation genuinely bit: the raw, unprojected centroid gap is well
    # above zero (not merely numerical noise) for these neighbourhoods.
    raw_gap_norms = _raw_centroid_gap_norms(X, k=30)
    assert np.median(raw_gap_norms) > 1e-3

    # (b) but the real estimator's ||H|| output stays at numerical-noise scale, because
    # that gap lives entirely in the tangent space the normal projection removes.
    H_est = cp.centroid_mean_curvature(X, k=30, d=2)
    h_est_norm = cp.mean_curvature_norm(H_est)
    assert np.median(h_est_norm) < 1e-8


# --- D-01's gating statistic behaves as claimed ------------------------------------------


def _monotone_noised_pair(seed: int, n: int = 500):
    """`h_true` strictly increasing; `h_est` a strictly monotone transform of `h_true`
    plus bounded noise, drawn from a fixed seed so the pair is exactly reproducible."""
    rng = np.random.default_rng(seed)
    h_true = np.linspace(0.0, 10.0, n)
    h_est = h_true**1.3 + rng.normal(scale=0.05, size=n)
    return h_true, h_est


def test_spearman_gate_recovers_ordering():
    """D-01's gating statistic behaves as claimed: it recovers a strong monotone
    relationship and reports near-zero correlation once that relationship is destroyed
    by shuffling. Follows `test_geometry_probes.py:31-51`'s same-seed/different-seed
    reproducibility convention."""
    h_true, h_est = _monotone_noised_pair(seed=42)
    rho = cp.spearman_gate_statistic(h_est, h_true)
    assert rho > 0.9

    rng_shuffle = np.random.default_rng(7)
    h_shuffled = rng_shuffle.permutation(h_est)
    rho_shuffled = cp.spearman_gate_statistic(h_shuffled, h_true)
    assert abs(rho_shuffled) < 0.2

    # same-seed reproducibility: identical inputs give a bit-for-bit identical statistic
    h_true_b, h_est_b = _monotone_noised_pair(seed=42)
    rho_b = cp.spearman_gate_statistic(h_est_b, h_true_b)
    assert rho == rho_b

    # a different seed gives a different value
    h_true_c, h_est_c = _monotone_noised_pair(seed=43)
    rho_c = cp.spearman_gate_statistic(h_est_c, h_true_c)
    assert rho_c != rho
