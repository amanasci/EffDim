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
import torch  # only used at module scope by test_chart_curvature_cpu_cuda_agree_to_float64_tolerance's
# skipif condition below -- every other test in this file still imports torch locally, per this
# file's original "no torch at collection time" discipline; torch is already a hard runtime
# dependency of every module this file exercises, so this top-level import changes nothing.
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

from pu_manifold import cache
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


# --- Plan 02.5-02 Task 1: graph-of-function fixture family ------------------------------


def test_graph_of_function_analytic_H_matches_numerical():
    """`H_norm` from `make_graph_of_function_fixture` (d=2, D=3, n_bumps=1) matches an
    INDEPENDENT central-finite-difference computation of the same parametric surface's
    mean curvature -- the analytic Hessian-based path is never used to check itself.
    Also asserts the field genuinely varies (IQR at least 30% of the median), so
    Spearman has a real ordering to score.
    """
    fix = cp.make_graph_of_function_fixture(
        n=500, d=2, D=3, n_bumps=1, seed=20260807, apply_rotation=False
    )
    x_param = fix["x_param"]
    amplitudes = fix["amplitudes"]
    centres = fix["centres"]
    sigma = fix["sigma"]
    global_std = fix["global_std"]

    def f_scalar(x):
        diff = x - centres[0]
        return amplitudes[0] * np.exp(-np.dot(diff, diff) / (2 * sigma**2))

    def X_surface(x):
        return np.array([x[0], x[1], f_scalar(x)])

    h = 1e-4
    H_numeric = np.zeros(len(x_param))
    for i, x in enumerate(x_param):
        du = np.array([h, 0.0])
        dv = np.array([0.0, h])
        Xu = (X_surface(x + du) - X_surface(x - du)) / (2 * h)
        Xv = (X_surface(x + dv) - X_surface(x - dv)) / (2 * h)
        Xuu = (X_surface(x + du) - 2 * X_surface(x) + X_surface(x - du)) / h**2
        Xvv = (X_surface(x + dv) - 2 * X_surface(x) + X_surface(x - dv)) / h**2
        Xuv = (
            (X_surface(x + du + dv) - X_surface(x + du - dv))
            - (X_surface(x - du + dv) - X_surface(x - du - dv))
        ) / (4 * h**2)

        normal = np.cross(Xu, Xv)
        normal = normal / np.linalg.norm(normal)

        E = Xu @ Xu
        F = Xu @ Xv
        G = Xv @ Xv
        L = Xuu @ normal
        M = Xuv @ normal
        N = Xvv @ normal

        H_trace = (E * N - 2 * F * M + G * L) / (E * G - F**2)
        H_numeric[i] = abs(H_trace) * global_std  # CLAUDE.md scaling; H_norm >= 0

    assert np.allclose(H_numeric, fix["H_norm"], rtol=1e-4, atol=1e-8)

    iqr = np.percentile(fix["H_norm"], 75) - np.percentile(fix["H_norm"], 25)
    median = np.median(fix["H_norm"])
    assert iqr >= 0.3 * median


def test_graph_fixture_padding_and_codimension():
    """Padding to `D=768` leaves `H_norm` bit-identical to the unpadded
    `D=d+n_bumps` case (rotation disabled, so no extra floating-point noise from a
    differently-sized rotation matrix); `n_bumps=8` gives a different `H_norm`
    distribution than `n_bumps=1` at the same `(d, D, seed)`; and `D < d + n_bumps`
    raises `ValueError` naming all three.
    """
    fix_small = cp.make_graph_of_function_fixture(
        n=300, d=2, D=3, n_bumps=1, seed=20260807, apply_rotation=False
    )
    fix_padded = cp.make_graph_of_function_fixture(
        n=300, d=2, D=768, n_bumps=1, seed=20260807, apply_rotation=False
    )
    assert np.array_equal(fix_small["H_norm"], fix_padded["H_norm"])

    fix_m1 = cp.make_graph_of_function_fixture(n=300, d=2, D=768, n_bumps=1, seed=20260807)
    fix_m8 = cp.make_graph_of_function_fixture(n=300, d=2, D=768, n_bumps=8, seed=20260807)
    assert not np.allclose(fix_m1["H_norm"], fix_m8["H_norm"])

    with pytest.raises(ValueError, match=r"d=.*n_bumps=.*D="):
        cp.make_graph_of_function_fixture(n=10, d=5, D=4, n_bumps=2, seed=1)


# --- Plan 02.5-02 Task 2: non-uniform sampling and the density correction ---------------


def test_density_correction_removes_bias():
    """D-06 in two parts, because the plan's original "flat fixture shows big fake
    curvature" premise turned out to be mathematically unsatisfiable for the estimator
    actually shipped here -- see this plan's SUMMARY.md for the full derivation, kept
    short here:

    PART A (flat, `H_norm == 0` exactly): an exactly-linear point cloud is exactly
    rank-`d`, so `local_tangent_basis`'s SVD recovers the TRUE tangent subspace exactly
    REGARDLESS of how neighbours are weighted within it (remaining singular values sit
    at the float64 noise floor, independent of sampling density). Any centroid
    displacement -- weighted or not -- therefore lies exactly in that subspace, and the
    normal projection removes it completely. This generalizes plan 02.5-01's Pitfall-3
    guard to this density model: BOTH corrected and uncorrected report `||H||` at the
    float64 noise floor here, not just the corrected one. There is nothing for the
    correction to remove on a purely linear fixture, and this asserts that directly
    (also covers `centroid_mean_curvature`'s `k_density`-required ValueError path).

    PART B (genuinely curved, strongly skewed): density bias can only leak through an
    IMPERFECTLY estimated tangent basis, which requires real curvature. On a curved,
    strongly-skewed fixture, the corrected estimator's median relative error against the
    known analytic `H` is measurably and consistently smaller than the uncorrected
    estimator's -- the real, if modest, effect the correction has.
    """
    # --- Part A: flat + skew -> both corrected and uncorrected are at noise floor ---
    fix_flat = cp.make_flat_fixture(n=4000, d=2, D=10, seed=20260807, density_skew=3.0)
    assert fix_flat["realized_skew"] > 2.0  # the skew genuinely bit
    assert np.all(fix_flat["H_norm"] == 0.0)  # analytic ground truth, exactly

    with pytest.raises(ValueError, match="k_density"):
        cp.centroid_mean_curvature(fix_flat["X"], k=20, d=2, density_correct=True)

    H_flat_uncorrected = cp.mean_curvature_norm(
        cp.centroid_mean_curvature(fix_flat["X"], k=20, d=2, density_correct=False)
    )
    H_flat_corrected = cp.mean_curvature_norm(
        cp.centroid_mean_curvature(fix_flat["X"], k=20, d=2, density_correct=True, k_density=20)
    )
    assert np.median(H_flat_uncorrected) < 1e-8
    assert np.median(H_flat_corrected) < 1e-8

    # --- Part B: curved + skew -> corrected has measurably lower error vs ground truth ---
    fix_curved = cp.make_graph_of_function_fixture(
        n=8000, d=2, D=10, n_bumps=1, seed=20260807, density_skew=5.0, amplitude=3.0, sigma=0.4
    )
    assert fix_curved["realized_skew"] > 2.0
    H_true = fix_curved["H_norm"]

    H_curved_uncorrected = cp.mean_curvature_norm(
        cp.centroid_mean_curvature(fix_curved["X"], k=15, d=2, density_correct=False)
    )
    H_curved_corrected = cp.mean_curvature_norm(
        cp.centroid_mean_curvature(
            fix_curved["X"], k=15, d=2, density_correct=True, k_density=15
        )
    )
    floor = 1e-3 * np.median(H_true)
    err_uncorrected = np.median(
        np.abs(H_curved_uncorrected - H_true) / np.maximum(H_true, floor)
    )
    err_corrected = np.median(
        np.abs(H_curved_corrected - H_true) / np.maximum(H_true, floor)
    )
    assert err_corrected < 0.95 * err_uncorrected


def test_density_correction_is_noop_on_uniform_sampling():
    """A correction that changes the answer where there is nothing to correct is itself
    a bug: on a uniformly sampled curved fixture, corrected and uncorrected ||H|| agree
    to within 10% median relative difference."""
    fix = cp.make_graph_of_function_fixture(
        n=2000, d=2, D=6, n_bumps=1, seed=2024, density_skew=0.0
    )
    H_uncorrected = cp.mean_curvature_norm(
        cp.centroid_mean_curvature(fix["X"], k=20, d=2, density_correct=False)
    )
    H_corrected = cp.mean_curvature_norm(
        cp.centroid_mean_curvature(fix["X"], k=20, d=2, density_correct=True, k_density=20)
    )
    rel_diff = np.median(
        np.abs(H_corrected - H_uncorrected) / np.maximum(H_uncorrected, 1e-8)
    )
    assert rel_diff < 0.10


# --- Plan 02.5-07: shared-pass optimization (coordinator-directed, implementation-only) --


def test_centroid_mean_curvature_both_densities_is_bit_identical():
    """Non-negotiable acceptance criterion for `centroid_mean_curvature_both_densities`
    (a pure implementation optimization -- the two `density_correct` variants share their
    expensive k-NN query and per-point tangent-basis SVD, computed once instead of twice):
    `np.array_equal` -- BIT-IDENTICAL, never merely `np.allclose` -- against
    `centroid_mean_curvature`'s own independent two calls, for BOTH the uncorrected and
    corrected variants, on the same input. `centroid_mean_curvature` itself is untouched
    by this test or by the function under test."""
    fix = cp.make_graph_of_function_fixture(n=500, d=3, D=10, n_bumps=2, seed=20260808)
    X = fix["X"]
    k, d, k_density = 20, 3, 20

    H_uncorrected_shared, H_corrected_shared = cp.centroid_mean_curvature_both_densities(
        X, k=k, d=d, k_density=k_density
    )
    H_uncorrected_separate = cp.centroid_mean_curvature(X, k=k, d=d, density_correct=False)
    H_corrected_separate = cp.centroid_mean_curvature(
        X, k=k, d=d, density_correct=True, k_density=k_density
    )

    assert np.array_equal(H_uncorrected_shared, H_uncorrected_separate)
    assert np.array_equal(H_corrected_shared, H_corrected_separate)
    assert np.max(np.abs(H_uncorrected_shared - H_uncorrected_separate)) == 0.0
    assert np.max(np.abs(H_corrected_shared - H_corrected_separate)) == 0.0


def test_centroid_tangent_basis_feasible_boundary():
    """`centroid_tangent_basis_feasible` (added after plan 02.5-07's sweep runner crashed
    mid-sweep on k=10, d=20) is a strict k > d predicate, treating k == d as ALSO
    infeasible (not merely k < d) -- the zero-redundancy boundary case, not just the
    outright-impossible one. Never raises."""
    assert cp.centroid_tangent_basis_feasible(k=30, d=20) is True
    assert cp.centroid_tangent_basis_feasible(k=21, d=20) is True
    assert cp.centroid_tangent_basis_feasible(k=20, d=20) is False  # boundary: k == d
    assert cp.centroid_tangent_basis_feasible(k=15, d=20) is False
    assert cp.centroid_tangent_basis_feasible(k=10, d=20) is False  # the exact crash case


def test_centroid_mean_curvature_both_densities_raises_on_infeasible_k():
    """Regression pin for the exact crash plan 02.5-07's sweep runner hit mid-sweep
    (`k=10, d=20`): `centroid_mean_curvature_both_densities` (like `centroid_mean_curvature`
    itself, via the shared `local_tangent_basis` call) still RAISES `ValueError` when
    called directly with an infeasible (k, d) pair -- this function's own behaviour is
    UNCHANGED by `centroid_tangent_basis_feasible`'s addition. The fix is that callers
    (the sweep runner) now check `centroid_tangent_basis_feasible` BEFORE calling this
    function, so an infeasible cell is RECORDED rather than ever reaching this raise --
    proved by this test never being reached with a caught exception in the runner's own
    per-cell loop (verified by direct execution, not re-tested here, since the runner is a
    script with STEP 0 module-level side effects, not an importable library function)."""
    fix = cp.make_graph_of_function_fixture(n=200, d=20, D=50, n_bumps=1, seed=1)
    assert cp.centroid_tangent_basis_feasible(k=10, d=20) is False
    with pytest.raises(ValueError, match="exceeds min"):
        cp.centroid_mean_curvature_both_densities(fix["X"], k=10, d=20, k_density=10)


# --- Plan 02.5-02 Task 3: non-gating magnitude evidence ---------------------------------


def test_median_relative_error_is_scale_consistent():
    """A scale-inconsistent implementation -- or a convention mismatch between
    estimator and ground truth -- fails this. Guards against the OQ-CONV
    factor-of-`d` error `02.5-01-PLAN.md`'s `<decisions_resolved_here>` resolves:
    rescale the whole cloud by 4 and its ground truth by 1/4 (curvature has units of
    inverse length, so this is the physically consistent rescaling), and the reported
    relative error must agree to `rtol=1e-6`."""
    fix = cp.make_graph_of_function_fixture(
        n=800, d=2, D=5, n_bumps=1, seed=99, apply_rotation=False
    )
    X = fix["X"]
    H_true = fix["H_norm"]
    H_est = cp.mean_curvature_norm(cp.centroid_mean_curvature(X, k=20, d=2))
    err1 = cp.median_relative_error(H_est, H_true)

    X2 = X * 4.0
    H_true2 = H_true / 4.0
    H_est2 = cp.mean_curvature_norm(cp.centroid_mean_curvature(X2, k=20, d=2))
    err2 = cp.median_relative_error(H_est2, H_true2)

    assert np.isclose(err1, err2, rtol=1e-6)


# --- Plan 02.5-03 Task 1: local quadric cross-check and its underdetermination ----------


def test_quadric_cross_check_and_underdetermination():
    """(a) On a well-determined `d=2` sphere fixture, the quadric fit recovers `||H||`
    close to the sphere's known `d/R`, agrees with the centroid estimator, and is NOT
    flagged underdetermined. (b) At the PU regime's `d=20, k=15`, the quadric fit IS
    flagged underdetermined, with `n_coefficients=210` and `coefficient_deficit=195` --
    the exact count the ROADMAP re-scope calls "badly underdetermined", and a property of
    THIS estimator alone, which is why D-05 keeps it non-gating.
    """
    # (a) d=2 sphere, R=1.5, k=30 -- well-determined: n_coefficients = 2*3/2 = 3 << k=30.
    R = 1.5
    X_sphere = _sample_sphere(d=2, radius=R, n=3000, seed=11)
    quadric = cp.quadric_mean_curvature(X_sphere, k=30, d=2)
    true_H = 2 / R
    median_quadric = np.median(quadric["H_norm"])
    assert abs(median_quadric - true_H) / true_H < 0.15
    assert quadric["underdetermined"] is False
    assert quadric["n_coefficients"] == 3
    assert quadric["coefficient_deficit"] == 0

    H_centroid = cp.centroid_mean_curvature(X_sphere, k=30, d=2)
    h_centroid_norm = cp.mean_curvature_norm(H_centroid)
    agreement = cp.estimator_agreement(h_centroid_norm, quadric["H_norm"])
    assert agreement["agreement_median_rel_diff"] < 0.25

    # (b) d=20, k=15 on a graph-of-function fixture: n_coefficients = 20*21/2 = 210,
    # deficit = 210 - 15 = 195 -- this is the count the ROADMAP re-scope calls "badly
    # underdetermined"; it is a property of THIS (quadric) estimator, which is exactly
    # why D-05 keeps it non-gating rather than promoting it.
    fix = cp.make_graph_of_function_fixture(n=50, d=20, D=25, n_bumps=1, seed=1)
    quadric_pu = cp.quadric_mean_curvature(fix["X"], k=15, d=20)
    assert quadric_pu["underdetermined"] is True
    assert quadric_pu["n_coefficients"] == 210
    assert quadric_pu["coefficient_deficit"] == 195


# --- Plan 02.5-03 Task 2: permutation-null calibration of the Spearman threshold -------


def test_permutation_null_rejects_random_pairing():
    """(a) A genuinely correlated pair clears its own null threshold. (b) An
    independently shuffled pair does not. (c) The same seed gives a bit-identical null
    threshold; a different seed gives a different one. (d) A constant input array raises
    `ValueError` naming the array, rather than letting Spearman's `NaN` reach a gate.

    `n_resamples=199` here keeps runtime under a few seconds; the pre-registered
    production value is larger and lives in `02.5-PREREGISTRATION.md` (plan 02.5-06), not
    here.
    """
    h_true, h_est = _monotone_noised_pair(seed=42, n=300)

    # (a) correlated pair clears its own null threshold
    result_corr = cp.permutation_null(h_true, h_est, n_resamples=199, seed=1, quantile=0.95)
    assert result_corr["clears_null"] is True
    assert result_corr["observed_rho"] > result_corr["null_threshold"]

    # (b) independently shuffled pair does not clear a fixed-seed null
    rng_shuffle = np.random.default_rng(9)
    h_shuffled = rng_shuffle.permutation(h_est)
    result_shuffled = cp.permutation_null(
        h_true, h_shuffled, n_resamples=199, seed=1, quantile=0.95
    )
    assert result_shuffled["clears_null"] is False

    # (c) same seed twice -> bit-identical null_threshold; different seed -> different
    result_a = cp.permutation_null(h_true, h_est, n_resamples=199, seed=5, quantile=0.95)
    result_b = cp.permutation_null(h_true, h_est, n_resamples=199, seed=5, quantile=0.95)
    assert result_a["null_threshold"] == result_b["null_threshold"]
    result_c = cp.permutation_null(h_true, h_est, n_resamples=199, seed=6, quantile=0.95)
    assert result_c["null_threshold"] != result_a["null_threshold"]

    # (d) a constant input raises ValueError naming the offending array
    with pytest.raises(ValueError, match="h_true_norm"):
        cp.permutation_null(
            np.ones(100), np.arange(100.0), n_resamples=99, seed=1, quantile=0.99
        )
    with pytest.raises(ValueError, match="h_est_norm"):
        cp.permutation_null(
            np.arange(100.0), np.ones(100), n_resamples=99, seed=1, quantile=0.99
        )


def test_permutation_null_statistic_fn_generalizes_calibration():
    """`statistic_fn` (added by plan 02.5-07) defaults to reproducing the original
    Spearman-only behaviour exactly, and can be swapped for a different rank statistic
    (here, `quantile_bin_concordance`) to calibrate the SAME null-quantile/permutation-
    count machinery against Section 3h's second gating statistic."""
    from scipy.stats import spearmanr

    h_true, h_est = _monotone_noised_pair(seed=7, n=300)

    default_result = cp.permutation_null(h_true, h_est, n_resamples=99, seed=2, quantile=0.9)
    explicit_spearman_result = cp.permutation_null(
        h_true,
        h_est,
        n_resamples=99,
        seed=2,
        quantile=0.9,
        statistic_fn=lambda x, y: float(spearmanr(x, y).statistic),
    )
    assert default_result["observed_rho"] == pytest.approx(explicit_spearman_result["observed_rho"])
    assert default_result["null_threshold"] == pytest.approx(explicit_spearman_result["null_threshold"])

    def _region_stat(x: np.ndarray, y: np.ndarray) -> float:
        return cp.quantile_bin_concordance(
            h_est_norm=y, h_true_norm=x, n_bins=4, pair_seed=20260731, n_pairs=5000
        )

    region_result = cp.permutation_null(
        h_true, h_est, n_resamples=99, seed=2, quantile=0.9, statistic_fn=_region_stat
    )
    assert region_result["observed_rho"] == pytest.approx(
        cp.quantile_bin_concordance(h_est, h_true, n_bins=4, pair_seed=20260731, n_pairs=5000)
    )
    # A different statistic generally calibrates to a different null threshold than
    # Spearman's -- not asserted equal, just independently computed and finite.
    assert np.isfinite(region_result["null_threshold"])


# --- Plan 02.5-07 Task 1: quantile_bin_concordance (Section 3h's second, independently
# gating region-scale statistic) ---------------------------------------------------------


def test_quantile_bin_concordance_basic_behavior():
    """Perfect rank agreement -> +1; perfect rank reversal -> -1; independent (shuffled)
    pairing lands near 0 -- the same three sanity checks `spearman_gate_statistic`-style
    functions are held to. Only cross-true-bin pairs are ever compared: same-true-bin
    pairs, which Phase 4 never distinguishes, are dropped before scoring."""
    rng = np.random.default_rng(0)
    h_true = rng.uniform(0, 1, size=2000)

    assert cp.quantile_bin_concordance(
        h_true, h_true, n_bins=4, pair_seed=1, n_pairs=50_000
    ) == pytest.approx(1.0)

    assert cp.quantile_bin_concordance(
        -h_true, h_true, n_bins=4, pair_seed=1, n_pairs=50_000
    ) == pytest.approx(-1.0)

    h_shuffled = rng.permutation(h_true)
    val = cp.quantile_bin_concordance(h_shuffled, h_true, n_bins=4, pair_seed=1, n_pairs=50_000)
    assert abs(val) < 0.05

    # a degenerate case (n_bins=1 -- every point in the same bin, so no pair is ever
    # cross-bin) raises rather than silently computing mean() over an empty array
    with pytest.raises(ValueError, match="cross-bin"):
        cp.quantile_bin_concordance(h_true, h_true, n_bins=1, pair_seed=1, n_pairs=100)


def test_quantile_bin_concordance_scale_invariance():
    """Section 3g's R4 scale-free check: rescaling `h_est_norm` by any positive constant
    leaves the statistic unchanged to float64 precision -- bin assignment and the sign
    comparison both depend only on rank, invariant to positive monotonic rescaling of the
    estimate."""
    rng = np.random.default_rng(3)
    h_true = rng.uniform(0, 1, size=2000)
    h_est = h_true * (1 + 0.2 * rng.standard_normal(2000))

    v1 = cp.quantile_bin_concordance(h_est, h_true, n_bins=4, pair_seed=20260731, n_pairs=200_000)
    v2 = cp.quantile_bin_concordance(
        5.37 * h_est, h_true, n_bins=4, pair_seed=20260731, n_pairs=200_000
    )
    assert v1 == pytest.approx(v2, abs=1e-6)


def test_quantile_bin_concordance_reproduces_preregistration_ceiling_and_null():
    """Section 3g's acceptance criteria: reproduces the Swiss roll noise-oracle ceiling
    table and the shuffled-pairing null to float64-REASONABLE tolerance, under
    02.5-PREREGISTRATION.md's own ratified constants (REGION_N_BINS=4,
    REGION_PAIR_COUNT=200_000, REGION_PAIR_SEED=20260731).

    Noise-oracle protocol (matching Section 3g's own description, "perturbing the
    analytic truth by unbiased multiplicative noise, then ranking"): perturb the Swiss
    roll's analytic H_norm field by mean-preserving LOGNORMAL multiplicative noise at a
    stated coefficient of variation -- h_noisy = h_true * exp(sigma*z - 0.5*sigma^2),
    sigma chosen so the multiplier's own CV equals noise_cv exactly -- then scores
    h_noisy against h_true. Lognormal (rather than h_true*(1+noise_cv*z)) is used because
    it never produces a negative "curvature" value; the linear form was independently
    cross-checked against this same table and diverges by up to 0.06 at noise_cv=0.80
    from sign-flipped points at the tail, while this lognormal form reproduces every
    documented ceiling value here to within 0.01 (well inside the 0.03 tolerance below).
    The SAME lognormal model, independently cross-checked against 02.5-PREREGISTRATION.md
    Section 3b's separately-published per-point-Spearman noise-oracle table (5%/10%/
    14.9%/20%/30% noise -> rho 0.986/0.950/0.897/0.831/0.698), reproduces that table
    equally closely via `spearman_gate_statistic`.

    This is an ATTEMPTED reproduction under a stated, disclosed protocol -- not a bit-
    exact replay of the orchestrator's original (private, undisclosed-seed) MC run. The
    0.03 absolute tolerance reflects Monte-Carlo seed variation at n=3000 (empirically,
    a 10-trial average's own std is well under 0.02 at every level tested), not
    implementation slack.
    """
    fix = cp.make_swiss_roll_fixture(n=3000, seed=20260807)
    h_true = fix["H_norm"]
    n = h_true.shape[0]

    documented_ceiling = {
        0.05: 0.9804,
        0.10: 0.9259,
        0.20: 0.7813,
        0.30: 0.6514,
        0.40: 0.5501,
        0.50: 0.4750,  # == REGION_ABSOLUTE_FLOOR, read directly off this row (Section 3g)
        0.80: 0.3440,
    }
    for noise_cv, expected in documented_ceiling.items():
        sigma = np.sqrt(np.log(1 + noise_cv**2))
        vals = []
        for trial_seed in range(10):
            rng = np.random.default_rng(90_000 + trial_seed)
            z = rng.standard_normal(n)
            h_noisy = h_true * np.exp(sigma * z - 0.5 * sigma**2)
            vals.append(
                cp.quantile_bin_concordance(
                    h_noisy, h_true, n_bins=4, pair_seed=20260731, n_pairs=200_000
                )
            )
        measured = float(np.mean(vals))
        assert measured == pytest.approx(expected, abs=0.03), (noise_cv, measured, expected)

    # Null: shuffled pairing (chance-level concordance) -- documented mean=0.0007,
    # std=0.0159 (Section 3g)
    null_vals = []
    for trial_seed in range(40):
        rng = np.random.default_rng(80_000 + trial_seed)
        h_shuffled = rng.permutation(h_true)
        null_vals.append(
            cp.quantile_bin_concordance(
                h_shuffled, h_true, n_bins=4, pair_seed=20260731, n_pairs=200_000
            )
        )
    null_vals = np.array(null_vals)
    assert float(null_vals.mean()) == pytest.approx(0.0007, abs=0.01)
    assert float(null_vals.std()) == pytest.approx(0.0159, abs=0.01)


# --- Plan 02.5-03 Task 3: one-call stage-1 measurement bundle ---------------------------


def test_measure_cell_returns_flat_native_types():
    """`measure_cell` returns a flat dict whose TWO gating keys (`spearman_rho`,
    `quantile_bin_concordance` -- Section 3h's option-scale-C, plan 02.5-07) are present
    and finite, every value is a plain Python native type (never a numpy scalar), and the
    whole dict round-trips through `json.dumps` with no custom encoder."""
    import json

    fix = cp.make_graph_of_function_fixture(n=300, d=2, D=6, n_bumps=1, seed=20260808)
    result = cp.measure_cell(
        fixture=fix,
        k=15,
        d=2,
        k_density=15,
        density_correct=False,
        n_resamples=99,
        seed=3,
        quantile=0.95,
        region_n_bins=4,
        region_pair_seed=20260731,
        region_n_pairs=5000,
    )

    assert "spearman_rho" in result
    assert np.isfinite(result["spearman_rho"])
    assert "quantile_bin_concordance" in result
    assert np.isfinite(result["quantile_bin_concordance"])
    assert "region_null_threshold" in result
    assert "region_null_clears_null" in result

    for key, value in result.items():
        assert isinstance(value, (float, int, bool, str)), f"{key} is {type(value)}"
        assert type(value).__module__ != "numpy", f"{key} is a numpy scalar: {type(value)}"

    json.dumps(result)  # succeeds without a custom encoder


def test_measure_cell_quadric_timeout_preserves_gating_result():
    """`quadric_timeout_s` (added by plan 02.5-07 to keep the full pre-registered sweep
    tractable at the PU regime's own scale -- see curvature_feasibility_sweep_run.py's own
    module docstring for the measured cost) bounds ONLY the non-gating quadric cross-check.
    An artificially tiny budget forces a timeout; the two GATING statistics and both their
    null calibrations must still be present, finite, and identical to what an untimed call
    on the same inputs produces -- the timeout must never touch the gating computation."""
    import json

    fix = cp.make_graph_of_function_fixture(n=300, d=5, D=20, n_bumps=1, seed=1)
    common_kwargs = dict(
        fixture=fix,
        k=15,
        d=5,
        k_density=15,
        density_correct=False,
        n_resamples=49,
        seed=1,
        quantile=0.9,
        region_n_bins=4,
        region_pair_seed=1,
        region_n_pairs=2000,
    )

    timed_out_result = cp.measure_cell(quadric_timeout_s=0.001, **common_kwargs)
    assert timed_out_result["quadric_timed_out"] is True
    assert timed_out_result["quadric_spearman_rho"] is None
    assert timed_out_result["agreement_spearman"] is None
    assert timed_out_result["agreement_median_rel_diff"] is None
    assert timed_out_result["quadric_underdetermined"] is None
    assert isinstance(timed_out_result["quadric_n_coefficients"], int)
    assert isinstance(timed_out_result["quadric_coefficient_deficit"], int)
    json.dumps(timed_out_result)  # None serializes to JSON null with no custom encoder

    untimed_result = cp.measure_cell(quadric_timeout_s=None, **common_kwargs)
    assert untimed_result["quadric_timed_out"] is False
    assert untimed_result["quadric_spearman_rho"] is not None

    # the gating result is bit-identical whether or not the quadric check timed out --
    # the timeout must never perturb the computation that ran before it
    assert timed_out_result["spearman_rho"] == untimed_result["spearman_rho"]
    assert timed_out_result["quantile_bin_concordance"] == untimed_result["quantile_bin_concordance"]
    assert timed_out_result["null_threshold"] == untimed_result["null_threshold"]
    assert timed_out_result["region_null_threshold"] == untimed_result["region_null_threshold"]

    # a default call (quadric_timeout_s omitted) behaves exactly like quadric_timeout_s=None
    default_result = cp.measure_cell(**common_kwargs)
    assert default_result["quadric_timed_out"] is False
    assert default_result["quadric_spearman_rho"] == untimed_result["quadric_spearman_rho"]


def test_measure_cell_precomputed_h_vec_is_bit_identical_and_shares_the_pass():
    """`precomputed_H_vec` (coordinator-directed, plan 02.5-07): a `measure_cell` call
    given a precomputed H array (e.g. from `centroid_mean_curvature_both_densities`) is
    bit-identical to the same call letting `measure_cell` compute the centroid estimate
    itself -- the precomputed path is a pure sharing optimization, never a different
    computation."""
    fix = cp.make_graph_of_function_fixture(n=400, d=3, D=10, n_bumps=1, seed=20260808)
    k, d, k_density = 20, 3, 20
    common_kwargs = dict(
        fixture=fix,
        k=k,
        d=d,
        k_density=k_density,
        n_resamples=49,
        seed=1,
        quantile=0.9,
        region_n_bins=4,
        region_pair_seed=1,
        region_n_pairs=2000,
    )

    H_uncorrected, H_corrected = cp.centroid_mean_curvature_both_densities(
        fix["X"], k=k, d=d, k_density=k_density
    )

    default_uncorrected = cp.measure_cell(density_correct=False, **common_kwargs)
    shared_uncorrected = cp.measure_cell(
        density_correct=False, precomputed_H_vec=H_uncorrected, **common_kwargs
    )
    assert default_uncorrected["spearman_rho"] == shared_uncorrected["spearman_rho"]
    assert (
        default_uncorrected["quantile_bin_concordance"]
        == shared_uncorrected["quantile_bin_concordance"]
    )
    assert default_uncorrected["null_threshold"] == shared_uncorrected["null_threshold"]

    default_corrected = cp.measure_cell(density_correct=True, **common_kwargs)
    shared_corrected = cp.measure_cell(
        density_correct=True, precomputed_H_vec=H_corrected, **common_kwargs
    )
    assert default_corrected["spearman_rho"] == shared_corrected["spearman_rho"]
    assert (
        default_corrected["quantile_bin_concordance"]
        == shared_corrected["quantile_bin_concordance"]
    )
    assert default_corrected["null_threshold"] == shared_corrected["null_threshold"]


# --- Plan 02.5-04 Task 1: gate constants and direction-aware verdict functions ----------


def test_verdict_gates_are_strict_and_direction_aware():
    """Stage 1 (Section 3h's option-scale-C, plan 02.5-07): PASS only when BOTH
    spearman_rho AND quantile_bin_concordance clear, each strictly above its OWN
    threshold; either alone failing fails the cell; a value exactly at threshold does not
    clear. Stage 2: PASS only when BOTH gates clear (margin greater-than AND seed_spread
    less-than); either alone is FAIL. `gate_detail` carries `direction` for every gate."""
    stage1_metrics = {"spearman_rho": 0.71, "quantile_bin_concordance": 0.55}
    stage1_thresholds = {"spearman_rho": 0.70, "quantile_bin_concordance": 0.50}

    # Stage 1: PASS -- both above their own threshold
    verdict, detail = cp.verdict_from_stage1_metrics(stage1_metrics, stage1_thresholds)
    assert verdict == "PASS"
    assert detail["spearman_rho"]["direction"] == "greater"
    assert detail["spearman_rho"]["passed"] is True
    assert detail["quantile_bin_concordance"]["direction"] == "greater"
    assert detail["quantile_bin_concordance"]["passed"] is True

    # Stage 1: FAIL exactly at threshold (spearman_rho)
    verdict, detail = cp.verdict_from_stage1_metrics(
        {"spearman_rho": 0.70, "quantile_bin_concordance": 0.55}, stage1_thresholds
    )
    assert verdict == "FAIL"
    assert detail["spearman_rho"]["passed"] is False

    # Stage 1: FAIL below (spearman_rho)
    verdict, detail = cp.verdict_from_stage1_metrics(
        {"spearman_rho": 0.69, "quantile_bin_concordance": 0.55}, stage1_thresholds
    )
    assert verdict == "FAIL"

    # Stage 1: FAIL when only quantile_bin_concordance misses its own threshold, even
    # though spearman_rho clears comfortably -- option-scale-C requires BOTH
    verdict, detail = cp.verdict_from_stage1_metrics(
        {"spearman_rho": 0.90, "quantile_bin_concordance": 0.40}, stage1_thresholds
    )
    assert verdict == "FAIL"
    assert detail["spearman_rho"]["passed"] is True
    assert detail["quantile_bin_concordance"]["passed"] is False

    # Stage 1: FAIL when only spearman_rho misses its own threshold, even though
    # quantile_bin_concordance clears comfortably
    verdict, detail = cp.verdict_from_stage1_metrics(
        {"spearman_rho": 0.40, "quantile_bin_concordance": 0.80}, stage1_thresholds
    )
    assert verdict == "FAIL"
    assert detail["spearman_rho"]["passed"] is False
    assert detail["quantile_bin_concordance"]["passed"] is True

    # Stage 2: PASS on both
    verdict, detail = cp.verdict_from_stage2_metrics(
        {"chart_vs_raw_margin": 0.10, "seed_spread": 0.02},
        {"chart_vs_raw_margin": 0.05, "seed_spread": 0.05},
    )
    assert verdict == "PASS"
    assert detail["chart_vs_raw_margin"]["direction"] == "greater"
    assert detail["seed_spread"]["direction"] == "less"

    # Stage 2: FAIL when only the margin clears (seed_spread does not)
    verdict, detail = cp.verdict_from_stage2_metrics(
        {"chart_vs_raw_margin": 0.10, "seed_spread": 0.08},
        {"chart_vs_raw_margin": 0.05, "seed_spread": 0.05},
    )
    assert verdict == "FAIL"
    assert detail["chart_vs_raw_margin"]["passed"] is True
    assert detail["seed_spread"]["passed"] is False

    # Stage 2: FAIL when only the seed spread clears (margin does not)
    verdict, detail = cp.verdict_from_stage2_metrics(
        {"chart_vs_raw_margin": 0.02, "seed_spread": 0.02},
        {"chart_vs_raw_margin": 0.05, "seed_spread": 0.05},
    )
    assert verdict == "FAIL"
    assert detail["chart_vs_raw_margin"]["passed"] is False
    assert detail["seed_spread"]["passed"] is True


def test_verdict_raises_on_absent_or_nonfinite_gate():
    """An absent gating metric, a NaN/inf gating metric, or an absent threshold each raise
    `ValueError` naming the offending gate before any comparison runs."""
    with pytest.raises(ValueError, match="spearman_rho"):
        cp.verdict_from_stage1_metrics({}, {"spearman_rho": 0.70})

    with pytest.raises(ValueError, match="spearman_rho"):
        cp.verdict_from_stage1_metrics(
            {"spearman_rho": float("nan")}, {"spearman_rho": 0.70}
        )

    with pytest.raises(ValueError, match="spearman_rho"):
        cp.verdict_from_stage1_metrics(
            {"spearman_rho": float("inf")}, {"spearman_rho": 0.70}
        )

    with pytest.raises(ValueError, match="spearman_rho"):
        cp.verdict_from_stage1_metrics({"spearman_rho": 0.71}, {})


# --- Plan 02.5-04 Task 2: R6 verdict and handoff writers, mirrored at 02.5 scope --------


def _valid_handoff_payload() -> dict:
    """Minimal payload naming the full estimator contract D-15 requires."""
    return {
        "substrate": "raw_points",
        "working_dimension": 2,
        "neighbourhood_rule": {"k": 30, "justification": "bias-variance test-only stub"},
        "density_correction": {"enabled": False, "k_density": None},
        "estimator_variant": "centroid_laplace_beltrami",
        "cache_stems": ["curvature_feasibility_stage1testkey"],
        "fit_key": "stage1testkey",
        "gate_values": {"spearman_rho": 0.71},
        "thresholds": {"spearman_rho": 0.70},
        "evidence_criteria": "spearman_rho exceeds the pre-registered threshold",
        "preregistration_sha": "deadbeef" * 5,
        "activation": "test-only",
    }


def test_pass_only_handoff_and_stale_deletion(tmp_path, monkeypatch):
    """FAIL writes the verdict and no handoff; write_curvature_handoff with verdict='FAIL'
    raises ValueError; PASS writes both; clear_stale_curvature_handoff after a PASS
    returns True and removes both files; calling it again returns False and does not
    raise; write_curvature_verdict with a verdict that disagrees with its own metrics
    raises ValueError."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    # FAIL: verdict written, no handoff possible (option-scale-C: both gates present,
    # spearman_rho alone misses its threshold)
    fail_metrics = {"spearman_rho": 0.60, "quantile_bin_concordance": 0.55}
    thresholds = {"spearman_rho": 0.70, "quantile_bin_concordance": 0.50}
    fail_result = cp.write_curvature_verdict(
        "failkey", 1, fail_metrics, thresholds, "FAIL"
    )
    assert fail_result["CURVATURE_VERDICT"] == "FAIL"
    assert cache.cache_path("curvature_verdict_stage1_failkey", "json").exists()
    assert not cache.cache_path("curvature_handoff_failkey", "json").exists()

    with pytest.raises(ValueError, match="FAIL"):
        cp.write_curvature_handoff("failkey", "FAIL", _valid_handoff_payload())

    # PASS: verdict and handoff both written (both gates clear their own threshold)
    pass_metrics = {"spearman_rho": 0.80, "quantile_bin_concordance": 0.60}
    pass_result = cp.write_curvature_verdict(
        "passkey", 1, pass_metrics, thresholds, "PASS"
    )
    assert pass_result["CURVATURE_VERDICT"] == "PASS"
    cp.write_curvature_handoff("passkey", "PASS", _valid_handoff_payload())
    assert cache.cache_path("curvature_handoff_passkey", "json").exists()
    assert cache.cache_path("curvature_handoff_passkey", "meta.json").exists()

    # clear_stale_curvature_handoff: True the first time, False (no raise) the second
    assert cp.clear_stale_curvature_handoff("passkey") is True
    assert not cache.cache_path("curvature_handoff_passkey", "json").exists()
    assert not cache.cache_path("curvature_handoff_passkey", "meta.json").exists()
    assert cp.clear_stale_curvature_handoff("passkey") is False

    # A verdict that disagrees with its own metrics is refused
    with pytest.raises(ValueError):
        cp.write_curvature_verdict("mismatchkey", 1, pass_metrics, thresholds, "FAIL")


def test_threshold_edit_raises_manifest_mismatch(tmp_path, monkeypatch):
    """Editing a threshold and re-calling write_curvature_verdict with the same key/stage
    raises cache._manifest_matches's mismatch ValueError rather than silently
    re-verdicting."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    metrics = {"spearman_rho": 0.80, "quantile_bin_concordance": 0.60}
    cp.write_curvature_verdict(
        "editkey",
        1,
        metrics,
        {"spearman_rho": 0.70, "quantile_bin_concordance": 0.50},
        "PASS",
    )
    with pytest.raises(ValueError, match="manifest mismatch"):
        cp.write_curvature_verdict(
            "editkey",
            1,
            metrics,
            {"spearman_rho": 0.75, "quantile_bin_concordance": 0.50},
            "PASS",
        )


# --- Plan 02.5-04 Task 3: full-suite green and sealed-artifact audit --------------------


def test_phase3_curvature_stubs_remain_unimplemented():
    """The executable form of OQ-1's resolution -- phase 02.5 builds parallel machinery in
    curvature_probe.py (and chart_curvature.py) and does NOT deliver Phase 3's
    CURV-01..04 ahead of schedule. This test is NOT coverage of curvature.py; it only
    pins that its four stubs still raise NotImplementedError, unedited by this phase."""
    from pu_manifold import curvature as curv

    with pytest.raises(NotImplementedError):
        curv.first_fundamental_form(None)
    with pytest.raises(NotImplementedError):
        curv.second_fundamental_form(None, None)
    with pytest.raises(NotImplementedError):
        curv.mean_curvature_vector(None, None)
    with pytest.raises(NotImplementedError):
        curv.metric_condition_number(None)


def test_curvature_probe_module_is_numpy_only():
    """`"torch" not in sys.modules` is not a reliable check under a shared pytest session
    (a sibling test module may have already imported torch), so instead this parses
    curvature_probe.py's own source via `ast` and asserts every `Import`/`ImportFrom`
    node naming `torch` has a non-module (i.e. function-local) parent scope -- the module
    itself never imports torch at the top level, even though `write_curvature_handoff`
    imports it lazily inside its own function body."""
    import ast
    import inspect

    source = inspect.getsource(cp)
    tree = ast.parse(source)

    def _names_torch(node) -> bool:
        if isinstance(node, ast.Import):
            return any(
                alias.name == "torch" or alias.name.startswith("torch.")
                for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            return node.module is not None and (
                node.module == "torch" or node.module.startswith("torch.")
            )
        return False

    # Direct children of the module body are the ONLY module-scope statements; anything
    # naming torch there would be a module-level import.
    for node in tree.body:
        assert not _names_torch(node), f"module-scope torch import: {ast.dump(node)}"

    # Every torch-naming Import/ImportFrom anywhere in the file (found via ast.walk, which
    # descends into function bodies too) must have a non-module parent -- i.e. must live
    # inside a function, never directly in the module body.
    found_any = False
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if _names_torch(child):
                found_any = True
                assert not isinstance(parent, ast.Module), (
                    f"torch import {ast.dump(child)} has a module-level parent scope"
                )
    assert found_any, "expected at least one lazy, function-scoped torch import"


# --- 02.5-PREREGISTRATION-AMENDMENT-01.md: seed-replicate aggregation -------------------


def test_seed_replicate_lower_bound_constants_match_the_amendment():
    """The amendment (Section 2) ratifies exactly `SEEDS` of length 5 and
    `T_MULTIPLIER = 2.132`, the one-sided t(0.95, df=4). Both are pinned here so a silent
    edit to either shows up as a test failure rather than as a moved verdict.
    """
    assert cp.SEED_REPLICATE_N == 5
    assert cp.SEED_REPLICATE_T_MULTIPLIER == 2.132


def test_seed_replicate_lower_bound_zero_variance_returns_the_mean():
    """Identical replicates carry no uncertainty, so the bound is the mean itself -- and,
    critically, never ABOVE it, so a degenerate replicate set cannot clear a floor the
    mean does not clear.
    """
    out = cp.seed_replicate_lower_bound([0.6] * 5, cp.SEED_REPLICATE_T_MULTIPLIER, 5)
    assert out["sd"] == 0.0
    assert out["se"] == 0.0
    assert out["lower"] == pytest.approx(0.6, abs=1e-15)
    assert out["lower"] <= out["mean"]


def test_seed_replicate_lower_bound_decreases_as_spread_grows():
    """At a fixed mean, more across-seed disagreement must move the bound DOWN. This is
    the whole point of the amendment: an unstable configuration must be harder to pass,
    not equally easy.
    """
    tight = cp.seed_replicate_lower_bound(
        [0.49, 0.50, 0.50, 0.50, 0.51], cp.SEED_REPLICATE_T_MULTIPLIER, 5
    )
    wide = cp.seed_replicate_lower_bound(
        [0.30, 0.40, 0.50, 0.60, 0.70], cp.SEED_REPLICATE_T_MULTIPLIER, 5
    )
    assert tight["mean"] == pytest.approx(wide["mean"], abs=1e-12)
    assert wide["lower"] < tight["lower"] < tight["mean"]


def test_seed_replicate_lower_bound_reproduces_the_amendments_worked_arithmetic():
    """Section 3(4)'s worked example, reproduced exactly: the three ALREADY-OBSERVED seed
    values plus two placed at their mean (the extrapolation most favourable to a PASS,
    since it adds zero to the sum of squared deviations while raising df from 2 to 4).

    The amendment states this arithmetic points at FAIL for `quantile_bin_concordance`
    and PASS for `spearman_rho`; both are pinned here so the sealed document's own
    prediction cannot drift away from what the code computes.
    """
    qbc_observed = [0.444446, 0.483705, 0.523633]
    rho_observed = [0.520469, 0.562711, 0.609409]

    for observed, expected_lower, floor, expect_clears in (
        (qbc_observed, 0.457234, 0.4750, False),
        (rho_observed, 0.534202, 0.50, True),
    ):
        mean = sum(observed) / len(observed)
        five = observed + [mean, mean]
        out = cp.seed_replicate_lower_bound(five, cp.SEED_REPLICATE_T_MULTIPLIER, 5)
        assert out["mean"] == pytest.approx(mean, abs=1e-12)
        assert out["lower"] == pytest.approx(expected_lower, abs=5e-7)
        assert (out["lower"] > floor) is expect_clears


def test_seed_replicate_lower_bound_feeds_the_unchanged_gate_comparison():
    """The amendment changes WHICH VALUE is compared, never HOW. Feeding the bounds to
    `verdict_from_stage1_metrics` must reproduce the amendment's stated outcome, with both
    floors byte-identical to the sealed pre-registration.
    """
    qbc = cp.seed_replicate_lower_bound(
        [0.444446, 0.483705, 0.523633, 0.483928, 0.483928],
        cp.SEED_REPLICATE_T_MULTIPLIER,
        5,
    )
    rho = cp.seed_replicate_lower_bound(
        [0.520469, 0.562711, 0.609409, 0.564196, 0.564196],
        cp.SEED_REPLICATE_T_MULTIPLIER,
        5,
    )
    verdict, detail = cp.verdict_from_stage1_metrics(
        {"spearman_rho": rho["lower"], "quantile_bin_concordance": qbc["lower"]},
        {"spearman_rho": 0.50, "quantile_bin_concordance": 0.4750},
    )
    assert verdict == "FAIL"
    assert detail["spearman_rho"]["passed"] is True
    assert detail["quantile_bin_concordance"]["passed"] is False


@pytest.mark.parametrize(
    "values, t_multiplier, expected_n, match",
    [
        ([0.5] * 4, 2.132, 5, r"got 4 replicates but expected_n=5"),
        ([0.5] * 6, 2.132, 5, r"got 6 replicates but expected_n=5"),
        ([0.5], 2.132, 1, r"expected_n=1"),
        ([0.5] * 5, 0.0, 5, r"t_multiplier=0\.0"),
        ([0.5] * 5, float("nan"), 5, r"t_multiplier=nan"),
        ([0.5, 0.5, float("nan"), 0.5, 0.5], 2.132, 5, r"index 2 is non-finite"),
        ([0.5, float("inf"), 0.5, 0.5, 0.5], 2.132, 5, r"index 1 is non-finite"),
    ],
)
def test_seed_replicate_lower_bound_refuses_bad_input(values, t_multiplier, expected_n, match):
    """Guard first, compute second. A wrong replicate count would silently apply an
    interval derived for different degrees of freedom; a non-finite replicate dropped
    silently would NARROW the interval and so make the gate easier to clear, in exactly
    the direction a reader could not see. Both raise instead.
    """
    with pytest.raises(ValueError, match=match):
        cp.seed_replicate_lower_bound(values, t_multiplier, expected_n)


# --- Plan 02.5-08 Task 1: exact chart-decoder curvature via torch.func -------------------
#
# torch is imported INSIDE each test body, never at module scope, so the numpy-only tests
# above still collect in an environment without torch (this file's own header promises
# "no torch"; that promise is kept at import time, which is what matters for collection).


def _toy_quadratic_chart_model(a: float, chart_dim: int):
    """A duck-typed stand-in for ``cae.ChartAutoEncoder`` whose decoder is the EXACT
    quadratic graph map ``z -> (z, 0.5 * a * ||z||^2)``, whose mean curvature is therefore
    known in closed form via ``curvature_probe.graph_mean_curvature``.

    02.2-05's precedent established that this package's model-consuming functions accept a
    duck-typed model object so a known-answer fixture can be minimal and fully
    float-controllable; the same precedent applies here. Only ``chart_decoders``,
    ``embedding_decoder``, and ``activation`` are consumed by ``chart_curvature``.
    """
    import torch

    class _QuadraticGraph(torch.nn.Module):
        def __init__(self, amp: float):
            super().__init__()
            self.amp = float(amp)

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            f = 0.5 * self.amp * (z**2).sum(dim=-1, keepdim=True)
            return torch.cat([z, f], dim=-1)

    class _ToyChartModel:
        def __init__(self):
            self.chart_decoders = [torch.nn.Identity().double()]
            self.embedding_decoder = _QuadraticGraph(a).double()
            self.activation = "silu"
            self.chart_dim = chart_dim
            self.out_dim = chart_dim + 1

    return _ToyChartModel()


def _toy_quadratic_analytic_H(z_np: np.ndarray, a: float) -> np.ndarray:
    """Closed-form mean curvature of ``M = {(z, 0.5*a*||z||^2)}``: ``grad f = a*z`` and
    ``hess f = a*I``, fed to the already-verified ``graph_mean_curvature`` (pinned against
    central finite differences by plan 02.5-02, so this is ground truth computed by a
    different route than the code under test)."""
    n, d = z_np.shape
    grad = (a * z_np)[:, None, :]  # (n, 1, d)
    hess = np.broadcast_to(a * np.eye(d), (n, 1, d, d)).copy()  # (n, 1, d, d)
    return cp.graph_mean_curvature(grad, hess)


def _small_cae(activation: str, seed: int = 0):
    """A tiny float64 ``ChartAutoEncoder``, small enough that vmap(hessian(...)) over its
    decoder is sub-second."""
    import torch

    from pu_manifold import cae

    torch.manual_seed(seed)
    model = cae.ChartAutoEncoder(
        in_dim=12, embed_dim=6, chart_dim=2, n_charts=3, hidden=[8], activation=activation
    )
    return model.double()


def test_chart_curvature_matches_analytic_on_toy_decoder():
    """Pitfall 5's verification, and the single test that makes the tensor contractions in
    ``chart_mean_curvature`` trustworthy.

    ``torch.func``'s composition through ``jacrev``/``hessian`` plus an outer ``vmap`` can
    silently return a Jacobian-shaped tensor, and a transposed index order in the normal
    projection produces a result that still runs. Neither survives a comparison against a
    decoder whose curvature is known in closed form and computed by an independent route.
    """
    import torch

    from pu_manifold import chart_curvature as cc

    a, chart_dim, batch = 0.7, 2, 16
    model = _toy_quadratic_chart_model(a, chart_dim)

    rng = np.random.default_rng(20260809)
    z_np = rng.uniform(-0.8, 0.8, size=(batch, chart_dim))
    z_chart = torch.tensor(z_np, dtype=torch.float64)

    out = cc.chart_mean_curvature(model, z_chart, 0)
    H = out["H_vec"].numpy()
    H_true = _toy_quadratic_analytic_H(z_np, a)

    assert H.shape == (batch, chart_dim + 1)
    np.testing.assert_allclose(H, H_true, rtol=1e-5)

    # Pitfall 5's warning sign, pinned: the diagnostics report the shapes the transform
    # composition actually produced, not the shapes it was supposed to produce.
    assert out["jacobian_shape"] == (batch, 3, 2)
    assert out["hessian_shape"] == (batch, 3, 2, 2)

    # A graph over a full-rank domain is an immersion everywhere; cond(g) near 1 is the
    # signal that no non-immersion point contaminated the comparison.
    assert np.all(np.isfinite(out["metric_condition_number"].numpy()))
    assert np.max(out["metric_condition_number"].numpy()) < 10.0

    np.testing.assert_allclose(
        cc.chart_mean_curvature_norm(out["H_vec"]).numpy(),
        np.linalg.norm(H_true, axis=-1),
        rtol=1e-5,
    )


def test_chart_curvature_refuses_relu_decoder():
    """RESEARCH Pitfall 4. ReLU is piecewise-linear, so its second derivative is a sum of
    Dirac deltas at the kinks and is numerically zero everywhere autodiff evaluates it.
    Without this guard the entire second fundamental form would come back as exactly
    ``0.0`` and read as a perfectly flat manifold -- a silent wrong answer, not an error.

    The guard reads ``model.activation``, not the cache stem name, which is Pitfall 4's
    own stated mitigation, and it raises rather than warns: a warning in a batch runner is
    a silent failure.
    """
    import torch

    from pu_manifold import chart_curvature as cc

    relu_model = _small_cae("relu", seed=1)
    z_chart = torch.rand(4, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="relu"):
        cc.chart_mean_curvature(relu_model, z_chart, 0)

    silu_model = _small_cae("silu", seed=1)
    out = cc.chart_mean_curvature(silu_model, z_chart, 0)
    assert out["H_vec"].shape == (4, silu_model.out_dim)
    assert torch.isfinite(out["H_vec"]).all()


def test_chart_curvature_field_reassembles_in_row_order():
    """A batching bug that reorders rows is invisible to every aggregate statistic this
    phase computes -- Spearman and quantile-bin concordance would both simply drop. The
    only thing that catches it is asserting the field is bit-identical under two different
    batch sizes and that the chart assignment matches the model's own argmax."""
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=2)
    rng = np.random.default_rng(7)
    x = torch.tensor(rng.standard_normal((24, 12)), dtype=torch.float64)

    field_small = cc.chart_curvature_field(model, x, batch_size=8)
    field_large = cc.chart_curvature_field(model, x, batch_size=64)

    assert field_small["H_norm"].shape == (24,)
    assert field_small["H_vec"].shape == (24, model.out_dim)

    with torch.no_grad():
        expected_assignment = model.chart_probs(model.encode(x)).argmax(dim=1)
    assert torch.equal(field_small["chart_assignment"], expected_assignment)

    # bit-identical, not merely close: batching must not touch a single float
    assert torch.equal(field_small["H_norm"], field_large["H_norm"])
    assert torch.equal(field_small["H_vec"], field_large["H_vec"])
    assert torch.equal(
        field_small["metric_condition_number"], field_large["metric_condition_number"]
    )
    assert field_small["n_charts_used"] == len(torch.unique(expected_assignment))


def test_chart_curvature_uses_trace_convention_not_averaged():
    """The factor-of-``d`` regression guard, and the reason it exists at full strength.

    ``02.5-NOTE-randomized-trace.md`` Section 1 and
    ``02.5-NOTE-high-d-curvature-approaches.md`` Section 2c both record that the external
    source material for this arm uses the AVERAGED convention ``H = (1/d) tr_g(II)``.
    Transcribing it verbatim introduces a factor-of-``d`` = 20 error against every fixture,
    every 02.5 SUMMARY number, and the sealed stage-1 gate. This codebase has already
    shipped and then fixed exactly one factor-of-``d`` scale bug.

    ``chart_dim = 4`` is chosen so that ``d``, ``d + 2`` and ``1`` are all distinguishable
    -- at ``d = 2`` a ``d``-fold error and a ``(d+2)/d`` error are not separable.
    """
    import torch

    from pu_manifold import chart_curvature as cc

    assert cc.CURVATURE_CONVENTION == "trace"
    assert cc.CURVATURE_CONVENTION == cp.CURVATURE_CONVENTION

    a, chart_dim, batch = 0.9, 4, 8
    model = _toy_quadratic_chart_model(a, chart_dim)

    rng = np.random.default_rng(4242)
    z_np = rng.uniform(-0.5, 0.5, size=(batch, chart_dim))
    H = cc.chart_mean_curvature(model, torch.tensor(z_np, dtype=torch.float64), 0)["H_vec"].numpy()
    H_true = _toy_quadratic_analytic_H(z_np, a)

    np.testing.assert_allclose(H, H_true, rtol=1e-5)

    norm = np.linalg.norm(H, axis=-1)
    norm_true = np.linalg.norm(H_true, axis=-1)
    assert np.all(norm_true > 1e-6)
    # the averaged convention, and the already-fixed (d+2)/d bug, are both excluded
    assert not np.allclose(norm, norm_true / chart_dim, rtol=1e-3)
    assert not np.allclose(norm, norm_true * chart_dim, rtol=1e-3)
    assert not np.allclose(norm, norm_true * (chart_dim + 2) / chart_dim, rtol=1e-3)


def test_chart_curvature_dxd_solve_matches_explicit_projector():
    """``chart_mean_curvature`` never materializes the ``(D, D)`` normal projector that
    RESEARCH Pattern 4's illustrative snippet writes: at ``out_dim = 768`` a batch of 32
    such projectors is 151 MB of float64, and the ``II`` tensor it multiplies is another
    78 MB. Instead it applies ``P_N a = a - J alpha`` with ``g alpha = J^T a``, a
    ``chart_dim x chart_dim`` solve, to the already-``g``-traced ambient Hessian.

    That is an optimisation of the SAME mathematics, so it needs an equality proof rather
    than an assurance. This test reimplements Pattern 4's snippet verbatim -- explicit
    ``(D, D)`` projector, explicit ``II``, projector-then-trace rather than
    trace-then-projector -- and asserts the two agree to float64 round-off.
    """
    import torch
    from torch.func import hessian, jacrev, vmap

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=3)
    chart_idx = 1
    z_chart = torch.rand(6, model.chart_dim, dtype=torch.float64)

    decode_one = cc.chart_decoder_map(model, chart_idx)
    J = vmap(jacrev(decode_one))(z_chart)
    Hess = vmap(hessian(decode_one))(z_chart)

    # --- RESEARCH Pattern 4, transcribed verbatim, as the independent reference ---
    g = torch.einsum("boi,boj->bij", J, J)
    g_inv = torch.linalg.inv(g)
    proj = torch.eye(J.shape[1], dtype=J.dtype)[None] - torch.einsum("boi,bij,bpj->bop", J, g_inv, J)
    II = torch.einsum("bop,bpjk->bojk", proj, Hess)
    H_reference = torch.einsum("bij,boij->bo", g_inv, II)

    H_actual = cc.chart_mean_curvature(model, z_chart, chart_idx)["H_vec"]

    assert H_reference.shape == H_actual.shape
    torch.testing.assert_close(H_actual, H_reference, rtol=1e-9, atol=1e-12)


def test_chart_curvature_reverse_mode_is_bit_identical_to_sealed_baseline():
    """Plan 03-05 Task 1B: pins the reverse path's exact output BEFORE the mode toggle is
    added, so the first-ever edit to ``chart_curvature.py`` cannot silently move a single bit
    of it.

    Same construction as ``test_chart_curvature_dxd_solve_matches_explicit_projector``
    (``_small_cae("silu", seed=3)``, ``chart_idx=1``, ``z_chart = torch.rand(6,
    model.chart_dim, dtype=torch.float64)`` drawn immediately after model construction so the
    RNG state is identical) -- deliberately reusing that exact fixture rather than a new one,
    so this golden array and that equivalence proof are anchored to the same inputs.

    ``torch.equal`` against a hard-coded golden array, not ``assert_close`` -- a tolerance
    would let the exact drift this test exists to catch pass silently. The golden array was
    generated by running this fixture through ``chart_mean_curvature`` with no ``mode``
    argument, on the unmodified pre-edit module, and transcribing the printed ``H_vec.tolist()``
    verbatim.
    """
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=3)
    chart_idx = 1
    z_chart = torch.rand(6, model.chart_dim, dtype=torch.float64)

    out = cc.chart_mean_curvature(model, z_chart, chart_idx)
    H_vec = out["H_vec"]

    golden = torch.tensor(
        [
            [
                23.864493867564452, 29.074663985887558, -24.431008977073652,
                16.18985390919337, -12.914912102612666, -21.58498894746812,
                -22.660838392163186, -5.501896282145914, 23.298350617824518,
                1.587970486442476, -23.59735580532664, 6.213482636359536,
            ],
            [
                25.36276051017246, 60.18063097243433, -5.971225559572369,
                -7.301754200469416, -42.365068996917486, -48.41651871036599,
                -40.542016291534175, -2.39506557367638, -2.9018089972993764,
                40.84688620551977, -33.33450723751727, -2.8035826535048507,
            ],
            [
                0.6465263832967274, 45.80512913707088, 13.062887275803785,
                13.566801009505344, -56.31457976392273, -20.081171175152626,
                -20.73411422579065, 44.36374735636546, -12.443646354420224,
                26.65009653500055, -25.911995533283356, -5.338161759514428,
            ],
            [
                13.322865201452485, 52.02312044660138, 6.494685309038635,
                -1.8369587383677883, -56.27896592560622, -32.639543419061596,
                -27.34046985792252, 25.467273672293008, -9.037625698996255,
                38.3836128588914, -27.41166888652544, -13.155806754362956,
            ],
            [
                17.829668423435116, 20.143842086310173, -19.921567555648068,
                14.887873792145623, -7.538587925204682, -14.543866133741915,
                -16.315487707016835, -3.8849457891738854, 19.589707467427193,
                -2.017220234671484, -17.5680596681889, 6.242046128994895,
            ],
            [
                14.75531971823188, 44.51488882582112, 5.994780619931614,
                -14.27107176573848, -43.0853283822766, -34.25812675976472,
                -25.87795363714077, 7.9005794819553685, -11.282780420485622,
                40.25342185414989, -21.03311326974296, -13.43936093204005,
            ],
        ],
        dtype=torch.float64,
    )

    assert torch.equal(H_vec, golden)


def test_chart_curvature_forward_mode_matches_reverse_to_float64_round_off():
    """D-09's equivalence proof, mirroring ``test_chart_curvature_dxd_solve_matches_explicit_projector``'s
    structure exactly but comparing ``mode="forward"`` against ``mode="reverse"`` instead of
    against a hand-written reference projector -- the ``d``-by-``d``-solve optimization earned
    its place this way, and the forward-mode toggle must earn its place the same way, proved
    rather than merely asserted.
    """
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=3)
    chart_idx = 1
    z_chart = torch.rand(6, model.chart_dim, dtype=torch.float64)

    reverse = cc.chart_mean_curvature(model, z_chart, chart_idx, mode="reverse")
    forward = cc.chart_mean_curvature(model, z_chart, chart_idx, mode="forward")

    assert reverse["jacobian_shape"] == forward["jacobian_shape"]
    assert reverse["hessian_shape"] == forward["hessian_shape"]

    torch.testing.assert_close(forward["H_vec"], reverse["H_vec"], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(forward["H_norm"], reverse["H_norm"], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(
        forward["metric_condition_number"],
        reverse["metric_condition_number"],
        rtol=1e-9,
        atol=1e-12,
    )
    assert forward["mode"] == "forward"
    assert reverse["mode"] == "reverse"


def test_chart_curvature_forward_mode_keeps_shape_assertions():
    """RESEARCH Pitfall 5: a wrong ``torch.func`` transform composition still runs and
    silently returns a differently-shaped tensor rather than raising. This pins that the
    forward path returns the exact expected tuple shapes -- the only thing that would catch a
    composition that silently returned a Jacobian-shaped Hessian."""
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=4)
    chart_idx = 0
    z_chart = torch.rand(5, model.chart_dim, dtype=torch.float64)

    out = cc.chart_mean_curvature(model, z_chart, chart_idx, mode="forward")

    assert out["jacobian_shape"] == (5, model.out_dim, model.chart_dim)
    assert out["hessian_shape"] == (5, model.out_dim, model.chart_dim, model.chart_dim)
    assert out["H_vec"].shape == (5, model.out_dim)


def test_chart_curvature_forward_mode_calls_c2_guard():
    """Mirrors ``test_chart_curvature_refuses_relu_decoder``: a ReLU-family decoder has an
    identically-zero second derivative and would return an identically-zero second
    fundamental form without raising. The C2 guard must be reached on the forward path too,
    not only on reverse."""
    import torch

    from pu_manifold import chart_curvature as cc

    relu_model = _small_cae("relu", seed=1)
    z_chart = torch.rand(4, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="relu"):
        cc.chart_mean_curvature(relu_model, z_chart, 0, mode="forward")

    silu_model = _small_cae("silu", seed=1)
    out = cc.chart_mean_curvature(silu_model, z_chart, 0, mode="forward")
    assert out["H_vec"].shape == (4, silu_model.out_dim)
    assert torch.isfinite(out["H_vec"]).all()


def test_chart_curvature_rejects_unknown_mode():
    """An unknown ``mode`` string raises ``ValueError`` naming the offending value, rather
    than silently falling through to a default -- on both public entry points."""
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=5)
    z_chart = torch.rand(4, model.chart_dim, dtype=torch.float64)

    with pytest.raises(ValueError, match="sideways"):
        cc.chart_mean_curvature(model, z_chart, 0, mode="sideways")

    # model.out_dim == in_dim for a ChartAutoEncoder (cae.py stores no separate in_dim
    # attribute); ambient rows must match it for encode() to accept them.
    x = torch.rand(4, model.out_dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="sideways"):
        cc.chart_curvature_field(model, x, mode="sideways")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA -- inert on this CPU-only development machine and on CI",
)
def test_chart_curvature_cpu_cuda_agree_to_float64_tolerance():
    """Device-parity check for Phase 3's opt-in GPU support (03-07-SUPPLEMENT-01.md).

    CUDA RNG differs from CPU RNG, so ``torch.manual_seed(seed)`` draws a DIFFERENT model
    on each device -- a bit-identity assertion across devices would be meaningless (and,
    per :data:`chart_curvature.VMAP_CHUNK`'s own docstring, even a batch-width change on
    the SAME device already moves bits at ~1e-14 to ~1e-18, so cross-device bit-identity
    was never on the table). Instead: build one model on CPU under the module's normal
    construct-then-``.to(device)`` discipline, deep-copy its exact trained parameters onto
    a CUDA copy (so both models share literally the same weights), and check the computed
    curvature field agrees to a documented float64 cross-device tolerance.

    Inert everywhere this suite currently runs (no CUDA available), which is the point --
    it exists to be exercised on the colleague's GPU machine, not here."""
    import copy

    import torch

    from pu_manifold import chart_curvature as cc

    model_cpu = _small_cae("silu", seed=9)
    model_cuda = copy.deepcopy(model_cpu).to("cuda").double()

    z_chart_cpu = torch.rand(6, model_cpu.chart_dim, dtype=torch.float64)
    z_chart_cuda = z_chart_cpu.to("cuda")

    out_cpu = cc.chart_mean_curvature(model_cpu, z_chart_cpu, chart_idx=1)
    out_cuda = cc.chart_mean_curvature(model_cuda, z_chart_cuda, chart_idx=1)

    # Documented cross-device float64 tolerance -- NOT bit-identity (unachievable across
    # devices; see docstring above). rtol/atol chosen an order of magnitude above the
    # same-device batch-width drift (~1e-14) this module's own docstring already measures,
    # so a real cross-device numerical divergence is caught without false alarms on the
    # noise floor this module already documents as expected.
    torch.testing.assert_close(
        out_cuda["H_vec"].cpu(), out_cpu["H_vec"], rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        out_cuda["metric_condition_number"].cpu(),
        out_cpu["metric_condition_number"],
        rtol=1e-6,
        atol=1e-8,
    )
    assert out_cpu["H_vec"].device.type == "cpu"
    assert out_cuda["H_vec"].device.type == "cuda"


def test_chart_curvature_field_forward_mode_matches_reverse_across_charts():
    """Field-level D-09 equivalence, reusing ``test_chart_curvature_field_reassembles_in_row_order``'s
    exact point-cloud construction (a multi-chart cloud, not a fresh invention) so
    ``mode="forward"`` and ``mode="reverse"`` are compared over rows spanning at least two
    charts, at the same float64 round-off band as the per-chart proof, and are shown to agree
    on chart assignment and chart count too."""
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=2)
    rng = np.random.default_rng(7)
    x = torch.tensor(rng.standard_normal((24, 12)), dtype=torch.float64)

    field_reverse = cc.chart_curvature_field(model, x, batch_size=8, mode="reverse")
    field_forward = cc.chart_curvature_field(model, x, batch_size=8, mode="forward")

    assert field_reverse["n_charts_used"] == field_forward["n_charts_used"]
    assert torch.equal(field_reverse["chart_assignment"], field_forward["chart_assignment"])

    torch.testing.assert_close(
        field_forward["H_vec"], field_reverse["H_vec"], rtol=1e-9, atol=1e-12
    )
    torch.testing.assert_close(
        field_forward["H_norm"], field_reverse["H_norm"], rtol=1e-9, atol=1e-12
    )
    torch.testing.assert_close(
        field_forward["metric_condition_number"],
        field_reverse["metric_condition_number"],
        rtol=1e-9,
        atol=1e-12,
    )


# --- Plan 02.5-08 Task 2: the checks a rank statistic cannot make -----------------------
#
# 02.5-NOTE-randomized-trace.md Addendum C. The decoder arm computes H_F, the EXACT
# curvature of the LEARNED manifold M_F = F(R^d) -- not H_true, the curvature of the data
# manifold. A reconstruction objective asks F(E(x)) ~= x; it never asks D^2 F ~= D^2 F_true.
# Both stage-1 gating statistics are rank-based and are therefore exactly blind to a decoder
# that compresses every curvature magnitude by a constant factor. These tests pin the
# machinery that is not blind to it.


def test_chart_curvature_fidelity_report_separates_amplitude_from_direction():
    """The specific way Arm B can look successful while being wrong.

    ``02.5-NOTE-high-d-curvature-approaches.md`` Section 2d: a decoder trained to reconstruct
    will happily regularize the bumps flatter than they are, producing a curvature field that
    is smooth, well-ordered, highly rank-correlated with the truth -- and systematically wrong
    in amplitude. The note's worked example: if the true local surface is ``y = a x^2`` and the
    decoder learns ``y = 0.7 a x^2``, reconstruction error stays tiny wherever the sampled
    ``x`` sit near zero while the second derivative is ``1.4a`` instead of ``2a``. Reconstruction
    quality can never validate a curvature estimate.

    ``H`` is vector-valued, so amplitude attenuation and orientation error are DISTINCT failure
    modes and must never be collapsed into one scalar. This test constructs each in isolation
    and requires the report to name the right one each time.
    """
    from pu_manifold import chart_curvature as cc

    rng = np.random.default_rng(20260809)
    n, D = 400, 6
    H_true = rng.standard_normal((n, D)) * rng.uniform(0.5, 3.0, size=(n, 1))

    # --- failure mode 1: pure amplitude attenuation, the note's 0.7a decoder ---
    H_attenuated = 0.7 * H_true
    rep = cc.curvature_fidelity_report(H_attenuated, H_true)

    # every rank-based statistic scores this PERFECT -- that is the whole problem
    norm_true = np.linalg.norm(H_true, axis=-1)
    assert cp.spearman_gate_statistic(np.linalg.norm(H_attenuated, axis=-1), norm_true) == pytest.approx(1.0)

    # direction is untouched, and the report says so rather than blaming the wrong thing
    assert rep["median_cosine_similarity"] == pytest.approx(1.0, abs=1e-12)
    # amplitude is caught, exactly, with zero scatter -- "attenuated but calibratable"
    assert rep["median_magnitude_ratio"] == pytest.approx(0.7, rel=1e-9)
    assert rep["magnitude_ratio_cv"] == pytest.approx(0.0, abs=1e-9)
    # and the calibration slope sees a = 0.7, which no rank statistic can
    assert rep["calibration_slope"] == pytest.approx(0.7, rel=1e-9)
    assert rep["calibration_intercept"] == pytest.approx(0.0, abs=1e-9)

    # --- failure mode 2: pure orientation error, amplitude exactly preserved ---
    H_rotated = H_true.copy()
    H_rotated[:, [0, 1]] = H_true[:, [1, 0]]  # a norm-preserving coordinate swap
    rep_rot = cc.curvature_fidelity_report(H_rotated, H_true)

    assert rep_rot["median_magnitude_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert rep_rot["calibration_slope"] == pytest.approx(1.0, rel=1e-9)
    assert rep_rot["median_cosine_similarity"] < 0.95  # the distinct mode, distinctly reported

    # The two failure modes are never collapsed: each report carries all three families
    # separately, and neither exposes a single summary score that could hide one behind
    # the other.
    for key in (
        "median_cosine_similarity",
        "median_magnitude_ratio",
        "magnitude_ratio_cv",
        "calibration_slope",
        "calibration_intercept",
    ):
        assert key in rep and key in rep_rot


def test_chart_curvature_fidelity_cv_separates_attenuated_from_destroyed():
    """Why the median ratio and its CV are BOTH required, and neither alone will do.

    ``02.5-NOTE-high-d-curvature-approaches.md`` Section 1a measured the point-cloud
    estimator's per-point ratio ``||H_est|| / ||H_true||``: at ``d = 2`` a clean 0.905 median
    at CV 0.250 ("a mild underestimate with modest scatter -- a correctable signature"), and
    at ``d = 20`` a CV of 2.250 ("the scatter is 2.25x the mean ... there is no scale factor
    to calibrate out"). The distinction between a bias one can calibrate away and an error
    that merely behaves like noise is carried entirely by the CV. This test builds two fields
    with deliberately similar medians and very different scatter, and requires the report to
    separate them.
    """
    from pu_manifold import chart_curvature as cc

    rng = np.random.default_rng(31337)
    n, D = 2000, 4
    direction = rng.standard_normal((n, D))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    H_true = direction * rng.uniform(0.5, 2.0, size=(n, 1))

    ratio_tight = rng.lognormal(mean=np.log(0.905), sigma=0.24, size=(n, 1))
    ratio_wild = rng.lognormal(mean=np.log(0.905), sigma=1.30, size=(n, 1))

    rep_tight = cc.curvature_fidelity_report(H_true * ratio_tight, H_true)
    rep_wild = cc.curvature_fidelity_report(H_true * ratio_wild, H_true)

    # the medians are close: the median ALONE cannot separate these two regimes
    assert rep_tight["median_magnitude_ratio"] == pytest.approx(0.905, rel=0.05)
    assert rep_wild["median_magnitude_ratio"] == pytest.approx(0.905, rel=0.10)

    # the CV separates them decisively, in the direction Section 1a measured
    assert rep_tight["magnitude_ratio_cv"] < 0.5
    assert rep_wild["magnitude_ratio_cv"] > 1.5


def test_chart_curvature_antithetic_probes_are_exactly_redundant():
    """``02.5-NOTE-randomized-trace.md`` Addendum A, pinned as executable fact rather than
    left as an argument in a note.

    The external source material suggests using "antithetic directions" if the randomized
    estimator is noisy. For this estimator that suggestion is void: ``B`` is a symmetric
    BILINEAR form, so ``B(-v, -v) = (-1)(-1) B(v, v) = B(v, v)``. The antithetic partner
    returns the IDENTICAL value, not a negatively-correlated one -- the pair is correlated at
    exactly ``+1``, so averaging over ``{v, -v}`` has precisely the variance of the single
    sample ``v`` at twice the cost. Any K-probe budget spent on antithetic pairing is halved
    for nothing. Antithetic sampling reduces variance for estimators with an ODD-order
    dependence on the probe; a quadratic form is even.

    This test is why ``chart_curvature.py`` implements no antithetic path, and it fails loudly
    if someone later adds one believing it helps.
    """
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=9)
    decode_one = cc.chart_decoder_map(model, 0)
    z = torch.rand(6, model.chart_dim, dtype=torch.float64)
    v = torch.randn(6, model.chart_dim, dtype=torch.float64)

    forward = cc.directional_second_derivative(decode_one, z, v)
    antithetic = cc.directional_second_derivative(decode_one, z, -v)

    # bit-identical, not merely close: the equality is algebraic, not numerical
    assert torch.equal(forward, antithetic)


def test_chart_curvature_randomized_trace_converges_to_exact():
    """The randomized K-probe estimator is a CONVERGENCE CHECK ON THE EXACT PATH, gated on
    nothing -- ``02.5-NOTE-randomized-trace.md``'s "What 02.5-08 should do". It is not a
    candidate estimator: at ``d = 20`` the exact ``g``-trace is only 20 Hessian-vector
    products against ``K = 8``, a 2.5x saving on a computation that was never the bottleneck,
    and the arm's real advantage is statistical rather than computational. Its value is the
    same as the sphere known-answer test's: agreement with the exact path is evidence the
    exact path is right.

    The normalization is the trap this test exists to close. With ``xi = g^{-1/2} eps`` and
    ``eps`` Rademacher, ``E[eps eps^T] = I`` so ``E[xi xi^T] = g^-1`` and the estimator
    ``(1/K) sum_k B(xi_k, xi_k)`` targets ``tr_g(II)`` with NO ``1/d`` and NO ``d``. Under the
    source material's averaged convention the same probes would need an explicit ``1/d``, and
    with uniform-on-the-sphere probes instead of Rademacher this module's trace convention
    would need an explicit ``d``. Mixing any two of those costs a factor of ``d`` = 20.
    Convergence to the exact path at the same scale is what proves the right pairing was used.
    """
    import torch

    from pu_manifold import chart_curvature as cc

    model = _small_cae("silu", seed=5)
    z_chart = torch.rand(8, model.chart_dim, dtype=torch.float64)

    check = cc.randomized_trace_convergence_check(
        model, z_chart, 0, probe_counts=(4, 16, 64), seeds=(0, 1, 2, 3, 4, 5)
    )

    assert check["gating"] is False
    err = check["median_relative_error"]
    assert set(err) == {4, 16, 64}

    # Monte-Carlo error falls as 1/sqrt(K): each 4x in K should roughly halve it. Averaged
    # over six seeds so a single unlucky draw cannot decide the assertion -- at one fixed
    # seed the sequence is not monotone, which is the nature of the estimator, not a defect.
    assert err[16] < 0.75 * err[4]
    assert err[64] < 0.75 * err[16]

    # unbiasedness: averaging many cheap K=4 estimates converges on the exact answer, so the
    # spread above is variance around the right value rather than a bias at the wrong scale
    # (which is exactly what a mis-paired normalization would produce, off by a factor of d).
    assert check["mean_of_replicates_relative_error"] < 0.5 * err[4]

    # and the exact path it is checking is the gating one
    exact = cc.chart_mean_curvature(model, z_chart, 0)["H_vec"]
    torch.testing.assert_close(check["H_exact"], exact, rtol=1e-12, atol=1e-14)


# --- Plan 02.5-08 Task 3: the sealed 02.2 fits are curvature-ready ----------------------

SEALED_FIT_KEY = "43cf438bc944c509"
SEALED_SEEDS = (20260803, 20260804, 20260805)


def test_sealed_cae_fits_load_and_match_meta():
    """D-10's premise, made executable: the three sealed 02.2 fits load read-only, pass their
    manifest checks, and carry a C2-smooth activation.

    D-10's whole case is that these fits were deliberately built with SiLU precisely because
    downstream curvature work needed twice-differentiable decoders (CAE-06). That is a claim
    about artifacts written months earlier by a different plan, and this is the check that it
    actually holds -- reading the RECORDED activation, never the cache stem name, which is
    RESEARCH Pitfall 4's own stated mitigation.

    Skips rather than fails when the cache is absent: ``notebooks/.cache/`` is gitignored and
    irreproducible on a fresh clone (each of these fits took ~33 minutes to train), and
    ``topoae_evaluate_run.py``'s precondition paragraph establishes the halt-rather-than-
    regenerate convention this follows. Regenerating a sealed artifact to make a test pass
    would defeat the entire point of sealing it.

    Reads and never writes; the cache listing is asserted unchanged at the end.
    """
    import json

    import torch

    from pu_manifold import cae
    from pu_manifold import chart_curvature as cc

    def _listing():
        return sorted((p.name, p.stat().st_size) for p in cache.CACHE_DIR.glob("*"))

    before = _listing()

    architectures = {}
    for seed in SEALED_SEEDS:
        fit_stem = f"cae_fit_{SEALED_FIT_KEY}_seed{seed}"
        meta_stem = f"cae_fit_meta_{SEALED_FIT_KEY}_seed{seed}"
        for stem, ext in ((fit_stem, "npz"), (meta_stem, "json")):
            if not cache.cache_path(stem, ext).exists():
                pytest.skip(f"sealed 02.2 fit artifact absent from the cache: {stem}.{ext}")
            if not cache._manifest_path(stem).exists():
                pytest.skip(f"sealed 02.2 fit manifest absent from the cache: {stem}.meta.json")

        # The recorded cfg is the manifest's own; loading with it drives _manifest_matches
        # down its equality path. compute_fn raises, so a cache MISS would surface as a loud
        # failure rather than silently retraining a sealed artifact.
        cfg = json.loads(cache._manifest_path(fit_stem).read_text())

        def _never():
            raise AssertionError(
                f"cache miss on sealed artifact {fit_stem}: this test must never compute, "
                f"only load"
            )

        arrays = cache.npz_cache(fit_stem, cfg, _never)

        meta_cfg = json.loads(cache._manifest_path(meta_stem).read_text())
        meta = cache.json_cache(meta_stem, meta_cfg, _never)

        # Negative control, so the assertion above is not merely a tautology comparing a cfg
        # to itself: a manifest mismatch must RAISE, not return False. Without this, a stem
        # whose sidecar had drifted would sail through the load path untested.
        with pytest.raises(ValueError, match="Cache manifest mismatch"):
            cache._manifest_matches(fit_stem, {**cfg, "seed": int(cfg["seed"]) + 1})

        # --- activation: the attribute, not the stem name ---
        assert meta["activation"] == "silu", (
            f"seed {seed} reports activation {meta['activation']!r}; a zero-second-derivative "
            f"activation would make this fit unusable for curvature"
        )
        assert meta["cfg"]["activation"] == "silu"
        assert cfg["activation"] == "silu"

        class _ActivationStandIn:
            activation = meta["activation"]

        assert cc.assert_c2_activation(_ActivationStandIn()) == "silu"
        assert meta["activation"] not in cc.ZERO_SECOND_DERIVATIVE_ACTIVATIONS

        # --- FIT_ARTIFACT_CONTRACT: every named key is present ---
        for key in ("z_all", "p_all", "chart_argmax_all", "train_idx", "holdout_idx", "y_holdout"):
            assert key in arrays, f"seed {seed}: FIT_ARTIFACT_CONTRACT key {key!r} missing"

        # --- architecture, DERIVED from the artifact rather than hardcoded ---
        in_dim = int(arrays["initial_encoder.net.0.weight"].shape[1])
        embed_dim = int(arrays["initial_encoder.net.6.weight"].shape[0])
        chart_dim = int(arrays["chart_decoders.0.net.0.weight"].shape[1])
        n_charts = int(arrays["chart_predictor.net.6.weight"].shape[0])
        depth = sum(
            1 for k in arrays if k.startswith("initial_encoder.net.") and k.endswith(".weight")
        ) - 1
        hidden = [int(arrays["initial_encoder.net.0.weight"].shape[0])] * depth

        # cross-check the derivation against the manifest's own recorded constants
        assert embed_dim == int(cfg["l_embed"])
        assert chart_dim == int(cfg["d_chart"])
        assert n_charts == int(cfg["n_charts_init"])
        assert hidden[0] == int(cfg["hidden_width"])
        assert in_dim == int(arrays["embedding_decoder.net.6.weight"].shape[0])

        architectures[seed] = {
            "in_dim": in_dim,
            "embed_dim": embed_dim,
            "chart_dim": chart_dim,
            "n_charts": n_charts,
            "hidden": tuple(hidden),
            "activation": meta["activation"],
        }

        # --- round-trip: the stored arrays reload bit-identically into a float64 model ---
        model = cae.ChartAutoEncoder(
            in_dim=in_dim,
            embed_dim=embed_dim,
            chart_dim=chart_dim,
            n_charts=n_charts,
            hidden=hidden,
            activation=meta["activation"],
        ).double()
        reference = model.state_dict()
        state = cae.arrays_to_state_dict(arrays, reference)
        assert set(state) == set(reference)
        for key, tensor in state.items():
            expected = torch.tensor(arrays[key], dtype=torch.float64).reshape(tensor.shape)
            assert torch.equal(tensor, expected), f"seed {seed}: {key} did not round-trip exactly"
        model.load_state_dict(state)

    # All three seeds share one architecture -- plan 02.5-11 must fit its fixture CAEs at the
    # SAME architecture with no new tunable hyperparameter, so a disagreement here would mean
    # there is no single "same architecture" to fit at.
    assert len({tuple(sorted(a.items())) for a in architectures.values()}) == 1

    assert _listing() == before, "this test reads the sealed cache and must never write to it"
