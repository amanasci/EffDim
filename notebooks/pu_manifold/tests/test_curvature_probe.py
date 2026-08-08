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
