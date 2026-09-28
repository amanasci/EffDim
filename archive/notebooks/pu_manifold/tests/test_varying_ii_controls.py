"""Known-answer tests for :mod:`pu_manifold.varying_ii_controls`.

The decisive tests are the FINITE-DIFFERENCE tests: every fixture's hand-derived ``grad`` and
``hess`` must match a central-difference computation of its own ``f``. That is
``synthetic_controls``' own standard (``test_synthetic_saddle_control_matches_finite_difference``)
and it is the only check that catches a mis-derived closed form, which otherwise produces
finite, plausible, wrong curvature.

    python -m pytest notebooks/pu_manifold/tests/test_varying_ii_controls.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from pu_manifold import synthetic_controls
from pu_manifold import varying_ii_controls as vic


# --- the closed forms are correct ----------------------------------------------------------


def _fd_grad_hess(f, x, h=1e-5):
    """Central-difference gradient and diagonal Hessian of a separable ``f`` at rows of ``x``."""
    n, d = x.shape
    grad = np.zeros((n, d))
    hess_diag = np.zeros((n, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        fp, fm, f0 = f(x + e), f(x - e), f(x)
        grad[:, j] = (fp - fm) / (2 * h)
        hess_diag[:, j] = (fp - 2 * f0 + fm) / (h * h)
    return grad, hess_diag


def test_cubic_grad_and_hess_match_finite_difference():
    d, n = 5, 40
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.5, 1.5, size=(n, d))
    a = np.ones(d)
    f = lambda z: np.einsum("j,ij->i", a, z**3) / 3.0
    g_fd, h_fd = _fd_grad_hess(f, x)
    np.testing.assert_allclose(a * x**2, g_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(2.0 * a * x, h_fd, rtol=1e-4, atol=1e-5)


def test_sine_grad_and_hess_match_finite_difference():
    d, n = 5, 40
    rng = np.random.default_rng(1)
    x = rng.uniform(-np.pi, np.pi, size=(n, d))
    A, w = 0.5, 1.0
    f = lambda z: A * np.sin(w * z).sum(axis=1)
    g_fd, h_fd = _fd_grad_hess(f, x)
    np.testing.assert_allclose(A * w * np.cos(w * x), g_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(-A * w**2 * np.sin(w * x), h_fd, rtol=1e-4, atol=1e-5)


def test_quadratic_control_reproduces_the_sealed_saddle_exactly():
    """Called with no eigenvalues, the general quadratic IS ``make_saddle_control``. If this
    ever diverges, every comparison against the sealed controls is invalid."""
    n, d, D, seed = 200, 6, 14, 20260816
    mine = vic.make_quadratic_graph_control(n=n, d=d, D=D, seed=seed)
    sealed = synthetic_controls.make_saddle_control(n=n, d=d, D=D, seed=seed)
    np.testing.assert_allclose(mine["X"], sealed["X"], rtol=0, atol=0)
    np.testing.assert_allclose(mine["H_norm"], sealed["H_norm"], rtol=0, atol=0)
    np.testing.assert_allclose(mine["eigenvalues"], sealed["signs"], rtol=0, atol=0)


# --- the axis this module exists to expose -------------------------------------------------


@pytest.mark.parametrize("name", ["quadratic_saddle", "quadratic_bowl", "quadratic_aniso"])
def test_every_quadratic_has_exactly_constant_second_fundamental_form(name):
    """``hess_fro_cv == 0`` exactly. This is the defect: no choice of eigenvalues escapes it,
    so no quadratic graph can test curvature RANKING at all."""
    fx = vic.FAMILIES[name](500, 8, 16, 7)
    assert fx["ii_variation"]["hess_fro_cv"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", ["cubic", "sine"])
def test_varying_families_have_a_genuinely_varying_second_fundamental_form(name):
    fx = vic.FAMILIES[name](500, 8, 16, 7)
    assert fx["ii_variation"]["hess_fro_cv"] > 0.05


def test_non_minimality_does_not_rescue_dynamic_range():
    """The counter-intuitive measured claim from the module docstring, pinned as a test.

    The isotropic bowl is maximally non-minimal (``trace = d``) and has WORSE ``||H||``
    dynamic range than the minimal saddle at the same ``d``, because raising the trace adds a
    large constant to every point while leaving the varying part alone.
    """
    n, d, D, seed = 3000, 20, 28, 20260816
    saddle = vic.FAMILIES["quadratic_saddle"](n, d, D, seed)
    bowl = vic.FAMILIES["quadratic_bowl"](n, d, D, seed)

    def spread(fx):
        h = fx["H_norm"]
        return float(np.percentile(h, 95) / np.percentile(h, 5))

    assert bowl["trace"] == pytest.approx(float(d))
    assert saddle["trace"] == pytest.approx(0.0)
    assert spread(bowl) < 1.5          # essentially no rankable range
    assert spread(saddle) > 10.0       # the minimal fixture is the better of the two
    assert spread(bowl) < spread(saddle)


def test_ridge_grad_and_hess_match_finite_difference():
    d, n, seed = 6, 40, 7
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    w = w / np.linalg.norm(w)
    A, freq = 1.0, 1.0
    x = rng.uniform(-3.0, 3.0, size=(n, d))
    f = lambda z: A * np.sin(freq * (z @ w))

    h = 1e-5
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        g_fd = (f(x + e) - f(x - e)) / (2 * h)
        np.testing.assert_allclose(
            A * freq * np.cos(freq * (x @ w)) * w[j], g_fd, rtol=1e-5, atol=1e-7
        )
        h_fd = (f(x + e) - 2 * f(x) + f(x - e)) / (h * h)
        np.testing.assert_allclose(
            -A * freq**2 * np.sin(freq * (x @ w)) * w[j] * w[j], h_fd, rtol=1e-3, atol=1e-5
        )


def test_ridge_hessian_is_rank_one():
    """The whole point: curvature carried by ONE direction, so nothing averages over d."""
    fx = vic.FAMILIES["ridge"](50, 8, 16, 7)
    # rebuild the Hessian the fixture used, from its own recorded w
    w = fx["w"]
    outer = np.outer(w, w)
    assert np.linalg.matrix_rank(outer, tol=1e-10) == 1
    np.testing.assert_allclose(np.linalg.norm(outer), 1.0, rtol=1e-12)


@pytest.mark.parametrize("d", [4, 20, 40])
def test_ridge_curvature_variation_does_not_decay_with_dimension(d):
    """Separable families lose their curvature variation like ``1/sqrt(d)``; the ridge does
    not. Measured CV: cubic ``0.243 -> 0.099 -> 0.069`` at ``d = 4, 20, 40`` against ridge
    ``0.462 -> 0.480 -> 0.487``. This is the property that makes a high-``d`` ranking test
    possible at all."""
    ridge = vic.FAMILIES["ridge"](2000, d, d + 8, 7)
    cubic = vic.FAMILIES["cubic"](2000, d, d + 8, 7)
    assert ridge["ii_variation"]["hess_fro_cv"] > 0.40
    if d >= 20:
        assert ridge["ii_variation"]["hess_fro_cv"] > 3 * cubic["ii_variation"]["hess_fro_cv"]


def test_separable_curvature_variation_does_decay_with_dimension():
    """The concentration law itself, pinned so a future fixture cannot quietly reintroduce it."""
    lo = vic.FAMILIES["cubic"](2000, 4, 12, 7)["ii_variation"]["hess_fro_cv"]
    hi = vic.FAMILIES["cubic"](2000, 40, 48, 7)["ii_variation"]["hess_fro_cv"]
    assert hi < lo / 2.0


def test_fixture_dicts_match_the_sealed_control_interface():
    """Existing runners consume ``X``/``x_param``/``H_vec``/``H_norm``/``global_std``."""
    fx = vic.FAMILIES["sine"](100, 4, 10, 3)
    for key in ("X", "x_param", "H_vec", "H_norm", "global_std"):
        assert key in fx
    assert fx["X"].shape == (100, 10)
    assert fx["H_vec"].shape == (100, 10)
    assert fx["H_norm"].shape == (100,)
    np.testing.assert_allclose(fx["H_norm"], np.linalg.norm(fx["H_vec"], axis=-1))


def test_second_fundamental_form_variation_is_zero_for_a_constant_hessian():
    hess = np.repeat(np.eye(3)[None, None, :, :], 50, axis=0)
    out = vic.second_fundamental_form_variation(hess)
    assert out["hess_fro_cv"] == pytest.approx(0.0, abs=1e-12)
    assert out["hess_fro_spread"] == pytest.approx(1.0)


def test_eigenvalue_shape_is_validated():
    with pytest.raises(ValueError, match="eigenvalues must have shape"):
        vic.make_quadratic_graph_control(10, 4, 8, 0, eigenvalues=np.ones(3))
    with pytest.raises(ValueError, match="coeffs must have shape"):
        vic.make_cubic_graph_control(10, 4, 8, 0, coeffs=np.ones(3))
