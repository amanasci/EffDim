"""
Fast synthetic-fixture tests for the ``pu_manifold.synthetic_controls`` module.

Every test here pins a function against an input whose answer is known independently (a flat
plane, a unit sphere, a mixed-sign saddle cross-checked against an independent
central-finite-difference computation) or against an equivalent reimplementation, never
merely "plausible" -- same discipline as ``test_decoder_curvature.py`` and
``test_curvature_probe.py``.

No HuggingFace access, no gitignored cache. Not collected by the core `effdim` test suite
(``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_synthetic_controls.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pu_manifold import chart_curvature, curvature_probe
from pu_manifold import synthetic_controls as sc


# --- Task 1: convention guard, flat and sphere controls -----------------------------------


def test_synthetic_controls_convention_agrees_with_sealed_modules():
    assert sc.CURVATURE_CONVENTION == "trace"
    assert chart_curvature.CURVATURE_CONVENTION == "trace"
    assert curvature_probe.CURVATURE_CONVENTION == "trace"


def test_synthetic_flat_control_is_exactly_zero():
    fixture = sc.make_flat_control(n=200, d=6, D=40, seed=0)
    assert float(np.abs(fixture["H_norm"]).max()) == 0.0


def test_synthetic_sphere_control_matches_d_over_R():
    for d, D, R in [(4, 12, 1.0), (20, 768, 2.0)]:
        fixture = sc.make_sphere_control(n=200, d=d, D=D, seed=0, R=R)
        H_norm = fixture["H_norm"]
        rel_spread = float((H_norm.max() - H_norm.min()) / H_norm.mean())
        assert rel_spread < 1e-12, (d, D, R, rel_spread)
        ratio = float(H_norm.mean() / fixture["global_std"])
        assert abs(ratio - d / R) < 1e-12, (d, D, R, ratio)


def test_synthetic_controls_convention_is_trace_not_averaged():
    fixture = sc.make_sphere_control(n=200, d=4, D=12, seed=0, R=1.0)
    ratio = float(fixture["H_norm"].mean() / fixture["global_std"])
    assert abs(ratio - 4.0) < 1e-9
    assert abs(ratio - 1.0) > 1e-6
    assert abs(ratio - (4 + 2) / 4) > 1e-6


def test_synthetic_controls_construct_at_pu_scale():
    flat = sc.make_flat_control(n=200, d=20, D=768, seed=0)
    sphere = sc.make_sphere_control(n=200, d=20, D=768, seed=0)
    for fixture in (flat, sphere):
        assert fixture["X"].shape == (200, 768)
        assert fixture["H_vec"].shape == (200, 768)
        assert fixture["X"].dtype == np.float64
        assert fixture["H_vec"].dtype == np.float64


# --- Task 2: saddle control and its finite-difference cross-check -------------------------


def test_synthetic_saddle_control_matches_finite_difference():
    """Cross-checks the hand-computed ``grad``/``hess`` construction
    ``make_saddle_control`` uses against an independent central-finite-difference
    computation of the same quadratic, before it is trusted as ground truth
    (``03-RESEARCH.md`` Assumption A2). Self-contained: builds its own signs and domain
    points rather than reaching into ``make_saddle_control``'s internals, so this test
    exercises the formula, not the wrapper."""
    d, n = 6, 50
    rng = np.random.default_rng(20260814)
    n_pos = d // 2
    signs = np.array([1.0] * n_pos + [-1.0] * (d - n_pos), dtype=np.float64)
    rng.shuffle(signs)
    x = rng.uniform(-2.0, 2.0, size=(n, d))

    grad_hand = (x * signs)[:, None, :]
    hess_hand = np.repeat(np.diag(signs)[None, None, :, :], n, axis=0)

    def f(xi):
        return 0.5 * float(np.sum(xi**2 * signs))

    step = 1e-4
    grad_fd = np.zeros((n, 1, d))
    hess_fd = np.zeros((n, 1, d, d))
    for i in range(n):
        xi = x[i]
        f0 = f(xi)
        for k in range(d):
            xp = xi.copy()
            xp[k] += step
            xm = xi.copy()
            xm[k] -= step
            grad_fd[i, 0, k] = (f(xp) - f(xm)) / (2.0 * step)
            hess_fd[i, 0, k, k] = (f(xp) - 2.0 * f0 + f(xm)) / step**2
        for k in range(d):
            for l in range(k + 1, d):
                xpp = xi.copy()
                xpp[k] += step
                xpp[l] += step
                xpm = xi.copy()
                xpm[k] += step
                xpm[l] -= step
                xmp = xi.copy()
                xmp[k] -= step
                xmp[l] += step
                xmm = xi.copy()
                xmm[k] -= step
                xmm[l] -= step
                val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * step**2)
                hess_fd[i, 0, k, l] = val
                hess_fd[i, 0, l, k] = val

    H_fd = curvature_probe.graph_mean_curvature(grad_fd, hess_fd)
    H_hand = curvature_probe.graph_mean_curvature(grad_hand, hess_hand)

    np.testing.assert_allclose(H_fd, H_hand, rtol=1e-8, atol=1e-6)


def test_synthetic_saddle_control_field_genuinely_varies():
    fixture = sc.make_saddle_control(n=200, d=6, D=40, seed=0)
    H_norm = fixture["H_norm"]
    cv = float(H_norm.std() / H_norm.mean())
    assert cv > 0.05, cv
    assert float(H_norm.min()) * 10.0 <= float(H_norm.max())


def test_synthetic_saddle_control_constructs_at_pu_scale():
    fixture = sc.make_saddle_control(n=200, d=20, D=768, seed=0)
    assert fixture["X"].shape == (200, 768)
    assert fixture["H_vec"].shape == (200, 768)
    assert fixture["X"].dtype == np.float64
    assert fixture["H_vec"].dtype == np.float64
    assert np.all(np.isfinite(fixture["H_norm"]))
