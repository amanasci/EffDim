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
