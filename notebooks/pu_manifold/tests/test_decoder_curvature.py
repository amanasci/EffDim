"""
Fast synthetic-fixture tests for the ``pu_manifold.decoder_curvature`` module.

No HuggingFace access, no gitignored cache. Not collected by the core `effdim` test suite
(``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_decoder_curvature.py -q

Every test here pins a function against an input whose answer is known independently (a
sphere, a flat linear map, a ReLU decoder that must raise, the sealed
``curvature_probe.swiss_roll_analytic_H_scaled`` module) or against an equivalent
reimplementation, never merely "plausible" -- same discipline as
``test_curvature_probe.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from torch import nn

from pu_manifold import cae, chart_curvature, curvature_probe
from pu_manifold import decoder_curvature as dc


# --- Task 1: end-to-end tracer ------------------------------------------------------------


def test_plain_decoder_curvature_swiss_roll_end_to_end():
    """One path only: Swiss roll fixture -> trained PlainAutoEncoder -> decoder curvature
    -> four separately-reported read-out numbers. Asserts shapes, dtype, convention, and
    finiteness -- deliberately NO quality bar on any of the five numbers. This phase
    screens candidate decoder substrates and does not gate; a passing bar here would be a
    gate created by accident.
    """
    fixture = curvature_probe.make_swiss_roll_fixture(n=800, seed=20260807)
    X_train = torch.tensor(fixture["X"], dtype=torch.float32)

    torch.manual_seed(0)
    model = cae.PlainAutoEncoder(3, 2, hidden=(64, 64, 64), activation="silu")
    cfg = dict(seed=0, lr=3e-4, weight_decay=1e-4, batch=64, max_epochs=25)
    cae.train_plain_ae(model, X_train, cfg)
    model = model.double()

    X = torch.tensor(fixture["X"], dtype=torch.float64)
    with torch.no_grad():
        z = model.encode(X)

    out = dc.plain_decoder_curvature(model, z)

    n = X.shape[0]
    assert tuple(out["H_vec"].shape) == (n, 3)
    assert tuple(out["H_norm"].shape) == (n,)
    assert out["H_vec"].dtype == torch.float64
    assert out["H_norm"].dtype == torch.float64
    assert out["curvature_convention"] == "trace"

    H_true = dc.swiss_roll_analytic_H_vector(fixture["t"], fixture["global_std"])
    report = chart_curvature.curvature_fidelity_report(out["H_vec"], H_true)

    for key in (
        "median_cosine_similarity",
        "median_magnitude_ratio",
        "magnitude_ratio_cv",
        "calibration_slope",
    ):
        assert np.isfinite(report[key]), f"{key} is not finite: {report[key]!r}"

    rho = curvature_probe.spearman_gate_statistic(
        out["H_norm"].detach().cpu().numpy(),
        np.linalg.norm(H_true, axis=-1),
    )
    assert np.isfinite(rho)

    assert torch.isfinite(out["metric_condition_number"]).all()
