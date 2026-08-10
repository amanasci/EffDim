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


# --- Task 2: known-answer and guard tests ------------------------------------------------


class _LinearDecoder(nn.Module):
    """A fixed (5, 2) float64 linear map, ``decode(z) = z @ A.T + b``. A linear map has an
    identically zero Hessian, so its mean curvature is EXACTLY zero -- no tolerance.
    Registers no ``decoder`` attribute, so ``getattr(model, "decoder", model)`` correctly
    falls back to the fixture itself; registers no ``nn.Linear``/activation submodule
    either (``A``/``b`` are plain ``nn.Parameter`` tensors consumed by hand in
    ``decode``), so ``modules()`` yields nothing but the fixture itself and
    ``assert_c2_decoder`` returns ``"no-activation-modules"``."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(20260810)
        self.A = nn.Parameter(torch.randn(5, 2, dtype=torch.float64, generator=gen))
        self.b = nn.Parameter(torch.randn(5, dtype=torch.float64, generator=gen))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.A.T + self.b


class _SphereDecoder(nn.Module):
    """Inverse stereographic map onto the radius-``R`` 2-sphere in ``R^3``:
    ``R * [2 z0/(1+s), 2 z1/(1+s), (s-1)/(1+s)]`` with ``s = z0^2 + z1^2``. Under this
    module's ``H = tr_g(II)`` trace convention a ``d``-sphere of radius ``R`` has
    ``||H|| = d/R`` at every point, independent of parameterization -- here
    ``d = 2``, so ``||H|| = 2/R`` exactly. The ``1e-12`` tolerance used against this
    fixture is machine round-off, deliberately much tighter than
    ``test_curvature_probe.py``'s 20% band: that band is finite-radius ``k``-NN bias from
    differentiating a POINT CLOUD, which exact autodiff of this closed-form map does not
    have. Registers no ``decoder``/activation submodule, same as ``_LinearDecoder``."""

    def __init__(self, R: float):
        super().__init__()
        self.R = float(R)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        s = z[:, 0] ** 2 + z[:, 1] ** 2
        denom = 1.0 + s
        x = 2.0 * z[:, 0] / denom
        y = 2.0 * z[:, 1] / denom
        w = (s - 1.0) / denom
        return self.R * torch.stack([x, y, w], dim=1)


def _fixed_z(batch: int = 64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(20260810)
    return torch.randn(batch, 2, dtype=torch.float64, generator=gen) * 0.7


def test_plain_decoder_curvature_flat_linear_decoder_is_exactly_zero():
    model = _LinearDecoder()
    z = _fixed_z()
    out = dc.plain_decoder_curvature(model, z)
    assert out["H_norm"].max().item() == 0.0
    assert out["activation"] == "no-activation-modules"


def test_plain_decoder_curvature_sphere_known_answer():
    R = 1.5
    model = _SphereDecoder(R)
    z = _fixed_z()
    out = dc.plain_decoder_curvature(model, z)
    true_H = 2.0 / R
    max_dev = (out["H_norm"] - true_H).abs().max().item()
    assert max_dev < 1e-12


def test_assert_c2_decoder_rejects_relu_plain_autoencoder():
    model = cae.PlainAutoEncoder(3, 2, hidden=(8,), activation="relu").double()
    z = _fixed_z()
    with pytest.raises(ValueError, match="relu"):
        dc.plain_decoder_curvature(model, z)


def test_assert_c2_decoder_accepts_default_plain_autoencoder():
    model = cae.PlainAutoEncoder(3, 2)
    assert dc.assert_c2_decoder(model) == "silu"
    # Measured during planning: the sealed guard hard-raises on every PlainAutoEncoder,
    # silu or otherwise, because it has no .activation attribute. This is the exact gap
    # assert_c2_decoder exists to close -- if a future cae.py edit adds the attribute,
    # this pin goes silently redundant rather than staying invisible.
    with pytest.raises(ValueError):
        chart_curvature.assert_c2_activation(model)


def test_plain_decoder_curvature_refuses_float32():
    model = _LinearDecoder()
    z = torch.randn(16, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        dc.plain_decoder_curvature(model, z)


def test_swiss_roll_analytic_H_vector_norm_pins_sealed_module():
    fixture = curvature_probe.make_swiss_roll_fixture(n=1000, seed=20260807)
    H_vec = dc.swiss_roll_analytic_H_vector(fixture["t"], fixture["global_std"])
    norms = np.linalg.norm(H_vec, axis=-1)

    sealed_norms = curvature_probe.swiss_roll_analytic_H_scaled(
        fixture["t"], fixture["global_std"]
    )
    assert np.max(np.abs(norms - fixture["H_norm"])) < 1e-12
    assert np.max(np.abs(norms - sealed_norms)) < 1e-12
    assert np.all(H_vec[:, 1] == 0.0)
