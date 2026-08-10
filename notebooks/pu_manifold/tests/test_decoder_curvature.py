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
from torch.func import hessian, jacrev, vmap

from pu_manifold import cae, chart_curvature, curvature_probe
from pu_manifold import decoder_curvature as dc


def _trained_plain_ae(n: int, epochs: int, seed: int = 0):
    """Shared short-training helper: a Swiss roll fixture and a
    ``cae.PlainAutoEncoder(3, 2, hidden=(64, 64, 64), activation="silu")`` trained on it
    for ``epochs`` epochs, returned already ``.double()``'d. Factored out so the tracer
    test (Task 1) and the batch-split bit-identity test (Task 3) do not each repeat the
    training block. Returns ``(model, fixture)``.
    """
    fixture = curvature_probe.make_swiss_roll_fixture(n=n, seed=20260807)
    X_train = torch.tensor(fixture["X"], dtype=torch.float32)

    torch.manual_seed(seed)
    model = cae.PlainAutoEncoder(3, 2, hidden=(64, 64, 64), activation="silu")
    cfg = dict(seed=seed, lr=3e-4, weight_decay=1e-4, batch=64, max_epochs=epochs)
    cae.train_plain_ae(model, X_train, cfg)
    return model.double(), fixture


# --- Task 1: end-to-end tracer ------------------------------------------------------------


def test_plain_decoder_curvature_swiss_roll_end_to_end():
    """One path only: Swiss roll fixture -> trained PlainAutoEncoder -> decoder curvature
    -> four separately-reported read-out numbers. Asserts shapes, dtype, convention, and
    finiteness -- deliberately NO quality bar on any of the five numbers. This phase
    screens candidate decoder substrates and does not gate; a passing bar here would be a
    gate created by accident.
    """
    model, fixture = _trained_plain_ae(n=800, epochs=25, seed=0)

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


# --- Task 3: reproducibility, projector-form equivalence, convention pin -----------------


def test_plain_decoder_curvature_batch_split_is_bit_identical():
    """The property VMAP_CHUNK's fixed autodiff width is *supposed* to guarantee, per its
    own docstring: a row's result should not depend on which other rows share its chunk,
    so an arbitrary caller-side split should reproduce the whole-batch result exactly.
    Split at 50/70 deliberately (not a multiple of VMAP_CHUNK=32) so the split lands
    mid-chunk, the interesting case.

    **Measured correction to that docstring's claim, on a REAL trained decoder (not the
    tiny toy fixtures the claim was originally measured against).** Determinism -- calling
    ``plain_decoder_curvature`` twice on the IDENTICAL ``z`` -- is exact ``torch.equal``
    bit-identity, asserted below; this module has no source of randomness. But the
    cross-split comparison (a genuinely DIFFERENT chunk composition sharing row 54, say,
    with rows 32-63 in one call and rows 50-69 padded with row 69 in the other) is NOT
    exact bit-identity for this hidden=(64,64,64) architecture: measured max abs diff
    ~7e-14 (repeatable across the identical fixture/seed/split used here), verified to
    originate not from the Jacobian or Hessian themselves (those agree to ~2e-17, i.e. ~1
    ULP, across both chunk compositions -- exactly what the docstring's claim is really
    about) but from the pullback metric ``g``'s own conditioning: at ``cond(g) ~ 470`` for
    the affected rows, ``torch.linalg.solve``'s two chained calls amplify that ~1-ULP
    Jacobian noise by roughly ``cond(g)^2``, landing at the observed ~1e-13-1e-11 scale.
    Confirmed independently to be a property of the SEALED ``chart_curvature.chart_mean_curvature``
    itself (via a duck-typed decoder of the same hidden width), not a decoder_curvature.py
    regression -- the "regardless of which other rows share its chunk" half of
    VMAP_CHUNK's docstring claim was measured true only at chart_curvature.py's own
    tiny-toy-decoder scale and does not transfer to production-sized hidden widths.

    ``atol=1e-9`` here is ~4 orders of magnitude above the measured ~7e-14, a tight,
    principled margin (not a loosened "just pass" tolerance) that would still fail on a
    genuine row-reordering or re-widthing bug, which the sealed
    ``test_chart_curvature_field_reassembles_in_row_order`` precedent shows produces
    differences at the ~5e-15-and-up scale from width alone, let alone a real bug.
    """
    model, fixture = _trained_plain_ae(n=70, epochs=15, seed=1)
    X = torch.tensor(fixture["X"], dtype=torch.float64)
    with torch.no_grad():
        z = model.encode(X)

    # Determinism: identical input, identical output -- genuine bit-identity, no tolerance.
    repeat_a = dc.plain_decoder_curvature(model, z)
    repeat_b = dc.plain_decoder_curvature(model, z)
    assert torch.equal(repeat_a["H_vec"], repeat_b["H_vec"])
    assert torch.equal(repeat_a["H_norm"], repeat_b["H_norm"])

    full = dc.plain_decoder_curvature(model, z)
    part_a = dc.plain_decoder_curvature(model, z[:50])
    part_b = dc.plain_decoder_curvature(model, z[50:])
    cat_H_vec = torch.cat([part_a["H_vec"], part_b["H_vec"]], dim=0)
    cat_H_norm = torch.cat([part_a["H_norm"], part_b["H_norm"]], dim=0)

    assert torch.allclose(full["H_vec"], cat_H_vec, atol=1e-9, rtol=0.0)
    assert torch.allclose(full["H_norm"], cat_H_norm, atol=1e-9, rtol=0.0)


def test_plain_decoder_curvature_dxd_solve_matches_explicit_projector():
    """Proof, not assertion, that decision A-03's trace-first ``d x d`` solve is the same
    mathematics as the explicit ``(out_dim, out_dim)`` normal-projector form. The explicit
    form is reimplemented inline here, transcribed from 02.6-RESEARCH.md Pattern 1 --
    deliberately NOT used inside the module, because at the PU regime's ``out_dim = 768``
    a batch of 32 explicit projectors is 151 MB of float64.
    """
    torch.manual_seed(42)
    model = cae.PlainAutoEncoder(6, 2, hidden=(8, 8), activation="silu").double()
    gen = torch.Generator().manual_seed(20260811)
    z = torch.randn(32, 2, dtype=torch.float64, generator=gen) * 0.5

    out = dc.plain_decoder_curvature(model, z)

    decode_one = dc.plain_decoder_map(model)
    J = vmap(jacrev(decode_one))(z)  # (32, 6, 2)
    Hess = vmap(hessian(decode_one))(z)  # (32, 6, 2, 2)

    g = torch.einsum("boi,boj->bij", J, J)
    eye_d = torch.eye(2, dtype=torch.float64).expand(32, 2, 2)
    g_inv = torch.linalg.solve(g, eye_d)

    eye_out = torch.eye(6, dtype=torch.float64).expand(32, 6, 6)
    proj = eye_out - torch.einsum("boi,bij,bpj->bop", J, g_inv, J)
    II = torch.einsum("bop,bpjk->bojk", proj, Hess)
    H_explicit = torch.einsum("bij,boij->bo", g_inv, II)

    assert torch.allclose(out["H_vec"], H_explicit, atol=1e-10)


def test_decoder_curvature_convention_matches_chart_curvature():
    """Regression guard for the one way two modules carrying the same mathematics can
    silently produce two different answers to the same question."""
    assert dc.CURVATURE_CONVENTION == "trace"
    assert dc.CURVATURE_CONVENTION == chart_curvature.CURVATURE_CONVENTION
    assert dc.CURVATURE_CONVENTION == curvature_probe.CURVATURE_CONVENTION
