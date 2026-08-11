"""
Fast synthetic-fixture tests for the ``pu_manifold.derivative_bridge`` module -- the
derivative-usability bridge (SC-5; D-16, D-17, D-18).

No HuggingFace access, no gitignored cache. Not collected by the core `effdim` test suite
(``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_derivative_bridge.py -q

Every test here pins a function against an input whose answer is known independently (a
sphere, a flat linear map whose Hessian is exactly zero, the sealed
``decoder_curvature.plain_decoder_curvature``'s own output) or against an equivalent
autodiff computation, never merely "plausible" -- same discipline as
``test_decoder_curvature.py``. ``_SphereDecoder`` and ``_LinearDecoder`` are imported from
that module rather than redefined here, per this plan's own instruction to reuse the
known-answer fixtures already on record.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from torch.func import hessian, jacrev, vmap

from pu_manifold import cae, chart_curvature, curvature_probe
from pu_manifold import decoder_curvature as dc
from pu_manifold import derivative_bridge as db

from test_decoder_curvature import _SphereDecoder, _LinearDecoder, _fixed_z


def _small_trained_plain_ae(n: int, epochs: int, seed: int = 0):
    """Shared short-training helper, matching ``test_decoder_curvature._trained_plain_ae``'s
    shape (not imported from there -- a fresh, independent helper, since the plan places no
    restriction on redefining ordinary functions, only the two known-answer fixture
    classes). Returns ``(model, fixture)``, ``model`` already ``.double()``'d."""
    fixture = curvature_probe.make_swiss_roll_fixture(n=n, seed=20260807)
    X_train = torch.tensor(fixture["X"], dtype=torch.float32)

    torch.manual_seed(seed)
    model = cae.PlainAutoEncoder(3, 2, hidden=(16, 16), activation="silu")
    cfg = dict(seed=seed, lr=3e-4, weight_decay=1e-4, batch=64, max_epochs=epochs)
    cae.train_plain_ae(model, X_train, cfg)
    return model.double(), fixture


# --- Task 1: end-to-end tracer --------------------------------------------------------------


def test_derivative_agreement_end_to_end_sphere_known_answer():
    """One path only: sphere fixture -> autodiff Hessian -> calibrated step ->
    finite-difference Hessian -> both agreement levels -> the pin of :func:`reduce_to_H_vec`
    against ``plain_decoder_curvature``'s own ``H_vec`` -- no acceptance rule on a trained
    model anywhere.
    """
    R = 1.5
    model = _SphereDecoder(R)
    z = _fixed_z(batch=16)

    out = db.derivative_agreement(model, z)

    for level in ("full_hessian_agreement", "reduced_mean_curvature_agreement"):
        assert level in out
    assert set(out["reduced_mean_curvature_agreement"].keys()) == {"H_vec", "H_norm"}
    for stats in (
        out["full_hessian_agreement"],
        out["reduced_mean_curvature_agreement"]["H_vec"],
        out["reduced_mean_curvature_agreement"]["H_norm"],
    ):
        for key in (
            "max_abs",
            "median_abs",
            "p90_abs",
            "max_abs_relative",
            "median_abs_relative",
            "p90_abs_relative",
        ):
            assert np.isfinite(stats[key]), f"{key} is not finite: {stats[key]!r}"

    assert out["activation"] == "no-activation-modules"
    assert torch.isfinite(out["metric_condition_number"]).all()
    assert 1e-5 <= out["fd_step_used"] <= 1e-3

    # never a key combining the two levels, never a boolean
    assert "combined" not in out
    assert not any(isinstance(v, bool) for v in out.values())

    # the pin: reduce_to_H_vec fed the autodiff J/Hessian reproduces plain_decoder_curvature's
    # own H_vec exactly.
    decode_one = dc.plain_decoder_map(model)
    J = vmap(jacrev(decode_one))(z)
    Hess = vmap(hessian(decode_one))(z)
    mirrored = db.reduce_to_H_vec(J, Hess)
    sealed = dc.plain_decoder_curvature(model, z)["H_vec"]
    max_dev = (mirrored - sealed).abs().max().item()
    assert max_dev < 1e-10

    print(
        f"derivative_agreement sphere end-to-end: fd_step_used={out['fd_step_used']!r}, "
        f"full_hessian max_abs={out['full_hessian_agreement']['max_abs']!r}, "
        f"H_vec max_abs={out['reduced_mean_curvature_agreement']['H_vec']['max_abs']!r}, "
        f"reduce_to_H_vec pin max_dev={max_dev!r}"
    )


# --- Task 1: per-bullet behavior tests -------------------------------------------------------


def test_finite_difference_hessian_matches_autodiff_sphere_known_answer():
    R = 1.5
    model = _SphereDecoder(R)
    z0 = torch.tensor([[0.3, -0.2]], dtype=torch.float64)

    fd_hess = db.finite_difference_hessian(model.decode, z0, h=1e-4)

    def decode_one(zz: torch.Tensor) -> torch.Tensor:
        return model.decode(zz.unsqueeze(0)).squeeze(0)

    autodiff_hess = vmap(hessian(decode_one))(z0)

    max_abs_err = (fd_hess - autodiff_hess).abs().max().item()
    print(f"finite_difference_hessian vs torch.func.hessian on sphere at h=1e-4: "
          f"max_abs_err={max_abs_err!r} (research measured 4.5e-8)")
    assert max_abs_err < 1e-6


def test_finite_difference_hessian_linear_decoder_near_exactly_zero():
    """A linear map has an identically zero Hessian for ANY step size; the finite-difference
    result differs from it only by roundoff cancellation, which scales as
    ``O(eps / h**2)`` and therefore SHRINKS as ``h`` grows -- opposite of the truncation-error
    regime the sphere test above lives in. ``h=1e-2`` (rather than
    :data:`derivative_bridge.DEFAULT_FD_STEP`, which was calibrated for the sphere's genuine
    curvature signal, not for this exactly-flat case) keeps that cancellation well under the
    bound; measured directly this session at ``h=1e-4`` on this same fixture, the cancellation
    noise is ``~8.9e-8`` -- an order of magnitude above the ``<1e-8`` bound purely from
    ``4*eps/h**2`` amplification, with no bearing on correctness.
    """
    model = _LinearDecoder()
    z = _fixed_z()
    fd_hess = db.finite_difference_hessian(model.decode, z, h=1e-2)
    max_abs = fd_hess.abs().max().item()
    print(f"finite_difference_hessian on linear decoder at h=1e-2: max_abs={max_abs!r}")
    assert max_abs < 1e-8


def test_calibrate_fd_step_reproduces_u_shape():
    R = 1.5
    model = _SphereDecoder(R)
    z = _fixed_z(batch=4)

    result = db.calibrate_fd_step(model.decode, z)
    errors = result["step_errors"]

    print(f"calibrate_fd_step ladder on sphere: {errors!r}, best_step={result['best_step']!r}")

    assert set(errors.keys()) == set(db.DEFAULT_CALIBRATION_STEPS)
    assert errors[1e-4] < errors[1e-2]
    assert errors[1e-4] < errors[1e-7]
    assert 1e-5 <= result["best_step"] <= 1e-3


def test_reduce_to_H_vec_pins_plain_decoder_curvature_sphere():
    model = _SphereDecoder(1.5)
    z = _fixed_z()
    decode_one = dc.plain_decoder_map(model)

    J = vmap(jacrev(decode_one))(z)
    Hess = vmap(hessian(decode_one))(z)
    mirrored = db.reduce_to_H_vec(J, Hess)

    sealed = dc.plain_decoder_curvature(model, z)["H_vec"]
    assert (mirrored - sealed).abs().max().item() < 1e-10


def test_reduce_to_H_vec_pins_plain_decoder_curvature_trained_net():
    model, fixture = _small_trained_plain_ae(n=200, epochs=5, seed=2)
    X = torch.tensor(fixture["X"], dtype=torch.float64)
    with torch.no_grad():
        z = model.encode(X)

    decode_one = dc.plain_decoder_map(model)
    J_parts, Hess_parts = [], []
    for start in range(0, z.shape[0], chart_curvature.VMAP_CHUNK):
        real = z[start : start + chart_curvature.VMAP_CHUNK]
        n_real = real.shape[0]
        chunk = chart_curvature._pad_to_chunk(real)
        J_parts.append(vmap(jacrev(decode_one))(chunk)[:n_real])
        Hess_parts.append(vmap(hessian(decode_one))(chunk)[:n_real])
    J = torch.cat(J_parts, dim=0)
    Hess = torch.cat(Hess_parts, dim=0)
    mirrored = db.reduce_to_H_vec(J, Hess)

    sealed = dc.plain_decoder_curvature(model, z)["H_vec"]
    assert (mirrored - sealed).abs().max().item() < 1e-10


def test_derivative_agreement_returns_separate_keys_no_combined_no_boolean():
    model, fixture = _small_trained_plain_ae(n=150, epochs=5, seed=3)
    X = torch.tensor(fixture["X"], dtype=torch.float64)
    with torch.no_grad():
        z = model.encode(X)[:12]

    out = db.derivative_agreement(model, z)

    assert set(out.keys()) == {
        "full_hessian_agreement",
        "reduced_mean_curvature_agreement",
        "fd_step_used",
        "metric_condition_number",
        "activation",
    }
    assert set(out["reduced_mean_curvature_agreement"].keys()) == {"H_vec", "H_norm"}
    assert not any(isinstance(v, bool) for v in out.values())
    assert out["activation"] == "silu"
    assert isinstance(out["fd_step_used"], float)
    assert tuple(out["metric_condition_number"].shape) == (z.shape[0],)


def test_derivative_agreement_accepts_caller_supplied_h_without_calibrating():
    model, fixture = _small_trained_plain_ae(n=120, epochs=5, seed=4)
    X = torch.tensor(fixture["X"], dtype=torch.float64)
    with torch.no_grad():
        z = model.encode(X)[:8]

    out = db.derivative_agreement(model, z, h=3e-4)
    assert out["fd_step_used"] == 3e-4


def test_finite_difference_hessian_invocation_count_matches_chunk_arithmetic():
    """Bounded cost (T-02.6R-16): the number of ``decode_batch`` calls
    :func:`derivative_bridge.finite_difference_hessian` makes is
    ``ceil(batch * n_offsets / MAX_FD_ROWS)``, verified at two different batch sizes sharing
    one latent dimension -- NOT a fixed count, and not simply proportional to ``batch`` alone
    (a naive one-call-per-batch implementation would give 1 at both sizes; an unchunked
    implementation issuing one call per stencil point would give ``batch * n_offsets`` at
    both, which is proportional to ``batch`` alone and ignores :data:`MAX_FD_ROWS` entirely).
    """
    torch.manual_seed(0)
    model = cae.PlainAutoEncoder(6, 8, hidden=(8,), activation="silu").double()
    d = 8
    n_offsets = 1 + 2 * d + 4 * d * (d - 1) // 2
    assert n_offsets == 129

    gen = torch.Generator().manual_seed(20260811)
    for batch in (50, 130):
        calls = {"n": 0}
        base = model.decode

        def counting_decode(zz: torch.Tensor, _base=base, _calls=calls) -> torch.Tensor:
            _calls["n"] += 1
            return _base(zz)

        z = torch.randn(batch, d, dtype=torch.float64, generator=gen) * 0.3
        db.finite_difference_hessian(counting_decode, z, h=db.DEFAULT_FD_STEP)

        expected_calls = -(-(batch * n_offsets) // db.MAX_FD_ROWS)  # ceil division
        assert calls["n"] == expected_calls, (
            f"batch={batch}: expected {expected_calls} decode_batch calls "
            f"(ceil({batch}*{n_offsets}/{db.MAX_FD_ROWS})), got {calls['n']}"
        )


# --- Task 3: reproducibility, symmetry, guard, convention pin -------------------------------


def test_finite_difference_hessian_is_symmetric():
    model = _SphereDecoder(1.5)
    z = _fixed_z(batch=8)
    H = db.finite_difference_hessian(model.decode, z, h=1e-4)
    assert torch.equal(H, H.transpose(-1, -2))


def test_finite_difference_jacobian_matches_autodiff_sphere():
    model = _SphereDecoder(1.5)
    z = _fixed_z(batch=8)
    fd_J = db.finite_difference_jacobian(model.decode, z, h=1e-4)

    decode_one = dc.plain_decoder_map(model)
    autodiff_J = vmap(jacrev(decode_one))(z)

    assert (fd_J - autodiff_J).abs().max().item() < 1e-6


def test_finite_difference_hessian_refuses_float32():
    model = _LinearDecoder()
    z = torch.randn(8, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        db.finite_difference_hessian(model.decode, z)


def test_derivative_bridge_convention_matches_sealed_modules():
    assert db.CURVATURE_CONVENTION == "trace"
    assert db.CURVATURE_CONVENTION == chart_curvature.CURVATURE_CONVENTION
    assert db.CURVATURE_CONVENTION == dc.CURVATURE_CONVENTION


# --- POST-COMPLETION REPAIR (2026-08-11): _p90 / quantile-cap regression --------------------
#
# Found during plan 02.6-14's Task 2 full PU-regime run: torch.quantile raises
# "RuntimeError: quantile() input tensor is too large" above its own hard 2**24-element
# cap. The PU regime's full Hessian tensor (BRIDGE_N_POINTS=32, out_dim=768, d=40 ->
# 39,321,600 elements) is 2.3x over that cap. None of this plan's original 13 tests
# exercised _agreement_stats at PU dimensionality, so the defect passed every acceptance
# criterion and surfaced only at real scale. These three tests are the guard.


def test_p90_matches_torch_quantile_below_the_cap():
    """Below db._QUANTILE_MAX_ELEMENTS, _p90 must delegate to torch.quantile UNCHANGED --
    pinned to exact agreement, not merely 'close', since the repair's whole claim is that
    behaviour below the cap is byte-for-byte untouched."""
    torch.manual_seed(0)
    x = torch.rand(10_000, dtype=torch.float64)
    assert db._p90(x) == float(torch.quantile(x, 0.90).item())


def test_p90_matches_numpy_linear_interpolation_above_the_cap():
    """Just above the cap, _p90 must fall back to numpy.quantile's default
    method="linear" -- the SAME linear-interpolation formula torch.quantile's own default
    computes. Uses a tensor just barely over db._QUANTILE_MAX_ELEMENTS so the test stays
    fast; also confirms torch.quantile itself refuses this exact input, so the test is
    provably exercising the fallback branch rather than accidentally staying under it."""
    torch.manual_seed(0)
    n = db._QUANTILE_MAX_ELEMENTS + 1_000
    x = torch.rand(n, dtype=torch.float64)
    got = db._p90(x)
    expected = float(np.quantile(x.numpy(), 0.90))
    assert got == expected
    with pytest.raises(RuntimeError):
        torch.quantile(x, 0.90)


def test_agreement_stats_handles_tensor_above_quantile_cap():
    """The regression this repair exists for: db._agreement_stats must not raise on a
    tensor shaped like the PU regime's full Hessian (BRIDGE_N_POINTS=32, out_dim=768,
    d=40 -> 39,321,600 elements, 2.3x db._QUANTILE_MAX_ELEMENTS) -- synthetic, no model fit
    needed, since the defect is in the reporting statistic, not the derivative computation
    itself. This is the test that would have caught the crash before it reached a real run."""
    torch.manual_seed(0)
    shape = (32, 768, 40, 40)
    assert torch.Size(shape).numel() > db._QUANTILE_MAX_ELEMENTS
    reference = torch.rand(shape, dtype=torch.float64)
    other = reference + 1e-6 * torch.rand(shape, dtype=torch.float64)
    stats = db._agreement_stats(reference, other)
    for key in (
        "max_abs",
        "median_abs",
        "p90_abs",
        "max_abs_relative",
        "median_abs_relative",
        "p90_abs_relative",
    ):
        assert key in stats
        assert np.isfinite(stats[key])
    assert stats["p90_abs"] <= stats["max_abs"]
    assert stats["median_abs"] <= stats["p90_abs"]
