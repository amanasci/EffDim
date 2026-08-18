"""
Fast synthetic-fixture tests for the ``pu_manifold.decoder_priors`` module -- known-answer
tests for :func:`metric_deviation`, a duck-typed-decoder tracer for
:func:`chart_decoder_jacobian`'s autograd graph, and shim-hygiene tests for
:func:`decoder_prior_active`, additive to the existing suite. Not collected by the core
`effdim` test suite (``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) --
run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_decoder_priors.py -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
from torch import nn

from pu_manifold import cae, curvature_probe
from pu_manifold import decoder_priors as dp


# --- metric_deviation: known answers, no model involved --------------------------------


def test_metric_deviation_isometry_zero_for_identity():
    g = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    out = dp.metric_deviation(g, mode="isometry")
    assert out.shape == (1,)
    assert out.item() == 0.0


def test_metric_deviation_isometry_known_value():
    # ||4I - I||_F^2 = ||3I||_F^2 = 2 * 9 = 18 at chart_dim=2.
    g = (4.0 * torch.eye(2, dtype=torch.float64)).unsqueeze(0)
    out = dp.metric_deviation(g, mode="isometry")
    assert out.item() == 18.0


def test_metric_deviation_conformal_zero_for_scaled_identity():
    # A uniformly scaled metric has no conformal deviation -- the whole distinction between
    # the two modes.
    g = (4.0 * torch.eye(2, dtype=torch.float64)).unsqueeze(0)
    out = dp.metric_deviation(g, mode="conformal")
    assert out.item() == 0.0


def test_metric_deviation_conformal_known_value():
    # c = trace(diag(1,4))/2 = 2.5; ||diag(-1.5, 1.5)||_F^2 = 2 * 2.25 = 4.5.
    g = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64)).unsqueeze(0)
    out = dp.metric_deviation(g, mode="conformal")
    assert out.item() == 4.5


def test_metric_deviation_unknown_mode_raises_naming_the_string():
    g = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(ValueError, match="bogus-mode"):
        dp.metric_deviation(g, mode="bogus-mode")


# --- chart_decoder_jacobian: known-answer matrix, and its autograd graph ----------------


class _DuckModel:
    """Exposes exactly what ``chart_curvature.chart_decoder_map`` consumes: an indexable
    ``chart_decoders`` and a callable ``embedding_decoder`` -- no other attribute of a real
    ``cae.ChartAutoEncoder`` is required (02.2-05's known-answer-fixture precedent)."""

    def __init__(self, chart_decoders, embedding_decoder):
        self.chart_decoders = chart_decoders
        self.embedding_decoder = embedding_decoder


def _orthonormal_duck_model(chart_dim: int, out_dim: int, seed: int = 0):
    """A duck-typed two-linear-layer decoder whose composed Jacobian is EXACTLY a known
    orthonormal matrix for every row: the chart decoder is the identity map (weight = I), so
    the composition's Jacobian equals the embedding decoder's weight matrix, which is set to
    have orthonormal columns via a QR decomposition."""
    torch.manual_seed(seed)
    chart_decoder = nn.Linear(chart_dim, chart_dim, bias=False).double()
    with torch.no_grad():
        chart_decoder.weight.copy_(torch.eye(chart_dim, dtype=torch.float64))

    embedding_decoder = nn.Linear(chart_dim, out_dim, bias=False).double()
    raw = torch.randn(out_dim, chart_dim, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw)  # (out_dim, chart_dim), orthonormal columns
    with torch.no_grad():
        embedding_decoder.weight.copy_(q)

    model = _DuckModel(chart_decoders=[chart_decoder], embedding_decoder=embedding_decoder)
    return model, q


def test_chart_decoder_jacobian_matches_known_orthonormal_matrix():
    chart_dim, out_dim, batch = 2, 3, 5
    model, q = _orthonormal_duck_model(chart_dim, out_dim)

    z = torch.randn(batch, chart_dim, dtype=torch.float64)
    J = dp.chart_decoder_jacobian(model, z, chart_idx=0)

    assert tuple(J.shape) == (batch, out_dim, chart_dim)
    for row in range(batch):
        torch.testing.assert_close(J[row], q, atol=1e-12, rtol=0)


def test_chart_decoder_jacobian_carries_live_autograd_graph():
    chart_dim, out_dim, batch = 2, 3, 5
    model, _ = _orthonormal_duck_model(chart_dim, out_dim)

    z = torch.randn(batch, chart_dim, dtype=torch.float64)
    J = dp.chart_decoder_jacobian(model, z, chart_idx=0)
    g = torch.einsum("boi,boj->bij", J, J)
    dev = dp.metric_deviation(g, mode="isometry")
    dev.sum().backward()

    chart_grad = model.chart_decoders[0].weight.grad
    embed_grad = model.embedding_decoder.weight.grad
    assert chart_grad is not None and float(chart_grad.abs().sum()) > 0.0
    assert embed_grad is not None and float(embed_grad.abs().sum()) > 0.0


# --- decoder_prior_active: shim hygiene --------------------------------------------------


def _tiny_cae(seed: int = 0) -> cae.ChartAutoEncoder:
    torch.manual_seed(seed)
    return cae.ChartAutoEncoder(
        in_dim=3, embed_dim=4, chart_dim=2, n_charts=2, hidden=[16, 16], activation="silu"
    )


def _tiny_batch(n: int = 40, seed: int = 0) -> torch.Tensor:
    fixture = curvature_probe.make_swiss_roll_fixture(n=n, seed=20260807)
    return torch.tensor(fixture["X"], dtype=torch.float32)


def test_decoder_prior_active_zero_weight_installs_nothing():
    model = _tiny_cae()
    original = cae.chart_loss
    with dp.decoder_prior_active(model, weight=0.0):
        assert cae.chart_loss is original
    assert cae.chart_loss is original


def test_decoder_prior_active_nonzero_weight_patches_and_restores():
    model = _tiny_cae()
    original = cae.chart_loss
    with dp.decoder_prior_active(model, weight=1e-2, mode="isometry"):
        assert cae.chart_loss is not original
    assert cae.chart_loss is original


def test_decoder_prior_active_restores_on_exit_by_exception():
    model = _tiny_cae()
    original = cae.chart_loss
    with pytest.raises(RuntimeError, match="boom"):
        with dp.decoder_prior_active(model, weight=1e-2, mode="isometry"):
            assert cae.chart_loss is not original
            raise RuntimeError("boom")
    assert cae.chart_loss is original


def test_decoder_prior_active_wrapper_adds_penalty_to_total_only():
    model = _tiny_cae()
    x = _tiny_batch()
    out = model(x)
    unpatched = cae.chart_loss(x, out["y_charts"], out["p"])

    with dp.decoder_prior_active(model, weight=1e-2, mode="isometry"):
        patched = cae.chart_loss(x, out["y_charts"], out["p"])

    expected_penalty = dp.isometry_penalty(model, x, 1e-2, mode="isometry")

    torch.testing.assert_close(patched["recon"], unpatched["recon"])
    torch.testing.assert_close(patched["xent"], unpatched["xent"])
    torch.testing.assert_close(patched["total"], unpatched["total"] + expected_penalty)


def _tiny_train_cfg(seed: int) -> dict:
    return dict(seed=seed, lr=3e-3, weight_decay=1e-4, batch=16, max_epochs=3)


def test_train_cae_with_prior_moves_the_optimizer():
    x_train = _tiny_batch(n=120)

    model_prior = _tiny_cae(seed=0)
    with dp.decoder_prior_active(model_prior, weight=1e-2, mode="isometry"):
        cae.train_cae(model_prior, x_train, _tiny_train_cfg(0))

    model_plain = _tiny_cae(seed=0)
    cae.train_cae(model_plain, x_train, _tiny_train_cfg(0))

    sd_prior = model_prior.state_dict()
    sd_plain = model_plain.state_dict()
    assert set(sd_prior.keys()) == set(sd_plain.keys())
    differs = any(not torch.equal(sd_prior[k], sd_plain[k]) for k in sd_prior)
    assert differs, "the prior did not move a single parameter away from the unpatched fit"


def test_train_cae_zero_weight_is_bit_identical_to_unpatched():
    x_train = _tiny_batch(n=120)

    model_shim = _tiny_cae(seed=0)
    with dp.decoder_prior_active(model_shim, weight=0.0):
        cae.train_cae(model_shim, x_train, _tiny_train_cfg(0))

    model_plain = _tiny_cae(seed=0)
    cae.train_cae(model_plain, x_train, _tiny_train_cfg(0))

    sd_shim = model_shim.state_dict()
    sd_plain = model_plain.state_dict()
    assert set(sd_shim.keys()) == set(sd_plain.keys())
    for k in sd_shim:
        assert torch.equal(sd_shim[k], sd_plain[k])


# --- second-order (Christoffel) prior -------------------------------------------------------
# The safety claim for the second-order mode is that it penalizes the TANGENTIAL part of D^2 F
# and provably not the normal part (the second fundamental form). These are the known answers
# that check it, using hand-built J/Hess rather than a trained model so the answers are exact.


def _graph_jacobian_hessian(Q, z):
    """J and Hess of the graph map F(z) = (z, 0.5 z^T Q z) at a point z, in R^(d+1)."""
    d = z.shape[0]
    J = torch.zeros(1, d + 1, d, dtype=torch.float64)
    J[0, :d, :] = torch.eye(d, dtype=torch.float64)
    J[0, d, :] = Q @ z
    Hess = torch.zeros(1, d + 1, d, d, dtype=torch.float64)
    Hess[0, d, :, :] = Q
    return J, Hess


def test_christoffel_deviation_is_zero_at_a_critical_point_despite_large_curvature():
    """A graph over its own critical point is in normal coordinates there: Gamma = 0 while the
    second fundamental form is exactly Q. This is THE test of the safety claim -- the penalty
    must not see curvature."""
    from pu_manifold import decoder_priors

    d = 4
    Q = torch.diag(torch.tensor([3.0, -2.0, 5.0, -7.0], dtype=torch.float64))
    z = torch.zeros(d, dtype=torch.float64)  # grad f = Qz = 0 here
    J, Hess = _graph_jacobian_hessian(Q, z)

    assert Hess.abs().max() > 1.0, "fixture must have genuinely large second derivatives"
    dev = decoder_priors.christoffel_deviation(J, Hess)
    assert dev.shape == (1,)
    assert float(dev[0]) < 1e-20, f"Gamma must vanish at a critical point; got {float(dev[0])}"


def test_christoffel_deviation_is_nonzero_away_from_the_critical_point():
    """Same surface, same curvature, different point: the coordinates are no longer normal, so
    Gamma is non-zero. Curvature did not change; the parameterization content did."""
    from pu_manifold import decoder_priors

    d = 4
    Q = torch.diag(torch.tensor([3.0, -2.0, 5.0, -7.0], dtype=torch.float64))
    J0, Hess0 = _graph_jacobian_hessian(Q, torch.zeros(d, dtype=torch.float64))
    J1, Hess1 = _graph_jacobian_hessian(Q, torch.full((d,), 0.4, dtype=torch.float64))

    assert torch.equal(Hess0, Hess1), "the surface's second derivative is the same at both points"
    assert float(decoder_priors.christoffel_deviation(J1, Hess1)[0]) > 1e-3


def test_christoffel_deviation_penalizes_a_curved_reparameterization_of_a_flat_plane():
    """Zero curvature, non-affine coordinates. The normal part of D^2 F is exactly zero, so a
    Hessian-norm penalty would report nothing to fix; Gamma is large. This is the case the
    second-order prior exists to catch."""
    from pu_manifold import decoder_priors

    # F(u) = (u_0 + u_0^2, u_1) -- image is the flat plane, coordinates are stretched.
    u0 = 0.7
    J = torch.tensor([[[1.0 + 2.0 * u0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    Hess = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    Hess[0, 0, 0, 0] = 2.0  # d^2 F_0 / du_0^2

    # The surface is flat: the second derivative lies entirely IN the tangent plane.
    dev = decoder_priors.christoffel_deviation(J, Hess)
    assert float(dev[0]) > 1e-3, "a non-affine chart of a flat plane must have non-zero Gamma"


def test_christoffel_deviation_is_invariant_to_the_normal_component_magnitude():
    """Adding purely NORMAL second-derivative content -- i.e. more curvature -- must not change
    Gamma at all. Directly asserts the estimand cannot be biased by this penalty."""
    from pu_manifold import decoder_priors

    d = 3
    J = torch.zeros(1, d + 2, d, dtype=torch.float64)
    J[0, :d, :] = torch.eye(d, dtype=torch.float64)  # tangent space = first d ambient axes
    Hess = torch.zeros(1, d + 2, d, d, dtype=torch.float64)
    Hess[0, 0, 0, 0] = 1.5  # a tangential component (ambient axis 0 is tangent)
    base = float(decoder_priors.christoffel_deviation(J, Hess)[0])

    for scale in (1.0, 10.0, 100.0):
        H2 = Hess.clone()
        H2[0, d, :, :] = scale * torch.eye(d, dtype=torch.float64)      # normal direction
        H2[0, d + 1, :, :] = scale * torch.ones(d, d, dtype=torch.float64)  # normal direction
        got = float(decoder_priors.christoffel_deviation(J, H2)[0])
        assert abs(got - base) < 1e-18, (
            f"normal (curvature) content at scale {scale} moved Gamma: {base} -> {got}"
        )


def test_metric_deviation_refuses_second_order_modes():
    """A silent fall-through to the conformal branch would compute a different quantity under
    the caller's chosen name."""
    from pu_manifold import decoder_priors

    g = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(ValueError, match="second-order"):
        decoder_priors.metric_deviation(g, "christoffel")


# --- the scale barrier ------------------------------------------------------------------------
# Phase 3's three-seed spread measured two fits whose ENTIRE metric spectrum had collapsed to
# ~1e-07. cond(g), christoffel and conformal are all exactly blind to that; isometry saturates.
# These pin the one mode that is not.


def test_scale_deviation_is_zero_at_the_identity():
    from pu_manifold import decoder_priors

    for d in (2, 3, 20):
        g = torch.eye(d, dtype=torch.float64).unsqueeze(0)
        assert float(decoder_priors.metric_deviation(g, "scale")[0]) < 1e-24


def test_scale_deviation_matches_the_closed_form_under_uniform_rescaling():
    """g = c^2 I has log det g / d = 2 log c, so the penalty is exactly 4 log^2 c -- and is
    independent of d, since it is the squared MEAN log eigenvalue."""
    from pu_manifold import decoder_priors

    for d in (3, 20):
        for c in (1e-1, 1e-3, 1e-6):
            g = (c**2) * torch.eye(d, dtype=torch.float64).unsqueeze(0)
            got = float(decoder_priors.metric_deviation(g, "scale")[0])
            assert abs(got - 4.0 * math.log(c) ** 2) < 1e-9


def test_scale_deviation_diverges_on_collapse_where_every_other_mode_is_blind():
    """The decisive property. Under J -> cJ the christoffel term is EXACTLY invariant, conformal
    is driven to zero (it would reward collapse), and isometry saturates at ||I||_F^2 = d."""
    from pu_manifold import decoder_priors

    d = 4
    torch.manual_seed(0)
    J = torch.randn(2, 10, d, dtype=torch.float64)
    H = torch.randn(2, 10, d, d, dtype=torch.float64)
    H = 0.5 * (H + H.transpose(-1, -2))

    scales, chris, conf, iso = [], [], [], []
    for c in (1.0, 1e-3, 1e-6):
        g = torch.einsum("boi,boj->bij", c * J, c * J)
        scales.append(float(decoder_priors.metric_deviation(g, "scale").mean()))
        conf.append(float(decoder_priors.metric_deviation(g, "conformal").mean()))
        iso.append(float(decoder_priors.metric_deviation(g, "isometry").mean()))
        chris.append(float(decoder_priors.christoffel_deviation(c * J, c * H).mean()))

    assert scales[0] < scales[1] < scales[2], "scale must grow as the metric collapses"
    assert abs(chris[0] - chris[2]) < 1e-9, "christoffel must be exactly scale-invariant"
    assert conf[2] < conf[0] * 1e-6, "conformal is minimized by collapse -- it rewards it"
    assert abs(iso[2] - d) < 1e-6, "isometry saturates at ||I||_F^2 = d"


def test_scale_deviation_is_invariant_to_rotation_of_the_chart_frame():
    """Scale is a property of the metric's determinant, so an orthogonal change of chart basis
    must not move it -- otherwise the penalty would depend on an arbitrary coordinate choice."""
    from pu_manifold import decoder_priors

    d = 5
    torch.manual_seed(3)
    A = torch.randn(d, d, dtype=torch.float64)
    g = (A @ A.T).unsqueeze(0) + d * torch.eye(d, dtype=torch.float64).unsqueeze(0)
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    g_rot = (Q.T @ g[0] @ Q).unsqueeze(0)
    a = float(decoder_priors.metric_deviation(g, "scale")[0])
    b = float(decoder_priors.metric_deviation(g_rot, "scale")[0])
    assert abs(a - b) < 1e-9


def test_scale_deviation_survives_a_determinant_that_underflows_float64():
    """At d=20 the measured collapse (det(g) ~ 1e-162) does NOT underflow float64, but it is
    only two decades of eigenvalue away from doing so: at lambda ~ 1e-17 the determinant is
    ~1e-340 and rounds to exactly zero, where log(det(g)) is -inf. slogdet keeps the penalty
    finite there, which is the whole reason this mode does not compute log(det(...))."""
    from pu_manifold import decoder_priors

    d = 20
    g_measured = (1e-8) * torch.eye(d, dtype=torch.float64).unsqueeze(0)
    assert float(torch.linalg.det(g_measured[0])) > 0.0, "the measured scale does not underflow"
    assert math.isfinite(float(decoder_priors.metric_deviation(g_measured, "scale")[0]))

    g_underflow = (1e-17) * torch.eye(d, dtype=torch.float64).unsqueeze(0)
    assert float(torch.linalg.det(g_underflow[0])) == 0.0, "fixture must actually underflow det"
    got = float(decoder_priors.metric_deviation(g_underflow, "scale")[0])
    assert math.isfinite(got) and got > 1000.0
