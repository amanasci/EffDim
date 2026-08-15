"""
Fast synthetic-fixture tests for the ``pu_manifold.decoder_priors`` module -- known-answer
tests for :func:`metric_deviation`, a duck-typed-decoder tracer for
:func:`chart_decoder_jacobian`'s autograd graph, and shim-hygiene tests for
:func:`decoder_prior_active`, additive to the existing suite. Not collected by the core
`effdim` test suite (``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) --
run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_decoder_priors.py -q
"""

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
