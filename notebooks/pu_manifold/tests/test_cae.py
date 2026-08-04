"""
Synthetic-fixture tests for the ``pu_manifold.cae`` module (Phase 02.2 Chart Auto-Encoder,
arXiv:1912.10094).

No HuggingFace access, no frozen cache reads -- torch is required (this module needs it),
unlike its sibling ``test_geometry_probes.py``. Not collected by the core `effdim` test
suite (``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run
explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_cae.py -q
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch

from pu_manifold import cache
from pu_manifold import cae as c


# --- fixtures --------------------------------------------------------------------------


def _make_synthetic_fixture(n: int = 300, ambient_dim: int = 30, seed: int = 0) -> torch.Tensor:
    """A few hundred rows sampled from a 3-d spiral-ribbon structure, linearly projected
    into a modest ambient dimension plus small noise -- a known-topology fixture cheap
    enough to fit a tiny CAE against in well under a second per epoch."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 4 * np.pi, size=n)
    s = rng.uniform(-1, 1, size=n)
    low = np.stack([np.cos(t) * (1 + 0.3 * s), np.sin(t) * (1 + 0.3 * s), s], axis=1)  # (n, 3)
    proj = rng.standard_normal((3, ambient_dim))
    x = low @ proj
    x += rng.normal(scale=0.01, size=x.shape)
    return torch.tensor(x, dtype=torch.float32)


# --- Task 1: end-to-end tracer ----------------------------------------------------------


def test_end_to_end_smoke_writes_verdict_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    torch.manual_seed(0)

    x = _make_synthetic_fixture(n=300, ambient_dim=30, seed=0)
    n = x.shape[0]
    perm = np.random.default_rng(1).permutation(n)
    n_holdout = int(0.2 * n)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    x_train = x[train_idx]
    x_holdout = x[holdout_idx]

    model = c.ChartAutoEncoder(
        in_dim=30,
        embed_dim=6,
        chart_dim=3,
        n_charts=3,
        hidden=[16, 16],
        activation="silu",
    )
    cfg = {"lr": 3e-3, "weight_decay": 1e-4, "batch": 32, "max_epochs": 15}
    fit = c.train_cae(model, x_train, cfg)

    # the path is real, not a no-op
    assert fit["history"][-1]["total"] < fit["history"][0]["total"]
    assert set(fit.keys()) == {
        "history",
        "epochs_run",
        "wallclock_s",
        "wallclock_truncated",
        "early_stopped",
        "cfg",
    }

    with torch.no_grad():
        out = model(x_holdout)
        y_hat = model.reconstruct(x_holdout)

    # forward() contract
    assert set(out.keys()) == {"z", "z_charts", "y_charts", "p", "e"}
    assert out["z_charts"].shape == (x_holdout.shape[0], model.n_charts, model.chart_dim)
    assert out["y_charts"].shape == (x_holdout.shape[0], model.n_charts, model.out_dim)
    assert out["e"].shape == (x_holdout.shape[0], model.n_charts)
    row_sums = out["p"].sum(dim=1)
    assert torch.all(out["p"] >= 0)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    stats = c.reconstruction_stats(x_holdout, y_hat)

    metrics = {
        "distortion": 0.05,  # cheap synthetic stand-in for T1 (real statistic: plan 02.2-04)
        "rcycle_ratio": 0.5,  # cheap synthetic stand-in for T2 (real statistic: plan 02.2-04)
        "recon_margin": stats["mse_per_dim"],  # real number from this fit's holdout split
    }
    thresholds = {"distortion": 0.15, "rcycle_ratio": 2.0, "recon_margin": 1.0}
    verdict, gate_detail = c.verdict_from_metrics(metrics, thresholds)
    artifact = c.write_cae_verdict(
        "test_fit_key", metrics, thresholds, verdict, extra={"gate_detail": gate_detail}
    )

    path = cache.cache_path("cae_verdict_test_fit_key", "json")
    assert path.exists()
    written = json.loads(path.read_text())
    assert written["CAE_VERDICT"] in ("PASS", "FAIL")
    assert written["CAE_VERDICT"] == artifact["CAE_VERDICT"]
    expected_keys = {"fit_key", "phase", "CAE_VERDICT", "metrics", "thresholds", "gate_detail"}
    assert expected_keys.issubset(written.keys())


def test_chart_decoder_targets_embedding_space():
    dec = c.ChartDecoder(chart_dim=3, embed_dim=7, hidden=[8], activation="silu")
    out = dec(torch.zeros(4, 3))
    assert out.shape == (4, 7)

    embed_dim = 7
    model = c.ChartAutoEncoder(
        in_dim=20, embed_dim=embed_dim, chart_dim=3, n_charts=5, hidden=[8], activation="silu"
    )
    embedding_decoder_ids = {id(m) for m in model.modules() if isinstance(m, c.EmbeddingDecoder)}
    assert len(embedding_decoder_ids) == 1

    encoder = c.ChartEncoder(embed_dim=embed_dim, chart_dim=3, hidden=[8], activation="silu")
    encoded = encoder(torch.randn(64, embed_dim))
    assert torch.all(encoded > 0)
    assert torch.all(encoded < 1)

    z = torch.randn(64, embed_dim)
    p = model.chart_probs(z)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(dim=1), torch.ones(64), atol=1e-6)


# --- Task 2: fit-artifact serialization contract and the PASS handoff writer ------------


def test_state_dict_array_roundtrip():
    torch.manual_seed(0)
    model_a = c.ChartAutoEncoder(
        in_dim=10, embed_dim=4, chart_dim=2, n_charts=3, hidden=[8], activation="silu"
    )
    model_b = c.ChartAutoEncoder(
        in_dim=10, embed_dim=4, chart_dim=2, n_charts=3, hidden=[8], activation="silu"
    )

    arrays = c.state_dict_to_arrays(model_a.state_dict())
    assert all(isinstance(v, np.ndarray) for v in arrays.values())
    assert set(arrays.keys()) == set(model_a.state_dict().keys())

    restored = c.arrays_to_state_dict(arrays, model_b.state_dict())
    model_b.load_state_dict(restored)

    x = torch.randn(5, 10)
    with torch.no_grad():
        za = model_a(x)["z"]
        zb = model_b(x)["z"]
    assert torch.equal(za, zb)


def test_write_cae_handoff_refuses_non_pass_and_names_consumables(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    payload = {
        "consumes": "cae_fit_test_fit_key_seed0",
        "global_embedding_key": "z_all",
        "chart_coords_key": "chart_coords_all",
        "chart_probs_key": "p_all",
        "chart_assignments_key": "chart_argmax_all",
        "decoder_state_keys": ["chart_decoders.", "embedding_decoder."],
        "surviving_charts": [0, 1, 2],
        "activation": "silu",
    }
    with pytest.raises(ValueError):
        c.write_cae_handoff("test_fit_key", "FAIL", payload)

    handoff = c.write_cae_handoff("test_fit_key", "PASS", payload)
    expected_keys = {
        "global_embedding_key",
        "chart_coords_key",
        "chart_probs_key",
        "chart_assignments_key",
        "decoder_state_keys",
        "surviving_charts",
        "activation",
        "consumes",
        "fit_key",
        "phase",
        "verdict",
        "timestamp",
    }
    assert expected_keys.issubset(handoff.keys())


# --- Task 3: harden the verdict rule against its boundary/empty/ordering/precision edges -

_BASE_METRICS = {"distortion": 0.05, "rcycle_ratio": 0.5, "recon_margin": 0.2}
_BASE_THRESHOLDS = {"distortion": 0.15, "rcycle_ratio": 2.0, "recon_margin": 1.0}


def test_verdict_rule_strict_inequality():
    for gate in c.GATING_METRICS:
        threshold = _BASE_THRESHOLDS[gate]

        at_threshold = dict(_BASE_METRICS)
        at_threshold[gate] = threshold
        verdict, gate_detail = c.verdict_from_metrics(at_threshold, _BASE_THRESHOLDS)
        assert verdict == "FAIL"
        assert gate_detail[gate]["passed"] is False

        one_ulp_below = dict(_BASE_METRICS)
        one_ulp_below[gate] = math.nextafter(threshold, -math.inf)
        _, gate_detail_below = c.verdict_from_metrics(one_ulp_below, _BASE_THRESHOLDS)
        assert gate_detail_below[gate]["passed"] is True


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("gate", list(_BASE_METRICS.keys()))
def test_verdict_rejects_missing_or_nonfinite_metric(gate, bad_value):
    metrics = dict(_BASE_METRICS)
    if bad_value is None:
        del metrics[gate]
    else:
        metrics[gate] = bad_value
    with pytest.raises(ValueError):
        c.verdict_from_metrics(metrics, _BASE_THRESHOLDS)


def test_verdict_json_is_byte_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    metrics_a = {"distortion": 0.05, "rcycle_ratio": 0.5, "recon_margin": 0.2}
    metrics_b = {"recon_margin": 0.2, "distortion": 0.05, "rcycle_ratio": 0.5}
    thresholds_a = {"distortion": 0.15, "rcycle_ratio": 2.0, "recon_margin": 1.0}
    thresholds_b = {"recon_margin": 1.0, "rcycle_ratio": 2.0, "distortion": 0.15}

    verdict_a, _ = c.verdict_from_metrics(metrics_a, thresholds_a)
    c.write_cae_verdict("byte_stable_a", metrics_a, thresholds_a, verdict_a)
    path_a = cache.cache_path("cae_verdict_byte_stable_a", "json")
    bytes_a = path_a.read_bytes()

    verdict_b, _ = c.verdict_from_metrics(metrics_b, thresholds_b)
    c.write_cae_verdict("byte_stable_b", metrics_b, thresholds_b, verdict_b)
    path_b = cache.cache_path("cae_verdict_byte_stable_b", "json")
    bytes_b = path_b.read_bytes()

    # both stems share fit_key-independent content modulo the fit_key field itself
    text_a = bytes_a.decode().replace("byte_stable_a", "STEM")
    text_b = bytes_b.decode().replace("byte_stable_b", "STEM")
    assert text_a == text_b


def test_verdict_metrics_are_native_float64_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    metrics = {
        "distortion": np.float32(0.05),
        "rcycle_ratio": np.float64(0.5),
        "recon_margin": np.float32(0.2),
    }
    verdict, _ = c.verdict_from_metrics(
        {k: float(v) for k, v in metrics.items()}, _BASE_THRESHOLDS
    )
    written = c.write_cae_verdict("native_roundtrip", metrics, _BASE_THRESHOLDS, verdict)

    for key, value in written["metrics"].items():
        assert isinstance(value, float)
        assert value == pytest.approx(float(metrics[key]))

    path = cache.cache_path("cae_verdict_native_roundtrip", "json")
    reread = json.loads(path.read_text())
    for key, value in reread["metrics"].items():
        assert isinstance(value, float)
        assert value == pytest.approx(float(metrics[key]))


def test_cae_verdict_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    verdict, _ = c.verdict_from_metrics(_BASE_METRICS, _BASE_THRESHOLDS)
    first = c.write_cae_verdict("cache_roundtrip", _BASE_METRICS, _BASE_THRESHOLDS, verdict)

    # identical cfg (same fit_key/phase) -> returns the stored artifact unchanged
    second = c.write_cae_verdict("cache_roundtrip", _BASE_METRICS, _BASE_THRESHOLDS, verdict)
    assert second == first

    # a cfg-mismatched sidecar manifest for the same stem (simulates a stale/tampered
    # manifest, same technique as test_pu_manifold.py's manifest-mismatch test) -> raise,
    # never a silent overwrite or a silently-returned stale artifact
    manifest_path = cache.cache_path("cae_verdict_cache_roundtrip", "meta.json")
    manifest_path.write_text(json.dumps({"fit_key": "cache_roundtrip", "phase": "01"}))
    with pytest.raises(ValueError):
        c.write_cae_verdict("cache_roundtrip", _BASE_METRICS, _BASE_THRESHOLDS, verdict)
