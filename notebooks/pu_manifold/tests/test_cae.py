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


# --- Plan 02.2-03 Task 1: eq. 4 Lipschitz penalty, FPS, eq. 5 pre-training loss ---------


def test_lipschitz_regularizer_matches_svd():
    torch.manual_seed(0)
    encoder = c.ChartEncoder(embed_dim=3, chart_dim=2, hidden=[4], activation="tanh").double()

    linears = [m for m in encoder.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2  # Linear(3,4) -> tanh -> Linear(4,2) -> sigmoid

    w0 = np.array([[1.0, 0.5, -0.3], [0.2, -1.5, 0.7], [0.1, 0.4, 2.0], [-0.6, 0.3, 0.9]])
    w1 = np.array([[0.5, -0.2, 1.1, 0.3], [1.2, 0.4, -0.6, 0.8]])
    with torch.no_grad():
        linears[0].weight.copy_(torch.tensor(w0, dtype=torch.float64))
        linears[1].weight.copy_(torch.tensor(w1, dtype=torch.float64))

    s0 = np.linalg.svd(w0, compute_uv=False)[0]
    s1 = np.linalg.svd(w1, compute_uv=False)[0]
    expected_product = s0 * s1
    lip_weight = 0.5
    # a single chart encoder -> products has one element -> max == mean == expected_product
    expected_penalty = lip_weight * (expected_product + expected_product)

    penalty = c.lipschitz_penalty([encoder], lip_weight)
    assert penalty.item() == pytest.approx(expected_penalty, rel=1e-10)

    # differentiable: .backward() populates non-None, non-zero gradients on every Linear
    # weight in the chart encoder
    for lin in linears:
        lin.weight.grad = None
    penalty.backward()
    for lin in linears:
        assert lin.weight.grad is not None
        assert torch.any(lin.weight.grad != 0)


def test_fps_selects_farthest_points():
    square = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    idx = c.farthest_point_sample(square, 4, seed=0)
    assert idx.shape == (4,)
    picks = idx.tolist()
    assert set(picks) == {0, 1, 2, 3}  # all four corners chosen, no repeats

    # each successive pick is a genuinely farthest remaining point relative to what's
    # already been chosen -- verified generally rather than assuming a fixed tie-break
    chosen: list = []
    remaining = set(range(4))
    for i, pick in enumerate(picks):
        if i > 0:
            min_dists = {
                cand: min(torch.dist(square[cand], square[ch]).item() for ch in chosen)
                for cand in remaining
            }
            best = max(min_dists.values())
            assert min_dists[pick] == pytest.approx(best)
        chosen.append(pick)
        remaining.discard(pick)


def test_fps_reproducible_under_fixed_seed():
    x = torch.randn(50, 5, generator=torch.Generator().manual_seed(123))
    a = c.farthest_point_sample(x, 8, seed=42)
    b = c.farthest_point_sample(x, 8, seed=42)
    assert np.array_equal(a.numpy(), b.numpy())

    d = c.farthest_point_sample(x, 8, seed=43)
    assert not np.array_equal(a.numpy(), d.numpy())


def test_loss_matches_hand_computation():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y_charts = torch.tensor(
        [
            [[0.1, 0.1], [2.0, 2.0]],
            [[1.1, 1.1], [0.0, 0.0]],
        ]
    )
    p = torch.tensor([[0.9, 0.1], [0.2, 0.8]])

    e_np = np.array([[0.02, 8.0], [0.02, 2.0]])  # ||x - y_charts||^2 by hand
    ell_np = np.exp(-e_np) / np.exp(-e_np).sum(axis=1, keepdims=True)
    p_np = p.numpy()
    xent_np = -(ell_np * np.log(p_np)).sum(axis=1)
    recon_np = e_np.min(axis=1)
    expected_total = (recon_np + xent_np).mean()
    expected_recon = recon_np.mean()
    expected_xent = xent_np.mean()

    out = c.chart_loss(x, y_charts, p)
    assert out["recon"].item() == pytest.approx(expected_recon, rel=1e-6)
    assert out["xent"].item() == pytest.approx(expected_xent, rel=1e-6)
    assert out["total"].item() == pytest.approx(expected_total, rel=1e-6)


def test_fps_pretrain_loss_returns_named_components_and_is_differentiable():
    torch.manual_seed(0)
    model = c.ChartAutoEncoder(
        in_dim=6, embed_dim=4, chart_dim=2, n_charts=3, hidden=[8], activation="silu"
    )
    x_seeds = torch.randn(3, 6)
    seed_chart_index = torch.arange(3)

    loss_dict = c.fps_pretrain_loss(model, x_seeds, seed_chart_index)
    assert set(loss_dict.keys()) == {"recon", "center", "xent", "total"}

    model.zero_grad()
    loss_dict["total"].backward()
    grads = [p.grad for p in model.chart_encoders.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)

    # nudging a chart encoder toward mapping its seed point to the chart-space centre
    # decreases the center term
    before = loss_dict["center"].item()
    optimizer = torch.optim.SGD(model.chart_encoders.parameters(), lr=0.5)
    optimizer.step()
    after = c.fps_pretrain_loss(model, x_seeds, seed_chart_index)["center"].item()
    assert after < before


# --- Plan 02.2-03 Task 2: full pre-registered protocol and the three-way stopping rule --


def test_train_cae_respects_wallclock_ceiling():
    torch.manual_seed(0)
    x = _make_synthetic_fixture(n=80, ambient_dim=12, seed=6)
    model = c.ChartAutoEncoder(
        in_dim=12, embed_dim=6, chart_dim=2, n_charts=2, hidden=[8], activation="silu"
    )
    cfg = {
        "seed": 0,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "batch": 16,
        "max_epochs": 10_000,
        "wallclock_ceiling_s": 1e-6,
    }
    fit = c.train_cae(model, x, cfg)
    assert fit["wallclock_truncated"] is True
    assert fit["early_stopped"] is False
    assert fit["epochs_run"] < cfg["max_epochs"]
    assert set(fit.keys()) == {
        "history",
        "epochs_run",
        "wallclock_s",
        "wallclock_truncated",
        "early_stopped",
        "cfg",
    }
    assert fit["cfg"]["lip_every_n_steps"] == 1  # echoed default, not hardcoded in the loop


def test_train_cae_reproducible_under_same_seed():
    x = _make_synthetic_fixture(n=60, ambient_dim=12, seed=7)
    cfg = {
        "seed": 5,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "batch": 16,
        "max_epochs": 4,
        "n_charts": 2,
        "fps_pretrain_epochs": 2,
        "lip_weight": 1e-2,
        "lip_every_n_steps": 2,
        "wallclock_ceiling_s": 60.0,
    }

    torch.manual_seed(11)
    model_a = c.ChartAutoEncoder(
        in_dim=12, embed_dim=6, chart_dim=2, n_charts=2, hidden=[8], activation="silu"
    )
    fit_a = c.train_cae(model_a, x, dict(cfg))

    torch.manual_seed(11)
    model_b = c.ChartAutoEncoder(
        in_dim=12, embed_dim=6, chart_dim=2, n_charts=2, hidden=[8], activation="silu"
    )
    fit_b = c.train_cae(model_b, x, dict(cfg))

    assert len(fit_a["history"]) == len(fit_b["history"])
    stages = {h["stage"] for h in fit_a["history"]}
    assert stages == {"pretrain", "main"}
    for ha, hb in zip(fit_a["history"], fit_b["history"]):
        assert ha["stage"] == hb["stage"]
        assert ha["total"] == hb["total"]  # exact equality, not approx
    assert fit_a["cfg"]["lip_every_n_steps"] == 2


def test_timing_probe_returns_expected_keys():
    def model_factory():
        return c.ChartAutoEncoder(
            in_dim=10, embed_dim=4, chart_dim=2, n_charts=2, hidden=[8], activation="silu"
        )

    x = _make_synthetic_fixture(n=40, ambient_dim=10, seed=8)
    cfg = {
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "batch": 8,
        "max_epochs": 5,
        "wallclock_ceiling_s": 7200.0,
        "lip_weight": 0.0,
    }
    probe = c.timing_probe(model_factory, x, cfg, n_steps=5)
    assert set(probe.keys()) == {"seconds_per_step", "projected_wallclock_s", "exceeds_ceiling"}
    assert probe["seconds_per_step"] >= 0
    assert probe["projected_wallclock_s"] >= 0
    assert isinstance(probe["exceeds_ceiling"], bool)


# --- Plan 02.2-03 Task 3: matched-capacity baseline trainers for the CAE-03 gate --------


def test_plain_autoencoder_matches_eq22_shape():
    model = c.PlainAutoEncoder(768, 20, hidden=(250, 250, 250), activation="silu")
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 8
    assert model.encoder[-1].out_features == 20


def test_train_plain_ae_matches_train_cae_key_set():
    x = _make_synthetic_fixture(n=40, ambient_dim=10, seed=9)
    plain_model = c.PlainAutoEncoder(10, 4, hidden=(8, 8), activation="silu")
    cfg = {"seed": 0, "lr": 3e-3, "weight_decay": 1e-4, "batch": 8, "max_epochs": 3}
    plain_fit = c.train_plain_ae(plain_model, x, cfg)

    cae_model = c.ChartAutoEncoder(
        in_dim=10, embed_dim=4, chart_dim=2, n_charts=2, hidden=[8], activation="silu"
    )
    cae_fit = c.train_cae(cae_model, x, dict(cfg))

    assert set(plain_fit.keys()) == set(cae_fit.keys())
    assert "protocol_difference" in plain_fit["cfg"]


def test_fit_linear_decoder_recovers_known_linear_map():
    rng = np.random.default_rng(0)
    n, latent_dim, out_dim = 200, 5, 12
    z_np = rng.standard_normal((n, latent_dim))
    w_true = rng.standard_normal((out_dim, latent_dim))
    b_true = rng.standard_normal(out_dim)
    x_np = z_np @ w_true.T + b_true

    z = torch.tensor(z_np, dtype=torch.float64)
    x = torch.tensor(x_np, dtype=torch.float64)

    fitted = c.fit_linear_decoder(z, x)
    rel_err_w = np.linalg.norm(fitted["weight"] - w_true) / np.linalg.norm(w_true)
    rel_err_b = np.linalg.norm(fitted["bias"] - b_true) / np.linalg.norm(b_true)
    assert rel_err_w < 1e-8
    assert rel_err_b < 1e-8


def test_train_mlp_decoder_returns_fit_artifact():
    rng = np.random.default_rng(5)
    n, latent_dim, out_dim = 60, 4, 10
    z = torch.tensor(rng.standard_normal((n, latent_dim)), dtype=torch.float32)
    x = torch.tensor(rng.standard_normal((n, out_dim)), dtype=torch.float32)
    model = c.MlpDecoder(latent_dim, out_dim, hidden=(8, 8), activation="silu")
    cfg = {"seed": 0, "lr": 3e-3, "weight_decay": 1e-4, "batch": 16, "max_epochs": 3}
    fit = c.train_mlp_decoder(model, z, x, cfg)
    assert set(fit.keys()) == {
        "history",
        "epochs_run",
        "wallclock_s",
        "wallclock_truncated",
        "early_stopped",
        "cfg",
    }
    with torch.no_grad():
        y = model(z)["y"]
    stats = c.reconstruction_stats(x, y)
    assert "mse_per_dim" in stats


def test_single_chart_fails_circle_topology():
    n = 200
    ambient_dim = 8
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    rng = np.random.default_rng(0)
    proj = rng.standard_normal((2, ambient_dim))
    x = torch.tensor(circle @ proj, dtype=torch.float32)

    cfg = {
        "seed": 0,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "batch": 32,
        "max_epochs": 150,
        "lip_weight": 0.0,
        "fps_pretrain_epochs": 8,  # eq. 5 pre-training -- without it the second chart in
        # the two-chart model is never activated (the "dead chart" failure mode) and both
        # models perform identically, which would defeat the point of this test
        "wallclock_ceiling_s": 60.0,
    }

    torch.manual_seed(1)
    one_chart = c.ChartAutoEncoder(
        in_dim=ambient_dim, embed_dim=4, chart_dim=1, n_charts=1, hidden=[16, 16], activation="silu"
    )
    c.train_cae(one_chart, x, dict(cfg))

    torch.manual_seed(1)
    two_chart = c.ChartAutoEncoder(
        in_dim=ambient_dim, embed_dim=4, chart_dim=1, n_charts=2, hidden=[16, 16], activation="silu"
    )
    c.train_cae(two_chart, x, dict(cfg))

    with torch.no_grad():
        y_one = one_chart.reconstruct(x)
        y_two = two_chart.reconstruct(x)
    mse_one = ((x - y_one) ** 2).sum(dim=-1).mean().item()
    mse_two = ((x - y_two) ** 2).sum(dim=-1).mean().item()

    assert mse_two < 0.7 * mse_one
