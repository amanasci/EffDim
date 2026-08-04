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


def test_robust_spectral_norm_falls_back_to_float64_on_svd_nonconvergence(monkeypatch):
    """Real training (plan 02.2-05, Task 2) hit torch.linalg.LinAlgError: "failed to
    converge" from torch.linalg.matrix_norm on an actual trained chart-encoder weight --
    a failure mode the small random matrices used elsewhere in this test file never
    happened to trigger. _robust_spectral_norm must retry in float64 and return the
    correct spectral norm rather than propagating the error."""
    weight = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    expected = torch.linalg.matrix_norm(weight, ord=2)

    real_matrix_norm = torch.linalg.matrix_norm
    call_count = {"n": 0}

    def _flaky_matrix_norm(w, ord=None):
        call_count["n"] += 1
        if w.dtype == torch.float32:
            raise torch.linalg.LinAlgError("linalg.svd: The algorithm failed to converge")
        return real_matrix_norm(w, ord=ord)

    monkeypatch.setattr(torch.linalg, "matrix_norm", _flaky_matrix_norm)
    result = c._robust_spectral_norm(weight)

    assert result.dtype == weight.dtype
    assert result.item() == pytest.approx(expected.item(), rel=1e-10)
    assert call_count["n"] == 2  # one failed float32 attempt, one successful float64 retry


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


def test_train_cae_early_stopping_on_plateau():
    torch.manual_seed(0)
    x = torch.randn(60, 8) * 0.01  # tiny-variance data -- loss plateaus almost immediately
    model = c.ChartAutoEncoder(
        in_dim=8, embed_dim=4, chart_dim=2, n_charts=2, hidden=[8], activation="silu"
    )
    cfg = {
        "seed": 0,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "batch": 16,
        "max_epochs": 200,
        "early_stop_min_delta": 0.01,
        "early_stop_patience": 3,
        "wallclock_ceiling_s": 60.0,
    }
    fit = c.train_cae(model, x, cfg)
    assert fit["early_stopped"] is True
    assert fit["wallclock_truncated"] is False
    assert fit["epochs_run"] < cfg["max_epochs"]


def test_train_cae_runs_full_epoch_cap_when_neither_limit_fires():
    torch.manual_seed(0)
    x = _make_synthetic_fixture(n=40, ambient_dim=10, seed=10)
    model = c.ChartAutoEncoder(
        in_dim=10, embed_dim=4, chart_dim=2, n_charts=2, hidden=[8], activation="silu"
    )
    cfg = {
        "seed": 0,
        "lr": 1e-4,  # deliberately tiny -- loss keeps moving, plateau rule never fires
        "weight_decay": 0.0,
        "batch": 8,
        "max_epochs": 5,
        "early_stop_min_delta": 0.0,
        "early_stop_patience": 10_000,
        "wallclock_ceiling_s": 60.0,
    }
    fit = c.train_cae(model, x, cfg)
    assert fit["early_stopped"] is False
    assert fit["wallclock_truncated"] is False
    assert fit["epochs_run"] == cfg["max_epochs"]


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


# --- Plan 02.2-04 Task 1: a-posteriori chart pruning ------------------------------------


class _ChartDecoderOnlyModel:
    """A duck-typed model exposing exactly the interface chart_survival needs
    (n_charts, chart_decoders) -- it need not be a full ChartAutoEncoder."""

    def __init__(self, decoders):
        self.chart_decoders = decoders
        self.n_charts = len(decoders)


def test_pruning_removes_dead_chart():
    torch.manual_seed(0)
    model = c.ChartAutoEncoder(
        in_dim=10, embed_dim=5, chart_dim=2, n_charts=4, hidden=[6], activation="silu"
    )
    with torch.no_grad():
        for layer in model.chart_decoders[1].modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

    result = c.chart_survival(model, prune_tol=1e-2)
    assert result["n_charts_initial"] == 4
    assert result["n_charts_surviving"] == 3
    assert result["pruned_indices"] == [1]
    assert set(result["surviving_indices"]) == {0, 2, 3}
    assert result["chart_logmass"].dtype == np.float64
    assert result["chart_mass_ratio"].dtype == np.float64
    assert set(result.keys()) == {
        "n_charts_initial",
        "n_charts_surviving",
        "surviving_indices",
        "pruned_indices",
        "chart_logmass",
        "chart_mass_ratio",
        "prune_tol",
    }


def test_pruning_boundary_is_strict_less_than():
    torch.manual_seed(0)
    model = c.ChartAutoEncoder(
        in_dim=6, embed_dim=3, chart_dim=2, n_charts=2, hidden=[4], activation="silu"
    )
    # read off the actual computed mass ratio of the smaller chart -- ratio computation
    # itself does not depend on prune_tol, only the survive/prune label does
    probe = c.chart_survival(model, prune_tol=0.5)
    minority = int(np.argmin(probe["chart_mass_ratio"]))
    exact_ratio = float(probe["chart_mass_ratio"][minority])

    # exactly at the tolerance -> survives (strict less-than, not less-than-or-equal)
    at_tol = c.chart_survival(model, prune_tol=exact_ratio)
    assert minority in at_tol["surviving_indices"]
    assert minority not in at_tol["pruned_indices"]

    # a tolerance one representable step above the chart's exact ratio has the identical
    # effect, for the strict `<` comparison, as the chart's ratio sitting one step below a
    # fixed tolerance -- bit-exact round-tripping of exp(log(w)) back to an arbitrarily
    # chosen target ratio is not guaranteed, so the boundary is exercised by nudging the
    # tolerance by one ULP instead of the weight -> now pruned
    just_above = float(np.nextafter(exact_ratio, 1.0))
    below_tol = c.chart_survival(model, prune_tol=just_above)
    assert minority in below_tol["pruned_indices"]
    assert minority not in below_tol["surviving_indices"]


def test_pruning_statistic_no_underflow():
    torch.manual_seed(0)
    decoder = c.ChartDecoder(chart_dim=2, embed_dim=2, hidden=[2, 2, 2, 2, 2], activation="silu")
    linears = [m for m in decoder.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 6
    with torch.no_grad():
        for lin in linears:
            lin.weight.copy_(1e-3 * torch.eye(2))
            if lin.bias is not None:
                lin.bias.zero_()

    model = _ChartDecoderOnlyModel([decoder])
    result = c.chart_survival(model, prune_tol=1e-2)
    logmass = float(result["chart_logmass"][0])
    assert math.isfinite(logmass)
    assert logmass < 0.0
    assert logmass > -math.inf


# --- Plan 02.2-04 Task 2: eq. 8 cycle residual and the overlap-pair selection rule ------


class _IdentityChartModel:
    """A duck-typed model exposing exactly the interface r_cycle (and, with chart_dim/
    out_dim set, unfaithfulness_coverage) needs, with every map an identity -- a
    known-answer fixture in the same spirit as the tree-metric delta-hyperbolicity test,
    where the correct answer is derivable independently of the function under test."""

    def __init__(self, n_charts: int):
        self.chart_decoders = [torch.nn.Identity() for _ in range(n_charts)]
        self.embedding_decoder = torch.nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def chart_coords(self, z: torch.Tensor) -> torch.Tensor:
        return z.unsqueeze(1).expand(-1, len(self.chart_decoders), -1).clone()


def test_r_cycle_zero_for_identity_charts():
    torch.manual_seed(0)
    model = _IdentityChartModel(n_charts=3)
    x = torch.randn(20, 5, dtype=torch.float64)

    residual_ab = c.r_cycle(model, x, alpha=0, beta=1)
    residual_ba = c.r_cycle(model, x, alpha=1, beta=0)

    assert residual_ab.shape == (20,)
    assert residual_ab.abs().max().item() < 1e-10
    assert torch.allclose(residual_ab, residual_ba)

    # shape and swap-symmetry hold generally on a real, trained model too, not just in
    # the degenerate identity fixture
    real_model = c.ChartAutoEncoder(
        in_dim=8, embed_dim=4, chart_dim=2, n_charts=3, hidden=[8], activation="silu"
    )
    x_real = torch.randn(10, 8)
    r_ab = c.r_cycle(real_model, x_real, alpha=0, beta=2)
    r_ba = c.r_cycle(real_model, x_real, alpha=2, beta=0)
    assert r_ab.shape == (10,)
    assert torch.allclose(r_ab, r_ba)


def test_overlap_selection_halts_below_min_points():
    rng = np.random.default_rng(0)
    n = 50
    n_charts = 4
    surviving = [0, 1, 2, 3]

    # only a handful of rows have two chart probabilities both clearing p_min=0.3 --
    # every other row is dominated by a single chart
    p_sparse = np.zeros((n, n_charts))
    p_sparse[:, 0] = 0.9
    p_sparse[:, 1] = 0.1
    p_sparse[:5, 0] = 0.5
    p_sparse[:5, 1] = 0.5
    with pytest.raises(ValueError) as excinfo:
        c.select_overlap_pairs(p_sparse, p_min=0.3, min_points=20, surviving_indices=surviving)
    assert "20" in str(excinfo.value)

    # enough rows qualify
    p_dense = rng.dirichlet(np.ones(n_charts), size=n)
    result = c.select_overlap_pairs(p_dense, p_min=0.05, min_points=5, surviving_indices=surviving)
    assert len(result["rows"]) == len(result["alpha"]) == len(result["beta"])
    assert len(result["rows"]) >= 5
    for row, a, b in zip(result["rows"], result["alpha"], result["beta"]):
        assert p_dense[row, a] >= 0.05
        assert p_dense[row, b] >= 0.05


# --- Plan 02.2-04 Task 3: unfaithfulness/coverage and the geodesic-distortion statistic -


def test_unfaithfulness_coverage_hand_check():
    chart_dim = 2
    out_dim = 2
    model = _IdentityChartModel(n_charts=1)
    model.chart_dim = chart_dim
    model.out_dim = out_dim

    x_train = torch.tensor(
        [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]], dtype=torch.float32
    )
    n_samples = 6
    seed = 0

    result = c.unfaithfulness_coverage(
        model, x_train, surviving_indices=[0], n_samples=n_samples, seed=seed
    )

    # independently reconstruct the exact same deterministic sample -- round-robin over a
    # single surviving chart routes every sample through chart 0, and the identity
    # chart-decoder/embedding-decoder fixture means the decoded sample equals the sampled
    # chart-space coordinate directly
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand((n_samples, chart_dim), generator=g)
    d2 = torch.cdist(coords, x_train) ** 2
    nearest_dist2, nearest_idx = d2.min(dim=1)
    expected_unfaithfulness = float(nearest_dist2.mean().item())
    expected_n_distinct = int(torch.unique(nearest_idx).numel())
    expected_coverage = expected_n_distinct / n_samples

    assert result["unfaithfulness"] == pytest.approx(expected_unfaithfulness)
    assert result["coverage"] == pytest.approx(expected_coverage)
    assert result["n_samples"] == n_samples
    assert result["n_distinct"] == expected_n_distinct
    assert 0.0 <= result["coverage"] <= 1.0


def test_distortion_uses_global_embedding_not_chart_coords():
    rng = np.random.default_rng(0)
    n = 60
    chart_dim = 2
    embed_dim = 4
    t = rng.uniform(-1, 1, size=(n, chart_dim))
    chart_id = np.array([0] * (n // 2) + [1] * (n - n // 2))

    # global embedding: near-isometric, the same low-dimensional structure zero-padded
    # to embed_dim
    global_z = np.zeros((n, embed_dim))
    global_z[:, :chart_dim] = t

    # chart-local "coordinates": each chart's local frame is offset from the other's --
    # unrelated parameterisations, exactly RESEARCH.md Pitfall 3's failure mode
    chart_local = t.copy()
    chart_local[chart_id == 1] += 50.0

    rows_list, cols_list = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rows_list.append(i)
            cols_list.append(j)
    rows = np.array(rows_list)
    cols = np.array(cols_list)

    geo_pairs = np.linalg.norm(t[rows] - t[cols], axis=1)  # unsquared, matching the
    # cached geo_pairs_r2 contract
    train_mask = np.ones(len(rows), dtype=bool)

    global_result = c.embedding_distortion(
        global_z, geo_pairs, rows, cols, train_mask, chart_dim=chart_dim, embed_dim=embed_dim
    )
    # exact isometry (same t, zero-padded) plus a scale of exactly 1 -> distortion below
    # 1e-12, which only holds if the squaring convention is applied on exactly one side
    assert global_result["median_abs_rel"] < 1e-12
    assert set(global_result.keys()) == {
        "median_abs_rel",
        "median_signed_rel",
        "p95_abs_rel",
        "global_scale_factor",
        "n_pairs",
    }

    # replicate the identical formula on the chart-local coordinates directly (bypassing
    # the guard on purpose -- this is the exact number the guard exists to prevent a
    # caller from ever reaching through embedding_distortion itself)
    d2_rep_local = ((chart_local[rows] - chart_local[cols]) ** 2).sum(axis=1)
    d2_geo = geo_pairs**2
    scale_local = c.fit_global_scale(d2_rep_local, d2_geo, train_mask)
    local_stats = c.geometry_probes.distortion_stats(scale_local * d2_rep_local, d2_geo)

    assert local_stats["median_abs_rel"] > 2 * max(global_result["median_abs_rel"], 1e-9)

    with pytest.raises(ValueError):
        c.embedding_distortion(
            chart_local, geo_pairs, rows, cols, train_mask, chart_dim=chart_dim, embed_dim=embed_dim
        )
