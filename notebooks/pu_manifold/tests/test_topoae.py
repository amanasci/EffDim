"""
Synthetic-fixture tests for the ``pu_manifold.topoae`` module (Phase 02.4 Topological
Auto-Encoder, arXiv:1906.00722).

No HuggingFace access, no frozen cache reads -- torch is required (this module needs
it), matching ``test_cae.py``'s discipline. Not collected by the core `effdim` test
suite (``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run
explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_topoae.py -q
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.manifold import trustworthiness

from pu_manifold import cache
from pu_manifold import cae as c
from pu_manifold import topoae as t
from pu_manifold.tests.test_cae import _make_synthetic_fixture


# --- Task 1: end-to-end tracer ------------------------------------------------------------


def test_topoae_tracer_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    x = _make_synthetic_fixture(n=300, ambient_dim=30, seed=0)
    n = x.shape[0]
    perm = np.random.default_rng(1).permutation(n)
    n_holdout = 60
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    x_train = x[train_idx]
    x_holdout = x[holdout_idx]

    latent_dim = 4
    train_cfg = {
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "batch": 32,
        "max_epochs": 3,
        "seed": 0,
        "lambda_topo": 0.5,
        "warmup_frac": 0.0,
        "ramp_frac": 0.34,
    }

    torch.manual_seed(0)
    topoae_model = c.PlainAutoEncoder(30, latent_dim)
    fit = t.train_topoae(topoae_model, x_train, train_cfg)

    # the returned shape matches cae._train_decoder_protocol's contract exactly
    assert set(fit.keys()) == {
        "history",
        "epochs_run",
        "wallclock_s",
        "wallclock_truncated",
        "early_stopped",
        "cfg",
    }
    assert fit["epochs_run"] == 3
    assert set(fit["history"][0].keys()) == {"epoch", "stage", "recon", "topo", "lambda_t", "total"}

    torch.manual_seed(0)
    baseline_model = c.PlainAutoEncoder(30, latent_dim)
    baseline_cfg = {"lr": 3e-3, "weight_decay": 1e-4, "batch": 32, "max_epochs": 3, "seed": 0}
    c.train_plain_ae(baseline_model, x_train, baseline_cfg)

    with torch.no_grad():
        z_topo = topoae_model.encode(x_holdout)
        y_topo = topoae_model.decode(z_topo)
        z_base = baseline_model.encode(x_holdout)
        y_base = baseline_model.decode(z_base)

    d_x = t.pairwise_distances_f64(x_holdout)

    def worse_direction(z: torch.Tensor) -> float:
        z_scaled = z.double() * t.latent_unit_scale(z)
        d_z = t.pairwise_distances_f64(z_scaled)
        loss = t.topological_loss(d_x, d_z)
        return max(loss["loss_x_to_z"].item(), loss["loss_z_to_x"].item())

    topo_worse = worse_direction(z_topo)
    base_worse = worse_direction(z_base)
    t1_topo_fidelity = topo_worse / max(base_worse, 1e-12)

    stats_topo = c.reconstruction_stats(x_holdout, y_topo)
    stats_base = c.reconstruction_stats(x_holdout, y_base)
    t2_recon_margin = stats_topo["mse_per_dim"] / max(stats_base["mse_per_dim"], 1e-12)

    x_np = x_holdout.detach().cpu().numpy().astype(np.float64)
    z_topo_np = z_topo.detach().cpu().numpy().astype(np.float64)
    z_base_np = z_base.detach().cpu().numpy().astype(np.float64)
    k = 5

    def rank_gate_value(z_np: np.ndarray) -> float:
        trust = trustworthiness(x_np, z_np, n_neighbors=k)
        cont = trustworthiness(z_np, x_np, n_neighbors=k)  # continuity = swapped args
        return 1.0 - min(trust, cont)

    rank_topo = rank_gate_value(z_topo_np)
    rank_base = rank_gate_value(z_base_np)
    t3_rank_structure = rank_topo / max(rank_base, 1e-12)

    # slot remap onto cae.verdict_from_metrics' generic GATING_METRICS names (plan 02
    # formalizes this remap; here it is inline, for the tracer only)
    metrics = {
        "distortion": t1_topo_fidelity,
        "rcycle_ratio": t2_recon_margin,
        "recon_margin": t3_rank_structure,
    }
    thresholds = {"distortion": 2.0, "rcycle_ratio": 2.0, "recon_margin": 2.0}
    verdict, gate_detail = c.verdict_from_metrics(metrics, thresholds)

    payload = {
        "fit_key": "tracer",
        "phase": "02.4",
        "TOPOAE_VERDICT": verdict,
        "metrics": {
            "t1_topo_fidelity": t1_topo_fidelity,
            "t2_recon_margin": t2_recon_margin,
            "t3_rank_structure": t3_rank_structure,
        },
        "thresholds": thresholds,
        "gate_detail": gate_detail,
    }
    written = cache.json_cache(
        "topoae_verdict_tracer", {"fit_key": "tracer", "phase": "02.4"}, lambda: c.to_native(payload)
    )

    path = cache.cache_path("topoae_verdict_tracer", "json")
    assert path.exists()
    reloaded = json.loads(path.read_text())
    assert reloaded["TOPOAE_VERDICT"] == written["TOPOAE_VERDICT"]
    assert reloaded["TOPOAE_VERDICT"] in ("PASS", "FAIL")
    assert set(reloaded["gate_detail"].keys()) == {"distortion", "rcycle_ratio", "recon_margin"}
    for gate_name, detail in reloaded["gate_detail"].items():
        assert set(detail.keys()) == {"value", "threshold", "passed"}
        assert isinstance(detail["value"], float)
        assert isinstance(detail["threshold"], float)
        assert isinstance(detail["passed"], bool)
