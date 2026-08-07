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
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PU_MANIFOLD_ROOT = str(Path(__file__).resolve().parents[2])

import numpy as np
import pytest
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


# --- Task 2: R1 hardening -- deterministic tie-break, batch floor, float64 precision ------


def test_persistence_pairs_deterministic_on_ties(tmp_path):
    # small integers guarantee exact ties rather than hoping for them
    D = np.array(
        [
            [0, 1, 1, 2, 2],
            [1, 0, 2, 1, 2],
            [1, 2, 0, 2, 1],
            [2, 1, 2, 0, 1],
            [2, 2, 1, 1, 0],
        ],
        dtype=np.float64,
    )

    results = [t.persistence_pairs(D) for _ in range(10)]
    for r in results[1:]:
        assert np.array_equal(r, results[0])

    # cross-process half: an in-process repeat alone would not catch a hash-seed or
    # ordering dependency
    script_path = tmp_path / "recompute_persistence_pairs.py"
    script_path.write_text(
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {PU_MANIFOLD_ROOT!r})\n"
        "from pu_manifold import topoae as t\n"
        f"D = np.array({D.tolist()!r}, dtype=np.float64)\n"
        "pairs = t.persistence_pairs(D)\n"
        "print(json.dumps(pairs.tolist()))\n"
    )
    proc = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, check=True
    )
    subprocess_pairs = np.array(json.loads(proc.stdout.strip()), dtype=np.int64)
    assert np.array_equal(subprocess_pairs, results[0])


def test_persistence_pairs_batch_below_two_raises():
    with pytest.raises(ValueError, match="1"):
        t.persistence_pairs(np.zeros((1, 1)))

    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    pairs = t.persistence_pairs(D)
    assert pairs.shape == (1, 2)


def test_persistence_pairs_is_a_spanning_tree():
    rng = np.random.default_rng(7)
    n = 40
    pts = rng.standard_normal((n, 5))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))

    pairs = t.persistence_pairs(D)
    assert pairs.shape == (n - 1, 2)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs.tolist():
        ri, rj = find(i), find(j)
        assert ri != rj, f"cycle detected at edge ({i}, {j}) -- not a valid spanning tree"
        parent[ri] = rj

    roots = {find(i) for i in range(n)}
    assert len(roots) == 1, "returned pairs do not span all vertices"


def test_persistence_pairs_rejects_float32():
    # persistence_pairs up-casts to float64 before any comparison (the chosen
    # behaviour, asserted here rather than left implicit) -- a float32 input and its
    # float64-cast equivalent must produce the identical pairing
    D_f32 = np.array(
        [
            [0, 2, 2, 4],
            [2, 0, 4, 2],
            [2, 4, 0, 2],
            [4, 2, 2, 0],
        ],
        dtype=np.float32,
    )
    pairs_f32 = t.persistence_pairs(D_f32)
    pairs_f64 = t.persistence_pairs(D_f32.astype(np.float64))
    assert np.array_equal(pairs_f32, pairs_f64)
    assert pairs_f32.dtype == np.int64


def test_topological_loss_gradient_flows_through_values_not_selection():
    n = 6
    rng = np.random.default_rng(0)
    d_x_np = rng.uniform(0.1, 5.0, size=(n, n))
    d_x_np = (d_x_np + d_x_np.T) / 2
    np.fill_diagonal(d_x_np, 0.0)
    d_x = torch.tensor(d_x_np, dtype=torch.float64)

    d_z_np = np.random.default_rng(1).uniform(0.1, 5.0, size=(n, n))
    d_z_np = (d_z_np + d_z_np.T) / 2
    np.fill_diagonal(d_z_np, 0.0)
    d_z = torch.tensor(d_z_np, dtype=torch.float64, requires_grad=True)

    loss = t.topological_loss(d_x, d_z)
    loss["total"].backward()

    assert d_z.grad is not None
    pairs_x = t.persistence_pairs(d_x.detach().cpu().numpy())
    pairs_z = t.persistence_pairs(d_z.detach().cpu().numpy())
    selected = set()
    for i, j in pairs_x.tolist():
        selected.add((i, j))
    for i, j in pairs_z.tolist():
        selected.add((i, j))

    grad = d_z.grad.numpy()
    for i in range(n):
        for j in range(n):
            if (i, j) in selected or (j, i) in selected:
                continue
            assert grad[i, j] == 0.0, f"gradient leaked into unselected position ({i}, {j})"
    assert np.any(grad != 0.0), "no gradient reached any selected position"
