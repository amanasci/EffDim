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
import math
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

    # the returned shape matches cae._train_decoder_protocol's contract, plus the
    # fidelity-correction addition latent_norm_final (2026-08-07)
    assert set(fit.keys()) == {
        "history",
        "epochs_run",
        "wallclock_s",
        "wallclock_truncated",
        "early_stopped",
        "cfg",
        "latent_norm_final",
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


# --- Task 3: R2 hardening -- reproducibility, drop_last, non-finite halt ------------------

_TOPOAE_CFG = {
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "batch": 32,
    "max_epochs": 4,
    "seed": 5,
    "lambda_topo": 0.3,
    "warmup_frac": 0.25,
    "ramp_frac": 0.25,
}


def test_train_topoae_reproducible():
    x = _make_synthetic_fixture(n=200, ambient_dim=20, seed=2)
    cfg = dict(_TOPOAE_CFG)

    torch.manual_seed(999)
    model1 = c.PlainAutoEncoder(20, 3)
    fit1 = t.train_topoae(model1, x, cfg)

    torch.manual_seed(999)
    model2 = c.PlainAutoEncoder(20, 3)
    fit2 = t.train_topoae(model2, x, cfg)

    # bit-identical, asserted with == -- no tolerance-based comparison
    assert fit1["history"] == fit2["history"]

    with torch.no_grad():
        y1 = model1.decode(model1.encode(x))
        y2 = model2.decode(model2.encode(x))
    stats1 = c.reconstruction_stats(x, y1)
    stats2 = c.reconstruction_stats(x, y2)
    assert stats1 == stats2


def test_train_topoae_different_seed_differs():
    x = _make_synthetic_fixture(n=200, ambient_dim=20, seed=2)
    cfg_a = dict(_TOPOAE_CFG, seed=5)
    cfg_b = dict(_TOPOAE_CFG, seed=6)

    torch.manual_seed(999)
    model_a = c.PlainAutoEncoder(20, 3)
    fit_a = t.train_topoae(model_a, x, cfg_a)

    torch.manual_seed(999)
    model_b = c.PlainAutoEncoder(20, 3)
    fit_b = t.train_topoae(model_b, x, cfg_b)

    assert fit_a["history"] != fit_b["history"]


def test_lambda_schedule_warmup_then_ramp_then_constant():
    cfg = {"lambda_topo": 2.0, "max_epochs": 20, "warmup_frac": 0.25, "ramp_frac": 0.25}
    # warmup_epochs = floor(0.25 * 20) = 5; ramp_epochs = floor(0.25 * 20) = 5

    for epoch in range(5):
        assert t.lambda_schedule(epoch, cfg) == 0.0

    ramp_values = [t.lambda_schedule(epoch, cfg) for epoch in range(5, 10)]
    assert all(a < b for a, b in zip(ramp_values, ramp_values[1:]))
    assert ramp_values[-1] == cfg["lambda_topo"]

    for epoch in range(10, 20):
        assert t.lambda_schedule(epoch, cfg) == cfg["lambda_topo"]


def test_train_topoae_drops_short_final_batch(monkeypatch):
    calls = []
    orig_persistence_pairs = t.persistence_pairs

    def spy(D):
        calls.append(D.shape[0])
        return orig_persistence_pairs(D)

    monkeypatch.setattr(t, "persistence_pairs", spy)

    n = 65  # 65 % 32 == 1 -- final batch of exactly 1 point, below the persistence floor
    x = _make_synthetic_fixture(n=n, ambient_dim=20, seed=3)
    cfg = dict(_TOPOAE_CFG, max_epochs=2, ramp_frac=0.5, warmup_frac=0.0)
    model = c.PlainAutoEncoder(20, 3)
    fit = t.train_topoae(model, x, cfg)

    assert fit["epochs_run"] == 2
    assert len(calls) > 0
    assert all(batch_n >= 2 for batch_n in calls), "persistence_pairs was called with a batch below 2 points"


def test_train_topoae_halts_on_non_finite_loss():
    x = _make_synthetic_fixture(n=64, ambient_dim=20, seed=4)
    cfg = dict(_TOPOAE_CFG, lr=1e8, weight_decay=0.0, max_epochs=5)
    model = c.PlainAutoEncoder(20, 3)
    with pytest.raises(ValueError, match="non-finite"):
        t.train_topoae(model, x, cfg)


# --- Plan 02.4-03 Swiss roll checkpoint: fidelity correction (2026-08-07) ------------------
#
# The Swiss roll gate found train_topoae() did not faithfully translate the paper
# (arXiv:1906.00722) / reference implementation (BorgwardtLab/topological-autoencoders):
# a missing jointly-trained latent_norm scale, a missing per-batch ambient d_x/d_x.max()
# normalization, a spurious /batch_size division, and a missing 1/2 factor on each
# directional term. These tests cover the correction.


def test_train_topoae_latent_norm_is_learned():
    # A dedicated, jointly-trained scalar (the paper's self.latent_norm) must actually
    # move away from its initial value of 1.0 when the topological term contributes to
    # the loss -- proving it receives gradient and is optimized, not just present.
    x = _make_synthetic_fixture(n=64, ambient_dim=10, seed=9)
    cfg = dict(_TOPOAE_CFG, max_epochs=3, warmup_frac=0.0, ramp_frac=0.0, lambda_topo=1.0)
    torch.manual_seed(123)
    model = c.PlainAutoEncoder(10, 3)
    fit = t.train_topoae(model, x, cfg)

    assert "latent_norm_final" in fit
    assert isinstance(fit["latent_norm_final"], float)
    assert fit["latent_norm_final"] != 1.0


def test_latent_norm_receives_gradient_directly():
    # Lower-level check, independent of the training loop: latent_norm's gradient is
    # nonzero when it participates in topological_loss through d_z / latent_norm --
    # the exact expression train_topoae now uses.
    rng = np.random.default_rng(2)
    n = 12
    x_np = rng.standard_normal((n, 4))
    z_np = rng.standard_normal((n, 3))
    d_x = t.pairwise_distances_f64(torch.tensor(x_np, dtype=torch.float64))
    d_x_norm = d_x / d_x.max()

    latent_norm = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    z = torch.tensor(z_np, dtype=torch.float64)
    d_z = t.pairwise_distances_f64(z)
    d_z_norm = d_z / latent_norm

    loss = t.topological_loss(d_x_norm, d_z_norm)["total"]
    loss.backward()

    assert latent_norm.grad is not None
    assert latent_norm.grad.item() != 0.0


def test_ambient_normalization_is_invariant_to_global_rescale():
    # The whole point of dividing d_x by its own per-batch max: the topological loss
    # must not care whether the ambient input is measured in millimetres or metres.
    # d_x_norm is scale-invariant to a global rescale of x, so the loss value must be
    # identical whether x or 100*x is passed in (with everything else held fixed).
    rng = np.random.default_rng(5)
    n = 20
    x_np = rng.standard_normal((n, 4))
    z_np = rng.standard_normal((n, 2))
    z = torch.tensor(z_np, dtype=torch.float64)
    d_z = t.pairwise_distances_f64(z)
    fixed_latent_norm = torch.tensor(1.7, dtype=torch.float64)
    d_z_norm = d_z / fixed_latent_norm

    def normalized_loss(scale: float) -> float:
        x = torch.tensor(x_np * scale, dtype=torch.float64)
        d_x = t.pairwise_distances_f64(x)
        d_x_norm = d_x / d_x.max()
        return t.topological_loss(d_x_norm, d_z_norm)["total"].item()

    loss_unit_scale = normalized_loss(1.0)
    loss_rescaled = normalized_loss(100.0)
    assert loss_unit_scale == pytest.approx(loss_rescaled, rel=1e-9)

    # A raw (unnormalized) comparison would NOT be invariant -- proves the assertion
    # above is testing the normalization, not an accidental invariance of the loss itself.
    def raw_loss(scale: float) -> float:
        x = torch.tensor(x_np * scale, dtype=torch.float64)
        d_x = t.pairwise_distances_f64(x)
        return t.topological_loss(d_x, d_z_norm)["total"].item()

    assert raw_loss(1.0) != pytest.approx(raw_loss(100.0), rel=1e-9)


def test_t1_gate_value_is_invariant_to_the_one_half_factor():
    # T1's gate value is a baseline-relative RATIO of the worse direction (D-02). The
    # paper's 1/2 factor on each directional term is a uniform multiplicative constant
    # applied identically to the model side and the baseline side of that ratio, so it
    # must cancel -- proved here by recomputing the same worse-of-two-directions
    # statistic with a hand-rolled loss that omits the 1/2, and confirming the ratio is
    # unchanged.
    rng_x = np.random.default_rng(41)
    rng_z1 = np.random.default_rng(42)
    rng_z2 = np.random.default_rng(43)
    x = torch.tensor(rng_x.standard_normal((30, 5)), dtype=torch.float64)
    z_model = torch.tensor(rng_z1.standard_normal((30, 3)), dtype=torch.float64)
    z_baseline = torch.tensor(rng_z2.standard_normal((30, 3)), dtype=torch.float64)

    fid_model = t.topological_fidelity(x, z_model)
    fid_baseline = t.topological_fidelity(x, z_baseline)
    ratio_with_half = t.t1_gate_value(fid_model, fid_baseline)

    def worse_without_half(x_: torch.Tensor, z_: torch.Tensor) -> float:
        d_x = t.pairwise_distances_f64(x_)
        z_scaled = z_ * t.latent_unit_scale(z_)
        d_z = t.pairwise_distances_f64(z_scaled)
        pairs_x = t.persistence_pairs(d_x.numpy())
        pairs_z = t.persistence_pairs(d_z.numpy())
        ix, jx = pairs_x[:, 0], pairs_x[:, 1]
        iz, jz = pairs_z[:, 0], pairs_z[:, 1]
        loss_x_to_z = ((d_x[ix, jx] - d_z[ix, jx]) ** 2).sum().item()
        loss_z_to_x = ((d_z[iz, jz] - d_x[iz, jz]) ** 2).sum().item()
        return max(loss_x_to_z, loss_z_to_x)

    model_worse = worse_without_half(x, z_model)
    baseline_worse = worse_without_half(x, z_baseline)
    ratio_without_half = model_worse / baseline_worse

    assert ratio_with_half == pytest.approx(ratio_without_half, rel=1e-9)


# --- Plan 02.4-02 Task 1: T1/T3 gate statistics and baseline-relative gate values ----------


def test_topological_fidelity_zero_on_identical_spaces():
    # every entry is +-1 and every column has zero mean, so mean per-dimension
    # variance is exactly 1.0 and latent_unit_scale(x) is exactly 1.0 -- multiplying by
    # exactly 1.0 is bit-identical, so d_x and d_z are bit-identical and the loss is
    # exactly zero, not just close to zero
    x = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    result = t.topological_fidelity(x, x)
    assert result["loss_x_to_z"] == 0.0
    assert result["loss_z_to_x"] == 0.0
    assert result["worse"] == 0.0


def test_topological_fidelity_gates_on_worse_direction():
    # x and z are drawn independently (no shared structure), so the two directional
    # terms are essentially guaranteed to differ
    rng_x = np.random.default_rng(11)
    rng_z = np.random.default_rng(97)
    x = torch.tensor(rng_x.standard_normal((40, 6)), dtype=torch.float64)
    z = torch.tensor(rng_z.standard_normal((40, 3)), dtype=torch.float64)

    result = t.topological_fidelity(x, z)
    a, b = result["loss_x_to_z"], result["loss_z_to_x"]
    assert a != b
    assert result["worse"] == max(a, b)
    # max of two distinct non-negative numbers is strictly greater than their mean --
    # proves the reduction is a max, not an average or a sum
    assert result["worse"] > (a + b) / 2


def test_topological_fidelity_is_scale_invariant_in_the_latent():
    rng_x = np.random.default_rng(21)
    rng_z = np.random.default_rng(22)
    x = torch.tensor(rng_x.standard_normal((40, 6)), dtype=torch.float64)
    z = torch.tensor(rng_z.standard_normal((40, 4)), dtype=torch.float64)

    result_a = t.topological_fidelity(x, z)
    result_b = t.topological_fidelity(x, z * 37.5)
    assert result_a["worse"] == pytest.approx(result_b["worse"], rel=1e-9)


def test_rank_structure_continuity_is_swapped_trustworthiness():
    rng = np.random.default_rng(31)
    x_np = rng.standard_normal((60, 8))
    z_np = rng.standard_normal((60, 3))
    k = 10

    result = t.rank_structure(torch.tensor(x_np), torch.tensor(z_np), k)
    expected_continuity = trustworthiness(z_np, x_np, n_neighbors=k)
    assert result["continuity"] == pytest.approx(expected_continuity)
    assert result["gate_value"] == pytest.approx(
        1.0 - min(result["trustworthiness"], result["continuity"])
    )


def test_gate_values_are_baseline_relative_ratios():
    fid_model = {"worse": 4.0}
    fid_baseline = {"worse": 2.0}
    assert t.t1_gate_value(fid_model, fid_baseline) == pytest.approx(2.0)
    assert t.t1_gate_value(fid_model, fid_model) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="t1_topo_fidelity"):
        t.t1_gate_value(fid_model, {"worse": 0.0})

    stats_model = {"mse_per_dim": 0.5}
    stats_baseline = {"mse_per_dim": 0.25}
    assert t.t2_gate_value(stats_model, stats_baseline) == pytest.approx(2.0)
    assert t.t2_gate_value(stats_model, stats_model) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="t2_recon_margin"):
        t.t2_gate_value(stats_model, {"mse_per_dim": float("nan")})

    rank_model = {"gate_value": 0.3}
    rank_baseline = {"gate_value": 0.15}
    assert t.t3_gate_value(rank_model, rank_baseline) == pytest.approx(2.0)
    assert t.t3_gate_value(rank_model, rank_model) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="t3_rank_structure"):
        t.t3_gate_value(rank_model, {"gate_value": float("inf")})


# --- Plan 02.4-02 Task 2: GATING_METRICS, the delegating verdict wrapper, the artifact -----

_TOPOAE_THRESHOLDS = {"t1_topo_fidelity": 1.0, "t2_recon_margin": 1.0, "t3_rank_structure": 1.0}
_TOPOAE_METRICS = {"t1_topo_fidelity": 0.5, "t2_recon_margin": 0.5, "t3_rank_structure": 0.5}


def test_gating_metrics_exactly_t1_t2_t3():
    assert t.GATING_METRICS == ("t1_topo_fidelity", "t2_recon_margin", "t3_rank_structure")


def test_verdict_strict_less_than_at_threshold():
    for gate in t.GATING_METRICS:
        at_threshold = dict(_TOPOAE_METRICS)
        at_threshold[gate] = _TOPOAE_THRESHOLDS[gate]
        verdict, gate_detail = t.verdict_from_topoae_metrics(at_threshold, _TOPOAE_THRESHOLDS)
        assert verdict == "FAIL"
        assert gate_detail[gate]["passed"] is False

        one_ulp_below = dict(_TOPOAE_METRICS)
        one_ulp_below[gate] = math.nextafter(_TOPOAE_THRESHOLDS[gate], -math.inf)
        _, gate_detail_below = t.verdict_from_topoae_metrics(one_ulp_below, _TOPOAE_THRESHOLDS)
        assert gate_detail_below[gate]["passed"] is True


def test_verdict_rejects_absent_or_nonfinite_topoae_metric():
    missing = dict(_TOPOAE_METRICS)
    del missing["t2_recon_margin"]
    with pytest.raises(ValueError, match="t2_recon_margin"):
        t.verdict_from_topoae_metrics(missing, _TOPOAE_THRESHOLDS)

    for bad_value in (float("nan"), float("inf")):
        bad = dict(_TOPOAE_METRICS)
        bad["t2_recon_margin"] = bad_value
        with pytest.raises(ValueError, match="t2_recon_margin"):
            t.verdict_from_topoae_metrics(bad, _TOPOAE_THRESHOLDS)


def test_verdict_gate_detail_keys_are_topoae_names():
    _, gate_detail = t.verdict_from_topoae_metrics(_TOPOAE_METRICS, _TOPOAE_THRESHOLDS)
    assert set(gate_detail.keys()) == set(t.GATING_METRICS)
    for name in t.GATING_METRICS:
        assert set(gate_detail[name].keys()) == {"value", "threshold", "passed"}


def test_verdict_artifact_carries_thresholds_in_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    verdict, _ = t.verdict_from_topoae_metrics(_TOPOAE_METRICS, _TOPOAE_THRESHOLDS)
    t.write_topoae_verdict("thr_test", _TOPOAE_METRICS, _TOPOAE_THRESHOLDS, verdict)

    changed_thresholds = dict(_TOPOAE_THRESHOLDS, t1_topo_fidelity=2.0)
    verdict2, _ = t.verdict_from_topoae_metrics(_TOPOAE_METRICS, changed_thresholds)
    with pytest.raises(ValueError):
        t.write_topoae_verdict("thr_test", _TOPOAE_METRICS, changed_thresholds, verdict2)


def test_verdict_artifact_written_on_fail_too(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    fail_metrics = {"t1_topo_fidelity": 5.0, "t2_recon_margin": 5.0, "t3_rank_structure": 5.0}
    verdict, _ = t.verdict_from_topoae_metrics(fail_metrics, _TOPOAE_THRESHOLDS)
    assert verdict == "FAIL"

    written = t.write_topoae_verdict("fail_test", fail_metrics, _TOPOAE_THRESHOLDS, verdict)
    assert written["TOPOAE_VERDICT"] == "FAIL"
    assert set(written["gate_detail"].keys()) == set(t.GATING_METRICS)

    path = cache.cache_path("topoae_verdict_fail_test", "json")
    assert path.exists()
    reloaded = json.loads(path.read_text())
    assert reloaded["TOPOAE_VERDICT"] == "FAIL"
    assert set(reloaded["gate_detail"].keys()) == set(t.GATING_METRICS)


def test_verdict_metrics_are_full_precision(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    precise = 1.2345678901234567  # 17 significant digits
    metrics = {"t1_topo_fidelity": precise, "t2_recon_margin": 0.5, "t3_rank_structure": 0.5}
    verdict, _ = t.verdict_from_topoae_metrics(metrics, _TOPOAE_THRESHOLDS)
    written = t.write_topoae_verdict("precision_test", metrics, _TOPOAE_THRESHOLDS, verdict)
    assert written["metrics"]["t1_topo_fidelity"] == precise

    path = cache.cache_path("topoae_verdict_precision_test", "json")
    reloaded = json.loads(path.read_text())
    assert reloaded["metrics"]["t1_topo_fidelity"] == precise


# --- Plan 02.4-02 Task 3: PASS-only handoff and R6's active deletion on FAIL ---------------

_HANDOFF_PAYLOAD = {
    "primary_d": 20,
    "fit_artifact_keys": ["topoae_fit_abc123_seed1"],
    "encoder_state_key": "topoae_fit_abc123_seed1_encoder",
    "gate_values": {"t1_topo_fidelity": 0.5, "t2_recon_margin": 0.6, "t3_rank_structure": 0.4},
    "thresholds": {"t1_topo_fidelity": 1.0, "t2_recon_margin": 1.0, "t3_rank_structure": 1.0},
    "evidence_criteria": "all three gates PASS on the primary rung, three seeds",
    "reinstated_requirements": ["DEC-01", "CURV-01"],
    "activation": "silu",
}


def test_handoff_refuses_non_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    with pytest.raises(ValueError):
        t.write_topoae_handoff("fk_refuse", "FAIL", _HANDOFF_PAYLOAD)
    assert not cache.cache_path("topoae_handoff_fk_refuse", "json").exists()


def test_handoff_carries_all_named_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    t.write_topoae_handoff("fk_fields", "PASS", _HANDOFF_PAYLOAD)

    path = cache.cache_path("topoae_handoff_fk_fields", "json")
    assert path.exists()
    reloaded = json.loads(path.read_text())
    for key in (
        "representation_id",
        "primary_d",
        "fit_artifact_keys",
        "encoder_state_key",
        "gate_values",
        "thresholds",
        "evidence_criteria",
        "reinstated_requirements",
        "activation",
        "timestamp",
    ):
        assert key in reloaded
    assert reloaded["representation_id"] == "topoae"


def test_fail_deletes_stale_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    json_path = cache.cache_path("topoae_handoff_fk_stale", "json")
    meta_path = cache.cache_path("topoae_handoff_fk_stale", "meta.json")
    json_path.write_text("{}")
    meta_path.write_text("{}")

    assert t.clear_stale_handoff("fk_stale") is True
    assert not json_path.exists()
    assert not meta_path.exists()


def test_clear_stale_handoff_is_a_noop_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    assert t.clear_stale_handoff("fk_absent") is False
