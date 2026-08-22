"""Unit tests for fixed global-probe curvature alignment."""

from __future__ import annotations

import numpy as np
import torch

from geometry.physics_activation_atlas.global_probe_curvature_alignment import (
    GlobalProbeAlignConfig,
    build_target_inventory,
    fit_global_probe,
    fixed_probe_decomposition,
    local_r2_fixed_predictions,
    projection_energies,
    train_global_probes_gpu,
    train_global_probes_sklearn,
    weighted_r2,
)


def test_fit_global_probe_matches_sklearn_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    y = X @ rng.normal(size=16) + 0.1 * rng.normal(size=200)
    w, b = fit_global_probe(X, y, alpha=100.0)
    assert w.shape == (16,)
    assert np.isfinite(b)


def test_gpu_multi_matches_sklearn_fit_global_probe():
    rng = np.random.default_rng(1)
    n, f, t = 300, 20, 3
    X = rng.normal(size=(n, f))
    Wtrue = rng.normal(size=(f, t))
    Y = X @ Wtrue + 0.05 * rng.normal(size=(n, t))
    Y[:20, 1] = np.nan  # different mask for target 1
    names = [f"t{i}" for i in range(t)]
    device = torch.device("cpu")
    gpu = train_global_probes_gpu(X, Y, names, alpha=100.0, device=device)
    sk = train_global_probes_sklearn(X, Y, names, alpha=100.0)
    for name in names:
        cos = abs(
            np.dot(gpu[name]["coef"], sk[name]["coef"])
            / (np.linalg.norm(gpu[name]["coef"]) * np.linalg.norm(sk[name]["coef"]) + 1e-12)
        )
        assert cos > 0.999
        assert abs(gpu[name]["intercept"] - sk[name]["intercept"]) < 1e-4


def test_uniform_weighted_r2_matches_sklearn():
    rng = np.random.default_rng(2)
    y = rng.normal(size=50)
    yhat = y + 0.1 * rng.normal(size=50)
    from sklearn.metrics import r2_score

    r_sk = r2_score(y, yhat)
    r_w = weighted_r2(y, yhat, np.ones(50))
    assert abs(r_sk - r_w) < 1e-10
    assert abs(local_r2_fixed_predictions(y, yhat) - r_sk) < 1e-10


def test_fixed_decomp_full_matches_global_pred():
    rng = np.random.default_rng(3)
    D, d, r, n = 32, 6, 4, 80
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    UB, _ = np.linalg.qr(rng.normal(size=(D, r)))
    Q, _ = np.linalg.qr(np.column_stack([x0, T]))
    UB = UB - Q @ (Q.T @ UB)
    UB, _ = np.linalg.qr(UB)
    UN = UB
    Xn = x0 + rng.normal(size=(n, D)) * 0.1
    w = rng.normal(size=D)
    b = 0.3
    y = b + Xn @ w + 0.01 * rng.normal(size=n)
    decomp = fixed_probe_decomposition(Xn, y, x0, w, b, T, UB, UN)
    assert decomp["R2_full"] > 0.9


def test_inventory_excludes_sfr():
    n = 500
    rng = np.random.default_rng(0)
    y_all = {
        "mag_r_desi": rng.normal(size=n),
        "photo_z": rng.normal(size=n),
        "smooth_fraction": rng.normal(size=n),
        "stellar_mass": rng.normal(size=n),
        "sfr": np.where(rng.random(n) < 0.08, rng.normal(size=n), np.nan),
    }
    inv = build_target_inventory(y_all, np.arange(n), GlobalProbeAlignConfig())
    assert bool(inv.loc[inv.target == "mag_r_desi", "included"].iloc[0])
    assert not bool(inv.loc[inv.target == "sfr", "included"].iloc[0])


def test_projection_energies_bounded():
    rng = np.random.default_rng(4)
    D, d, r = 24, 5, 3
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    UB, _ = np.linalg.qr(rng.normal(size=(D, r)))
    UN = UB
    w = rng.normal(size=D)
    e = projection_energies(w, T, x0, UB, UN)
    assert 0 <= e["A_T"] <= 1 + 1e-6
    assert e["A_B_normal"] >= -1e-6
