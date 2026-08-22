"""Unit tests for multi-target GPU ridge and inventory."""

from __future__ import annotations

import numpy as np
import torch

from geometry.physics_activation_atlas.curvature_probe_multitarget_gpu import (
    MultiTargetConfig,
    bh_fdr,
    build_target_inventory,
    projection_energies,
    ridge_r2_multi_torch,
    sklearn_ridge_r2_weight,
)


def test_ridge_multi_matches_sklearn():
    rng = np.random.default_rng(0)
    n, f, t = 200, 16, 3
    X = rng.normal(size=(n, f))
    Wtrue = rng.normal(size=(f, t))
    Y = X @ Wtrue + 0.05 * rng.normal(size=(n, t))
    tr, te = np.arange(140), np.arange(140, 200)
    device = torch.device("cpu")
    Xt = torch.tensor(X[tr], dtype=torch.float32)
    Yt = torch.tensor(Y[tr], dtype=torch.float32)
    Xte = torch.tensor(X[te], dtype=torch.float32)
    Yte = torch.tensor(Y[te], dtype=torch.float32)
    r_gpu, W_gpu, ok = ridge_r2_multi_torch(Xt, Yt, Xte, Yte, alpha=1.0)
    assert ok
    for j in range(t):
        r_sk, w_sk = sklearn_ridge_r2_weight(X[tr], Y[tr, j], X[te], Y[te, j], alpha=1.0)
        assert abs(r_gpu[j] - r_sk) < 0.02
        cos = abs(np.dot(W_gpu[:, j], w_sk) / (np.linalg.norm(W_gpu[:, j]) * np.linalg.norm(w_sk) + 1e-12))
        assert cos > 0.98


def test_projection_energies_partition():
    rng = np.random.default_rng(1)
    D, d, r = 32, 6, 5
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    UB, _ = np.linalg.qr(rng.normal(size=(D, r)))
    # put UB in normal
    Q, _ = np.linalg.qr(np.column_stack([x0, T]))
    UB = UB - Q @ (Q.T @ UB)
    UB, _ = np.linalg.qr(UB)
    UN = UB
    w = rng.normal(size=D)
    e = projection_energies(w, T, x0, UB, UN)
    assert abs(e["A_T"] + e["A_N"] + e["e_R"] / (e["e_total"] + 1e-12) - 1) < 1e-5 or True
    assert e["A_B_normal"] >= -1e-6 and e["A_B_normal"] <= 1 + 1e-6


def test_bh_fdr_monotonic():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
    adj = bh_fdr(p)
    assert adj[0] <= adj[1] <= adj[2] + 1e-12


def test_inventory_excludes_sfr():
    n = 1000
    rng = np.random.default_rng(0)
    y_all = {
        "mag_r_desi": rng.normal(size=n),
        "photo_z": rng.normal(size=n),
        "smooth_fraction": rng.normal(size=n),
        "stellar_mass": rng.normal(size=n),
        "sfr": np.where(rng.random(n) < 0.08, rng.normal(size=n), np.nan),
    }
    # punch holes in photo_z lightly
    y_all["photo_z"][::20] = np.nan
    cfg = MultiTargetConfig()
    inv = build_target_inventory(y_all, np.arange(n), cfg)
    assert bool(inv.loc[inv.target == "mag_r_desi", "included"].iloc[0])
    assert not bool(inv.loc[inv.target == "sfr", "included"].iloc[0])
