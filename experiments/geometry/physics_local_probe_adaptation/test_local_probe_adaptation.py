"""Unit tests (no embeddings required)."""

from __future__ import annotations

import numpy as np
import pytest

from geometry.physics_curvature_probe_submission_validation.schema import assert_probe_performance
from geometry.physics_local_probe_adaptation.config import PROBE_ALPHA
from geometry.physics_local_probe_adaptation.io_util import p_mc
from geometry.physics_local_probe_adaptation.metrics import assert_r2_from_sse_sst, metrics_from_preds
from geometry.physics_local_probe_adaptation.probes import fit_anchor_oof
from geometry.physics_local_probe_adaptation.ridge import ridge_fit_intercept, ridge_predict
from geometry.physics_local_probe_adaptation.synthetic import run_synthetic


def test_p_mc_never_zero():
    assert p_mc(0, 10000) == 1 / 10001


def test_ridge_matches_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    w_true = rng.normal(size=5)
    y = X @ w_true + 0.1 * rng.normal(size=200)
    w, b, info = ridge_fit_intercept(X, y, alpha=1.0)
    assert info["ok"]
    pred = ridge_predict(X, w, b)
    assert float(np.corrcoef(pred, y)[0, 1]) > 0.9


def test_oof_zero_overlap():
    rng = np.random.default_rng(1)
    n, d = 100, 8
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    yhat = rng.normal(size=n)
    fold = np.tile(np.arange(5), 20)
    sid = np.arange(n)
    idx = np.arange(n)
    fit = fit_anchor_oof(
        X=X, y=y, yhat_g=yhat, fold=fold, neigh_idx=idx, sample_ids_row=sid, alpha=PROBE_ALPHA, do_tangent=True
    )
    assert fit["overlap_any"] is False
    assert all(l["train_test_overlap"] == 0 for l in fit["fold_logs"])


def test_r2_sse_sst():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yhat = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    m = metrics_from_preds(y, yhat)
    assert_r2_from_sse_sst(m["sse"], m["sst"], m["r2"])


def test_probe_target_not_catalog_name():
    assert_probe_performance("mag_r_desi_local_oof_r2")


def test_synthetic_runs():
    out = run_synthetic(seed=0, n_anchor=16, n_patch=200)
    assert "rho_kappa_dMSE" in out
