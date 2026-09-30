"""Review robustness: concerns 2-5 on stored geometry."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), RUNNERS / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


rr = _load("11_review_robustness_run.py")


def _ridge_data(n=600, D=12, noise=0.5, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D)); beta = rng.standard_normal(D)
    y = X @ beta + noise * rng.standard_normal(n)
    if nan_frac:
        y[rng.random(n) < nan_frac] = np.nan
    return X, y


def test_oof_fixed_alpha_matches_published_helper():
    X, y = _ridge_data(nan_frac=0.1)
    got, alphas = rr.oof_predictions(X, y, (100.0, 100.0))
    want = rr.runner._oof_predictions_for_label(X, y, 100.0, rr.pcp.N_OOF_FOLDS, rr.pcp.OOF_FOLD_SEED)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[np.isfinite(got)], want[np.isfinite(want)])
    assert alphas == [100.0] * rr.pcp.N_OOF_FOLDS


def test_tuned_oof_never_uses_held_out_rows():
    """Scrambling the labels of fold 0's held-out rows must not change the alpha chosen for fold 0."""
    from sklearn.model_selection import KFold
    X, y = _ridge_data()
    _, a_ref = rr.oof_predictions(X, y, rr.ALPHA_GRID)
    test0 = next(iter(KFold(rr.pcp.N_OOF_FOLDS, shuffle=True, random_state=rr.pcp.OOF_FOLD_SEED).split(X)))[1]
    y2 = y.copy(); y2[test0] = np.random.default_rng(9).standard_normal(len(test0)) * 100
    _, a_new = rr.oof_predictions(X, y2, rr.ALPHA_GRID)
    assert a_new[0] == a_ref[0]


def test_select_alpha_small_for_strong_signal():
    X, y = _ridge_data(noise=0.01)
    assert rr.select_alpha(X, y) <= 1.0
    assert rr.PUBLISHED_ALPHA in rr.ALPHA_GRID and 1.0 in [round(a, 12) for a in rr.ALPHA_GRID]


def test_global_probe_matches_ridge():
    from sklearn.linear_model import Ridge
    X, y = _ridge_data(nan_frac=0.1)
    w, b0 = rr.global_probe(X, y, 100.0)
    fin = np.isfinite(y)
    ref = Ridge(alpha=100.0).fit(X[fin], y[fin])
    np.testing.assert_array_equal(w, ref.coef_); assert b0 == float(ref.intercept_)


@pytest.fixture(scope="module")
def smoke():
    """The split runner's smoke fixture: generator data, a 3-epoch decoder, its geometry at the anchors."""
    data = rr.ppf.load_smoke(argparse.Namespace(seed=20260905))
    X = data["X"]; n = X.shape[0]
    k, n_anchors, d = rr.adj.SMOKE["k"], rr.adj.SMOKE["n_anchors"], rr.adj.SMOKE["d"]
    a = rr.pcp.anchor_indices(n, rr.pcp.SPLIT_SEED, rr.pcp.HOLDOUT_FRACTION, n_anchors, rr.pcp.ANCHOR_DRAW_SEED)["anchor_idx"]
    panel = rr.pcp.knn_panel(X, a, k)
    fit = rr.ppf.fit_decoder(X, d, X.shape[1], 3)
    with torch.no_grad():
        z = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long)])
    geo = rr.ppf.decoder_geometry(fit["curvature_model"], z)
    return {"X": X, "y": np.asarray(data["labels"]["lin"], float), "a": a, "panel": panel, "geo": geo, "d": d}


def test_probe_panel_matches_published_construction(smoke):
    s = smoke
    pp = rr.probe_panel(s["X"], s["y"], s["a"], s["panel"], (100.0, 100.0))
    y_hat = rr.runner._oof_predictions_for_label(s["X"], s["y"], 100.0, rr.pcp.N_OOF_FOLDS, rr.pcp.OOF_FOLD_SEED)
    loc = rr.pcp.local_r2_panel(s["y"], y_hat, s["panel"]["indices"], rr.pcp.MIN_FINITE_NEIGHBOURS)
    np.testing.assert_array_equal(pp["r2"], loc["r2"])
    ks = [kk for kk in rr.ppf.MULTISCALE_KS if kk <= s["panel"]["indices"].shape[1]]
    assert pp["Z_multi"].shape == (len(s["a"]), len(ks) + 2)


def test_extended_controls_and_partials(smoke):
    s = smoke
    pp = rr.probe_panel(s["X"], s["y"], s["a"], s["panel"], (100.0, 100.0))
    w, b0 = rr.global_probe(s["X"], s["y"], 100.0)
    sq = rr.split_quantities(s["X"], s["y"], s["a"], s["panel"], s["geo"], w, b0, s["d"])
    Z_ext = rr.extended_controls(pp["Z_multi"], sq["cols"], sq["roughness"])
    assert Z_ext.shape[1] == pp["Z_multi"].shape[1] + 2
    np.testing.assert_array_equal(Z_ext[:, -2], sq["cols"]["hess_label"])
    parts = rr.partials(sq["cols"], pp["r2"], Z_ext, 50)
    ref = rr.ppf.partial_row(sq["cols"][rr.MISMATCH], pp["r2"], Z_ext, 50)
    assert parts[rr.MISMATCH]["partial"] == ref["partial"] and set(parts) == {rr.MISMATCH, rr.ALIGN}
