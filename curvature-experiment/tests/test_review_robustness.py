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
