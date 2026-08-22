"""Unit tests for audit helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_local_probe_adaptation_audit.io_util import p_mc
from geometry.physics_local_probe_adaptation_audit.paired import paired_contrast
from geometry.physics_local_probe_adaptation_audit.shuffle import global_oof_from_ops, _fold_ops


def test_p_mc():
    assert p_mc(0, 200) == 1 / 201


def test_paired_contrast_runs():
    rng = np.random.default_rng(0)
    n = 64
    df = pd.DataFrame(
        {
            "K_H_cross": rng.normal(size=n),
            "mse_G": rng.normal(size=n),
            "mse_P": rng.normal(size=n) * 0.9,
            "log_knn_radius": rng.normal(size=n),
            "local_label_variance": rng.normal(size=n),
            "local_evaluation_count": rng.normal(size=n),
        }
    )
    out = paired_contrast(df, "mse_G", "mse_P", n_boot=50, seed=0, name="t")
    assert np.isfinite(out["delta_rho"])


def test_global_oof_no_leak():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))
    y = rng.normal(size=80)
    fold = np.tile(np.arange(5), 16)
    ops = _fold_ops(X, fold, alpha=1.0)
    yhat = global_oof_from_ops(X, y, ops)
    assert np.isfinite(yhat).all()
