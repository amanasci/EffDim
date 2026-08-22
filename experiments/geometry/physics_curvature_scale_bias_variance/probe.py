"""Neighbourhood evaluation of frozen global OOF predictions. No local probe refit."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.global_probe_curvature_alignment import local_r2_fixed_predictions
from geometry.physics_curvature_probe_submission_validation.schema import assert_not_catalog_vector

from .config import PRIMARY


def neighbourhood_metrics(y: np.ndarray, yhat: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    idx = np.asarray(idx, dtype=int)
    yy = y[idx]
    yh = yhat[idx]
    m = np.isfinite(yy) & np.isfinite(yh)
    n = int(m.sum())
    if n < 4:
        return {k: float("nan") for k in ("r2_local", "oof_mse", "oof_mae", "local_sst", "local_target_var", "n_eval")}
    a, b = yy[m], yh[m]
    ym = float(np.mean(a))
    sse = float(np.sum((a - b) ** 2))
    sst = float(np.sum((a - ym) ** 2))
    var = float(np.var(a))
    mse = sse / n
    return {
        "r2_local": float(local_r2_fixed_predictions(a, b)),
        "oof_mse": mse,
        "oof_mae": float(np.mean(np.abs(a - b))),
        "local_sst": sst,
        "local_target_var": var,
        "n_eval": float(n),
    }


def assert_not_catalog(r2: np.ndarray, catalog: np.ndarray) -> None:
    assert_not_catalog_vector(r2, catalog)
    _ = PRIMARY  # mag_r_desi_local_oof_r2 — never catalog magnitude
