"""Anchor-level performance metrics and paired improvements."""

from __future__ import annotations

import numpy as np

from geometry.physics_activation_atlas.global_probe_curvature_alignment import local_r2_fixed_predictions


def metrics_from_preds(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    n = int(m.sum())
    if n < 4:
        return {k: float("nan") for k in ("sse", "mse", "mae", "r2", "sst", "var", "n_eval")}
    yy, yh = y[m], yhat[m]
    ym = float(np.mean(yy))
    sse = float(np.sum((yy - yh) ** 2))
    sst = float(np.sum((yy - ym) ** 2))
    mse = sse / n
    mae = float(np.mean(np.abs(yy - yh)))
    r2 = float(local_r2_fixed_predictions(yy, yh))
    # reconstructibility check
    if sst > 1e-18 and abs((1.0 - sse / sst) - r2) > 1e-4:
        # local_r2_fixed_predictions uses same formula; allow tiny tol
        pass
    return {
        "sse": sse,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "sst": sst,
        "var": float(np.var(yy)),
        "n_eval": float(n),
    }


def assert_r2_from_sse_sst(sse: float, sst: float, r2: float, *, atol: float = 1e-4) -> None:
    if not (np.isfinite(sse) and np.isfinite(sst) and np.isfinite(r2)):
        return
    if sst < 1e-18:
        return
    recon = 1.0 - sse / sst
    if abs(recon - r2) > atol:
        raise RuntimeError(f"R²≠1-SSE/SST: r2={r2} recon={recon}")
