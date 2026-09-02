"""Local OOF probe scores. Catalog values are never the outcome."""

from __future__ import annotations

import numpy as np

from .config import MIN_FINITE_NEIGH


def neighbourhood_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    idx: np.ndarray,
) -> dict[str, float]:
    yy, yh = np.asarray(y, float)[idx], np.asarray(yhat, float)[idx]
    m = np.isfinite(yy) & np.isfinite(yh)
    n = int(m.sum())
    if n < 4:
        return {
            "r2_G": float("nan"),
            "mse_G": float("nan"),
            "local_label_variance": float("nan"),
            "local_evaluation_count": float(n),
        }
    yv, hv = yy[m], yh[m]
    sse = float(np.sum((yv - hv) ** 2))
    sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-18 else float("nan")
    return {
        "r2_G": float(r2),
        "mse_G": sse / n,
        "local_label_variance": float(np.var(yv)),
        "local_evaluation_count": float(n),
    }


def finite_enough(y: np.ndarray, idx: np.ndarray) -> bool:
    return int(np.isfinite(np.asarray(y, float)[idx]).sum()) >= MIN_FINITE_NEIGH
