"""Per-model Q reconstruction. Tensors discarded after contraction."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_task_aligned_curvature.geometry_q import q_for_weights

from .audit import load_q_frame
from .config import N_Q_SPLITS


def eval_q_model(shared: dict, bundle: dict, weights: dict[str, np.ndarray], *, n_splits: int = N_Q_SPLITS) -> dict[str, Any]:
    sids = shared["sids"]
    n = len(sids)
    out = {t: {"E_Q_cross": np.full(n, np.nan), "T_Q_cross": np.full(n, np.nan)} for t in weights}
    KH = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    model = bundle["model"]
    for i, sid in enumerate(sids):
        x0, J = load_q_frame(shared, model, int(sid))
        ai = bundle["sid_to_ai"][int(sid)]
        rec = q_for_weights(bundle["X"], bundle["neigh"][i], x0, J, weights, ai=ai, n_splits=n_splits)
        KH[i] = rec.get("K_H_cross_recon", np.nan)
        ok[i] = bool(rec.get("ok"))
        for t, vals in rec.get("per_target", {}).items():
            out[t]["E_Q_cross"][i] = vals.get("E_Q_cross", np.nan)
            out[t]["T_Q_cross"][i] = vals.get("T_Q_cross", np.nan)
    return {"per_target": out, "K_H_cross_recon": KH, "ok": ok, "model": model}
