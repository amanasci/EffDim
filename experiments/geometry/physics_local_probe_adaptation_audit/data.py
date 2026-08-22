"""Load frozen LPA outputs and alignment inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X

from .config import MODEL, PRIMARY_K, SOURCE_ALIGN, SOURCE_GEO, SOURCE_LPA, SOURCE_MM, SOURCE_NDC, TARGET
from .io_util import platonic_root, resolve_path


def load_audit_tables() -> dict[str, Any]:
    root = platonic_root()
    lpa = resolve_path(root, SOURCE_LPA)
    mm = resolve_path(root, SOURCE_MM)
    imp = pd.read_csv(lpa / "anchor_improvements.csv")
    met = pd.read_parquet(lpa / "anchor_model_metrics.parquet")
    diag = pd.read_parquet(lpa / "probe_direction_diagnostics.parquet")
    primary = __import__("json").loads((lpa / "primary_inference.json").read_text())
    parity_lpa = __import__("json").loads((lpa / "parity.json").read_text())

    X = load_model_X(mm, MODEL)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_all = [int(s) for s in anchors["anchors_sample_id"]]
    sid_to_ai = {int(s): i for i, s in enumerate(sid_all)}
    ai_to_sid = {i: int(s) for s, i in sid_to_ai.items()}

    gw = np.load(resolve_path(root, SOURCE_ALIGN) / "global_probe_weights.npz")
    w_g = np.asarray(gw[f"w_{TARGET}"], dtype=np.float64)
    b_g = float(np.asarray(gw[f"b_{TARGET}"]).reshape(-1)[0])

    y = folds["y_mag_r_desi"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    sid_row = folds["sample_id"].to_numpy(int) if "sample_id" in folds.columns else np.arange(len(folds))
    yhat = np.asarray(np.load(mm / "global_probes" / "oof_predictions" / f"{MODEL}_{TARGET}.npz")["oof"], float).reshape(-1)

    return {
        "root": root,
        "lpa": lpa,
        "mm": mm,
        "imp": imp,
        "met": met,
        "diag": diag,
        "primary_lpa": primary,
        "parity_lpa": parity_lpa,
        "X": X,
        "y": y,
        "yhat": yhat,
        "fold": fold,
        "sid_row": sid_row,
        "neigh": neigh,
        "sid_to_ai": sid_to_ai,
        "ai_to_sid": ai_to_sid,
        "w_g": w_g,
        "b_g": b_g,
        "geo_cache": resolve_path(root, SOURCE_GEO),
        "h_dir": resolve_path(root, SOURCE_NDC) / "H_vectors",
    }
