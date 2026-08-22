"""Load frozen embeddings, OOF preds, folds, neighbours, K_H_cross, controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_curvature_probe_submission_validation.schema import (
    PRIMARY,
    assert_not_catalog_vector,
    assert_probe_performance,
)

from .config import CATALOG_FIELD, CONTROLS, MODEL, PRIMARY_D, PRIMARY_K, SOURCE_CPRS, SOURCE_MM, ExpConfig
from .io_util import platonic_root, resolve_path


def load_bundle(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    cprs = resolve_path(root, SOURCE_CPRS)
    X = load_model_X(mm, MODEL)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_all = [int(s) for s in anchors["anchors_sample_id"]]
    sid_to_ai = {int(s): i for i, s in enumerate(sid_all)}
    anchors_local = np.asarray(anchors["anchors_local"], dtype=int)

    folds = pd.read_parquet(mm / "sample_folds.parquet")
    y = folds["y_mag_r_desi"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    sample_id_row = folds["sample_id"].to_numpy(int) if "sample_id" in folds.columns else np.arange(len(folds))
    oof = np.load(mm / "global_probes" / "oof_predictions" / f"{MODEL}_{CATALOG_FIELD}.npz")
    yhat = np.asarray(oof["oof"], dtype=float).reshape(-1)
    if len(yhat) != len(y):
        raise RuntimeError(f"OOF length {len(yhat)} != labels {len(y)}")

    # Global fold weights if stored (optional)
    global_W = None
    gw = mm / "global_probes" / f"{MODEL}_{CATALOG_FIELD}_weights.npz"
    if gw.exists():
        z = np.load(gw)
        global_W = {int(k.split("_")[-1]): z[k] for k in z.files if k.startswith("coef")}

    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    panel = panel[(panel.d == PRIMARY_D)].copy()
    if "k" in panel.columns:
        panel = panel[panel.k == PRIMARY_K]
    elif "scale_k" in panel.columns:
        panel = panel[panel.scale_k == PRIMARY_K]
    panel = panel.drop_duplicates("sample_id")

    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == MODEL) & (geo.target == CATALOG_FIELD) & (geo.neighbourhood == "model") & (geo.scale_k == PRIMARY_K)]
    geo = geo.drop_duplicates("sample_id").set_index("sample_id")

    sids = sorted(int(s) for s in panel.sample_id.unique() if int(s) in sid_to_ai and int(s) in geo.index)
    sids = sids[: cfg.n_anc()]

    catalog = y.copy()  # labels are catalog mag; association outcomes must not be this vector at anchors
    assert_probe_performance(PRIMARY.value)

    return {
        "root": root,
        "mm": mm,
        "cprs": cprs,
        "X": X,
        "y": y,
        "yhat": yhat,
        "fold": fold,
        "sample_id_row": sample_id_row,
        "neigh": neigh,
        "sid_to_ai": sid_to_ai,
        "anchors_local": anchors_local,
        "panel": panel.set_index("sample_id"),
        "geo": geo,
        "sids": sids,
        "catalog": catalog,
        "global_W": global_W,
    }


def kh_controls(bundle: dict, sid: int) -> dict[str, float]:
    panel = bundle["panel"]
    geo = bundle["geo"]
    kh = float(panel.loc[int(sid), "K_H_cross"]) if int(sid) in panel.index else float("nan")
    out = {"K_H_cross": kh, "sample_id": int(sid)}
    for c in CONTROLS:
        if c in geo.columns:
            out[c] = float(geo.loc[int(sid), c])
        elif c.replace("local_evaluation_count", "local_evaluation_count") in geo.columns:
            out[c] = float(geo.loc[int(sid), c])
        else:
            out[c] = float("nan")
    return out
