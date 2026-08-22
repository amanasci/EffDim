"""Load frozen embeddings, ordered neighbours, 128 anchors, and probe fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_curvature_probe_submission_validation.pipeline import hash_select_cprs
from geometry.physics_curvature_probe_submission_validation.schema import (
    PRIMARY,
    assert_not_catalog_vector,
    assert_probe_performance,
)

from .config import (
    CATALOG_FIELD,
    CONTROLS,
    MODEL,
    N_ANCHORS,
    PRIMARY_K,
    SEED,
    SOURCE_CPRS,
    SOURCE_MM,
    SOURCE_VAL,
    ExpConfig,
)
from .io_util import platonic_root, resolve_path


def _pack_neigh(pack: dict) -> np.ndarray:
    for k in ("neigh", "neighbors", "idx", "knn"):
        if k in pack:
            return np.asarray(pack[k], dtype=np.int64)
    raise KeyError(f"no neighbour matrix in pack keys {list(pack)}")


def load_bundle(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    cprs = resolve_path(root, SOURCE_CPRS)
    val = resolve_path(root, SOURCE_VAL)
    X = load_model_X(mm, MODEL)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax{PRIMARY_K}.npz"))
    neigh = _pack_neigh(pack)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_all = [int(s) for s in anchors["anchors_sample_id"]]
    sid_to_ai = {int(s): i for i, s in enumerate(sid_all)}
    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    sids = sorted(int(s) for s in panel.sample_id.unique())
    n_take = cfg.n_anc()
    scale_sids = hash_select_cprs(sids, N_ANCHORS if not cfg.smoke else n_take, seed=SEED)
    scale_sids = scale_sids[:n_take]
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == MODEL) & (geo.target == CATALOG_FIELD) & (geo.neighbourhood == "model")]
    oof_p = mm / "global_probes" / "oof_predictions" / f"{MODEL}_{CATALOG_FIELD}.npz"
    z = np.load(oof_p)
    yhat = np.asarray(z["oof"], dtype=float).reshape(-1)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    y = folds["y_mag_r_desi"].to_numpy(float) if "y_mag_r_desi" in folds.columns else folds.iloc[:, 0].to_numpy(float)
    catalog = None
    for col in ("mag_r_desi", "y_mag_r_desi_catalog", "mag_r_desi_catalog_value"):
        if col in folds.columns:
            catalog = folds[col].to_numpy(float)
            break
    if catalog is None:
        # Fall back to label vector only to exercise the identity guard; R² must not match it.
        catalog = y.copy()
    folds_sample_id = folds["sample_id"].to_numpy(int) if "sample_id" in folds.columns else None
    assert_probe_performance(PRIMARY.value)
    return {
        "root": root,
        "mm": mm,
        "cprs": cprs,
        "val": val,
        "X": X,
        "neigh": neigh,
        "sid_to_ai": sid_to_ai,
        "all_sids": sids,
        "scale_sids": [int(s) for s in scale_sids],
        "geo": geo,
        "y": y,
        "yhat": yhat,
        "catalog": catalog,
        "folds_sample_id": folds_sample_id,
        "panel": panel,
        "anchors_local": np.asarray(anchors["anchors_local"], dtype=int)
        if "anchors_local" in anchors.files
        else np.asarray(anchors["anchors_sample_id"], dtype=int),
    }


def geo_row(geo: pd.DataFrame, sid: int, k: int) -> pd.Series:
    g = geo[(geo.sample_id == int(sid)) & (geo.scale_k == int(k))]
    if not len(g) and "k" in geo.columns:
        g = geo[(geo.sample_id == int(sid)) & (geo.k == int(k))]
    if not len(g):
        raise KeyError(f"missing geo sample_id={sid} k={k}")
    return g.iloc[0]


def control_row(geo: pd.DataFrame, sid: int, k: int) -> dict[str, float]:
    r = geo_row(geo, sid, k)
    out = {}
    aliases = {
        "log_knn_radius": ("log_knn_radius", "log_knn_radius"),
        "local_label_variance": ("local_label_variance", "local_label_variance"),
        "local_evaluation_count": ("local_evaluation_count", "local_evaluation_count"),
    }
    for canon, opts in aliases.items():
        for c in opts:
            if c in r.index:
                out[canon] = float(r[c])
                break
        else:
            out[canon] = float("nan")
    return out


load_bundle = load_bundle
