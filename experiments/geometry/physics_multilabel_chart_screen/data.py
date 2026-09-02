"""Load frozen charts, neighbourhoods, and per-label OOF probes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_quadratic_label_chart_alignment.io_util import platonic_root, resolve_path

from .config import (
    CONTROLS,
    MODEL,
    PRIMARY_D,
    PRIMARY_K,
    SOURCE_CPRS,
    SOURCE_MM,
    SOURCE_NDC,
    ScreenConfig,
)
from .inventory import assert_not_desi_resurrected, record_for


def load_shared(cfg: ScreenConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    cprs = resolve_path(root, SOURCE_CPRS)
    ndc = resolve_path(root, SOURCE_NDC)

    X = load_model_X(mm, MODEL)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_all = [int(s) for s in anchors["anchors_sample_id"]]
    sid_to_ai = {int(s): i for i, s in enumerate(sid_all)}

    folds = pd.read_parquet(mm / "sample_folds.parquet")
    fold = folds["fold"].to_numpy(int)

    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    panel = panel[(panel.d == PRIMARY_D)].copy()
    if "k" in panel.columns:
        panel = panel[panel.k == PRIMARY_K]
    elif "scale_k" in panel.columns:
        panel = panel[panel.scale_k == PRIMARY_K]
    panel = panel.drop_duplicates("sample_id").set_index("sample_id")

    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[
        (geo.model == MODEL)
        & (geo.target == "mag_r_desi")
        & (geo.neighbourhood == "model")
        & (geo.scale_k == PRIMARY_K)
    ].drop_duplicates("sample_id").set_index("sample_id")

    sids = sorted(int(s) for s in panel.index if int(s) in sid_to_ai and int(s) in geo.index)
    sids = sids[: cfg.n_anc()]

    return {
        "root": root,
        "mm": mm,
        "cprs": cprs,
        "ndc": ndc,
        "X": X,
        "fold": fold,
        "neigh": neigh,
        "sid_to_ai": sid_to_ai,
        "panel": panel,
        "geo": geo,
        "sids": sids,
        "folds": folds,
    }


def load_label_vectors(shared: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    assert_not_desi_resurrected(field)
    record_for(field)
    mm = shared["mm"]
    folds = shared["folds"]
    ycol = f"y_{field}"
    if ycol not in folds.columns:
        raise KeyError(f"{ycol} missing from sample_folds.parquet")
    y = folds[ycol].to_numpy(float)
    oof_p = mm / "global_probes" / "oof_predictions" / f"{MODEL}_{field}.npz"
    if not oof_p.exists():
        raise FileNotFoundError(oof_p)
    yhat = np.asarray(np.load(oof_p)["oof"], dtype=float).reshape(-1)
    if len(y) != len(yhat):
        raise RuntimeError(f"{field}: OOF length {len(yhat)} != labels {len(y)}")
    return y, yhat


def kh_controls(shared: dict, sid: int, local_var: float, n_eval: float) -> dict[str, float]:
    panel = shared["panel"]
    geo = shared["geo"]
    out = {
        "sample_id": int(sid),
        "K_H_cross": float(panel.loc[int(sid), "K_H_cross"]),
        "log_knn_radius": float(geo.loc[int(sid), "log_knn_radius"]),
        "local_label_variance": float(local_var),
        "local_evaluation_count": float(n_eval),
    }
    for c in CONTROLS:
        if c not in out:
            out[c] = float(geo.loc[int(sid), c])
    return out


