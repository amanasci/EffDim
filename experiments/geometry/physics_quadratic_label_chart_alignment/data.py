"""Load frozen MM / CPRS / NDC / LPA artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X

from .config import (
    CATALOG_FIELD,
    CONTROLS,
    MODEL,
    PRIMARY_D,
    PRIMARY_K,
    SOURCE_CPRS,
    SOURCE_LPA,
    SOURCE_MM,
    SOURCE_NDC,
    ExpConfig,
)
from .features import bs_prod_to_frob, n_quad
from .io_util import file_sha256, platonic_root, resolve_path


def load_bundle(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    cprs = resolve_path(root, SOURCE_CPRS)
    ndc = resolve_path(root, SOURCE_NDC)
    lpa = resolve_path(root, SOURCE_LPA)

    X = load_model_X(mm, MODEL)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_all = [int(s) for s in anchors["anchors_sample_id"]]
    sid_to_ai = {int(s): i for i, s in enumerate(sid_all)}

    folds = pd.read_parquet(mm / "sample_folds.parquet")
    y = folds["y_mag_r_desi"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    oof = np.load(mm / "global_probes" / "oof_predictions" / f"{MODEL}_{CATALOG_FIELD}.npz")
    yhat = np.asarray(oof["oof"], dtype=float).reshape(-1)

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
        & (geo.target == CATALOG_FIELD)
        & (geo.neighbourhood == "model")
        & (geo.scale_k == PRIMARY_K)
    ].drop_duplicates("sample_id").set_index("sample_id")

    lpa_imp = pd.read_csv(lpa / "anchor_improvements.csv")
    lpa_imp = lpa_imp.drop_duplicates("sample_id").set_index("sample_id")

    sids = sorted(int(s) for s in panel.index if int(s) in sid_to_ai and int(s) in geo.index)
    sids = sids[: cfg.n_anc()]

    return {
        "root": root,
        "mm": mm,
        "cprs": cprs,
        "ndc": ndc,
        "lpa": lpa,
        "X": X,
        "y": y,
        "yhat": yhat,
        "fold": fold,
        "neigh": neigh,
        "sid_to_ai": sid_to_ai,
        "panel": panel,
        "geo": geo,
        "lpa_imp": lpa_imp,
        "sids": sids,
    }


def load_chart(ndc: Path, sid: int) -> dict[str, np.ndarray]:
    z = np.load(ndc / "H_vectors" / f"{int(sid)}.npz")
    x0 = np.asarray(z["x0"], dtype=np.float64)
    J = np.asarray(z["J16"], dtype=np.float64)
    BSA = np.asarray(z["BS16_A"], dtype=np.float64)
    BSB = np.asarray(z["BS16_B"], dtype=np.float64)
    H = np.asarray(z["H16"], dtype=np.float64)
    assert J.shape[1] == PRIMARY_D
    assert BSA.shape[1] == n_quad(PRIMARY_D)
    BS_mean = 0.5 * (BSA + BSB)
    return {
        "x0": x0,
        "J": J,
        "BS_A_prod": BSA,
        "BS_B_prod": BSB,
        "BS_mean_prod": BS_mean,
        "BS_A_frob": bs_prod_to_frob(BSA, PRIMARY_D),
        "BS_B_frob": bs_prod_to_frob(BSB, PRIMARY_D),
        "BS_mean_frob": bs_prod_to_frob(BS_mean, PRIMARY_D),
        "H": H,
    }


def tangent_coords(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    x0n = x0 / max(float(np.linalg.norm(x0)), 1e-12)
    return (Xloc - x0n) @ J


def kh_row(bundle: dict, sid: int) -> dict[str, float]:
    panel = bundle["panel"]
    geo = bundle["geo"]
    out = {"sample_id": int(sid), "K_H_cross": float(panel.loc[int(sid), "K_H_cross"])}
    for c in CONTROLS:
        out[c] = float(geo.loc[int(sid), c])
    return out


def build_reuse_manifest(bundle: dict) -> dict:
    paths = {
        "mm_X": bundle["mm"] / "prepare" / "models" / f"{MODEL}.npz",
        "neigh": bundle["mm"] / "model_neighbourhoods" / f"{MODEL}_kmax{PRIMARY_K}.npz",
        "folds": bundle["mm"] / "sample_folds.parquet",
        "oof": bundle["mm"] / "global_probes" / "oof_predictions" / f"{MODEL}_{CATALOG_FIELD}.npz",
        "cprs": bundle["cprs"] / "per_anchor_rank_curve.parquet",
        "lpa_imp": bundle["lpa"] / "anchor_improvements.csv",
        "ndc_example": bundle["ndc"] / "H_vectors" / f"{bundle['sids'][0]}.npz",
    }
    man = {"n_anchors": len(bundle["sids"]), "files": {}}
    for k, p in paths.items():
        p = Path(p)
        man["files"][k] = {
            "path": str(p),
            "exists": p.exists(),
            "sha16": file_sha256(p) if p.exists() else None,
            "shape": _shape_hint(p) if p.exists() else None,
        }
    man["X_shape"] = list(np.asarray(bundle["X"]).shape)
    man["neigh_shape"] = list(np.asarray(bundle["neigh"]).shape)
    man["n_labels"] = int(len(bundle["y"]))
    man["n_quad_d16"] = n_quad(PRIMARY_D)
    return man


def _shape_hint(p: Path):
    if p.suffix == ".npz":
        z = np.load(p)
        return {k: list(np.asarray(z[k]).shape) for k in list(z.files)[:12]}
    if p.suffix == ".parquet":
        return {"rows": int(pd.read_parquet(p).shape[0]), "cols": int(pd.read_parquet(p).shape[1])}
    return None
