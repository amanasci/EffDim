"""Read-only access to frozen embeddings, anchors, frames, and G/P probes."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X

from .config import (
    CATALOG_FIELD,
    CONTROLS,
    MODELS,
    PRIMARY_K,
    SOURCE_CMCLA,
    SOURCE_EDM,
    SOURCE_MM,
    SOURCE_NDC,
    ExpConfig,
)
from .io_util import platonic_root, resolve_path


def load_shared(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    cmcla = resolve_path(root, SOURCE_CMCLA)
    ndc = resolve_path(root, SOURCE_NDC)
    edm = resolve_path(root, SOURCE_EDM)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    sample_id_row = folds["sample_id"].to_numpy(int)
    sid_to_row = {int(s): i for i, s in enumerate(sample_id_row)}
    y = folds[f"y_{CATALOG_FIELD}"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    orig_sids = [int(s) for s in anchors["anchors_sample_id"]]
    sid_to_ai = {int(s): i for i, s in enumerate(orig_sids)}
    man = json.loads((cmcla / "common_anchor_manifest.json").read_text())
    sids = [int(s) for s in man["sample_ids"]][: cfg.n_anc()]
    return {
        "root": root,
        "mm": mm,
        "cmcla": cmcla,
        "ndc": ndc,
        "edm": edm,
        "folds": folds,
        "sample_id_row": sample_id_row,
        "sid_to_row": sid_to_row,
        "y": y,
        "fold": fold,
        "orig_sids": orig_sids,
        "sid_to_ai": sid_to_ai,
        "sids": sids,
        "n_obj": int(len(folds)),
    }


def models_used(cfg: ExpConfig) -> list[str]:
    if cfg.models_override:
        return list(cfg.models_override)
    return list(MODELS)


def load_model_bundle(shared: dict, model: str) -> dict[str, Any]:
    mm = shared["mm"]
    X = load_model_X(mm, model)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{model}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    return {"model": model, "X": X, "neigh": neigh, "pack": pack}


def load_frozen_probes(shared: dict, model: str, sids: list[int]) -> pd.DataFrame:
    path = shared["cmcla"] / "probes" / f"{model}_anchor_metrics.parquet"
    df = pd.read_parquet(path)
    hit = df[df.sample_id.astype(int).isin(sids)].copy()
    need = ["sample_id", "mse_G", "r2_G", "mse_P", "r2_P", "delta_adapt"] + list(CONTROLS)
    missing = [c for c in need if c not in hit.columns]
    if missing:
        raise RuntimeError(f"{model}: frozen probe table missing {missing}")
    hit = hit.drop_duplicates("sample_id")
    # drop frozen KH so newly computed curvature columns cannot collide on merge
    drop_kh = [c for c in ("K_H_cross", "R_H") if c in hit.columns]
    return hit.drop(columns=drop_kh).reset_index(drop=True)


def load_frozen_kh(shared: dict, model: str, sids: list[int]) -> pd.DataFrame:
    path = shared["cmcla"] / "probes" / f"{model}_anchor_metrics.parquet"
    df = pd.read_parquet(path)
    hit = df[df.sample_id.astype(int).isin(sids)][["sample_id", "K_H_cross", "R_H"]].copy()
    return hit.drop_duplicates("sample_id").reset_index(drop=True)


def load_ndc_full_vitb(sids: list[int], shared: dict) -> pd.DataFrame:
    path = shared["ndc"] / "nested_curvature_metrics.parquet"
    df = pd.read_parquet(path)
    df = df[(df.d == 16) & (df.k == PRIMARY_K) & (df.model == "vit_base")]
    g = (
        df.groupby("sample_id", as_index=False)
        .agg(
            K_H_cross=("K_H_cross", "mean"),
            K_aniso_cross=("K_aniso_cross", "mean"),
            K_dir_cross=("K_dir_cross", "mean"),
            R_H=("R_H", "mean"),
            R_B0=("R_B0", "mean"),
            R_BS=("R_BS", "mean"),
            n_splits=("split", "nunique"),
        )
    )
    g = g[g.sample_id.astype(int).isin(sids)].copy()
    g["model"] = "vit_base"
    g["source"] = "physics_nested_dimension_curvature"
    return g.reset_index(drop=True)


def load_frame(shared: dict, model: str, sid: int) -> tuple[np.ndarray, np.ndarray] | None:
    jp = shared["cmcla"] / "geometry" / model / f"J_{int(sid)}.npz"
    if not jp.exists():
        return None
    z = np.load(jp)
    return np.asarray(z["x0"], dtype=np.float64), np.asarray(z["J"], dtype=np.float64)
