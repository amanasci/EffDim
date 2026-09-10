"""Shared object universe, folds, embeddings, neighbours. Align by sample_id."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X

from .config import CATALOG_FIELD, CONTROLS, POSITIVE_CONTROL, PRIMARY_D, PRIMARY_K, SOURCE_CPRS, SOURCE_LPA, SOURCE_MM, ExpConfig
from .io_util import platonic_root, resolve_path, write_json


def load_shared(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    sample_id_row = folds["sample_id"].to_numpy(int)
    local_index = folds["local_index"].to_numpy(int) if "local_index" in folds.columns else np.arange(len(folds))
    sid_to_row = {int(s): int(i) for s, i in zip(sample_id_row, local_index)}
    y = folds[f"y_{CATALOG_FIELD}"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    orig_sids = [int(s) for s in anchors["anchors_sample_id"]]
    orig_local = np.asarray(anchors["anchors_local"], dtype=int)
    sid_to_ai = {int(s): i for i, s in enumerate(orig_sids)}
    return {
        "root": root,
        "mm": mm,
        "folds": folds,
        "sample_id_row": sample_id_row,
        "sid_to_row": sid_to_row,
        "y": y,
        "fold": fold,
        "orig_sids": orig_sids,
        "orig_local": orig_local,
        "sid_to_ai": sid_to_ai,
        "n_obj": int(len(folds)),
    }


def load_model_bundle(shared: dict, model: str) -> dict[str, Any]:
    mm = shared["mm"]
    X = load_model_X(mm, model)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{model}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    oof = np.load(mm / "global_probes" / "oof_predictions" / f"{model}_{CATALOG_FIELD}.npz")
    yhat = np.asarray(oof["oof"], dtype=float).reshape(-1)
    if len(yhat) != len(shared["y"]) or X.shape[0] != len(shared["y"]):
        raise RuntimeError(f"{model}: embedding/OOF/label length mismatch")
    return {"model": model, "X": X, "neigh": neigh, "yhat": yhat, "pack": pack}


def fill_controls_from_pack(
    shared: dict,
    model: str,
    sids: list[int],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """If local_probe_fields missed a control, compute it from the knn pack + labels."""
    out = df.copy()
    pack = dict(np.load(shared["mm"] / "model_neighbourhoods" / f"{model}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    dists = np.asarray(pack["dists"], dtype=float) if "dists" in pack else None
    y = shared["y"]
    for i, sid in enumerate(out.sample_id.astype(int).tolist()):
        ai = shared["sid_to_ai"].get(int(sid))
        if ai is None:
            continue
        N = neigh[ai, :PRIMARY_K]
        if "log_knn_radius" not in out.columns or not np.isfinite(out.loc[out.index[i], "log_knn_radius"]):
            if dists is not None:
                rho = float(dists[ai, PRIMARY_K - 1])
            else:
                rho = float("nan")
            out.loc[out.index[i], "log_knn_radius"] = float(np.log(max(rho, 1e-12)))
            out.loc[out.index[i], "knn_radius"] = rho
        if "local_label_variance" not in out.columns or not np.isfinite(
            out.loc[out.index[i], "local_label_variance"]
        ):
            yy = y[N]
            out.loc[out.index[i], "local_label_variance"] = float(np.nanvar(yy))
        if "local_evaluation_count" not in out.columns or not np.isfinite(
            out.loc[out.index[i], "local_evaluation_count"]
        ):
            out.loc[out.index[i], "local_evaluation_count"] = float(np.isfinite(y[N]).sum())
    return out


def geo_controls(shared: dict, model: str, sids: list[int]) -> pd.DataFrame:
    geo = pd.read_parquet(shared["mm"] / "local_probe_fields.parquet")
    geo = geo[
        (geo.model == model)
        & (geo.target == CATALOG_FIELD)
        & (geo.neighbourhood == "model")
        & (geo.scale_k == PRIMARY_K)
    ].drop_duplicates("sample_id")
    geo = geo.set_index("sample_id")
    rows = []
    for sid in sids:
        rec = {"sample_id": int(sid), "model": model}
        if int(sid) not in geo.index:
            for c in CONTROLS:
                rec[c] = float("nan")
            rows.append(rec)
            continue
        for c in CONTROLS:
            rec[c] = float(geo.loc[int(sid), c]) if c in geo.columns else float("nan")
        rec["knn_radius"] = float(geo.loc[int(sid), "knn_radius"]) if "knn_radius" in geo.columns else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def freeze_common_anchors(shared: dict, eligible: list[str], cfg: ExpConfig, out) -> dict[str, Any]:
    """Prefer the original 512 ViT-B anchors; they exist in every eligible model here."""
    orig = list(shared["orig_sids"])
    sid_set = set(int(s) for s in shared["sample_id_row"])
    present = [s for s in orig if s in sid_set]
    n_req = cfg.n_anc()
    common = present[:n_req]
    payload = {
        "n_original_anchors": len(orig),
        "n_original_present_in_object_table": len(present),
        "n_common_used": len(common),
        "used_original_512_prefix": bool(common == orig[: len(common)]),
        "min_required": 256,
        "ok": len(common) >= (8 if cfg.smoke else 256),
        "sample_ids": common,
    }
    write_json(out / "common_anchor_manifest.json", payload, force=True)
    write_json(
        out / "common_object_manifest.json",
        {
            "n_objects": shared["n_obj"],
            "n_folds": int(len(set(shared["fold"].tolist()))),
            "target_field": CATALOG_FIELD,
            "sample_id_is_row_index": bool(
                np.array_equal(shared["sample_id_row"], np.arange(shared["n_obj"]))
            ),
            "eligible_models": eligible,
        },
        force=True,
    )
    if not payload["ok"]:
        raise RuntimeError(f"common anchors {len(common)} below minimum")
    return payload


def vitb_frozen_kh(shared: dict, sids: list[int]) -> pd.DataFrame:
    cprs = resolve_path(shared["root"], SOURCE_CPRS)
    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    panel = panel[(panel.d == PRIMARY_D)].copy()
    if "k" in panel.columns:
        panel = panel[panel.k == PRIMARY_K]
    panel = panel.drop_duplicates("sample_id")
    hit = panel[panel.sample_id.isin(sids)][["sample_id", "K_H_cross", "R_H"]].copy()
    hit["model"] = POSITIVE_CONTROL
    hit["source"] = "physics_curvature_probe_rank_sweep"
    return hit.reset_index(drop=True)


def load_lpa_vitb() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = platonic_root()
    lpa = resolve_path(root, SOURCE_LPA)
    imp = pd.read_csv(lpa / "anchor_improvements.csv")
    met = pd.read_parquet(lpa / "anchor_model_metrics.parquet")
    return imp, met
