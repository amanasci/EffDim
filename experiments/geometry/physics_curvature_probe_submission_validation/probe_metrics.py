"""Phase 2: recover local OOF probe metrics and test denominator vs error."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.global_probe_curvature_alignment import local_r2_fixed_predictions
from geometry.physics_curvature_probe_rank_sweep.inference import (
    associate,
    control_matrix,
    paired_bootstrap_curves,
    permutation_curves,
)

from .config import CONTROLS, DS, MODEL, PRIMARY_K, SEED
from .pipeline import ValConfig, p_report, resolve_path, write_df, write_json
from .schema import PRIMARY, ProbeTargetId, assert_not_catalog_vector, assert_probe_performance
from .parity import load_catalog


PERM_TARGETS = {
    ProbeTargetId.MAG_R_DESI_LOCAL_OOF_R2,
    ProbeTargetId.MAG_R_DESI_OOF_SSE,
    ProbeTargetId.MAG_R_DESI_OOF_MAE,
    ProbeTargetId.MAG_R_DESI_OOF_MSE,
    ProbeTargetId.MAG_R_DESI_LOCAL_SST,
    ProbeTargetId.MAG_R_DESI_LOCAL_TARGET_VAR,
}
YCOL = {
    ProbeTargetId.MAG_R_DESI_LOCAL_OOF_R2: "r2_local",
    ProbeTargetId.MAG_R_DESI_OOF_SSE: "oof_sse",
    ProbeTargetId.MAG_R_DESI_LOCAL_SST: "local_sst",
    ProbeTargetId.MAG_R_DESI_LOCAL_TARGET_VAR: "local_target_var",
    ProbeTargetId.MAG_R_DESI_OOF_MAE: "oof_mae",
    ProbeTargetId.MAG_R_DESI_OOF_MSE: "oof_mse",
    ProbeTargetId.MAG_R_DESI_OOF_NMSE: "oof_nmse",
    ProbeTargetId.MAG_R_DESI_NORMALIZED_MSE: "normalized_mse",
}


def _load_oof(mm: Path) -> tuple[np.ndarray, np.ndarray]:
    p = mm / "global_probes" / "oof_predictions" / f"{MODEL}_mag_r_desi.npz"
    z = np.load(p)
    yhat = np.asarray(z["oof"], dtype=float).reshape(-1)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    y = folds["y_mag_r_desi"].to_numpy(float)
    if len(y) != len(yhat):
        raise RuntimeError(f"OOF length {len(yhat)} != fold labels {len(y)}")
    return y, yhat


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    m = np.isfinite(y) & np.isfinite(yhat)
    n = int(m.sum())
    if n < 4:
        return {k: float("nan") for k in ("r2_local", "oof_sse", "local_sst", "local_target_var", "oof_mae", "oof_mse", "oof_nmse", "n_eval")}
    yy, yh = y[m], yhat[m]
    ym = float(np.mean(yy))
    sse = float(np.sum((yy - yh) ** 2))
    sst = float(np.sum((yy - ym) ** 2))
    var = float(np.var(yy))
    mse = sse / n
    mae = float(np.mean(np.abs(yy - yh)))
    nmse = mse / max(var, 1e-18)
    r2 = local_r2_fixed_predictions(yy, yh)
    return {
        "r2_local": r2,
        "oof_sse": sse,
        "local_sst": sst,
        "local_target_var": var,
        "oof_mae": mae,
        "oof_mse": mse,
        "oof_nmse": nmse,
        "n_eval": float(n),
    }


def compute_anchor_metrics(root: Path, cfg: ValConfig, *, slice_mode: str = "full") -> pd.DataFrame:
    mm = resolve_path(root, cfg.mm_dir)
    cprs = resolve_path(root, cfg.cprs_dir)
    y, yhat = _load_oof(mm)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=int)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_to_ai = {int(s): i for i, s in enumerate(anchors["anchors_sample_id"])}
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == MODEL) & (geo.target == "mag_r_desi") & (geo.neighbourhood == "model") & (geo.scale_k == PRIMARY_K)]
    geo = geo.drop_duplicates("sample_id").set_index("sample_id")
    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    sids = sorted(set(panel.sample_id.astype(int)) & set(geo.index.astype(int)) & set(sid_to_ai))
    if cfg.smoke:
        sids = sids[:16]
    catalog = load_catalog(root, sids)
    y_r2 = np.asarray([float(geo.loc[s, "local_r2"]) for s in sids], dtype=float)
    assert_not_catalog_vector(y_r2, catalog)

    folds = pd.read_parquet(mm / "sample_folds.parquet")
    fold = folds.fold.to_numpy(int) if "fold" in folds.columns else None
    rows = []
    leak_hits = 0
    self_in_nb = 0
    for sid in sids:
        ai = sid_to_ai[int(sid)]
        N = neigh[ai, :PRIMARY_K]
        if int(anchors["anchors_local"][ai]) in set(N.tolist()):
            self_in_nb += 1
        if slice_mode == "outer_half":
            N = N[PRIMARY_K // 2 :]
        elif slice_mode == "inner_half":
            N = N[: PRIMARY_K // 2]
        rec = _metrics(y[N], yhat[N])
        rec.update(
            {
                "sample_id": int(sid),
                "slice_mode": slice_mode,
                "normalized_mse": float(geo.loc[sid, "normalized_mse"]),
                **{c: float(geo.loc[sid, c]) for c in CONTROLS},
                "r2_geo_cached": float(geo.loc[sid, "local_r2"]),
            }
        )
        if fold is not None and slice_mode == "full":
            rec["n_unique_folds"] = int(len(set(fold[N].tolist())))
        rows.append(rec)
        if abs(rec["r2_local"] - rec["r2_geo_cached"]) > 1e-5 and slice_mode == "full":
            leak_hits += 1
    df = pd.DataFrame(rows)
    df.attrs["leak_r2_mismatch"] = leak_hits
    df.attrs["self_in_neighbourhood"] = self_in_nb
    return df


def _wide(panel: pd.DataFrame, metrics: pd.DataFrame, ycol: str) -> pd.DataFrame:
    base = metrics[["sample_id", ycol, *CONTROLS]].drop_duplicates("sample_id")
    kh = panel.groupby(["sample_id", "d"], as_index=False)["K_H_cross"].mean()
    for d in DS:
        gd = kh[kh.d == d][["sample_id", "K_H_cross"]].rename(columns={"K_H_cross": f"KH{d}"})
        base = base.merge(gd, on="sample_id", how="left")
    return base


def run_probe_validation(root: Path, cfg: ValConfig) -> dict[str, Any]:
    assert_probe_performance(PRIMARY.value)
    out = cfg.resolved(root)
    cprs = resolve_path(root, cfg.cprs_dir)
    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    n_perm, n_boot = cfg.perm_boot()
    metrics_full = compute_anchor_metrics(root, cfg, slice_mode="full")
    metrics_outer = compute_anchor_metrics(root, cfg, slice_mode="outer_half")
    write_df(out / "probe_metrics_full.csv", metrics_full, force=cfg.force)
    write_df(out / "probe_metrics_outer_half.csv", metrics_outer, force=cfg.force)

    r2_ok = bool(np.nanmax(np.abs(metrics_full.r2_local - metrics_full.r2_geo_cached)) < 1e-4)
    assoc_rows = []
    perm_rows = []
    boot_rows = []
    for slice_mode, met in (("full", metrics_full), ("outer_half", metrics_outer)):
        for tid, col in YCOL.items():
            wide = _wide(panel, met, col)
            ds = [d for d in DS if f"KH{d}" in wide.columns]
            for d in ds:
                rec = associate(wide[f"KH{d}"].to_numpy(float), wide[col].to_numpy(float), control_matrix(wide))
                rec.update({"d": int(d), "target_id": tid.value, "slice_mode": slice_mode, "analysis": "confirmatory" if slice_mode == "full" and tid is PRIMARY else "sensitivity"})
                assoc_rows.append(rec)
            if slice_mode == "full" and tid in PERM_TARGETS:
                perm = permutation_curves(wide, ds, ycol=col, x_prefix="KH", n_perm=n_perm, seed=SEED, controlled=True)
                pt = perm["table"].copy()
                pt["target_id"] = tid.value
                pt["slice_mode"] = slice_mode
                pt["p_global"] = perm["p_global"]
                pt["p_global_report"] = p_report(float(perm["p_global"]), n_perm)
                perm_rows.append(pt)
                boot = paired_bootstrap_curves(wide, ds, ycol=col, x_prefix="KH", n_boot=n_boot, seed=SEED)
                bt = boot["table"].copy()
                bt["target_id"] = tid.value
                bt["slice_mode"] = slice_mode
                boot_rows.append(bt)

    assoc = pd.DataFrame(assoc_rows)
    perm = pd.concat(perm_rows, ignore_index=True) if perm_rows else pd.DataFrame()
    boot = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()
    write_df(out / "metric_associations.csv", assoc, force=cfg.force)
    write_df(out / "metric_permutation.csv", perm, force=cfg.force)
    write_df(out / "metric_bootstrap.csv", boot, force=cfg.force)

    # shuffle: y permutation already in permutation_curves; also a one-shot label shuffle vs catalog
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(metrics_full.r2_local.to_numpy(float))
    wide = _wide(panel, metrics_full.assign(r2_shuf=y_shuf), "r2_shuf")
    shuffle = associate(wide["KH16"].to_numpy(float), wide["r2_shuf"].to_numpy(float), control_matrix(wide))

    leakage = {
        "oof_predictions": "global_probes/oof_predictions/vit_base_mag_r_desi.npz",
        "construction": "5-fold ridge; test fold never in training fold",
        "r2_matches_cached_geography": r2_ok,
        "self_may_appear_in_knn": True,
        "self_prediction_still_oof": True,
        "shuffle_d16_controlled": shuffle.get("controlled"),
        "disjoint_neighbour_slice": "outer_half uses neighbours [k/2, k) for probe metrics; K_H unchanged",
    }
    write_json(out / "leakage_report.json", leakage, force=cfg.force)
    return {"r2_ok": r2_ok, "n": int(metrics_full.sample_id.nunique()), "shuffle": shuffle}
