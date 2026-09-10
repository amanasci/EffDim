"""Phase 0: locate frozen artifacts, sample-ID parity, optional historical decoder field."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix
from geometry.physics_curvature_probe_submission_validation.schema import (
    PRIMARY,
    assert_not_catalog_vector,
    assert_probe_performance,
)

from .config import (
    CATALOG_FIELD,
    CONTROLS,
    HISTORICAL_DECODER_R2_RHO,
    HISTORICAL_DECODER_RHO_ATOL,
    MODEL,
    N_ANCHORS,
    N_FOLDS,
    PARITY_ATOL,
    PARITY_DMSE,
    PARITY_MSE_G,
    PARITY_R2,
    PRIMARY_D,
    PRIMARY_K,
    SOURCE_CMCLA,
    SOURCE_CPRS,
    SOURCE_FCR,
    SOURCE_LPA,
    SOURCE_MM,
    SOURCE_NDC,
    SOURCE_QLCA,
    ExpConfig,
)
from .io_util import file_meta, platonic_root, resolve_path, write_json


def load_bundle(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    cprs = resolve_path(root, SOURCE_CPRS)
    lpa = resolve_path(root, SOURCE_LPA)
    cmcla = resolve_path(root, SOURCE_CMCLA)
    fcr = resolve_path(root, SOURCE_FCR)
    qlca = resolve_path(root, SOURCE_QLCA)
    ndc = resolve_path(root, SOURCE_NDC)

    X = load_model_X(mm, MODEL)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    sample_id_row = folds["sample_id"].to_numpy(int)
    sid_to_row = {int(s): int(i) for i, s in enumerate(sample_id_row)}
    y = folds[f"y_{CATALOG_FIELD}"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    orig_sids = np.asarray(anchors["anchors_sample_id"], dtype=np.int64)
    orig_local = np.asarray(anchors["anchors_local"], dtype=np.int64)
    sid_to_ai = {int(s): i for i, s in enumerate(orig_sids.tolist())}

    oof_p = mm / "global_probes" / "oof_predictions" / f"{MODEL}_{CATALOG_FIELD}.npz"
    oof = np.load(oof_p)
    yhat = np.asarray(oof["oof"], dtype=float).reshape(-1)

    probes = pd.read_parquet(cmcla / "probes" / f"{MODEL}_anchor_metrics.parquet")
    lpa_imp = pd.read_csv(lpa / "anchor_improvements.csv")
    lpa_met = pd.read_parquet(lpa / "anchor_model_metrics.parquet")
    fcr_tab = pd.read_parquet(fcr / "tables" / f"{MODEL}_per_anchor_curvature.parquet")
    cprs_tab = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    cprs_tab = cprs_tab[(cprs_tab.d == PRIMARY_D) & (cprs_tab.k == PRIMARY_K)].drop_duplicates("sample_id")
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[
        (geo.model == MODEL)
        & (geo.target == CATALOG_FIELD)
        & (geo.neighbourhood == "model")
        & (geo.scale_k == PRIMARY_K)
    ].drop_duplicates("sample_id")

    sids = [int(s) for s in orig_sids.tolist()][: cfg.n_anc()]
    return {
        "root": root,
        "mm": mm,
        "cprs": cprs,
        "lpa": lpa,
        "cmcla": cmcla,
        "fcr": fcr,
        "qlca": qlca,
        "ndc": ndc,
        "X": np.asarray(X),
        "folds": folds,
        "sample_id_row": sample_id_row,
        "sid_to_row": sid_to_row,
        "y": y,
        "fold": fold,
        "yhat": yhat,
        "orig_sids": orig_sids,
        "orig_local": orig_local,
        "sid_to_ai": sid_to_ai,
        "probes": probes,
        "lpa_imp": lpa_imp,
        "lpa_met": lpa_met,
        "fcr_tab": fcr_tab,
        "cprs_tab": cprs_tab,
        "geo": geo,
        "sids": sids,
        "oof_path": oof_p,
        "n_obj": int(len(folds)),
    }


def aligned_outcomes(bundle: dict) -> pd.DataFrame:
    """Join G/P, Q scalars, and controls by sample_id. Never by row position."""
    sids = [int(s) for s in bundle["sids"]]
    gp = bundle["probes"].copy()
    gp["sample_id"] = gp["sample_id"].astype(int)
    gp = gp[gp.sample_id.isin(sids)].drop_duplicates("sample_id")
    lpa = bundle["lpa_imp"].copy()
    lpa["sample_id"] = lpa["sample_id"].astype(int)
    lpa = lpa[lpa.sample_id.isin(sids)].drop_duplicates("sample_id")
    fcr = bundle["fcr_tab"].copy()
    fcr["sample_id"] = fcr["sample_id"].astype(int)
    fcr = fcr[fcr.sample_id.isin(sids)].drop_duplicates("sample_id")
    cprs = bundle["cprs_tab"].copy()
    cprs["sample_id"] = cprs["sample_id"].astype(int)
    cprs = cprs[cprs.sample_id.isin(sids)].drop_duplicates("sample_id")
    geo = bundle["geo"].copy()
    geo["sample_id"] = geo["sample_id"].astype(int)
    geo = geo[geo.sample_id.isin(sids)].drop_duplicates("sample_id")
    met = bundle["lpa_met"].copy()
    met["sample_id"] = met["sample_id"].astype(int)
    met = met[met.sample_id.isin(sids)].drop_duplicates("sample_id")

    df = pd.DataFrame({"sample_id": sids})
    gp_cols = ["sample_id", "r2_G", "r2_P", "mse_G", "mse_P", "delta_adapt"]
    for c in ("mae_G", "mae_P", "mse_T", "r2_T", "n_eval"):
        if c in gp.columns:
            gp_cols.append(c)
    df = df.merge(gp[gp_cols], on="sample_id", how="inner")
    keep_lpa = [c for c in ("dMAE_G_to_P", "dR2_G_to_P") if c in lpa.columns]
    if keep_lpa:
        df = df.merge(lpa[["sample_id", *keep_lpa]], on="sample_id", how="left")
    qcols = [c for c in ("K_H_cross", "K_dir_cross", "K_aniso_cross", "R_H") if c in fcr.columns]
    df = df.merge(fcr[["sample_id", *qcols]], on="sample_id", how="left")
    if "K_H_cross" not in df.columns or df["K_H_cross"].isna().all():
        df = df.merge(cprs[["sample_id", "K_H_cross", "K_dir_cross"]], on="sample_id", how="left", suffixes=("", "_cprs"))
    ctrl_src = geo if all(c in geo.columns for c in CONTROLS) else lpa
    for c in CONTROLS:
        if c not in df.columns:
            df = df.merge(ctrl_src[["sample_id", c]], on="sample_id", how="left")
    if "n_eval" in met.columns and "local_evaluation_count" in df.columns:
        pass
    df["delta_adapt"] = df["mse_G"] - df["mse_P"]
    # catalog magnitude at the anchor object (for the forbidden-substitution test only)
    cat = []
    fold_ids = []
    rows = []
    for s in df.sample_id.astype(int):
        r = bundle["sid_to_row"].get(int(s))
        cat.append(float(bundle["y"][r]) if r is not None else float("nan"))
        fold_ids.append(int(bundle["fold"][r]) if r is not None else -1)
        rows.append(r if r is not None else -1)
    df["mag_r_desi_catalog_value"] = cat
    df["global_oof_fold"] = fold_ids
    df["row_index"] = rows
    return df.reset_index(drop=True)


def reuse_manifest(bundle: dict) -> dict[str, Any]:
    mm = bundle["mm"]
    files = {
        "vit_b_embeddings": mm / "prepare" / "models" / f"{MODEL}.npz",
        "anchors": mm / "prepare" / "anchors.npz",
        "folds": mm / "sample_folds.parquet",
        "oof": bundle["oof_path"],
        "local_probe_fields": mm / "local_probe_fields.parquet",
        "cprs_rank_curve": bundle["cprs"] / "per_anchor_rank_curve.parquet",
        "lpa_improvements": bundle["lpa"] / "anchor_improvements.csv",
        "lpa_metrics": bundle["lpa"] / "anchor_model_metrics.parquet",
        "cmcla_vit_base": bundle["cmcla"] / "probes" / f"{MODEL}_anchor_metrics.parquet",
        "fcr_vit_base": bundle["fcr"] / "tables" / f"{MODEL}_per_anchor_curvature.parquet",
        "qlca_risks": bundle["qlca"] / "anchor_risks.csv",
        "decoder_impl": Path(__file__).resolve().parents[3] / "notebooks" / "pu_manifold" / "decoder_curvature.py",
        "cae_impl": Path(__file__).resolve().parents[3] / "notebooks" / "pu_manifold" / "cae.py",
    }
    X = bundle["X"]
    folds = bundle["folds"]
    return {
        "n_objects": int(bundle["n_obj"]),
        "X_shape": list(np.asarray(X).shape),
        "X_dtype": str(np.asarray(X).dtype),
        "median_embedding_norm": float(np.median(np.linalg.norm(np.asarray(X), axis=1))),
        "n_frozen_anchors": int(len(bundle["orig_sids"])),
        "n_used_anchors": int(len(bundle["sids"])),
        "n_folds": int(len(set(bundle["fold"].tolist()))),
        "fold_ids": sorted(int(v) for v in set(bundle["fold"].tolist())),
        "sample_id_is_row_index": bool(np.array_equal(bundle["sample_id_row"], np.arange(bundle["n_obj"]))),
        "align_by": "sample_id",
        "d": PRIMARY_D,
        "k": PRIMARY_K,
        "model": MODEL,
        "target_field": CATALOG_FIELD,
        "estimator": "pointwise_sphere_residual_decoder_curvature",
        "historical_full_is_control_only": True,
        "Q_is_empirical_statistic_only": True,
        "files": {k: file_meta(p) for k, p in files.items()},
        "fold_table_columns": list(folds.columns),
    }


def run_parity(bundle: dict, df: pd.DataFrame, cfg: ExpConfig, out) -> dict[str, Any]:
    assert_probe_performance(PRIMARY.value)
    Z = control_matrix(df)
    a_r2 = associate(df.K_H_cross.to_numpy(float), df.r2_G.to_numpy(float), Z)
    a_mse = associate(df.K_H_cross.to_numpy(float), df.mse_G.to_numpy(float), Z)
    a_dm = associate(df.K_H_cross.to_numpy(float), df.delta_adapt.to_numpy(float), Z)
    cat = df.mag_r_desi_catalog_value.to_numpy(float)
    assert_not_catalog_vector(df.r2_G.to_numpy(float), cat)
    assert_not_catalog_vector(df.mse_G.to_numpy(float), cat)
    assert_not_catalog_vector(df.r2_P.to_numpy(float), cat)
    sid_ok = bool(np.array_equal(df.sample_id.to_numpy(int), np.asarray(bundle["sids"][: len(df)], dtype=int))) or set(df.sample_id.astype(int)) <= set(int(s) for s in bundle["orig_sids"])
    overlap0 = True
    if "overlap_any" in bundle["probes"].columns:
        sub = bundle["probes"][bundle["probes"].sample_id.astype(int).isin(df.sample_id.astype(int))]
        overlap0 = bool((~sub.overlap_any.astype(bool)).all())
    hist = _historical_decoder_optional(bundle, df)
    report = {
        "ok": True,
        "n": int(len(df)),
        "rho_ctl_KH_R2G": a_r2,
        "rho_ctl_KH_MSEG": a_mse,
        "rho_ctl_KH_DeltaAdapt": a_dm,
        "expected": {"r2": PARITY_R2, "mse_G": PARITY_MSE_G, "dMSE": PARITY_DMSE},
        "match_r2": abs(float(a_r2["controlled"]) - PARITY_R2) <= PARITY_ATOL,
        "match_mse_G": abs(float(a_mse["controlled"]) - PARITY_MSE_G) <= PARITY_ATOL,
        "match_dMSE": abs(float(a_dm["controlled"]) - PARITY_DMSE) <= PARITY_ATOL,
        "n_folds": int(len(set(df.global_oof_fold.tolist()))),
        "zero_overlap": overlap0,
        "sample_id_alignment_ok": bool(sid_ok and len(df) == len(set(df.sample_id))),
        "catalog_not_used_as_outcome": True,
        "historical_decoder_field": hist,
        "note": "Outcomes are frozen OOF probe metrics, never mag_r_desi_catalog_value.",
    }
    report["ok"] = bool(
        report["match_r2"]
        and report["match_mse_G"]
        and report["match_dMSE"]
        and report["n_folds"] == N_FOLDS
        and (cfg.smoke or report["n"] >= 500)
        and report["sample_id_alignment_ok"]
    )
    write_json(out / "parity.json", report, force=True)
    return report


def _historical_decoder_optional(bundle: dict, df: pd.DataFrame) -> dict[str, Any]:
    """Reproduce ~+0.328 if a compatible historical decoder field exists. Optional."""
    root = bundle["root"]
    candidates = [
        root / "outputs/geometry/physics_ae_local_patch_scale_match" / "per_object_curvature.parquet",
        root / "outputs/geometry/physics_ae_local_patch_scale_match" / "anchor_curvature.parquet",
        bundle["mm"] / "decoder_curvature_vit_base.parquet",
    ]
    found = [p for p in candidates if p.exists()]
    scale = root / "outputs/geometry/physics_ae_local_patch_scale_match" / "summary.json"
    note = {
        "available": False,
        "reason": "no compatible historical pointwise decoder field on the frozen 512 anchors",
        "searched": [str(p) for p in candidates],
        "scale_match_summary_exists": scale.exists(),
        "expected_rho_approx": HISTORICAL_DECODER_R2_RHO,
        "identities_to_check_if_found": [
            "raw-decoder full H^E",
            "normalized-decoder residual H^S",
            "norm vs squared norm",
            "averaged vs unaveraged trace",
        ],
    }
    if scale.exists():
        import json

        sm = json.loads(scale.read_text())
        note["scale_match_rho_point_R2"] = sm.get("rho_point_R2")
        note["scale_match_protocol"] = (
            "PlainAutoEncoder 600 epochs, sphere-projected H_tan at evaluation; "
            "not the 400-epoch raw-decode reproduction protocol; single seed; not reused."
        )
    if not found:
        return note
    # If a table exists, try sample_id join
    try:
        tab = pd.read_parquet(found[0]) if found[0].suffix == ".parquet" else pd.read_csv(found[0])
    except Exception as exc:
        note["read_error"] = str(exc)
        return note
    if "sample_id" not in tab.columns:
        note["available"] = False
        note["reason"] = "historical table missing sample_id"
        return note
    col = None
    for c in ("H_norm", "C_H", "H_tan", "H_E_norm", "H_full"):
        if c in tab.columns:
            col = c
            break
    if col is None:
        note["reason"] = "no H-norm column"
        return note
    m = df.merge(tab[["sample_id", col]].drop_duplicates("sample_id"), on="sample_id", how="inner")
    if len(m) < 32:
        note["reason"] = f"join too small n={len(m)}"
        return note
    Z = control_matrix(m)
    a = associate(m[col].to_numpy(float), m.r2_G.to_numpy(float), Z)
    note.update(
        {
            "available": True,
            "path": str(found[0]),
            "column": col,
            "n": int(len(m)),
            "rho_ctl_vs_R2G": a,
            "matches_plus_0.328": abs(float(a["controlled"]) - HISTORICAL_DECODER_R2_RHO) <= HISTORICAL_DECODER_RHO_ATOL,
        }
    )
    return note
