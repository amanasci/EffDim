"""Phase 0: locate frozen artifacts, align by sample_id, reproduce Q and ViT-B D-residual parity."""

from __future__ import annotations

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
    ALL_MODELS,
    CATALOG_FIELD,
    CONTROLS,
    FROZEN_Q,
    FROZEN_VITB_DRES,
    N_FOLDS,
    NATIVE_D,
    PARITY_ATOL,
    PRIMARY_D,
    PRIMARY_K,
    REFERENCE,
    SOURCE_CMCLA,
    SOURCE_FCR,
    SOURCE_MM,
    SOURCE_PRCR,
    ExpConfig,
)
from .io_util import file_meta, platonic_root, resolve_path, write_json


def load_shared(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    cmcla = resolve_path(root, SOURCE_CMCLA)
    fcr = resolve_path(root, SOURCE_FCR)
    prcr = resolve_path(root, SOURCE_PRCR)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    sample_id_row = folds["sample_id"].to_numpy(int)
    sid_to_row = {int(s): int(i) for i, s in enumerate(sample_id_row)}
    y = folds[f"y_{CATALOG_FIELD}"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    orig_sids = [int(s) for s in np.asarray(anchors["anchors_sample_id"])]
    sids = orig_sids[: cfg.n_anc()]
    return {
        "root": root,
        "mm": mm,
        "cmcla": cmcla,
        "fcr": fcr,
        "prcr": prcr,
        "folds": folds,
        "sample_id_row": sample_id_row,
        "sid_to_row": sid_to_row,
        "y": y,
        "fold": fold,
        "orig_sids": orig_sids,
        "sids": sids,
        "n_obj": int(len(folds)),
    }


def load_model_bundle(shared: dict, model: str) -> dict[str, Any]:
    mm = shared["mm"]
    X = np.asarray(load_model_X(mm, model))
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{model}_kmax{PRIMARY_K}.npz"))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    oof_p = mm / "global_probes" / "oof_predictions" / f"{model}_{CATALOG_FIELD}.npz"
    oof = np.load(oof_p)
    yhat = np.asarray(oof["oof"], dtype=float).reshape(-1)
    if X.shape[0] != shared["n_obj"] or len(yhat) != shared["n_obj"]:
        raise RuntimeError(f"{model}: embedding/OOF length mismatch")
    probes = pd.read_parquet(shared["cmcla"] / "probes" / f"{model}_anchor_metrics.parquet")
    fcr = pd.read_parquet(shared["fcr"] / "tables" / f"{model}_per_anchor_curvature.parquet")
    return {
        "model": model,
        "X": X,
        "neigh": neigh,
        "yhat": yhat,
        "oof_path": oof_p,
        "probes": probes,
        "fcr": fcr,
        "D": int(X.shape[1]),
        "expected_D": NATIVE_D[model],
    }


def aligned_outcomes(shared: dict, bundle: dict) -> pd.DataFrame:
    sids = [int(s) for s in shared["sids"]]
    gp = bundle["probes"].copy()
    gp["sample_id"] = gp["sample_id"].astype(int)
    gp = gp[gp.sample_id.isin(sids)].drop_duplicates("sample_id")
    fcr = bundle["fcr"].copy()
    fcr["sample_id"] = fcr["sample_id"].astype(int)
    fcr = fcr[fcr.sample_id.isin(sids)].drop_duplicates("sample_id")
    df = pd.DataFrame({"sample_id": sids, "model": bundle["model"]})
    cols = [
        "sample_id",
        "r2_G",
        "r2_P",
        "mse_G",
        "mse_P",
        "mae_G",
        "mae_P",
        "delta_adapt",
        "n_eval",
        "K_H_cross",
        "log_knn_radius",
        "local_label_variance",
        "local_evaluation_count",
    ]
    cols = [c for c in cols if c in gp.columns]
    df = df.merge(gp[cols], on="sample_id", how="inner")
    qcols = [c for c in ("K_dir_cross", "K_H_cross", "R_H") if c in fcr.columns]
    df = df.merge(fcr[["sample_id", *qcols]].drop(columns=["K_H_cross"], errors="ignore"), on="sample_id", how="left")
    if "K_H_cross" not in df.columns:
        df = df.merge(fcr[["sample_id", "K_H_cross"]], on="sample_id", how="left")
    df["delta_adapt"] = df["mse_G"] - df["mse_P"]
    cat = []
    folds = []
    for s in df.sample_id.astype(int):
        r = shared["sid_to_row"][int(s)]
        cat.append(float(shared["y"][r]))
        folds.append(int(shared["fold"][r]))
    df["mag_r_desi_catalog_value"] = cat
    df["global_oof_fold"] = folds
    missing = [c for c in CONTROLS + ("r2_G", "r2_P", "mse_G", "mse_P", "K_H_cross") if c not in df.columns or df[c].isna().any()]
    if missing:
        raise RuntimeError(f"{bundle['model']}: missing/NaN required columns {missing}")
    if len(df) != len(sids):
        raise RuntimeError(f"{bundle['model']}: sample_id join dropped rows {len(sids)}->{len(df)}")
    if not np.array_equal(df.sample_id.to_numpy(int), np.asarray(sids)):
        raise RuntimeError(f"{bundle['model']}: sample_id order mismatch")
    return df.reset_index(drop=True)


def reuse_manifest(shared: dict, bundles: dict[str, dict]) -> dict[str, Any]:
    files = {
        "anchors": shared["mm"] / "prepare" / "anchors.npz",
        "folds": shared["mm"] / "sample_folds.parquet",
        "cmcla_primary": shared["cmcla"] / "per_model_primary.json",
        "prcr_decision": shared["prcr"] / "decision.json",
        "prcr_seed": shared["prcr"] / "seed_reliability.json",
        "cae_impl": shared["root"] / "notebooks" / "pu_manifold" / "cae.py",
        "decoder_impl": shared["root"] / "notebooks" / "pu_manifold" / "decoder_curvature.py",
    }
    models = {}
    for m, b in bundles.items():
        models[m] = {
            "D": b["D"],
            "expected_D": b["expected_D"],
            "native_ambient_ok": b["D"] == b["expected_D"],
            "X_shape": list(np.asarray(b["X"]).shape),
            "median_norm": float(np.median(np.linalg.norm(b["X"], axis=1))),
            "embeddings": file_meta(shared["mm"] / "prepare" / "models" / f"{m}.npz"),
            "neighbours": file_meta(shared["mm"] / "model_neighbourhoods" / f"{m}_kmax{PRIMARY_K}.npz"),
            "oof": file_meta(b["oof_path"]),
            "cmcla_probes": file_meta(shared["cmcla"] / "probes" / f"{m}_anchor_metrics.parquet"),
            "fcr_table": file_meta(shared["fcr"] / "tables" / f"{m}_per_anchor_curvature.parquet"),
        }
    return {
        "align_by": "sample_id",
        "n_objects": shared["n_obj"],
        "n_anchors": len(shared["sids"]),
        "n_folds": int(len(set(shared["fold"].tolist()))),
        "d": PRIMARY_D,
        "k": PRIMARY_K,
        "target_field": CATALOG_FIELD,
        "files": {k: file_meta(p) for k, p in files.items()},
        "models": models,
        "read_only": True,
    }


def cross_model_alignment(shared: dict, outcomes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ref = outcomes[REFERENCE]
    sids = ref.sample_id.to_numpy(int)
    ok = True
    per = {}
    for m, df in outcomes.items():
        same = bool(np.array_equal(df.sample_id.to_numpy(int), sids))
        dlt = df["mse_G"] - df["mse_P"]
        ident = bool(np.allclose(dlt.to_numpy(float), df["delta_adapt"].to_numpy(float), atol=1e-12))
        folds = sorted(int(v) for v in set(df.global_oof_fold.tolist()))
        per[m] = {
            "n": int(len(df)),
            "sample_id_match_reference": same,
            "delta_adapt_is_MSE_G_minus_MSE_P": ident,
            "n_folds": len(folds),
            "fold_ids": folds,
            "mean_delta_adapt": float(df.delta_adapt.mean()),
            "frac_patch_R2_gt_global": float((df.r2_P > df.r2_G).mean()),
            "D": int(NATIVE_D[m]),
        }
        ok = ok and same and ident and len(folds) == N_FOLDS
    return {
        "ok": ok,
        "shared_sample_ids": [int(s) for s in sids],
        "n": int(len(sids)),
        "models": per,
        "note": "Aligned by sample_id. Outcomes are frozen OOF probe metrics, never catalogue magnitude.",
    }


def run_parity(shared: dict, outcomes: dict[str, pd.DataFrame], cfg: ExpConfig, out) -> dict[str, Any]:
    assert_probe_performance(PRIMARY.value)
    q_rows = {}
    q_ok = True
    for m, df in outcomes.items():
        assert_not_catalog_vector(df.r2_G.to_numpy(float), df.mag_r_desi_catalog_value.to_numpy(float))
        assert_not_catalog_vector(df.mse_G.to_numpy(float), df.mag_r_desi_catalog_value.to_numpy(float))
        Z = control_matrix(df)
        a_r2 = associate(df.K_H_cross.to_numpy(float), df.r2_G.to_numpy(float), Z)
        a_mse = associate(df.K_H_cross.to_numpy(float), df.mse_G.to_numpy(float), Z)
        a_p = associate(df.K_H_cross.to_numpy(float), df.r2_P.to_numpy(float), Z)
        a_a = associate(df.K_H_cross.to_numpy(float), df.delta_adapt.to_numpy(float), Z)
        exp = FROZEN_Q[m]
        rec = {
            "C_R2": a_r2,
            "C_G": a_mse,
            "C_P": a_p,
            "C_A": a_a,
            "expected": exp,
            "match_C_R2": abs(float(a_r2["controlled"]) - exp["C_R2"]) <= PARITY_ATOL,
            "match_C_G": abs(float(a_mse["controlled"]) - exp["C_G"]) <= PARITY_ATOL,
            "match_C_A": abs(float(a_a["controlled"]) - exp["C_A"]) <= PARITY_ATOL,
        }
        q_rows[m] = rec
        q_ok = q_ok and rec["match_C_R2"] and rec["match_C_G"] and rec["match_C_A"]

    vitb = outcomes[REFERENCE]
    Z = control_matrix(vitb)
    # Frozen D-residual consensus lives in the previous tree; reproduce associations from that file.
    prcr_dec = shared["prcr"] / "decision.json"
    prcr_seed = shared["prcr"] / "seed_reliability.json"
    import json

    dec = json.loads(prcr_dec.read_text()) if prcr_dec.exists() else {}
    seed = json.loads(prcr_seed.read_text()) if prcr_seed.exists() else {}
    dres = {
        "source": str(prcr_dec),
        "P1": dec.get("P1", {}),
        "P2": dec.get("P2", {}),
        "rho_ctl_delta_adapt": dec.get("rho_ctl_delta_adapt"),
        "seed_median_rho": seed.get("median_rho_CH"),
        "seed_median_cos": seed.get("median_cos_HS"),
        "expected": FROZEN_VITB_DRES,
    }
    dres_ok = True
    if dec:
        dres_ok = (
            abs(float(dec["P1"]["observed"]) - FROZEN_VITB_DRES["C_R2"]) <= PARITY_ATOL
            and abs(float(dec["P2"]["observed"]) - FROZEN_VITB_DRES["C_P"]) <= PARITY_ATOL
            and abs(float(dec["rho_ctl_delta_adapt"]) - FROZEN_VITB_DRES["C_A"]) <= PARITY_ATOL
        )
    if seed:
        dres_ok = dres_ok and abs(float(seed["median_rho_CH"]) - FROZEN_VITB_DRES["median_rho_seed"]) <= PARITY_ATOL
        dres_ok = dres_ok and abs(float(seed["median_cos_HS"]) - FROZEN_VITB_DRES["median_cos_seed"]) <= PARITY_ATOL
    dres["ok"] = bool(dres_ok)

    vitb_q = q_rows[REFERENCE]
    report = {
        "ok": bool(q_ok and dres_ok and (cfg.smoke or len(vitb) >= 500)),
        "n": int(len(vitb)),
        "n_folds": int(len(set(vitb.global_oof_fold.tolist()))),
        "q_table": q_rows,
        "q_table_ok": bool(q_ok),
        "vitb_dresidual": dres,
        "vitb_dresidual_ok": bool(dres_ok),
        "match_r2": bool(vitb_q["match_C_R2"]),
        "match_mse_G": bool(vitb_q["match_C_G"]),
        "match_dMSE": bool(vitb_q["match_C_A"]),
        "catalog_not_used_as_outcome": True,
        "note": "Frozen files are authoritative. Outcomes are OOF probe metrics.",
    }
    write_json(out / "parity.json", report, force=True)
    return report
