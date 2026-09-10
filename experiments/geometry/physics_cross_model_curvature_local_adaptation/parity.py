"""Phase 0: reproduce frozen ViT-B global/local associations exactly."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix
from geometry.physics_curvature_probe_submission_validation.schema import (
    PRIMARY,
    assert_not_catalog_vector,
    assert_probe_performance,
)

from .config import (
    CONTROLS,
    PARITY_ATOL,
    PARITY_DM_MEAN,
    PARITY_DMSE,
    PARITY_MSE_G,
    PARITY_MSE_P,
    PARITY_R2,
    SOURCE_LPA,
    ExpConfig,
)
from .data import load_lpa_vitb
from .io_util import resolve_path, write_json


def run_parity(shared: dict, cfg: ExpConfig, out) -> dict[str, Any]:
    assert_probe_performance(PRIMARY.value)
    imp, met = load_lpa_vitb()
    n = int(len(imp))
    Z = control_matrix(imp)
    a_r2 = associate(imp.K_H_cross.to_numpy(float), imp.r2_G.to_numpy(float), Z)
    a_mse_g = associate(imp.K_H_cross.to_numpy(float), imp.mse_G.to_numpy(float), Z)
    a_mse_p = associate(imp.K_H_cross.to_numpy(float), imp.mse_P.to_numpy(float), Z)
    a_dm = associate(imp.K_H_cross.to_numpy(float), imp.dMSE_G_to_P.to_numpy(float), Z)
    mean_dm = float(imp.dMSE_G_to_P.mean())
    median_dm = float(imp.dMSE_G_to_P.median())
    frac_pos = float((imp.dMSE_G_to_P > 0).mean())
    overlap0 = bool((~met.overlap_any).all()) if "overlap_any" in met.columns else False
    n_eval = int(met.n_eval.median()) if "n_eval" in met.columns else 0

    cat = []
    for s in imp.sample_id.astype(int):
        hits = np.where(shared["sample_id_row"] == int(s))[0]
        cat.append(float(shared["y"][hits[0]]) if len(hits) else float("nan"))
    assert_not_catalog_vector(imp.r2_G.to_numpy(float), np.asarray(cat, float))
    assert_not_catalog_vector(imp.mse_G.to_numpy(float), np.asarray(cat, float))
    assert_not_catalog_vector(imp.dMSE_G_to_P.to_numpy(float), np.asarray(cat, float))

    lpa = resolve_path(shared["root"], SOURCE_LPA)
    import json

    prim = json.loads((lpa / "primary_inference.json").read_text())
    report = {
        "ok": True,
        "n_association": n,
        "n_probe_eval_median": n_eval,
        "zero_overlap": overlap0,
        "rho_r2_G": a_r2,
        "rho_mse_G": a_mse_g,
        "rho_mse_P": a_mse_p,
        "rho_dMSE_GP": a_dm,
        "mean_delta_adapt": mean_dm,
        "median_delta_adapt": median_dm,
        "frac_positive_delta_adapt": frac_pos,
        "patch_worse_on_average": bool(mean_dm < 0),
        "primary_reused": {
            "controlled": float(prim["observed"]["controlled"]),
            "ci95": prim["ci95"],
            "p_mc": prim["p_mc"],
            "n_perm": prim["n_perm"],
            "n_boot": prim["n_boot"],
        },
        "expected": {
            "rho_r2": PARITY_R2,
            "rho_mse_G": PARITY_MSE_G,
            "rho_mse_P": PARITY_MSE_P,
            "rho_dMSE": PARITY_DMSE,
            "mean_dMSE": PARITY_DM_MEAN,
        },
        "match_r2": abs(float(a_r2["controlled"]) - PARITY_R2) <= PARITY_ATOL,
        "match_mse_G": abs(float(a_mse_g["controlled"]) - PARITY_MSE_G) <= PARITY_ATOL,
        "match_mse_P": abs(float(a_mse_p["controlled"]) - PARITY_MSE_P) <= PARITY_ATOL,
        "match_dMSE": abs(float(a_dm["controlled"]) - PARITY_DMSE) <= PARITY_ATOL,
        "match_mean_sign": mean_dm < 0,
        "match_primary_point": abs(float(prim["observed"]["controlled"]) - float(a_dm["controlled"])) < 1e-12,
        "oof_folds": int(len(set(shared["fold"].tolist()))),
        "note": "Outcomes are OOF probe metrics, never mag_r_desi_catalog_value. Δ_adapt=MSE_G-MSE_P.",
    }
    report["ok"] = bool(
        report["match_r2"]
        and report["match_mse_G"]
        and report["match_dMSE"]
        and report["match_mean_sign"]
        and report["zero_overlap"]
        and (cfg.smoke or n >= 500)
        and (cfg.smoke or n_eval >= 1000)
        and report["oof_folds"] == 5
    )
    write_json(out / "parity.json", report, force=True)
    return report
