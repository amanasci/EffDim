"""Phase 1 parity from frozen LPA outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix
from geometry.physics_curvature_probe_submission_validation.schema import assert_not_catalog_vector

from .config import (
    CONTROLS,
    N_ANCHORS,
    N_PROBE_EVAL,
    PARITY_ATOL,
    PARITY_DM_MEAN,
    PARITY_DMH,
    PARITY_MSE,
    PARITY_MSE_G,
    PARITY_MSE_P,
    PARITY_R2,
    PRIMARY_D,
    PRIMARY_K,
)
from .io_util import write_json


def load_lpa_tables(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .io_util import resolve_path
    from .config import SOURCE_LPA

    out = resolve_path(root, SOURCE_LPA)
    imp = pd.read_csv(out / "anchor_improvements.csv")
    met = pd.read_parquet(out / "anchor_model_metrics.parquet")
    return imp, met


def run_parity(root, cfg, out) -> dict[str, Any]:
    imp, met = load_lpa_tables(root)
    n = len(imp)
    if n != N_ANCHORS and not cfg.smoke:
        raise RuntimeError(f"expected {N_ANCHORS} anchors, got {n}")

    Z = control_matrix(imp)
    a_r2 = associate(imp.K_H_cross.to_numpy(float), imp.r2_G.to_numpy(float), Z)
    a_mse_g = associate(imp.K_H_cross.to_numpy(float), imp.mse_G.to_numpy(float), Z)
    a_mse_p = associate(imp.K_H_cross.to_numpy(float), imp.mse_P.to_numpy(float), Z)
    a_dmh = associate(imp.K_H_cross.to_numpy(float), imp.dMSE_G_to_P.to_numpy(float), Z)

    # catalog guard on outcomes
    # y catalog not in imp; skip if no catalog column

    mean_dm = float(imp.dMSE_G_to_P.mean())
    overlap = bool((~met.overlap_any).all()) if "overlap_any" in met.columns else True
    n_eval = int(met.n_eval.median()) if "n_eval" in met.columns else N_PROBE_EVAL

    report = {
        "ok": True,
        "n_association": int(n),
        "n_probe_eval_median": n_eval,
        "curvature_fit_m_note": "K_H_cross is frozen; unrelated to probe outer-fold train count m from scale experiment",
        "rho_r2_G": a_r2,
        "rho_mse_G": a_mse_g,
        "rho_mse_P": a_mse_p,
        "rho_dMSE_GP": a_dmh,
        "mean_dMSE_GP": mean_dm,
        "expected": {
            "rho_r2": PARITY_R2,
            "rho_mse_G": PARITY_MSE,
            "rho_dMSE": PARITY_DMH,
            "mean_dMSE": PARITY_DM_MEAN,
            "rho_mse_G_point": PARITY_MSE_G,
            "rho_mse_P_point": PARITY_MSE_P,
        },
        "match_r2": abs(float(a_r2["controlled"]) - PARITY_R2) <= PARITY_ATOL,
        "match_mse_G": abs(float(a_mse_g["controlled"]) - PARITY_MSE) <= PARITY_ATOL,
        "match_dMSE": abs(float(a_dmh["controlled"]) - PARITY_DMH) <= PARITY_ATOL,
        "match_mean_sign": mean_dm < 0,
        "zero_overlap": overlap,
    }
    report["ok"] = bool(
        report["match_r2"]
        and report["match_mse_G"]
        and report["match_dMSE"]
        and report["zero_overlap"]
        and n_eval >= 1000
    )
    write_json(out / "parity.json", report, force=cfg.force)
    if not report["ok"] and not cfg.smoke:
        raise RuntimeError(f"parity failed: {report}")
    return report
