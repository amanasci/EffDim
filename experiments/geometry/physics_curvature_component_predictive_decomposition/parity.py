"""Reproduce FCR, trace, and QLCA point estimates from frozen tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import (
    FCR_ATOL,
    FCR_DIR_ADAPT,
    FCR_DIR_R2,
    FCR_DIR_R2P,
    PARITY_VITB_DMSE,
    PARITY_VITB_MSE_G,
    PARITY_VITB_R2,
    QLCA_AB,
    QLCA_ATOL,
    QLCA_MED_DQ,
    QLCA_RHO_KH_DQ,
    STAB_MIN,
    ExpConfig,
)
from geometry.physics_cross_model_full_curvature_reconciliation.config import TRACE_EXPECTED as TRACE_TABLE
from .data import load_qlca_risks, merge_components
from .io_util import write_json, write_text


def _match(a: float, b: float, atol: float) -> bool:
    return bool(np.isfinite(a) and abs(float(a) - float(b)) <= atol)


def run_parity(shared: dict, cfg: ExpConfig) -> dict[str, Any]:
    sids = shared["sids"]
    models = shared["models"]
    fcr_rows = {}
    dir_r2, dir_r2p, dir_ca = [], [], []
    kh_rows = {}
    for m in models:
        df = merge_components(shared, m, sids)
        Z = control_matrix(df)
        rec = {
            "n": int(len(df)),
            "identity_max": float((df.K_dir_cross - df.K_H_cross - df.K_TF_cross).abs().max()),
            "rho_Kdir_R2G": associate(df.K_dir_cross.to_numpy(float), df.r2_G.to_numpy(float), Z)["controlled"],
            "rho_Kdir_R2P": associate(df.K_dir_cross.to_numpy(float), df.r2_P.to_numpy(float), Z)["controlled"],
            "rho_Kdir_Dadapt": associate(df.K_dir_cross.to_numpy(float), df.delta_adapt.to_numpy(float), Z)["controlled"],
            "rho_KH_R2G": associate(df.K_H_cross.to_numpy(float), df.r2_G.to_numpy(float), Z)["controlled"],
            "rho_KH_MSEG": associate(df.K_H_cross.to_numpy(float), df.mse_G.to_numpy(float), Z)["controlled"],
            "rho_KH_Dadapt": associate(df.K_H_cross.to_numpy(float), df.delta_adapt.to_numpy(float), Z)["controlled"],
        }
        fcr_rows[m] = rec
        dir_r2.append(rec["rho_Kdir_R2G"])
        dir_r2p.append(rec["rho_Kdir_R2P"])
        dir_ca.append(rec["rho_Kdir_Dadapt"])
        exp = TRACE_TABLE[m]
        kh_rows[m] = {
            "C_R2": rec["rho_KH_R2G"],
            "C_G": rec["rho_KH_MSEG"],
            "C_A": rec["rho_KH_Dadapt"],
            "match_R2": _match(rec["rho_KH_R2G"], exp["C_R2"], 1e-6),
            "match_CG": _match(rec["rho_KH_MSEG"], exp["C_G"], 1e-6),
            "match_CA": _match(rec["rho_KH_Dadapt"], exp["C_A"], 1e-6),
        }
    agg = {
        "rho_Kdir_R2G": float(np.mean(dir_r2)),
        "rho_Kdir_R2P": float(np.mean(dir_r2p)),
        "rho_Kdir_Dadapt": float(np.mean(dir_ca)),
    }
    vitb = fcr_rows.get("vit_base", {})
    qlca = load_qlca_risks(shared, sids)
    Zq = control_matrix(qlca)
    rho_dq = associate(qlca.K_H_cross.to_numpy(float), qlca.delta_Q.to_numpy(float), Zq)["controlled"]
    qlca_rec = {
        "n": int(len(qlca)),
        "median_delta_Q": float(qlca.delta_Q.median()),
        "rho_KH_delta_Q": float(rho_dq),
        "A_B_median": float(qlca.A_B.median()),
        "frac_stable": float((qlca.gamma_fold_cosine >= STAB_MIN).mean()),
        "all_stable": bool((qlca.gamma_fold_cosine >= STAB_MIN).all()),
        "match_dq": _match(float(qlca.delta_Q.median()), QLCA_MED_DQ, QLCA_ATOL),
        "match_rho": _match(float(rho_dq), QLCA_RHO_KH_DQ, QLCA_ATOL),
        "match_AB": _match(float(qlca.A_B.median()), QLCA_AB, QLCA_ATOL),
    }
    ok_fcr = (
        _match(agg["rho_Kdir_R2G"], FCR_DIR_R2, FCR_ATOL)
        and _match(agg["rho_Kdir_R2P"], FCR_DIR_R2P, FCR_ATOL)
        and _match(agg["rho_Kdir_Dadapt"], FCR_DIR_ADAPT, FCR_ATOL)
        and all(r["identity_max"] < 1e-10 for r in fcr_rows.values())
    )
    ok_kh = (
        _match(vitb.get("rho_KH_R2G", 0), PARITY_VITB_R2, FCR_ATOL)
        and _match(vitb.get("rho_KH_MSEG", 0), PARITY_VITB_MSE_G, FCR_ATOL)
        and _match(vitb.get("rho_KH_Dadapt", 0), PARITY_VITB_DMSE, FCR_ATOL)
        and all(v["match_R2"] and v["match_CG"] and v["match_CA"] for v in kh_rows.values())
    )
    ok_qlca = qlca_rec["match_dq"] and qlca_rec["match_rho"] and qlca_rec["match_AB"] and qlca_rec["all_stable"]
    ok = bool((ok_fcr and ok_kh and ok_qlca) or cfg.smoke)
    report = {
        "ok": ok,
        "fcr": {"ok": ok_fcr or cfg.smoke, "aggregate": agg, "per_model": fcr_rows},
        "trace": {"ok": ok_kh or cfg.smoke, "per_model": kh_rows, "vitb": vitb},
        "qlca": {"ok": ok_qlca or cfg.smoke, **qlca_rec},
        "smoke": cfg.smoke,
    }
    return report
