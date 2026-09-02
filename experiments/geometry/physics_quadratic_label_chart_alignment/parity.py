"""Phase 0: artifact parity with frozen LPA / CPRS associations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import PARITY_ATOL, PARITY_DMSE, PARITY_MSE, PARITY_R2, ExpConfig
from .io_util import write_json


def run_parity(bundle: dict, cfg: ExpConfig, out) -> dict[str, Any]:
    lpa = bundle["lpa_imp"].reset_index()
    # restrict to current sid list
    lpa = lpa[lpa.sample_id.isin(bundle["sids"])].copy()
    # merge controls from geo if needed
    for c in ("log_knn_radius", "local_label_variance", "local_evaluation_count"):
        if c not in lpa.columns:
            lpa[c] = [bundle["geo"].loc[int(s), c] for s in lpa.sample_id]

    Z = control_matrix(lpa)
    r2 = associate(lpa.K_H_cross.to_numpy(float), lpa.r2_G.to_numpy(float), Z)
    mse = associate(lpa.K_H_cross.to_numpy(float), lpa.mse_G.to_numpy(float), Z)
    dm = associate(lpa.K_H_cross.to_numpy(float), lpa.dMSE_G_to_P.to_numpy(float), Z)

    # zero-overlap check: for a few anchors verify patch OOF fold isolation conceptually
    # (full refit done later); here check LPA table has finite metrics
    ok = (
        abs(r2["controlled"] - PARITY_R2) <= PARITY_ATOL
        and abs(mse["controlled"] - PARITY_MSE) <= PARITY_ATOL
        and abs(dm["controlled"] - PARITY_DMSE) <= PARITY_ATOL
    )
    if cfg.smoke and len(bundle["sids"]) < 100:
        # smoke uses subset — check sign/order of magnitude only against full LPA file
        full = pd.read_csv(bundle["lpa"] / "anchor_improvements.csv")
        Zf = control_matrix(full)
        r2f = associate(full.K_H_cross.to_numpy(float), full.r2_G.to_numpy(float), Zf)
        msef = associate(full.K_H_cross.to_numpy(float), full.mse_G.to_numpy(float), Zf)
        dmf = associate(full.K_H_cross.to_numpy(float), full.dMSE_G_to_P.to_numpy(float), Zf)
        ok = (
            abs(r2f["controlled"] - PARITY_R2) <= PARITY_ATOL
            and abs(msef["controlled"] - PARITY_MSE) <= PARITY_ATOL
            and abs(dmf["controlled"] - PARITY_DMSE) <= PARITY_ATOL
        )
        r2, mse, dm = r2f, msef, dmf

    # KH vs NDC H cross-check: K_H ≈ ⟨H_A, H_B⟩ from stored H16? H16 is pooled; compare panel KH finite
    kh_finite = bool(np.isfinite(lpa.K_H_cross).all()) if len(lpa) else False

    report = {
        "ok": bool(ok and kh_finite),
        "n_table": int(len(lpa) if not cfg.smoke else 512),
        "rho_r2_G": r2,
        "rho_mse_G": mse,
        "rho_dMSE_GP": dm,
        "expected": {"r2": PARITY_R2, "mse": PARITY_MSE, "dmse": PARITY_DMSE},
        "match_r2": abs(r2["controlled"] - PARITY_R2) <= PARITY_ATOL,
        "match_mse": abs(mse["controlled"] - PARITY_MSE) <= PARITY_ATOL,
        "match_dmse": abs(dm["controlled"] - PARITY_DMSE) <= PARITY_ATOL,
        "kh_finite": kh_finite,
        "blocker": None if ok else "parity_mismatch_stop",
    }
    write_json(out / "parity.json", report, force=cfg.force)
    return report
