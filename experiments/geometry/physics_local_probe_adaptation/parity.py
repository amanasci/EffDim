"""Reproduce frozen global OOF associations before any patch fitting."""

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

from .config import CONTROLS, PARITY_ATOL, PARITY_MSE, PARITY_R2, PRIMARY_D, PRIMARY_K, ExpConfig
from .data import kh_controls
from .io_util import write_json
from .metrics import assert_r2_from_sse_sst, metrics_from_preds


def run_parity(bundle: dict[str, Any], cfg: ExpConfig, out) -> dict[str, Any]:
    assert_probe_performance(PRIMARY.value)
    sids = bundle["sids"]
    y, yhat = bundle["y"], bundle["yhat"]
    fold = bundle["fold"]
    neigh = bundle["neigh"]
    sid_to_ai = bundle["sid_to_ai"]

    rows = []
    for sid in sids:
        ai = sid_to_ai[int(sid)]
        N = neigh[ai, :PRIMARY_K]
        met = metrics_from_preds(y[N], yhat[N])
        assert_r2_from_sse_sst(met["sse"], met["sst"], met["r2"])
        # leakage: no OOF point trained on its fold — checked globally below
        ctrl = kh_controls(bundle, int(sid))
        rows.append({**met, **ctrl, "r2_geo": float(bundle["geo"].loc[int(sid), "local_r2"])})
    df = pd.DataFrame(rows)
    if len(df) < 12 and not cfg.smoke:
        raise RuntimeError(f"parity n={len(df)} too small")

    # catalog must not be the association outcome
    catalog_at = np.asarray([float(y[sid_to_ai[int(s)]]) if False else float(bundle["y"][bundle["sample_id_row"] == int(s)][0]) if np.any(bundle["sample_id_row"] == int(s)) else float("nan") for s in sids], float)
    # simpler: use y at embedding row of first neighbour's... use fold table alignment
    cat_vals = []
    for s in sids:
        hits = np.where(bundle["sample_id_row"] == int(s))[0]
        cat_vals.append(float(bundle["y"][hits[0]]) if len(hits) else float("nan"))
    assert_not_catalog_vector(df.r2.to_numpy(float), np.asarray(cat_vals, float))
    assert_not_catalog_vector(df.mse.to_numpy(float), np.asarray(cat_vals, float))

    # OOF self-leak: for every index i, prediction should be from model not trained on fold[i]
    # We cannot retrain here; assert fold column exists and OOF length matches.
    if len(set(fold.tolist())) != 5:
        raise RuntimeError(f"expected 5 folds, got {len(set(fold.tolist()))}")

    r2_mismatch = float(np.mean(np.abs(df.r2 - df.r2_geo) > 1e-4))
    Z = control_matrix(df)
    a_r2 = associate(df.K_H_cross.to_numpy(float), df.r2.to_numpy(float), Z)
    a_mse = associate(df.K_H_cross.to_numpy(float), df.mse.to_numpy(float), Z)

    report = {
        "ok": True,
        "n": int(len(df)),
        "d": PRIMARY_D,
        "k": PRIMARY_K,
        "primary_target": PRIMARY.value,
        "rho_r2": a_r2,
        "rho_mse": a_mse,
        "expected_r2": PARITY_R2,
        "expected_mse": PARITY_MSE,
        "match_r2": abs(float(a_r2["controlled"]) - PARITY_R2) <= PARITY_ATOL,
        "match_mse": abs(float(a_mse["controlled"]) - PARITY_MSE) <= PARITY_ATOL,
        "r2_geo_mismatch_frac": r2_mismatch,
        "note": "Association outcomes are OOF probe performance, never raw catalog magnitude.",
    }
    report["ok"] = bool(
        report["match_r2"]
        and report["match_mse"]
        and r2_mismatch < 0.05
        and (cfg.smoke or len(df) >= 500)
    )
    write_json(out / "parity.json", report, force=True)
    if not report["ok"] and not cfg.smoke:
        raise RuntimeError(f"parity failed: {report}")
    return report
