"""Resumable analysis stages for order-stratified geometry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    cross_metric_pair,
    metric_scalars,
)
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame
from geometry.physics_activation_atlas.quadratic import quadratic_features
from geometry.physics_activation_atlas.sphere_normal_quadratic import NestedChart, chart_errors
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices
from geometry.physics_activation_atlas.tangent_reliability import principal_angles
from geometry.physics_stable_tangent_dimension.dimension import paired_bootstrap_ci
from geometry.physics_stable_tangent_dimension.nested_pca import (
    nested_uncentred_svd,
    radial_stratified_halves,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import angular_radii, rms_tangent_radius

from .algebra import (
    EPS,
    ambient_mse,
    cross_frobenius,
    fit_quadratic_map,
    intersection_rank,
    mix_shares,
    mixed_scale_nnls,
    odd_even_displacements,
    pair_antipodes,
    pca_subspace,
    per_col_r2,
    predict_quadratic_map,
    projector_overlap,
    r2_score,
    refine_chart_coords,
    svd_quadratic_image,
    truncate_bs_left,
    whiten_tangent,
)
from .pipeline import (
    FREEZE_HASH_EXPECTED,
    PARITY_D12_RHO,
    PARITY_D16_RHO,
    PARITY_TOL,
    OrderStratConfig,
    _b_path,
    _budget_ok,
    _done,
    _j_path,
    displacements,
    ensure_neigh,
    load_ctx,
    stage_prepare,
)
from .rank import DEFAULT_Q_THRESHOLDS, classify_hypothesis, select_q2
from .synthetics import SYNTH_KINDS, closest_synthetic, make_order_synthetic, split_seeds

STAGES = [
    "prepare",
    "parity",
    "carrier",
    "quadratic_rank",
    "tail_overlap",
    "conditional_tail",
    "mixed_scaling",
    "odd_even",
    "model_comparison",
    "normal_complement",
    "synthetic_calibration",
    "synthetic_evaluation",
    "associations",
    "replication",
    "analyze",
    "report",
]


def stage_parity(root: Path, cfg: OrderStratConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "parity.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    result: dict[str, Any] = {"ok": True, "corrections": []}
    geo = ctx["geo"][ctx["geo"].scale_k == cfg.primary_k]
    try:
        cov_df = pd.read_parquet(ctx["cov"] / "model_reliability_anchor_mean.parquet")
        c16 = cov_df[(cov_df.model == cfg.model) & (cov_df.k == cfg.primary_k) & (cov_df.d == 16)]
        m16 = geo.merge(c16, on="sample_id", how="inner")
        rho16, _ = spearmanr(m16.K_H_cross, m16.local_r2)
        edm_df = pd.read_parquet(ctx["edm"] / "crossfit_curvature_metrics.parquet")
        e12 = edm_df[
            (edm_df.model == cfg.model)
            & (edm_df.k == cfg.primary_k)
            & (edm_df.role == "d_star")
            & (edm_df.d == cfg.d_core)
        ]
        m12 = geo.merge(e12, on="sample_id", how="inner")
        rho12, _ = spearmanr(m12.K_H_cross, m12.local_r2)
        result["d16"] = {
            "rho_KH_cross": float(rho16),
            "expected": PARITY_D16_RHO,
            "ok": abs(float(rho16) - PARITY_D16_RHO) <= PARITY_TOL,
            "n": int(len(m16)),
        }
        result["d12"] = {
            "rho_KH_cross": float(rho12),
            "expected": PARITY_D12_RHO,
            "ok": abs(float(rho12) - PARITY_D12_RHO) <= PARITY_TOL,
            "n": int(len(m12)),
        }
        result["ok"] = bool(result["d16"]["ok"] and result["d12"]["ok"])
    except Exception as e:  # noqa: BLE001
        result["ok"] = False
        result["artifact_error"] = str(e)
    freeze_hash = ctx["freeze"].get("dimension_config_hash")
    result["freeze_hash"] = freeze_hash
    result["freeze_hash_ok"] = freeze_hash == FREEZE_HASH_EXPECTED
    result["ok"] = bool(result.get("ok", True) and result["freeze_hash_ok"])
    std_sum = ctx["std"] / "tangent_dimension_summary.csv"
    if std_sum.exists():
        sdf = pd.read_csv(std_sum)
        ref = sdf[sdf.k == cfg.primary_k]
        result["stable_core"] = {
            "median_dT": float(ref.iloc[0].median_dT) if len(ref) else float("nan"),
            "expected": 12,
            "ok": bool(len(ref) and abs(float(ref.iloc[0].median_dT) - 12) < 0.5),
        }
        result["ok"] = bool(result["ok"] and result["stable_core"]["ok"])
    else:
        result["stable_core"] = {"ok": False, "reason": "missing_stable_tangent_summary"}
        result["corrections"].append("stable_tangent_summary_missing")
    X, device = ctx["X"], ctx["device"]
    rows = []
    for sid in ctx["use_sids"][: cfg.n_parity_anchors]:
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, cfg.primary_k)
        Xloc = X[N].astype(np.float64)
        x_anchor = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        _, Jold, _, _ = nested_pca_frame(Xloc, max(cfg.R, cfg.d_ref), device)
        Z = displacements(x_anchor, Xloc, cfg.coord)
        Jlog, _ = nested_uncentred_svd(Z, max(cfg.R, cfg.d_ref), device=device)
        jp = ctx["ndc"] / "cache" / f"J_{int(sid)}_k{cfg.primary_k}.npz"
        cache_cos12 = float("nan")
        if jp.exists() and Jold.shape[1] >= 12:
            zc = np.load(jp)
            if zc["J"].shape[1] >= 12:
                cache_cos12 = float(np.mean(np.cos(principal_angles(Jold[:, :12], zc["J"][:, :12]))))
        rec: dict[str, Any] = {
            "sample_id": int(sid),
            "nested_cache_cos12": cache_cos12,
            "log_vs_mean_cos12": float(np.mean(np.cos(principal_angles(Jlog[:, :12], Jold[:, :12]))))
            if min(Jlog.shape[1], Jold.shape[1]) >= 12
            else float("nan"),
        }
        th = angular_radii(x_anchor, Xloc)
        A, B = radial_stratified_halves(th, cfg.seed + ai)
        if min(len(A), len(B)) >= cfg.d_core + 8 and Jlog.shape[1] >= cfg.d_core:
            fA, vA = _half_fit_indices(A, cfg.seed + 17 * ai)
            chA, _, infoA = fit_quad(Xloc, x_anchor, Jlog[:, : cfg.d_core], fA, vA, B, ridges=RIDGES, device=device)
            if chA is not None:
                sc = metric_scalars(chA.BS_flat, cfg.d_core)
                rec.update({f"parity_{kk}": vv for kk, vv in sc.items()})
                rec["parity_dS"] = infoA.get("dS", float("nan"))
        rows.append(rec)
    dfp = pd.DataFrame(rows)
    result["nested_pca"] = {
        "n": int(len(dfp)),
        "median_cache_cos12": float(dfp.nested_cache_cos12.median()) if len(dfp) else float("nan"),
        "median_log_vs_mean_cos12": float(dfp.log_vs_mean_cos12.median()) if len(dfp) else float("nan"),
        "note": "log-map uncentred vs mean-centred nested frames is a documented coordinate correction",
    }
    if "parity_K_H" in dfp.columns:
        result["curvature_parity"] = {
            "median_K_H": float(dfp.parity_K_H.median()),
            "median_K_aniso": float(dfp.parity_K_aniso.median()) if "parity_K_aniso" in dfp.columns else float("nan"),
            "median_K_dir": float(dfp.parity_K_dir.median()) if "parity_K_dir" in dfp.columns else float("nan"),
            "median_dS": float(dfp.parity_dS.median()) if "parity_dS" in dfp.columns else float("nan"),
        }
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[osg] parity ok={result['ok']}", flush=True)
    return result


from .stages_models import run  # noqa: E402

