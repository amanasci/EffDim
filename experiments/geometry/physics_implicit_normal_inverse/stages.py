"""Resumable stages for the implicit normal-space inverse."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from geometry.physics_activation_atlas.effdim_curvature_metrics import metric_scalars
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES as CURV_RIDGES, fit_quad
from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices
from geometry.physics_activation_atlas.tangent_reliability import principal_angles
from geometry.physics_stable_tangent_dimension.nested_pca import nested_uncentred_svd, radial_stratified_halves
from geometry.physics_stable_tangent_dimension.sphere_coords import (
    angular_radii,
    parallel_transport_basis_yx,
    rms_tangent_radius,
    sphere_log_map,
)

from .algebra import (
    EPS,
    projector_overlap,
    qr_orthonormal,
    sampson_batch,
    unpack_h,
    weighted_phi,
    intersection_rank,
)
from .classify import DEFAULT_THRESHOLDS, primary_label
from .pipeline import (
    FREEZE_HASH_EXPECTED,
    PARITY_D12_RHO,
    PARITY_D16_RHO,
    PARITY_TOL,
    ImplicitNormalConfig,
    _budget_ok,
    _done,
    _j_ours,
    cache_path,
    carrier_coords,
    classify_fit,
    dimension_from_labels,
    ensure_neigh,
    fit_constraints,
    implicit_q2_from_pack,
    load_or_compute_J,
    platonic_root,
    scaling_for_directions,
    stage_prepare,
)
from .synthetics import SYNTH_KINDS, make_implicit_synthetic, split_seeds

STAGES = [
    "prepare",
    "parity",
    "carrier",
    "linear_constraints",
    "quadratic_constraints",
    "geometric_refine",
    "constraint_scaling",
    "normal_classification",
    "dimension_bounds",
    "implicit_curvature",
    "tail_analysis",
    "gauss_validation",
    "synthetic_calibration",
    "synthetic_evaluation",
    "associations",
    "analyze",
    "report",
]


def _local_pack(ctx: dict, cfg: ImplicitNormalConfig, sid: int, k: int):
    ai = ctx["sid_to_ai"][int(sid)]
    X = ctx["X"]
    N = ensure_neigh(ctx, ai, k)
    Xloc = X[N].astype(np.float64)
    x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
    Z = sphere_log_map(x0, Xloc)
    return x0, Xloc, Z, N


def _save_fit_cache(path: Path, fit: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    np.savez_compressed(
        path,
        ok=np.array([1 if fit.get("ok") else 0]),
        lam=np.array([fit.get("lam", np.nan)]),
        df=np.array([fit.get("df", np.nan)]),
        ev_K=np.asarray(fit.get("ev_K", []), dtype=np.float64),
        ev_lin=np.asarray(fit.get("ev_lin", []), dtype=np.float64),
        UA=np.asarray(fit.get("UA", np.zeros((0, 0))), dtype=np.float64),
        UB=np.asarray(fit.get("UB", np.zeros((0, 0))), dtype=np.float64),
        Ulin=np.asarray(fit.get("Ulin", np.zeros((0, 0))), dtype=np.float64),
        h_pack=np.asarray(fit.get("h_pack", np.zeros((0, 0))), dtype=np.float64),
        null_mse=np.array([fit.get("null_mse", np.nan)]),
        tot_var=np.array([fit.get("tot_var", np.nan)]),
        dir_rows=np.array(json.dumps(fit.get("dir_rows", []), default=str)),
        q_rows=np.array(json.dumps(fit.get("q_rows", []), default=str)),
        n_tr=np.array([fit.get("n_tr", 0)]),
        n_te=np.array([fit.get("n_te", 0)]),
        R=np.array([fit.get("R", 0)]),
        outside=np.array([extra.get("outside", np.nan)]),
        theta_med=np.array([extra.get("theta_med", np.nan)]),
        rms=np.array([extra.get("rms", np.nan)]),
        ev_nested=np.asarray(extra.get("ev_nested", []), dtype=np.float64),
    )


def _load_fit_cache(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {
        "ok": bool(z["ok"][0]),
        "lam": float(z["lam"][0]),
        "df": float(z["df"][0]),
        "ev_K": np.asarray(z["ev_K"], dtype=np.float64),
        "ev_lin": np.asarray(z["ev_lin"], dtype=np.float64),
        "UA": np.asarray(z["UA"], dtype=np.float64),
        "UB": np.asarray(z["UB"], dtype=np.float64),
        "Ulin": np.asarray(z["Ulin"], dtype=np.float64),
        "h_pack": np.asarray(z["h_pack"], dtype=np.float64),
        "null_mse": float(z["null_mse"][0]),
        "tot_var": float(z["tot_var"][0]),
        "dir_rows": json.loads(z["dir_rows"].item() if getattr(z["dir_rows"], "ndim", 1) == 0 else str(z["dir_rows"])),
        "q_rows": json.loads(z["q_rows"].item() if getattr(z["q_rows"], "ndim", 1) == 0 else str(z["q_rows"])),
        "n_tr": int(z["n_tr"][0]),
        "n_te": int(z["n_te"][0]),
        "R": int(z["R"][0]),
        "outside": float(z["outside"][0]) if "outside" in z.files else float("nan"),
        "theta_med": float(z["theta_med"][0]) if "theta_med" in z.files else float("nan"),
        "rms": float(z["rms"][0]) if "rms" in z.files else float("nan"),
        "ev_nested": np.asarray(z["ev_nested"], dtype=np.float64) if "ev_nested" in z.files else np.zeros(0),
    }


def ensure_fit_cache(out: Path, ctx: dict, cfg: ImplicitNormalConfig, sid: int, k: int, R: int, *, refine_steps: int = 0):
    path = cache_path(out, cfg, sid, k, R)
    if path.exists() and not cfg.force:
        return _load_fit_cache(path)
    x0, Xloc, Z, _ = _local_pack(ctx, cfg, sid, k)
    J, ev, _meta = load_or_compute_J(out, ctx, cfg, sid, k, Z)
    if J.shape[1] < R:
        return None
    Y, outside = carrier_coords(Z, J, R)
    th = angular_radii(x0, Xloc)
    fit = fit_constraints(
        Y,
        th,
        q_max=min(cfg.q_max, R),
        seed=cfg.seed + 31 * int(sid) + k,
        n_null=cfg.n_null_draw,
        refine_steps=refine_steps,
    )
    if not fit.get("ok"):
        return None
    extra = {
        "outside": outside,
        "theta_med": float(np.median(th)),
        "rms": float(rms_tangent_radius(Z)),
        "ev_nested": ev[: min(len(ev), max(R, 8))],
    }
    _save_fit_cache(path, fit, extra)
    fit.update(extra)
    return fit


def stage_parity(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> dict:
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
        result["corrections"].append("coverage_or_effdim_merge_failed")
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
    osg_q = ctx["osg"] / "quadratic_rank_summary.csv"
    if osg_q.exists():
        qdf = pd.read_csv(osg_q)
        hit = qdf[qdf.k == cfg.primary_k] if "k" in qdf.columns else qdf
        med = float(hit.iloc[0].median_q2) if len(hit) and "median_q2" in hit.columns else float("nan")
        result["prior_q2_given_12"] = {
            "median_q2": med,
            "expected": 1.0,
            "ok": bool(np.isfinite(med) and abs(med - 1.0) <= 1.5),
            "source": str(osg_q),
        }
        if not result["prior_q2_given_12"]["ok"]:
            result["corrections"].append("prior_q2_mismatch")
    else:
        result["prior_q2_given_12"] = {"ok": False, "reason": "missing_osg_quadratic_rank_summary"}
        result["corrections"].append("osg_q2_summary_missing")
    X, device = ctx["X"], ctx["device"]
    rows = []
    for sid in ctx["use_sids"][: cfg.n_parity_anchors]:
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, cfg.primary_k)
        Xloc = X[N].astype(np.float64)
        x_anchor = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        _, Jold, _, _ = nested_pca_frame(Xloc, max(cfg.R, cfg.d_ref), device)
        Z = sphere_log_map(x_anchor, Xloc)
        Jlog, ev = nested_uncentred_svd(Z, max(cfg.R, cfg.d_ref), device=device)
        jp = ctx["osg"] / "J" / f"{cfg.model}_{int(sid)}_k{cfg.primary_k}.npz"
        cache_cos12 = float("nan")
        energy20 = float("nan")
        if jp.exists():
            zc = np.load(jp)
            Jsrc = zc["J"]
            if Jsrc.shape[1] >= 12:
                cache_cos12 = float(np.mean(np.cos(principal_angles(Jlog[:, :12], Jsrc[:, :12]))))
            if "ev" in zc.files and len(zc["ev"]) >= cfg.R:
                e = np.asarray(zc["ev"], dtype=np.float64)
                energy20 = float(np.sum(e[: cfg.R]) / max(float(np.sum(e)), EPS))
        rec: dict[str, Any] = {
            "sample_id": int(sid),
            "osg_J_cos12": cache_cos12,
            "log_vs_mean_cos12": float(np.mean(np.cos(principal_angles(Jlog[:, :12], Jold[:, :12]))))
            if min(Jlog.shape[1], Jold.shape[1]) >= 12
            else float("nan"),
            "nested_energy_frac_R20": energy20,
        }
        th = angular_radii(x_anchor, Xloc)
        A, B = radial_stratified_halves(th, cfg.seed + ai)
        if min(len(A), len(B)) >= cfg.d_core + 8 and Jlog.shape[1] >= cfg.d_ref:
            fA, vA = _half_fit_indices(A, cfg.seed + 17 * ai)
            for dname, duse in (("d12", cfg.d_core), ("d16", cfg.d_ref)):
                chA, _, infoA = fit_quad(Xloc, x_anchor, Jlog[:, :duse], fA, vA, B, ridges=CURV_RIDGES, device=device)
                if chA is not None:
                    sc = metric_scalars(chA.BS_flat, duse)
                    rec[f"parity_{dname}_K_H"] = sc.get("K_H", float("nan"))
                    rec[f"parity_{dname}_K_dir"] = sc.get("K_dir", float("nan"))
        rows.append(rec)
    dfp = pd.DataFrame(rows)
    result["nested_carrier_S20"] = {
        "n": int(len(dfp)),
        "median_osg_J_cos12": float(dfp.osg_J_cos12.median()) if len(dfp) else float("nan"),
        "median_energy_frac_R20": float(dfp.nested_energy_frac_R20.median()) if len(dfp) else float("nan"),
        "ok": bool(len(dfp) and (not np.isfinite(dfp.osg_J_cos12).any() or float(dfp.osg_J_cos12.median()) > 0.85)),
        "note": "log-map uncentred nested PCA through the anchor; origin is the chart origin",
    }
    if "parity_d12_K_H" in dfp.columns:
        result["curvature_parity"] = {
            "median_K_H_d12": float(dfp.parity_d12_K_H.median()),
            "median_K_H_d16": float(dfp.parity_d16_K_H.median()) if "parity_d16_K_H" in dfp.columns else float("nan"),
        }
    result["l2_unit_normalized"] = bool(ctx["l2"].get("unit_normalized", False))
    if not result["l2_unit_normalized"]:
        result["corrections"].append("activations_not_unit_normalized")
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[ini] parity ok={result['ok']} corrections={result['corrections']}", flush=True)
    return result

