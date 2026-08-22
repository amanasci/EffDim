"""Parity, synthetic calibration, and per-anchor fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from geometry.physics_activation_atlas.effdim_curvature_metrics import metric_scalars
from geometry.physics_activation_atlas.paths import resolve_path
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES as CURV_RIDGES, fit_quad
from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices
from geometry.physics_activation_atlas.tangent_reliability import principal_angles
from geometry.physics_order_stratified_geometry.algebra import (
    fit_quadratic_map,
    predict_quadratic_map,
    r2_score,
    whiten_tangent,
)
from geometry.physics_stable_tangent_dimension.nested_pca import nested_uncentred_svd, radial_stratified_halves
from geometry.physics_stable_tangent_dimension.sphere_coords import angular_radii, sphere_log_map

from .classify import DEFAULT_THRESHOLDS, plateau_from_curve
from .pipeline import (
    FREEZE_HASH_EXPECTED,
    PARITY_D12_RHO,
    PARITY_D16_RHO,
    PARITY_E4_R2,
    PARITY_E4_TOL,
    PARITY_TOL,
    PRESERVED,
    QuadPredConfig,
    _budget_ok,
    _done,
    _file_sha,
    ensure_neigh,
    fit_neighbourhood,
    load_frozen_J,
    local_pack,
    write_df,
)
from .synthetics import SYNTH_KINDS, make_predictive_synthetic, split_seeds

STAGES = [
    "prepare",
    "parity",
    "synthetic_calibration",
    "fit_primary",
    "scale_sensitivity",
    "aggregate",
    "tail_adequacy",
    "synthetic_evaluation",
    "analyze",
    "report",
]


def stage_parity(root: Path, cfg: QuadPredConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "parity.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    result: dict[str, Any] = {"ok": True, "corrections": []}
    freeze_hash = ctx["freeze"].get("dimension_config_hash")
    result["freeze_hash"] = freeze_hash
    result["freeze_hash_ok"] = freeze_hash == FREEZE_HASH_EXPECTED
    result["n_anchors"] = len(ctx["use_sids"])
    result["n_anchors_ok"] = len(ctx["use_sids"]) >= (8 if cfg.smoke else 500)
    result["l2_unit_normalized"] = bool(ctx["l2"].get("unit_normalized", False))
    result["coord"] = "sphere_log"
    hard = bool(result["freeze_hash_ok"] and result["n_anchors_ok"] and result["l2_unit_normalized"])
    result["preserved_complete_hashes"] = {}
    markers = ("COMPLETE.json", "decision_labels.json", "freeze_manifest.json", "REPORT.md", "summary.json")
    for rel in PRESERVED:
        base = resolve_path(root, rel)
        rec = {}
        for name in markers:
            cpath = base / name
            rec[name] = {"sha": _file_sha(cpath), "mtime": cpath.stat().st_mtime, "size": cpath.stat().st_size} if cpath.exists() else None
        result["preserved_complete_hashes"][rel] = rec
    geo = ctx["geo"][ctx["geo"].scale_k == cfg.primary_k]
    try:
        cov_df = pd.read_parquet(ctx["cov"] / "model_reliability_anchor_mean.parquet")
        c16 = cov_df[(cov_df.model == cfg.model) & (cov_df.k == cfg.primary_k) & (cov_df.d == 16)]
        m16 = geo.merge(c16, on="sample_id", how="inner")
        rho16, _ = spearmanr(m16.K_H_cross, m16.local_r2)
        edm_df = pd.read_parquet(ctx["edm"] / "crossfit_curvature_metrics.parquet")
        e12 = edm_df[
            (edm_df.model == cfg.model) & (edm_df.k == cfg.primary_k) & (edm_df.role == "d_star") & (edm_df.d == cfg.d_core)
        ]
        m12 = geo.merge(e12, on="sample_id", how="inner")
        rho12, _ = spearmanr(m12.K_H_cross, m12.local_r2)
        result["d16"] = {"rho_KH_cross": float(rho16), "expected": PARITY_D16_RHO, "ok": abs(float(rho16) - PARITY_D16_RHO) <= PARITY_TOL, "n": int(len(m16))}
        result["d12"] = {"rho_KH_cross": float(rho12), "expected": PARITY_D12_RHO, "ok": abs(float(rho12) - PARITY_D12_RHO) <= PARITY_TOL, "n": int(len(m12))}
    except Exception as e:  # noqa: BLE001
        result["corrections"].append(f"kh_magr_merge:{e}")
        result["d16"] = {"ok": False}
        result["d12"] = {"ok": False}
    std_sum = ctx["std"] / "tangent_dimension_summary.csv"
    if std_sum.exists():
        sdf = pd.read_csv(std_sum)
        ref = sdf[sdf.k == cfg.primary_k]
        result["stable_core"] = {"median_dT": float(ref.iloc[0].median_dT) if len(ref) else float("nan"), "expected": 12, "ok": bool(len(ref) and abs(float(ref.iloc[0].median_dT) - 12) < 0.5)}
        hard = bool(hard and result["stable_core"]["ok"])
    else:
        result["stable_core"] = {"ok": False}
        hard = False
        result["corrections"].append("stable_tangent_summary_missing")
    osg_q = ctx["osg"] / "quadratic_rank_summary.csv"
    if osg_q.exists():
        qdf = pd.read_csv(osg_q)
        hit = qdf[qdf.k == cfg.primary_k] if "k" in qdf.columns else qdf
        med = float(hit.iloc[0].median_q2) if len(hit) and "median_q2" in hit.columns else float("nan")
        result["prior_q2_given_12"] = {"median_q2": med, "expected": 1.0, "ok": bool(np.isfinite(med) and abs(med - 1) <= 1.5)}
    else:
        result["prior_q2_given_12"] = {"ok": False}
        result["corrections"].append("osg_q2_missing")
    osg_e4 = ctx["osg"] / "conditional_tail_prediction.parquet"
    if osg_e4.exists():
        edf = pd.read_parquet(osg_e4)
        r2e = float(edf.r2_E4.median()) if "r2_E4" in edf.columns else float("nan")
        result["prior_E4_quadratic_R2"] = {
            "median_r2_E4": r2e,
            "expected": PARITY_E4_R2,
            "explained_fraction": r2e,
            "unexplained_fraction": float(1.0 - r2e) if np.isfinite(r2e) else float("nan"),
            "ok": bool(np.isfinite(r2e) and abs(r2e - PARITY_E4_R2) <= PARITY_E4_TOL),
            "note": "This is ~15% explained and ~85% unexplained. The earlier experiment did not explain 85%.",
        }
    else:
        result["prior_E4_quadratic_R2"] = {"ok": False}
        result["corrections"].append("osg_E4_missing")
    rows = []
    X, device = ctx["X"], ctx["device"]
    for sid in ctx["use_sids"][: cfg.n_parity_anchors]:
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, cfg.primary_k)
        Xloc = X[N].astype(np.float64)
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        Z = sphere_log_map(x0, Xloc)
        Jlog, _ = nested_uncentred_svd(Z, max(cfg.R, cfg.d_ref), device=device)
        Jfr = load_frozen_J(ctx, cfg, sid, cfg.primary_k)
        rec: dict[str, Any] = {"sample_id": int(sid)}
        if Jfr is not None and min(Jfr.shape[1], Jlog.shape[1]) >= 12:
            rec["frozen_cos12"] = float(np.mean(np.cos(principal_angles(Jlog[:, :12], Jfr[:, :12]))))
        _, Jold, _, _ = nested_pca_frame(Xloc, max(cfg.R, cfg.d_ref), device)
        if min(Jlog.shape[1], Jold.shape[1]) >= 12:
            rec["log_vs_mean_cos12"] = float(np.mean(np.cos(principal_angles(Jlog[:, :12], Jold[:, :12]))))
        th = angular_radii(x0, Xloc)
        A, B = radial_stratified_halves(th, cfg.seed + ai)
        if min(len(A), len(B)) >= cfg.d_core + 8 and Jlog.shape[1] >= cfg.d_ref:
            fA, vA = _half_fit_indices(A, cfg.seed + 17 * ai)
            chA, _, _info = fit_quad(Xloc, x0, Jlog[:, : cfg.d_core], fA, vA, B, ridges=CURV_RIDGES, device=device)
            if chA is not None:
                rec["parity_K_H"] = metric_scalars(chA.BS_flat, cfg.d_core).get("K_H", float("nan"))
            Jc, E4 = Jlog[:, : cfg.d_core], Jlog[:, cfg.d_core : cfg.d_ref]
            Utr, scw = whiten_tangent(Z[A] @ Jc)
            Ute, _ = whiten_tangent(Z[B] @ Jc, scw)
            Yhat = predict_quadratic_map(Ute, fit_quadratic_map(Utr, Z[A] @ E4, 1e-2))
            rec["parity_r2_E4"] = r2_score(Z[B] @ E4, Yhat)
        rows.append(rec)
    dfp = pd.DataFrame(rows)
    result["nested_carrier_S20"] = {
        "n": int(len(dfp)),
        "median_frozen_cos12": float(dfp.frozen_cos12.median()) if "frozen_cos12" in dfp.columns else float("nan"),
        "ok": True,
    }
    if "parity_r2_E4" in dfp.columns:
        result["reproduced_E4_R2"] = {"median": float(dfp.parity_r2_E4.median()), "note": "earlier estimator: whitened u, unweighted phi, ridge=1e-2"}
    result["decomposition"] = {"T12": "ranks 1-12", "E4": "ranks 13-16", "U4": "ranks 17-20", "U8": "E4 + U4"}
    result["ok"] = bool(hard)
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[qpd] parity ok={result['ok']} corrections={result['corrections']}", flush=True)
    if not result["ok"]:
        raise RuntimeError("parity failed; see parity.json")
    return result


def _curve_from_rows(rows: list[dict[str, Any]], ds: list[int], thr: dict) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if not len(df):
        return {"dQ": float("nan"), "dL": float("nan"), "r2_q": np.array([])}
    g = df.groupby("d").mean(numeric_only=True)
    nmse_q = np.array([float(g.loc[d].quad_close_nmse) if d in g.index else np.nan for d in ds])
    nmse_l = np.array([float(g.loc[d].lin_nmse) if d in g.index else np.nan for d in ds])
    df_frac = np.array([float(g.loc[d].df_frac) if d in g.index and "df_frac" in g.columns else np.nan for d in ds])
    pq = plateau_from_curve(np.array(ds), nmse_q, df_frac, thr)
    pl = plateau_from_curve(np.array(ds), nmse_l, None, thr)
    return {"dQ": pq["d_plat"], "dL": pl["d_plat"], "nmse_q": nmse_q, "nmse_l": nmse_l, "r2_q": 1.0 - nmse_q}


def _eval_synth(pack: dict, cfg: QuadPredConfig, thr: dict) -> dict[str, Any]:
    rows = fit_neighbourhood(
        pack["Z"], pack["radii"], np.zeros(pack["Z"].shape[1]),
        ds=cfg.ds(), thr=thr, seed=cfg.seed + 3, frozen_J=None,
        d_core=min(12, pack["Z"].shape[1]), d_ref=min(16, pack["Z"].shape[1]),
        R=min(cfg.R, pack["Z"].shape[1]), n_inner_cp=min(cfg.n_inner_cp, 48), device=None,
    )
    curve = _curve_from_rows(rows, cfg.ds(), thr)
    return {
        "kind": pack["kind"], "true_d": pack["true_d"], "dQ": curve["dQ"], "dL": curve["dL"],
        "r2_best": float(np.nanmax(curve["r2_q"])) if np.size(curve.get("r2_q", [])) else float("nan"),
        "ok": bool(len(rows)),
    }


def stage_synthetic_calibration(root: Path, cfg: QuadPredConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "synthetic_calibration.csv"
    tpath = out / "thresholds.json"
    if _done(path, cfg.force) and _done(tpath, cfg.force):
        return json.loads(tpath.read_text())
    seeds = split_seeds(cfg.n_synth_cal, cfg.n_synth_eval)
    variants = [
        dict(DEFAULT_THRESHOLDS),
        {**DEFAULT_THRESHOLDS, "delta_practical": 0.002, "plateau_rel_tol": 0.01},
        {**DEFAULT_THRESHOLDS, "delta_practical": 0.008, "plateau_rel_tol": 0.04},
    ]
    n, D = (200, 20) if cfg.smoke else (480, 32)
    best_thr, best_score, best_rows = dict(DEFAULT_THRESHOLDS), float("-inf"), []
    for vi, thr in enumerate(variants):
        rows = []
        for kind in SYNTH_KINDS:
            for seed in seeds["calibration_seeds"]:
                pack = make_predictive_synthetic(kind, n=n, D=D, seed=seed)
                est = _eval_synth(pack, cfg, thr)
                est.update({"variant": vi, "seed": seed, "split": "calibration"})
                rows.append(est)
        df = pd.DataFrame(rows)
        err = np.abs(df.dQ - df.true_d)
        mask = df.kind.str.contains("flat_|curved_d")
        mae = float(np.nanmean(err[mask])) if mask.any() else 9.0
        unique = int(df.dQ.nunique(dropna=True))
        iso = df[df.kind == "isotropic"]
        iso_pen = 3.0 if len(iso) and abs(float(iso.dQ.median()) - 12) < 1.5 else 0.0
        only12 = 4.0 if unique <= 1 and abs(float(df.dQ.median()) - 12) < 1 else 0.0
        score = -mae + 0.3 * unique - iso_pen - only12
        if score > best_score:
            best_score, best_thr, best_rows = score, dict(thr), rows
    write_df(path, pd.DataFrame(best_rows), force=cfg.force)
    best_thr["calibration_score"] = best_score
    tpath.write_text(json.dumps(best_thr, indent=2))
    print(f"[qpd] synthetic_calibration score={best_score:.3f}", flush=True)
    return best_thr


def _fit_anchor(out: Path, ctx: dict, cfg: QuadPredConfig, sid: int, k: int, thr: dict) -> pd.DataFrame:
    cache = out / "cache" / f"{cfg.model}_{int(sid)}_k{int(k)}.parquet"
    if cache.exists() and not cfg.force:
        return pd.read_parquet(cache)
    x0, Xloc, Z, th, _ = local_pack(ctx, cfg, sid, k)
    Jfr = load_frozen_J(ctx, cfg, sid, k)
    if Jfr is None:
        Jfr, _ = nested_uncentred_svd(Z, max(cfg.R, cfg.d_max), device=ctx["device"], centre=False)
        dest = out / "J" / f"{cfg.model}_{int(sid)}_k{int(k)}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, J=Jfr)
    rows = fit_neighbourhood(
        Z, th, x0, ds=cfg.ds(), thr=thr, seed=cfg.seed + 31 * int(sid) + int(k),
        frozen_J=Jfr, d_core=cfg.d_core, d_ref=cfg.d_ref, R=cfg.R,
        n_inner_cp=cfg.n_inner_cp, device=ctx["device"],
    )
    df = pd.DataFrame(rows)
    df["sample_id"] = int(sid)
    df["k"] = int(k)
    df["R"] = int(cfg.R)
    write_df(cache, df, force=cfg.force)
    return df


def stage_fit_primary(root: Path, cfg: QuadPredConfig, ctx: dict, t0: float, thr: dict) -> None:
    out = cfg.resolved(root)
    marker = out / "fit_primary_done.json"
    if _done(marker, cfg.force):
        return
    k = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    n_done = 0
    for sid in ctx["use_sids"]:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        _fit_anchor(out, ctx, cfg, sid, k, thr)
        n_done += 1
        if n_done % 16 == 0:
            print(f"[qpd] fit_primary {n_done}/{len(ctx['use_sids'])} k={k}", flush=True)
    marker.write_text(json.dumps({"n": n_done, "k": k}))
    print(f"[qpd] fit_primary n={n_done}", flush=True)


def stage_scale_sensitivity(root: Path, cfg: QuadPredConfig, ctx: dict, t0: float, thr: dict) -> None:
    out = cfg.resolved(root)
    marker = out / "scale_sensitivity_done.json"
    if _done(marker, cfg.force):
        return
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    extra = [k for k in ctx["ks"] if k != k_ref]
    n_done = 0
    for k in extra:
        for sid in ctx["scale_sids"]:
            if not _budget_ok(t0, cfg, reserve=True):
                break
            _fit_anchor(out, ctx, cfg, sid, k, thr)
            n_done += 1
        print(f"[qpd] scale k={k}", flush=True)
    marker.write_text(json.dumps({"n_jobs": n_done, "scale_sids": ctx["scale_sids"], "ks": extra}))
    print(f"[qpd] scale_sensitivity jobs={n_done}", flush=True)
