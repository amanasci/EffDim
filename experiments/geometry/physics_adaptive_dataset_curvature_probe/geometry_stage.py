"""Stage 1: geometry-only dimensional ranges. Physics labels are not loaded here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import knn_torch_ip, load_model_X
from geometry.physics_activation_atlas.paths import resolve_path
from geometry.physics_quadratic_predictive_dimension.algebra import (
    closest_point_project,
    n_quad_features,
    nmse,
    phi2,
    predict_f,
    project_B_normal,
    remove_radial_cols,
    ridge_df,
    ridge_fit,
    ridge_grid_from_gram,
    scale_phi_train,
)
from geometry.physics_quadratic_predictive_dimension.classify import plateau_from_curve
from geometry.physics_stable_tangent_dimension.nested_pca import nested_uncentred_svd, radial_stratified_halves
from geometry.physics_stable_tangent_dimension.sphere_coords import angular_radii, row_l2_status, sphere_log_map

from .config import (
    DEFAULT_THRESHOLDS,
    DELTA_PRACTICAL,
    NOISE_TAIL_RANKS_PAST_PLATEAU,
    SOURCE_MM,
    SOURCE_QPD,
    SPECTRAL_HARD_CAP,
    SPECTRAL_START,
    SPECTRAL_STEP,
    TAU_GRID,
)
from .pipeline import (
    AdaptiveProbeConfig,
    _done,
    crossing_d,
    existing_max,
    existing_min,
    file_sha_full,
    hash_select,
    primary_k,
    scale_list,
    sha16,
    write_df,
    write_json,
)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-12)).astype(np.float32)


def load_physics_X(root: Path, cfg: AdaptiveProbeConfig) -> np.ndarray:
    return load_model_X(cfg.mm(root), "vit_base")


def load_desi_X(root: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    t = pq.read_table(root / "data_hf/desi/desi_vit_base.parquet", columns=["vit_base_hsc"])
    X = np.stack(t.column("vit_base_hsc").to_pylist()).astype(np.float32)
    return l2_normalize(X)


def ensure_knn(path: Path, X: np.ndarray, query_rows: np.ndarray, k: int, device: torch.device, force: bool) -> np.ndarray:
    if path.exists() and not force:
        z = np.load(path)
        return z["neigh"]
    queries = X[np.asarray(query_rows, dtype=np.int64)]
    idx = knn_torch_ip(X, queries, k, device, batch=64 if device.type == "cpu" else 256)
    neigh = []
    for qi, row in enumerate(idx):
        qrow = int(query_rows[qi])
        r = [int(j) for j in row if int(j) != qrow]
        if len(r) < k:
            r = [int(j) for j in row][:k]
        neigh.append(r[:k])
    neigh = np.asarray(neigh, dtype=np.int64)
    np.savez_compressed(path, neigh=neigh)
    return neigh


def heldout_linear_r2(Z: np.ndarray, J: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, float]:
    """R²_L(d) using the full spherical-log ambient energy on held-out rows."""
    Zte = Z[te]
    energy = float(np.sum(Zte * Zte))
    if energy <= 0:
        return np.full(J.shape[1], np.nan), 0.0
    U = Zte @ J
    r2 = []
    for d in range(1, J.shape[1] + 1):
        Zhat = U[:, :d] @ J[:, :d].T
        sse = float(np.sum((Zte - Zhat) ** 2))
        r2.append(1.0 - sse / energy)
    return np.asarray(r2, dtype=np.float64), energy


def spectral_pass_anchor(
    X: np.ndarray,
    neigh: np.ndarray,
    ai: int,
    sid: int,
    k: int,
    d_max: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    N = neigh[ai, :k]
    Xloc = X[N].astype(np.float64)
    x0 = X[int(N[0])].astype(np.float64) if True else X[ai]
    # use the query point as anchor: queries are X[sids], neigh rows align to anchors
    x0 = X[ai].astype(np.float64) if X.shape[0] == neigh.shape[0] else Xloc[0]
    Z = sphere_log_map(x0, Xloc)
    th = angular_radii(x0, Xloc)
    tr, te = radial_stratified_halves(th, seed + ai)
    if min(len(tr), len(te)) < 8:
        return {"ok": False}
    J, ev = nested_uncentred_svd(Z[tr], d_max, device=device, centre=False)
    r2, energy = heldout_linear_r2(Z, J, te)
    return {
        "ok": True,
        "sample_id": int(sid),
        "k": int(k),
        "d_avail": int(J.shape[1]),
        "r2": r2,
        "energy": energy,
        "n_tr": int(len(tr)),
        "n_te": int(len(te)),
        "ev": ev,
    }


def pool_r2(per: list[dict[str, Any]], d_max: int) -> np.ndarray:
    num = np.zeros(d_max)
    den = np.zeros(d_max)
    for rec in per:
        if not rec.get("ok"):
            continue
        r2 = rec["r2"]
        e = rec["energy"]
        for i, v in enumerate(r2):
            if i >= d_max or not np.isfinite(v):
                continue
            # recover sse from r2 and energy
            sse = (1.0 - v) * e
            num[i] += sse
            den[i] += e
    out = np.full(d_max, np.nan)
    m = den > 0
    out[m] = 1.0 - num[m] / den[m]
    return out


def coarse_quadratic_ranks(crossings: dict[str, Any], d_avail: int) -> list[int]:
    """Geometry-derived coarse grid, capped at the curvature-reliability rank."""
    ds = set()
    for key in ("d_70", "d_75", "d_80", "d_825", "d_85", "d_875", "d_90"):
        v = crossings.get(key)
        if isinstance(v, int) and v <= d_avail:
            ds.add(v)
    # d_95 is recorded in the spectral table but is often a noise-tail rank
    for key in ("dL_plat", "dQ_plat"):
        v = crossings.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v) and int(v) <= d_avail:
            ds.add(int(v))
    if not ds:
        ds.update(range(4, min(d_avail, 20) + 1, 2))
    lo, hi = min(ds), max(ds)
    for d in range(max(2, lo - 3), min(d_avail, hi + 3) + 1, 2):
        ds.add(d)
    ds.add(min(d_avail, max(ds) + 2))
    return sorted(d for d in ds if 2 <= d <= d_avail)


def fit_quad_fixed_and_close(
    Z: np.ndarray,
    x0: np.ndarray,
    radii: np.ndarray,
    d: int,
    seed: int,
    thr: dict,
    device: torch.device,
    closest: bool,
) -> dict[str, float]:
    tr, te = radial_stratified_halves(radii, seed)
    if min(len(tr), len(te)) < d + 8:
        return {"ok": 0.0}
    J, _ = nested_uncentred_svd(Z[tr], d, device=device, centre=False)
    if J.shape[1] < d:
        return {"ok": 0.0}
    J = J[:, :d]
    Utr = Z[tr] @ J
    Phi, rms = scale_phi_train(phi2(Utr))
    Ytr = Z[tr] - Utr @ J.T
    G = Phi.T @ Phi
    grid = ridge_grid_from_gram(G, len(tr), int(thr["ridge_n_grid"]))
    # inner split for λ
    cut = max(len(tr) // 5, 8)
    inner_tr, inner_va = tr[cut:], tr[:cut]
    Ui = Z[inner_tr] @ J
    Phii, rmsi = scale_phi_train(phi2(Ui))
    Yi = Z[inner_tr] - Ui @ J.T
    Gi = Phii.T @ Phii
    Uva = Z[inner_va] @ J
    best = None
    for lam in grid:
        Bsc = ridge_fit(Phii, Yi, float(lam), G=Gi, C=Phii.T @ Yi)
        B = remove_radial_cols(Bsc / rmsi[None, :], x0)
        Zhat = predict_f(Uva, J, B)
        loss = nmse(Z[inner_va], Zhat)
        if best is None or loss < best[0]:
            best = (loss, float(lam), ridge_df(Gi, float(lam)))
    lam = best[1] if best else float(grid[0])
    Bsc = ridge_fit(Phi, Ytr, lam, G=G, C=Phi.T @ Ytr)
    B = remove_radial_cols(Bsc / rms[None, :], x0)
    BN = project_B_normal(B, J)
    Ute = Z[te] @ J
    Zlin = Ute @ J.T
    energy = float(np.sum(Z[te] * Z[te]))
    lin_nmse = nmse(Z[te], Zlin)
    if closest:
        u_max = float(np.quantile(np.linalg.norm(Utr, axis=1), float(thr["u_bound_q"]))) if len(Utr) else 1.0
        cpu = closest_point_project(
            Z[te], J, B, Ute, u_max=u_max, max_iter=int(thr["gn_max_iter"]), damp=float(thr["gn_damp"]), x_anchor=x0, device=device
        )
        q_nmse = cpu["close_nmse"]
        qn = closest_point_project(
            Z[te], J, BN, Ute, u_max=u_max, max_iter=int(thr["gn_max_iter"]), damp=float(thr["gn_damp"]), x_anchor=x0, device=device
        )
        qn_nmse = qn["close_nmse"]
    else:
        q_nmse = nmse(Z[te], predict_f(Ute, J, B))
        qn_nmse = nmse(Z[te], predict_f(Ute, J, BN))
    return {
        "ok": 1.0,
        "d": float(d),
        "lin_nmse": float(lin_nmse),
        "quad_nmse": float(q_nmse),
        "quadN_nmse": float(qn_nmse),
        "lin_r2": float(1.0 - lin_nmse),
        "quad_r2": float(1.0 - q_nmse),
        "lam": lam,
        "df": float(best[2]) if best else float("nan"),
        "df_frac": float((best[2] if best else np.nan) / max(n_quad_features(d), 1)),
        "n_tr": float(len(tr)),
        "n_te": float(len(te)),
        "m_d": float(n_quad_features(d)),
        "energy": energy,
        "test_sse_lin": float(np.sum((Z[te] - Zlin) ** 2)),
    }


def construct_interval(cross: dict[str, Any], d_curv_max: int) -> dict[str, Any]:
    d75 = cross.get("d_75")
    d90 = cross.get("d_90")
    dL_lo = cross.get("dL_plat_lo", cross.get("dL_plat"))
    dQ_lo = cross.get("dQ_plat_lo", cross.get("dQ_plat"))
    dL_hi = cross.get("dL_plat_hi", cross.get("dL_plat"))
    dQ_hi = cross.get("dQ_plat_hi", cross.get("dQ_plat"))
    lo_src = existing_min(d75, dL_lo, dQ_lo)
    hi_src = existing_max(d90, dL_hi, dQ_hi)
    if not np.isfinite(lo_src):
        lo_src = 2.0
    if not np.isfinite(hi_src):
        hi_src = float(d_curv_max)
        right_trunc_geom = True
    else:
        right_trunc_geom = d90 == "not_reached" or d90 is None
    d_low = int(max(2, lo_src - 2))
    d_high = int(hi_src + 2)
    d_high = min(d_high, int(d_curv_max))
    if d_low > d_high:
        d_low, d_high = 2, int(d_curv_max)
    # narrower primary if a high-τ crossing or linear plateau is a noise tail
    dL = cross.get("dL_plat")
    d95 = cross.get("d_95")
    d90 = cross.get("d_90")
    dQ = cross.get("dQ_plat")
    narrow_reason = ""
    d_low_p, d_high_p = d_low, d_high
    if (
        isinstance(d95, int)
        and isinstance(dL, (int, float))
        and np.isfinite(dL)
        and d95 > int(dL) + NOISE_TAIL_RANKS_PAST_PLATEAU
    ):
        d_high_p = min(d_high, int(dL) + 2, int(d_curv_max))
        narrow_reason = "d_95 sits past linear plateau + frozen noise-tail ranks; primary interval uses plateau+2"
    if (
        isinstance(dL, (int, float))
        and np.isfinite(dL)
        and isinstance(d90, int)
        and int(dL) > int(d90) + NOISE_TAIL_RANKS_PAST_PLATEAU
    ):
        hi_p = existing_max(d90, dQ if isinstance(dQ, (int, float)) else None)
        if np.isfinite(hi_p):
            d_high_p = min(d_high, int(hi_p) + 2, int(d_curv_max))
            narrow_reason = (narrow_reason + "; " if narrow_reason else "") + "linear plateau lies past d_90 + frozen noise-tail ranks"
    return {
        "d_low": d_low,
        "d_high": d_high,
        "d_low_primary": d_low_p,
        "d_high_primary": max(d_low_p, d_high_p),
        "right_truncated": bool(right_trunc_geom or d_high >= d_curv_max and (d90 == "not_reached" or d90 is None)),
        "narrow_reason": narrow_reason,
    }


def reuse_physics_qpd(root: Path) -> dict[str, Any] | None:
    qpd = resolve_path(root, SOURCE_QPD)
    curves = qpd / "aggregate_risk_curves.csv"
    per = qpd / "per_anchor_metrics.parquet"
    if not curves.exists():
        return None
    df = pd.read_csv(curves)
    return {"curves": df, "per_anchor": pd.read_parquet(per) if per.exists() else None, "path": str(curves)}


def run_geometry_dataset(
    root: Path,
    cfg: AdaptiveProbeConfig,
    *,
    dataset_id: str,
    X: np.ndarray,
    sids: list[int],
    sid_to_row: dict[int, int],
    k: int,
    device: torch.device,
    reuse_knn: np.ndarray | None,
    reuse_qpd: dict[str, Any] | None,
    t0: float,
) -> dict[str, Any]:
    out = cfg.resolved(root)
    ddir = out / "datasets" / dataset_id
    ddir.mkdir(parents=True, exist_ok=True)
    if (ddir / "interval.json").exists() and not cfg.force:
        import json

        interval = json.loads((ddir / "interval.json").read_text())
        crossings = {k: interval.get(k) for k in ("d_70", "d_75", "d_80", "d_825", "d_85", "d_875", "d_90", "d_95", "dL_plat", "dQ_plat", "dL_plat_lo", "dL_plat_hi", "dQ_plat_lo", "dQ_plat_hi", "quadratic_source")}
        query_rows = np.array([sid_to_row[s] for s in sids], dtype=np.int64)
        neigh = reuse_knn if reuse_knn is not None else ensure_knn(ddir / f"knn_k{int(interval.get('k', k))}.npz", X, query_rows, k, device, False)
        pool = pd.read_csv(ddir / "linear_risk_pooled.csv") if (ddir / "linear_risk_pooled.csv").exists() else None
        r2_pool = pool.r2_L_pooled.to_numpy(float) if pool is not None else None
        ds = pool.d.to_numpy(int) if pool is not None else np.arange(1, 2)
        return {
            "interval": interval,
            "crossings": crossings,
            "r2_pool": r2_pool,
            "ds": ds,
            "neigh": neigh,
            "sids": sids,
            "sid_to_row": sid_to_row,
            "k": k,
            "X": X,
            "ddir": ddir,
        }
    thr = DEFAULT_THRESHOLDS
    query_rows = np.array([sid_to_row[s] for s in sids], dtype=np.int64)
    if reuse_knn is not None:
        neigh = reuse_knn
    else:
        neigh = ensure_knn(ddir / f"knn_k{k}.npz", X, query_rows, k, device, cfg.force)

    # map neigh rows: if reuse from multimodel, neigh is over the 16384-row X and
    # sid_to_row[sid] is the local row. We need neigh[local_ai].
    def neigh_for(sid: int, ai: int) -> np.ndarray:
        return neigh[ai, :k]

    # Spectral: start at SPECTRAL_START and expand until 0.95 or cap
    d_spec = SPECTRAL_START if not cfg.smoke else 12
    lin_rows = []
    per_anchor = []
    r2_pool = None
    while d_spec <= (16 if cfg.smoke else SPECTRAL_HARD_CAP):
        per = []
        for ai, sid in enumerate(sids):
            N = neigh_for(sid, ai)
            x0 = X[sid_to_row[sid]].astype(np.float64)
            Xloc = X[N].astype(np.float64)
            Z = sphere_log_map(x0, Xloc)
            th = angular_radii(x0, Xloc)
            tr, te = radial_stratified_halves(th, cfg.seed + int(sid))
            if min(len(tr), len(te)) < 8:
                continue
            J, ev = nested_uncentred_svd(Z[tr], d_spec, device=device, centre=False)
            r2, energy = heldout_linear_r2(Z, J, te)
            rec = {"ok": True, "sample_id": int(sid), "r2": r2, "energy": energy, "d_avail": J.shape[1], "ev": ev}
            per.append(rec)
            for d, v in enumerate(r2, start=1):
                lin_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": int(sid),
                        "k": k,
                        "d": d,
                        "r2_L": float(v),
                        "test_energy": energy,
                        "test_sse_lin": float((1.0 - v) * energy),
                    }
                )
        r2_pool = pool_r2(per, d_spec)
        per_anchor = per
        need_more = any(crossing_d(np.arange(1, d_spec + 1), r2_pool, tau) == "not_reached" for tau in (0.90, 0.95))
        if not need_more or d_spec >= SPECTRAL_HARD_CAP or cfg.smoke:
            break
        d_spec = min(d_spec + SPECTRAL_STEP, SPECTRAL_HARD_CAP)

    ds = np.arange(1, len(r2_pool) + 1)
    crossings = {f"d_{str(tau).replace('.', '')}": crossing_d(ds, r2_pool, tau) for tau in TAU_GRID}
    # nicer keys
    crossings = {
        "d_70": crossing_d(ds, r2_pool, 0.70),
        "d_75": crossing_d(ds, r2_pool, 0.75),
        "d_80": crossing_d(ds, r2_pool, 0.80),
        "d_825": crossing_d(ds, r2_pool, 0.825),
        "d_85": crossing_d(ds, r2_pool, 0.85),
        "d_875": crossing_d(ds, r2_pool, 0.875),
        "d_90": crossing_d(ds, r2_pool, 0.90),
        "d_95": crossing_d(ds, r2_pool, 0.95),
    }
    nmse_l = 1.0 - r2_pool
    plat_L = plateau_from_curve(ds, nmse_l, None, thr)
    crossings["dL_plat"] = plat_L["d_plat"]
    crossings["dL_plat_lo"] = plat_L["d_plat"]
    crossings["dL_plat_hi"] = plat_L["d_plat"]

    n_tr = k // 2
    d_curv_max = 2
    for d in range(2, int(ds.max()) + 1):
        m = n_quad_features(d)
        if m < n_tr or (m < 2 * n_tr and d <= 24):
            d_curv_max = d
        else:
            break
    if cfg.smoke:
        d_curv_max = min(d_curv_max, 8)

    # Reuse qpd quadratic if physics
    q_rows = []
    q_ref_rows = []
    if reuse_qpd is not None and reuse_qpd.get("curves") is not None:
        cdf = reuse_qpd["curves"]
        if "d" in cdf.columns:
            for _, r in cdf.iterrows():
                q_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "d": int(r.d),
                        "source": "reused_qpd",
                        "lin_nmse": float(r.nmse_lin_med) if "nmse_lin_med" in cdf.columns else float("nan"),
                        "quad_nmse": float(r.nmse_quad_med) if "nmse_quad_med" in cdf.columns else float("nan"),
                        "quad_close_nmse": float(r.nmse_quad_med) if "nmse_quad_med" in cdf.columns else float("nan"),
                        "stage": "reuse",
                    }
                )
            nmse_q = np.array([float(cdf.loc[cdf.d == d, "nmse_quad_med"].iloc[0]) if (cdf.d == d).any() and "nmse_quad_med" in cdf.columns else np.nan for d in ds if d <= int(cdf.d.max())])
            ds_q = np.array([d for d in ds if d <= int(cdf.d.max())])
            plat_Q = plateau_from_curve(ds_q, nmse_q, None, thr)
            crossings["dQ_plat"] = plat_Q["d_plat"]
            crossings["dQ_plat_lo"] = plat_Q["d_plat"]
            crossings["dQ_plat_hi"] = plat_Q["d_plat"]
            crossings["quadratic_source"] = "reused_physics_qpd_closest_point"
    else:
        crossings["dQ_plat"] = "not_reached"
        grid = coarse_quadratic_ranks(crossings, int(d_curv_max))
        if cfg.smoke:
            grid = [d for d in grid if d <= 8][:4] or [4, 6]
        n_coarse = 8 if cfg.smoke else min(128, len(sids))
        for ai, sid in enumerate(sids[:n_coarse]):
            N = neigh_for(sid, ai)
            x0 = X[sid_to_row[sid]].astype(np.float64)
            Xloc = X[N].astype(np.float64)
            Z = sphere_log_map(x0, Xloc)
            th = angular_radii(x0, Xloc)
            for d in grid:
                rec = fit_quad_fixed_and_close(Z, x0, th, d, cfg.seed + int(sid), thr, device, closest=False)
                rec.update({"dataset_id": dataset_id, "sample_id": int(sid), "k": k, "stage": "coarse_fixed"})
                q_rows.append(rec)
        qdf = pd.DataFrame(q_rows)
        if len(qdf) and "quad_nmse" in qdf.columns:
            g = qdf.groupby("d").mean(numeric_only=True)
            ds_q = np.array(sorted(int(x) for x in g.index))
            nmse_q = np.array([float(g.loc[d].quad_nmse) for d in ds_q])
            plat_Q = plateau_from_curve(ds_q, nmse_q, None, thr)
            crossings["dQ_plat"] = plat_Q["d_plat"]
            crossings["dQ_plat_lo"] = plat_Q["d_plat"]
            crossings["dQ_plat_hi"] = plat_Q["d_plat"]
        crossings["quadratic_source"] = "coarse_fixed_then_fine_closest"
        # fine closest-point around candidates
        fine = set()
        for key in ("d_80", "d_85", "d_90", "dL_plat", "dQ_plat"):
            v = crossings.get(key)
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                iv = int(v)
                if iv > d_curv_max:
                    continue
                fine.update(range(max(2, iv - 2), min(d_curv_max, iv + 2) + 1))
        if cfg.smoke:
            fine = {d for d in fine if d <= 8} or {4, 6}
        for ai, sid in enumerate(sids[: (4 if cfg.smoke else min(64, len(sids)))]):
            N = neigh_for(sid, ai)
            x0 = X[sid_to_row[sid]].astype(np.float64)
            Xloc = X[N].astype(np.float64)
            Z = sphere_log_map(x0, Xloc)
            th = angular_radii(x0, Xloc)
            for d in sorted(fine):
                rec = fit_quad_fixed_and_close(Z, x0, th, d, cfg.seed + 17 + int(sid), thr, device, closest=True)
                rec.update({"dataset_id": dataset_id, "sample_id": int(sid), "k": k, "stage": "fine_closest"})
                q_ref_rows.append(rec)
        if q_ref_rows:
            rdf = pd.DataFrame(q_ref_rows)
            g = rdf.groupby("d").mean(numeric_only=True)
            ds_q = np.array(sorted(int(x) for x in g.index))
            nmse_q = np.array([float(g.loc[d].quad_nmse) for d in ds_q])
            plat_Q = plateau_from_curve(ds_q, nmse_q, None, thr)
            crossings["dQ_plat"] = plat_Q["d_plat"]
            crossings["dQ_plat_lo"] = plat_Q["d_plat"]
            crossings["dQ_plat_hi"] = plat_Q["d_plat"]

    interval = construct_interval(crossings, d_curv_max)
    interval.update(
        {
            "dataset_id": dataset_id,
            "k": k,
            "n_obs": int(len(X)),
            "n_anchors": int(len(sids)),
            "d_spec_max": int(ds.max()),
            "d_curv_max": int(d_curv_max),
            "curvature_range_right_truncated": bool(interval["right_truncated"]),
            **{k2: crossings[k2] for k2 in crossings},
        }
    )
    write_df(ddir / "linear_risk_curves.csv", pd.DataFrame(lin_rows), force=cfg.force)
    write_df(ddir / "quadratic_screening.csv", pd.DataFrame(q_rows) if q_rows else pd.DataFrame([{"dataset_id": dataset_id}]), force=cfg.force)
    write_df(ddir / "quadratic_refinement.csv", pd.DataFrame(q_ref_rows) if q_ref_rows else pd.DataFrame([{"dataset_id": dataset_id}]), force=cfg.force)
    cross_row = {"dataset_id": dataset_id, "k": k, **{k2: crossings[k2] for k2 in crossings}}
    for i, tau in enumerate(TAU_GRID):
        cross_row[f"r2_at_max_d_vs_tau_{tau}"] = float(r2_pool[-1]) if r2_pool is not None else float("nan")
    write_df(ddir / "spectral_crossings.csv", pd.DataFrame([cross_row]), force=cfg.force)
    # pooled linear curve
    pool_df = pd.DataFrame({"dataset_id": dataset_id, "d": ds, "r2_L_pooled": r2_pool, "k": k})
    write_df(ddir / "linear_risk_pooled.csv", pool_df, force=cfg.force)
    write_json(ddir / "interval.json", interval, force=cfg.force)
    write_json(
        ddir / "geometry_meta.json",
        {
            "l2": row_l2_status(X),
            "k": k,
            "k_rule": f"largest preset <= {0.125}*n",
            "n": int(len(X)),
            "anchors": sids,
            "labels_loaded": False,
        },
        force=cfg.force,
    )
    return {
        "interval": interval,
        "crossings": crossings,
        "r2_pool": r2_pool,
        "ds": ds,
        "neigh": neigh,
        "sids": sids,
        "sid_to_row": sid_to_row,
        "k": k,
        "X": X,
        "ddir": ddir,
    }
