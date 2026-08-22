"""Staged falsification of global-probe × sphere-normal curvature associations.

Corrected UniverseTBD protocol only: fixed global probes, geographic local scores.
Never refits probes at anchors. Primary cell: mag_r_desi, k=2048, α=100.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew, spearmanr
from sklearn.neighbors import NearestNeighbors

from .confirmatory_object_curvature import _fit_neighborhood, decompose_BS, select_anchors
from .curvature_probe_alignment import B0_flat_for_svd, traceless_B0
from .curvature_probe_screen import (
    EXPECTED_HASH,
    LOCAL_DIM,
    ScreenConfig,
    load_frozen_curvature,
    partial_spearman,
    spearman_dict,
)
from .curvature_probe_subspace_ablation import normal_pca_basis
from .data import load_prepare
from .global_probe_curvature_alignment import (
    CANONICAL_PROBE_LOCAL,
    CANONICAL_PROBE_TRAIN,
    PROBE_ALPHA,
    local_r2_fixed_predictions,
    projection_energies,
    weighted_r2,
)
from .paths import platonic_root, resolve_path
from .sphere_normal_quadratic import normalize_rows

EPS = 1e-12
PRIMARY_TARGET = "mag_r_desi"
PRIMARY_K = 2048
DIMS_SMOKE = (8, 12, 16, 24, 32)
DIMS_CONFIRM = (8, 16, 32)


def n_quad_features(d: int) -> int:
    return int(d * (d + 1) // 2)


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def mc_p(real: float, nulls: np.ndarray, *, greater: bool = False) -> tuple[float, int]:
    nulls = np.asarray(nulls, float)
    nulls = nulls[np.isfinite(nulls)]
    B = len(nulls)
    if B == 0 or not np.isfinite(real):
        return float("nan"), 0
    if greater:
        return float((1 + np.sum(nulls >= real)) / (B + 1)), B
    return float((1 + np.sum(np.abs(nulls) >= abs(real))) / (B + 1)), B


@dataclass
class FalsificationConfig:
    output_dir: str = "outputs/geometry/physics_global_probe_curvature_falsification"
    align_dir: str = "outputs/geometry/physics_global_probe_curvature_alignment"
    mag_dir: str = "outputs/geometry/physics_global_probe_curvature_magnitude"
    geometry_cache: str = (
        "outputs/geometry/physics_curvature_probe_multitarget/geometry_cache"
    )
    curvature_path: str = (
        "outputs/geometry/physics_quadratic_atlas_sphere_normal/"
        "object_curvature_features_aggregated.parquet"
    )
    prepare_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    expected_hash: str = EXPECTED_HASH
    primary_target: str = PRIMARY_TARGET
    secondary_target: str = "photo_z"
    primary_k: int = PRIMARY_K
    probe_alpha: float = PROBE_ALPHA
    smoke_anchors: int = 96
    n_null: int = 40
    n_permute: int = 300
    seed: int = 0
    force: bool = False
    device: str = "cuda"
    batch_anchors: int = 16
    near_zero_q: float = 0.05  # training-activation quantile for near-zero
    stages: str = "all"  # or comma list 1-8
    confirm_full: bool = True  # run 384 if smoke gates pass

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


# -------------------- shared stats --------------------


def controls_C0(g: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            g.log_knn_radius.to_numpy(float),
            g.local_target_variance.to_numpy(float),
            g.reconstruction_error.to_numpy(float),
            g.local_evaluation_count.to_numpy(float),
        ]
    )


def controls_C2(g: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            controls_C0(g),
            g.A_N.to_numpy(float),
            g.A_T.to_numpy(float),
            g.A_PCA_normal.to_numpy(float),
        ]
    )


def unit_normal_component(w: np.ndarray, T: np.ndarray, x0u: np.ndarray) -> np.ndarray:
    wn = float(np.linalg.norm(w))
    if wn < EPS:
        return np.zeros_like(w)
    wh = w / wn
    x0 = x0u / max(np.linalg.norm(x0u), EPS)
    wN = wh - T @ (T.T @ wh) - x0 * float(np.dot(x0, wh))
    n = float(np.linalg.norm(wN))
    return wN / n if n > EPS else wN


def probe_facing(B0: np.ndarray, H: np.ndarray, wN_u: np.ndarray, d: int) -> dict:
    Bflat = B0_flat_for_svd(B0, d)
    k_probe = float(np.linalg.norm(Bflat.T @ wN_u))
    h_probe = float(abs(np.dot(H, wN_u)))
    return {"K_probe": k_probe, "H_probe": h_probe}


def assoc_block(r2: np.ndarray, feats: dict[str, np.ndarray], C0: np.ndarray, C2: np.ndarray, cfg: FalsificationConfig) -> dict:
    out = {}
    for name, x in feats.items():
        out[f"raw_{name}"] = spearman_dict(r2, x)["rho"]
        Z = C2 if name.startswith("A_B") else C0
        part = partial_spearman(x, r2, Z)
        out[f"partial_{name}"] = part["rho"]
        # perm
        rng = np.random.default_rng(cfg.seed + hash(name) % 10000)
        nulls = []
        for _ in range(cfg.n_permute):
            rp = r2.copy()
            m = np.isfinite(rp)
            rp[m] = rng.permutation(rp[m])
            nulls.append(partial_spearman(x, rp, Z)["rho"])
        out[f"p_perm_{name}"], _ = mc_p(part["rho"], np.asarray(nulls))
    # interaction coef via rank residualized OLS
    ab = feats.get("A_B_normal")
    kt = feats.get("K_traceless")
    if ab is not None and kt is not None:
        from .global_probe_curvature_magnitude import rank_z

        yz = rank_z(r2)
        abz, ktz = rank_z(ab), rank_z(kt)
        # residualize on C2 ranks
        Zr = np.column_stack([rank_z(C2[:, j]) for j in range(C2.shape[1])])
        m = np.isfinite(yz) & np.isfinite(abz) & np.isfinite(ktz) & np.all(np.isfinite(Zr), axis=1)
        if m.sum() > 20:
            A = np.column_stack([np.ones(m.sum()), Zr[m]])
            def resid(v):
                b, *_ = np.linalg.lstsq(A, v[m], rcond=None)
                return v[m] - A @ b
            yr, ar, kr = resid(yz), resid(abz), resid(ktz)
            X = np.column_stack([np.ones(len(yr)), ar, kr, ar * kr])
            beta, *_ = np.linalg.lstsq(X, yr, rcond=None)
            out["coef_interaction"] = float(beta[3])
            # bootstrap CI
            rng = np.random.default_rng(cfg.seed + 11)
            boots = []
            for _ in range(min(400, cfg.n_permute)):
                take = rng.choice(len(yr), size=len(yr), replace=True)
                b, *_ = np.linalg.lstsq(X[take], yr[take], rcond=None)
                boots.append(float(b[3]))
            out["ci_interaction"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        else:
            out["coef_interaction"] = float("nan")
            out["ci_interaction"] = [float("nan"), float("nan")]
    return out


# -------------------- subset selection --------------------


def select_smoke_anchors(mag_g: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Chart/radius-stratified subset; does NOT use probe R²."""
    g = mag_g.drop_duplicates("sample_id").copy()
    # stratify by radius tertile × recon tertile
    g["r_bin"] = pd.qcut(g.knn_radius, 3, labels=False, duplicates="drop")
    g["e_bin"] = pd.qcut(g.reconstruction_error, 3, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    picks = []
    groups = list(g.groupby(["r_bin", "e_bin"], observed=True))
    per = max(1, n // max(len(groups), 1))
    for _, sub in groups:
        take = min(per, len(sub))
        picks.extend(rng.choice(sub.sample_id.to_numpy(), size=take, replace=False).tolist())
    picks = list(dict.fromkeys(picks))
    if len(picks) < n:
        rest = g[~g.sample_id.isin(picks)].sample_id.to_numpy()
        need = n - len(picks)
        if len(rest):
            picks.extend(rng.choice(rest, size=min(need, len(rest)), replace=False).tolist())
    return np.asarray(sorted(picks[:n]), dtype=np.int64)


# -------------------- Stage 1: geometry audit --------------------


def stage1_audit(
    root: Path,
    cfg: FalsificationConfig,
    X: np.ndarray,
    pack_map: dict,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
    w_probe: np.ndarray,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / "geometry_audit.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    rows = []
    h_dir = resolve_path(root, cfg.mag_dir) / "H_cache"
    for sid in smoke_ids:
        p = pack_map.get((int(sid), cfg.primary_k))
        if p is None:
            continue
        z = np.load(p)
        neigh = z["neigh"]
        T, x0u, B0 = z["T"], z["x0u"], z["B0"]
        d = T.shape[1]
        q = n_quad_features(d)
        rho = float(z["rho"])
        # rematerialize for H + audit diagnostics
        chart, chart_RS, info, U, glob, reason = _fit_neighborhood(
            X, neigh, d, seed=cfg.seed + 17 * int(z["ai"]) + cfg.primary_k
        )
        if chart is None:
            rows.append({"sample_id": int(sid), "scale_k": cfg.primary_k, "ok": False, "reason": reason})
            continue
        B0c, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
        d_eff = chart.J.shape[1]
        x0 = chart.x0 / max(np.linalg.norm(chart.x0), EPS)
        J = chart.J
        # orthogonality
        pt_H = float(np.linalg.norm(J.T @ H))
        r_H = float(abs(np.dot(x0, H)))
        # ||P_T B°|| : apply P_T to each B0[:,a,b]
        pt_B = 0.0
        r_B = 0.0
        for a in range(d_eff):
            for b in range(a, d_eff):
                v = B0c[:, a, b]
                pt_B += float(np.sum((J.T @ v) ** 2))
                r_B += float(np.dot(x0, v) ** 2)
        pt_B = float(np.sqrt(pt_B))
        r_B = float(np.sqrt(r_B))
        # thickness
        dx = X[neigh] - x0u[None, :]
        u = dx @ T
        tang_var = float(np.var(u))
        resid = dx - u @ T.T
        # remove radial in resid
        resid = resid - np.outer(resid @ x0u, x0u)
        normal_var = float(np.var(resid))
        total_var = tang_var + normal_var + EPS
        # eigengap of tangent cov
        C = u.T @ u / max(len(u), 1)
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        gap = float(ev[min(d_eff - 1, len(ev) - 1)] / max(ev[0], EPS)) if len(ev) else float("nan")
        # if next eigenvalue exists
        if len(ev) > d_eff:
            gap = float(ev[d_eff - 1] / max(ev[d_eff], EPS))
        recon = float(info.get("val_E_TRS", np.nan))
        n_eff = float(len(neigh))
        row_m = mag_g[mag_g.sample_id == sid]
        rows.append(
            {
                "sample_id": int(sid),
                "scale_k": cfg.primary_k,
                "ok": True,
                "tangent_dim": int(d_eff),
                "quadratic_feature_count": int(n_quad_features(d_eff)),
                "n_eff": n_eff,
                "neff_over_q": n_eff / max(n_quad_features(d_eff), 1),
                "tangent_eigengap": gap,
                "tangent_variance_frac": tang_var / total_var,
                "normal_residual_variance_frac": normal_var / total_var,
                "recon_over_radius": recon / max(rho, EPS) if np.isfinite(recon) else float("nan"),
                "val_E_TRS": recon,
                "knn_radius": rho,
                "P_T_H_norm": pt_H,
                "x0_dot_H_abs": r_H,
                "P_T_B0_fro": pt_B,
                "x0_dot_B0_fro": r_B,
                "H_orth_ok": pt_H < 1e-4 and r_H < 1e-4,
                "B0_orth_ok": pt_B < 1e-3 and r_B < 1e-3,
                "K_traceless": float(np.linalg.norm(B0c)),
                "K_mean": float(np.linalg.norm(H)),
                "local_r2": float(row_m.local_r2.iloc[0]) if len(row_m) else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    thin = float(df.loc[df.ok, "normal_residual_variance_frac"].median()) if df.ok.any() else float("nan")
    (out / "stage1_summary.json").write_text(
        json.dumps(
            {
                "median_normal_var_frac": thin,
                "frac_H_orth_ok": float(df.loc[df.ok, "H_orth_ok"].mean()) if df.ok.any() else float("nan"),
                "frac_B0_orth_ok": float(df.loc[df.ok, "B0_orth_ok"].mean()) if df.ok.any() else float("nan"),
                "median_neff_over_q": float(df.loc[df.ok, "neff_over_q"].median()) if df.ok.any() else float("nan"),
                "density_ridge_language": thin > 0.25,
            },
            indent=2,
        )
    )
    print(f"[falsify] stage1 n={len(df)} thin_median={thin:.3f}", flush=True)
    return df


# -------------------- Stage 2: dimension sweep --------------------


def fit_anchor_dim(
    X: np.ndarray,
    neigh: np.ndarray,
    ai: int,
    k: int,
    d: int,
    seed: int,
    w_probe: np.ndarray,
    rho: float,
) -> dict | None:
    chart, chart_RS, info, U, glob, reason = _fit_neighborhood(X, neigh, d, seed=seed + 17 * ai + k + 1000 * d)
    if chart is None:
        return {"ok": False, "reason": reason, "d_requested": d}
    B0, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
    T, x0u = chart.J, chart.x0
    d_eff = T.shape[1]
    # A_B via left singular vectors of Bflat
    Bflat = B0_flat_for_svd(B0, d_eff)
    U_b, s, _ = np.linalg.svd(Bflat, full_matrices=False)
    keep = s > 1e-8 * (s[0] if len(s) else 1.0)
    UB = U_b[:, keep] if np.any(keep) else U_b
    en = projection_energies(w_probe, T, x0u, UB, UB)
    wN = unit_normal_component(w_probe, T, x0u)
    pf = probe_facing(B0, H, wN, d_eff)
    dec = decompose_BS(chart.BS_flat, d_eff)
    return {
        "ok": True,
        "reason": "",
        "d_requested": d,
        "d_eff": d_eff,
        "K_traceless": dec["B_traceless_fro"],
        "K_mean": dec["H_norm"],
        "A_B_normal": en["A_B_normal"],
        "A_B_total": en["A_B_total"],
        "A_N": en["A_N"],
        "A_T": en["A_T"],
        "K_probe": pf["K_probe"],
        "H_probe": pf["H_probe"],
        "val_E_TRS": float(info.get("val_E_TRS", np.nan)),
        "neff_over_q": float(len(neigh)) / max(n_quad_features(d_eff), 1),
    }


def stage2_dimension_sweep(
    root: Path,
    cfg: FalsificationConfig,
    X: np.ndarray,
    pack_map: dict,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
    w_probe: np.ndarray,
    dims: tuple[int, ...],
    tag: str,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / f"dimension_sensitivity_{tag}.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    rows = []
    for d in dims:
        shard = out / f"dim_d{d}_{tag}.parquet"
        if _done(shard, cfg.force):
            rows.append(pd.read_parquet(shard))
            continue
        batch = []
        t0 = time.time()
        for i, sid in enumerate(smoke_ids):
            p = pack_map.get((int(sid), cfg.primary_k))
            if p is None:
                continue
            z = np.load(p)
            fit = fit_anchor_dim(
                X, z["neigh"], int(z["ai"]), cfg.primary_k, d, cfg.seed, w_probe, float(z["rho"])
            )
            if fit is None:
                continue
            row_m = mag_g[mag_g.sample_id == sid].iloc[0]
            fit.update(
                {
                    "sample_id": int(sid),
                    "scale_k": cfg.primary_k,
                    "local_r2": float(row_m.local_r2),
                    "log_knn_radius": float(row_m.log_knn_radius),
                    "local_target_variance": float(row_m.local_target_variance),
                    "reconstruction_error": float(row_m.reconstruction_error),
                    "local_evaluation_count": float(row_m.local_evaluation_count),
                    "A_PCA_normal": float(row_m.A_PCA_normal),
                }
            )
            # for d!=8 A_PCA from cache is frozen-d=8; keep as nuisance proxy
            batch.append(fit)
            if (i + 1) % 16 == 0:
                print(f"[falsify] stage2 d={d} {i+1}/{len(smoke_ids)} rss={_rss():.0f}", flush=True)
        bdf = pd.DataFrame(batch)
        bdf.to_parquet(shard, index=False)
        rows.append(bdf)
        print(f"[falsify] stage2 d={d} done in {time.time()-t0:.1f}s n={len(bdf)}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    # associations per d
    assoc_rows = []
    for d, g in df.groupby("d_requested"):
        gok = g[g.ok].copy()
        if len(gok) < 20:
            continue
        feats = {
            "A_B_normal": gok.A_B_normal.to_numpy(float),
            "K_traceless": gok.K_traceless.to_numpy(float),
            "K_mean": gok.K_mean.to_numpy(float),
            "K_probe": gok.K_probe.to_numpy(float),
        }
        ab = assoc_block(gok.local_r2.to_numpy(float), feats, controls_C0(gok), controls_C2(gok), cfg)
        ab.update({"d_requested": int(d), "n": int(len(gok)), "tag": tag})
        assoc_rows.append(ab)
    assoc = pd.DataFrame(assoc_rows)
    assoc.to_parquet(out / f"dimension_sensitivity.parquet" if tag == "smoke" else out / f"dimension_sensitivity_{tag}_assoc.parquet", index=False)
    if tag == "smoke":
        # also write combined path expected by outputs list
        assoc.to_parquet(out / "dimension_sensitivity.parquet", index=False)
    df.to_parquet(path, index=False)
    return df


def stage2_smoke_gate(assoc: pd.DataFrame) -> dict:
    """Pass if primary associations keep sign from d=8 to d=32 (or stay null)."""
    if assoc.empty:
        return {"pass": False, "reason": "empty"}
    a8 = assoc[assoc.d_requested == 8]
    a32 = assoc[assoc.d_requested == 32]
    if a8.empty or a32.empty:
        return {"pass": False, "reason": "missing_d"}
    r8, r32 = a8.iloc[0], a32.iloc[0]
    flips = []
    for key in ["partial_K_mean", "partial_K_probe", "coef_interaction"]:
        v8, v32 = r8.get(key, np.nan), r32.get(key, np.nan)
        if np.isfinite(v8) and np.isfinite(v32) and abs(v8) > 0.08 and np.sign(v8) != np.sign(v32) and abs(v32) > 0.05:
            flips.append(key)
    return {
        "pass": len(flips) == 0,
        "flips": flips,
        "label_if_fail": "tangent_underfit",
        "partial_K_mean_d8": float(r8.get("partial_K_mean", np.nan)),
        "partial_K_mean_d32": float(r32.get("partial_K_mean", np.nan)),
        "partial_K_probe_d8": float(r8.get("partial_K_probe", np.nan)),
        "partial_K_probe_d32": float(r32.get("partial_K_probe", np.nan)),
        "coef_interaction_d8": float(r8.get("coef_interaction", np.nan)),
        "coef_interaction_d32": float(r32.get("coef_interaction", np.nan)),
    }


# -------------------- Stage 3: sparsity / boundary --------------------


def stage3_sparsity(
    root: Path,
    cfg: FalsificationConfig,
    X: np.ndarray,
    train: np.ndarray,
    pack_map: dict,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / "sparsity_boundary_metrics.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    Xtr = X[train]
    thr = float(np.quantile(np.abs(Xtr), cfg.near_zero_q))
    rows = []
    for sid in smoke_ids:
        p = pack_map.get((int(sid), cfg.primary_k))
        if p is None:
            continue
        z = np.load(p)
        neigh = z["neigh"]
        Xn = X[neigh]
        T, x0u = z["T"], z["x0u"]
        u = (Xn - x0u) @ T
        absx = np.abs(Xn)
        exact0 = float((Xn == 0).mean())
        near0 = float((absx < thr).mean())
        # Hoyer sparsity per row then mean
        l1 = np.sum(absx, axis=1)
        l2 = np.linalg.norm(Xn, axis=1)
        n = Xn.shape[1]
        hoyer = (np.sqrt(n) - l1 / np.maximum(l2, EPS)) / (np.sqrt(n) - 1 + EPS)
        # Gini of |x| mean
        def gini(v):
            v = np.sort(np.abs(v).ravel())
            if v.sum() < EPS:
                return 0.0
            idx = np.arange(1, len(v) + 1)
            return float((2 * np.sum(idx * v) / (len(v) * v.sum())) - (len(v) + 1) / len(v))

        # participation ratio of mean |coord|
        mabs = absx.mean(0)
        pr = float((mabs.sum() ** 2) / max(np.sum(mabs**2), EPS)) / n
        # support turnover: Jaccard of near-zero sets between consecutive neighbors (sorted by ||u||)
        order = np.argsort(np.linalg.norm(u, axis=1))
        turns = []
        for a, b in zip(order[:-1], order[1:]):
            sa = absx[a] < thr
            sb = absx[b] < thr
            inter = np.logical_and(sa, sb).sum()
            union = np.logical_or(sa, sb).sum()
            turns.append(1.0 - inter / max(union, 1))
        # tangent skewness / directional imbalance
        skew_u = float(np.mean([skew(u[:, j]) for j in range(u.shape[1])]))
        # antipodal: for each tangent dir, fraction with positive vs negative projection on top PC
        v1 = np.linalg.svd(u, full_matrices=False)[2][0]
        proj = u @ v1
        antip = float(min((proj > 0).mean(), (proj < 0).mean()) * 2)  # 1=balanced, 0=one-sided
        imb = float(abs((proj > 0).mean() - 0.5))
        row_m = mag_g[mag_g.sample_id == sid].iloc[0]
        rows.append(
            {
                "sample_id": int(sid),
                "scale_k": cfg.primary_k,
                "near_zero_threshold": thr,
                "exact_zero_fraction": exact0,
                "near_zero_fraction": near0,
                "Hoyer_sparsity": float(np.mean(hoyer)),
                "abs_gini": gini(mabs),
                "coord_participation_ratio": pr,
                "support_turnover": float(np.mean(turns)) if turns else float("nan"),
                "tangent_skewness": skew_u,
                "tangent_directional_imbalance": imb,
                "antipodal_coverage": antip,
                "knn_radius": float(row_m.knn_radius),
                "K_traceless": float(row_m.K_traceless),
                "K_mean": float(row_m.K_mean),
                "local_r2": float(row_m.local_r2),
                "A_B_normal": float(row_m.A_B_normal),
                "log_knn_radius": float(row_m.log_knn_radius),
                "local_target_variance": float(row_m.local_target_variance),
                "reconstruction_error": float(row_m.reconstruction_error),
                "local_evaluation_count": float(row_m.local_evaluation_count),
                "A_N": float(row_m.A_N),
                "A_T": float(row_m.A_T),
                "A_PCA_normal": float(row_m.A_PCA_normal),
            }
        )
    df = pd.DataFrame(rows)
    # explain curvature / R2
    expl = {}
    for yname in ["K_traceless", "K_mean", "local_r2"]:
        for xname in [
            "near_zero_fraction",
            "Hoyer_sparsity",
            "support_turnover",
            "tangent_directional_imbalance",
            "antipodal_coverage",
            "knn_radius",
        ]:
            expl[f"rho_{yname}__{xname}"] = spearman_dict(df[yname].to_numpy(float), df[xname].to_numpy(float))["rho"]
    # partial curvature with sparsity nuisances
    C_sp = np.column_stack(
        [
            df.log_knn_radius,
            df.local_target_variance,
            df.reconstruction_error,
            df.local_evaluation_count,
            df.near_zero_fraction,
            df.support_turnover,
            df.tangent_directional_imbalance,
        ]
    ).astype(float)
    for feat in ["K_mean", "K_traceless", "A_B_normal"]:
        expl[f"partial_{feat}_with_sparsity"] = partial_spearman(
            df[feat].to_numpy(float), df.local_r2.to_numpy(float), C_sp
        )["rho"]
    (out / "stage3_explanations.json").write_text(json.dumps(expl, indent=2))
    df.to_parquet(path, index=False)
    print(f"[falsify] stage3 n={len(df)}", flush=True)
    return df


# -------------------- Stage 4: flat matched nulls --------------------


def make_null_neighborhood(
    Xn: np.ndarray, x0u: np.ndarray, T: np.ndarray, rng: np.random.Generator, n_bins: int = 8
) -> np.ndarray:
    """Resample normal residuals within ||u|| bins; destroy E[resid|direction]."""
    dx = Xn - x0u[None, :]
    u = dx @ T
    tang = u @ T.T
    resid = dx - tang
    # remove radial component from resid
    x0 = x0u / max(np.linalg.norm(x0u), EPS)
    resid = resid - np.outer(resid @ x0, x0)
    ru = np.linalg.norm(u, axis=1)
    # bin by radius
    qs = np.quantile(ru, np.linspace(0, 1, n_bins + 1))
    resid_null = resid.copy()
    for b in range(n_bins):
        if b < n_bins - 1:
            m = (ru >= qs[b]) & (ru < qs[b + 1])
        else:
            m = ru >= qs[b]
        idx = np.where(m)[0]
        if len(idx) <= 1:
            continue
        perm = rng.permutation(idx)
        resid_null[idx] = resid[perm]
    Y = x0u[None, :] + tang + resid_null
    return normalize_rows(Y)


def fit_metrics_on_points(
    Xn: np.ndarray, d: int, seed: int, w_probe: np.ndarray, rho: float
) -> dict | None:
    n = len(Xn)
    if n < max(40, 5 * d):
        return None
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_g = max(15, int(0.4 * n))
    n_f = max(15, int(0.4 * n))
    from .sphere_normal_quadratic import fit_nested_chart

    w = np.ones(n)
    g, f, v = order[:n_g], order[n_g : n_g + n_f], order[n_g + n_f :]
    if len(v) < 8:
        return None
    try:
        chart, _, info, _U = fit_nested_chart(Xn, np.zeros((n, d)), w, g, f, v)
    except Exception:
        return None
    if chart.J.shape[1] < max(2, d // 2):
        return None
    B0, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
    d_eff = chart.J.shape[1]
    Bflat = B0_flat_for_svd(B0, d_eff)
    Ub, s, _ = np.linalg.svd(Bflat, full_matrices=False)
    keep = s > 1e-8 * (s[0] if len(s) else 1.0)
    UB = Ub[:, keep] if np.any(keep) else Ub
    en = projection_energies(w_probe, chart.J, chart.x0, UB, UB)
    wN = unit_normal_component(w_probe, chart.J, chart.x0)
    pf = probe_facing(B0, H, wN, d_eff)
    dec = decompose_BS(chart.BS_flat, d_eff)
    return {
        "K_traceless": dec["B_traceless_fro"],
        "K_mean": dec["H_norm"],
        "K_probe": pf["K_probe"],
        "H_probe": pf["H_probe"],
        "A_B_normal": en["A_B_normal"],
    }


def synthetic_controls(d: int, D: int, n: int, rng: np.random.Generator, w_probe: np.ndarray) -> list[dict]:
    rows = []
    # flat full plane
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - J @ (J.T @ x0)
    x0 /= np.linalg.norm(x0)
    u = rng.normal(size=(n, d)) * 0.3
    Xflat = normalize_rows(x0 + u @ J.T)
    m = fit_metrics_on_points(Xflat, d, int(rng.integers(1e9)), w_probe, 0.5)
    if m:
        rows.append({"synth": "flat_full_plane", **m})
    # flat half-ball
    u2 = u.copy()
    u2 = u2[u2[:, 0] > 0]
    if len(u2) > 40:
        Xh = normalize_rows(x0 + u2 @ J.T)
        m = fit_metrics_on_points(Xh, d, int(rng.integers(1e9)), w_probe, 0.5)
        if m:
            rows.append({"synth": "flat_half_ball", **m})
    # curved: quadratic normal bump
    Hdir = rng.normal(size=D)
    Hdir = Hdir - J @ (J.T @ Hdir) - x0 * np.dot(x0, Hdir)
    Hdir /= max(np.linalg.norm(Hdir), EPS)
    Xc = normalize_rows(x0 + u @ J.T + 0.15 * ((u**2).sum(1, keepdims=True)) * Hdir)
    m = fit_metrics_on_points(Xc, d, int(rng.integers(1e9)), w_probe, 0.5)
    if m:
        rows.append({"synth": "genuinely_curved", **m})
    # heteroskedastic flat
    scale = 0.1 + 0.5 * np.linalg.norm(u, axis=1, keepdims=True)
    noise = rng.normal(size=(n, D))
    noise = noise - (noise @ J) @ J.T - np.outer(noise @ x0, x0)
    Xhet = normalize_rows(x0 + u @ J.T + scale * noise * 0.05)
    m = fit_metrics_on_points(Xhet, d, int(rng.integers(1e9)), w_probe, 0.5)
    if m:
        rows.append({"synth": "heteroskedastic_flat", **m})
    return rows


def stage4_nulls(
    root: Path,
    cfg: FalsificationConfig,
    X: np.ndarray,
    pack_map: dict,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
    w_probe: np.ndarray,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / "flat_null_results.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    rng = np.random.default_rng(cfg.seed + 21)
    real_rows = []
    null_rows = []
    for i, sid in enumerate(smoke_ids):
        p = pack_map.get((int(sid), cfg.primary_k))
        if p is None:
            continue
        z = np.load(p)
        neigh = z["neigh"]
        Xn = X[neigh]
        T, x0u = z["T"], z["x0u"]
        rho = float(z["rho"])
        d = T.shape[1]
        m_real = fit_metrics_on_points(Xn, d, cfg.seed + int(z["ai"]), w_probe, rho)
        if m_real:
            m_real.update({"sample_id": int(sid), "kind": "real", "replicate": -1})
            real_rows.append(m_real)
        for r in range(cfg.n_null):
            Xnull = make_null_neighborhood(Xn, x0u, T, rng)
            m = fit_metrics_on_points(Xnull, d, cfg.seed + 10007 * r + int(sid), w_probe, rho)
            if m:
                m.update({"sample_id": int(sid), "kind": "null", "replicate": r})
                null_rows.append(m)
        if (i + 1) % 16 == 0:
            print(f"[falsify] stage4 nulls {i+1}/{len(smoke_ids)}", flush=True)
    synth = synthetic_controls(LOCAL_DIM, X.shape[1], 400, rng, w_probe)
    for s in synth:
        s.update({"sample_id": -1, "kind": "synth", "replicate": -1})
    df = pd.DataFrame(real_rows + null_rows + synth)
    # MC: real mean vs null means
    summary = {}
    if real_rows and null_rows:
        rdf = pd.DataFrame(real_rows)
        ndf = pd.DataFrame(null_rows)
        for feat in ["K_traceless", "K_mean", "K_probe"]:
            real_mean = float(rdf[feat].mean())
            # null distribution: per-replicate mean across anchors
            null_means = ndf.groupby("replicate")[feat].mean().to_numpy()
            p, B = mc_p(real_mean, null_means, greater=True)
            summary[feat] = {"real_mean": real_mean, "null_mean": float(np.mean(null_means)), "p_mc": p, "B": B}
    (out / "stage4_summary.json").write_text(json.dumps(summary, indent=2))
    df.to_parquet(path, index=False)
    print(f"[falsify] stage4 real={len(real_rows)} null={len(null_rows)}", flush=True)
    return df


# -------------------- Stage 5: disjoint samples --------------------


def stage5_disjoint(
    root: Path,
    cfg: FalsificationConfig,
    X: np.ndarray,
    pack_map: dict,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
    w_probe: np.ndarray,
    b_probe: float,
    y: np.ndarray,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / "disjoint_sample_results.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    rows = []
    for swap in (0, 1):
        for sid in smoke_ids:
            p = pack_map.get((int(sid), cfg.primary_k))
            if p is None:
                continue
            z = np.load(p)
            neigh = np.asarray(z["neigh"])
            rng = np.random.default_rng(cfg.seed + int(sid) + 333 * swap)
            order = neigh.copy()
            rng.shuffle(order)
            n = len(order)
            n_g = n // 2
            geom_idx, eval_idx = order[:n_g], order[n_g:]
            if swap:
                geom_idx, eval_idx = eval_idx, geom_idx
            if len(geom_idx) < 40 or len(eval_idx) < 20:
                continue
            d = z["T"].shape[1]
            m = fit_metrics_on_points(X[geom_idx], d, cfg.seed + int(sid), w_probe, float(z["rho"]))
            if m is None:
                continue
            yhat = X @ w_probe + b_probe
            r2 = local_r2_fixed_predictions(y[eval_idx], yhat[eval_idx])
            # MSE / global var etc.
            yg = y[eval_idx]
            pred = yhat[eval_idx]
            mm = np.isfinite(yg) & np.isfinite(pred)
            if mm.sum() < 8:
                continue
            gvar = float(np.nanvar(y[np.isfinite(y)]))
            gstd = float(np.sqrt(max(gvar, EPS)))
            mse = float(np.mean((yg[mm] - pred[mm]) ** 2))
            mae = float(np.mean(np.abs(yg[mm] - pred[mm])))
            pear = float(spearmanr(yg[mm], pred[mm]).correlation)  # will overwrite with pearson
            pear = float(np.corrcoef(yg[mm], pred[mm])[0, 1])
            # calibration
            A = np.column_stack([np.ones(mm.sum()), pred[mm]])
            beta, *_ = np.linalg.lstsq(A, yg[mm], rcond=None)
            row_m = mag_g[mag_g.sample_id == sid].iloc[0]
            rows.append(
                {
                    "sample_id": int(sid),
                    "swap": swap,
                    "local_r2_disjoint": r2,
                    "mse_over_gvar": mse / max(gvar, EPS),
                    "mae_over_gstd": mae / gstd,
                    "pearson": pear,
                    "cal_intercept": float(beta[0]),
                    "cal_slope": float(beta[1]),
                    "K_mean": m["K_mean"],
                    "K_probe": m["K_probe"],
                    "K_traceless": m["K_traceless"],
                    "A_B_normal": m["A_B_normal"],
                    "log_knn_radius": float(row_m.log_knn_radius),
                    "local_target_variance": float(row_m.local_target_variance),
                    "reconstruction_error": float(row_m.reconstruction_error),
                    "local_evaluation_count": float(len(eval_idx)),
                    "A_N": float(row_m.A_N),
                    "A_T": float(row_m.A_T),
                    "A_PCA_normal": float(row_m.A_PCA_normal),
                }
            )
    df = pd.DataFrame(rows)
    # associations on swap=0
    summary = {}
    for swap, g in df.groupby("swap"):
        feats = {
            "K_mean": g.K_mean.to_numpy(float),
            "K_probe": g.K_probe.to_numpy(float),
            "A_B_normal": g.A_B_normal.to_numpy(float),
            "K_traceless": g.K_traceless.to_numpy(float),
        }
        summary[f"swap{swap}"] = assoc_block(
            g.local_r2_disjoint.to_numpy(float), feats, controls_C0(g), controls_C2(g), cfg
        )
    (out / "stage5_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    df.to_parquet(path, index=False)
    print(f"[falsify] stage5 n={len(df)}", flush=True)
    return df


# -------------------- Stage 6: spatial dependence --------------------


def stage6_spatial(
    root: Path,
    cfg: FalsificationConfig,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
    pack_map: dict,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / "spatial_block_results.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    g = mag_g[mag_g.sample_id.isin(smoke_ids)].copy()
    # overlap graph via Jaccard of neighbour sets
    neighs = {}
    for sid in smoke_ids:
        p = pack_map.get((int(sid), cfg.primary_k))
        if p is None:
            continue
        neighs[int(sid)] = set(np.load(p)["neigh"].tolist())
    ids = sorted(neighs)
    n = len(ids)
    # build edges if Jaccard > 0.05
    adj = {i: set() for i in ids}
    jaccs = []
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = ids[a], ids[b]
            inter = len(neighs[ia] & neighs[ib])
            union = len(neighs[ia] | neighs[ib])
            jac = inter / max(union, 1)
            if jac > 0.05:
                adj[ia].add(ib)
                adj[ib].add(ia)
                jaccs.append(jac)
    # greedy coloring → blocks of independent set? For block bootstrap: connected components of strong overlap
    # Use graph-block: connected components with jac>0.15
    adj2 = {i: set() for i in ids}
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = ids[a], ids[b]
            inter = len(neighs[ia] & neighs[ib])
            union = len(neighs[ia] | neighs[ib])
            if inter / max(union, 1) > 0.15:
                adj2[ia].add(ib)
                adj2[ib].add(ia)
    seen = set()
    blocks = []
    for i in ids:
        if i in seen:
            continue
        stack = [i]
        comp = []
        seen.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj2[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        blocks.append(comp)
    # block bootstrap of partial K_mean
    r2 = g.set_index("sample_id").local_r2
    km = g.set_index("sample_id").K_mean
    C0 = controls_C0(g.set_index("sample_id").loc[ids].reset_index() if False else g)
    # align to ids order
    gg = g.set_index("sample_id").loc[ids]
    r2v = gg.local_r2.to_numpy(float)
    kmv = gg.K_mean.to_numpy(float)
    kpv = gg.get("K_probe", gg.C_w).to_numpy(float) if "K_probe" in gg else gg.C_w.to_numpy(float)
    # use K_mean / A_B / interaction from mag
    abv = gg.A_B_normal.to_numpy(float)
    ktv = gg.K_traceless.to_numpy(float)
    C0v = controls_C0(gg.reset_index())
    C2v = controls_C2(gg.reset_index())
    real = {
        "partial_K_mean": partial_spearman(kmv, r2v, C0v)["rho"],
        "partial_A_B": partial_spearman(abv, r2v, C2v)["rho"],
        "partial_K_traceless": partial_spearman(ktv, r2v, C0v)["rho"],
    }
    from .global_probe_curvature_magnitude import rank_z

    yz, az, kz = rank_z(r2v), rank_z(abv), rank_z(ktv)
    m = np.isfinite(yz) & np.isfinite(az) & np.isfinite(kz)
    if m.sum() > 20:
        X = np.column_stack([np.ones(m.sum()), az[m], kz[m], az[m] * kz[m]])
        beta, *_ = np.linalg.lstsq(X, yz[m], rcond=None)
        real["coef_interaction"] = float(beta[3])
    else:
        real["coef_interaction"] = float("nan")

    rng = np.random.default_rng(cfg.seed + 44)
    nulls = {k: [] for k in real}
    id_to_pos = {sid: i for i, sid in enumerate(ids)}
    for _ in range(cfg.n_permute):
        # resample blocks with replacement, then permute R2 within concatenated
        chosen = [blocks[j] for j in rng.choice(len(blocks), size=len(blocks), replace=True)]
        boot_ids = [x for bl in chosen for x in bl]
        # truncate/pad to n
        if len(boot_ids) >= n:
            boot_ids = boot_ids[:n]
        else:
            boot_ids = boot_ids + list(rng.choice(ids, size=n - len(boot_ids), replace=True))
        pos = [id_to_pos[s] for s in boot_ids]
        r_b = r2v[pos]
        # permute R2 across blocks for null
        block_vals = []
        cursor = 0
        r_perm = r_b.copy()
        # simpler: permute all r2
        r_perm = rng.permutation(r2v)
        nulls["partial_K_mean"].append(partial_spearman(kmv, r_perm, C0v)["rho"])
        nulls["partial_A_B"].append(partial_spearman(abv, r_perm, C2v)["rho"])
        nulls["partial_K_traceless"].append(partial_spearman(ktv, r_perm, C0v)["rho"])
        yz = rank_z(r_perm)
        if m.sum() > 20:
            X = np.column_stack([np.ones(m.sum()), az[m], kz[m], az[m] * kz[m]])
            beta, *_ = np.linalg.lstsq(X, yz[m], rcond=None)
            nulls["coef_interaction"].append(float(beta[3]))
    # block bootstrap CI for real effects: resample blocks, recompute
    boots = {k: [] for k in real}
    for _ in range(min(400, cfg.n_permute)):
        chosen = [blocks[j] for j in rng.choice(len(blocks), size=len(blocks), replace=True)]
        boot_ids = [x for bl in chosen for x in bl][:n]
        if len(boot_ids) < 30:
            continue
        pos = [id_to_pos[s] for s in boot_ids if s in id_to_pos]
        if len(pos) < 30:
            continue
        boots["partial_K_mean"].append(partial_spearman(kmv[pos], r2v[pos], C0v[pos])["rho"])
        boots["partial_A_B"].append(partial_spearman(abv[pos], r2v[pos], C2v[pos])["rho"])
        boots["partial_K_traceless"].append(partial_spearman(ktv[pos], r2v[pos], C0v[pos])["rho"])
    rows = []
    for k, val in real.items():
        p, B = mc_p(val, np.asarray(nulls[k]))
        ci = (
            [float(np.quantile(boots[k], 0.025)), float(np.quantile(boots[k], 0.975))]
            if boots.get(k)
            else [float("nan"), float("nan")]
        )
        rows.append(
            {
                "statistic": k,
                "estimate": val,
                "p_block_perm": p,
                "B": B,
                "ci95_block_boot": ci,
                "n_blocks": len(blocks),
                "mean_jaccard_edges": float(np.mean(jaccs)) if jaccs else 0.0,
                "n_anchors": n,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"[falsify] stage6 blocks={len(blocks)} mean_jac={np.mean(jaccs) if jaccs else 0:.3f}", flush=True)
    return df


# -------------------- Stage 7: curvature vs NPCA --------------------


def stage7_specificity(
    root: Path,
    cfg: FalsificationConfig,
    X: np.ndarray,
    pack_map: dict,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
    w_probe: np.ndarray,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / "curvature_pca_decomposition.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    rows = []
    for sid in smoke_ids:
        p = pack_map.get((int(sid), cfg.primary_k))
        if p is None:
            continue
        z = np.load(p)
        neigh = z["neigh"]
        T, x0u, B0, UN = z["T"], z["x0u"], z["B0"], z["UNPCA"]
        d = T.shape[1]
        # H from rematerialize
        chart, *_rest = _fit_neighborhood(X, neigh, d, seed=cfg.seed + 17 * int(z["ai"]) + cfg.primary_k)
        if chart is None:
            continue
        B0, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
        d = chart.J.shape[1]
        T, x0u = chart.J, chart.x0
        dx = X[neigh] - x0u[None, :]
        r = B0_flat_for_svd(B0, d).shape[1]
        # matched rank NPCA
        UN = normal_pca_basis(dx, x0u, T, min(r, UN.shape[1] if UN is not None else r))
        # project each B0 slice
        # P = UN UN.T
        B_shared = np.zeros_like(B0)
        B_unique = np.zeros_like(B0)
        for a in range(d):
            for b in range(d):
                v = B0[:, a, b]
                sh = UN @ (UN.T @ v)
                B_shared[:, a, b] = sh
                B_unique[:, a, b] = v - sh
        wN = unit_normal_component(w_probe, T, x0u)
        k_sh = probe_facing(B_shared, H * 0, wN, d)["K_probe"]
        k_un = probe_facing(B_unique, H * 0, wN, d)["K_probe"]
        k_full = probe_facing(B0, H, wN, d)["K_probe"]
        # principal angles between UB and UN
        Bflat = B0_flat_for_svd(B0, d)
        Ub, s, _ = np.linalg.svd(Bflat, full_matrices=False)
        keep = s > 1e-8 * (s[0] if len(s) else 1.0)
        UB = Ub[:, keep] if np.any(keep) else Ub
        # overlap
        M = UB.T @ UN
        svals = np.linalg.svd(M, compute_uv=False)
        overlap = float(np.mean(svals**2)) if len(svals) else float("nan")
        mean_angle = float(np.mean(np.arccos(np.clip(svals, 0, 1)))) if len(svals) else float("nan")
        row_m = mag_g[mag_g.sample_id == sid].iloc[0]
        rows.append(
            {
                "sample_id": int(sid),
                "K_probe": k_full,
                "K_probe_shared": k_sh,
                "K_probe_unique": k_un,
                "subspace_overlap": overlap,
                "mean_principal_angle": mean_angle,
                "local_r2": float(row_m.local_r2),
                "log_knn_radius": float(row_m.log_knn_radius),
                "local_target_variance": float(row_m.local_target_variance),
                "reconstruction_error": float(row_m.reconstruction_error),
                "local_evaluation_count": float(row_m.local_evaluation_count),
                "A_N": float(row_m.A_N),
                "A_T": float(row_m.A_T),
                "A_PCA_normal": float(row_m.A_PCA_normal),
                "K_mean": float(row_m.K_mean),
            }
        )
    df = pd.DataFrame(rows)
    feats = {
        "K_probe": df.K_probe.to_numpy(float),
        "K_probe_shared": df.K_probe_shared.to_numpy(float),
        "K_probe_unique": df.K_probe_unique.to_numpy(float),
    }
    summary = assoc_block(df.local_r2.to_numpy(float), feats, controls_C0(df), controls_C2(df), cfg)
    summary["mean_overlap"] = float(df.subspace_overlap.mean())
    (out / "stage7_summary.json").write_text(json.dumps(summary, indent=2))
    df.to_parquet(path, index=False)
    print(f"[falsify] stage7 n={len(df)} unique_partial={summary.get('partial_K_probe_unique')}", flush=True)
    return df


# -------------------- Stage 8: performance metrics --------------------


def stage8_metrics(
    root: Path,
    cfg: FalsificationConfig,
    X: np.ndarray,
    pack_map: dict,
    mag_g: pd.DataFrame,
    smoke_ids: np.ndarray,
    w_probe: np.ndarray,
    b_probe: float,
    y: np.ndarray,
) -> pd.DataFrame:
    out = cfg.resolved_out(root)
    path = out / "performance_metric_robustness.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    gvar = float(np.nanvar(y[np.isfinite(y)]))
    gstd = float(np.sqrt(max(gvar, EPS)))
    yhat = X @ w_probe + b_probe
    rows = []
    for sid in smoke_ids:
        p = pack_map.get((int(sid), cfg.primary_k))
        if p is None:
            continue
        z = np.load(p)
        neigh = z["neigh"]
        yg, pred = y[neigh], yhat[neigh]
        m = np.isfinite(yg) & np.isfinite(pred)
        if m.sum() < 8:
            continue
        r2 = local_r2_fixed_predictions(yg, pred)
        mse = float(np.mean((yg[m] - pred[m]) ** 2)) / max(gvar, EPS)
        mae = float(np.mean(np.abs(yg[m] - pred[m]))) / gstd
        pear = float(np.corrcoef(yg[m], pred[m])[0, 1])
        A = np.column_stack([np.ones(m.sum()), pred[m]])
        beta, *_ = np.linalg.lstsq(A, yg[m], rcond=None)
        row_m = mag_g[mag_g.sample_id == sid].iloc[0]
        rows.append(
            {
                "sample_id": int(sid),
                "local_r2": r2,
                "mse_over_gvar": mse,
                "mae_over_gstd": mae,
                "pearson": pear,
                "cal_slope": float(beta[1]),
                "cal_intercept": float(beta[0]),
                "K_mean": float(row_m.K_mean),
                "K_traceless": float(row_m.K_traceless),
                "A_B_normal": float(row_m.A_B_normal),
                "C_w": float(row_m.C_w),
                "log_knn_radius": float(row_m.log_knn_radius),
                "local_target_variance": float(row_m.local_target_variance),
                "reconstruction_error": float(row_m.reconstruction_error),
                "local_evaluation_count": float(row_m.local_evaluation_count),
                "A_N": float(row_m.A_N),
                "A_T": float(row_m.A_T),
                "A_PCA_normal": float(row_m.A_PCA_normal),
            }
        )
    df = pd.DataFrame(rows)
    summary = {}
    for metric in ["local_r2", "mse_over_gvar", "mae_over_gstd", "pearson", "cal_slope"]:
        # for error metrics, association sign flips expected
        yv = df[metric].to_numpy(float)
        if metric.startswith("mse") or metric.startswith("mae"):
            yv = -yv  # higher error = worse; align so positive means better performance association
        feats = {
            "K_mean": df.K_mean.to_numpy(float),
            "A_B_normal": df.A_B_normal.to_numpy(float),
            "K_traceless": df.K_traceless.to_numpy(float),
        }
        summary[metric] = assoc_block(yv, feats, controls_C0(df), controls_C2(df), cfg)
    (out / "stage8_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    df.to_parquet(path, index=False)
    print(f"[falsify] stage8 n={len(df)}", flush=True)
    return df


# -------------------- verdict / report --------------------


def decide_verdicts(out: Path) -> list[str]:
    labels: list[str] = []
    s1 = json.loads((out / "stage1_summary.json").read_text()) if (out / "stage1_summary.json").exists() else {}
    if s1.get("density_ridge_language"):
        labels.append("density_ridge_not_thin_manifold")
    gate = json.loads((out / "stage2_gate.json").read_text()) if (out / "stage2_gate.json").exists() else {}
    if gate and not gate.get("pass", True):
        labels.append("tangent_underfit")
    s3 = json.loads((out / "stage3_explanations.json").read_text()) if (out / "stage3_explanations.json").exists() else {}
    if abs(s3.get("partial_K_mean_with_sparsity", 1)) < 0.08 and abs(s3.get("rho_local_r2__near_zero_fraction", 0)) > 0.2:
        labels.append("coordinate_sparsity_artifact")
    if abs(s3.get("rho_K_mean__knn_radius", 0)) > 0.7:
        labels.append("density_or_radius_artifact")
    if abs(s3.get("rho_local_r2__tangent_directional_imbalance", 0)) > 0.25:
        labels.append("boundary_or_stratification_effect")
    s4 = json.loads((out / "stage4_summary.json").read_text()) if (out / "stage4_summary.json").exists() else {}
    s5 = json.loads((out / "stage5_summary.json").read_text()) if (out / "stage5_summary.json").exists() else {}
    if s5:
        p0 = s5.get("swap0", {})
        if abs(p0.get("partial_K_mean", 1)) < 0.08 and abs(p0.get("coef_interaction", 1)) < 0.05:
            labels.append("shared_sample_artifact")
    s6 = pd.read_parquet(out / "spatial_block_results.parquet") if (out / "spatial_block_results.parquet").exists() else None
    if s6 is not None and len(s6):
        n_blocks = int(s6.iloc[0].get("n_blocks", 0))
        if n_blocks <= 2:
            labels.append("spatial_dependence_underpowered")
        else:
            weak = s6[s6.statistic.isin(["partial_K_mean", "coef_interaction"])]
            if len(weak) and (weak.p_block_perm > 0.1).all():
                labels.append("spatial_dependence_underpowered")
    s7 = json.loads((out / "stage7_summary.json").read_text()) if (out / "stage7_summary.json").exists() else {}
    if abs(s7.get("partial_K_probe_unique", 1)) < 0.08 and abs(s7.get("partial_K_probe_shared", 0)) >= 0.12:
        labels.append("generic_normal_frame_effect")
    s8 = json.loads((out / "stage8_summary.json").read_text()) if (out / "stage8_summary.json").exists() else {}
    if s8:
        r2_km = abs(s8.get("local_r2", {}).get("partial_K_mean", 0))
        other_ok = any(
            abs(s8.get(m, {}).get("partial_K_mean", 0)) >= 0.1
            for m in ["pearson", "mse_over_gvar", "mae_over_gstd"]
        )
        if r2_km >= 0.12 and not other_ok:
            labels.append("performance_metric_specific")

    blockers = {
        "tangent_underfit",
        "shared_sample_artifact",
        "coordinate_sparsity_artifact",
        "performance_metric_specific",
        "generic_normal_frame_effect",
        "spatial_dependence_underpowered",
    }
    blocked = bool(blockers & set(labels))
    # K_mean can remain a geographic marker even if interaction fails dim-sweep
    km_ok = s4.get("K_mean", {}).get("p_mc", 1) <= 0.05
    if s8:
        km_ok = km_ok and abs(s8.get("local_r2", {}).get("partial_K_mean", 0)) >= 0.12
    if km_ok and not blocked:
        labels.insert(0, "survives_primary_falsification")
    elif not labels:
        labels.append("inconclusive")

    seen: set[str] = set()
    out_labs = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            out_labs.append(lab)
    return out_labs


# -------------------- main --------------------


def run_falsification(cfg: FalsificationConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    profile: dict[str, Any] = {"stages": {}}

    print(f"[falsify] train_fn={CANONICAL_PROBE_TRAIN}", flush=True)
    print(f"[falsify] local_fn={CANONICAL_PROBE_LOCAL}", flush=True)

    mag_dir = resolve_path(root, cfg.mag_dir)
    align_dir = resolve_path(root, cfg.align_dir)
    mag = pd.read_parquet(mag_dir / "anchor_target_curvature_metrics.parquet")
    mag_g = mag[(mag.target == cfg.primary_target) & (mag.scale_k == cfg.primary_k)].copy()
    if "config_hash" in mag_g.columns and set(mag_g.config_hash.astype(str).unique()) != {cfg.expected_hash}:
        raise RuntimeError("hash mismatch")
    scfg = ScreenConfig(curvature_path=cfg.curvature_path, expected_hash=cfg.expected_hash)
    _ = load_frozen_curvature(root, scfg)

    data = load_prepare(resolve_path(root, cfg.prepare_dir))
    X = data["X"].astype(np.float64)
    train = np.asarray(data["train_local"])
    sample_ids = data["sample_ids"]
    lab = np.load(resolve_path(root, cfg.labels_path))
    y = np.asarray(lab[cfg.primary_target], dtype=np.float64)[sample_ids]

    wz = np.load(align_dir / "global_probe_weights.npz")
    w_probe = np.asarray(wz[f"w_{cfg.primary_target}"], dtype=np.float64)
    b_probe = float(np.asarray(wz[f"b_{cfg.primary_target}"]).ravel()[0])

    cache = resolve_path(root, cfg.geometry_cache)
    pack_map = {}
    for p in cache.glob(f"k{cfg.primary_k}_ai*.npz"):
        try:
            z = np.load(p)
            pack_map[(int(z["sample_id"]), int(z["scale_k"]))] = p
        except Exception:
            continue

    smoke_path = out / "smoke_anchor_ids.json"
    if _done(smoke_path, cfg.force):
        smoke_ids = np.asarray(json.loads(smoke_path.read_text())["sample_ids"], dtype=np.int64)
    else:
        smoke_ids = select_smoke_anchors(mag_g, cfg.smoke_anchors, cfg.seed)
        smoke_path.write_text(json.dumps({"sample_ids": smoke_ids.tolist(), "n": len(smoke_ids)}, indent=2))
    print(f"[falsify] smoke anchors n={len(smoke_ids)}", flush=True)

    want = set(cfg.stages.split(",")) if cfg.stages != "all" else {str(i) for i in range(1, 9)}

    if "1" in want:
        t1 = time.time()
        stage1_audit(root, cfg, X, pack_map, mag_g, smoke_ids, w_probe)
        profile["stages"]["stage1_s"] = time.time() - t1

    if "2" in want:
        t1 = time.time()
        # runtime estimate before full confirm
        t_est0 = time.time()
        _ = stage2_dimension_sweep(
            root, cfg, X, pack_map, mag_g, smoke_ids[:8], w_probe, (8,), "timing8"
        )
        per_anchor = (time.time() - t_est0) / 8
        est_smoke = per_anchor * len(smoke_ids) * len(DIMS_SMOKE)
        est_full = per_anchor * 384 * len(DIMS_CONFIRM)
        (out / "stage2_runtime_estimate.json").write_text(
            json.dumps(
                {
                    "sec_per_anchor_d8": per_anchor,
                    "est_smoke_all_dims_s": est_smoke,
                    "est_confirm_384_s": est_full,
                },
                indent=2,
            )
        )
        print(f"[falsify] stage2 estimate smoke≈{est_smoke/60:.1f}min confirm≈{est_full/60:.1f}min", flush=True)
        dim_df = stage2_dimension_sweep(
            root, cfg, X, pack_map, mag_g, smoke_ids, w_probe, DIMS_SMOKE, "smoke"
        )
        assoc = pd.read_parquet(out / "dimension_sensitivity.parquet")
        gate = stage2_smoke_gate(assoc)
        (out / "stage2_gate.json").write_text(json.dumps(gate, indent=2))
        print(f"[falsify] stage2 gate pass={gate.get('pass')} flips={gate.get('flips')}", flush=True)
        if cfg.confirm_full and gate.get("pass"):
            stage2_dimension_sweep(
                root, cfg, X, pack_map, mag_g, mag_g.sample_id.unique(), w_probe, DIMS_CONFIRM, "confirm384"
            )
            # merge assoc
            ap = out / "dimension_sensitivity_confirm384_assoc.parquet"
            if ap.exists():
                assoc = pd.concat([assoc, pd.read_parquet(ap)], ignore_index=True)
                assoc.to_parquet(out / "dimension_sensitivity.parquet", index=False)
        profile["stages"]["stage2_s"] = time.time() - t1

    if "3" in want:
        t1 = time.time()
        stage3_sparsity(root, cfg, X, train, pack_map, mag_g, smoke_ids)
        profile["stages"]["stage3_s"] = time.time() - t1

    if "4" in want:
        t1 = time.time()
        stage4_nulls(root, cfg, X, pack_map, mag_g, smoke_ids, w_probe)
        profile["stages"]["stage4_s"] = time.time() - t1

    if "5" in want:
        t1 = time.time()
        stage5_disjoint(root, cfg, X, pack_map, mag_g, smoke_ids, w_probe, b_probe, y)
        profile["stages"]["stage5_s"] = time.time() - t1

    if "6" in want:
        t1 = time.time()
        stage6_spatial(root, cfg, mag_g, smoke_ids, pack_map)
        profile["stages"]["stage6_s"] = time.time() - t1

    if "7" in want:
        t1 = time.time()
        stage7_specificity(root, cfg, X, pack_map, mag_g, smoke_ids, w_probe)
        profile["stages"]["stage7_s"] = time.time() - t1

    if "8" in want:
        t1 = time.time()
        stage8_metrics(root, cfg, X, pack_map, mag_g, smoke_ids, w_probe, b_probe, y)
        profile["stages"]["stage8_s"] = time.time() - t1

    # secondary photo_z only after primary stages
    if cfg.stages == "all":
        mag_z = mag[(mag.target == cfg.secondary_target) & (mag.scale_k == cfg.primary_k)].copy()
        if len(mag_z):
            wz_p = np.asarray(wz[f"w_{cfg.secondary_target}"], dtype=np.float64)
            # quick assoc replication on smoke using cached mag features
            g = mag_z[mag_z.sample_id.isin(smoke_ids)]
            feats = {
                "K_mean": g.K_mean.to_numpy(float),
                "A_B_normal": g.A_B_normal.to_numpy(float),
                "K_traceless": g.K_traceless.to_numpy(float),
            }
            sec = assoc_block(g.local_r2.to_numpy(float), feats, controls_C0(g), controls_C2(g), cfg)
            (out / "secondary_photo_z_summary.json").write_text(json.dumps(sec, indent=2, default=str))

    labels = decide_verdicts(out)
    pd.DataFrame([{"label": lab} for lab in labels]).to_csv(out / "falsification_summary.csv", index=False)

    profile.update(
        {
            "total_seconds": time.time() - t0,
            "peak_rss_mb": _rss(),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0.0,
            "n_smoke": int(len(smoke_ids)),
            "labels": labels,
            "canonical_train_fn": CANONICAL_PROBE_TRAIN,
            "canonical_local_fn": CANONICAL_PROBE_LOCAL,
            "config_hash": cfg.expected_hash,
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    # REPORT
    def _loadj(name):
        fp = out / name
        return json.loads(fp.read_text()) if fp.exists() else {}

    s1 = _loadj("stage1_summary.json")
    gate = _loadj("stage2_gate.json")
    s3 = _loadj("stage3_explanations.json")
    s4 = _loadj("stage4_summary.json")
    s5 = _loadj("stage5_summary.json")
    s7 = _loadj("stage7_summary.json")
    s8 = _loadj("stage8_summary.json")
    s6 = ""
    if (out / "spatial_block_results.parquet").exists():
        s6 = pd.read_parquet(out / "spatial_block_results.parquet").to_string(index=False)
    dim = ""
    if (out / "dimension_sensitivity.parquet").exists():
        dim = pd.read_parquet(out / "dimension_sensitivity.parquet").to_string(index=False)

    headline = labels[0] if labels else "inconclusive"
    report = f"""# Global-probe curvature falsification

Frozen hash `{cfg.expected_hash}`. Primary cell: `{cfg.primary_target}`, k={cfg.primary_k}, α={cfg.probe_alpha}.
Smoke anchors: {len(smoke_ids)}. Protocol: fixed global probes only.

## Canonical functions

- train: `{CANONICAL_PROBE_TRAIN}`
- local score: `{CANONICAL_PROBE_LOCAL}`

## Verdict labels

{chr(10).join(f"- `{lab}`" for lab in labels)}

**Headline:** `{headline}`

## Stage 1 — geometry audit

- median normal residual variance fraction: {s1.get("median_normal_var_frac")}
- density-ridge language (thick): {s1.get("density_ridge_language")}
- H / B° orthogonality OK fractions: {s1.get("frac_H_orth_ok")} / {s1.get("frac_B0_orth_ok")}
- median n_eff / q: {s1.get("median_neff_over_q")}

## Stage 2 — tangent dimension

Gate pass={gate.get("pass")} flips={gate.get("flips")}
{dim}

## Stage 3 — sparsity / boundary

{json.dumps(s3, indent=2)[:2000]}

## Stage 4 — flat matched nulls

{json.dumps(s4, indent=2)}

## Stage 5 — disjoint geometry/scoring

{json.dumps(s5, indent=2, default=str)[:2500]}

## Stage 6 — spatial blocks

{s6}

## Stage 7 — unique vs shared curvature

{json.dumps(s7, indent=2)}

## Stage 8 — performance-metric robustness

{json.dumps(s8, indent=2, default=str)[:2500]}

## Answers

1. Which tests passed/failed: see stage summaries and labels.
2. d=8 underfit: {"yes" if "tangent_underfit" in labels else "no / not indicated"}.
3. Density/sparsity/boundaries: see stage 3 and labels.
4. Disjoint samples: see stage 5; shared_sample_artifact={"yes" if "shared_sample_artifact" in labels else "no"}.
5. Spatially blocked inference: see stage 6.
6. Probe-facing curvature: see K_probe associations in stages 2/7.
7. Unique beyond NPCA: {"no — generic_normal_frame_effect" if "generic_normal_frame_effect" in labels else "see stage 7 partial_K_probe_unique"}.
8. Alternative metrics: {"R²-specific" if "performance_metric_specific" in labels else "see stage 8"}.
9. Manifold terminology: {"prefer density-ridge language" if s1.get("density_ridge_language") else "thin-manifold language not ruled out by thickness alone"}.
10. Strongest defensible headline: `{headline}`.

## Runtime

{profile["total_seconds"]:.1f}s; peak RSS={profile["peak_rss_mb"]:.1f} MB.

## Exact command

```bash
cd ~/platonic-universe && source .venv/bin/activate && \
PYTHONPATH=experiments python -m geometry.run_global_probe_curvature_falsification \
  --force --seed 0
```
"""
    (out / "REPORT.md").write_text(report)
    analysis = {
        "labels": labels,
        "headline": headline,
        "stage1": s1,
        "stage2_gate": gate,
        "runtime": profile,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"[falsify] done in {profile['total_seconds']:.1f}s labels={labels}", flush=True)
    return analysis
