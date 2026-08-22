"""Global-probe curvature magnitude × alignment analysis.

Preserves the corrected UniverseTBD protocol:
  fit_global_probe + weighted_r2; α=100 from select_ridge_alpha.
Never refits probes at anchors. Primarily joins cached alignment rows with
frozen sphere-normal magnitude features; rematerializes H^S only for A_H.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import LeaveOneGroupOut

from .confirmatory_object_curvature import _fit_neighborhood
from .curvature_probe_alignment import traceless_B0
from .curvature_probe_screen import (
    EXPECTED_HASH,
    LOCAL_DIM,
    ScreenConfig,
    load_frozen_curvature,
    partial_spearman,
    spearman_dict,
)
from .data import load_prepare
from .global_probe_curvature_alignment import (
    CANONICAL_PROBE_LOCAL,
    CANONICAL_PROBE_TRAIN,
    PROBE_ALPHA,
    GlobalProbeAlignConfig,
    build_target_inventory,
    fit_global_probe,
)
from .paths import platonic_root, resolve_path

EPS = 1e-12
PRIMARY_TARGET = "mag_r_desi"
SCALES = (1024, 2048)
H_UNSTABLE_FRAC = 1e-4  # ||H|| / ||B°|| threshold


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


@dataclass
class MagnitudeConfig:
    output_dir: str = "outputs/geometry/physics_global_probe_curvature_magnitude"
    align_dir: str = "outputs/geometry/physics_global_probe_curvature_alignment"
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
    scales: list[int] = field(default_factory=lambda: list(SCALES))
    primary_k: int = 2048
    primary_target: str = PRIMARY_TARGET
    probe_alpha: float = PROBE_ALPHA
    n_bootstrap: int = 1000
    n_permute: int = 500
    seed: int = 0
    force: bool = False
    h_unstable_rel: float = H_UNSTABLE_FRAC

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


# -------------------- helpers --------------------


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.ones(n, dtype=float)
    prev = 1.0
    for i, idx in enumerate(order[::-1]):
        rank = n - i
        val = min(prev, p[idx] * n / rank) if np.isfinite(p[idx]) else float("nan")
        adj[idx] = val
        if np.isfinite(val):
            prev = val
    return adj


def mc_p_twosided(real, nulls):
    nulls = np.asarray(nulls, float)
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0 or not np.isfinite(real):
        return float("nan"), 0
    return float((1 + np.sum(np.abs(nulls) >= abs(real))) / (len(nulls) + 1)), len(nulls)


def mc_p_greater(real, nulls):
    nulls = np.asarray(nulls, float)
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0 or not np.isfinite(real):
        return float("nan"), 0
    return float((1 + np.sum(nulls >= real)) / (len(nulls) + 1)), len(nulls)


def fisher_meta(rhos: np.ndarray, ns: np.ndarray) -> dict:
    m = np.isfinite(rhos) & np.isfinite(ns) & (ns > 3)
    if m.sum() == 0:
        return {"rho": float("nan"), "n_targets": 0}
    z = np.arctanh(np.clip(rhos[m], -0.999, 0.999))
    w = ns[m] - 3.0
    return {"rho": float(np.tanh(np.sum(w * z) / max(w.sum(), EPS))), "n_targets": int(m.sum())}


def rank_z(x: np.ndarray) -> np.ndarray:
    m = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if m.sum() < 2:
        return out
    r = rankdata(x[m]).astype(float)
    r = (r - r.mean()) / (r.std() + EPS)
    out[m] = r
    return out


def a_h_from_w_H(w: np.ndarray, T: np.ndarray, x0u: np.ndarray, H: np.ndarray) -> tuple[float, bool]:
    """A_H = |H·w_N|² / ((||H||²+eps)(||w_N||²+eps)); flag unstable if ||H|| tiny."""
    hn = float(np.linalg.norm(H))
    unstable = hn < EPS
    wn = float(np.linalg.norm(w))
    if wn < EPS or unstable:
        return float("nan"), True
    w_hat = w / wn
    x0 = x0u / max(np.linalg.norm(x0u), EPS)
    w_T = T @ (T.T @ w_hat)
    w_R = x0 * float(np.dot(x0, w_hat))
    w_N = w_hat - w_T - w_R
    nN = float(np.linalg.norm(w_N))
    if nN < EPS:
        return float("nan"), unstable
    a = (float(np.dot(H, w_N)) ** 2) / ((hn**2 + EPS) * (nN**2 + EPS))
    return float(a), unstable


def rematerialize_H(X: np.ndarray, neigh: np.ndarray, ai: int, k: int, seed: int) -> np.ndarray | None:
    seed_fit = seed + 17 * ai + k
    chart, *_rest = _fit_neighborhood(X, neigh, LOCAL_DIM, seed=seed_fit)
    if chart is None:
        return None
    _B0, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
    return H.astype(np.float64)


# -------------------- build metrics table --------------------


def build_metrics_table(
    root: Path,
    cfg: MagnitudeConfig,
    align_df: pd.DataFrame,
    curv: pd.DataFrame,
    X: np.ndarray,
    y_all: dict[str, np.ndarray],
    sample_ids: np.ndarray,
    probes: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Join cached alignment + frozen magnitude; add A_H and local_evaluation_count."""
    out = cfg.resolved_out(root)
    metrics_path = out / "anchor_target_curvature_metrics.parquet"
    if _done(metrics_path, cfg.force):
        return pd.read_parquet(metrics_path)

    d = LOCAL_DIM
    curv_use = curv[curv.scale_k.isin(cfg.scales)].copy()
    if set(curv_use.config_hash.astype(str).unique()) != {cfg.expected_hash}:
        raise RuntimeError(f"hash mismatch: {curv_use.config_hash.unique()}")

    mag_cols = {
        "sample_id": "sample_id",
        "scale_k": "scale_k",
        "B_traceless_fro": "K_traceless",
        "H_norm": "K_mean",
        "B_fro": "B_fro_frozen",
        "mean_frac": "mean_fraction",
        "rho_times_B_traceless_fro": "C_traceless",
        "rho_times_H_norm": "C_mean",
        "stable_rank": "stable_rank",
        "rank95": "rank95",
        "knn_radius": "knn_radius_frozen",
        "reconstruction_error": "reconstruction_error_frozen",
        "n_eff": "n_eff_frozen",
    }
    keep = [c for c in mag_cols if c in curv_use.columns]
    mag = curv_use[keep].rename(columns={k: mag_cols[k] for k in keep})
    mag["K_total"] = np.sqrt(
        np.maximum(mag["K_traceless"].to_numpy(float), 0) ** 2
        + d * np.maximum(mag["K_mean"].to_numpy(float), 0) ** 2
    )
    # recompute C from align knn_radius after merge for consistency with local score radius
    df = align_df.merge(mag, on=["sample_id", "scale_k"], how="left", suffixes=("", "_mag"))
    # prefer align reconstruction_error; fill from frozen if needed
    if "reconstruction_error" in df.columns:
        df["reconstruction_error"] = df["reconstruction_error"].fillna(df.get("reconstruction_error_frozen"))
    else:
        df["reconstruction_error"] = df["reconstruction_error_frozen"]
    rho = df["knn_radius"].to_numpy(float)
    df["C_traceless"] = rho * df["K_traceless"].to_numpy(float)
    df["C_mean"] = rho * df["K_mean"].to_numpy(float)
    df["H_unstable"] = (
        df["K_mean"].to_numpy(float)
        < cfg.h_unstable_rel * np.maximum(df["K_traceless"].to_numpy(float), EPS)
    )

    # local_evaluation_count + A_H from geometry packs / rematerialized H
    cache = resolve_path(root, cfg.geometry_cache)
    h_dir = out / "H_cache"
    h_dir.mkdir(exist_ok=True)
    sid_to_local = {int(s): i for i, s in enumerate(np.asarray(sample_ids))}

    # map (sid,k) -> pack path
    pack_map: dict[tuple[int, int], Path] = {}
    for k in cfg.scales:
        for p in cache.glob(f"k{k}_ai*.npz"):
            try:
                z = np.load(p)
                pack_map[(int(z["sample_id"]), int(z["scale_k"]))] = p
            except Exception:
                continue

    targets = sorted(df.target.unique())
    mag_by_sk = (
        df.drop_duplicates(["sample_id", "scale_k"])
        .set_index(["sample_id", "scale_k"])[["K_mean", "K_traceless"]]
        .to_dict(orient="index")
    )
    eval_counts: dict[tuple[int, int, str], int] = {}
    a_h_rows: dict[tuple[int, int, str], tuple[float, bool]] = {}
    t0 = time.time()
    n_packs = 0
    for (sid, k), p in pack_map.items():
        h_path = h_dir / f"sid{sid}_k{k}.npy"
        z = np.load(p)
        neigh = z["neigh"]
        T, x0u = z["T"], z["x0u"]
        ai = int(z["ai"])
        if h_path.exists():
            H = np.load(h_path)
        else:
            H = rematerialize_H(X, neigh, ai, k, cfg.seed)
            if H is None:
                H = np.full(X.shape[1], np.nan)
            else:
                np.save(h_path, H)
        meta = mag_by_sk.get((sid, k), {})
        kn = float(meta.get("K_mean", float("nan")))
        kt = float(meta.get("K_traceless", float("nan")))
        mag_uns = np.isfinite(kn) and np.isfinite(kt) and kn < cfg.h_unstable_rel * max(kt, EPS)
        for tname in targets:
            y = y_all[tname][neigh]
            eval_counts[(sid, k, tname)] = int(np.isfinite(y).sum())
            ah, uns = a_h_from_w_H(probes[tname], T, x0u, H)
            a_h_rows[(sid, k, tname)] = (ah, bool(uns or mag_uns))
        n_packs += 1
        if n_packs % 64 == 0:
            print(f"[mag] A_H packs {n_packs}/{len(pack_map)} rss={_rss():.0f}", flush=True)

    df["local_evaluation_count"] = [
        eval_counts.get((int(r.sample_id), int(r.scale_k), r.target), 0) for r in df.itertuples()
    ]
    df["A_H"] = [
        a_h_rows.get((int(r.sample_id), int(r.scale_k), r.target), (float("nan"), True))[0]
        for r in df.itertuples()
    ]
    df["H_unstable"] = [
        a_h_rows.get((int(r.sample_id), int(r.scale_k), r.target), (float("nan"), True))[1]
        for r in df.itertuples()
    ]
    print(f"[mag] metrics table n={len(df)} built in {time.time()-t0:.1f}s", flush=True)
    df.to_parquet(metrics_path, index=False)
    return df


# -------------------- correlations --------------------


def correlate_target_scale(g: pd.DataFrame, target: str, k: int, cfg: MagnitudeConfig) -> dict:
    r2 = g.local_r2.to_numpy(float)
    feats = {
        "A_B_normal": g.A_B_normal.to_numpy(float),
        "A_B_total": g.A_B_total.to_numpy(float),
        "K_traceless": g.K_traceless.to_numpy(float),
        "K_mean": g.K_mean.to_numpy(float),
        "C_traceless": g.C_traceless.to_numpy(float),
        "C_mean": g.C_mean.to_numpy(float),
        "C_w": g.C_w.to_numpy(float),
        "A_PCA_normal": g.A_PCA_normal.to_numpy(float),
        "A_H": g.A_H.to_numpy(float),
        "mean_fraction": g.mean_fraction.to_numpy(float),
    }
    log_r = g.log_knn_radius.to_numpy(float)
    labvar = g.local_target_variance.to_numpy(float)
    recon = g.reconstruction_error.to_numpy(float)
    n_eval = g.local_evaluation_count.to_numpy(float)
    an = g.A_N.to_numpy(float)
    at = g.A_T.to_numpy(float)
    apn = g.A_PCA_normal.to_numpy(float)

    C0 = np.column_stack([log_r, labvar, recon, n_eval])
    C1 = np.column_stack([C0, an, at])
    C2 = np.column_stack([C1, apn])

    row: dict[str, Any] = {
        "target": target,
        "scale_k": k,
        "n": int(np.isfinite(r2).sum()),
        "mean_local_r2": float(np.nanmean(r2)),
        "mean_A_N": float(np.nanmean(an)),
        "mean_A_B_normal": float(np.nanmean(feats["A_B_normal"])),
        "mean_K_traceless": float(np.nanmean(feats["K_traceless"])),
        "mean_K_mean": float(np.nanmean(feats["K_mean"])),
        "frac_H_unstable": float(np.nanmean(g.H_unstable.to_numpy(float))),
        "corr_radius_C_traceless": spearman_dict(log_r, feats["C_traceless"])["rho"],
    }
    for name, x in feats.items():
        st = spearman_dict(r2, x)
        row[f"raw_{name}"] = st["rho"]
        row[f"p_raw_{name}"] = st["pvalue"]

    # primary partials
    part_ab = partial_spearman(feats["A_B_normal"], r2, C2)
    part_k = partial_spearman(feats["K_traceless"], r2, C0)
    part_hm = partial_spearman(feats["K_mean"], r2, C0)
    part_ab_c0 = partial_spearman(feats["A_B_normal"], r2, C0)
    part_ab_c1 = partial_spearman(feats["A_B_normal"], r2, C1)
    part_ct = partial_spearman(feats["C_traceless"], r2, C0)
    part_pca = partial_spearman(feats["A_PCA_normal"], r2, np.column_stack([C1, feats["A_B_normal"]]))
    # A_H only on stable anchors
    stable = ~g.H_unstable.to_numpy(bool)
    if stable.sum() >= 30:
        part_ah = partial_spearman(feats["A_H"][stable], r2[stable], C0[stable])
    else:
        part_ah = {"rho": float("nan"), "pvalue": float("nan"), "n": int(stable.sum())}

    row.update(
        {
            "partial_A_B_C0": part_ab_c0["rho"],
            "partial_A_B_C1": part_ab_c1["rho"],
            "partial_A_B_C2": part_ab["rho"],
            "p_partial_A_B_C2": part_ab["pvalue"],
            "partial_K_traceless_C0": part_k["rho"],
            "p_partial_K_traceless_C0": part_k["pvalue"],
            "partial_K_mean_C0": part_hm["rho"],
            "p_partial_K_mean_C0": part_hm["pvalue"],
            "partial_C_traceless_C0": part_ct["rho"],
            "partial_A_H_C0_stable": part_ah["rho"],
            "partial_PCA_given_AB": part_pca["rho"],
        }
    )

    # permutation p for primary statistics
    rng = np.random.default_rng(cfg.seed + 17 * k + (hash(target) % 997))
    null_ab, null_k, null_hm = [], [], []
    for _ in range(cfg.n_permute):
        rp = r2.copy()
        m = np.isfinite(rp)
        rp[m] = rng.permutation(rp[m])
        null_ab.append(partial_spearman(feats["A_B_normal"], rp, C2)["rho"])
        null_k.append(partial_spearman(feats["K_traceless"], rp, C0)["rho"])
        null_hm.append(partial_spearman(feats["K_mean"], rp, C0)["rho"])
    row["p_perm_A_B_C2"], row["B_perm"] = mc_p_twosided(part_ab["rho"], null_ab)
    row["p_perm_K_traceless_C0"], _ = mc_p_twosided(part_k["rho"], null_k)
    row["p_perm_K_mean_C0"], _ = mc_p_twosided(part_hm["rho"], null_hm)
    return row


# -------------------- interaction models --------------------


def _design(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df.local_r2.to_numpy(float)
    X = df[cols].to_numpy(float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[m], y[m], np.where(m)[0]


def fit_nested_interaction(g: pd.DataFrame, cfg: MagnitudeConfig) -> dict:
    """Rank-transform + standardize A_B and K; nested OLS on ranked R²."""
    gg = g.copy()
    gg["A_B_z"] = rank_z(gg.A_B_normal.to_numpy(float))
    gg["K_z"] = rank_z(gg.K_traceless.to_numpy(float))
    gg["Kmean_z"] = rank_z(gg.K_mean.to_numpy(float))
    gg["mean_frac_z"] = rank_z(gg.mean_fraction.to_numpy(float))
    gg["R2_rank"] = rank_z(gg.local_r2.to_numpy(float))
    # controls C2 ranked
    for c in ["log_knn_radius", "local_target_variance", "reconstruction_error", "local_evaluation_count", "A_N", "A_T", "A_PCA_normal"]:
        gg[c + "_z"] = rank_z(gg[c].to_numpy(float))
    ctrl_all = [
        c + "_z"
        for c in [
            "log_knn_radius",
            "local_target_variance",
            "reconstruction_error",
            "local_evaluation_count",
            "A_N",
            "A_T",
            "A_PCA_normal",
        ]
    ]
    # drop near-constant / ultra-collinear controls (keep science, fix VIF)
    ctrl = []
    for c in ctrl_all:
        v = gg[c].to_numpy(float)
        if np.nanstd(v) < 1e-8:
            continue
        if ctrl:
            mat = np.column_stack([gg[x].to_numpy(float) for x in ctrl + [c]])
            m = np.all(np.isfinite(mat), axis=1)
            if m.sum() > 10:
                corr = np.corrcoef(mat[m].T)
                if np.any(np.abs(corr[:-1, -1]) > 0.97):
                    continue
        ctrl.append(c)
    gg["A_B_x_K"] = gg["A_B_z"] * gg["K_z"]

    models = {
        "M0": ctrl,
        "M1": ctrl + ["A_B_z"],
        "M2": ctrl + ["K_z"],
        "M3": ctrl + ["A_B_z", "K_z"],
        "M4": ctrl + ["A_B_z", "K_z", "A_B_x_K"],
        "M5": ctrl + ["A_B_z", "K_z", "A_B_x_K", "Kmean_z", "mean_frac_z"],
    }

    y = gg.R2_rank.to_numpy(float)
    groups = gg.sample_id.to_numpy()  # leave-one-chart-out ≈ leave-one-anchor
    rows = {}
    rng = np.random.default_rng(cfg.seed + 3)

    def ols_stats(cols):
        X = np.column_stack([np.ones(len(gg)), gg[cols].to_numpy(float)])
        m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        Xr, yr = X[m], y[m]
        if len(yr) < len(cols) + 5:
            return None
        beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        pred = Xr @ beta
        ss_res = float(np.sum((yr - pred) ** 2))
        ss_tot = float(np.sum((yr - yr.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, EPS)
        # VIF / cond on design without intercept
        Z = Xr[:, 1:]
        # standardize columns
        Zs = (Z - Z.mean(0)) / (Z.std(0) + EPS)
        try:
            cond = float(np.linalg.cond(Zs.T @ Zs))
        except Exception:
            cond = float("nan")
        vifs = []
        for j in range(Zs.shape[1]):
            others = np.delete(Zs, j, axis=1)
            if others.size == 0:
                vifs.append(1.0)
                continue
            bj, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Zs)), others]), Zs[:, j], rcond=None)
            pr = np.column_stack([np.ones(len(Zs)), others]) @ bj
            r2j = 1 - np.sum((Zs[:, j] - pr) ** 2) / max(np.sum((Zs[:, j] - Zs[:, j].mean()) ** 2), EPS)
            vifs.append(float(1.0 / max(1.0 - r2j, EPS)))
        # LOGO CV
        logo = LeaveOneGroupOut()
        preds = np.full(len(yr), np.nan)
        Xm, ym, idx = Xr, yr, np.where(m)[0]
        grp = groups[idx]
        # subsample groups if too many for speed: use 5-fold grouped by hashing
        # LeaveOneGroupOut on 384 is fine for small p
        try:
            for tr, te in logo.split(Xm, ym, grp):
                if len(tr) < len(cols) + 2:
                    continue
                b, *_ = np.linalg.lstsq(Xm[tr], ym[tr], rcond=None)
                preds[te] = Xm[te] @ b
            mcv = np.isfinite(preds)
            if mcv.sum() > 10:
                ss_r = float(np.sum((ym[mcv] - preds[mcv]) ** 2))
                ss_t = float(np.sum((ym[mcv] - ym[mcv].mean()) ** 2))
                cv_r2 = 1.0 - ss_r / max(ss_t, EPS)
            else:
                cv_r2 = float("nan")
        except Exception:
            cv_r2 = float("nan")
        # bootstrap coefs for key terms
        boot = {c: [] for c in cols}
        for _ in range(min(cfg.n_bootstrap, 500)):
            take = rng.choice(len(yr), size=len(yr), replace=True)
            try:
                b, *_ = np.linalg.lstsq(Xr[take], yr[take], rcond=None)
                for j, c in enumerate(cols):
                    boot[c].append(float(b[j + 1]))
            except Exception:
                continue
        coef = {c: float(beta[j + 1]) for j, c in enumerate(cols)}
        ci = {
            c: [
                float(np.quantile(boot[c], 0.025)) if boot[c] else float("nan"),
                float(np.quantile(boot[c], 0.975)) if boot[c] else float("nan"),
            ]
            for c in cols
        }
        return {
            "r2_in": r2,
            "cv_r2_logo": cv_r2,
            "cond": cond,
            "max_vif": float(np.nanmax(vifs)) if vifs else float("nan"),
            "coef": coef,
            "ci95": ci,
            "n": int(len(yr)),
        }

    fitted = {name: ols_stats(cols) for name, cols in models.items()}
    # incremental held-out ≈ CV differences
    def cv(name):
        f = fitted.get(name)
        return f["cv_r2_logo"] if f else float("nan")

    inter = fitted.get("M4")
    out = {
        "n": int(np.isfinite(y).sum()),
        "cv_M0": cv("M0"),
        "cv_M1": cv("M1"),
        "cv_M2": cv("M2"),
        "cv_M3": cv("M3"),
        "cv_M4": cv("M4"),
        "cv_M5": cv("M5"),
        "dcv_M1": cv("M1") - cv("M0") if np.isfinite(cv("M1")) else float("nan"),
        "dcv_M2": cv("M2") - cv("M0") if np.isfinite(cv("M2")) else float("nan"),
        "dcv_M3": cv("M3") - cv("M0") if np.isfinite(cv("M3")) else float("nan"),
        "dcv_M4": cv("M4") - cv("M3") if np.isfinite(cv("M4")) else float("nan"),
        "dcv_M5": cv("M5") - cv("M4") if np.isfinite(cv("M5")) else float("nan"),
        "coef_A_B_M4": inter["coef"].get("A_B_z", float("nan")) if inter else float("nan"),
        "coef_K_M4": inter["coef"].get("K_z", float("nan")) if inter else float("nan"),
        "coef_interaction_M4": inter["coef"].get("A_B_x_K", float("nan")) if inter else float("nan"),
        "ci_interaction_M4": inter["ci95"].get("A_B_x_K", [float("nan"), float("nan")]) if inter else [float("nan"), float("nan")],
        "cond_M4": inter["cond"] if inter else float("nan"),
        "max_vif_M4": inter["max_vif"] if inter else float("nan"),
        "r2_in_M4": inter["r2_in"] if inter else float("nan"),
    }
    # also M5 mean curvature coefs
    m5 = fitted.get("M5")
    if m5:
        out["coef_Kmean_M5"] = m5["coef"].get("Kmean_z", float("nan"))
        out["coef_mean_frac_M5"] = m5["coef"].get("mean_frac_z", float("nan"))
    return out


def stratified_corrs(g: pd.DataFrame) -> list[dict]:
    rows = []
    ab = g.A_B_normal.to_numpy(float)
    kt = g.K_traceless.to_numpy(float)
    r2 = g.local_r2.to_numpy(float)
    # tertiles on finite values
    def tertile_masks(x):
        m = np.isfinite(x)
        qs = np.nanquantile(x[m], [1 / 3, 2 / 3])
        lo = m & (x <= qs[0])
        mid = m & (x > qs[0]) & (x <= qs[1])
        hi = m & (x > qs[1])
        return [("low", lo), ("mid", mid), ("high", hi)], qs

    for strat_name, x, other_name, other in [
        ("A_B_tertile", ab, "K_traceless", kt),
        ("K_traceless_tertile", kt, "A_B_normal", ab),
    ]:
        bands, qs = tertile_masks(x)
        for band, mask in bands:
            st = spearman_dict(r2[mask], other[mask])
            rows.append(
                {
                    "stratifier": strat_name,
                    "band": band,
                    "q_lo": float(qs[0]),
                    "q_hi": float(qs[1]),
                    "outcome_feature": other_name,
                    "rho": st["rho"],
                    "pvalue": st["pvalue"],
                    "n": st["n"],
                }
            )
    return rows


def classify_target(corr: dict, inter: dict) -> tuple[str, str]:
    if not np.isfinite(corr["mean_local_r2"]) or corr["mean_local_r2"] < 0.05:
        return "not_locally_probeable", "Mean local R² of fixed global probe too low."
    ab = corr["partial_A_B_C2"]
    k = corr["partial_K_traceless_C0"]
    hm = corr["partial_K_mean_C0"]
    inter_c = inter.get("coef_interaction_M4", float("nan"))
    inter_ci = inter.get("ci_interaction_M4", [float("nan"), float("nan")])
    inter_sig = np.isfinite(inter_c) and np.isfinite(inter_ci[0]) and (inter_ci[0] * inter_ci[1] > 0)
    pca_dom = abs(corr.get("partial_PCA_given_AB", 0)) >= abs(ab) and abs(corr.get("raw_A_PCA_normal", 0)) >= abs(
        corr.get("raw_A_B_normal", 0)
    )
    if inter_sig and abs(inter_c) >= 0.05 and inter.get("dcv_M4", 0) > 0.005:
        return "alignment_magnitude_interaction", "Interaction term improves held-out fit beyond additive A_B+K."
    if abs(ab) >= 0.12 and corr["p_perm_A_B_C2"] <= 0.05 and not pca_dom:
        return "orientation_mismatch", "A_B_normal associates with local R² after C2; orientation marks mismatch/use."
    if abs(ab) >= 0.12 and corr["p_perm_A_B_C2"] <= 0.05 and pca_dom:
        return "generic_normal_frame_effect", "Alignment signal present but normal-PCA explains as well or better."
    if abs(k) >= 0.12 and corr["p_perm_K_traceless_C0"] <= 0.05 and abs(ab) < 0.1:
        return "curvature_magnitude_association", "K_traceless associates with local R² after C0; A_B weak."
    if abs(hm) >= 0.12 and corr["p_perm_K_mean_C0"] <= 0.05:
        return "mean_curvature_association", "K_mean associates with local R² after C0."
    if abs(ab) < 0.08 and abs(k) < 0.08 and abs(hm) < 0.08:
        return "no_curvature_association", "No robust partial association with A_B, K_traceless, or K_mean."
    return "inconclusive", "Mixed or weak signals without a single primary pattern."


# -------------------- plots --------------------


def make_figures(df: pd.DataFrame, corr_df: pd.DataFrame, cfg: MagnitudeConfig, fig_dir: Path) -> None:
    fig_dir.mkdir(exist_ok=True)
    prim = df[(df.target == cfg.primary_target) & (df.scale_k == cfg.primary_k)]
    for xcol, xlab, fname in [
        ("A_B_normal", r"$A_{B,\mathrm{normal}}$", "r2_vs_AB_normal.png"),
        ("K_traceless", r"$K_{\mathrm{traceless}}$", "r2_vs_K_traceless.png"),
        ("K_mean", r"$K_{\mathrm{mean}}$", "r2_vs_K_mean.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(prim[xcol], prim.local_r2, s=12, alpha=0.55, c="#2c5f7c")
        ax.set_xlabel(xlab)
        ax.set_ylabel("local R² (fixed global probe)")
        ax.set_title(f"{cfg.primary_target} k={cfg.primary_k}")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=140)
        plt.close(fig)

    # interaction surface: binned mean R2
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = rank_z(prim.A_B_normal.to_numpy(float))
    y = rank_z(prim.K_traceless.to_numpy(float))
    z = prim.local_r2.to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    xb = pd.qcut(x[m], 8, duplicates="drop")
    yb = pd.qcut(y[m], 8, duplicates="drop")
    grid = pd.DataFrame({"xb": xb, "yb": yb, "z": z[m]}).groupby(["xb", "yb"], observed=True)["z"].mean().unstack()
    im = ax.imshow(grid.to_numpy(float), origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_xlabel("K_traceless rank bin")
    ax.set_ylabel("A_B_normal rank bin")
    ax.set_title("Mean local R² surface")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "interaction_surface_AB_K.png", dpi=140)
    plt.close(fig)

    # heatmaps target × metric
    targets = list(corr_df[corr_df.scale_k == cfg.primary_k].target)
    metrics = [
        ("raw_A_B_normal", "raw A_B"),
        ("partial_A_B_C2", "partial A_B|C2"),
        ("raw_K_traceless", "raw K"),
        ("partial_K_traceless_C0", "partial K|C0"),
        ("raw_K_mean", "raw K_mean"),
        ("partial_K_mean_C0", "partial K_mean|C0"),
        ("raw_A_PCA_normal", "raw A_PCA"),
    ]
    mat = np.full((len(targets), len(metrics)), np.nan)
    for i, t in enumerate(targets):
        row = corr_df[(corr_df.target == t) & (corr_df.scale_k == cfg.primary_k)].iloc[0]
        for j, (key, _) in enumerate(metrics):
            mat[i, j] = float(row[key])
    fig, ax = plt.subplots(figsize=(1.8 + 0.85 * len(metrics), 1.2 + 0.45 * len(targets)))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([lab for _, lab in metrics], rotation=30, ha="right")
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Correlations k={cfg.primary_k}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmap_target_metrics.png", dpi=140)
    plt.close(fig)

    # raw vs partial A_B and K
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    sub = corr_df[corr_df.scale_k == cfg.primary_k]
    x = np.arange(len(sub))
    axes[0].bar(x - 0.15, sub.raw_A_B_normal, 0.3, label="raw")
    axes[0].bar(x + 0.15, sub.partial_A_B_C2, 0.3, label="partial C2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sub.target, rotation=25, ha="right")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].legend(fontsize=8)
    axes[0].set_title("A_B_normal")
    axes[1].bar(x - 0.15, sub.raw_K_traceless, 0.3, label="raw")
    axes[1].bar(x + 0.15, sub.partial_K_traceless_C0, 0.3, label="partial C0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub.target, rotation=25, ha="right")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].legend(fontsize=8)
    axes[1].set_title("K_traceless")
    fig.tight_layout()
    fig.savefig(fig_dir / "raw_vs_partial.png", dpi=140)
    plt.close(fig)

    # AB vs APCA
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - 0.15, sub.raw_A_B_normal, 0.3, label="A_B_normal")
    ax.bar(x + 0.15, sub.raw_A_PCA_normal, 0.3, label="A_PCA_normal")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.target, rotation=25, ha="right")
    ax.legend()
    ax.set_title("Curvature vs normal-PCA alignment (raw)")
    fig.tight_layout()
    fig.savefig(fig_dir / "AB_vs_APCA.png", dpi=140)
    plt.close(fig)


# -------------------- main --------------------


def run_magnitude_analysis(cfg: MagnitudeConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    profile: dict[str, Any] = {"stages": {}, "cache_reuse": {}}

    align_dir = resolve_path(root, cfg.align_dir)
    align_path = align_dir / "anchor_target_probe_results.parquet"
    if not align_path.exists():
        raise FileNotFoundError(
            f"Missing {align_path}; run global_probe_curvature_alignment first."
        )
    align_df = pd.read_parquet(align_path)
    inv_src = align_dir / "target_inventory.csv"
    if inv_src.exists():
        inv = pd.read_csv(inv_src)
        inv.to_csv(out / "target_inventory.csv", index=False)
    else:
        inv = None

    # verify hash on align rows
    if "config_hash" in align_df.columns:
        hashes = set(align_df.config_hash.astype(str).unique())
        if hashes != {cfg.expected_hash}:
            raise RuntimeError(f"align hash mismatch {hashes}")

    scfg = ScreenConfig(curvature_path=cfg.curvature_path, expected_hash=cfg.expected_hash)
    curv = load_frozen_curvature(root, scfg)
    data = load_prepare(resolve_path(root, cfg.prepare_dir))
    X = data["X"].astype(np.float64)
    sample_ids = data["sample_ids"]
    train = np.asarray(data["train_local"])

    lab = np.load(resolve_path(root, cfg.labels_path))
    y_all = {k: np.asarray(lab[k], dtype=np.float64)[sample_ids] for k in lab.files}
    if inv is None:
        inv = build_target_inventory(
            y_all, train, GlobalProbeAlignConfig(primary_target=cfg.primary_target)
        )
        inv.to_csv(out / "target_inventory.csv", index=False)

    targets = inv.loc[inv.included, "target"].tolist()
    print(f"[mag] targets={targets}", flush=True)
    print(f"[mag] train_fn={CANONICAL_PROBE_TRAIN}", flush=True)
    print(f"[mag] local_fn={CANONICAL_PROBE_LOCAL}", flush=True)

    # load frozen global probe weights
    t1 = time.time()
    wpath = align_dir / "global_probe_weights.npz"
    probes: dict[str, np.ndarray] = {}
    if wpath.exists():
        wz = np.load(wpath)
        for t in targets:
            key = f"w_{t}"
            if key not in wz.files:
                raise KeyError(f"{key} missing from {wpath}")
            probes[t] = np.asarray(wz[key], dtype=np.float64)
        profile["cache_reuse"]["global_probe_weights"] = True
    else:
        # rematerialize all targets jointly via sklearn fit_global_probe (CPU; small)
        profile["cache_reuse"]["global_probe_weights"] = False
        for t in targets:
            y = y_all[t]
            m = np.isfinite(y[train])
            coef, _b = fit_global_probe(X[train][m], y[train][m], cfg.probe_alpha)
            probes[t] = coef
    profile["stages"]["load_probes_s"] = time.time() - t1

    t1 = time.time()
    df = build_metrics_table(root, cfg, align_df, curv, X, y_all, sample_ids, probes)
    # keep only included targets / scales
    df = df[df.target.isin(targets) & df.scale_k.isin(cfg.scales)].copy()
    df.to_parquet(out / "anchor_target_curvature_metrics.parquet", index=False)
    profile["stages"]["metrics_s"] = time.time() - t1
    profile["cache_reuse"]["align_rows"] = True
    profile["cache_reuse"]["frozen_curvature"] = True

    # correlations per target×scale with checkpoints
    t1 = time.time()
    corr_rows = []
    for k in cfg.scales:
        for t in targets:
            shard = out / f"corr_{t}_k{k}.json"
            if _done(shard, cfg.force):
                corr_rows.append(json.loads(shard.read_text()))
                continue
            g = df[(df.target == t) & (df.scale_k == k)]
            row = correlate_target_scale(g, t, k, cfg)
            shard.write_text(json.dumps(row, indent=2))
            corr_rows.append(row)
            print(f"[mag] corr {t} k={k} A_B|C2={row['partial_A_B_C2']:.3f} K|C0={row['partial_K_traceless_C0']:.3f}", flush=True)
    corr_df = pd.DataFrame(corr_rows)
    # FDR on secondary targets for primary hypotheses at primary k
    for hyp, pcol, fcol in [
        ("A_B_C2", "p_perm_A_B_C2", "p_fdr_A_B_C2"),
        ("K_C0", "p_perm_K_traceless_C0", "p_fdr_K_traceless_C0"),
        ("Kmean_C0", "p_perm_K_mean_C0", "p_fdr_K_mean_C0"),
    ]:
        corr_df[fcol] = np.nan
        sec = (corr_df.scale_k == cfg.primary_k) & (corr_df.target != cfg.primary_target)
        if sec.any():
            corr_df.loc[sec, fcol] = bh_fdr(corr_df.loc[sec, pcol].to_numpy(float))
        prim = (corr_df.scale_k == cfg.primary_k) & (corr_df.target == cfg.primary_target)
        corr_df.loc[prim, fcol] = corr_df.loc[prim, pcol].to_numpy()
    corr_df.to_parquet(out / "target_curvature_correlations.parquet", index=False)
    corr_df.to_csv(out / "target_curvature_correlations.csv", index=False)
    profile["stages"]["correlations_s"] = time.time() - t1

    # interaction models + stratified
    t1 = time.time()
    inter_rows = []
    strat_rows = []
    for k in cfg.scales:
        for t in targets:
            g = df[(df.target == t) & (df.scale_k == k)]
            inter = fit_nested_interaction(g, cfg)
            inter.update({"target": t, "scale_k": k})
            inter_rows.append(inter)
            for r in stratified_corrs(g):
                r.update({"target": t, "scale_k": k})
                strat_rows.append(r)
    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_parquet(out / "target_interaction_models.parquet", index=False)
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_parquet(out / "target_stratified_results.parquet", index=False)
    profile["stages"]["interaction_s"] = time.time() - t1

    # max-stat for secondary A_B and K at primary k
    sec_targets = [t for t in targets if t != cfg.primary_target]
    maxstat = {}
    for hyp, col in [("A_B_C2", "partial_A_B_C2"), ("K_traceless_C0", "partial_K_traceless_C0")]:
        sub = corr_df[(corr_df.scale_k == cfg.primary_k) & (corr_df.target.isin(sec_targets))]
        obs = sub[col].abs().to_numpy(float)
        obs_max = float(np.nanmax(obs)) if len(obs) else float("nan")
        # joint permutation using primary-k metrics
        g0 = df[df.scale_k == cfg.primary_k]
        ids = sorted(g0.sample_id.unique())
        rng = np.random.default_rng(cfg.seed + hash(hyp) % 1000)
        nulls = []
        mats = []
        for t in sec_targets:
            gt = g0[g0.target == t].set_index("sample_id").reindex(ids)
            r2 = gt.local_r2.to_numpy(float)
            if hyp.startswith("A_B"):
                x = gt.A_B_normal.to_numpy(float)
                Z = np.column_stack(
                    [
                        gt.log_knn_radius,
                        gt.local_target_variance,
                        gt.reconstruction_error,
                        gt.local_evaluation_count,
                        gt.A_N,
                        gt.A_T,
                        gt.A_PCA_normal,
                    ]
                ).astype(float)
            else:
                x = gt.K_traceless.to_numpy(float)
                Z = np.column_stack(
                    [
                        gt.log_knn_radius,
                        gt.local_target_variance,
                        gt.reconstruction_error,
                        gt.local_evaluation_count,
                    ]
                ).astype(float)
            mats.append((x, r2, Z))
        for _ in range(cfg.n_permute):
            perm = rng.permutation(len(ids))
            mxs = [abs(partial_spearman(x, r2[perm], Z)["rho"]) for x, r2, Z in mats]
            nulls.append(float(np.nanmax(mxs)))
        p_ms, B = mc_p_greater(obs_max, np.asarray(nulls))
        maxstat[hyp] = {"obs_max_abs": obs_max, "p_maxstat": p_ms, "B_perm": B}
    (out / "maxstat_secondary.json").write_text(json.dumps(maxstat, indent=2))

    # classification
    class_rows = []
    for t in targets:
        c = corr_df[(corr_df.target == t) & (corr_df.scale_k == cfg.primary_k)].iloc[0].to_dict()
        inter = inter_df[(inter_df.target == t) & (inter_df.scale_k == cfg.primary_k)].iloc[0].to_dict()
        lab, why = classify_target(c, inter)
        fam = inv.loc[inv.target == t, "family"].iloc[0] if "family" in inv.columns else "other"
        class_rows.append(
            {
                "target": t,
                "family": fam,
                "role": "primary" if t == cfg.primary_target else "secondary",
                "label": lab,
                "reason": why,
                "partial_A_B_C2": c["partial_A_B_C2"],
                "partial_K_traceless_C0": c["partial_K_traceless_C0"],
                "partial_K_mean_C0": c["partial_K_mean_C0"],
                "coef_interaction_M4": inter["coef_interaction_M4"],
                "p_perm_A_B_C2": c["p_perm_A_B_C2"],
                "p_fdr_A_B_C2": c.get("p_fdr_A_B_C2", c["p_perm_A_B_C2"]),
                "p_fdr_K_traceless_C0": c.get("p_fdr_K_traceless_C0", c["p_perm_K_traceless_C0"]),
                "mean_local_r2": c["mean_local_r2"],
            }
        )
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(out / "target_classification.csv", index=False)

    # family summary
    fam_rows = []
    for fam, sub in class_df.groupby("family"):
        rh = corr_df[(corr_df.scale_k == cfg.primary_k) & (corr_df.target.isin(sub.target))]
        fam_rows.append(
            {
                "family": fam,
                "n_targets": int(len(sub)),
                "fisher_partial_A_B_C2": fisher_meta(rh.partial_A_B_C2.to_numpy(float), rh.n.to_numpy(float))["rho"],
                "fisher_partial_K_C0": fisher_meta(rh.partial_K_traceless_C0.to_numpy(float), rh.n.to_numpy(float))["rho"],
                "labels": ",".join(sub.label.tolist()),
            }
        )
    pd.DataFrame(fam_rows).to_csv(out / "target_family_summary.csv", index=False)

    make_figures(df, corr_df, cfg, out / "figures")

    mag = corr_df[(corr_df.target == cfg.primary_target) & (corr_df.scale_k == cfg.primary_k)].iloc[0]
    mag_i = inter_df[(inter_df.target == cfg.primary_target) & (inter_df.scale_k == cfg.primary_k)].iloc[0]
    # replication check vs prior align summary
    prior_path = align_dir / "target_alignment_summary.csv"
    prior_ab = float("nan")
    if prior_path.exists():
        pr = pd.read_csv(prior_path)
        prow = pr[(pr.target == cfg.primary_target) & (pr.scale_k == cfg.primary_k)]
        if len(prow) and "rho_R2_A_B_normal" in prow.columns:
            prior_ab = float(prow.iloc[0]["rho_R2_A_B_normal"])

    profile.update(
        {
            "peak_rss_mb": _rss(),
            "peak_vram_mb": 0.0,
            "total_seconds": time.time() - t0,
            "n_targets": len(targets),
            "n_anchors": 384,
            "canonical_train_fn": CANONICAL_PROBE_TRAIN,
            "canonical_local_fn": CANONICAL_PROBE_LOCAL,
            "probe_alpha": cfg.probe_alpha,
            "config_hash": cfg.expected_hash,
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    n_sec = int((class_df.role == "secondary").sum())
    n_fdr_ab = int(((class_df.role == "secondary") & (class_df.p_fdr_A_B_C2 <= 0.05) & (class_df.partial_A_B_C2.abs() >= 0.1)).sum())
    n_fdr_k = int(((class_df.role == "secondary") & (class_df.p_fdr_K_traceless_C0 <= 0.05) & (class_df.partial_K_traceless_C0.abs() >= 0.1)).sum())

    # interpretation sentence for mag_r
    ab, kpart, hmp = mag.partial_A_B_C2, mag.partial_K_traceless_C0, mag.partial_K_mean_C0
    inter_c = float(mag_i.coef_interaction_M4)
    if abs(ab) >= 0.1 and abs(kpart) < 0.1:
        reading = "orientation_mismatch: curvature orientation marks global-probe mismatch; magnitude null/weak after controls."
    elif abs(kpart) >= 0.1 and abs(ab) < 0.1:
        reading = "curvature_magnitude_association: stronger local bending associates with weak/strong probe performance; A_B null/weak."
    elif np.isfinite(inter_c) and abs(inter_c) >= 0.05 and mag_i.ci_interaction_M4[0] * mag_i.ci_interaction_M4[1] > 0:
        reading = "alignment_magnitude_interaction: joint A_B×K term is supported by the nested models."
    elif abs(hmp) >= 0.1:
        reading = "mean_curvature_association: bowl/cap-like mean curvature associates with local performance."
    else:
        reading = "mixed/weak; see tables. Prior fixed-probe ablation had ΔB_fixed<ΔPCA_fixed — do not claim unique curvature information from A_B alone."

    report = f"""# Global-probe curvature magnitude × alignment

Frozen hash `{cfg.expected_hash}`. Alpha=`{cfg.probe_alpha}`.
Protocol: **fixed global ridge probes only** (no per-anchor refits).

## Canonical functions

| Role | Source |
|------|--------|
| Global ridge train | `{CANONICAL_PROBE_TRAIN}` |
| Local performance | `{CANONICAL_PROBE_LOCAL}` |
| Alpha (frozen) | `curvature_probe_screen.select_ridge_alpha` → 100 |

## Cache reuse

{json.dumps(profile['cache_reuse'], indent=2)}

## Target inventory

{inv.to_string(index=False)}

## Confirmatory replication (`mag_r_desi`, k={cfg.primary_k})

Prior corrected raw corr(R², A_B_normal) ≈ {prior_ab:.4f}.
This run raw = {mag.raw_A_B_normal:.4f}; partial A_B\\|C2 = {mag.partial_A_B_C2:.4f} (perm p={mag.p_perm_A_B_C2:.4g}).

| Statistic | ρ | perm p |
|-----------|---|--------|
| raw A_B_normal | {mag.raw_A_B_normal:.4f} | {mag.p_raw_A_B_normal:.4g} |
| partial A_B \\| C2 (primary alignment) | {mag.partial_A_B_C2:.4f} | {mag.p_perm_A_B_C2:.4g} |
| raw K_traceless | {mag.raw_K_traceless:.4f} | {mag.p_raw_K_traceless:.4g} |
| partial K_traceless \\| C0 (primary magnitude) | {mag.partial_K_traceless_C0:.4f} | {mag.p_perm_K_traceless_C0:.4g} |
| partial K_mean \\| C0 | {mag.partial_K_mean_C0:.4f} | {mag.p_perm_K_mean_C0:.4g} |
| raw C_traceless | {mag.raw_C_traceless:.4f} | — |
| partial C_traceless \\| C0 | {mag.partial_C_traceless_C0:.4f} | — |
| corr(log ρ, C_traceless) | {mag.corr_radius_C_traceless:.4f} | — |
| partial PCA \\| C1+A_B | {mag.partial_PCA_given_AB:.4f} | — |

Interaction (M4): coef(A_B×K)={mag_i.coef_interaction_M4:.4f}, CI95={mag_i.ci_interaction_M4}, ΔCV(M4−M3)={mag_i.dcv_M4:.4f}, cond={mag_i.cond_M4:.3g}, maxVIF={mag_i.max_vif_M4:.3g}.

**Reading:** {reading}

## Secondary targets (k={cfg.primary_k})

- FDR survivors A_B\\|C2: {n_fdr_ab}/{n_sec}
- FDR survivors K\\|C0: {n_fdr_k}/{n_sec}
- max-stat A_B: obs={maxstat.get('A_B_C2',{}).get('obs_max_abs', float('nan')):.4f}, p={maxstat.get('A_B_C2',{}).get('p_maxstat', float('nan')):.4g}
- max-stat K: obs={maxstat.get('K_traceless_C0',{}).get('obs_max_abs', float('nan')):.4f}, p={maxstat.get('K_traceless_C0',{}).get('p_maxstat', float('nan')):.4g}

## Classification

{class_df.to_string(index=False)}

## Correlations (all targets × scales)

{corr_df[['target','scale_k','raw_A_B_normal','partial_A_B_C2','raw_K_traceless','partial_K_traceless_C0','partial_K_mean_C0','partial_PCA_given_AB']].to_string(index=False)}

## Interaction models (primary k)

{inter_df[inter_df.scale_k==cfg.primary_k][['target','coef_A_B_M4','coef_K_M4','coef_interaction_M4','dcv_M1','dcv_M2','dcv_M4','max_vif_M4']].to_string(index=False)}

## Scale confirmation (mag_r_desi k=1024)

{corr_df[(corr_df.target=='mag_r_desi')&(corr_df.scale_k==1024)][['raw_A_B_normal','partial_A_B_C2','partial_K_traceless_C0','partial_K_mean_C0']].to_string(index=False)}

## Caveats

- C_traceless is scale-integrated (ρ·K); report only with radius controls / radius correlation.
- Prior fixed-probe ablation: ΔB_fixed < ΔPCA_fixed — A_B associations are geographic markers, not proof of unique curvature information.
- No causal interpretation of partial correlations.

## Answers

1. Alignment replication: raw A_B={mag.raw_A_B_normal:.3f} (prior≈{prior_ab:.3f}); partial C2={mag.partial_A_B_C2:.3f}.
2. Magnitude: partial K\\|C0={mag.partial_K_traceless_C0:.3f} (p={mag.p_perm_K_traceless_C0:.4g}).
3. Mean curvature: partial K_mean\\|C0={mag.partial_K_mean_C0:.3f} (p={mag.p_perm_K_mean_C0:.4g}).
4. Interaction: coef={mag_i.coef_interaction_M4:.3f}, ΔCV={mag_i.dcv_M4:.4f}.
5. Beyond normal-PCA: partial PCA\\|AB={mag.partial_PCA_given_AB:.3f}.
6. Multiplicity: A_B FDR {n_fdr_ab}/{n_sec}; K FDR {n_fdr_k}/{n_sec}.
7. Shared vs specific: see classifications / family Fisher.
8. Runtime: {profile['total_seconds']:.1f}s; peak RSS={profile['peak_rss_mb']:.1f} MB; VRAM≈0 (CPU stats).

## Strongest defensible conclusion

{reading}

## Exact command

```bash
cd ~/platonic-universe && source .venv/bin/activate && \\
PYTHONPATH=experiments python -m geometry.run_global_probe_curvature_magnitude \\
  --force --seed 0
```
"""
    (out / "REPORT.md").write_text(report)
    analysis = {
        "mag_r_primary": mag.to_dict(),
        "mag_r_interaction": mag_i.to_dict(),
        "prior_raw_A_B": prior_ab,
        "classifications": class_df.to_dict(orient="records"),
        "maxstat": maxstat,
        "reading": reading,
        "runtime": profile,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"[mag] done in {profile['total_seconds']:.1f}s label={class_df.loc[class_df.target==cfg.primary_target,'label'].iloc[0]}", flush=True)
    return analysis
