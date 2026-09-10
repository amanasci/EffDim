"""Seed reliability, primary P1–P3, secondary, Q comparison, nulls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y

from .config import CONTROLS, INFER_SEED, SEED_COS_GATE, SEED_RHO_GATE
from .io_util import p_mc


def spearman_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8 or float(np.std(a[m])) < 1e-15 or float(np.std(b[m])) < 1e-15:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def pearson_rank(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return float("nan")
    ra, rb = rankdata(a[m]), rankdata(b[m])
    if float(np.std(ra)) < 1e-15 or float(np.std(rb)) < 1e-15:
        return float("nan")
    return float(pearsonr(ra, rb).statistic)


def holm(ps: np.ndarray) -> np.ndarray:
    ps = np.asarray(ps, dtype=np.float64)
    m = len(ps)
    order = np.argsort(ps)
    out = np.empty(m, dtype=np.float64)
    prev = 0.0
    for rank, i in enumerate(order):
        prev = min(1.0, max(prev, (m - rank) * ps[i]))
        out[i] = prev
    return out


def _assoc(df: pd.DataFrame, xcol: str, ycol: str, extra_controls: list[str] | None = None) -> dict[str, float]:
    sub = df.reset_index(drop=True).copy()
    cols = list(CONTROLS) + list(extra_controls or [])
    for c in cols:
        if c not in sub.columns:
            sub[c] = np.nan
    Z = np.column_stack([sub[c].to_numpy(float) for c in cols])
    return associate(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float), Z)


def seed_reliability(
    per_seed: dict[int, dict[str, np.ndarray]],
    df: pd.DataFrame,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Do not average C_H before this gate. No best-seed selection."""
    pairs = []
    C = {s: np.asarray(per_seed[s]["C_H"], dtype=np.float64) for s in seeds}
    H = {s: np.asarray(per_seed[s]["H_S"], dtype=np.float64) for s in seeds}
    recon = {s: np.asarray(per_seed[s]["recon"], dtype=np.float64) for s in seeds}
    cond = {s: np.asarray(per_seed[s]["cond_g"], dtype=np.float64) for s in seeds}
    extra = pd.DataFrame(
        {
            "sample_id": df.sample_id.to_numpy(int),
            "log_knn_radius": df.log_knn_radius.to_numpy(float),
            "local_label_variance": df.local_label_variance.to_numpy(float),
            "local_evaluation_count": df.local_evaluation_count.to_numpy(float),
        }
    )
    for i, s in enumerate(seeds):
        for t in seeds[i + 1 :]:
            cos = np.sum(H[s] * H[t], axis=1) / np.clip(
                np.linalg.norm(H[s], axis=1) * np.linalg.norm(H[t], axis=1), 1e-30, None
            )
            rel = np.abs(C[s] - C[t]) / np.clip(0.5 * (C[s] + C[t]), 1e-15, None)
            tmp = extra.copy()
            tmp["Cs"] = C[s]
            tmp["Ct"] = C[t]
            tmp["recon_mean"] = 0.5 * (recon[s] + recon[t])
            Z = control_matrix(tmp)
            Z2 = np.column_stack([Z, tmp.recon_mean.to_numpy(float)])
            rec = {
                "pair": f"{s}-{t}",
                "seed_a": int(s),
                "seed_b": int(t),
                "rho_CH": spearman_safe(C[s], C[t]),
                "pearson_rank_CH": pearson_rank(C[s], C[t]),
                "median_cos_HS": float(np.median(cos)),
                "median_rel_norm": float(np.median(rel)),
                "rho_recon": spearman_safe(recon[s], recon[t]),
                "rho_cond": spearman_safe(cond[s], cond[t]),
                "rho_CH_ctl_radius": associate(C[s], C[t], Z)["controlled"],
                "rho_CH_ctl_radius_recon": associate(C[s], C[t], Z2)["controlled"],
            }
            pairs.append(rec)
    rho_med = float(np.median([p["rho_CH"] for p in pairs])) if pairs else float("nan")
    cos_med = float(np.median([p["median_cos_HS"] for p in pairs])) if pairs else float("nan")
    pass_rho = bool(np.isfinite(rho_med) and rho_med >= SEED_RHO_GATE)
    pass_cos = bool(np.isfinite(cos_med) and cos_med >= SEED_COS_GATE)
    passed = bool(pass_rho and pass_cos)
    consensus_rank = None
    consensus_mag = None
    if passed:
        ranks = np.column_stack([rankdata(C[s]) for s in seeds])
        consensus_rank = np.median(ranks, axis=1)
        consensus_mag = np.median(np.column_stack([C[s] for s in seeds]), axis=1)
    return {
        "pairs": pairs,
        "median_rho_CH": rho_med,
        "median_cos_HS": cos_med,
        "pass_rho": pass_rho,
        "pass_cos": pass_cos,
        "passed": passed,
        "n_seeds": int(len(seeds)),
        "gate_rho": SEED_RHO_GATE,
        "gate_cos": SEED_COS_GATE,
        "best_seed_selected": False,
        "consensus_is_median_rank": bool(passed),
        "consensus_rank": consensus_rank,
        "consensus_mag": consensus_mag,
    }


def primary_family(
    df: pd.DataFrame,
    xcol: str,
    *,
    n_perm: int,
    n_boot: int,
    seed: int = INFER_SEED,
) -> dict[str, Any]:
    Z = control_matrix(df.reset_index(drop=True))
    x = df[xcol].to_numpy(float)
    yG = df["r2_G"].to_numpy(float)
    yP = df["r2_P"].to_numpy(float)
    p1 = associate(x, yG, Z)
    p2 = associate(x, yP, Z)
    delta = float(p2["controlled"]) - float(p1["controlled"])
    rng = np.random.default_rng(seed)
    n = len(df)
    boot = {k: np.empty(n_boot) for k in ("P1", "P2", "P3")}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        a1 = _assoc(sub, xcol, "r2_G")
        a2 = _assoc(sub, xcol, "r2_P")
        boot["P1"][b] = a1["controlled"]
        boot["P2"][b] = a2["controlled"]
        boot["P3"][b] = a2["controlled"] - a1["controlled"]
    null1 = np.empty(n_perm)
    null2 = np.empty(n_perm)
    null3 = np.empty(n_perm)
    for b in range(n_perm):
        xp = freedman_lane_y(x, Z, rng)
        a1 = associate(xp, yG, Z)
        a2 = associate(xp, yP, Z)
        null1[b] = a1["controlled"]
        null2[b] = a2["controlled"]
        null3[b] = a2["controlled"] - a1["controlled"]

    def pack_two_sided(name, obs, null, bt):
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        b_count = int(np.sum(np.abs(null) >= abs(obs))) if np.isfinite(obs) else n_perm
        return {
            "name": name,
            "observed": float(obs),
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "side": "two-sided",
        }

    def pack_greater(name, obs, null, bt):
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        b_count = int(np.sum(null >= obs)) if np.isfinite(obs) else n_perm
        return {
            "name": name,
            "observed": float(obs),
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "side": "greater",
        }

    fam = {
        "P1": pack_two_sided("P1_CH_R2G", float(p1["controlled"]), null1, boot["P1"]),
        "P2": pack_two_sided("P2_CH_R2P", float(p2["controlled"]), null2, boot["P2"]),
        "P3": pack_greater("P3_delta_rho", delta, null3, boot["P3"]),
    }
    ps = np.array([fam["P1"]["p_mc"], fam["P2"]["p_mc"], fam["P3"]["p_mc"]], dtype=np.float64)
    hs = holm(ps)
    for i, k in enumerate(("P1", "P2", "P3")):
        fam[k]["p_holm"] = float(hs[i])
        fam[k]["raw"] = float(associate(x, {"P1": yG, "P2": yP, "P3": yP}[k] if k != "P3" else yP, None)["raw"]) if k != "P3" else float("nan")
        fam[k]["n"] = int(p1["n"])
        fam[k]["n_perm"] = int(n_perm)
        fam[k]["n_boot"] = int(n_boot)
    fam["P1"]["raw"] = float(p1["raw"])
    fam["P2"]["raw"] = float(p2["raw"])
    fam["P1"]["point"] = p1
    fam["P2"]["point"] = p2
    fam["P3"]["components"] = {"rho_R2P": float(p2["controlled"]), "rho_R2G": float(p1["controlled"])}
    return fam


def secondary_table(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    specs = [
        ("rho_CH_MSE_G", "mse_G"),
        ("rho_CH_MSE_P", "mse_P"),
        ("rho_CH_DeltaAdapt", "delta_adapt"),
        ("rho_CH_MAE_G", "mae_G"),
        ("rho_CH_MAE_P", "mae_P"),
        ("rho_CH_dMAE", "dMAE_G_to_P"),
        ("rho_CH_R2_T", "r2_T"),
        ("rho_CH_MSE_T", "mse_T"),
    ]
    rows = []
    for name, col in specs:
        if col not in df.columns:
            continue
        rec = _assoc(df, xcol, col)
        rec["name"] = name
        rec["ycol"] = col
        rec["xcol"] = xcol
        rows.append(rec)
    return pd.DataFrame(rows)


def q_comparison(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    rows = []
    for qcol in ("K_H_cross", "K_dir_cross"):
        if qcol not in df.columns:
            continue
        raw = associate(df[xcol].to_numpy(float), df[qcol].to_numpy(float), None)
        ctl = _assoc(df, xcol, qcol)
        rows.append({"contrast": f"{xcol}_vs_{qcol}", "raw": raw["raw"], **{f"ctl_{k}": v for k, v in ctl.items()}})
    # Q associations after conditioning on C_H, and conversely
    extra_ch = [xcol]
    extra_kh = ["K_H_cross"] if "K_H_cross" in df.columns else []
    for ycol, label in (("r2_G", "R2G"), ("delta_adapt", "DeltaAdapt")):
        if extra_kh:
            rec = _assoc(df, "K_H_cross", ycol, extra_controls=extra_ch)
            rec["name"] = f"rho_ctl_KH_{label}_given_CH"
            rec["ycol"] = ycol
            rows.append(rec)
        rec2 = _assoc(df, xcol, ycol, extra_controls=extra_kh)
        rec2["name"] = f"rho_ctl_CH_{label}_given_KH"
        rec2["ycol"] = ycol
        rows.append(rec2)
        rec3 = _assoc(df, xcol, ycol)
        rec3["name"] = f"rho_ctl_CH_{label}"
        rec3["ycol"] = ycol
        rows.append(rec3)
        if extra_kh:
            rec4 = _assoc(df, "K_H_cross", ycol)
            rec4["name"] = f"rho_ctl_KH_{label}"
            rec4["ycol"] = ycol
            rows.append(rec4)
    return pd.DataFrame(rows)


def sensitivity_table(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    rows = []
    extra = [c for c in ("recon", "cond_g") if c in df.columns]
    for ycol in ("r2_G", "r2_P", "mse_G", "mse_P", "delta_adapt"):
        rec = _assoc(df, xcol, ycol, extra_controls=extra)
        rec["name"] = f"rho_ctl_{ycol}_plus_decoder_controls"
        rec["ycol"] = ycol
        rows.append(rec)
    if "recon" in df.columns:
        rec = _assoc(df, xcol, "recon")
        rec["name"] = "rho_ctl_CH_recon"
        rec["ycol"] = "recon"
        rows.append(rec)
    if "cond_g" in df.columns:
        rec = _assoc(df, xcol, "cond_g")
        rec["name"] = "rho_ctl_CH_cond"
        rec["ycol"] = "cond_g"
        rows.append(rec)
    rec = _assoc(df, xcol, "log_knn_radius")
    rec["name"] = "rho_ctl_CH_log_knn_radius"
    rec["ycol"] = "log_knn_radius"
    rows.append(rec)
    return pd.DataFrame(rows)


def drop_worst(df: pd.DataFrame, col: str, frac: float = 0.05) -> pd.DataFrame:
    s = df[col].to_numpy(float)
    thr = np.nanquantile(s, 1.0 - frac)
    return df.loc[s <= thr].reset_index(drop=True)


def null_results(df: pd.DataFrame, xcol: str, *, n_perm: int, seed: int = INFER_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 7)
    x = df[xcol].to_numpy(float)
    Z = control_matrix(df.reset_index(drop=True))
    yG = df.r2_G.to_numpy(float)
    obs = associate(x, yG, Z)["controlled"]
    null = np.empty(n_perm)
    for b in range(n_perm):
        xp = rng.permutation(x)
        null[b] = associate(xp, yG, Z)["controlled"]
    b_count = int(np.sum(np.abs(null) >= abs(obs))) if np.isfinite(obs) else n_perm
    return {
        "anchor_perm_CH_vs_R2G": {
            "observed": float(obs),
            "p_mc": p_mc(b_count, n_perm),
            "null_median": float(np.median(null)),
        }
    }


def seed_label_permutation(per_seed: dict[int, dict[str, np.ndarray]], seeds: tuple[int, ...], n: int = 200, seed: int = INFER_SEED) -> dict[str, Any]:
    """Consensus median-rank is a symmetric function of seeds; shuffling labels must not select a winner."""
    rng = np.random.default_rng(seed + 13)
    C = np.column_stack([per_seed[s]["C_H"] for s in seeds])
    ranks = np.column_stack([rankdata(C[:, i]) for i in range(C.shape[1])])
    cons0 = np.median(ranks, axis=1)
    rhos = []
    for _ in range(n):
        perm = rng.permutation(C.shape[1])
        rp = ranks[:, perm]
        cons = np.median(rp, axis=1)
        rhos.append(spearman_safe(cons0, cons))
    return {
        "n": int(n),
        "median_rho_vs_unpermuted_consensus": float(np.median(rhos)),
        "min_rho": float(np.min(rhos)),
        "seed_selective": bool(float(np.min(rhos)) < 0.999),
        "note": "Median rank is permutation-invariant across seed labels; min rho should be 1.",
    }


def decile_curves(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    q = pd.qcut(df[xcol].rank(method="first"), 10, labels=False, duplicates="drop")
    g = df.groupby(q, dropna=False)
    out = g.agg(
        C_H_mean=(xcol, "mean"),
        r2_G_mean=("r2_G", "mean"),
        r2_P_mean=("r2_P", "mean"),
        mse_G_mean=("mse_G", "mean"),
        mse_P_mean=("mse_P", "mean"),
        delta_adapt_mean=("delta_adapt", "mean"),
        n=("sample_id", "size"),
    ).reset_index(names="decile")
    out["patch_R2_exceeds_global"] = out["r2_P_mean"] > out["r2_G_mean"]
    return out
