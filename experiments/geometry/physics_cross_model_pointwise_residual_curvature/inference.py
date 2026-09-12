"""Per-model associations, seed gates, synchronized H1–H3, heterogeneity, D vs Q."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y

from .config import CONTROLS, EQUIV_ABS, INFER_SEED, REFERENCE, SEED_COS_GATE, SEED_RHO_GATE
from .io_util import p_mc


def spearman_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8 or float(np.std(a[m])) < 1e-15 or float(np.std(b[m])) < 1e-15:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


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


def bh(ps: np.ndarray) -> np.ndarray:
    ps = np.asarray(ps, dtype=np.float64)
    m = len(ps)
    order = np.argsort(ps)
    out = np.empty(m, dtype=np.float64)
    prev = 1.0
    for k, i in enumerate(order[::-1]):
        rank = m - k
        prev = min(prev, ps[i] * m / rank)
        out[i] = min(1.0, prev)
    return out


def _assoc(df: pd.DataFrame, xcol: str, ycol: str, extra_controls: list[str] | None = None, controls: tuple[str, ...] | None = None) -> dict[str, float]:
    sub = df.reset_index(drop=True).copy()
    cols = list(controls if controls is not None else CONTROLS) + list(extra_controls or [])
    Z = np.column_stack([sub[c].to_numpy(float) if c in sub.columns else np.full(len(sub), np.nan) for c in cols]) if cols else None
    if Z is not None and Z.size == 0:
        Z = None
    return associate(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float), Z)


def seed_reliability(per_seed: dict[int, dict[str, np.ndarray]], df: pd.DataFrame, seeds: tuple[int, ...]) -> dict[str, Any]:
    pairs = []
    C = {s: np.asarray(per_seed[s]["C_H"], dtype=np.float64) for s in seeds}
    H = {s: np.asarray(per_seed[s]["H_S"], dtype=np.float64) for s in seeds}
    recon = {s: np.asarray(per_seed[s]["recon"], dtype=np.float64) for s in seeds}
    extra = pd.DataFrame(
        {
            "log_knn_radius": df.log_knn_radius.to_numpy(float),
            "local_label_variance": df.local_label_variance.to_numpy(float),
            "local_evaluation_count": df.local_evaluation_count.to_numpy(float),
        }
    )
    for i, s in enumerate(seeds):
        for t in seeds[i + 1 :]:
            cos = np.sum(H[s] * H[t], axis=1) / np.clip(np.linalg.norm(H[s], axis=1) * np.linalg.norm(H[t], axis=1), 1e-30, None)
            rel = np.abs(C[s] - C[t]) / np.clip(0.5 * (C[s] + C[t]), 1e-15, None)
            rs, rt = rankdata(C[s]), rankdata(C[t])
            q = np.quantile(rs, 0.75)
            jacc = float(len(set(np.where(rs >= q)[0]) & set(np.where(rt >= q)[0])) / max(len(set(np.where(rs >= q)[0]) | set(np.where(rt >= q)[0])), 1))
            tmp = extra.copy()
            tmp["recon_mean"] = 0.5 * (recon[s] + recon[t])
            Z = control_matrix(df.reset_index(drop=True))
            Z2 = np.column_stack([Z, tmp.recon_mean.to_numpy(float)])
            # rank ICC (two-way consistency on ranks)
            R = np.column_stack([rs, rt])
            ms_rows = float(np.var(R.mean(axis=1), ddof=1) * 2)
            ms_err = float(np.mean(np.var(R, axis=1, ddof=1)))
            icc = (ms_rows - ms_err) / (ms_rows + ms_err) if (ms_rows + ms_err) > 0 else float("nan")
            pairs.append(
                {
                    "pair": f"{s}-{t}",
                    "seed_a": int(s),
                    "seed_b": int(t),
                    "rho_CH": spearman_safe(C[s], C[t]),
                    "median_cos_HS": float(np.median(cos)),
                    "median_rel_norm": float(np.median(rel)),
                    "rank_icc": float(icc),
                    "top_quartile_jaccard": jacc,
                    "rho_recon": spearman_safe(recon[s], recon[t]),
                    "rho_CH_ctl_radius_recon": associate(C[s], C[t], Z2)["controlled"],
                }
            )
    rho_med = float(np.median([p["rho_CH"] for p in pairs])) if pairs else float("nan")
    cos_med = float(np.median([p["median_cos_HS"] for p in pairs])) if pairs else float("nan")
    passed = bool(np.isfinite(rho_med) and rho_med >= SEED_RHO_GATE and np.isfinite(cos_med) and cos_med >= SEED_COS_GATE)
    consensus_rank = np.median(np.column_stack([rankdata(C[s]) for s in seeds]), axis=1) if passed else None
    consensus_mag = np.median(np.column_stack([C[s] for s in seeds]), axis=1) if passed else None
    return {
        "pairs": pairs,
        "median_rho_CH": rho_med,
        "median_cos_HS": cos_med,
        "passed": passed,
        "n_seeds": int(len(seeds)),
        "gate_rho": SEED_RHO_GATE,
        "gate_cos": SEED_COS_GATE,
        "best_seed_selected": False,
        "consensus_is_median_rank": bool(passed),
        "consensus_rank": consensus_rank,
        "consensus_mag": consensus_mag,
    }


def numerical_eligible(per_seed: dict[int, dict[str, np.ndarray]], seeds: tuple[int, ...]) -> dict[str, Any]:
    fracs = []
    ranks = []
    for s in seeds:
        fin = np.asarray(per_seed[s]["finite"], dtype=bool)
        fracs.append(float(fin.mean()))
        ranks.append(float(np.median(per_seed[s]["jac_rank"])))
    return {
        "min_finite_frac": float(min(fracs)) if fracs else 0.0,
        "median_jac_rank": float(np.median(ranks)) if ranks else float("nan"),
        "ok": bool(fracs and min(fracs) >= 0.95 and abs(float(np.median(ranks)) - 16.0) < 0.51),
    }


def model_associations(df: pd.DataFrame, xcol: str, *, n_perm: int, n_boot: int, seed: int = INFER_SEED) -> dict[str, Any]:
    Z = control_matrix(df.reset_index(drop=True))
    x = df[xcol].to_numpy(float)
    specs = {
        "C_R2": "r2_G",
        "C_P": "r2_P",
        "C_A": "delta_adapt",
        "C_G": "mse_G",
        "C_MSE_P": "mse_P",
        "C_MAE_G": "mae_G",
        "C_MAE_P": "mae_P",
    }
    point = {k: associate(x, df[c].to_numpy(float), Z) for k, c in specs.items() if c in df.columns}
    rng = np.random.default_rng(seed)
    n = len(df)
    boot = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        for k, c in specs.items():
            if k in point:
                boot[k][b] = _assoc(sub, xcol, c)["controlled"]
    ymap = {k: df[c].to_numpy(float) for k, c in specs.items() if k in point}
    null = {k: np.empty(n_perm) for k in point}
    for b in range(n_perm):
        xp = freedman_lane_y(x, Z, rng)
        for k in point:
            null[k][b] = associate(xp, ymap[k], Z)["controlled"]
    out = {}
    for k, rec in point.items():
        obs = float(rec["controlled"])
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        b_count = int(np.sum(np.abs(null[k]) >= abs(obs))) if np.isfinite(obs) else n_perm
        out[k] = {
            "observed": obs,
            "raw": float(rec["raw"]),
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "n": int(rec["n"]),
            "n_perm": int(n_perm),
            "n_boot": int(n_boot),
        }
    out["delta_C_PG"] = {
        "observed": float(out["C_P"]["observed"] - out["C_R2"]["observed"]),
        "ci95": [
            float(np.nanpercentile(boot["C_P"] - boot["C_R2"], 2.5)),
            float(np.nanpercentile(boot["C_P"] - boot["C_R2"], 97.5)),
        ],
    }
    # practical equivalence for global R2
    lo, hi = out["C_R2"]["ci95"]
    out["C_R2_equivalence"] = {
        "interval": [-EQUIV_ABS, EQUIV_ABS],
        "ci95": [lo, hi],
        "inside": bool(lo > -EQUIV_ABS and hi < EQUIV_ABS),
        "sign": "negative" if hi < 0 else ("positive" if lo > 0 else "null_or_mixed"),
    }
    return out


def synchronized_h123(
    frames: dict[str, pd.DataFrame],
    xcol: str,
    *,
    n_perm: int,
    n_boot: int,
    seed: int = INFER_SEED,
) -> dict[str, Any]:
    models = list(frames)
    M = len(models)
    n = len(next(iter(frames.values())))
    sids = next(iter(frames.values())).sample_id.to_numpy(int)
    for df in frames.values():
        if not np.array_equal(df.sample_id.to_numpy(int), sids):
            raise RuntimeError("synchronized tests require identical sample_id order")
    Zs = {m: control_matrix(frames[m].reset_index(drop=True)) for m in models}
    xs = {m: frames[m][xcol].to_numpy(float) for m in models}

    def stats(xmap):
        c_p, c_a, dlt = [], [], []
        for m in models:
            df = frames[m]
            Z = Zs[m]
            rp = associate(xmap[m], df.r2_P.to_numpy(float), Z)["controlled"]
            rg = associate(xmap[m], df.r2_G.to_numpy(float), Z)["controlled"]
            ra = associate(xmap[m], df.delta_adapt.to_numpy(float), Z)["controlled"]
            c_p.append(rp)
            c_a.append(ra)
            dlt.append(rp - rg)
        return np.mean(c_p), np.mean(c_a), np.mean(dlt)

    obs = stats(xs)
    rng = np.random.default_rng(seed + 3)
    boot = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        c_p, c_a, dlt = [], [], []
        for m in models:
            df = frames[m].iloc[idx].reset_index(drop=True)
            Z = control_matrix(df)
            x = df[xcol].to_numpy(float)
            rp = associate(x, df.r2_P.to_numpy(float), Z)["controlled"]
            rg = associate(x, df.r2_G.to_numpy(float), Z)["controlled"]
            ra = associate(x, df.delta_adapt.to_numpy(float), Z)["controlled"]
            c_p.append(rp)
            c_a.append(ra)
            dlt.append(rp - rg)
        boot[b] = (np.mean(c_p), np.mean(c_a), np.mean(dlt))
    null = np.empty((n_perm, 3))
    for b in range(n_perm):
        perm = rng.permutation(n)
        xmap = {}
        for m in models:
            y = xs[m]
            Z = Zs[m]
            msk = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
            y2 = y.copy()
            yr = rankdata(y[msk]).astype(np.float64)
            Zr = np.column_stack([rankdata(Z[msk, j]) for j in range(Z.shape[1])])
            A = np.column_stack([np.ones(int(msk.sum())), Zr])
            bhat, *_ = np.linalg.lstsq(A, yr, rcond=None)
            fit = A @ bhat
            resid = yr - fit
            idx = np.where(msk)[0]
            y2[idx] = fit + resid[np.argsort(perm[idx])]
            xmap[m] = y2
        null[b] = stats(xmap)

    names = ("H1_bar_C_P", "H2_bar_C_A", "H3_bar_delta_C_PG")
    fam = {}
    ps = []
    for i, name in enumerate(names):
        o = float(obs[i])
        lo, hi = np.nanpercentile(boot[:, i], [2.5, 97.5])
        # preregistered negative direction
        b_count = int(np.sum(null[:, i] <= o)) if np.isfinite(o) else n_perm
        rec = {
            "name": name,
            "observed": o,
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "side": "less",
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "n_models": int(M),
            "n_perm": int(n_perm),
            "n_boot": int(n_boot),
            "equal_model_weight": True,
        }
        fam[name] = rec
        ps.append(rec["p_mc"])
    hs = holm(np.asarray(ps))
    for i, name in enumerate(names):
        fam[name]["p_holm"] = float(hs[i])
        fam[name]["pass_holm"] = bool(fam[name]["observed"] < 0 and hs[i] <= 0.05)
    fam["models"] = models
    return fam


def heterogeneity(assocs: dict[str, dict], key: str) -> dict[str, Any]:
    rhos = np.array([assocs[m][key]["observed"] for m in assocs], dtype=float)
    names = list(assocs)
    z = np.arctanh(np.clip(rhos, -0.999, 0.999))
    zbar = float(np.mean(z))
    Q = float(np.sum((z - zbar) ** 2))
    df = max(len(rhos) - 1, 1)
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    loo = []
    for i, m in enumerate(names):
        keep = np.delete(rhos, i)
        loo.append({"left_out": m, "mean": float(np.mean(keep)), "median": float(np.median(keep))})
    return {
        "key": key,
        "models": names,
        "rhos": rhos.tolist(),
        "mean": float(np.mean(rhos)),
        "median": float(np.median(rhos)),
        "sign_frac_neg": float(np.mean(rhos < 0)),
        "sign_frac_pos": float(np.mean(rhos > 0)),
        "Q": Q,
        "I2": I2,
        "leave_one_out": loo,
    }


def d_vs_q(df: pd.DataFrame, xcol: str) -> list[dict[str, Any]]:
    rows = []
    geo_ctrls = [c for c in ("log_knn_radius", "recon", "cond_g") if c in df.columns]
    for qcol in ("K_H_cross", "K_dir_cross"):
        if qcol not in df.columns:
            continue
        raw = associate(df[xcol].to_numpy(float), df[qcol].to_numpy(float), None)
        Z_r = df[["log_knn_radius"]].to_numpy(float)
        rad = associate(df[xcol].to_numpy(float), df[qcol].to_numpy(float), Z_r)
        Z_g = np.column_stack([df[c].to_numpy(float) for c in geo_ctrls]) if geo_ctrls else None
        geo = associate(df[xcol].to_numpy(float), df[qcol].to_numpy(float), Z_g)
        rows.append({"contrast": f"{xcol}_vs_{qcol}", "raw": raw["raw"], "radius_partial": rad["controlled"], "geometry_partial": geo["controlled"]})
    extra_ch = [xcol]
    extra_kh = ["K_H_cross"] if "K_H_cross" in df.columns else []
    for ycol, lab in (("r2_G", "R2G"), ("r2_P", "R2P"), ("delta_adapt", "DeltaAdapt")):
        if extra_kh:
            rec = _assoc(df, "K_H_cross", ycol, extra_controls=extra_ch)
            rec["name"] = f"rho_ctl_KH_{lab}_given_CH"
            rows.append(rec)
        rec2 = _assoc(df, xcol, ycol, extra_controls=extra_kh)
        rec2["name"] = f"rho_ctl_CH_{lab}_given_KH"
        rows.append(rec2)
    return rows
