"""Controlled associations, synchronized P1/P2, seed gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from geometry.physics_curvature_probe_rank_sweep.inference import associate, freedman_lane_y

from .algebra import holm
from .config import CONTROLS, INFER_SEED
from .io_util import p_mc


def spearman_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def control_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df[c].to_numpy(float) for c in CONTROLS])


def controlled(df: pd.DataFrame, xcol: str, ycol: str) -> dict[str, float]:
    Z = control_matrix(df)
    return associate(df[xcol].to_numpy(float), df[ycol].to_numpy(float), Z)


def seed_reliability_E(per_seed: dict[int, np.ndarray], b_seed: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    seeds = tuple(sorted(per_seed))
    pairs = []
    for i, s in enumerate(seeds):
        for t in seeds[i + 1 :]:
            rec = {"pair": f"{s}-{t}", "rho_E": spearman_safe(per_seed[s], per_seed[t])}
            if b_seed is not None:
                ba, bb = b_seed[s], b_seed[t]
                cos = []
                for k in range(len(ba)):
                    u, v = ba[k].ravel(), bb[k].ravel()
                    den = float(np.linalg.norm(u) * np.linalg.norm(v))
                    cos.append(float(np.dot(u, v) / den) if den > 1e-15 else float("nan"))
                rec["median_cos_b"] = float(np.nanmedian(cos))
            pairs.append(rec)
    med = float(np.nanmedian([p["rho_E"] for p in pairs])) if pairs else float("nan")
    passed = bool(np.isfinite(med) and med >= 0.70)
    cons = np.median(np.column_stack([rankdata(per_seed[s]) for s in seeds]), axis=1) if passed else None
    return {
        "pairs": pairs,
        "median_rho_E": med,
        "passed": passed,
        "best_seed_selected": False,
        "consensus_rank": cons,
        "n_seeds": int(len(seeds)),
    }


def per_target_assoc(df: pd.DataFrame, xcol: str, ycol: str, *, n_perm: int, n_boot: int, seed: int = INFER_SEED) -> dict[str, Any]:
    Z = control_matrix(df)
    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)
    obs = associate(x, y, Z)
    rng = np.random.default_rng(seed)
    n = len(df)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        boot[b] = controlled(sub, xcol, ycol)["controlled"]
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = freedman_lane_y(y, Z, rng)
        null[b] = associate(x, yp, Z)["controlled"]
    o = float(obs["controlled"])
    # one-sided positive
    b_count = int(np.sum(null >= o)) if np.isfinite(o) else n_perm
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "observed": o,
        "raw": float(obs["raw"]),
        "ci95": [float(lo), float(hi)],
        "p_mc": p_mc(b_count, n_perm),
        "side": "greater",
        "n": int(obs["n"]),
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
    }


def synchronized_aggregate(
    frames: dict[str, pd.DataFrame],
    xcol: str,
    ycol: str,
    *,
    n_perm: int,
    n_boot: int,
    seed: int = INFER_SEED,
) -> dict[str, Any]:
    targets = list(frames)
    n = len(next(iter(frames.values())))
    sids = next(iter(frames.values())).sample_id.to_numpy(int)
    for df in frames.values():
        if not np.array_equal(df.sample_id.to_numpy(int), sids):
            raise RuntimeError("synchronized tests require identical sample_id order")

    def stat(xmap=None):
        rhos = []
        for t in targets:
            df = frames[t]
            x = df[xcol].to_numpy(float) if xmap is None else xmap[t]
            Z = control_matrix(df)
            rhos.append(associate(x, df[ycol].to_numpy(float), Z)["controlled"])
        return float(np.mean(rhos)), np.asarray(rhos)

    obs_mean, obs_rhos = stat()
    rng = np.random.default_rng(seed + 11)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rhos = []
        for t in targets:
            sub = frames[t].iloc[idx].reset_index(drop=True)
            rhos.append(controlled(sub, xcol, ycol)["controlled"])
        boot[b] = float(np.mean(rhos))
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(n)
        rhos = []
        for t in targets:
            df = frames[t]
            y = df[ycol].to_numpy(float)
            Z = control_matrix(df)
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
            rhos.append(associate(df[xcol].to_numpy(float), y2, Z)["controlled"])
        null[b] = float(np.mean(rhos))
    b_count = int(np.sum(null >= obs_mean)) if np.isfinite(obs_mean) else n_perm
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "observed": float(obs_mean),
        "per_target": {t: float(obs_rhos[i]) for i, t in enumerate(targets)},
        "all_predicted_sign": bool(np.all(obs_rhos > 0)),
        "ci95": [float(lo), float(hi)],
        "p_mc": p_mc(b_count, n_perm),
        "side": "greater",
        "n_targets": int(len(targets)),
        "equal_target_weight": True,
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
    }


def dep_bootstrap_mse(
    y: np.ndarray,
    yhat: np.ndarray,
    neigh: np.ndarray,
    eval_mask: np.ndarray,
    E: np.ndarray,
    controls: np.ndarray,
    *,
    n_boot: int,
    min_eval: int,
    seed: int = INFER_SEED,
) -> np.ndarray:
    """Resample evaluation objects, recompute local MSE, then controlled ρ(E, MSE)."""
    rng = np.random.default_rng(seed + 29)
    eval_idx = np.where(eval_mask & np.isfinite(y) & np.isfinite(yhat))[0]
    out = np.full(n_boot, np.nan)
    n_a = len(neigh)
    for b in range(n_boot):
        draw = rng.choice(eval_idx, size=len(eval_idx), replace=True)
        # membership count per object among the resampled eval bag
        # use a hash set of drawn indices (with replacement → treat as new eval population)
        drawn_set = draw  # local_risk analogue with replacement via repeats
        mse = np.full(n_a, np.nan)
        var = np.full(n_a, np.nan)
        nct = np.zeros(n_a)
        for i in range(n_a):
            nbr = neigh[i]
            # keep neighbours that appear in the resampled eval bag
            keep = np.intersect1d(nbr, drawn_set, assume_unique=False)
            if len(keep) < min_eval:
                continue
            yi, yh = y[keep], yhat[keep]
            mse[i] = float(np.mean((yi - yh) ** 2))
            var[i] = float(np.var(yi, ddof=1)) if len(keep) > 1 else np.nan
            nct[i] = len(keep)
        m = np.isfinite(E) & np.isfinite(mse) & np.isfinite(controls).all(axis=1)
        if int(m.sum()) < 12:
            continue
        Z = np.column_stack([controls[m, 0], var[m], nct[m]])
        out[b] = associate(E[m], mse[m], Z)["controlled"]
    return out
