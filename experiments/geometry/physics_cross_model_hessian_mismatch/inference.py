"""Controlled associations, P1/P2, leave-one-out."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from geometry.physics_curvature_probe_rank_sweep.inference import associate
from geometry.physics_task_aligned_curvature.io_util import p_mc

from .config import INFER_SEED, PAPER_CONTROLS, PRIMARY_CONTROLS


def Z_of(df: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([df[c].to_numpy(float) for c in cols])


def controlled(df: pd.DataFrame, xcol: str, ycol: str, cols: tuple[str, ...] = PRIMARY_CONTROLS) -> dict[str, float]:
    return associate(df[xcol].to_numpy(float), df[ycol].to_numpy(float), Z_of(df, cols))


def vif_and_cond(df: pd.DataFrame, cols: tuple[str, ...] = PRIMARY_CONTROLS) -> dict[str, Any]:
    Z = Z_of(df, cols)
    Zr = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    Zr = (Zr - Zr.mean(0)) / np.maximum(Zr.std(0), 1e-12)
    G = Zr.T @ Zr / max(len(Zr) - 1, 1)
    cond = float(np.linalg.cond(G))
    try:
        inv = np.linalg.inv(G)
        vif = {c: float(inv[i, i]) for i, c in enumerate(cols)}
    except np.linalg.LinAlgError:
        vif = {c: float("nan") for c in cols}
    return {"cond": cond, "vif": vif, "n": int(len(df))}


def cell_assoc(df: pd.DataFrame, xcol: str, ycol: str, cols: tuple[str, ...], *, n_perm: int, n_boot: int, side: str, seed: int) -> dict[str, Any]:
    obs = controlled(df, xcol, ycol, cols)
    rng = np.random.default_rng(seed)
    n = len(df)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[b] = controlled(df.iloc[idx].reset_index(drop=True), xcol, ycol, cols)["controlled"]
    null = np.empty(n_perm)
    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)
    Z = Z_of(df, cols)
    for b in range(n_perm):
        msk = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        y2 = y.copy()
        yr = rankdata(y[msk]).astype(np.float64)
        Zr = np.column_stack([rankdata(Z[msk, j]) for j in range(Z.shape[1])])
        A = np.column_stack([np.ones(int(msk.sum())), Zr])
        bhat, *_ = np.linalg.lstsq(A, yr, rcond=None)
        fit = A @ bhat
        resid = yr - fit
        idx = np.where(msk)[0]
        perm = rng.permutation(int(msk.sum()))
        y2[idx] = fit + resid[perm]
        null[b] = associate(x, y2, Z)["controlled"]
    o = float(obs["controlled"])
    if side == "greater":
        b_count = int(np.sum(null >= o)) if np.isfinite(o) else n_perm
    else:
        b_count = int(np.sum(null <= o)) if np.isfinite(o) else n_perm
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "observed": o,
        "raw": float(obs["raw"]),
        "ci95": [float(lo), float(hi)],
        "p_mc": p_mc(b_count, n_perm),
        "side": side,
        "n": int(obs["n"]),
    }


def aggregate_mean(cells: dict[tuple[str, str], float]) -> float:
    return float(np.mean(list(cells.values()))) if cells else float("nan")


def synchronized_p1_p2(
    frames: dict[tuple[str, str], pd.DataFrame],
    *,
    n_perm: int,
    n_boot: int,
    seed: int = INFER_SEED,
    cols: tuple[str, ...] = PRIMARY_CONTROLS,
) -> dict[str, Any]:
    keys = list(frames)
    ref = frames[keys[0]]
    n = len(ref)
    sids = ref.sample_id.to_numpy(int)
    for df in frames.values():
        if not np.array_equal(df.sample_id.to_numpy(int), sids):
            raise RuntimeError("sample_id order mismatch")

    def stats(idx=None, ymap=None):
        rd, ra = [], []
        per = {}
        for (m, t), df in frames.items():
            sub = df if idx is None else df.iloc[idx].reset_index(drop=True)
            y = sub["mse_G"].to_numpy(float) if ymap is None else (ymap[(m, t)] if idx is None else ymap[(m, t)][idx])
            Z = Z_of(sub, cols)
            md = associate(sub["M_Delta"].to_numpy(float), y, Z)["controlled"]
            aa = associate(sub["A_full"].to_numpy(float), y, Z)["controlled"]
            rd.append(md)
            ra.append(aa)
            per[(m, t)] = (md, aa)
        models = sorted({k[0] for k in per})
        targets = sorted({k[1] for k in per})
        # equal target weight then equal model weight
        p1s, p2s = [], []
        for m in models:
            p1s.append(float(np.mean([per[(m, t)][0] for t in targets])))
            p2s.append(float(np.mean([per[(m, t)][1] for t in targets])))
        return float(np.mean(p1s)), float(np.mean(p2s)), per, p1s, p2s, models

    o1, o2, per, p1s, p2s, models = stats()
    rng = np.random.default_rng(seed + 3)
    b1, b2 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        b1[b], b2[b], *_ = stats(idx=idx)
    n1, n2 = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(n)
        ymap = {}
        for (m, t), df in frames.items():
            y = df["mse_G"].to_numpy(float)
            Z = Z_of(df, cols)
            msk = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
            y2 = y.copy()
            yr = rankdata(y[msk]).astype(np.float64)
            Zr = np.column_stack([rankdata(Z[msk, j]) for j in range(Z.shape[1])])
            A = np.column_stack([np.ones(int(msk.sum())), Zr])
            bhat, *_ = np.linalg.lstsq(A, yr, rcond=None)
            fit = A @ bhat
            resid = yr - fit
            ii = np.where(msk)[0]
            y2[ii] = fit + resid[np.argsort(perm[ii])]
            ymap[(m, t)] = y2
        n1[b], n2[b], *_ = stats(ymap=ymap)
    def pack(obs, null, boot, side, name, per_model, models_):
        if side == "greater":
            bc = int(np.sum(null >= obs)) if np.isfinite(obs) else len(null)
        else:
            bc = int(np.sum(null <= obs)) if np.isfinite(obs) else len(null)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        return {
            "name": name,
            "observed": float(obs),
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(bc, len(null)),
            "side": side,
            "per_model": {m: float(per_model[i]) for i, m in enumerate(models_)},
            "equal_model_weight": True,
            "equal_target_weight": True,
        }

    p1 = pack(o1, n1, b1, "greater", "P1_bar_rho_Delta", p1s, models)
    p2 = pack(o2, n2, b2, "less", "P2_bar_rho_A", p2s, models)
    p1["per_cell"] = {f"{m}:{t}": float(per[(m, t)][0]) for m, t in per}
    p2["per_cell"] = {f"{m}:{t}": float(per[(m, t)][1]) for m, t in per}
    return {"P1": p1, "P2": p2}


def leave_one_out(per_cell: dict[str, float], *, axis: str) -> pd.DataFrame:
    rows = []
    items = [(k.split(":", 1)[0], k.split(":", 1)[1], v) for k, v in per_cell.items()]
    models = sorted({m for m, _, _ in items})
    targets = sorted({t for _, t, _ in items})
    if axis == "model":
        for drop in models:
            keep = [v for m, t, v in items if m != drop]
            rows.append({"dropped": drop, "mean": float(np.mean(keep)), "n": len(keep)})
    else:
        for drop in targets:
            keep = [v for m, t, v in items if t != drop]
            rows.append({"dropped": drop, "mean": float(np.mean(keep)), "n": len(keep)})
    return pd.DataFrame(rows)
