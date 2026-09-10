"""Controlled rank-space associations for every curvature definition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y

from .config import CONTROLS, CURVATURE_COLS, OUTCOME_COLS
from .io_util import p_mc

OUTCOME_SIGN = {
    "r2_G": "neg",
    "mse_G": "pos",
    "r2_P": "pos",
    "mse_P": "neg",
    "delta_adapt": "pos",
}


def _assoc(df: pd.DataFrame, xcol: str, ycol: str) -> dict[str, float]:
    sub = df.reset_index(drop=True).copy()
    for c in CONTROLS:
        if c not in sub.columns:
            sub[c] = np.nan
    Z = control_matrix(sub)
    return associate(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float), Z)


def point_associations(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for xcol in CURVATURE_COLS:
        if xcol not in df.columns:
            continue
        out[xcol] = {}
        for ycol in OUTCOME_COLS:
            if ycol not in df.columns:
                continue
            out[xcol][ycol] = _assoc(df, xcol, ycol)
        if "mse_G" in df.columns and "mse_P" in df.columns:
            cg = out[xcol]["mse_G"]["controlled"]
            cp = out[xcol]["mse_P"]["controlled"]
            out[xcol]["C_G"] = {"controlled": float(cg)}
            out[xcol]["C_P"] = {"controlled": float(cp)}
            out[xcol]["C_A"] = {"controlled": float(out[xcol]["delta_adapt"]["controlled"])}
            out[xcol]["A"] = {"controlled": float(cg - cp)}
    return out


def curvature_vs_curvature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [c for c in CURVATURE_COLS if c in df.columns]
    Z = control_matrix(df.reset_index(drop=True))
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            raw = associate(df[a].to_numpy(float), df[b].to_numpy(float), None)
            ctl = associate(df[a].to_numpy(float), df[b].to_numpy(float), Z)
            rows.append(
                {
                    "x": a,
                    "y": b,
                    "raw": raw["raw"],
                    "controlled": ctl["controlled"],
                    "n": ctl["n"],
                }
            )
    return pd.DataFrame(rows)


def model_inference(
    df: pd.DataFrame,
    xcol: str,
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    Z = control_matrix(df.reset_index(drop=True))
    x = df[xcol].to_numpy(float)
    point = {y: _assoc(df, xcol, y) for y in OUTCOME_COLS if y in df.columns}
    cg = float(point["mse_G"]["controlled"])
    cp = float(point["mse_P"]["controlled"])
    ca = float(point["delta_adapt"]["controlled"])
    names = list(point) + ["A"]
    rng = np.random.default_rng(seed)
    n = len(df)
    boot = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        for y in point:
            boot[y][b] = _assoc(sub, xcol, y)["controlled"]
        boot["A"][b] = boot["mse_G"][b] - boot["mse_P"][b]

    null = {k: np.empty(n_perm) for k in ("mse_G", "mse_P", "delta_adapt", "r2_G", "r2_P", "A")}
    for b in range(n_perm):
        xp = freedman_lane_y(x, Z, rng)
        tmp = df.copy()
        tmp[xcol] = xp
        for y in ("mse_G", "mse_P", "delta_adapt", "r2_G", "r2_P"):
            null[y][b] = _assoc(tmp, xcol, y)["controlled"]
        null["A"][b] = null["mse_G"][b] - null["mse_P"][b]

    def pack(name: str, obs: float, expected: str) -> dict[str, Any]:
        bt = boot[name]
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        nt = null[name]
        if expected == "pos":
            b_count = int(np.sum(nt >= obs))
        elif expected == "neg":
            b_count = int(np.sum(nt <= obs))
        else:
            b_count = int(np.sum(np.abs(nt) >= abs(obs)))
        return {
            "observed": float(obs),
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "expected_sign": expected,
        }

    family = {
        "C_G": pack("mse_G", cg, "pos"),
        "C_P": pack("mse_P", cp, "neg"),
        "C_A": pack("delta_adapt", ca, "pos"),
        "C_R2": pack("r2_G", float(point["r2_G"]["controlled"]), "neg"),
        "C_R2P": pack("r2_P", float(point["r2_P"]["controlled"]), "pos"),
        "A": pack("A", cg - cp, "pos"),
    }
    return {
        "n": int(n),
        "xcol": xcol,
        "point": point,
        **family,
        "n_perm": n_perm,
        "n_boot": n_boot,
        "mean_delta_adapt": float(df.delta_adapt.mean()),
        "median_delta_adapt": float(df.delta_adapt.median()),
        "frac_positive_delta_adapt": float((df.delta_adapt > 0).mean()),
        "patch_worse_on_average": bool(float(df.delta_adapt.mean()) < 0),
    }
