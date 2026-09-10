"""Equal-weight and Fisher-z aggregates with joint-anchor resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import control_matrix, freedman_lane_y

from .inference import _assoc
from .io_util import p_mc


def _aligned(by_model: dict[str, pd.DataFrame]) -> tuple[list[str], list[int], dict[str, pd.DataFrame]]:
    models = list(by_model)
    common = None
    for df in by_model.values():
        s = set(df.sample_id.astype(int).tolist())
        common = s if common is None else (common & s)
    sids = sorted(common)
    aligned = {m: by_model[m].set_index("sample_id").loc[sids].reset_index() for m in models}
    return models, sids, aligned


def fisher_z_mean(rhos: list[float]) -> dict[str, float]:
    arr = np.clip(np.asarray(rhos, dtype=float), -0.999999, 0.999999)
    z = np.arctanh(arr)
    return {
        "equal_weight_mean": float(np.mean(arr)),
        "fisher_z_mean": float(np.mean(z)),
        "fisher_z_corr": float(np.tanh(np.mean(z))),
    }


def _stats(aligned: dict[str, pd.DataFrame], xcol: str) -> dict[str, Any]:
    per: dict[str, list[float]] = {k: [] for k in ("C_G", "C_P", "C_A", "C_R2", "C_R2P", "A")}
    models = list(aligned)
    for df in aligned.values():
        cg = float(_assoc(df, xcol, "mse_G")["controlled"])
        cp = float(_assoc(df, xcol, "mse_P")["controlled"])
        ca = float(_assoc(df, xcol, "delta_adapt")["controlled"])
        cr2 = float(_assoc(df, xcol, "r2_G")["controlled"])
        cr2p = float(_assoc(df, xcol, "r2_P")["controlled"])
        per["C_G"].append(cg)
        per["C_P"].append(cp)
        per["C_A"].append(ca)
        per["C_R2"].append(cr2)
        per["C_R2P"].append(cr2p)
        per["A"].append(cg - cp)
    out: dict[str, Any] = {"models": models, "per_model": per}
    for k, vals in per.items():
        fz = fisher_z_mean(vals)
        out[f"{k}_bar"] = fz["equal_weight_mean"]
        out[f"{k}_fisher"] = fz
        out[f"n_neg_{k}"] = int(sum(v < 0 for v in vals))
        out[f"n_pos_{k}"] = int(sum(v > 0 for v in vals))
    return out


def _leave_one_out(aligned: dict[str, pd.DataFrame], xcol: str) -> dict[str, Any]:
    loo = {}
    models = list(aligned)
    for leave in models:
        sub = {m: df for m, df in aligned.items() if m != leave}
        if len(sub) < 2:
            continue
        st = _stats(sub, xcol)
        loo[leave] = {k: st[k] for k in st if k.endswith("_bar")}
    return loo


def synchronized_inference(
    by_model: dict[str, pd.DataFrame],
    xcol: str,
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    models, sids, aligned = _aligned(by_model)
    obs = _stats(aligned, xcol)
    obs["leave_one_encoder_out"] = _leave_one_out(aligned, xcol)
    rng = np.random.default_rng(seed)
    n = len(sids)
    keys = ("C_G_bar", "C_P_bar", "C_A_bar", "C_R2_bar", "C_R2P_bar", "A_bar")
    boot = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = {m: df.iloc[idx].reset_index(drop=True) for m, df in aligned.items()}
        st = _stats(sub, xcol)
        for k in keys:
            boot[k][b] = st[k]

    null = {k: np.empty(n_perm) for k in keys}
    Zs = {m: control_matrix(df.reset_index(drop=True)) for m, df in aligned.items()}
    xs = {m: df[xcol].to_numpy(float) for m, df in aligned.items()}
    for b in range(n_perm):
        perm = rng.permutation(n)
        sub = {}
        for m, df in aligned.items():
            xp = freedman_lane_y(xs[m], Zs[m], rng)
            tmp = df.copy()
            tmp[xcol] = xp[perm]
            sub[m] = tmp
        st = _stats(sub, xcol)
        for k in keys:
            null[k][b] = st[k]

    expect = {
        "C_G_bar": "pos",
        "C_P_bar": "neg",
        "C_A_bar": "pos",
        "C_R2_bar": "neg",
        "C_R2P_bar": "pos",
        "A_bar": "pos",
    }
    out: dict[str, Any] = {
        "models": models,
        "n_common_anchors": n,
        "xcol": xcol,
        "observed": {k: obs[k] for k in keys},
        "per_model": obs["per_model"],
        "fisher": {k: obs[f"{k[:-4]}_fisher"] for k in keys},
        "sign": {k: {"n_neg": obs[f"n_neg_{k[:-4]}"], "n_pos": obs[f"n_pos_{k[:-4]}"]} for k in keys},
        "leave_one_encoder_out": obs["leave_one_encoder_out"],
        "n_perm": n_perm,
        "n_boot": n_boot,
    }
    for k, exp in expect.items():
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        obs_v = float(obs[k])
        nt = null[k]
        if exp == "pos":
            b_count = int(np.sum(nt >= obs_v))
        elif exp == "neg":
            b_count = int(np.sum(nt <= obs_v))
        else:
            b_count = int(np.sum(np.abs(nt) >= abs(obs_v)))
        out[k] = {
            "observed": obs_v,
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "expected_sign": exp,
        }
    return out
