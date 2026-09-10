"""Controlled and component-conditional rank associations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, freedman_lane_y

from .config import CONTROLS
from .io_util import p_mc

OUTCOMES = ("r2_G", "mse_G", "r2_P", "mse_P", "delta_adapt")


def _Z(df: pd.DataFrame, extra: np.ndarray | None = None) -> np.ndarray:
    cols = [df[c].fillna(0).to_numpy(float) for c in CONTROLS]
    if extra is not None:
        cols.append(np.asarray(extra, dtype=float))
    return np.column_stack(cols)


def assoc(df: pd.DataFrame, xcol: str, ycol: str, *, cond: str | None = None) -> dict[str, float]:
    extra = df[cond].to_numpy(float) if cond else None
    return associate(df[xcol].to_numpy(float), df[ycol].to_numpy(float), _Z(df, extra))


def model_component_inference(
    df: pd.DataFrame,
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
    ycol: str = "mse_G",
) -> dict[str, Any]:
    """Holm family: unique K_H and unique K_TF vs ycol."""
    rng = np.random.default_rng(seed)
    n = len(df)
    point = {
        "KH": assoc(df, "K_H_cross", ycol),
        "KTF": assoc(df, "K_TF_cross", ycol),
        "Kdir": assoc(df, "K_dir_cross", ycol),
        "unique_KH": assoc(df, "K_H_cross", ycol, cond="K_TF_cross"),
        "unique_KTF": assoc(df, "K_TF_cross", ycol, cond="K_H_cross"),
    }
    names = list(point)
    boot = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        boot["KH"][b] = assoc(sub, "K_H_cross", ycol)["controlled"]
        boot["KTF"][b] = assoc(sub, "K_TF_cross", ycol)["controlled"]
        boot["Kdir"][b] = assoc(sub, "K_dir_cross", ycol)["controlled"]
        boot["unique_KH"][b] = assoc(sub, "K_H_cross", ycol, cond="K_TF_cross")["controlled"]
        boot["unique_KTF"][b] = assoc(sub, "K_TF_cross", ycol, cond="K_H_cross")["controlled"]

    Z0 = _Z(df)
    kh = df.K_H_cross.to_numpy(float)
    ktf = df.K_TF_cross.to_numpy(float)
    kdir = df.K_dir_cross.to_numpy(float)
    null = {k: np.empty(n_perm) for k in ("unique_KH", "unique_KTF")}
    for b in range(n_perm):
        tmp = df.copy()
        tmp["K_H_cross"] = freedman_lane_y(kh, np.column_stack([Z0, ktf]), rng)
        null["unique_KH"][b] = assoc(tmp, "K_H_cross", ycol, cond="K_TF_cross")["controlled"]
        tmp = df.copy()
        tmp["K_TF_cross"] = freedman_lane_y(ktf, np.column_stack([Z0, kh]), rng)
        null["unique_KTF"][b] = assoc(tmp, "K_TF_cross", ycol, cond="K_H_cross")["controlled"]

    def pack(name: str, two_sided: bool = True) -> dict[str, Any]:
        obs = float(point[name]["controlled"])
        finite = boot[name][np.isfinite(boot[name])]
        if finite.size >= 8:
            lo, hi = np.nanpercentile(finite, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        rec = {
            "observed": obs,
            "raw": float(point[name]["raw"]),
            "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
        if name in null:
            nt = null[name]
            rec["p_mc"] = p_mc(int(np.sum(np.abs(nt) >= abs(obs))), n_perm)
        return rec

    family = {"unique_KH": pack("unique_KH"), "unique_KTF": pack("unique_KTF")}
    ps = [family[k]["p_mc"] for k in ("unique_KH", "unique_KTF")]
    order = np.argsort(ps)
    holm = [1.0, 1.0]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (2 - rank) * float(ps[i])))
        holm[i] = run
    family["unique_KH"]["p_holm"] = float(holm[0])
    family["unique_KTF"]["p_holm"] = float(holm[1])
    return {
        "n": n,
        "ycol": ycol,
        "KH": pack("KH"),
        "KTF": pack("KTF"),
        "Kdir": pack("Kdir"),
        **family,
        "n_perm": n_perm,
        "n_boot": n_boot,
    }


def point_all_outcomes(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    out = {}
    for x in ("K_H_cross", "K_TF_cross", "K_dir_cross"):
        out[x] = {}
        cond = "K_TF_cross" if x == "K_H_cross" else ("K_H_cross" if x == "K_TF_cross" else None)
        for y in OUTCOMES:
            if y not in df.columns:
                continue
            out[x][y] = {
                "ordinary": assoc(df, x, y),
                "unique": assoc(df, x, y, cond=cond) if cond else None,
            }
    return out


def joint_unique(
    by_model: dict[str, pd.DataFrame],
    ycol: str,
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    models = list(by_model)
    common = None
    for df in by_model.values():
        s = set(df.sample_id.astype(int))
        common = s if common is None else common & s
    sids = sorted(common)
    aligned = {m: by_model[m].set_index("sample_id").loc[sids].reset_index() for m in models}

    def stats(al):
        khs, ktfs = [], []
        for df in al.values():
            khs.append(assoc(df, "K_H_cross", ycol, cond="K_TF_cross")["controlled"])
            ktfs.append(assoc(df, "K_TF_cross", ycol, cond="K_H_cross")["controlled"])
        def _fz(vals):
            z = np.arctanh(np.clip(np.asarray(vals, float), -0.999, 0.999))
            return float(np.tanh(np.mean(z)))

        return {
            "unique_KH_bar": float(np.mean(khs)),
            "unique_KTF_bar": float(np.mean(ktfs)),
            "unique_KH_fisher": _fz(khs),
            "unique_KTF_fisher": _fz(ktfs),
            "n_pos_KH": int(sum(v > 0 for v in khs)),
            "n_pos_KTF": int(sum(v > 0 for v in ktfs)),
            "per_KH": khs,
            "per_KTF": ktfs,
        }

    obs = stats(aligned)
    rng = np.random.default_rng(seed)
    n = len(sids)
    boot = {k: np.empty(n_boot) for k in ("unique_KH_bar", "unique_KTF_bar")}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        st = stats({m: df.iloc[idx].reset_index(drop=True) for m, df in aligned.items()})
        boot["unique_KH_bar"][b] = st["unique_KH_bar"]
        boot["unique_KTF_bar"][b] = st["unique_KTF_bar"]
    null = {k: np.empty(n_perm) for k in boot}
    for b in range(n_perm):
        perm = rng.permutation(n)
        sub = {}
        for m, df in aligned.items():
            tmp = df.copy()
            Z = _Z(df, df.K_TF_cross.to_numpy(float))
            tmp["K_H_cross"] = freedman_lane_y(df.K_H_cross.to_numpy(float), Z, rng)[perm]
            Z2 = _Z(df, df.K_H_cross.to_numpy(float))
            tmp["K_TF_cross"] = freedman_lane_y(df.K_TF_cross.to_numpy(float), Z2, rng)[perm]
            sub[m] = tmp
        st = stats(sub)
        null["unique_KH_bar"][b] = st["unique_KH_bar"]
        null["unique_KTF_bar"][b] = st["unique_KTF_bar"]
    loo = {}
    for leave in models:
        st = stats({m: df for m, df in aligned.items() if m != leave})
        loo[leave] = {k: st[k] for k in ("unique_KH_bar", "unique_KTF_bar")}
    out = {
        "models": models,
        "n_common": n,
        "ycol": ycol,
        "observed": {
            k: obs[k]
            for k in ("unique_KH_bar", "unique_KTF_bar", "unique_KH_fisher", "unique_KTF_fisher")
        },
        "per_model": {"unique_KH": obs["per_KH"], "unique_KTF": obs["per_KTF"]},
        "sign": {"n_pos_KH": obs["n_pos_KH"], "n_pos_KTF": obs["n_pos_KTF"]},
        "leave_one_encoder_out": loo,
    }
    for k in ("unique_KH_bar", "unique_KTF_bar"):
        finite = boot[k][np.isfinite(boot[k])]
        if finite.size >= 8:
            lo, hi = np.nanpercentile(finite, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        obs_v = float(obs[k])
        out[k] = {
            "observed": obs_v,
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(int(np.sum(np.abs(null[k]) >= abs(obs_v))), n_perm),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
    ps = [out["unique_KH_bar"]["p_mc"], out["unique_KTF_bar"]["p_mc"]]
    order = np.argsort(ps)
    holm = [1.0, 1.0]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (2 - rank) * float(ps[i])))
        holm[i] = run
    out["unique_KH_bar"]["p_holm"] = float(holm[0])
    out["unique_KTF_bar"]["p_holm"] = float(holm[1])
    return out
