"""Controlled associations, Freedman–Lane primary inference, decision label."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import (
    CONTROLS,
    associate,
    control_matrix,
    freedman_lane_y,
)

from .config import DECISION_LABELS, SECONDARY_FAMILY
from .io_util import p_mc


def _assoc_df(df: pd.DataFrame, ycol: str) -> dict[str, float]:
    sub = df.reset_index(drop=True).copy()
    for c in CONTROLS:
        if c not in sub.columns:
            sub[c] = np.nan
    Z = control_matrix(sub)
    return associate(sub["K_H_cross"].to_numpy(float), sub[ycol].to_numpy(float), Z)


def primary_inference(df: pd.DataFrame, *, n_perm: int, n_boot: int, seed: int) -> dict[str, Any]:
    ycol = "dMSE_G_to_P"
    obs = _assoc_df(df, ycol)
    Z = control_matrix(df.reset_index(drop=True))
    x = df["K_H_cross"].to_numpy(float)
    y = df[ycol].to_numpy(float)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = freedman_lane_y(y, Z, rng)
        null[b] = associate(x, yp, Z)["controlled"]
    b_count = int(np.sum(np.abs(null) >= abs(obs["controlled"]))) if np.isfinite(obs["controlled"]) else n_perm
    boot = np.empty(n_boot)
    sids = df.sample_id.to_numpy()
    for b in range(n_boot):
        draw = rng.choice(len(df), size=len(df), replace=True)
        sub = df.iloc[draw].reset_index(drop=True)
        boot[b] = _assoc_df(sub, ycol)["controlled"]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "endpoint": ycol,
        "observed": obs,
        "ci95": [float(lo), float(hi)],
        "p_mc": p_mc(b_count, n_perm),
        "n_perm": n_perm,
        "n_boot": n_boot,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def secondary_table(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("rho_KH_R2_P", "r2_P"),
        ("rho_KH_MSE_P", "mse_P"),
        ("rho_KH_dMAE_GP", "dMAE_G_to_P"),
        ("rho_KH_dR2_GP", "dR2_G_to_P"),
        ("rho_KH_dMSE_CP", "dMSE_C_to_P"),
        ("rho_KH_MSE_G", "mse_G"),
        ("rho_KH_MSE_C", "mse_C"),
        ("rho_KH_dMSE_G_to_T", "dMSE_G_to_T"),
        ("rho_KH_dMSE_C_to_T", "dMSE_C_to_T"),
        ("rho_KH_SST", "sst"),
        ("rho_KH_var", "var"),
        ("rho_KH_alpha", "selected_alpha"),
    ]
    rows = []
    for name, col in specs:
        if col not in df.columns:
            continue
        rec = _assoc_df(df, col)
        rec["name"] = name
        rec["ycol"] = col
        rows.append(rec)
    out = pd.DataFrame(rows)
    # Holm over predeclared secondary family only
    mask = out.name.isin(SECONDARY_FAMILY)
    if mask.any():
        ps = out.loc[mask, "p_ctl"].to_numpy(float)
        order = np.argsort(ps)
        holm = np.empty_like(ps)
        m = len(ps)
        for rank, i in enumerate(order):
            holm[i] = min(1.0, (m - rank) * ps[i])
        out.loc[mask, "p_holm"] = holm
    return out


def decide(
    primary: dict,
    sec: pd.DataFrame,
    *,
    parity_ok: bool,
    oof_ok: bool,
    shuffle_pass: bool,
    outer_half_pass: bool,
    hist_insample_only: bool,
) -> dict[str, Any]:
    rho = float(primary["observed"]["controlled"])
    ci_ok = bool(primary.get("ci_excludes_zero") and rho > 0)
    p_ok = float(primary["p_mc"]) <= 0.05

    def get(name):
        hit = sec[sec.name == name]
        return float(hit.iloc[0]["controlled"]) if len(hit) else float("nan")

    mse_g = get("rho_KH_MSE_G")
    mse_p = get("rho_KH_MSE_P")
    d_cp = get("rho_KH_dMSE_CP")
    d_mae = get("rho_KH_dMAE_GP")
    mse_weaker = np.isfinite(mse_g) and np.isfinite(mse_p) and (mse_p < mse_g - 0.02)
    mae_agrees = np.isfinite(d_mae) and d_mae > 0
    hit_cp = sec[sec.name == "rho_KH_dMSE_CP"]
    beats_c = bool(
        np.isfinite(d_cp)
        and d_cp > 0
        and len(hit_cp)
        and float(hit_cp.iloc[0].get("p_holm", hit_cp.iloc[0].get("p_ctl", 1.0)) or 1.0) <= 0.05
    )

    if hist_insample_only and not (ci_ok and p_ok):
        label = "in_sample_patch_probe_artifact"
    elif not (parity_ok and oof_ok):
        label = "local_probe_result_unresolved"
    elif ci_ok and p_ok and mae_agrees and mse_weaker and beats_c and shuffle_pass and outer_half_pass:
        label = "curvature_predicts_local_direction_adaptation"
    elif ci_ok and p_ok and mae_agrees and (not beats_c) and shuffle_pass:
        label = "curvature_predicts_local_calibration_gain"
    else:
        label = "local_probe_result_unresolved"
    assert label in DECISION_LABELS
    return {
        "label": label,
        "primary_rho": rho,
        "primary_p_mc": primary["p_mc"],
        "mse_G_assoc": mse_g,
        "mse_P_assoc": mse_p,
        "dMSE_C_to_P_assoc": d_cp,
        "checks": {
            "parity_ok": parity_ok,
            "oof_ok": oof_ok,
            "ci_ok": ci_ok,
            "p_ok": p_ok,
            "mae_agrees": mae_agrees,
            "mse_weaker_for_P": mse_weaker,
            "beats_C": beats_c,
            "shuffle_pass": shuffle_pass,
            "outer_half_pass": outer_half_pass,
        },
    }


def manuscript_action(label: str) -> dict[str, str]:
    if label == "curvature_predicts_local_direction_adaptation":
        return {
            "action": "include_as_main_result",
            "paragraph": (
                "Curvature predicts the advantage of locally adapted probes over a fixed global probe, "
                "consistent with locally accessible predictive directions rotating across the representation."
            ),
        }
    if label == "curvature_predicts_local_calibration_gain":
        return {
            "action": "include_as_exploratory_appendix",
            "paragraph": (
                "Curvature predicts gains from local intercept/rescaling of the global probe, but not a "
                "clear additional benefit from patch-specific representation directions."
            ),
        }
    if label == "in_sample_patch_probe_artifact":
        return {
            "action": "do_not_include",
            "paragraph": "The accidental positive association does not survive genuine OOF evaluation.",
        }
    return {
        "action": "do_not_include",
        "paragraph": "Local-probe adaptation vs curvature remains unresolved; leave the accidental observation out.",
    }
