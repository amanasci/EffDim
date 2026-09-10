"""Per-model controlled Spearman inference and paired attenuation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y

from .config import CONTROLS, PRIMARY_FAMILY
from .io_util import p_mc


def _assoc(df: pd.DataFrame, ycol: str) -> dict[str, float]:
    sub = df.reset_index(drop=True).copy()
    for c in CONTROLS:
        if c not in sub.columns:
            sub[c] = np.nan
    Z = control_matrix(sub)
    return associate(sub["K_H_cross"].to_numpy(float), sub[ycol].to_numpy(float), Z)


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps)
    out = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * float(ps[i]))
        running = max(running, val)
        out[i] = running
    return out


def model_primary(df: pd.DataFrame, *, n_perm: int, n_boot: int, seed: int) -> dict[str, Any]:
    """C_G, C_A, C_P, A with paired bootstrap and KH Freedman–Lane permutations."""
    need = ["K_H_cross", "mse_G", "mse_P", "delta_adapt", "r2_G"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"missing {c}")
    Z = control_matrix(df.reset_index(drop=True))
    kh = df.K_H_cross.to_numpy(float)
    cg = _assoc(df, "mse_G")
    cp = _assoc(df, "mse_P")
    ca = _assoc(df, "delta_adapt")
    cr2 = _assoc(df, "r2_G")
    a_obs = float(cg["controlled"] - cp["controlled"])
    rng = np.random.default_rng(seed)
    n = len(df)

    boot = {k: np.empty(n_boot) for k in ("C_G", "C_P", "C_A", "A", "C_R2")}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx].reset_index(drop=True)
        boot["C_G"][b] = _assoc(sub, "mse_G")["controlled"]
        boot["C_P"][b] = _assoc(sub, "mse_P")["controlled"]
        boot["C_A"][b] = _assoc(sub, "delta_adapt")["controlled"]
        boot["C_R2"][b] = _assoc(sub, "r2_G")["controlled"]
        boot["A"][b] = boot["C_G"][b] - boot["C_P"][b]

    # permute KH residuals; G/P/adapt stay paired on the same anchors
    null = {k: np.empty(n_perm) for k in ("C_G", "C_P", "C_A", "A")}
    for b in range(n_perm):
        khp = freedman_lane_y(kh, Z, rng)
        tmp = df.copy()
        tmp["K_H_cross"] = khp
        null["C_G"][b] = _assoc(tmp, "mse_G")["controlled"]
        null["C_P"][b] = _assoc(tmp, "mse_P")["controlled"]
        null["C_A"][b] = _assoc(tmp, "delta_adapt")["controlled"]
        null["A"][b] = null["C_G"][b] - null["C_P"][b]

    def pack(name: str, obs: float, expected_pos: bool) -> dict[str, Any]:
        bt = boot[name]
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        nt = null[name]
        if expected_pos:
            b_count = int(np.sum(nt >= obs))
        else:
            b_count = int(np.sum(np.abs(nt) >= abs(obs)))
        return {
            "observed": float(obs),
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "expected_positive": expected_pos,
        }

    family = {
        "C_G": pack("C_G", float(cg["controlled"]), True),
        "C_A": pack("C_A", float(ca["controlled"]), True),
        "A": pack("A", a_obs, True),
    }
    ps = [family[k]["p_mc"] for k in PRIMARY_FAMILY]
    holm_ps = holm(ps)
    for k, hp in zip(PRIMARY_FAMILY, holm_ps):
        family[k]["p_holm"] = float(hp)

    return {
        "n": int(n),
        "C_G": family["C_G"],
        "C_A": family["C_A"],
        "A": family["A"],
        "C_P": {
            "observed": float(cp["controlled"]),
            "ci95": [float(x) for x in np.nanpercentile(boot["C_P"], [2.5, 97.5])],
        },
        "C_R2": {
            "observed": float(cr2["controlled"]),
            "ci95": [float(x) for x in np.nanpercentile(boot["C_R2"], [2.5, 97.5])],
            "note": "expected negative; not in Holm family",
        },
        "n_perm": n_perm,
        "n_boot": n_boot,
        "holm_family": list(PRIMARY_FAMILY),
        "mean_delta_adapt": float(df.delta_adapt.mean()),
        "median_delta_adapt": float(df.delta_adapt.median()),
        "frac_positive_delta_adapt": float((df.delta_adapt > 0).mean()),
        "patch_worse_on_average": bool(float(df.delta_adapt.mean()) < 0),
    }


def calibration_assoc(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for name, col in (
        ("delta_intercept", "delta_intercept"),
        ("delta_affine", "delta_affine"),
        ("delta_direction", "delta_direction"),
        ("delta_adapt", "delta_adapt"),
    ):
        if col not in df.columns:
            continue
        out[name] = _assoc(df, col)
    # direction exceeds intercept/affine if rho(KH, delta_direction) > 0 and
    # rho(KH, delta_adapt) is not explained by intercept/affine alone.
    return out
