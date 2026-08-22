"""Paired bootstrap contrasts between controlled Spearman correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import CONTROLS
from .io_util import p_mc


def _rho(df: pd.DataFrame, ycol: str) -> float:
    sub = df.reset_index(drop=True).copy()
    for c in CONTROLS:
        if c not in sub.columns:
            sub[c] = np.nan
    return float(associate(sub.K_H_cross.to_numpy(float), sub[ycol].to_numpy(float), control_matrix(sub))["controlled"])


def paired_contrast(
    df: pd.DataFrame,
    y_a: str,
    y_b: str,
    *,
    n_boot: int,
    seed: int,
    name: str,
) -> dict:
    obs_a = _rho(df, y_a)
    obs_b = _rho(df, y_b)
    delta = obs_a - obs_b
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(df)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[idx]
        boot[b] = _rho(sub, y_a) - _rho(sub, y_b)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    # BCa optional skip if unstable
    p_pos = float(np.mean(boot > 0))
    b_count = int(np.sum(boot <= 0)) if delta > 0 else int(np.sum(boot >= 0))
    return {
        "name": name,
        "y_a": y_a,
        "y_b": y_b,
        "rho_a": obs_a,
        "rho_b": obs_b,
        "delta_rho": delta,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_boot_positive": p_pos,
        "p_mc_two_sided": p_mc(int(np.sum(np.abs(boot) >= abs(delta))), n_boot),
    }


def run_all_paired(df: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    specs = [
        ("delta_rho_MSE_GP", "mse_G", "mse_P"),
        ("delta_rho_MAE_GP", "mae_G", "mae_P"),
        ("delta_rho_MSE_CP", "mse_C", "mse_P"),
        ("delta_rho_MSE_GT", "mse_G", "mse_T"),
    ]
    rows = []
    for i, (name, ya, yb) in enumerate(specs):
        if ya not in df.columns or yb not in df.columns:
            continue
        rows.append(paired_contrast(df, ya, yb, n_boot=n_boot, seed=seed + i, name=name))
    return pd.DataFrame(rows)
