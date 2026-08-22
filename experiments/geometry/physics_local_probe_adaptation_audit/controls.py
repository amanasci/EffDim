"""Alignment-controlled associations and interaction sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import CONTROLS


def _partial_rho(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    if int(m.sum()) < 12:
        return {"controlled": float("nan"), "n": int(m.sum())}
    sub = pd.DataFrame({"x": x[m], "y": y[m]})
    for j, c in enumerate(CONTROLS):
        sub[c] = Z[m, j]
    return associate(sub.x.to_numpy(float), sub.y.to_numpy(float), control_matrix(sub))


def alignment_models(df: pd.DataFrame) -> pd.DataFrame:
    """Models A–D on dMSE_G_to_P."""
    y = df.dMSE_G_to_P.to_numpy(float)
    x = df.K_H_cross.to_numpy(float)
    Z0 = control_matrix(df)
    rows = [
        {"model": "model_A_baseline", **_partial_rho(y, x, Z0)},
        {"model": "model_B_AH", **_partial_rho(y, x, np.column_stack([Z0, df.A_H_G.to_numpy(float)]))},
        {"model": "model_C_AB", **_partial_rho(y, x, np.column_stack([Z0, df.A_B_G.to_numpy(float)]))},
        {
            "model": "model_D_both",
            **_partial_rho(
                y,
                x,
                np.column_stack([Z0, df.A_H_G.to_numpy(float), df.A_B_G.to_numpy(float)]),
            ),
        },
    ]
    out = pd.DataFrame(rows)
    # VIF on rank controls
    Zd = np.column_stack([Z0, df.A_H_G.to_numpy(float), df.A_B_G.to_numpy(float)])
    m = np.all(np.isfinite(Zd), axis=1)
    if m.sum() >= 12:
        s = np.linalg.svd(Zd[m], compute_uv=False)
        out.attrs["cond_design"] = float(s[0] / max(s[-1], 1e-12))
    return out


def interaction_sensitivity(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["dMSE_G_to_P", "K_H_cross", "A_H_G", "A_B_G", *CONTROLS])
    if len(sub) < 20:
        return {"ok": False}
    y = rankdata(sub.dMSE_G_to_P.to_numpy(float))
    kh = rankdata(sub.K_H_cross.to_numpy(float))
    ah = rankdata(sub.A_H_G.to_numpy(float))
    ab = rankdata(sub.A_B_G.to_numpy(float))
    z = (kh - kh.mean()) * (ab - ab.mean())
    Z = np.column_stack(
        [
            np.ones(len(sub)),
            kh - kh.mean(),
            ah - ah.mean(),
            ab - ab.mean(),
            z,
            *[rankdata(sub[c].to_numpy(float)) - 0.5 for c in CONTROLS],
        ]
    )
    beta, *_ = np.linalg.lstsq(Z, y - y.mean(), rcond=None)
    return {
        "ok": True,
        "beta_KH": float(beta[1]),
        "beta_interaction": float(beta[4]),
        "n": int(len(sub)),
    }
