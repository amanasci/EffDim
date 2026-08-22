"""Pathway / compatibility diagnostics (not causal mediation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import CONTROLS


def _rho_extra(df: pd.DataFrame, xcol: str, ycol: str, extra: list[str]) -> float:
    sub = df.reset_index(drop=True).copy()
    Z = control_matrix(sub)
    cols = [sub[c].to_numpy(float) for c in extra if c in sub.columns]
    if cols:
        Z = np.column_stack([Z] + cols)
    return float(associate(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float), Z)["controlled"])


def pathway_table(df: pd.DataFrame, align: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(align, on="sample_id", how="inner")
    m["D_PG"] = m["weight_angle"]
    m["D_PG_tangent"] = m["tangent_weight_angle"]
    rel = m[m.direction_reliable.fillna(False)]
    rows = []
    if len(m):
        rows.append(
            {
                "test": "KH_predicts_DPG_given_align",
                "rho": _rho_extra(m, "K_H_cross", "D_PG", ["A_H_G", "A_B_G"]),
                "n": int(len(m)),
                "reliable_only": False,
            }
        )
        rows.append(
            {
                "test": "DPG_predicts_dMSE_given_KH_align",
                "rho": _rho_extra(m, "D_PG", "dMSE_G_to_P", ["K_H_cross", "A_H_G", "A_B_G"]),
                "n": int(len(m)),
                "reliable_only": False,
            }
        )
        rows.append(
            {
                "test": "dAB_predicts_dMSE",
                "rho": _rho_extra(m, "dA_B", "dMSE_G_to_P", CONTROLS),
                "n": int(len(m)),
                "reliable_only": False,
            }
        )
        rows.append(
            {
                "test": "dAH_predicts_dMSE",
                "rho": _rho_extra(m, "dA_H", "dMSE_G_to_P", CONTROLS),
                "n": int(len(m)),
                "reliable_only": False,
            }
        )
    if len(rel) >= 12:
        rows.append(
            {
                "test": "KH_predicts_DPG_reliable",
                "rho": _rho_extra(rel, "K_H_cross", "D_PG", ["A_H_G", "A_B_G"]),
                "n": int(len(rel)),
                "reliable_only": True,
            }
        )
    return pd.DataFrame(rows)
