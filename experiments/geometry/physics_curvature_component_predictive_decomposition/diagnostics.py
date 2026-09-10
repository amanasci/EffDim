"""Phase 2: curvature organization diagnostics from frozen component tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PRIMARY_D
from geometry.physics_cross_model_full_curvature_reconciliation.metrics import aniso_prefactor


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def enrich_organization(df: pd.DataFrame, d: int = PRIMARY_D) -> pd.DataFrame:
    out = df.copy()
    kd = out["K_dir_cross"].to_numpy(float)
    kh = out["K_H_cross"].to_numpy(float)
    ktf = out["K_TF_cross"].to_numpy(float)
    out["KH_share_of_Kdir"] = np.where(np.abs(kd) > 1e-12, kh / kd, np.nan)
    out["KTF_share_of_Kdir"] = np.where(np.abs(kd) > 1e-12, ktf / kd, np.nan)
    pref = aniso_prefactor(d)
    if "trace_energy_mean" in out.columns and "traceless_energy_mean" in out.columns:
        e_h = out["trace_energy_mean"].to_numpy(float) / float(d)
        e_tf = pref * out["traceless_energy_mean"].to_numpy(float)
        den = e_h + e_tf
        out["E_H"] = e_h
        out["E_TF"] = e_tf
        out["E_dir"] = den
        out["F_H"] = np.where(den > 1e-18, e_h / den, np.nan)
    if "trace_fraction" in out.columns and "C_trace_mean" not in out.columns:
        out["C_trace_mean"] = out["trace_fraction"]
    if "C_trace_mean" not in out.columns and "trace_fraction" in out.columns:
        out["C_trace_mean"] = out["trace_fraction"]
    return out


def organization_report(df: pd.DataFrame, model: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"model": model, "n": int(len(df))}
    for col in ("K_H_cross", "K_TF_cross", "K_dir_cross", "F_H", "C_trace_mean", "R_H", "R_B0", "R_BS"):
        if col not in df.columns:
            continue
        x = df[col].to_numpy(float)
        rec[f"median_{col}"] = float(np.nanmedian(x))
        rec[f"mean_{col}"] = float(np.nanmean(x))
        rec[f"frac_pos_{col}"] = float(np.nanmean(x > 0))
        rec[f"p05_{col}"] = float(np.nanpercentile(x, 5))
        rec[f"p95_{col}"] = float(np.nanpercentile(x, 95))
    rec["spearman_KH_KTF"] = _spearman(df.K_H_cross.to_numpy(float), df.K_TF_cross.to_numpy(float))
    rec["spearman_KH_Kdir"] = _spearman(df.K_H_cross.to_numpy(float), df.K_dir_cross.to_numpy(float))
    rec["spearman_KTF_Kdir"] = _spearman(df.K_TF_cross.to_numpy(float), df.K_dir_cross.to_numpy(float))
    if "F_H" in df.columns:
        rec["spearman_Kdir_FH"] = _spearman(df.K_dir_cross.to_numpy(float), df.F_H.to_numpy(float))
    rec["high_total_is_mostly_traceless"] = bool(
        np.isfinite(rec["spearman_KTF_Kdir"])
        and rec["spearman_KTF_Kdir"] > rec.get("spearman_KH_Kdir", -np.inf)
        and rec.get("spearman_Kdir_FH", 0.0) < 0
    )
    if "aniso_share_of_Kdir" in df.columns:
        rec["median_KTF_share_of_Kdir"] = float(np.nanmedian(df.aniso_share_of_Kdir))
    elif "KTF_share_of_Kdir" in df.columns:
        rec["median_KTF_share_of_Kdir"] = float(np.nanmedian(df.KTF_share_of_Kdir))
    return rec
