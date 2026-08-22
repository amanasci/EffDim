"""Phase 8: high-rank reliability sensitivity. Cutoffs are not chosen by significance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import R_H_FAIL, R_H_STRICT
from .pipeline import AuditConfig, peak_abs, write_df


def run_reliability(root: Path, cfg: AuditConfig) -> pd.DataFrame:
    out = cfg.resolved(root)
    adcp = cfg.adcp(root)
    rank = pd.read_csv(adcp / "dataset_rank_associations.csv")
    rel = pd.read_csv(adcp / "curvature_reliability.csv")
    panel = pd.read_parquet(adcp / "per_anchor_curvature.parquet")

    extra = []
    if "m_d" in panel.columns:
        extra.append(panel.groupby(["dataset_id", "d"], as_index=False).agg(median_m_d=("m_d", "median"), median_df_ratio=("df_ratio", "median"), median_dS=("dS", "median")))
    extra_df = extra[0] if extra else pd.DataFrame()

    rows = []
    cutoffs = (R_H_FAIL, *R_H_STRICT)
    for (ds, lab), g in rank.groupby(["dataset_id", "label"]):
        rg = rel[rel.dataset_id == ds] if "dataset_id" in rel.columns else rel
        for cut in cutoffs:
            keep = []
            for _, r in g.iterrows():
                rr = rg[rg.d == r.d]
                rh = float(rr.median_R_H.iloc[0]) if len(rr) and "median_R_H" in rr.columns else float("nan")
                ok = bool(np.isfinite(rh) and rh >= cut)
                keep.append((int(r.d), float(r.controlled), float(r.raw), rh, ok))
            okd = {d: ctl for d, ctl, raw, rh, ok in keep if ok and np.isfinite(ctl)}
            peak_d, peak_v = peak_abs(okd)
            rows.append(
                {
                    "dataset_id": ds,
                    "label": lab,
                    "cutoff": float(cut),
                    "cutoff_source": "frozen_gate" if cut == R_H_FAIL else "descriptive_stricter",
                    "n_ranks_kept": len(okd),
                    "peak_d": peak_d if peak_d is not None else "",
                    "peak_controlled": peak_v,
                    "smooth_fraction_d43_kept": bool(ds == "physics_vit_base" and lab == "smooth_fraction" and 43 in okd),
                    "desi_mag_d36_kept": bool(ds == "desi_vit_base_hsc" and lab == "mag_r" and 36 in okd),
                }
            )
        for _, r in g.iterrows():
            rr = rg[rg.d == r.d]
            rh = float(rr.median_R_H.iloc[0]) if len(rr) and "median_R_H" in rr.columns else np.nan
            md = extra_df[(extra_df.dataset_id == ds) & (extra_df.d == r.d)] if len(extra_df) else pd.DataFrame()
            rows.append(
                {
                    "dataset_id": ds,
                    "label": lab,
                    "cutoff": "overlay",
                    "cutoff_source": "per_rank",
                    "n_ranks_kept": "",
                    "peak_d": int(r.d),
                    "peak_controlled": float(r.controlled),
                    "R_H": rh,
                    "m_d": float(md.median_m_d.iloc[0]) if len(md) else np.nan,
                    "df_ratio": float(md.median_df_ratio.iloc[0]) if len(md) else np.nan,
                    "dS": float(md.median_dS.iloc[0]) if len(md) else np.nan,
                    "fail_frozen": bool(np.isfinite(rh) and rh < R_H_FAIL),
                    "weak_0.4": bool(np.isfinite(rh) and rh < 0.4),
                    "weak_0.5": bool(np.isfinite(rh) and rh < 0.5),
                    "weak_0.6": bool(np.isfinite(rh) and rh < 0.6),
                }
            )
    df = pd.DataFrame(rows)
    write_df(out / "high_rank_reliability_sensitivity.csv", df, force=cfg.force)
    return df
