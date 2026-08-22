"""Gate-derived primary labels. Frozen before seeing associations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_THRESHOLDS


def primary_label(
    *,
    n_included: int,
    n_reliable_datasets: int,
    n_confirmatory_fwer: int,
    p_global: float,
    mag_deltas: list[float],
    transition_aligned_var: bool,
    transition_aligned_rank: bool,
    scale_stable: bool,
    missing_ok: bool,
    thr: dict[str, Any] | None = None,
) -> str:
    thr = thr or DEFAULT_THRESHOLDS
    alpha = float(thr["alpha"])
    if not missing_ok:
        return "cross_dataset_curvature_replication_unresolved"
    if n_reliable_datasets < 1:
        return "adaptive_curvature_sweeps_underidentified"
    if n_included < 2 and n_confirmatory_fwer == 0:
        return "cross_dataset_curvature_replication_unresolved"
    if not scale_stable:
        return "cross_dataset_curvature_replication_unresolved"
    mag_ok = [d for d in mag_deltas if np.isfinite(d)]
    mag_neg = [d for d in mag_ok if d < 0]
    global_ok = np.isfinite(p_global) and p_global <= alpha
    if global_ok and len(mag_neg) >= 1 and (transition_aligned_var or transition_aligned_rank) and n_included >= 2:
        return "cross_dataset_curvature_transition_replicated"
    if global_ok and n_confirmatory_fwer >= 1 and n_included >= 2 and not (transition_aligned_var or transition_aligned_rank):
        return "curvature_probe_association_replicates_without_common_transition"
    if n_confirmatory_fwer >= 1 and not global_ok:
        return "dataset_specific_curvature_probe_associations"
    if n_confirmatory_fwer >= 1 and n_included < 2:
        return "dataset_specific_curvature_probe_associations"
    if global_ok and n_confirmatory_fwer >= 1:
        return "dataset_specific_curvature_probe_associations"
    return "curvature_probe_replication_not_supported"


def summarize_peaks(rank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if rank is None or not len(rank):
        return pd.DataFrame(rows)
    for (ds, lab), g in rank.groupby(["dataset_id", "label"]):
        g = g.copy()
        i_raw = int(np.nanargmax(np.abs(g["raw"].to_numpy(float)))) if g["raw"].notna().any() else 0
        i_ctl = int(np.nanargmax(np.abs(g["controlled"].to_numpy(float)))) if g["controlled"].notna().any() else 0
        rows.append(
            {
                "dataset_id": ds,
                "label": lab,
                "is_discovery": bool(g.iloc[0].is_discovery),
                "peak_raw_d": int(g.iloc[i_raw].d),
                "peak_raw": float(g.iloc[i_raw]["raw"]),
                "peak_ctl_d": int(g.iloc[i_ctl].d),
                "peak_ctl": float(g.iloc[i_ctl]["controlled"]),
                "any_fwer_ctl": bool((g.p_ctl_fwer <= 0.05).any()),
            }
        )
    return pd.DataFrame(rows)
