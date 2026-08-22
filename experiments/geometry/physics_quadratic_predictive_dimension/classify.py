"""Plateau, adequacy ranks, and gate-derived labels. Frozen on synthetics."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "plateau_rel_tol": 0.02,
    "delta_practical": 0.004,
    "n_lookahead": 3,
    "df_collapse": 0.12,
    "r2_90": 0.90,
    "r2_95": 0.95,
    "r2_99": 0.99,
    "tail_poor": 0.30,
    "tail_ok": 0.50,
    "u_bound_q": 0.99,
    "gn_max_iter": 8,
    "gn_damp": 1e-4,
    "n_inner_cp": 96,
    "ridge_n_grid": 11,
    "n_boot": 400,
}


def incremental(nmse: np.ndarray) -> np.ndarray:
    dlt = np.full_like(nmse, np.nan, dtype=np.float64)
    dlt[1:] = nmse[:-1] - nmse[1:]
    return dlt


def plateau_from_curve(
    ds: np.ndarray,
    nmse: np.ndarray,
    df_frac: np.ndarray | None,
    thr: dict[str, Any],
) -> dict[str, Any]:
    ds = np.asarray(ds, dtype=int)
    y = np.asarray(nmse, dtype=np.float64)
    if len(y) == 0 or not np.any(np.isfinite(y)):
        return {"d_plat": float("nan"), "d_lo": float("nan"), "d_hi": float("nan"), "reason": "empty"}
    best = float(np.nanmin(y))
    se_like = float(thr["plateau_rel_tol"]) * max(abs(best), 1e-8)
    band = best + max(se_like, float(thr["delta_practical"]))
    look = int(thr["n_lookahead"])
    dlt = incremental(y)
    chosen = None
    for i, d in enumerate(ds):
        if not np.isfinite(y[i]) or y[i] > band:
            continue
        nxt = dlt[i + 1 : i + 1 + look]
        nxt = nxt[np.isfinite(nxt)]
        if len(nxt) and float(np.mean(nxt)) > float(thr["delta_practical"]):
            continue
        if df_frac is not None and i < len(df_frac) and np.isfinite(df_frac[i]):
            if float(df_frac[i]) < float(thr["df_collapse"]) and i > 0:
                continue
        chosen = int(d)
        break
    if chosen is None:
        chosen = int(ds[int(np.nanargmin(y))])
    return {"d_plat": chosen, "band": band, "best_nmse": best, "reason": "ok"}


def adequacy_ranks(
    ds: np.ndarray,
    r2_lo: np.ndarray,
    thr: dict[str, Any],
) -> dict[str, Any]:
    out = {}
    for name, t in (("d90", thr["r2_90"]), ("d95", thr["r2_95"]), ("d99", thr["r2_99"])):
        hit = [int(d) for d, lo in zip(ds, r2_lo) if np.isfinite(lo) and lo >= t]
        out[name] = int(min(hit)) if hit else "not_reached"
    return out


def primary_label(
    *,
    dQ: float,
    dL: float,
    d95,
    r2_total: float,
    r2_E4: float,
    r2_U8: float,
    delta_Q_12_16: float,
    delta_L_12_16: float,
    synth_not_only12: bool,
    scale_stable: bool,
    thr: dict[str, Any],
) -> str:
    if not synth_not_only12:
        return "quadratic_predictive_dimension_unresolved"
    if not scale_stable:
        return "quadratic_predictive_dimension_unresolved"
    poor_tail = (np.isfinite(r2_E4) and r2_E4 < thr["tail_poor"]) or (
        np.isfinite(r2_U8) and r2_U8 < thr["tail_poor"]
    )
    high_total = np.isfinite(r2_total) and r2_total >= thr["r2_95"]
    plat12 = np.isfinite(dQ) and abs(dQ - 12) <= 1.5
    lin_beyond = np.isfinite(dL) and dL >= 14
    q_extends = np.isfinite(dQ) and dQ >= 14 and dQ <= 16.5
    q_gain_1612 = np.isfinite(delta_Q_12_16) and delta_Q_12_16 > thr["delta_practical"]
    lin_gain = np.isfinite(delta_L_12_16) and delta_L_12_16 > thr["delta_practical"]
    q_vs_lin = np.isfinite(delta_Q_12_16) and np.isfinite(delta_L_12_16) and abs(delta_Q_12_16 - delta_L_12_16) <= thr["delta_practical"]

    if high_total and poor_tail:
        return "high_total_low_tail_adequacy"
    if plat12 and lin_beyond and high_total and not poor_tail and d95 not in ("not_reached", None):
        return "quadratic_predictive_plateau_at_12_adequate"
    if plat12 and (not high_total or poor_tail or d95 == "not_reached"):
        return "quadratic_predictive_plateau_at_12_but_inadequate"
    if q_extends and q_gain_1612 and not q_vs_lin:
        return "extended_quadratic_dimension_14_16"
    if lin_gain and q_vs_lin:
        return "linear_extra_dimensions_preferred"
    if (not np.isfinite(dQ) or dQ >= 19) and (not high_total):
        return "quadratic_model_inadequate_within_S20"
    return "quadratic_predictive_dimension_unresolved"
