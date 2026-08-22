"""Label-free classification of implicit constraints and dimension bounds."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_stable_tangent_dimension.nested_pca import degenerate_blocks

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "overlap_min": 0.40,
    "cancel_r2_min": 0.30,
    "cancel_r2_flat_max": 0.15,
    "persist_min": 0.35,
    "rel_gap_min": 0.12,
    "null_q": 0.99,
    "flat_ratio_max": 0.35,
    "tangent_ratio_min": 0.55,
    "amp_quad_lo": 1.3,
    "amp_quad_hi": 3.2,
    "amp_lin_lo": 0.5,
    "amp_lin_hi": 1.5,
    "q_max": 10,
}


def classify_one(
    *,
    cancel_r2: float,
    split_overlap: float,
    persist: float,
    raw_mse: float,
    corr_mse: float,
    null_mse: float,
    amp_exp: float,
    corr_exp: float,
    var_share: float,
    thr: dict[str, Any],
) -> str:
    ov_ok = np.isfinite(split_overlap) and split_overlap >= float(thr["overlap_min"])
    persist_ok = (not np.isfinite(persist)) or persist >= float(thr["persist_min"])
    beats_null = np.isfinite(corr_mse) and np.isfinite(null_mse) and corr_mse <= null_mse
    high_cancel = np.isfinite(cancel_r2) and cancel_r2 >= float(thr["cancel_r2_min"])
    low_cancel = (not np.isfinite(cancel_r2)) or cancel_r2 <= float(thr["cancel_r2_flat_max"])
    quad_amp = np.isfinite(amp_exp) and float(thr["amp_quad_lo"]) <= amp_exp <= float(thr["amp_quad_hi"])
    lin_amp = np.isfinite(amp_exp) and float(thr["amp_lin_lo"]) <= amp_exp <= float(thr["amp_lin_hi"])
    low_var = np.isfinite(var_share) and var_share <= float(thr["flat_ratio_max"])
    high_var = np.isfinite(var_share) and var_share >= float(thr["tangent_ratio_min"])

    if ov_ok and persist_ok and high_cancel and beats_null and (quad_amp or not np.isfinite(amp_exp)):
        return "curvature_active_normal"
    if ov_ok and persist_ok and low_cancel and low_var and beats_null and not high_var:
        return "approximately_flat_normal"
    if high_var and lin_amp and not high_cancel:
        return "first_order_tangent"
    if ov_ok and (not high_cancel) and (not low_var) and (not high_var):
        return "structured_thickness_normal_candidate"
    if high_cancel and high_var:
        return "mixed_order"
    return "unresolved"


def consecutive_normal_count(labels: list[str], *, allow: tuple[str, ...] | None = None) -> int:
    allow = allow or ("curvature_active_normal", "approximately_flat_normal")
    n = 0
    for lab in labels:
        if lab in allow:
            n += 1
        else:
            break
    return n


def blockwise_prefix(evals: np.ndarray, flags: np.ndarray, rel_gap_min: float) -> np.ndarray:
    accepted = np.zeros(len(flags), dtype=bool)
    blocks = degenerate_blocks(np.asarray(evals, dtype=np.float64), rel_gap_min=rel_gap_min)
    for a, b in blocks:
        b = min(b, len(flags) - 1)
        if a > b:
            continue
        if not bool(np.all(flags[a : b + 1])):
            break
        accepted[a : b + 1] = True
    return accepted


def primary_label(
    *,
    cN_minus: float,
    d1_minus: float,
    d1_plus: float,
    q2: float,
    R: int,
    e4_normal_frac: float,
    synth_not_only12: bool,
) -> str:
    if not synth_not_only12:
        return "implicit_normal_inverse_unresolved"
    supports12 = np.isfinite(cN_minus) and abs(cN_minus - (R - 12)) <= 1.5 and abs(d1_plus - 12) <= 1.5
    supports16 = np.isfinite(cN_minus) and abs(cN_minus - (R - 16)) <= 1.5 and abs(d1_plus - 16) <= 1.5
    interval = np.isfinite(d1_minus) and np.isfinite(d1_plus) and (d1_plus - d1_minus) >= 2
    if supports12 and not supports16 and np.isfinite(d1_minus) and d1_minus >= 10:
        return "normal_codimension_8_supports_tangent12"
    if supports16 and not supports12:
        return "normal_codimension_4_supports_tangent16"
    if interval and np.isfinite(e4_normal_frac) and 0.2 <= e4_normal_frac <= 0.8:
        return "mixed_normal_and_tangent_tail"
    if np.isfinite(cN_minus) and cN_minus <= 1 and np.isfinite(e4_normal_frac) and e4_normal_frac < 0.2:
        return "structured_thickness_tail"
    if np.isfinite(cN_minus) and cN_minus >= 1 and interval:
        return "normal_inverse_partially_identified"
    return "implicit_normal_inverse_unresolved"
