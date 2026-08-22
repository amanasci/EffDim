"""Operational stable tangent dimension from consecutive accepted blocks.

Gates (all label-free): split stability, held-out linear gain, cross-scale
persistence, tangent-like scaling. Isolated accepted ranks above a failed
block are not allowed. Graph dimension d_G is reported, never used to select.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .sphere_coords import EPS


DEFAULT_THRESHOLDS: dict[str, Any] = {
    "agreement_q": 0.99,
    "gain_q": 0.99,
    "persistence_min": 0.40,
    "tangent_lo": 1.2,
    "tangent_hi": 2.8,
    "curve_lo": 3.2,
    "curve_hi": 5.2,
    "thick_lo": -0.6,
    "thick_hi": 0.6,
    "z_agree_min": 1.5,
    "z_gain_min": 1.0,
    "require_scaling": True,
    "ainc_null_q99": 0.12,
    "agree_null_q99": 0.12,
    "gain_floor": 0.001,
    "prefix_A_min": 0.45,
    "rel_gap_min": 0.15,
    "block_A_min": 0.50,
}


def accept_block(
    *,
    agree: float,
    agree_null_q: float,
    z_agree: float,
    gain: float,
    gain_null_q: float,
    z_gain: float,
    persistence: float,
    scaling_label: str,
    scaling_resolved: bool,
    thr: dict[str, Any],
) -> tuple[bool, str]:
    if not np.isfinite(agree) or agree < agree_null_q:
        return False, "fail_stability"
    if np.isfinite(z_agree) and z_agree < float(thr["z_agree_min"]):
        return False, "fail_stability_z"
    if not np.isfinite(gain) or gain < gain_null_q:
        return False, "fail_heldout_gain"
    if np.isfinite(z_gain) and z_gain < float(thr["z_gain_min"]):
        return False, "fail_heldout_gain_z"
    if np.isfinite(persistence) and persistence < float(thr["persistence_min"]):
        return False, "fail_persistence"
    if thr.get("require_scaling", True):
        if not scaling_resolved:
            return False, "scaling_unresolved"
        if scaling_label != "tangent_like":
            return False, f"fail_scaling_{scaling_label}"
    return True, "accepted"


def consecutive_prefix(block_accept: list[bool]) -> int:
    """End rank of the largest consecutive accepted prefix (1-based d_T).

    `block_accept[i]` refers to the i-th eigengap block in rank order.
    Returns 0 if the first block fails.
    """
    d = 0
    # caller should pass per-rank accept flags for ranks 1..d_max
    for i, ok in enumerate(block_accept, start=1):
        if not ok:
            break
        d = i
    return d


def dT_from_rank_flags(accepted: np.ndarray) -> int:
    """accepted[d-1] True iff rank d passes. Isolated holes stop the prefix."""
    d_T = 0
    for i, ok in enumerate(accepted, start=1):
        if not ok:
            break
        d_T = i
    return int(d_T)


def survival_curve(dT: np.ndarray, d_max: int) -> np.ndarray:
    dT = np.asarray(dT, dtype=np.float64)
    dT = dT[np.isfinite(dT)]
    p = np.zeros(d_max)
    n = max(len(dT), 1)
    for d in range(1, d_max + 1):
        p[d - 1] = float(np.mean(dT >= d))
    return p


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    stat="median",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    fn = np.median if stat == "median" else np.mean
    point = float(fn(x))
    boots = []
    n = len(x)
    for _ in range(n_boot):
        boots.append(float(fn(x[rng.integers(0, n, size=n)])))
    lo, hi = float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
    return {"point": point, "lo": lo, "hi": hi}


def bootstrap_survival(
    dT: np.ndarray,
    d_max: int,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(dT, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    p = survival_curve(x, d_max)
    boot = np.zeros((n_boot, d_max))
    for t in range(n_boot):
        boot[t] = survival_curve(x[rng.integers(0, n, size=n)], d_max)
    return {
        "p": p,
        "lo": np.quantile(boot, alpha / 2, axis=0),
        "hi": np.quantile(boot, 1 - alpha / 2, axis=0),
    }


def model_label(
    *,
    median_dT: float,
    iqr_dT: float,
    p_adj: np.ndarray | None,
    scale_medians: dict[int, float],
    extra_block_label: str | None,
    concentrated: bool,
) -> str:
    ks = sorted(scale_medians)
    meds = [scale_medians[k] for k in ks]
    if not np.isfinite(median_dT):
        return "tangent_dimension_unresolved"
    scale_span = float(np.nanmax(meds) - np.nanmin(meds)) if meds else 0.0
    if extra_block_label == "scale_independent_thickness":
        return "stable_thickness_beyond_tangent"
    if extra_block_label in ("mixed_or_crossing", "curvature_normal_like"):
        return "finite_scale_stratification"
    if scale_span >= 2.0 or (not concentrated):
        return "scale_dependent_tangent_dimension"
    if concentrated and iqr_dT <= 3.0:
        return "stable_tangent_dimension_identified"
    return "tangent_dimension_unresolved"


def dimension_sensitivity_label(
    values_by_d: dict[int, float],
    *,
    band: list[int],
    sign_flip_ok: bool = False,
) -> str:
    """Classify whether a metric's qualitative reading survives the d band."""
    vals = [values_by_d.get(d) for d in band]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return "unresolved"
    signs = [np.sign(v) if abs(v) > 1e-6 else 0.0 for v in vals]
    if not sign_flip_ok and len(set(int(s) for s in signs if s != 0)) > 1:
        return "dimension_sensitive"
    rng = float(np.max(vals) - np.min(vals))
    mag = float(np.median(np.abs(vals)))
    if mag > EPS and rng / max(mag, EPS) < 0.35:
        return "dimension_robust"
    return "dimension_sensitive"
