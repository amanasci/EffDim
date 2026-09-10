"""Operating-characteristic labels. Does not overwrite neither_estimator_validated."""

from __future__ import annotations

from .config import PRIOR_DECISION_LABEL
from .metrics import BAND_RANK_MODERATE, BAND_RANK_STRONG, BAND_REL_MODERATE


def _abs(x) -> float:
    try:
        return abs(float(x))
    except (TypeError, ValueError):
        return float("nan")


def decide(ctx: dict) -> dict:
    d_full = ctx["d_full"]
    q = ctx["q"]
    d_res = ctx["d_res"]
    dims = ctx["dimensions"]

    d_full_label = _label_d_full(d_full)
    q_label = _label_q(q)
    d_res_label = _label_d_res(d_res)

    return {
        "prior_exact_recovery_label": PRIOR_DECISION_LABEL,
        "prior_label_not_overwritten": True,
        "d_full_summary_label": d_full_label,
        "q_summary_label": q_label,
        "d_residual_summary_label": d_res_label,
        "dimensions": dims,
        "distinction": (
            "Exact-recovery validation asked whether D-full and Q recovered the matched "
            "tensor at a frozen gate. Operating characteristics ask whether the same "
            "frozen estimators still rank, discriminate, and repeat under sampling and "
            "noise, even when those gates fail. The two questions are not substitutes."
        ),
        "notes": {
            "d_full": d_full,
            "q": q,
            "d_res": d_res,
        },
    }


def _label_d_full(d: dict) -> str:
    if d.get("unresolved"):
        return "d_full_operating_characteristics_unresolved"
    sphere_deg = bool(d.get("sphere_rank_degenerate"))
    cubic_strong = _abs(d.get("cubic_rho")) >= BAND_RANK_STRONG
    resid_rank = _abs(d.get("residualized_f4_clean_rho"))
    stress_rank = _abs(d.get("residualized_f4_stress_rho"))
    vec = float(d.get("clean_vector_cosine") or 0.0)
    if sphere_deg and cubic_strong and vec >= 0.95:
        return "d_full_useful_on_non_spherical_clean_geometry"
    if sphere_deg:
        return "d_full_sphere_rank_target_degenerate"
    if resid_rank >= BAND_RANK_MODERATE and stress_rank < BAND_RANK_MODERATE:
        return "d_full_noise_or_sampling_limited"
    if resid_rank < BAND_RANK_MODERATE:
        return "d_full_operating_characteristics_weak"
    return "d_full_useful_on_non_spherical_clean_geometry"


def _label_q(q: dict) -> str:
    if q.get("unresolved"):
        return "q_operating_characteristics_unresolved"
    t2_rho = _abs(q.get("t2_kh_clean_rho"))
    t2_tensor = _abs(q.get("t2_tensor_clean"))
    stress = _abs(q.get("t2_kh_stress_rho"))
    samp_dep = bool(q.get("sampling_measure_dependence"))
    rel = q.get("sampling_reliability")
    rel_v = _abs(rel) if rel is not None else float("nan")
    if t2_rho >= BAND_RANK_STRONG and t2_tensor >= BAND_RANK_MODERATE:
        return "q_useful_finite_patch_rank_statistic"
    if t2_rho >= BAND_RANK_MODERATE and samp_dep:
        return "q_moderately_informative_sampling_dependent_statistic"
    if np_finite(rel_v) and rel_v >= BAND_REL_MODERATE and t2_rho < BAND_RANK_MODERATE:
        return "q_reliable_but_geometrically_biased"
    if t2_rho >= BAND_RANK_MODERATE and stress < BAND_RANK_MODERATE:
        return "q_noise_or_sampling_limited"
    if t2_rho < BAND_RANK_MODERATE:
        return "q_operating_characteristics_weak"
    return "q_moderately_informative_sampling_dependent_statistic"


def _label_d_res(d: dict) -> str:
    if d.get("unresolved"):
        return "d_residual_operating_characteristics_unresolved"
    clean = _abs(d.get("f4_clean_rho"))
    stress = _abs(d.get("f4_stress_rho"))
    if clean >= BAND_RANK_STRONG and stress >= BAND_RANK_STRONG:
        return "d_residual_robust_pointwise_instrument"
    if clean >= BAND_RANK_STRONG and stress >= BAND_RANK_MODERATE:
        return "d_residual_useful_but_stress_sensitive"
    if clean >= BAND_RANK_MODERATE:
        return "d_residual_useful_but_stress_sensitive"
    return "d_residual_operating_characteristics_weak"


def np_finite(x) -> bool:
    import math

    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
