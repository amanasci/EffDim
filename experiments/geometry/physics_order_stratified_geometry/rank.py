"""Reliable quadratic-normal rank q_2: split, prediction, scale, consecutive prefix."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_stable_tangent_dimension.dimension import dT_from_rank_flags
from geometry.physics_stable_tangent_dimension.nested_pca import degenerate_blocks

from .algebra import (
    EPS,
    cross_frobenius,
    projector_overlap,
    svd_quadratic_image,
    truncate_bs_left,
)

DEFAULT_Q_THRESHOLDS: dict[str, Any] = {
    "rel_gap_min": 0.12,
    "mode_overlap_min": 0.45,
    "dS_gain_min": 1e-6,
    "persist_min": 0.35,
    "energy_null_q": 0.99,
    "q_max": 8,
    "require_heldout_gain": True,
    "require_split_overlap": True,
}


def mode_energies(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return s**2


def aligned_mode_energy(UA: np.ndarray, sA: np.ndarray, UB: np.ndarray, sB: np.ndarray, q: int) -> float:
    """Cross energy of the q-th 1-based mode after overlapping left subspaces."""
    if min(UA.shape[1], UB.shape[1], len(sA), len(sB)) < q or q < 1:
        return float("nan")
    # use prefix-q projectors; incremental energy ≈ s_q^A s_q^B * overlap of added direction
    u = UA[:, q - 1]
    Pb = UB[:, :q] @ (UB[:, :q].T @ u)
    ov = float(np.dot(Pb, Pb) / max(np.dot(u, u), EPS))
    return float(sA[q - 1] * sB[q - 1] * ov)


def prefix_heldout_gain(dS: np.ndarray) -> np.ndarray:
    """dS[q] is held-out sphere-normal gain using q modes; return incremental gains."""
    dS = np.asarray(dS, dtype=np.float64)
    g = np.full(len(dS), np.nan)
    prev = 0.0
    for i, v in enumerate(dS):
        if np.isfinite(v):
            g[i] = float(v - prev)
            prev = float(v)
    return g


def select_q2(
    *,
    sA: np.ndarray,
    sB: np.ndarray,
    UA: np.ndarray,
    UB: np.ndarray,
    dS: np.ndarray,
    persist: np.ndarray | None,
    energy_null: float,
    thr: dict[str, Any],
) -> dict[str, Any]:
    """Accept consecutive eigengap blocks of quadratic-normal modes."""
    q_max = int(min(thr.get("q_max", 8), len(sA), len(sB), UA.shape[1], UB.shape[1], len(dS)))
    if q_max < 1:
        return {"q2": 0, "flags": np.zeros(0, dtype=bool), "reason": "no_modes"}
    ev = 0.5 * (sA[:q_max] ** 2 + sB[:q_max] ** 2)
    blocks = degenerate_blocks(ev, rel_gap_min=float(thr.get("rel_gap_min", 0.12)))
    gains = prefix_heldout_gain(dS[:q_max])
    accepted = np.zeros(q_max, dtype=bool)
    ov_min = float(thr.get("mode_overlap_min", 0.45))
    g_min = float(thr.get("dS_gain_min", 1e-6))
    p_min = float(thr.get("persist_min", 0.35))
    e_null = float(energy_null)
    for a, b in blocks:
        b = min(b, q_max - 1)
        if a > b:
            continue
        ov = projector_overlap(UA[:, a : b + 1], UB[:, a : b + 1])
        gsum = float(np.nansum(gains[a : b + 1]))
        eblk = float(np.nansum([aligned_mode_energy(UA, sA, UB, sB, q + 1) for q in range(a, b + 1)]))
        persist_ok = True
        if persist is not None and len(persist) > b:
            persist_ok = float(np.nanmin(persist[a : b + 1])) >= p_min
        ok = (
            np.isfinite(ov)
            and ov >= ov_min
            and eblk >= e_null * max(b - a + 1, 1) * 0.25
            and (not thr.get("require_heldout_gain", True) or gsum >= g_min)
            and persist_ok
        )
        if not ok:
            break
        accepted[a : b + 1] = True
    q2 = int(dT_from_rank_flags(accepted))
    return {"q2": q2, "flags": accepted, "ev": ev, "gains": gains}


def classify_hypothesis(
    *,
    q2: float,
    overlap_e4: float,
    r2_quad: float,
    residual_r2_linear: float,
    pi_lin: float,
    pi_quad: float,
    pi_thick: float,
    m12_vs_m16: float,
    mix_resolved: bool,
) -> str:
    """Primary label. m12_vs_m16 = held-out risk(M12q) - risk(M16); negative means M12q better."""
    high_ov = np.isfinite(overlap_e4) and overlap_e4 >= 0.45
    high_r2 = np.isfinite(r2_quad) and r2_quad >= 0.25
    res_lin = np.isfinite(residual_r2_linear) and residual_r2_linear >= 0.08
    q_ok = np.isfinite(q2) and q2 >= 2
    quad_dom = (not mix_resolved) or (np.isfinite(pi_quad) and pi_quad >= 0.35 and (not np.isfinite(pi_lin) or pi_lin < 0.35))
    lin_dom = mix_resolved and np.isfinite(pi_lin) and pi_lin >= 0.35
    thick_dom = mix_resolved and np.isfinite(pi_thick) and pi_thick >= 0.45
    m16_wins = np.isfinite(m12_vs_m16) and m12_vs_m16 > 0.0 and res_lin

    if q_ok and high_ov and high_r2 and (not res_lin) and (quad_dom or not mix_resolved) and not m16_wins:
        return "linear12_plus_quadratic_normal_modes"
    if m16_wins and not high_r2:
        return "weak_tangent_dimension_beyond12"
    if (high_r2 or q_ok) and res_lin:
        return "mixed_linear_and_quadratic_tail"
    if thick_dom and not high_r2:
        return "structured_thickness_beyond_core"
    return "order_stratification_unresolved"
