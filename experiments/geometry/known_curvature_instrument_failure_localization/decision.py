"""Exactly one mechanical localization label."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import (
    F0_FALSE_MAX,
    F1_TF_FRAC_MAX,
    F2_H_FRAC_MAX,
    MATERIAL_REL_DELTA,
    MATERIAL_RHO_DROP,
    Q1_REL_OK,
    Q1_RHO_OK,
    SEED_SPEARMAN_OK,
    T2_VS_T1_REL_STRONG,
)


def _f(d, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def decide(bundle: dict) -> dict[str, Any]:
    q1_ok = bool(bundle.get("q1_recovers_t2"))
    q1_rel = _f(bundle, "q1", "median_rel_Kdir_T2")
    q1_rho = _f(bundle, "q1", "rho_Kdir_T2_F4")
    if not np.isfinite(q1_rel):
        q1_ok = q1_ok or (np.isfinite(q1_rho) and q1_rho >= Q1_RHO_OK)
    else:
        q1_ok = q1_ok or (q1_rel <= Q1_REL_OK)

    d_pca = _f(bundle, "increments", "q1_to_q3_rel")
    d_ridge = _f(bundle, "increments", "q1_to_q2_rel")
    d_split = _f(bundle, "increments", "q4_to_q5_rel")
    rho_drop_pca = _f(bundle, "increments", "q1_to_q3_rho_drop")
    rho_drop_ridge = _f(bundle, "increments", "q1_to_q2_rho_drop")
    rho_drop_split = _f(bundle, "increments", "q4_to_q5_rho_drop")

    pca_fail = (np.isfinite(d_pca) and d_pca >= MATERIAL_REL_DELTA) or (
        np.isfinite(rho_drop_pca) and rho_drop_pca >= MATERIAL_RHO_DROP
    )
    ridge_fail = (np.isfinite(d_ridge) and d_ridge >= MATERIAL_REL_DELTA) or (
        np.isfinite(rho_drop_ridge) and rho_drop_ridge >= MATERIAL_RHO_DROP
    )
    split_fail = (np.isfinite(d_split) and d_split >= MATERIAL_REL_DELTA) or (
        np.isfinite(rho_drop_split) and rho_drop_split >= MATERIAL_RHO_DROP
    )

    patch_bias = bool(bundle.get("patch_differs_from_t1")) or (
        _f(bundle, "t2_vs_t1", "median_rel_Kdir") >= T2_VS_T1_REL_STRONG
    )

    d1_done = bool(bundle.get("d1_available"))
    fd_ok = bool(bundle.get("d2_fd_agrees"))
    false_survives = bool(bundle.get("d_false_survives_true_projector"))
    seed_bad = _f(bundle, "decoder", "spearman_Kdir") < SEED_SPEARMAN_OK
    d_nonid = d1_done and fd_ok and false_survives and seed_bad

    q_labels = []
    if not q1_ok:
        q_labels.append("quadratic_oracle_or_convention_mismatch")
    else:
        if pca_fail:
            q_labels.append("quadratic_tangent_estimation_failure")
        if ridge_fail:
            q_labels.append("quadratic_ridge_attenuation_failure")
        if split_fail:
            q_labels.append("quadratic_split_variance_failure")
        if patch_bias and not (pca_fail or ridge_fail or split_fail):
            q_labels.append("quadratic_patch_model_bias")

    n_mat = len(q_labels) + int(d_nonid)
    if n_mat > 1:
        label = "multiple_instrument_failure_sources"
        reason = "more than one localized mechanism is material: " + ", ".join(q_labels + (["decoder_learned_surface_hessian_nonidentifiability"] if d_nonid else []))
    elif n_mat == 1:
        label = q_labels[0] if q_labels else "decoder_learned_surface_hessian_nonidentifiability"
        reason = {
            "quadratic_oracle_or_convention_mismatch": "Q1 fails with exact frames, noiseless neighbours, and adequate design rank",
            "quadratic_tangent_estimation_failure": "Q1 succeeds; replacing the exact frame with PCA is the principal loss",
            "quadratic_ridge_attenuation_failure": "Q1 succeeds; production ridge is the principal loss",
            "quadratic_split_variance_failure": "full-neighbourhood Q4 succeeds; split-half Q5 is the principal loss",
            "quadratic_patch_model_bias": "Q1 recovers T2/T3 but those targets differ strongly from T1 at production radius",
            "decoder_learned_surface_hessian_nonidentifiability": "D false curvature survives the true projector; autodiff agrees with FD; seeds disagree",
        }[label]
    else:
        if not bundle.get("primary_complete"):
            label = "bounded_failure_localization_unresolved"
            reason = "the 64-anchor cached diagnostic did not isolate a dominant mechanism"
        else:
            label = "bounded_failure_localization_unresolved"
            reason = "no single mechanism exceeded frozen materiality thresholds"

    return {
        "label": label,
        "reason": reason,
        "q1_ok": bool(q1_ok),
        "pca_fail": bool(pca_fail),
        "ridge_fail": bool(ridge_fail),
        "split_fail": bool(split_fail),
        "patch_bias": bool(patch_bias),
        "d_nonidentifiable": bool(d_nonid),
        "d1_available": bool(d1_done),
        "thresholds": {
            "Q1_REL_OK": Q1_REL_OK,
            "Q1_RHO_OK": Q1_RHO_OK,
            "MATERIAL_REL_DELTA": MATERIAL_REL_DELTA,
            "MATERIAL_RHO_DROP": MATERIAL_RHO_DROP,
            "SEED_SPEARMAN_OK": SEED_SPEARMAN_OK,
        },
    }
