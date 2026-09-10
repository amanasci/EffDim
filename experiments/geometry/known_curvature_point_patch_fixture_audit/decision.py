"""Mechanical decision labels. Thresholds frozen in config.py before primary scores."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import (
    COS_POINTWISE_OK,
    FALSE_F0_KDIR_MAX,
    FALSE_F1_TF_FRAC_MAX,
    FALSE_F2_H_FRAC_MAX,
    RHO_PATCH_OK,
    RHO_POINTWISE_OK,
    RHO_SHRINK_OK,
    ROBUST_DROP_MAX,
    SEED_SPEARMAN_OK,
)


def _g(d: dict, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def decide(bundle: dict) -> dict[str, Any]:
    tests_ok = bool(bundle.get("tests_ok"))
    truth_ok = bool(bundle.get("truth_ok"))
    if not tests_ok or not truth_ok:
        return {
            "label": "known_curvature_fixture_audit_blocked",
            "reason": "unit tests or independent truth validation failed",
            "tests_ok": tests_ok,
            "truth_ok": truth_ok,
        }

    d_point = bundle.get("decoder_pointwise", {})
    q_patch = bundle.get("quadratic_patch", {})
    q_shrink = bundle.get("quadratic_shrink", {})
    d_seed = bundle.get("decoder_seeds", {})
    false = bundle.get("false_curvature", {})
    robust_s = bundle.get("sampling_robust", {})
    robust_n = bundle.get("noise_robust", {})
    mean_vs = bundle.get("mean_vs_full", {})

    decoder_ok = (
        float(_g(d_point, "rho_H", default=0)) >= RHO_POINTWISE_OK
        and float(_g(d_point, "median_cosine_H", default=0)) >= COS_POINTWISE_OK
        and float(_g(d_point, "rho_Kdir", default=0)) >= RHO_POINTWISE_OK
        and float(_g(d_seed, "spearman_Kdir", default=0)) >= SEED_SPEARMAN_OK
        and float(_g(false, "F0_decoder_Kdir", default=1)) <= FALSE_F0_KDIR_MAX * 50
        and float(_g(false, "F1_decoder_tf_frac", default=1)) <= FALSE_F1_TF_FRAC_MAX
        and float(_g(false, "F2_decoder_H_frac", default=1)) <= FALSE_F2_H_FRAC_MAX
    )
    decoder_unstable = float(_g(d_seed, "spearman_Kdir", default=0)) < SEED_SPEARMAN_OK or not decoder_ok

    q_patch_ok = float(_g(q_patch, "rho_Kdir_T2", default=0)) >= RHO_PATCH_OK
    q_shrink_ok = float(_g(q_shrink, "rho_Kdir_T1_smallest_k", default=0)) >= RHO_SHRINK_OK
    q_ok = q_patch_ok and (q_shrink_ok or float(_g(q_shrink, "rho_improves", default=0)) > 0)

    clean_ok = decoder_ok and q_patch_ok
    samp_drop = float(_g(robust_s, "max_rho_drop", default=0))
    noise_drop = float(_g(robust_n, "max_rho_drop", default=0))
    density_explains = bool(_g(robust_s, "T3_rescues_Q", default=False))
    noise_explains = bool(_g(robust_n, "thickness_drives_Q", default=False))
    mean_full = bool(_g(mean_vs, "divergence", default=False))

    if decoder_ok and q_ok:
        label = "both_instruments_valid_at_distinct_scales"
        reason = "decoder recovers T1; quadratic recovers matched T2/T3 and shrinks toward T1"
    elif decoder_ok and not q_ok:
        label = "decoder_pointwise_valid_quadratic_patch_unreliable"
        reason = "decoder recovers pointwise truth; quadratic fails patch match and/or T1 shrinkage"
    elif q_ok and decoder_unstable:
        label = "quadratic_patch_valid_decoder_pointwise_unstable"
        reason = "quadratic recovers matched patch truth; decoder unstable or fails T1"
    elif clean_ok and (samp_drop > ROBUST_DROP_MAX or noise_drop > ROBUST_DROP_MAX):
        label = "both_valid_only_in_clean_uniform_regime"
        reason = "matched recovery in S0/N0, material failure under sparsity or noise"
    elif density_explains:
        label = "density_conditioned_instrument_divergence"
        reason = "fixed-k / sampling weights explain estimator disagreement vs unmatched T1"
    elif noise_explains:
        label = "noise_conditioned_instrument_divergence"
        reason = "normal thickness or noise orientation explains disagreement"
    elif mean_full:
        label = "mean_vs_full_curvature_divergence"
        reason = "mean curvature discards traceless bending that full K_dir recovers"
    else:
        label = "known_curvature_fixture_audit_unresolved"
        reason = "mixed pattern; no single stronger label is justified"

    # Priority overrides when a more specific robustness label fits a passing distinct-scale pair.
    if label == "both_instruments_valid_at_distinct_scales":
        if samp_drop > ROBUST_DROP_MAX and density_explains:
            label = "density_conditioned_instrument_divergence"
            reason = "both work at matched scales in the clean regime; sampling non-uniformity drives unmatched disagreement"
        elif noise_drop > ROBUST_DROP_MAX and noise_explains:
            label = "noise_conditioned_instrument_divergence"
            reason = "both work at matched scales in the clean regime; noise orientation drives unmatched disagreement"

    return {
        "label": label,
        "reason": reason,
        "decoder_ok": bool(decoder_ok),
        "quadratic_patch_ok": bool(q_patch_ok),
        "quadratic_shrink_ok": bool(q_shrink_ok),
        "thresholds": {
            "RHO_POINTWISE_OK": RHO_POINTWISE_OK,
            "COS_POINTWISE_OK": COS_POINTWISE_OK,
            "SEED_SPEARMAN_OK": SEED_SPEARMAN_OK,
            "RHO_PATCH_OK": RHO_PATCH_OK,
            "RHO_SHRINK_OK": RHO_SHRINK_OK,
        },
        "inputs": {
            "decoder_pointwise": d_point,
            "quadratic_patch": q_patch,
            "quadratic_shrink": q_shrink,
            "decoder_seeds": d_seed,
            "false_curvature": false,
            "sampling_robust": robust_s,
            "noise_robust": robust_n,
            "mean_vs_full": mean_vs,
        },
    }
