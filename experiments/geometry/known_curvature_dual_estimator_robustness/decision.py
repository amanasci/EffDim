"""Frozen-gate Boolean findings. Do not retune thresholds after seeing scores."""

from __future__ import annotations

import numpy as np

from .config import (
    GATE_D_FULL_CLEAN_COS,
    GATE_D_FULL_CLEAN_RATIO,
    GATE_D_FULL_CLEAN_RHO,
    GATE_D_FULL_STRESS_COS,
    GATE_D_FULL_STRESS_RATIO,
    GATE_D_FULL_STRESS_RHO,
    GATE_F0_RES_ENERGY_FRAC,
    GATE_F1_H_REL,
    GATE_F4_RES_RHO,
    GATE_Q_CONST_CAL,
    GATE_Q_PW_RHO,
    GATE_Q_T2_COS,
    GATE_Q_T2_RATIO,
    GATE_Q_T2_SCAL_RHO,
    GATE_SAMP_COS_DROP,
    GATE_SAMP_RHO_DROP,
)


def _row(rows, fixture, sampling, noise):
    cell = f"{fixture}_{sampling}_{noise}"
    for r in rows:
        if r.get("cell") == cell:
            return r
    return None


def _in(x, lo, hi):
    return x is not None and np.isfinite(x) and lo <= float(x) <= hi


def decide(d_full, d_res, q_t2, q_t3, q_pw, scal, skipped) -> dict:
    f4c = _row(d_full, "F4", "S0", "N0")
    f4s = _row(d_full, "F4", "S3", "N4")
    decoder_full_clean_ok = bool(
        f4c
        and (f4c.get("rho") or -1) >= GATE_D_FULL_CLEAN_RHO
        and (f4c.get("median_cosine") or -1) >= GATE_D_FULL_CLEAN_COS
        and _in(f4c.get("median_ratio"), *GATE_D_FULL_CLEAN_RATIO)
    )
    decoder_full_stress_ok = bool(
        f4s
        and (f4s.get("rho") or -1) >= GATE_D_FULL_STRESS_RHO
        and (f4s.get("median_cosine") or -1) >= GATE_D_FULL_STRESS_COS
        and _in(f4s.get("median_ratio"), *GATE_D_FULL_STRESS_RATIO)
    )

    r0 = _row(d_res, "F0", "S0", "N0")
    r1 = _row(d_res, "F1", "S0", "N0")
    r2 = _row(d_res, "F2", "S0", "N0")
    r4 = _row(d_res, "F4", "S0", "N0")
    r4s = _row(d_res, "F4", "S3", "N4")
    f0_ok = bool(r0 and (r0.get("energy_frac") or 1) <= GATE_F0_RES_ENERGY_FRAC)
    f1_ok = bool(r1 and abs((r1.get("median_ratio") or 99) - 1.0) <= GATE_F1_H_REL)
    f2_ok = bool(r2 and (r2.get("median_cosine") is not None) and (r2.get("energy_B_S_mean") or 0) > 1e-6)
    f4_ok = bool(r4 and ((r4.get("rho") or -1) >= GATE_F4_RES_RHO or (r4.get("scal_rho") or -1) >= GATE_F4_RES_RHO))
    decoder_residual_clean_ok = bool(f0_ok and f1_ok and f2_ok and f4_ok)
    decoder_residual_stress_ok = bool(
        r4s and ((r4s.get("rho") or -1) >= 0.60 or (r4s.get("scal_rho") or -1) >= 0.60)
    )

    qt2 = _row(q_t2, "F4", "S0", "N0")
    qpw = _row(q_pw, "F4", "S0", "N0")
    sc = _row(scal, "F4", "S0", "N0")
    quadratic_matched_patch_ok = bool(
        qt2
        and (qt2.get("median_tensor_cos") or -1) >= GATE_Q_T2_COS
        and _in(qt2.get("median_tensor_ratio"), *GATE_Q_T2_RATIO)
        and (qt2.get("scal_rho") or -1) >= GATE_Q_T2_SCAL_RHO
    )
    qt3 = _row(q_t3, "F4", "S0", "N0")
    quadratic_uniform_patch_ok = bool(qt3 and (qt3.get("median_T2_T3_cos") or -1) >= 0.80)
    quadratic_pointwise_residual_ok = bool(
        qpw and ((qpw.get("median_tensor_cos") or -1) >= 0.70 or (qpw.get("scal_rho") or -1) >= GATE_Q_PW_RHO)
        and (qpw.get("scal_rho") or -1) >= GATE_Q_PW_RHO
    )
    quadratic_intrinsic_ok = bool(sc and (sc.get("Q_scal_T2_rho") or -1) >= GATE_Q_T2_SCAL_RHO)

    f4s2 = _row(d_full, "F4", "S2", "N0")
    f4s1 = _row(d_full, "F4", "S1", "N0")
    sampling_robust = True
    if f4c and f4s2:
        sampling_robust = sampling_robust and (f4c.get("rho") or 0) - (f4s2.get("rho") or 0) <= GATE_SAMP_RHO_DROP
        sampling_robust = sampling_robust and (f4c.get("median_cosine") or 0) - (f4s2.get("median_cosine") or 0) <= GATE_SAMP_COS_DROP
    else:
        sampling_robust = False
    f4n2 = _row(d_full, "F4", "S0", "N2")
    noise_robust = bool(f4c and f4n2) and (
        (f4c.get("rho") or 0) - (f4n2.get("rho") or 0) <= GATE_SAMP_RHO_DROP
        and (f4c.get("median_cosine") or 0) - (f4n2.get("median_cosine") or 0) <= GATE_SAMP_COS_DROP
    )
    qt2_s2 = _row(q_t2, "F4", "S2", "N0")
    sampling_measure_dependence_detected = bool(
        qt3 and (qt3.get("median_T2_T3_cos") is not None) and (qt3.get("median_T2_T3_cos") or 1) < 0.95
    )

    d_ok = decoder_full_clean_ok
    q_patch = quadratic_matched_patch_ok
    q_pw_ok = quadratic_pointwise_residual_ok
    if not d_full or not q_t2:
        summary = "bounded_dual_estimator_validation_unresolved"
    elif not sampling_robust or not noise_robust:
        if d_ok or q_patch:
            summary = "both_estimators_sampling_or_noise_sensitive"
        else:
            summary = "neither_estimator_validated"
    elif d_ok and q_patch and q_pw_ok:
        summary = "both_estimators_validated_for_matched_targets"
    elif d_ok and q_patch and not q_pw_ok:
        summary = "decoder_validated_quadratic_finite_patch_only"
    elif d_ok and not q_patch:
        summary = "decoder_only_validated"
    elif q_patch and not d_ok:
        summary = "quadratic_only_validated"
    else:
        summary = "neither_estimator_validated"

    return {
        "decoder_full_clean_ok": decoder_full_clean_ok,
        "decoder_full_stress_ok": decoder_full_stress_ok,
        "decoder_residual_clean_ok": decoder_residual_clean_ok,
        "decoder_residual_stress_ok": decoder_residual_stress_ok,
        "quadratic_matched_patch_ok": quadratic_matched_patch_ok,
        "quadratic_uniform_patch_ok": quadratic_uniform_patch_ok,
        "quadratic_pointwise_residual_ok": quadratic_pointwise_residual_ok,
        "quadratic_intrinsic_ok": quadratic_intrinsic_ok,
        "sampling_measure_dependence_detected": sampling_measure_dependence_detected,
        "sampling_robust": sampling_robust,
        "noise_robust": noise_robust,
        "summary_label": summary,
        "skipped": skipped,
        "notes": {
            "f0_residual_energy_ok": f0_ok,
            "f1_mean_ok": f1_ok,
            "f2_tracefree_ok": f2_ok,
            "sparse_row": None if f4s1 is None else {"rho": f4s1.get("rho")},
        },
    }
