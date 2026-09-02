"""Mechanical decision labels for quadratic chart alignment."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DECISION_LABELS


def decide(
    *,
    primary: dict,
    secondary: dict,
    alignment: dict,
    synth_ok: bool,
    unstable: bool,
) -> dict[str, Any]:
    med_dq = float(primary.get("median_delta_Q", np.nan))
    ci_lo = float(primary.get("delta_Q_ci_lo", np.nan))
    rho = float(primary.get("rho_KH_delta_Q", np.nan))
    rho_p = float(primary.get("rho_KH_delta_Q_p_mc", 1.0))
    holm_ok = bool(primary.get("holm_both_pass", False))

    uq_gain = bool(np.isfinite(med_dq) and med_dq > 0 and ci_lo > 0)
    curv_pred = bool(np.isfinite(rho) and rho > 0 and rho_p <= 0.05 and holm_ok)

    med_dbs = float(secondary.get("median_delta_BS", np.nan))
    frac_bs = float(secondary.get("frac_UQ_captured_by_BS", np.nan))
    ab_med = float(alignment.get("A_B_median", np.nan))
    ab_null = float(alignment.get("A_B_null_median", 1.0))
    stab = float(alignment.get("gamma_fold_cosine_median", np.nan))

    chart_explains = bool(
        (np.isfinite(med_dbs) and med_dbs > 0 and med_dbs >= 0.4 * max(med_dq, 1e-12))
        or (np.isfinite(frac_bs) and frac_bs >= 0.4)
        or (np.isfinite(ab_med) and ab_med > ab_null + 0.15 and (not np.isfinite(stab) or stab >= 0.5))
    )

    if unstable or not synth_ok:
        label = "quadratic_probe_unstable_or_overfit" if unstable else "quadratic_chart_link_unresolved"
        if unstable:
            label = "quadratic_probe_unstable_or_overfit"
    elif uq_gain and curv_pred and chart_explains:
        label = "curvature_aligned_quadratic_decoding"
    elif uq_gain and not chart_explains:
        label = "generic_quadratic_local_decoding"
    elif not uq_gain:
        label = "tangent_rotation_preferred"
    else:
        label = "quadratic_chart_link_unresolved"

    assert label in DECISION_LABELS
    return {
        "label": label,
        "uq_gain": uq_gain,
        "curv_predicts_gain": curv_pred,
        "chart_explains_gain": chart_explains,
        "checks": {
            "median_delta_Q": med_dq,
            "delta_Q_ci_lo": ci_lo,
            "rho_KH_delta_Q": rho,
            "rho_p_mc": rho_p,
            "holm_both_pass": holm_ok,
            "median_delta_BS": med_dbs,
            "frac_UQ_captured_by_BS": frac_bs,
            "A_B_median": ab_med,
            "gamma_stability": stab,
            "synth_ok": synth_ok,
            "unstable": unstable,
        },
    }
