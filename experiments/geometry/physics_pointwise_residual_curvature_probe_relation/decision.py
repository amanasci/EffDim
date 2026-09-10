"""Mechanically determined decision label. Do not force the old Q narrative."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DECISION_LABELS


def decide(
    *,
    seed_passed: bool,
    primary: dict[str, Any] | None,
    secondary: Any,
    parity_ok: bool,
    tests_ok: bool,
    resource_capped: bool,
) -> dict[str, Any]:
    if not tests_ok:
        label = "pointwise_residual_probe_relation_unresolved"
        return _pack(label, seed_passed, primary, reason="unit_tests_failed")
    if not parity_ok:
        label = "pointwise_residual_probe_relation_unresolved"
        return _pack(label, seed_passed, primary, reason="parity_failed")
    if resource_capped and primary is None:
        label = "pointwise_residual_probe_relation_unresolved"
        return _pack(label, seed_passed, primary, reason="resource_cap_before_inference")
    if not seed_passed:
        return _pack("decoder_residual_seed_unstable", False, primary, reason="seed_reliability_gate")

    p1 = primary["P1"]
    p2 = primary["P2"]
    p3 = primary["P3"]
    rho_g = float(p1["observed"])
    rho_p = float(p2["observed"])
    d_rho = float(p3["observed"])
    p1s = float(p1["p_holm"]) <= 0.05
    p2s = float(p2["p_holm"]) <= 0.05
    p3s = float(p3["p_holm"]) <= 0.05
    global_penalty = bool(rho_g < 0 and p1s)
    patch_positive = bool(rho_p > 0 and p2s)
    patch_penalty = bool(rho_p < 0 and p2s)
    patch_not_reverse = not patch_positive
    beneficial = bool((rho_g > 0 and p1s) or (rho_p > 0 and p2s))

    d_adapt = float("nan")
    if secondary is not None and hasattr(secondary, "name"):
        hit = secondary[secondary.name == "rho_CH_DeltaAdapt"]
        if len(hit):
            d_adapt = float(hit.iloc[0]["controlled"])
    rel_adapt = bool(np.isfinite(d_adapt) and d_adapt > 0)

    if global_penalty and patch_positive and p3s and d_rho > 0:
        label = "stable_global_penalty_and_absolute_local_reversal"
    elif global_penalty and patch_not_reverse and rel_adapt:
        label = "stable_global_penalty_with_relative_local_adaptation"
    elif global_penalty and patch_not_reverse and not rel_adapt:
        label = "stable_global_penalty_without_local_relief"
    elif beneficial and (not global_penalty) and (not patch_penalty):
        label = "stable_positive_pointwise_decodability_link"
    elif not (p1s or p2s or p3s):
        label = "stable_pointwise_residual_null"
    else:
        label = "pointwise_residual_probe_relation_unresolved"

    return _pack(
        label,
        True,
        primary,
        reason="mechanical",
        extra={
            "global_penalty": global_penalty,
            "patch_positive": patch_positive,
            "patch_penalty": patch_penalty,
            "contrast_sig": p3s,
            "rho_ctl_delta_adapt": d_adapt,
            "relative_adaptation": rel_adapt,
        },
    )


def _pack(label: str, seed_passed: bool, primary, reason: str, extra=None) -> dict[str, Any]:
    assert label in DECISION_LABELS
    out = {
        "label": label,
        "seed_reliability_passed": bool(seed_passed),
        "reason": reason,
        "estimator": "pointwise_sphere_residual_decoder_curvature",
        "Q_not_treated_as_ground_truth": True,
        "historical_full_not_primary": True,
    }
    if primary:
        out["P1"] = primary["P1"]
        out["P2"] = primary["P2"]
        out["P3"] = primary["P3"]
        out["signs_differ"] = bool(np.sign(primary["P1"]["observed"]) != np.sign(primary["P2"]["observed"]))
    if extra:
        out.update(extra)
    return out
