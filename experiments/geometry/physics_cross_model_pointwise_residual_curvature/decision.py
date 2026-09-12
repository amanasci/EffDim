"""Mechanical D-residual and optional Q labels. Does not overwrite prior trees."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DECISION_LABELS, Q_LABELS


def decide(
    *,
    n_reliable_rep: int,
    h123: dict[str, Any] | None,
    tests_ok: bool,
    parity_ok: bool,
    resource_capped: bool,
    assocs: dict[str, dict] | None = None,
    hetero: dict[str, Any] | None = None,
    q_panel_run: bool = False,
    q_n_rep: int = 0,
) -> dict[str, Any]:
    if not tests_ok or not parity_ok:
        d = _pack("cross_model_pointwise_residual_unresolved", reason="tests_or_parity")
    elif resource_capped and (h123 is None or n_reliable_rep < 3):
        d = _pack("cross_model_pointwise_residual_unresolved", reason="resource_or_incomplete")
    elif n_reliable_rep < 3:
        d = _pack("pointwise_residual_seed_unstable_cross_model", reason="fewer_than_three_reliable_replication_encoders")
    else:
        assert h123 is not None
        h1, h2, h3 = h123["H1_bar_C_P"], h123["H2_bar_C_A"], h123["H3_bar_delta_C_PG"]
        all_neg = bool(h1["pass_holm"] and h2["pass_holm"] and h3["pass_holm"])
        patch_only = bool(h1["pass_holm"] and not (h2["pass_holm"] and h3["pass_holm"]))
        signs = []
        if assocs:
            signs = [np.sign(assocs[m]["C_P"]["observed"]) for m in assocs if m != "vit_base"]
        mixed = bool(signs) and (min(signs) < 0 < max(signs))
        I2 = float((hetero or {}).get("C_P", {}).get("I2", 0.0))
        near0 = bool(abs(h1["observed"]) < 0.05 and abs(h2["observed"]) < 0.05)
        if all_neg:
            d = _pack("cross_model_pointwise_local_degradation_replication", reason="H1_H2_H3_holm")
        elif mixed or I2 > 0.5:
            d = _pack("heterogeneous_pointwise_residual_probe_relation", reason="sign_or_I2")
        elif patch_only:
            d = _pack("cross_model_pointwise_patch_degradation_only", reason="H1_only")
        elif near0:
            d = _pack("cross_model_pointwise_residual_null", reason="aggregate_near_zero")
        else:
            d = _pack("vitb_specific_pointwise_local_degradation", reason="replication_does_not_support_vitb_pattern")
        d["H1"] = h1
        d["H2"] = h2
        d["H3"] = h3
    if not q_panel_run:
        q = "q_cross_model_resampling_not_run"
    elif q_n_rep < 16:
        q = "q_cross_model_resampling_exploratory"
    else:
        q = "q_cross_model_resampling_exploratory"
    d["q_label"] = q
    d["n_reliable_replication"] = int(n_reliable_rep)
    d["prior_labels_not_overwritten"] = True
    return d


def _pack(label: str, *, reason: str) -> dict[str, Any]:
    assert label in DECISION_LABELS
    return {
        "label": label,
        "reason": reason,
        "estimator": "pointwise_sphere_residual_decoder_curvature",
        "Q_not_treated_as_ground_truth": True,
        "vitb_excluded_from_replication_aggregate": True,
    }
