"""Frozen thresholds and gate-derived labels. Not tuned on mag_r_desi."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "n_perm": 10000,
    "n_boot": 2000,
    "alpha": 0.05,
    "r_h_fail": 0.20,
    "valid_frac_fail": 0.85,
    "persist_neighbours": 3,
    "narrow_width": 2,
    "reliability_track": 0.70,
    "scale_disagree": 0.15,
    "var_taus": [0.80, 0.825, 0.85, 0.875, 0.90],
    "primary_d_min": 12,
    "primary_d_max": 20,
    "ref_d_min": 8,
    "n_trace_anchors": 32,
    "trace_atol": 1e-8,
}


def primary_label(
    *,
    fwer_hits: list[int],
    reliable: dict[int, bool],
    tracks_rel: bool,
    scale_stable: bool,
    missing_ok: bool,
    thr: dict[str, Any],
) -> str:
    if not missing_ok:
        return "curvature_probe_rank_sweep_unresolved"
    if not scale_stable:
        return "curvature_probe_rank_sweep_unresolved"
    if tracks_rel and not fwer_hits:
        return "curvature_probe_association_tracks_estimator_reliability"
    rel_hits = [d for d in fwer_hits if reliable.get(d, False)]
    if not rel_hits:
        if fwer_hits:
            return "curvature_probe_rank_sweep_unresolved"
        return "curvature_probe_association_not_familywise_supported"
    rel_hits = sorted(rel_hits)
    persist = False
    need = int(thr["persist_neighbours"])
    for d in rel_hits:
        neigh = [x for x in rel_hits if abs(x - d) <= 1]
        if len(neigh) >= need:
            persist = True
            break
        # also count consecutive run
    run = 1
    best_run = 1
    for a, b in zip(rel_hits, rel_hits[1:]):
        run = run + 1 if b == a + 1 else 1
        best_run = max(best_run, run)
    persist = persist or best_run >= need
    width = rel_hits[-1] - rel_hits[0]
    if persist:
        return "curvature_probe_association_rank_robust"
    if width <= int(thr["narrow_width"]) or best_run <= 2:
        return "curvature_probe_association_dimension_localized"
    return "curvature_probe_association_dimension_localized"
