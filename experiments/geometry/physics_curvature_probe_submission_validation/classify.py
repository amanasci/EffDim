"""Decision labels. Gates are not retuned after seeing results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ALPHA, DECISION_LABELS, FROZEN_CTL, PREDECLARED_D, PRIMARY_K
from .pipeline import ValConfig, write_json
from .schema import DENOMINATOR, DIRECT_ERROR, PRIMARY, ProbeTargetId


def _ctl(assoc: pd.DataFrame, *, d: int, target: str, slice_mode: str = "full") -> float:
    g = assoc[(assoc.d == d) & (assoc.target_id == target) & (assoc.slice_mode == slice_mode)]
    return float(g.controlled.iloc[0]) if len(g) else float("nan")


def _fwer(perm: pd.DataFrame, *, d: int, target: str, slice_mode: str = "full") -> float:
    g = perm[(perm.d == d) & (perm.target_id == target) & (perm.slice_mode == slice_mode) & (perm.kind == "controlled")]
    return float(g.p_fwer.iloc[0]) if len(g) else float("nan")


def assign_label(root: Path, cfg: ValConfig, parity: dict[str, Any], probe: dict[str, Any], scale: dict[str, Any]) -> dict[str, Any]:
    out = cfg.resolved(root)
    assoc = pd.read_csv(out / "metric_associations.csv")
    perm = pd.read_csv(out / "metric_permutation.csv")
    scale_tbl = pd.read_csv(out / "scale_sensitivity.csv")
    leakage = True
    if (out / "leakage_report.json").exists():
        import json

        leakage = bool(json.loads((out / "leakage_report.json").read_text()).get("r2_matches_cached_geography", False))

    parity_ok = bool(parity.get("exact_parity"))
    primary_fwer = [_fwer(perm, d=d, target=PRIMARY.value) for d in (16, 17, 18, 19, 20)]
    primary_survives = any(np.isfinite(p) and p <= ALPHA for p in primary_fwer)

    error_ok = False
    error_hits = []
    for tid in DIRECT_ERROR:
        rho16 = _ctl(assoc, d=16, target=tid.value)
        # higher curvature should go with worse probes: positive rho for error, negative for R²
        if np.isfinite(rho16) and abs(rho16) >= 0.10:
            pf = _fwer(perm, d=16, target=tid.value)
            error_hits.append({"target": tid.value, "rho16": rho16, "p_fwer": pf})
            if rho16 > 0:
                error_ok = True

    denom_rhos = {tid.value: _ctl(assoc, d=16, target=tid.value) for tid in DENOMINATOR}
    error_rhos = {tid.value: _ctl(assoc, d=16, target=tid.value) for tid in DIRECT_ERROR}
    denom_driven = False
    if all(not np.isfinite(v) or abs(v) < 0.08 for v in error_rhos.values()) and any(
        np.isfinite(v) and abs(v) >= 0.15 for v in denom_rhos.values()
    ):
        denom_driven = True

    shuffle = probe.get("shuffle") or {}
    shuffle_ok = abs(float(shuffle.get("controlled", 1.0))) < 0.12 if shuffle else False

    # Scale recurrence at predeclared middle/upper (negative controlled ρ at k=2048).
    def _sign_match(d: int) -> list[bool]:
        ref = float(FROZEN_CTL[d])
        outb = []
        for k in (512, 1024, 1536, 2048):
            g = scale_tbl[(scale_tbl.k == k) & (scale_tbl.d == d)]
            if not len(g) or not np.isfinite(g.controlled.iloc[-1]):
                continue
            outb.append(np.sign(float(g.controlled.iloc[-1])) == np.sign(ref) or abs(float(g.controlled.iloc[-1])) < 0.05 and abs(ref) < 0.05)
        return outb

    def _rho_at(d: int, k: int) -> float:
        if k == PRIMARY_K:
            g = scale_tbl[(scale_tbl.k == k) & (scale_tbl.d == d) & (scale_tbl.source.isin(["reused_ndc", "reused_rank_sweep_cache"]))]
        else:
            g = scale_tbl[(scale_tbl.k == k) & (scale_tbl.d == d) & (scale_tbl.source.isin(["refit_scale_subset", "refit_scale_cache"]))]
        return float(g.controlled.iloc[0]) if len(g) and np.isfinite(g.controlled.iloc[0]) else float("nan")

    mid_signs = _sign_match(PREDECLARED_D["middle"])
    up_signs = _sign_match(PREDECLARED_D["upper"])
    frac = float(np.mean(mid_signs + up_signs)) if (mid_signs + up_signs) else float("nan")
    scale_recurrent = bool(np.isfinite(frac) and frac >= 0.75)
    atten = []
    for d in (PREDECLARED_D["middle"], PREDECLARED_D["upper"]):
        ref = abs(_rho_at(d, PRIMARY_K))
        for k in (512, 1024, 1536):
            v = abs(_rho_at(d, k))
            if np.isfinite(ref) and ref >= 0.10 and np.isfinite(v):
                atten.append(v < 0.5 * ref)
    magnitude_varies = bool(atten) and (sum(atten) / len(atten) >= 0.5)
    scale_material = bool(np.isfinite(frac) and frac < 0.75) or magnitude_varies

    reasons = []
    if not parity_ok:
        reasons.append("parity_failed")
        label = "submission_claim_unresolved"
    elif not leakage or not probe.get("r2_ok", False):
        reasons.append("leakage_or_r2_mismatch")
        label = "submission_claim_unresolved"
    elif denom_driven:
        reasons.append("denominator_driven")
        label = "local_r2_denominator_driven"
    elif not primary_survives:
        reasons.append("primary_fwer_failed")
        label = "submission_claim_unresolved"
    elif not error_ok:
        reasons.append("direct_error_did_not_persist")
        # still an R² association; not unresolved if primary survived — but the claim is decodability
        label = "local_r2_denominator_driven" if any(abs(v) >= 0.15 for v in denom_rhos.values() if np.isfinite(v)) else "submission_claim_unresolved"
        if label == "submission_claim_unresolved":
            reasons.append("error_metrics_weak_and_denom_weak")
    elif scale_material:
        reasons.append("scale_direction_or_magnitude_varies")
        label = "claim_supported_but_scale_dependent"
    elif scale_recurrent and error_ok and primary_survives and parity_ok:
        label = "submission_claim_supported"
        reasons.append("all_gates_passed")
    else:
        label = "submission_claim_unresolved"
        reasons.append("sensitivity_incomplete")

    if label not in DECISION_LABELS:
        raise RuntimeError(label)

    rec = {
        "label": label,
        "reasons": reasons,
        "parity_ok": parity_ok,
        "primary_survives_fwer": primary_survives,
        "direct_error_ok": error_ok,
        "error_hits": error_hits,
        "denom_rhos_d16": denom_rhos,
        "error_rhos_d16": error_rhos,
        "shuffle_ok": shuffle_ok,
        "scale_sign_frac_mid_upper": frac,
        "scale_recurrent": scale_recurrent,
        "scale_magnitude_varies": magnitude_varies,
        "n_pending_scale": int(scale.get("n_pending", 0)),
        "leakage_ok": leakage,
    }
    write_json(out / "decision.json", rec, force=cfg.force)
    return rec
