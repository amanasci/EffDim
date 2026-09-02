"""Reproduce frozen QLCA tables (Phase 0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ORIGINAL_LABEL, PARITY_ATOL
from .io_util import file_sha256


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def inventory_frozen(qlca: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "summary.json",
        "parity.json",
        "primary_inference.json",
        "secondary_inference.json",
        "alignment_summary.json",
        "synthetic_results.json",
        "reuse_manifest.json",
        "CONFIG.json",
        "COMPLETE.json",
        "anchor_risks.csv",
        "chart_alignment.csv",
        "METHODS.md",
        "REPORT.md",
    ]
    files = {}
    missing = []
    for name in required:
        p = qlca / name
        files[name] = {
            "path": str(p),
            "exists": p.is_file(),
            "sha16": file_sha256(p) if p.is_file() else None,
            "bytes": p.stat().st_size if p.is_file() else None,
        }
        if not p.is_file():
            missing.append(name)
    return {"qlca_dir": str(qlca), "files": files, "missing": missing}


def reproduce_from_tables(qlca: Path) -> dict[str, Any]:
    primary = _load_json(qlca / "primary_inference.json")
    secondary = _load_json(qlca / "secondary_inference.json")
    align = _load_json(qlca / "alignment_summary.json")
    parity = _load_json(qlca / "parity.json")
    synth = _load_json(qlca / "synthetic_results.json")
    decision = _load_json(qlca / "decision.json")
    complete = _load_json(qlca / "COMPLETE.json")
    anchor = pd.read_csv(qlca / "anchor_risks.csv")

    obs = {
        "median_delta_Q": float(np.nanmedian(anchor.delta_Q)),
        "rho_KH_delta_Q": float(primary["rho_KH_delta_Q"]),
        "median_delta_BS": float(np.nanmedian(anchor.delta_BS)),
        "frac_UQ_captured_by_BS": float(secondary["frac_UQ_captured_by_BS"]),
        "median_delta_FQ": float(np.nanmedian(anchor.delta_FQ)),
        "A_B_median": float(np.nanmedian(anchor.A_B)),
        "A_B_null_median": float(align["A_B_null_median"]),
        "gamma_fold_cosine_median": float(np.nanmedian(anchor.gamma_fold_cosine)),
        "rho_r2": float(parity["rho_r2_G"]["controlled"]),
        "rho_mse": float(parity["rho_mse_G"]["controlled"]),
        "rho_dmse": float(parity["rho_dMSE_GP"]["controlled"]),
        "rho_dmse_adj": float(secondary.get("rho_KH_dMSE_GP_adj_deltaQ", np.nan)),
        "shuffle_dQ": float(synth["deltas"]["shuffle_dQ"]),
        "label": decision["label"],
        "n_anchors": int(len(anchor)),
        "frac_positive_delta_Q": float(np.mean(np.isfinite(anchor.delta_Q) & (anchor.delta_Q > 0))),
    }
    expected = {
        "median_delta_Q": 0.020581617601622228,
        "rho_KH_delta_Q": 0.111248619551161,
        "median_delta_BS": 0.01960613735261897,
        "frac_UQ_captured_by_BS": 0.9376366120634971,
        "median_delta_FQ": 0.019695030963293697,
        "A_B_median": 2.4271836244410787,
        "A_B_null_median": 0.9783494151098924,
        "gamma_fold_cosine_median": 0.9243428097633845,
        "rho_r2": -0.240,
        "rho_mse": 0.227,
        "rho_dmse": 0.153,
        "rho_dmse_adj": 0.20518047401609046,
        "shuffle_dQ": -7.561259848029579,
        "label": ORIGINAL_LABEL,
    }
    checks = {}
    ok = True
    for k, exp in expected.items():
        got = obs[k]
        if k == "label":
            checks[k] = {"expected": exp, "observed": got, "ok": got == exp}
        else:
            atol = PARITY_ATOL.get(k, 0.01)
            match = bool(np.isfinite(got) and abs(float(got) - float(exp)) <= atol)
            checks[k] = {"expected": exp, "observed": got, "atol": atol, "ok": match}
        ok = ok and bool(checks[k]["ok"])
    synth_gates = synth.get("gates", {})
    return {
        "ok": ok,
        "original_label": decision["label"],
        "original_label_unchanged": decision["label"] == ORIGINAL_LABEL,
        "n_perm": complete.get("n_perm"),
        "n_boot": complete.get("n_boot"),
        "seconds": complete.get("seconds"),
        "obs": obs,
        "expected": expected,
        "checks": checks,
        "synth_gates": synth_gates,
        "synth_deltas": synth.get("deltas"),
        "primary_json": primary,
        "secondary_json": secondary,
        "blocker": None if ok else "phase0_reproduction_mismatch",
    }
