"""Reproduce the completed trace table and historical K_dir storage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate  # noqa: F401

from .config import (
    PARITY_ATOL,
    PARITY_VITB_DMSE,
    PARITY_VITB_MSE_G,
    PARITY_VITB_R2,
    TRACE_EXPECTED,
    TRACE_PARITY_ATOL,
    ExpConfig,
)
from .data import load_frozen_kh, load_frozen_probes
from .inference import _assoc
from .io_util import write_json


def _match(obs: float, exp: float, atol: float) -> bool:
    return bool(np.isfinite(obs) and abs(float(obs) - float(exp)) <= atol)


def run_trace_parity(shared: dict, sids: list[int], models: list[str], cfg: ExpConfig) -> dict[str, Any]:
    per = {}
    ok = True
    for m in models:
        probes = load_frozen_probes(shared, m, sids)
        kh = load_frozen_kh(shared, m, sids)
        df = probes.merge(kh, on="sample_id", how="inner")
        if cfg.smoke and len(df) < 8:
            continue
        rec = {
            "n": int(len(df)),
            "C_R2": _assoc(df, "K_H_cross", "r2_G")["controlled"],
            "C_G": _assoc(df, "K_H_cross", "mse_G")["controlled"],
            "C_P": _assoc(df, "K_H_cross", "mse_P")["controlled"],
            "C_A": _assoc(df, "K_H_cross", "delta_adapt")["controlled"],
            "mean_delta_adapt": float(df.delta_adapt.mean()),
        }
        rec["A"] = float(rec["C_G"] - rec["C_P"])
        exp = TRACE_EXPECTED[m]
        rec["expected"] = exp
        rec["match"] = {
            k: _match(rec[k], exp[k], PARITY_ATOL if m == "vit_base" and k in ("C_R2", "C_G", "C_A") else 1e-6)
            for k in ("C_R2", "C_G", "C_P", "C_A", "A", "mean_delta_adapt")
        }
        if m == "vit_base":
            rec["match"]["named_r2"] = _match(rec["C_R2"], PARITY_VITB_R2, PARITY_ATOL)
            rec["match"]["named_mse_G"] = _match(rec["C_G"], PARITY_VITB_MSE_G, PARITY_ATOL)
            rec["match"]["named_dadapt"] = _match(rec["C_A"], PARITY_VITB_DMSE, PARITY_ATOL)
        if not cfg.smoke:
            ok = ok and all(rec["match"].values())
        per[m] = rec
    return {"ok": bool(ok or cfg.smoke), "per_model": per, "note": "frozen CMCLA probes + KH; no refit"}


def kh_recompute_parity(tables: dict[str, pd.DataFrame], shared: dict, sids: list[int]) -> dict[str, Any]:
    rows = {}
    ok = True
    for m, df in tables.items():
        frozen = load_frozen_kh(shared, m, sids)
        merged = df.merge(frozen, on="sample_id", suffixes=("_new", "_frozen"))
        if "K_H_cross_new" not in merged.columns:
            continue
        d = np.abs(merged.K_H_cross_new.to_numpy(float) - merged.K_H_cross_frozen.to_numpy(float))
        rec = {
            "n": int(len(merged)),
            "max_abs": float(np.nanmax(d)) if len(d) else float("nan"),
            "median_abs": float(np.nanmedian(d)) if len(d) else float("nan"),
        }
        rec["ok"] = bool(np.isfinite(rec["max_abs"]) and rec["max_abs"] <= max(TRACE_PARITY_ATOL, 1e-6))
        # ViT-B reused NDC, which already matches CMCLA at 1e-12 for sid 0; allow 1e-8
        if m == "vit_base":
            rec["ok"] = bool(np.isfinite(rec["max_abs"]) and rec["max_abs"] <= 1e-8)
        ok = ok and rec["ok"]
        rows[m] = rec
    return {"ok": ok, "per_model": rows}
