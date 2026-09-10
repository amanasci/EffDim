"""Load read-only prior audits. Never write into those trees."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DUAL_OUT, FAIL_OUT, FCR_VIT, HEADLINES, PATCH_OUT, PHYS_OUT, REPRO_OUT


def _repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(rel: str, name: str) -> dict | None:
    p = _repo() / rel / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_prior() -> dict:
    repo = _repo()
    dual = repo / DUAL_OUT
    repro = repo / REPRO_OUT
    files = {
        "dual_per_anchor": dual / "per_anchor_metrics.parquet",
        "dual_d_full": dual / "decoder_full_metrics.csv",
        "dual_d_res": dual / "decoder_residual_metrics.csv",
        "dual_q_t2": dual / "quadratic_T2_metrics.csv",
        "dual_q_t3": dual / "quadratic_T3_metrics.csv",
        "dual_q_pw": dual / "quadratic_pointwise_metrics.csv",
        "dual_scal": dual / "intrinsic_curvature_metrics.csv",
        "dual_decision": dual / "decision.json",
        "repro_cells": repro / "reproduction_cells.csv",
        "phys_decision": repo / PHYS_OUT / "decision.json",
        "fail_decision": repo / FAIL_OUT / "decision.json" if (repo / FAIL_OUT / "decision.json").exists() else None,
        "patch_complete": repo / PATCH_OUT / "COMPLETE.json" if (repo / PATCH_OUT / "COMPLETE.json").exists() else None,
        "fcr_vit": repo / FCR_VIT,
        "phys_train": repo / PHYS_OUT / "decoder_training_manifest.json",
    }
    missing = [k for k, p in files.items() if p is not None and not Path(p).exists()]
    pa = pd.read_parquet(files["dual_per_anchor"])
    d_full = pd.read_csv(files["dual_d_full"])
    d_res = pd.read_csv(files["dual_d_res"])
    q_t2 = pd.read_csv(files["dual_q_t2"])
    q_t3 = pd.read_csv(files["dual_q_t3"])
    q_pw = pd.read_csv(files["dual_q_pw"])
    scal = pd.read_csv(files["dual_scal"])
    repro_cells = pd.read_csv(files["repro_cells"])
    dual_dec = json.loads(files["dual_decision"].read_text())
    phys_train = json.loads(files["phys_train"].read_text()) if files["phys_train"].exists() else {}
    fcr = None
    if files["fcr_vit"].exists():
        try:
            fcr = pd.read_parquet(files["fcr_vit"])
        except Exception as exc:  # noqa: BLE001
            fcr = {"error": str(exc)}
    return {
        "missing": missing,
        "per_anchor": pa,
        "d_full": d_full,
        "d_res": d_res,
        "q_t2": q_t2,
        "q_t3": q_t3,
        "q_pw": q_pw,
        "scal": scal,
        "repro_cells": repro_cells,
        "dual_decision": dual_dec,
        "phys_train": phys_train,
        "fcr": fcr,
        "files": {k: str(v) if v is not None else None for k, v in files.items()},
    }


def reproduction_headlines(repro_cells: pd.DataFrame) -> dict:
    def rho(cell):
        row = repro_cells.loc[repro_cells["cell"] == cell].iloc[0]
        return float(row["rho"]), float(row["median_cosine"])

    r2_rho, r2_cos = rho("R2")
    r3_rho, r3_cos = rho("R3")
    return {
        "d_full_cubic_rho": r2_rho,
        "d_full_cubic_cosine": r2_cos,
        "d_full_ridge_rho": r3_rho,
        "d_full_ridge_cosine": r3_cos,
        "cubic_ok": abs(r2_rho - HEADLINES["d_full_cubic_rho"]["expect"]) <= 0.02,
        "ridge_ok": abs(r3_rho - HEADLINES["d_full_ridge_rho"]["expect"]) <= 0.02,
    }


def reuse_manifest(prior: dict) -> dict:
    return {
        "read_only_trees": [DUAL_OUT, REPRO_OUT, PATCH_OUT, FAIL_OUT, PHYS_OUT],
        "files": prior["files"],
        "missing": prior["missing"],
        "n_per_anchor": int(len(prior["per_anchor"])),
        "prior_dual_label": prior["dual_decision"].get("summary_label"),
        "prior_label_not_overwritten": True,
        "new_decoder_checkpoints_in_prior": False,
        "note": "Historical decision.json / COMPLETE.json / reports were not modified.",
    }
