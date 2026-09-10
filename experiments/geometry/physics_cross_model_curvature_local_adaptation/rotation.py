"""Readout-rotation audit in the frozen tangent basis. Gate before correlating."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import WEIGHT_COS_RELIABLE
from .inference import _assoc


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def rotation_for_anchor(
    *,
    J: np.ndarray,
    weights: list[dict],
) -> dict[str, Any]:
    """Project foldwise G/P ambient weights into J. Do not use test-fold observations."""
    g = [r for r in weights if r.get("model") == "G"]
    p = [r for r in weights if r.get("model") == "P"]
    J = np.asarray(J, dtype=np.float64)
    rec = {
        "n_G_folds": len(g),
        "n_P_folds": len(p),
        "P_fold_cosine_med": float("nan"),
        "direction_reliable": False,
        "cos_vG_vP": float("nan"),
        "angle_rad": float("nan"),
        "norm_ratio": float("nan"),
        "vP_norm": float("nan"),
        "vG_norm": float("nan"),
    }
    if len(p) < 2:
        return rec
    pcos = []
    vP = []
    for i in range(len(p)):
        vP.append(J.T @ np.asarray(p[i]["w"], dtype=np.float64))
        for j in range(i + 1, len(p)):
            pcos.append(_cos(p[i]["w"], p[j]["w"]))
    rec["P_fold_cosine_med"] = float(np.median(pcos)) if pcos else float("nan")
    rec["direction_reliable"] = bool(
        np.isfinite(rec["P_fold_cosine_med"]) and rec["P_fold_cosine_med"] >= WEIGHT_COS_RELIABLE
    )
    vP_mean = np.mean(np.stack(vP, axis=0), axis=0)
    rec["vP_norm"] = float(np.linalg.norm(vP_mean))
    if g:
        vG = np.mean(np.stack([J.T @ np.asarray(r["w"], dtype=np.float64) for r in g], axis=0), axis=0)
        rec["vG_norm"] = float(np.linalg.norm(vG))
        rec["cos_vG_vP"] = _cos(vG, vP_mean)
        rec["angle_rad"] = float(np.arccos(np.clip(rec["cos_vG_vP"], -1.0, 1.0))) if np.isfinite(rec["cos_vG_vP"]) else float("nan")
        rec["norm_ratio"] = rec["vP_norm"] / rec["vG_norm"] if rec["vG_norm"] > 0 else float("nan")
    return rec


def rotation_associations(df: pd.DataFrame) -> dict[str, Any]:
    """Correlate rotation with KH / Δ_adapt only on the predeclared reliability gate."""
    out: dict[str, Any] = {
        "stability_gate": WEIGHT_COS_RELIABLE,
        "n_reliable": int(df.direction_reliable.fillna(False).sum()) if "direction_reliable" in df.columns else 0,
        "n_total": int(len(df)),
    }
    rel = df[df.direction_reliable.fillna(False)].copy() if "direction_reliable" in df.columns else df.iloc[0:0]
    if len(rel) >= 32 and "angle_rad" in rel.columns:
        out["rho_KH_angle_reliable"] = _assoc(rel, "angle_rad")
        if "delta_adapt" in rel.columns:
            out["rho_angle_delta_adapt_reliable"] = _assoc(rel, "delta_adapt")
            # split-fold style: this table already uses fold-median weights, not test residuals
            out["rho_KH_delta_adapt_reliable"] = _assoc(rel, "delta_adapt")
    else:
        out["rho_KH_angle_reliable"] = {"controlled": float("nan"), "n": int(len(rel))}
    return out
