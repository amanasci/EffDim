"""Cached-decoder diagnostics. No new training. Weights were never written."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import SEED_SPEARMAN_OK


def load_decoder_seeds(cell_dir: Path) -> pd.DataFrame | None:
    p = cell_dir / "decoder_seeds.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def pairwise_spearman(df: pd.DataFrame, col: str) -> dict:
    seeds = sorted(df["seed"].unique())
    if len(seeds) < 2:
        return {"n_seeds": int(len(seeds)), "pairs": [], "mean_rho": float("nan")}
    piv = df.pivot_table(index="sample_id", columns="seed", values=col)
    pairs = []
    rhos = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            r = spearmanr(piv[a], piv[b], nan_policy="omit").correlation
            r = float(r) if r is not None and np.isfinite(r) else float("nan")
            pairs.append({"seed_a": int(a), "seed_b": int(b), "rho": r})
            rhos.append(r)
    return {"n_seeds": int(len(seeds)), "pairs": pairs, "mean_rho": float(np.nanmean(rhos)) if rhos else float("nan")}


def seed_decomposition(df: pd.DataFrame, recon: dict | None) -> dict:
    out = {
        "weights_cached": False,
        "limitation": "no decoder .pt/.pth in audit cells; D1/D2 unavailable",
        "spearman_H": pairwise_spearman(df, "H_norm_D"),
        "spearman_Kdir": pairwise_spearman(df, "K_dir_D"),
        "mean_H_cosine": float(np.nanmean(df["D_H_cosine"])) if "D_H_cosine" in df.columns else float("nan"),
        "mean_tensor_cos": float(np.nanmean(df["D_tensor_dir_cos"])) if "D_tensor_dir_cos" in df.columns else float("nan"),
        "recon": {},
    }
    if recon:
        for sd, rec in recon.items():
            if sd in ("error",) or not isinstance(rec, dict):
                continue
            ho = rec.get("holdout", {})
            out["recon"][str(sd)] = {"holdout_r2": ho.get("r2"), "holdout_mse": ho.get("mse")}
    r2s = [v["holdout_r2"] for v in out["recon"].values() if v.get("holdout_r2") is not None]
    out["recon_r2_range"] = [float(min(r2s)), float(max(r2s))] if r2s else None
    out["curvature_disagrees_despite_recon"] = bool(
        out["spearman_Kdir"]["mean_rho"] < SEED_SPEARMAN_OK
        and r2s
        and min(r2s) > 0.9
    )
    return out


def d0_from_cached(anchors: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    sub = anchors[anchors["sample_id"].isin(ids)].copy()
    keep = [
        c
        for c in (
            "sample_id",
            "H_norm_D",
            "K_dir_D",
            "K_tf_D",
            "K_H_D",
            "D_H_cosine",
            "D_tensor_dir_cos",
            "H_norm_T1",
            "K_dir_T1",
            "K_tf_T1",
        )
        if c in sub.columns
    ]
    return sub[keep]
