"""Stage 2: rank-conditioned sphere-normal curvature. Frozen K_H definition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from geometry.physics_activation_atlas.nested_dimension_curvature import _fit_rank, nested_pca_frame
from geometry.physics_activation_atlas.paths import resolve_path
from geometry.physics_quadratic_predictive_dimension.algebra import n_quad_features

from .config import R_H_FAIL, SOURCE_NDC, VALID_FRAC_FAIL
from .pipeline import AdaptiveProbeConfig, _done, write_df, write_json

METRIC_COLS = ["K_H_cross", "K_aniso_cross", "K_dir_cross", "R_H", "dS"]


def reuse_physics_kh(root: Path, sids: list[int], ds: list[int], k: int) -> pd.DataFrame | None:
    """Return whatever cached nested ranks exist. Partial reuse is allowed."""
    p = resolve_path(root, SOURCE_NDC) / "nested_curvature_metrics.parquet"
    if not p.exists():
        return None
    ndc = pd.read_parquet(p)
    if "k" in ndc.columns:
        ndc = ndc[ndc.k == k]
    if "model" in ndc.columns:
        ndc = ndc[ndc.model == "vit_base"]
    ndc = ndc[ndc.sample_id.isin(sids) & ndc.d.isin(ds)]
    if not len(ndc):
        return None
    keep = [c for c in METRIC_COLS if c in ndc.columns]
    agg = ndc.groupby(["sample_id", "d"], as_index=False)[keep].mean(numeric_only=True)
    sizes = ndc.groupby(["sample_id", "d"]).size().reset_index(name="n_splits_ok")
    agg = agg.merge(sizes, on=["sample_id", "d"])
    agg["k"] = k
    agg["m_d"] = [n_quad_features(int(d)) for d in agg.d]
    agg["n_tr"] = k // 2
    agg["df_ratio"] = agg["m_d"] / max(k // 2, 1)
    agg["source"] = "reused_nested_curvature"
    return agg


def _empty_row(sid: int, d: int, k: int) -> dict[str, Any]:
    return {
        "sample_id": int(sid),
        "d": int(d),
        "k": k,
        "K_H_cross": np.nan,
        "K_aniso_cross": np.nan,
        "K_dir_cross": np.nan,
        "R_H": np.nan,
        "dS": np.nan,
        "n_splits_ok": 0,
        "m_d": n_quad_features(d),
        "n_tr": k // 2,
        "df_ratio": n_quad_features(d) / max(k // 2, 1),
        "source": "fit_rank",
    }


def _rows_from_fits(sid: int, d: int, k: int, fits: list[dict]) -> dict[str, Any]:
    rec = _empty_row(sid, d, k)
    if not fits:
        return rec
    rec["K_H_cross"] = float(np.nanmean([f["K_H_cross"] for f in fits]))
    rec["K_aniso_cross"] = float(np.nanmean([f["K_aniso_cross"] for f in fits]))
    rec["K_dir_cross"] = float(np.nanmean([f["K_dir_cross"] for f in fits]))
    rec["R_H"] = float(np.nanmean([f["R_H"] for f in fits]))
    rec["dS"] = float(np.nanmean([f["dS"] for f in fits]))
    rec["n_splits_ok"] = int(len(fits))
    return rec


def _have_pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    if df is None or not len(df):
        return set()
    return {(int(s), int(d)) for s, d in zip(df.sample_id, df.d)}


def fit_kh_panel(
    X: np.ndarray,
    neigh: np.ndarray,
    sids: list[int],
    sid_to_row: dict[int, int],
    ds: list[int],
    k: int,
    seed: int,
    device: torch.device,
    n_splits: int = 3,
    *,
    reuse: pd.DataFrame | None = None,
    ckpt: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    parts = []
    if reuse is not None and len(reuse):
        parts.append(reuse)
    if ckpt is not None and ckpt.exists() and not force:
        prev = pd.read_parquet(ckpt)
        parts.append(prev)
        print(f"[adcp] resume curvature checkpoint rows={len(prev)} anchors={prev.sample_id.nunique()}", flush=True)
    have = _have_pairs(pd.concat(parts, ignore_index=True) if parts else None)
    need = [(int(s), int(d)) for s in sids for d in ds if (int(s), int(d)) not in have]
    if not need:
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return out.drop_duplicates(["sample_id", "d"], keep="first")

    missing_by_sid: dict[int, list[int]] = {}
    for sid, d in need:
        missing_by_sid.setdefault(sid, []).append(d)
    sid_index = {int(s): i for i, s in enumerate(sids)}
    new_rows: list[dict[str, Any]] = []
    todo_sids = [s for s in sids if int(s) in missing_by_sid]
    for n_done, sid in enumerate(todo_sids, start=1):
        ai = sid_index[int(sid)]
        miss = sorted(missing_by_sid[int(sid)])
        N = neigh[ai, :k]
        Xloc = X[N].astype(np.float64)
        x0, J, ev, _diag = nested_pca_frame(Xloc, max(miss), device)
        for d in miss:
            if J.shape[1] < d:
                new_rows.append(_empty_row(int(sid), d, k))
                continue
            fits = _fit_rank(Xloc, x0, J, d, k, n_splits, seed, ai)
            new_rows.append(_rows_from_fits(int(sid), d, k, fits))
        if ckpt is not None and (n_done % 8 == 0 or n_done == len(todo_sids)):
            chunk = pd.DataFrame(new_rows)
            allp = pd.concat([*(parts), chunk], ignore_index=True) if parts else chunk
            allp = allp.drop_duplicates(["sample_id", "d"], keep="first")
            write_df(ckpt, allp, force=True)
            print(
                f"[adcp] curvature ckpt {n_done}/{len(todo_sids)} new_anchors "
                f"pairs={len(allp)} last_sid={int(sid)} missing_d={miss[0]}..{miss[-1]}",
                flush=True,
            )
    chunk = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    out = pd.concat([*(parts), chunk], ignore_index=True) if (parts or len(chunk)) else pd.DataFrame()
    return out.drop_duplicates(["sample_id", "d"], keep="first")


def reliability_table(panel: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    rows = []
    for d, g in panel.groupby("d"):
        n = len(g)
        valid = g["K_H_cross"].notna()
        r_h = float(g.R_H.median()) if "R_H" in g.columns else float("nan")
        vf = float(valid.mean()) if n else 0.0
        fail = (np.isfinite(r_h) and r_h < R_H_FAIL) or vf < VALID_FRAC_FAIL
        rows.append(
            {
                "dataset_id": dataset_id,
                "d": int(d),
                "n": n,
                "valid_frac": vf,
                "median_R_H": r_h,
                "median_dS": float(g.dS.median()) if "dS" in g.columns else float("nan"),
                "median_df_ratio": float(g.df_ratio.median()) if "df_ratio" in g.columns else float("nan"),
                "m_d": n_quad_features(int(d)),
                "fail_reliability": bool(fail),
            }
        )
    return pd.DataFrame(rows)


def run_curvature_dataset(
    root: Path,
    cfg: AdaptiveProbeConfig,
    *,
    dataset_id: str,
    X: np.ndarray,
    neigh: np.ndarray,
    sids: list[int],
    sid_to_row: dict[int, int],
    ds: list[int],
    k: int,
    device: torch.device,
    reuse: pd.DataFrame | None,
) -> pd.DataFrame:
    out = cfg.resolved(root)
    ddir = out / "datasets" / dataset_id
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / "cache").mkdir(exist_ok=True)
    path = ddir / "per_anchor_curvature.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    n_splits = 1 if cfg.smoke else 3
    ckpt = ddir / "cache" / "curvature_checkpoint.parquet"
    reused_d = sorted(int(x) for x in reuse.d.unique()) if reuse is not None and len(reuse) else []
    missing_d = [d for d in ds if d not in set(reused_d)]
    print(
        f"[adcp] curvature {dataset_id} anchors={len(sids)} ds={ds[0]}..{ds[-1]} "
        f"reuse_d={reused_d} fit_d={missing_d}",
        flush=True,
    )
    panel = fit_kh_panel(
        X,
        neigh,
        sids,
        sid_to_row,
        ds,
        k,
        cfg.seed,
        device,
        n_splits=n_splits,
        reuse=reuse,
        ckpt=ckpt,
        force=cfg.force,
    )
    panel = panel.copy()
    panel["dataset_id"] = dataset_id
    if "source" not in panel.columns:
        panel["source"] = "fit_rank"
    write_df(path, panel, force=cfg.force)
    rel = reliability_table(panel, dataset_id)
    write_df(ddir / "curvature_reliability.csv", rel, force=cfg.force)
    write_json(
        ddir / "curvature_meta.json",
        {
            "estimator": "K_H_cross = <H_A,H_B>; Q_R removed; B_S sphere-normal; metric whitening",
            "rank_conditioned": True,
            "not_intrinsic_dimension": True,
            "n": int(panel.sample_id.nunique()) if len(panel) else 0,
            "ds": ds,
            "reused_d": reused_d,
            "fitted_d": missing_d,
        },
        force=cfg.force,
    )
    return panel
