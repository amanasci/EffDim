"""Phase 3: targeted k-scale sensitivity on the hash-selected 128-anchor subset.

Chart positions are the geometry-only PREDECLARED_D. Correlations are not used to pick ranks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import CONTROLS, MODEL, N_SCALE_ANCHORS, PREDECLARED_D, PRIMARY_K, SCALE_KS, SEED
from .pipeline import ValConfig, hash_select_cprs, resolve_path, write_df, write_json
from .schema import PRIMARY


def _load_scale_cache(cprs: Path) -> pd.DataFrame:
    recs = []
    cache = cprs / "cache"
    if cache.is_dir():
        for p in sorted(cache.glob("scale_vit_base_*_k*.parquet")):
            recs.append(pd.read_parquet(p))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def run_scale(root: Path, cfg: ValConfig) -> dict[str, Any]:
    out = cfg.resolved(root)
    cprs = resolve_path(root, cfg.cprs_dir)
    mm = resolve_path(root, cfg.mm_dir)
    qpd = resolve_path(root, cfg.qpd_dir)
    write_json(out / "predeclared_chart_positions.json", {
        "rule": "geometry-only held-out variance/risk family; not correlation-maximised",
        "positions": PREDECLARED_D,
        "tau_80": 12,
        "tau_85": 20,
        "middle": "midpoint of the τ∈[0.80,0.85] family",
        "k": list(SCALE_KS),
    }, force=cfg.force)

    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == MODEL) & (geo.target == "mag_r_desi") & (geo.neighbourhood == "model")]
    sids = sorted(panel.sample_id.astype(int).unique())
    scale_sids = hash_select_cprs(sids, N_SCALE_ANCHORS if not cfg.smoke else min(8, len(sids)), seed=SEED)

    existing = pd.read_csv(cprs / "scale_sensitivity.csv")
    existing["analysis"] = "reused_rank_sweep_cache"
    existing["target_id"] = PRIMARY.value

    # Common-anchor k=2048 subset of the primary 512-anchor result
    kh = panel.groupby(["sample_id", "d"], as_index=False)["K_H_cross"].mean()
    g2048 = geo[geo.scale_k == PRIMARY_K].drop_duplicates("sample_id")
    rows = []
    for d in PREDECLARED_D.values():
        sub = kh[kh.d == d].merge(g2048, on="sample_id", how="inner")
        sub = sub[sub.sample_id.isin(scale_sids)]
        rec = associate(sub.K_H_cross.to_numpy(float), sub.local_r2.to_numpy(float), control_matrix(sub))
        rec.update({"k": PRIMARY_K, "d": int(d), "n": rec["n"], "source": "common_anchor_k2048_subset", "target_id": PRIMARY.value, "analysis": "sensitivity"})
        rows.append(rec)
    common = pd.DataFrame(rows)

    caches = _load_scale_cache(cprs)
    rel_rows = []
    if len(caches):
        keep = [c for c in ("K_H_cross", "R_H", "dS") if c in caches.columns]
        agg = caches.groupby(["sample_id", "d", "k"], as_index=False)[keep].mean(numeric_only=True)
        for k in SCALE_KS:
            if k == PRIMARY_K:
                continue
            gk = geo.copy()
            scales = sorted(gk.scale_k.unique())
            sk = int(k) if int(k) in set(int(s) for s in scales) else min(scales, key=lambda s: abs(int(s) - int(k)))
            g = gk[gk.scale_k == sk].drop_duplicates("sample_id")
            for d in PREDECLARED_D.values():
                sub = agg[(agg.k == k) & (agg.d == d)].merge(g, on="sample_id", how="inner")
                if not len(sub):
                    continue
                rec = associate(sub.K_H_cross.to_numpy(float), sub.local_r2.to_numpy(float), control_matrix(sub))
                rec.update({
                    "k": int(k),
                    "d": int(d),
                    "probe_scale_k": int(sk),
                    "source": "refit_scale_cache",
                    "target_id": PRIMARY.value,
                    "analysis": "sensitivity",
                    "R_H_med": float(sub.R_H.median()) if "R_H" in sub.columns else float("nan"),
                    "n_cache": int(sub.sample_id.nunique()),
                })
                rows.append(rec)
                rel_rows.append({"k": int(k), "d": int(d), "R_H_med": rec["R_H_med"], "n": rec["n_cache"], "fail_reliability": bool(rec["R_H_med"] < 0.20) if np.isfinite(rec["R_H_med"]) else True})

    ve = pd.read_csv(cprs / "variance_explained.csv")
    qpd_scale = pd.read_csv(qpd / "scale_sensitivity.csv") if (qpd / "scale_sensitivity.csv").exists() else pd.DataFrame()
    mapped = []
    for pos, d in PREDECLARED_D.items():
        r2 = float(ve.loc[ve.d == d, "r2_L_pooled"].iloc[0]) if d in set(ve.d.astype(int)) else float("nan")
        mapped.append({"position": pos, "d_at_k2048": int(d), "r2_L_k2048": r2})

    scale_tbl = pd.concat([existing, common, pd.DataFrame(rows)], ignore_index=True, sort=False)
    if scale_tbl.source.eq("predeclared_pending").any() if "source" in scale_tbl.columns else False:
        scale_tbl = scale_tbl[scale_tbl.source != "predeclared_pending"]
    write_df(out / "scale_sensitivity.csv", scale_tbl, force=cfg.force)
    write_df(out / "scale_reliability.csv", pd.DataFrame(rel_rows), force=cfg.force)
    write_json(out / "scale_variance_map.json", {"positions": mapped, "qpd_scale_rows": int(len(qpd_scale))}, force=cfg.force)
    pending = int((scale_tbl.source == "predeclared_pending").sum()) if "source" in scale_tbl.columns else 0
    return {"n_scale_anchors": len(scale_sids), "n_pending": pending, "common_n": int(common.n.min()) if len(common) else 0}
