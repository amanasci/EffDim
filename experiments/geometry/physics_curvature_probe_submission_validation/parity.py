"""Phase 1: reproduce frozen d∈{12,16,20} correlations after sample_id alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import (
    CATALOG_FIELD,
    CONTROLS,
    FROZEN_CTL,
    FROZEN_DELTA_20_12,
    FROZEN_RAW,
    MODEL,
    PARITY_ATOL,
    PARITY_RANKS,
    PRIMARY_K,
)
from .pipeline import ValConfig, file_sha_full, resolve_path, write_df, write_json
from .schema import PRIMARY, assert_not_catalog_vector, assert_probe_performance


def load_catalog(root: Path, sids: list[int]) -> np.ndarray:
    z = np.load(root / "data_hf/physics/vit_base_test_labels.npz")
    y = np.asarray(z[CATALOG_FIELD], dtype=float)
    return np.asarray([y[int(s)] if 0 <= int(s) < len(y) else np.nan for s in sids], dtype=float)


def _agg_kh(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ("K_H_cross", "K_aniso_cross", "R_H", "dS") if c in df.columns]
    return df.groupby(["sample_id", "d"], as_index=False)[keep].mean(numeric_only=True)


def run_parity(root: Path, cfg: ValConfig) -> dict[str, Any]:
    assert_probe_performance(PRIMARY.value)
    out = cfg.resolved(root)
    cprs = resolve_path(root, cfg.cprs_dir)
    mm = resolve_path(root, cfg.mm_dir)
    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == MODEL) & (geo.target == CATALOG_FIELD) & (geo.neighbourhood == "model") & (geo.scale_k == PRIMARY_K)]
    geo = geo.drop_duplicates("sample_id").set_index("sample_id")
    kh = _agg_kh(panel)
    sids = sorted(set(kh.sample_id.astype(int)) & set(geo.index.astype(int)))
    if cfg.smoke:
        sids = sids[:16]
    catalog = load_catalog(root, sids)
    y_r2 = np.asarray([float(geo.loc[s, "local_r2"]) for s in sids], dtype=float)
    assert_not_catalog_vector(y_r2, catalog)

    rows = []
    for d in PARITY_RANKS:
        gd = kh[kh.d == d].set_index("sample_id")
        x = np.asarray([float(gd.loc[s, "K_H_cross"]) if s in gd.index else np.nan for s in sids], dtype=float)
        sub = pd.DataFrame({"K_H_cross": x, "local_r2": y_r2, **{c: [float(geo.loc[s, c]) for s in sids] for c in CONTROLS}})
        rec = associate(sub.K_H_cross.to_numpy(float), sub.local_r2.to_numpy(float), control_matrix(sub))
        rec.update(
            {
                "d": int(d),
                "target_id": PRIMARY.value,
                "frozen_raw": FROZEN_RAW[d],
                "frozen_ctl": FROZEN_CTL[d],
                "raw_match": bool(abs(rec["raw"] - FROZEN_RAW[d]) <= PARITY_ATOL),
                "ctl_match": bool(abs(rec["controlled"] - FROZEN_CTL[d]) <= PARITY_ATOL),
                "n": int(rec["n"]),
            }
        )
        rows.append(rec)
    table = pd.DataFrame(rows)
    delta = float(table.loc[table.d == 20, "controlled"].iloc[0] - table.loc[table.d == 12, "controlled"].iloc[0])
    hashes = {
        "per_anchor_rank_curve": file_sha_full(cprs / "per_anchor_rank_curve.parquet"),
        "local_probe_fields": file_sha_full(mm / "local_probe_fields.parquet"),
        "vit_base_npz": file_sha_full(mm / "prepare" / "models" / "vit_base.npz"),
        "knn2048": file_sha_full(mm / "model_neighbourhoods" / "vit_base_kmax2048.npz"),
    }
    report = {
        "n_anchors": len(sids),
        "k": PRIMARY_K,
        "target_id": PRIMARY.value,
        "catalog_field": "mag_r_desi_catalog_value",
        "not_catalog": True,
        "delta_20_12": delta,
        "delta_20_12_frozen": FROZEN_DELTA_20_12,
        "delta_match": bool(abs(delta - FROZEN_DELTA_20_12) <= 5e-4),
        "all_raw_match": bool(table.raw_match.all()),
        "all_ctl_match": bool(table.ctl_match.all()),
        "exact_parity": bool(table.raw_match.all() and table.ctl_match.all()),
        "hashes": hashes,
        "controls": list(CONTROLS),
        "neighbourhood": "vit_base_kmax2048.npz k=2048, aligned by sample_id",
    }
    write_df(out / "parity_correlations.csv", table, force=cfg.force)
    write_json(out / "parity_report.json", report, force=cfg.force)
    if not cfg.smoke and not report["exact_parity"]:
        raise RuntimeError(f"parity failed: {table.to_dict(orient='records')}")
    return report
