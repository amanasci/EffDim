"""Phase 5: anchor-level inferential sample sizes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MIN_VALID_ANCHORS, SHARED_CORE_CONTROLS
from .parity import load_catalog_label
from .pipeline import AuditConfig, write_df


PHYSICS_LABELS = ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass", "sfr")


def underpowered(n: int, min_n: int = MIN_VALID_ANCHORS) -> bool:
    return int(n) < int(min_n)


def run_sample_sizes(root: Path, cfg: AuditConfig, inv: dict[str, Any], desi: dict[str, Any]) -> pd.DataFrame:
    out = cfg.resolved(root)
    adcp = cfg.adcp(root)
    panel = pd.read_parquet(adcp / "per_anchor_curvature.parquet")
    rows = []

    phys = panel[panel.dataset_id == "physics_vit_base"]
    sids = [int(s) for s in phys.sample_id.drop_duplicates().tolist()]
    z = np.load(root / "data_hf/physics/vit_base_test_labels.npz")
    n_full = int(len(z["mag_r_desi"]))
    for name in PHYSICS_LABELS:
        y = np.asarray(z[name], dtype=float)
        if name == "stellar_mass":
            y = y.copy()
            y[y == -99.0] = np.nan
        ya = load_catalog_label(root, name, sids)
        n_lab_full = int(np.isfinite(y).sum())
        n_anc = int(np.isfinite(ya).sum())
        n_ctrl = n_anc  # same mask; controls filled with 0
        rows.append(
            {
                "dataset_id": "physics_vit_base",
                "label": name,
                "full_dataset_rows": n_full,
                "valid_labelled_rows": n_lab_full,
                "total_curvature_anchors": len(sids),
                "valid_labelled_anchors": n_anc,
                "controlled_analysis_anchors": n_ctrl,
                "n_controls": len(SHARED_CORE_CONTROLS),
                "residual_df": max(n_ctrl - (1 + len(SHARED_CORE_CONTROLS)), 0),
                "missingness": f"{len(sids) - n_anc}/{len(sids)} anchors unlabeled",
                "underpowered": underpowered(n_anc),
                "scientific": True,
            }
        )

    desi_p = panel[panel.dataset_id == "desi_vit_base_hsc"]
    if len(desi_p):
        dsids = [int(s) for s in desi_p.sample_id.drop_duplicates().tolist()]
        lab = np.load(adcp / "cache" / "desi_smith42_labels.npz", allow_pickle=True)
        n_desi = int(len(lab["spec_z"]))
        for key, canon in (("spec_z", "spec_z"), ("mag_r", "mag_r")):
            y = np.asarray(lab[key], dtype=float)
            ya = np.asarray([y[s] if 0 <= s < len(y) else np.nan for s in dsids], dtype=float)
            n_anc = int(np.isfinite(ya).sum())
            rows.append(
                {
                    "dataset_id": "desi_vit_base_hsc",
                    "label": canon,
                    "full_dataset_rows": n_desi,
                    "valid_labelled_rows": int(np.isfinite(y).sum()),
                    "total_curvature_anchors": len(dsids),
                    "valid_labelled_anchors": n_anc,
                    "controlled_analysis_anchors": n_anc,
                    "n_controls": len(SHARED_CORE_CONTROLS),
                    "residual_df": max(n_anc - (1 + len(SHARED_CORE_CONTROLS)), 0),
                    "missingness": f"{len(dsids) - n_anc}/{len(dsids)} anchors unlabeled",
                    "underpowered": underpowered(n_anc),
                    "scientific": False,
                    "note": desi.get("status", "desi_label_alignment_unresolved"),
                }
            )

    df = pd.DataFrame(rows)
    write_df(out / "association_sample_sizes.csv", df, force=cfg.force)
    return df
