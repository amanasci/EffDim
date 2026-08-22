"""Reproduce common-128 scale correlations before any factorial inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix
from geometry.physics_curvature_probe_submission_validation.schema import PRIMARY, assert_probe_performance

from .config import (
    CONTROLS,
    PARITY_ATOL,
    PARITY_COMMON128_D16,
    PARITY_FULL512_D16,
    PRIMARY_D,
    PRIMARY_K,
    ExpConfig,
)
from .data import control_row, geo_row
from .io_util import write_json
from .probe import neighbourhood_metrics, assert_not_catalog


def run_parity(bundle: dict[str, Any], cfg: ExpConfig, out) -> dict[str, Any]:
    assert_probe_performance(PRIMARY.value)
    panel = bundle["panel"]
    geo = bundle["geo"]
    sids = bundle["scale_sids"]
    y, yhat = bundle["y"], bundle["yhat"]
    neigh, sid_to_ai = bundle["neigh"], bundle["sid_to_ai"]

    r2_2048 = []
    catalog_at_anchor = []
    cat = np.asarray(bundle["catalog"], dtype=float)
    folds_sid = bundle.get("folds_sample_id")
    for sid in sids:
        ai = sid_to_ai[int(sid)]
        met = neighbourhood_metrics(y, yhat, neigh[ai, :PRIMARY_K])
        r2_2048.append(met["r2_local"])
        if folds_sid is not None:
            hits = np.where(folds_sid == int(sid))[0]
            catalog_at_anchor.append(float(cat[hits[0]]) if len(hits) else float("nan"))
        elif ai < len(cat):
            catalog_at_anchor.append(float(cat[ai]))
        else:
            catalog_at_anchor.append(float("nan"))
    r2_2048 = np.asarray(r2_2048, float)
    assert_not_catalog(r2_2048, np.asarray(catalog_at_anchor, float))

    rows = []
    kh = panel.groupby(["sample_id", "d"], as_index=False)["K_H_cross"].mean()
    # Recompute common-128 at k=2048 from frozen K_H and fixed probe geography.
    expected2048 = PARITY_COMMON128_D16[PRIMARY_K]
    sub = kh[kh.d == PRIMARY_D].merge(
        pd.DataFrame({"sample_id": sids, "r2_local": r2_2048}),
        on="sample_id",
    )
    ctrl = pd.DataFrame([control_row(geo, int(s), PRIMARY_K) for s in sub.sample_id])
    for c in CONTROLS:
        sub[c] = ctrl[c].to_numpy() if c in ctrl else np.nan
    rec = associate(sub.K_H_cross.to_numpy(float), sub.r2_local.to_numpy(float), control_matrix(sub))
    rec.update({"k": int(PRIMARY_K), "n": int(len(sub)), "expected": expected2048, "source": "common128_k2048_recomputed"})
    rec["match"] = abs(float(rec["controlled"]) - expected2048) <= PARITY_ATOL
    rows.append(rec)

    # Remaining common-128 cells from the frozen validation scale table (read-only).
    scale_p = bundle["val"] / "scale_sensitivity.csv"
    if scale_p.exists():
        sc = pd.read_csv(scale_p)
        for k, expected in PARITY_COMMON128_D16.items():
            hit = sc[(sc.k == int(k)) & (sc.d == PRIMARY_D)]
            if int(k) == PRIMARY_K:
                hit = hit[hit.source.astype(str).str.contains("common", na=False)]
            else:
                hit = hit[hit.source.astype(str).str.contains("refit_scale", na=False)]
            if not len(hit):
                continue
            got = float(hit.iloc[-1].controlled)
            rows.append(
                {
                    "k": int(k),
                    "controlled": got,
                    "expected": expected,
                    "n": int(hit.iloc[-1].get("n", 128)),
                    "source": str(hit.iloc[-1].source),
                    "match": abs(got - expected) <= PARITY_ATOL,
                }
            )

    full = None
    full_p = bundle["val"] / "metric_associations.csv"
    if full_p.exists():
        assoc = pd.read_csv(full_p)
        hit = assoc[(assoc.d == PRIMARY_D) & (assoc.target_id == PRIMARY.value) & (assoc.slice_mode == "full")]
        if len(hit):
            full = float(hit.iloc[0].controlled)

    report = {
        "ok": bool(all(r.get("match", False) for r in rows if r.get("k") in PARITY_COMMON128_D16) and (full is None or abs(full - PARITY_FULL512_D16) <= PARITY_ATOL)),
        "primary_target": PRIMARY.value,
        "catalog_field": "mag_r_desi_catalog_value",
        "common128": rows,
        "full512_k2048_d16": full,
        "full512_expected": PARITY_FULL512_D16,
        "note": "n=128 estimates are not compared for significance against n=512",
        "anchors": [int(s) for s in sids],
        "n_anchors": len(sids),
    }
    write_json(out / "parity.json", report, force=True)
    if not report["ok"] and not cfg.smoke:
        raise RuntimeError(f"parity failed: {report}")
    return report


run_parity = run_parity
