"""Phase 3: control-model audit. Do not select a spec by which result is stronger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FROZEN_D80, FROZEN_D85, FROZEN_CTL, FROZEN_RAW, PARITY_RANKS, SHARED_CORE_CONTROLS
from .inventory import load_frozen_probe
from .parity import load_catalog_mag
from .pipeline import AuditConfig, associate, control_matrix, delta_85_80, spearman_dict, write_df


def _vif(Z: np.ndarray) -> list[float]:
    vifs = []
    for j in range(Z.shape[1]):
        y = Z[:, j]
        X = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
        m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if int(m.sum()) < X.shape[1] + 2:
            vifs.append(float("nan"))
            continue
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        pred = X[m] @ b
        ss_res = float(np.sum((y[m] - pred) ** 2))
        ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs.append(float(1.0 / max(1.0 - r2, 1e-12)))
    return vifs


def run_controls(root: Path, cfg: AuditConfig, inv: dict[str, Any], parity: dict[str, Any]) -> dict[str, Any]:
    out = cfg.resolved(root)
    mm = cfg.mm(root)
    shared = inv["shared_sids"]
    probe = load_frozen_probe(mm)
    old_p = parity["old_panel"]
    new_p = parity["new_panel"]

    frozen_ctrl = probe.set_index("sample_id").reindex(shared)
    # Adaptive controls were recomputed from catalog y; reconstruct from the
    # published adaptive association table is not enough. Rebuild from the
    # same neighbour cache + catalog mag used by the adaptive run.
    knn = inv.get("knn") or {}
    sid_to_ai = inv.get("sid_to_ai") or {}
    y_full = np.load(root / "data_hf/physics/vit_base_test_labels.npz")["mag_r_desi"].astype(float)
    X = np.load(mm / "prepare" / "models" / "vit_base.npz")["X"].astype(np.float32)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    sid_to_row = {int(s): int(i) for i, s in zip(folds.sample_id, folds.local_index)} if "local_index" in folds.columns else {int(s): int(i) for i, s in enumerate(folds.sample_id)}

    harm_rows = []
    for sid in shared:
        ai = sid_to_ai.get(int(sid))
        row = sid_to_row.get(int(sid))
        if ai is None or row is None or "neigh" not in knn:
            harm_rows.append({"sample_id": int(sid), "log_knn_radius": np.nan, "local_label_variance": np.nan, "local_evaluation_count": np.nan})
            continue
        N = knn["neigh"][ai, :2048]
        yn = y_full[N] if np.max(N) < len(y_full) else np.full(len(N), np.nan)
        # N are local indices into the 16384-row X, not full-table sample_ids
        if "sample_ids" in knn:
            full_ids = knn["sample_ids"][N]
            yn = y_full[full_ids]
        m = np.isfinite(yn)
        x0 = X[row]
        xn = X[N]
        d = np.linalg.norm(xn - x0[None, :], axis=1)
        harm_rows.append(
            {
                "sample_id": int(sid),
                "log_knn_radius": float(np.log(max(float(np.max(d)), 1e-12))),
                "local_label_variance": float(np.var(yn[m])) if int(m.sum()) >= 2 else np.nan,
                "local_evaluation_count": int(m.sum()),
            }
        )
    harm = pd.DataFrame(harm_rows).set_index("sample_id")

    design_rows = []
    for name, src, yname in (
        ("frozen_discovery", frozen_ctrl, "local_r2"),
        ("harmonized_catalog", harm, "catalog_mag_r_desi"),
    ):
        for c in SHARED_CORE_CONTROLS:
            col = src[c] if c in src.columns else pd.Series(np.nan, index=src.index)
            design_rows.append(
                {
                    "design": name,
                    "control": c,
                    "n": int(col.notna().sum()),
                    "mean": float(col.mean()) if col.notna().any() else np.nan,
                    "std": float(col.std()) if col.notna().any() else np.nan,
                    "missing_filled_with_zero": True,
                    "rank_transform": "yes (partial Spearman residualizes ranked Z)",
                    "residualization_order": "rank(x), rank(y), rank(Z); OLS residual; Spearman of residuals",
                    "standardization": "none beyond ranking",
                    "nonlinear_terms": "none",
                    "strata": "none",
                    "y": yname,
                }
            )
    write_df(out / "control_design_comparison.csv", pd.DataFrame(design_rows), force=cfg.force)

    y_old = frozen_ctrl["local_r2"].to_numpy(float)
    y_new = load_catalog_mag(root, shared)
    Z_f = control_matrix(frozen_ctrl.reset_index())
    Z_h = control_matrix(harm.reset_index())

    sens_rows = []
    side_rows = []
    for d in PARITY_RANKS:
        kh = old_p[old_p.d == d].set_index("sample_id").reindex(shared)["K_H_cross"].to_numpy(float)
        raw_disc = associate(kh, y_old, None)
        ctl_disc = associate(kh, y_old, Z_f)
        raw_cat = associate(kh, y_new, None)
        ctl_harm = associate(kh, y_new, Z_h)
        ctl_disc_on_cat = associate(kh, y_new, Z_f)
        side_rows.append(
            {
                "d": int(d),
                "raw_discovery_local_r2": raw_disc["raw"],
                "frozen_discovery_control_local_r2": ctl_disc["controlled"],
                "raw_catalog_mag": raw_cat["raw"],
                "harmonized_control_catalog_mag": ctl_harm["controlled"],
                "frozen_controls_on_catalog_mag": ctl_disc_on_cat["controlled"],
                "n": raw_disc["n"],
                "frozen_raw_published": FROZEN_RAW[d],
                "frozen_ctl_published": FROZEN_CTL[d],
                "discovery_parity_label": "frozen_discovery_control uses local_r2 + local_probe_fields controls",
            }
        )
        # leave-one-control-out on frozen discovery
        if Z_f is not None:
            for j, c in enumerate(SHARED_CORE_CONTROLS):
                Zj = np.delete(Z_f, j, axis=1)
                rec = associate(kh, y_old, Zj)
                sens_rows.append(
                    {
                        "d": int(d),
                        "y": "local_r2",
                        "dropped": c,
                        "controlled": rec["controlled"],
                        "raw": raw_disc["raw"],
                    }
                )
            rec0 = associate(kh, y_old, Z_f)
            sens_rows.append({"d": int(d), "y": "local_r2", "dropped": "none", "controlled": rec0["controlled"], "raw": raw_disc["raw"]})
            for j, c in enumerate(SHARED_CORE_CONTROLS):
                rho_xk = spearman_dict(kh, Z_f[:, j])["rho"]
                rho_yk = spearman_dict(y_old, Z_f[:, j])["rho"]
                sens_rows.append(
                    {
                        "d": int(d),
                        "y": "local_r2",
                        "dropped": f"corr_with_{c}",
                        "controlled": rho_xk,
                        "raw": rho_yk,
                    }
                )
            vifs = _vif(Z_f)
            for c, v in zip(SHARED_CORE_CONTROLS, vifs):
                sens_rows.append({"d": int(d), "y": "local_r2", "dropped": f"vif_{c}", "controlled": v, "raw": np.nan})

    write_df(out / "control_sensitivity.csv", pd.DataFrame(sens_rows), force=cfg.force)
    side = pd.DataFrame(side_rows)
    write_df(out / "discovery_curves_side_by_side.csv", side, force=cfg.force)

    rho_f = {int(r.d): float(r.frozen_discovery_control_local_r2) for _, r in side.iterrows()}
    rho_h = {int(r.d): float(r.harmonized_control_catalog_mag) for _, r in side.iterrows()}
    rho_r = {int(r.d): float(r.raw_discovery_local_r2) for _, r in side.iterrows()}
    return {
        "side": side,
        "delta_frozen_ctl": delta_85_80(rho_f, FROZEN_D80, FROZEN_D85),
        "delta_harmonized_ctl": delta_85_80(rho_h, FROZEN_D80, FROZEN_D85),
        "delta_frozen_raw": delta_85_80(rho_r, FROZEN_D80, FROZEN_D85),
        "sign_reversal_control": bool(
            np.sign(side.loc[side.d == 12, "raw_discovery_local_r2"].iloc[0])
            != np.sign(side.loc[side.d == 12, "frozen_discovery_control_local_r2"].iloc[0])
        )
        if len(side[side.d == 12])
        else False,
    }
