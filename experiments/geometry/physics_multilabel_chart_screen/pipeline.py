"""Screen frozen charts across eligible physics probe labels."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import (
    associate,
    control_matrix,
    freedman_lane_y,
)
from geometry.physics_quadratic_label_chart_alignment.data import load_chart, tangent_coords
from geometry.physics_quadratic_label_chart_alignment.io_util import (
    assert_not_preserved,
    p_mc,
    platonic_root,
    resolve_path,
    write_df,
    write_json,
)
from geometry.physics_quadratic_label_chart_alignment.pipeline import _process_anchor

from .config import (
    MIN_ANALYZED_ANCHORS,
    MODEL,
    PARITY_ATOL,
    PARITY_MSE,
    PARITY_R2,
    ScreenConfig,
)
from .data import kh_controls, load_label_vectors, load_shared
from .metrics import finite_enough, neighbourhood_metrics
from .inventory import exclusion_table, record_for


def run(cfg: ScreenConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=cfg.force)
    write_json(
        out / "inventory.json",
        {"eligible": [record_for(f) for f in cfg.labels], "excluded": exclusion_table()},
        force=cfg.force,
    )

    shared = load_shared(cfg)
    summaries = []
    for field in cfg.labels:
        rec = _run_one(shared, field, cfg, out)
        summaries.append(rec)

    table = pd.DataFrame(summaries)
    write_df(out / "label_summary.csv", table, force=cfg.force)
    write_json(out / "label_summary.json", summaries, force=cfg.force)

    mag = next((s for s in summaries if s["field"] == "mag_r_desi"), None)
    parity = {"ok": True, "blocker": None}
    if mag is not None and not cfg.smoke:
        ok_r2 = abs(float(mag["rho_ctl_r2_G"]) - PARITY_R2) <= PARITY_ATOL
        ok_mse = abs(float(mag["rho_ctl_mse_G"]) - PARITY_MSE) <= PARITY_ATOL
        parity = {
            "ok": bool(ok_r2 and ok_mse),
            "rho_ctl_r2_G": mag["rho_ctl_r2_G"],
            "rho_ctl_mse_G": mag["rho_ctl_mse_G"],
            "expected_r2": PARITY_R2,
            "expected_mse": PARITY_MSE,
        }
        if not parity["ok"]:
            write_json(out / "BLOCKER.json", {"reason": "mag_r_desi_parity_failed", "parity": parity}, force=True)
            raise RuntimeError("mag_r_desi global-decoding parity failed")

    _write_report(out, summaries, parity, time.time() - t0, cfg)
    payload = {
        "ok": True,
        "parity": parity,
        "n_labels": len(summaries),
        "seconds": time.time() - t0,
        "smoke": cfg.smoke,
    }
    write_json(out / "summary.json", payload, force=cfg.force)
    if not cfg.smoke:
        write_json(out / "COMPLETE.json", payload, force=cfg.force)
    return payload


def _run_one(shared: dict, field: str, cfg: ScreenConfig, out: Path) -> dict:
    meta = record_for(field)
    print(f"[multilabel] {field} ({meta['family']})", flush=True)
    y, yhat = load_label_vectors(shared, field)
    X = shared["X"]
    fold_all = shared["fold"]
    neigh = shared["neigh"]
    ndc = shared["ndc"]

    global_rows = []
    payloads = []
    skipped = 0
    for sid in shared["sids"]:
        ai = shared["sid_to_ai"][int(sid)]
        idx = np.asarray(neigh[ai], dtype=int)
        if not finite_enough(y, idx):
            skipped += 1
            continue
        mets = neighbourhood_metrics(y, yhat, idx)
        kh = kh_controls(shared, int(sid), mets["local_label_variance"], mets["local_evaluation_count"])
        global_rows.append({**kh, **mets, "field": field})
        if not cfg.skip_quadratic:
            payloads.append(
                {
                    "sid": int(sid),
                    "Xloc": np.asarray(X[idx], dtype=np.float64),
                    "yloc": y[idx].copy(),
                    "floc": fold_all[idx].copy(),
                    "g_pred": yhat[idx].copy(),
                    "chart": load_chart(ndc, int(sid)),
                    "seed": cfg.seed,
                    "kh": kh,
                }
            )

    glob = pd.DataFrame(global_rows)
    if len(glob) < (8 if cfg.smoke else MIN_ANALYZED_ANCHORS):
        raise RuntimeError(f"{field}: only {len(glob)} labelled anchors (skipped {skipped})")

    lab_dir = out / field
    lab_dir.mkdir(parents=True, exist_ok=True)
    write_df(lab_dir / "global_anchor_metrics.csv", glob, force=cfg.force)

    g_r2 = _assoc(glob, "r2_G", cfg)
    g_mse = _assoc(glob, "mse_G", cfg)

    q_primary = {}
    if payloads:
        q_primary = _quadratic(payloads, cfg, lab_dir)

    rec = {
        **meta,
        "n_anchors": int(len(glob)),
        "n_skipped_sparse": int(skipped),
        "rho_ctl_r2_G": g_r2["controlled"],
        "rho_raw_r2_G": g_r2["raw"],
        "p_ctl_r2_G": g_r2.get("p_mc"),
        "rho_ctl_mse_G": g_mse["controlled"],
        "rho_raw_mse_G": g_mse["raw"],
        "p_ctl_mse_G": g_mse.get("p_mc"),
        "median_delta_Q": q_primary.get("median_delta_Q"),
        "delta_Q_ci_lo": q_primary.get("delta_Q_ci_lo"),
        "delta_Q_ci_hi": q_primary.get("delta_Q_ci_hi"),
        "frac_positive_delta_Q": q_primary.get("frac_positive_delta_Q"),
        "p_mc_median_delta_Q": q_primary.get("p_mc_median_delta_Q"),
        "rho_KH_delta_Q": q_primary.get("rho_KH_delta_Q"),
        "rho_KH_delta_Q_p_mc": q_primary.get("rho_KH_delta_Q_p_mc"),
        "A_B_median": q_primary.get("A_B_median"),
        "gamma_fold_cosine_median": q_primary.get("gamma_fold_cosine_median"),
    }
    write_json(lab_dir / "summary.json", rec, force=cfg.force)
    return rec


def _assoc(df: pd.DataFrame, ycol: str, cfg: ScreenConfig) -> dict:
    Z = control_matrix(df)
    xv = df.K_H_cross.to_numpy(float)
    yv = df[ycol].to_numpy(float)
    assoc = associate(xv, yv, Z)
    n_perm = cfg.n_perm_eff()
    rng = np.random.default_rng(cfg.seed + 11 + sum(ord(c) for c in ycol))
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = freedman_lane_y(yv, Z, rng)
        null[b] = associate(xv, yp, Z)["controlled"]
    obs = float(assoc["controlled"])
    b_count = int(np.sum(null >= obs)) if obs >= 0 else int(np.sum(null <= obs))
    assoc["p_mc"] = p_mc(b_count, n_perm)
    assoc["n_perm"] = n_perm
    return assoc


def _quadratic(payloads: list[dict], cfg: ScreenConfig, lab_dir: Path) -> dict:
    n_workers = 1 if cfg.smoke else min(8, max(1, (os.cpu_count() or 8) // 2))
    rows = []
    if n_workers == 1:
        for p in payloads:
            row, _ = _process_anchor(p)
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_process_anchor, p) for p in payloads]
            for fut in as_completed(futs):
                row, _ = fut.result()
                rows.append(row)
    anc = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    write_df(lab_dir / "anchor_risks.csv", anc, force=cfg.force)

    dq = anc.delta_Q.to_numpy(float)
    m = np.isfinite(dq)
    dq = dq[m]
    med = float(np.median(dq)) if len(dq) else float("nan")
    rng = np.random.default_rng(cfg.seed)
    B = cfg.n_boot_eff()
    boots = np.array([float(np.median(rng.choice(dq, size=len(dq), replace=True))) for _ in range(B)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p_pos = p_mc(int(np.sum(dq <= 0)), B) if len(dq) and med > 0 else float("nan")

    Z = control_matrix(anc)
    assoc = associate(anc.K_H_cross.to_numpy(float), anc.delta_Q.to_numpy(float), Z)
    n_perm = cfg.n_perm_eff()
    yv = anc.delta_Q.to_numpy(float)
    xv = anc.K_H_cross.to_numpy(float)
    rng2 = np.random.default_rng(cfg.seed + 7)
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = freedman_lane_y(yv, Z, rng2)
        null[b] = associate(xv, yp, Z)["controlled"]
    obs = float(assoc["controlled"])
    b_count = int(np.sum(null >= obs)) if obs >= 0 else int(np.sum(null <= obs))
    return {
        "median_delta_Q": med,
        "delta_Q_ci_lo": float(lo),
        "delta_Q_ci_hi": float(hi),
        "frac_positive_delta_Q": float(np.mean(dq > 0)) if len(dq) else float("nan"),
        "p_mc_median_delta_Q": float(p_pos),
        "rho_KH_delta_Q": obs,
        "rho_KH_delta_Q_p_mc": p_mc(b_count, n_perm),
        "A_B_median": float(np.nanmedian(anc.A_B)),
        "gamma_fold_cosine_median": float(np.nanmedian(anc.gamma_fold_cosine)),
        "n": int(len(anc)),
    }


def _write_report(out: Path, summaries: list[dict], parity: dict, seconds: float, cfg: ScreenConfig) -> None:
    lines = [
        "# Multi-label frozen-chart screen",
        "",
        "Charts, neighbourhoods, and global OOF probes are reused. Geometry is not refit.",
        "Targets are physics-table labels with a proven `sample_id` join.",
        "DESI spectroscopic / DESI imaging labels are excluded.",
        "",
        "## Excluded",
        "",
    ]
    for r in exclusion_table():
        lines.append(f"- `{r['field']}`: {r['reason']}")
    lines += ["", "## Results", ""]
    for s in summaries:
        lines.append(
            f"- **{s['display']}** (`{s['field']}`, {s['family']}, n={s['n_anchors']}): "
            f"ρ_ctl(K_H, R²_G)={s['rho_ctl_r2_G']:.3f}, "
            f"ρ_ctl(K_H, MSE_G)={s['rho_ctl_mse_G']:.3f}"
        )
        if s.get("median_delta_Q") is not None:
            lines.append(
                f"  Δ_Q median={s['median_delta_Q']:.4f}, "
                f"ρ_ctl(K_H, Δ_Q)={s['rho_KH_delta_Q']:.3f}, "
                f"A_B={s['A_B_median']:.3f}"
            )
    lines += [
        "",
        f"mag_r_desi parity ok={parity.get('ok')} (expected ρ_R²={PARITY_R2}, ρ_MSE={PARITY_MSE}).",
        f"Runtime {seconds:.1f}s smoke={cfg.smoke}.",
        "",
        "These secondary labels are a screen, not a replacement for the frozen r-band confirmatory analysis.",
        "Catalog vectors are never used as the global-decoding outcome; only local OOF probe metrics are.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "METHODS.md").write_text(
        "\n".join(
            [
                "# Methods",
                "",
                f"Model `{MODEL}`, k={2048}, d=16, frozen nested charts.",
                "Global decoding: five-fold ridge OOF predictions already stored per target.",
                "Local R² / MSE: geographic score of those fixed predictions in the k=2048 neighbourhood.",
                "Quadratic models: same nested-CV L / UQ / BS path as QLCA (`_process_anchor`).",
                "Controls: log kNN radius (neighbourhood geometry), local label variance (this target), evaluation count.",
                "Inference: rank-space Freedman–Lane permutations and bootstrap medians.",
                "",
            ]
        )
    )
