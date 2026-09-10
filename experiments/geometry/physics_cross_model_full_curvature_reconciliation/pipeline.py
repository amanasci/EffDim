"""Orchestrate audit → tests → parity → geometry → inference. No manuscript edits."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import synchronized_inference
from .audit import run_audit
from .config import (
    PARITY_VITB_DMSE,
    PARITY_VITB_MSE_G,
    PARITY_VITB_R2,
    POSITIVE_CONTROL,
    PRIMARY_METRIC,
    ExpConfig,
)
from .data import load_frozen_probes, load_shared, models_used
from .decision import decide
from .figures import write_figures
from .geometry import fit_model_full, reliability_table, vitb_refit_parity
from .inference import curvature_vs_curvature, model_inference, point_associations
from .io_util import assert_not_preserved, peak_rss_mb, platonic_root, resolve_path, write_df, write_json, write_text
from .parity import kh_recompute_parity, run_trace_parity
from .reports import write_manuscript_recommendation, write_methods, write_report, write_reuse_manifest, write_tables
from .tests_unit import run_unit_tests


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("geometry", "figures", "tables", "inference"):
        (out / sub).mkdir(exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=True)

    shared = load_shared(cfg)
    sids = list(shared["sids"])
    models = models_used(cfg)

    audit = run_audit(shared, out)
    if audit.get("blocked") and not cfg.smoke:
        write_json(
            out / "decision.json",
            decide(
                audit_ok=False,
                tests_ok=False,
                parity_ok=False,
                historical_parity_ok=False,
                full_agg={},
                kh_agg={},
                n_models=len(models),
            ),
            force=True,
        )
        return {"ok": False, "blocker": True, "audit": audit}

    tests = run_unit_tests()
    write_json(out / "unit_synthetic_test_results.json", tests, force=True)
    if not tests["ok"] and not cfg.smoke:
        write_text(out / "BLOCKER.md", "# BLOCKER\n\nUnit/synthetic tests failed.\n", force=True)
        return {"ok": False, "blocker": "unit_tests", "tests": tests}

    trace_parity = run_trace_parity(shared, sids, models, cfg)
    write_json(out / "parity_trace_table.json", trace_parity, force=True)

    if cfg.stage == "audit":
        return audit
    if cfg.stage == "tests":
        return tests
    if cfg.stage == "parity":
        return trace_parity

    geo: dict[str, pd.DataFrame] = {}
    if not cfg.skip_geometry:
        for m in models:
            print(f"[fcrr] geometry {m}", flush=True)
            geo[m] = fit_model_full(shared, m, sids, cfg, out)
    else:
        for m in models:
            p = out / "geometry" / m / "anchor_curvature.parquet"
            if p.exists():
                geo[m] = pd.read_parquet(p)

    hist = {"ok": cfg.smoke, "skipped": True}
    if POSITIVE_CONTROL in models and not cfg.smoke:
        print("[fcrr] ViT-B historical K_dir refit parity", flush=True)
        hist = vitb_refit_parity(shared, sids, cfg, n_check=8)
    write_json(out / "historical_kdir_parity.json", hist, force=True)

    khp = kh_recompute_parity(geo, shared, sids) if geo else {"ok": cfg.smoke, "per_model": {}}
    write_json(out / "parity_kh_recompute.json", khp, force=True)

    tables: dict[str, pd.DataFrame] = {}
    pairwise: dict[str, pd.DataFrame] = {}
    points: dict[str, dict] = {}
    for m, gdf in geo.items():
        probes = load_frozen_probes(shared, m, sids)
        df = gdf.merge(probes, on="sample_id", how="inner")
        if "model_x" in df.columns:
            df["model"] = df["model_x"]
        df = df.sort_values("sample_id").reset_index(drop=True)
        if "K_dir_cross" in df.columns and "K_aniso_cross" in df.columns:
            kd = df.K_dir_cross.to_numpy(float)
            ka = df.K_aniso_cross.to_numpy(float)
            df["aniso_share_of_Kdir"] = np.where(np.abs(kd) > 1e-12, ka / kd, np.nan)
        write_df(out / "tables" / f"{m}_per_anchor_curvature.parquet", df, force=True)
        tables[m] = df
        pairwise[m] = curvature_vs_curvature(df)
        points[m] = point_associations(df)
        write_df(out / "tables" / f"{m}_controlled_correlations.csv", _points_to_df(points[m]), force=True)

    if not tables:
        return {"ok": False, "reason": "no_geometry_tables"}

    rel = reliability_table(geo)
    write_df(out / "reliability_table.csv", rel, force=True)

    primaries: dict[str, dict] = {}
    aggs: dict[str, dict] = {}
    infer_metrics = ["K_dir_cross", "K_H_cross", "K_B_cross"]
    if cfg.smoke:
        infer_metrics = ["K_dir_cross", "K_H_cross"]
    for xcol in infer_metrics:
        if any(xcol not in df.columns for df in tables.values()):
            continue
        primaries[xcol] = {}
        for m, df in tables.items():
            print(f"[fcrr] inference {xcol} {m}", flush=True)
            primaries[xcol][m] = model_inference(
                df,
                xcol,
                n_perm=cfg.n_perm_eff(),
                n_boot=cfg.n_boot_eff(),
                seed=cfg.seed + int(md5(f"{xcol}:{m}".encode()).hexdigest()[:6], 16) % 1000,
            )
            write_json(out / "inference" / f"{m}_{xcol}.json", primaries[xcol][m], force=True)
        print(f"[fcrr] joint bootstrap {xcol}", flush=True)
        aggs[xcol] = synchronized_inference(
            tables,
            xcol,
            n_perm=cfg.n_perm_eff(),
            n_boot=cfg.n_boot_eff(),
            seed=cfg.seed + 17 + abs(hash(xcol)) % 100,
        )
        write_json(out / "inference" / f"aggregate_{xcol}.json", aggs[xcol], force=True)

    write_json(out / "per_model_primary.json", primaries, force=True)
    write_json(out / "cross_model_aggregates.json", aggs, force=True)
    write_json(out / "point_associations.json", points, force=True)

    write_tables(out, tables, primaries, aggs, rel, pairwise)
    _write_decomp_and_perms(out, tables, primaries, aggs)
    if "K_dir_cross" in primaries and "K_H_cross" in primaries:
        write_figures(out, primaries["K_dir_cross"], primaries["K_H_cross"], aggs.get("K_dir_cross", {}), tables)

    parity = {
        "ok": bool(trace_parity.get("ok") and khp.get("ok") and (hist.get("ok") or cfg.smoke)),
        "trace_table": trace_parity,
        "kh_recompute": khp,
        "historical_kdir_ok": bool(hist.get("ok")),
        "historical_kdir": hist,
        "vitb_named": {
            "rho_KH_R2G": PARITY_VITB_R2,
            "rho_KH_MSEG": PARITY_VITB_MSE_G,
            "rho_KH_Dadapt": PARITY_VITB_DMSE,
        },
    }
    write_json(out / "parity.json", parity, force=True)

    decision = decide(
        audit_ok=bool(audit.get("ok")),
        tests_ok=bool(tests.get("ok")),
        parity_ok=bool(parity.get("ok")),
        historical_parity_ok=bool(hist.get("ok") or cfg.smoke),
        full_agg=aggs.get("K_dir_cross", {}),
        kh_agg=aggs.get("K_H_cross", {}),
        n_models=len(tables),
    )
    write_json(out / "decision.json", decision, force=True)
    write_methods(out, audit)
    runtime = time.time() - t0
    write_report(
        out,
        decision=decision,
        audit=audit,
        parity=parity,
        primaries=primaries,
        aggs=aggs,
        tests=tests,
        runtime_s=runtime,
    )
    write_manuscript_recommendation(out, decision, aggs)
    write_reuse_manifest(out, audit, parity)

    summary = {
        "decision": decision["label"],
        "reason": decision["reason"],
        "primary_metric": PRIMARY_METRIC,
        "formula": audit.get("formula"),
        "n_anchors": len(sids),
        "models": list(tables),
        "runtime_s": runtime,
        "rss_mb": peak_rss_mb(),
        "tests_ok": tests.get("ok"),
        "parity_ok": parity.get("ok"),
        "aggregates": {k: v.get("observed") for k, v in aggs.items()},
        "output_dir": str(out),
    }
    write_json(out / "summary.json", summary, force=True)

    all_ok = bool(
        audit.get("ok")
        and tests.get("ok")
        and parity.get("ok")
        and decision["label"] != "full_curvature_reconciliation_blocked"
        and not cfg.smoke
    )
    if all_ok:
        write_json(
            out / "COMPLETE.json",
            {"ok": True, "decision": decision["label"], "runtime_s": runtime},
            force=True,
        )
    return summary


def _write_decomp_and_perms(out, tables, primaries, aggs) -> None:
    rows = []
    for m, df in tables.items():
        rec = {"model": m, "n": int(len(df))}
        for col in ("K_dir_cross", "K_B_cross", "K_H_cross", "K_aniso_cross"):
            if col in df.columns:
                rec[f"median_{col}"] = float(np.nanmedian(df[col]))
                rec[f"mean_{col}"] = float(np.nanmean(df[col]))
                rec[f"frac_pos_{col}"] = float((df[col] > 0).mean())
        if "aniso_share_of_Kdir" in df.columns:
            rec["median_aniso_share_of_Kdir"] = float(np.nanmedian(df.aniso_share_of_Kdir))
        if "traceless_fraction" in df.columns:
            rec["median_traceless_energy_fraction"] = float(np.nanmedian(df.traceless_fraction))
            rec["median_trace_energy_fraction"] = float(np.nanmedian(df.trace_fraction)) if "trace_fraction" in df.columns else float("nan")
        rows.append(rec)
    write_df(out / "trace_traceless_decomposition.csv", pd.DataFrame(rows), force=True)
    perm = {}
    for xcol, by_m in primaries.items():
        perm[xcol] = {
            "per_model": {
                m: {k: rec[k] for k in ("C_R2", "C_G", "C_P", "C_R2P", "C_A", "A") if k in rec}
                for m, rec in by_m.items()
            },
            "aggregate": aggs.get(xcol, {}),
        }
    write_json(out / "permutation_results.json", perm, force=True)


def _points_to_df(points: dict) -> pd.DataFrame:
    rows = []
    for xcol, ys in points.items():
        for ycol, rec in ys.items():
            if not isinstance(rec, dict) or "controlled" not in rec:
                rows.append({"x": xcol, "y": ycol, "controlled": rec.get("controlled") if isinstance(rec, dict) else rec})
                continue
            rows.append({"x": xcol, "y": ycol, **{k: rec[k] for k in rec}})
    return pd.DataFrame(rows)
