"""Orchestrate parity → organization → cross-model unique associations → ViT-B pass."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CONTROLS, MODELS, PRIMARY_D, STAB_MIN, ExpConfig
from .data import load_qlca_risks, load_shared, merge_components
from .decision import decide
from .diagnostics import enrich_organization, organization_report
from .figures import write_figures
from .inference import assoc, joint_unique, model_component_inference, point_all_outcomes
from .io_util import (
    assert_not_preserved,
    peak_rss_mb,
    platonic_root,
    resolve_path,
    sha256_file16,
    write_df,
    write_json,
    write_text,
)
from .parity import run_parity
from .probes import verify_quadratic_span
from .reports import summarize_vitb, write_artifact_audit, write_manuscript_recommendation, write_methods, write_report
from .tests_unit import run_unit_tests
from .vitb import run_vitb_pass


def _hash_if(p: Path) -> dict[str, Any]:
    return {
        "path": str(p),
        "exists": p.exists(),
        "sha16": sha256_file16(p) if p.exists() else None,
        "bytes": int(p.stat().st_size) if p.exists() else 0,
    }


def build_reuse_manifest(shared: dict, sids: list[int]) -> dict[str, Any]:
    files = {
        "cmcla_manifest": _hash_if(shared["cmcla"] / "common_anchor_manifest.json"),
        "qlca_risks": _hash_if(shared["qlca"] / "anchor_risks.csv"),
        "qlca_primary": _hash_if(shared["qlca"] / "primary_inference.json"),
        "qlca_align": _hash_if(shared["qlca"] / "alignment_summary.json"),
        "ndc_example": _hash_if(shared["ndc"] / "H_vectors" / f"{sids[0]}.npz"),
    }
    for m in shared["models"]:
        files[f"fcr_{m}"] = _hash_if(shared["fcr"] / "tables" / f"{m}_per_anchor_curvature.parquet")
        files[f"cmcla_{m}"] = _hash_if(shared["cmcla"] / "probes" / f"{m}_anchor_metrics.parquet")
    return {
        "n_anchors": len(sids),
        "models": list(shared["models"]),
        "d": PRIMARY_D,
        "coordinate_convention": "orthonormal_pca_chart_euclidean_frobenius",
        "align_by": "sample_id",
        "files": files,
    }


def _hessian_flags(vitb: pd.DataFrame) -> dict[str, Any]:
    if vitb is None or vitb.empty or "frac_Gamma_TF" not in vitb.columns:
        return {"primarily_traceless": False, "primarily_isotropic": False, "component_unstable": True}
    med_tf = float(np.nanmedian(vitb.frac_Gamma_TF))
    med_h = float(np.nanmedian(vitb.frac_Gamma_H))
    stab_h = float(np.nanmedian(vitb.gamma_H_fold_cosine))
    stab_tf = float(np.nanmedian(vitb.gamma_TF_fold_cosine))
    unstable = bool((med_h >= 0.5 and stab_h < STAB_MIN) or (med_tf >= 0.5 and stab_tf < STAB_MIN))
    return {
        "primarily_traceless": bool(med_tf > 0.5 and stab_tf >= STAB_MIN),
        "primarily_isotropic": bool(med_h > 0.5 and stab_h >= STAB_MIN),
        "component_unstable": unstable,
        "median_frac_H": med_h,
        "median_frac_TF": med_tf,
        "median_stab_H": stab_h,
        "median_stab_TF": stab_tf,
    }


def _align_flags(vitb: pd.DataFrame) -> dict[str, Any]:
    if vitb is None or vitb.empty or "A_H" not in vitb.columns:
        return {"driven_by_mean": False, "driven_by_traceless": False, "driven_by_interaction": False, "driver": "unavailable"}
    aH = float(np.nanmedian(vitb.A_H))
    aTF = float(np.nanmedian(vitb.A_TF))
    aB = float(np.nanmedian(vitb.A_B))
    nH = float(np.nanmedian(vitb.A_H_null_median))
    nTF = float(np.nanmedian(vitb.A_TF_null_median))
    pH = float(np.nanmedian(vitb.A_H_null_p95))
    pTF = float(np.nanmedian(vitb.A_TF_null_p95))
    mean = bool(aH > pH and aTF <= pTF)
    tf = bool(aTF > pTF and aH <= pH)
    cross = float(np.nanmedian(np.abs(vitb.induced_cross))) if "induced_cross" in vitb.columns else 0.0
    eB = float(np.nanmedian(np.abs(vitb.induced_EB))) if "induced_EB" in vitb.columns else 1.0
    inter = bool(cross > 0.25 * max(eB, 1e-12)) or bool(aH > pH and aTF > pTF)
    if mean and not tf:
        driver = "mean"
    elif tf and not mean:
        driver = "traceless"
    elif inter:
        driver = "interaction"
    else:
        driver = "unresolved"
    return {
        "driven_by_mean": mean,
        "driven_by_traceless": tf,
        "driven_by_interaction": inter,
        "driver": driver,
        "median_A_B": aB,
        "median_A_H": aH,
        "median_A_TF": aTF,
        "median_null_H": nH,
        "median_null_TF": nTF,
    }


def _probe_flags(vitb: pd.DataFrame, qlca: pd.DataFrame | None) -> dict[str, Any]:
    if vitb is None or vitb.empty or "delta_UQ2" not in vitb.columns:
        return {"iq_explains_uq2": False, "tq_explains_uq2": False, "bstf_explains_bs": False}
    du = vitb.delta_UQ2.to_numpy(float)
    di = vitb.delta_IQ.to_numpy(float)
    dt = vitb.delta_TQ.to_numpy(float)
    m = np.isfinite(du) & (np.abs(du) > 1e-12)
    frac_iq = float(np.nanmedian(np.clip(di[m] / du[m], -2, 2))) if m.any() else float("nan")
    frac_tq = float(np.nanmedian(np.clip(dt[m] / du[m], -2, 2))) if m.any() else float("nan")
    bstf = False
    if qlca is not None and "delta_BS" in qlca.columns:
        mm = vitb.merge(qlca[["sample_id", "delta_BS"]], on="sample_id", how="inner")
        if "delta_BSTF" in mm.columns and len(mm):
            a, b = mm.delta_BSTF.to_numpy(float), mm.delta_BS.to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
            frac_bs = float(np.nanmedian(np.clip(a[ok] / b[ok], -2, 2))) if ok.any() else float("nan")
            bstf = bool(np.isfinite(frac_bs) and frac_bs > 0.7)
        else:
            frac_bs = float("nan")
    else:
        frac_bs = float("nan")
    return {
        "iq_explains_uq2": bool(np.isfinite(frac_iq) and frac_iq > 0.7 and frac_iq > (frac_tq if np.isfinite(frac_tq) else -np.inf)),
        "tq_explains_uq2": bool(np.isfinite(frac_tq) and frac_tq > 0.7 and frac_tq > (frac_iq if np.isfinite(frac_iq) else -np.inf)),
        "bstf_explains_bs": bstf,
        "median_frac_IQ_of_UQ2": frac_iq,
        "median_frac_TQ_of_UQ2": frac_tq,
        "median_frac_BSTF_of_BS": frac_bs,
        "median_delta_IQ": float(np.nanmedian(di)),
        "median_delta_TQ": float(np.nanmedian(dt)),
        "median_delta_UQ2": float(np.nanmedian(du)),
        "win_IQ": float(np.nanmean(di > 0)),
        "win_TQ": float(np.nanmean(dt > 0)),
        "win_UQ2": float(np.nanmean(du > 0)),
    }


def run(cfg: ExpConfig) -> dict[str, Any]:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("tables", "figures", "inference"):
        (out / sub).mkdir(exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=True)

    shared = load_shared(cfg)
    sids = list(shared["sids"])
    models = list(shared["models"])

    man = build_reuse_manifest(shared, sids)
    write_json(out / "reuse_manifest.json", man, force=True)

    tests = run_unit_tests()
    write_json(out / "unit_synthetic_test_results.json", tests, force=True)
    write_df(out / "tables" / "synthetic_validation.csv", pd.DataFrame(tests["rows"]), force=True)
    if not tests["ok"] and not cfg.smoke:
        write_text(out / "BLOCKER.md", "# BLOCKER\n\nUnit/synthetic tests failed.\n", force=True)
        write_json(out / "decision.json", {"label": "curvature_component_audit_blocked", "reason": "tests"}, force=True)
        return {"ok": False, "blocker": "tests", "tests": tests}

    print("[ccpd] Phase 0 parity", flush=True)
    parity = run_parity(shared, cfg)
    write_json(out / "parity.json", parity, force=True)
    write_artifact_audit(out, man, parity)
    if not parity.get("ok"):
        write_text(out / "BLOCKER.md", "# BLOCKER\n\nPhase 0 parity failed. No scientific inference.\n", force=True)
        write_json(out / "decision.json", {"label": "curvature_component_audit_blocked", "reason": "parity"}, force=True)
        return {"ok": False, "blocker": "parity", "parity": parity}

    if cfg.stage in ("audit", "tests", "parity"):
        return {"ok": True, "stage": cfg.stage, "parity": parity, "tests": tests}

    tables: dict[str, pd.DataFrame] = {}
    org_rows = []
    for m in models:
        df = enrich_organization(merge_components(shared, m, sids))
        write_df(out / "tables" / f"{m}_per_anchor_components.parquet", df, force=True)
        tables[m] = df
        org_rows.append(organization_report(df, m))
        write_json(out / "inference" / f"{m}_point_associations.json", point_all_outcomes(df), force=True)
    write_df(out / "tables" / "organization_table.csv", pd.DataFrame(org_rows), force=True)

    print("[ccpd] Phase 3 unique associations", flush=True)
    per_model_mseg = {}
    per_model_r2 = {}
    per_model_adapt = {}
    for m, df in tables.items():
        print(f"[ccpd] inference {m} mse_G", flush=True)
        per_model_mseg[m] = model_component_inference(
            df, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed, ycol="mse_G"
        )
        write_json(out / "inference" / f"{m}_unique_mse_G.json", per_model_mseg[m], force=True)
        per_model_r2[m] = model_component_inference(
            df, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 3, ycol="r2_G"
        )
        write_json(out / "inference" / f"{m}_unique_r2_G.json", per_model_r2[m], force=True)
        per_model_adapt[m] = model_component_inference(
            df, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 5, ycol="delta_adapt"
        )
        write_json(out / "inference" / f"{m}_unique_delta_adapt.json", per_model_adapt[m], force=True)

    print("[ccpd] joint-anchor bootstrap", flush=True)
    joint_mseg = joint_unique(tables, "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 7)
    joint_r2 = joint_unique(tables, "r2_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 9)
    joint_adapt = joint_unique(tables, "delta_adapt", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 11)
    write_json(out / "inference" / "joint_unique_mse_G.json", joint_mseg, force=True)
    write_json(out / "inference" / "joint_unique_r2_G.json", joint_r2, force=True)
    write_json(out / "inference" / "joint_unique_delta_adapt.json", joint_adapt, force=True)

    assoc_rows = []
    for m in models:
        rec = {"model": m}
        for name, store in (("mse_G", per_model_mseg), ("r2_G", per_model_r2), ("delta_adapt", per_model_adapt)):
            for key in ("KH", "KTF", "Kdir", "unique_KH", "unique_KTF"):
                rec[f"{name}_{key}"] = store[m][key]["observed"]
        assoc_rows.append(rec)
    write_df(out / "tables" / "cross_model_component_associations.csv", pd.DataFrame(assoc_rows), force=True)
    write_df(
        out / "tables" / "joint_anchor_bootstrap.csv",
        pd.DataFrame(
            [
                {"y": "mse_G", **{k: joint_mseg[k] for k in ("unique_KH_bar", "unique_KTF_bar") if k in joint_mseg}},
                {"y": "r2_G", **{k: joint_r2[k] for k in ("unique_KH_bar", "unique_KTF_bar") if k in joint_r2}},
                {"y": "delta_adapt", **{k: joint_adapt[k] for k in ("unique_KH_bar", "unique_KTF_bar") if k in joint_adapt}},
            ]
        ),
        force=True,
    )

    qlca = load_qlca_risks(shared, sids)
    if "vit_base" in tables:
        qlca = qlca.merge(
            tables["vit_base"][["sample_id", "K_TF_cross", "K_dir_cross"]],
            on="sample_id",
            how="inner",
        )
        write_df(out / "tables" / "vitb_qlca_merged.csv", qlca, force=True)
        print("[ccpd] Phase 4 ViT-B QLCA unique associations", flush=True)
        vitb_outcomes = {}
        for ycol in ("delta_Q", "delta_BS", "delta_FQ", "A_B", "gamma_fold_cosine"):
            if ycol not in qlca.columns:
                continue
            vitb_outcomes[ycol] = model_component_inference(
                qlca, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 19, ycol=ycol
            )
        if "mse_UQ" in qlca.columns and "mse_BS" in qlca.columns:
            qlca = qlca.copy()
            qlca["delta_UQ_minus_BS"] = qlca.mse_UQ - qlca.mse_BS
            vitb_outcomes["delta_UQ_minus_BS"] = model_component_inference(
                qlca, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 23, ycol="delta_UQ_minus_BS"
            )
        write_json(out / "inference" / "vitb_qlca_component_associations.json", vitb_outcomes, force=True)
    else:
        vitb_outcomes = {}

    if cfg.stage == "cross":
        return {"ok": True, "stage": "cross", "joint": joint_mseg}

    print("[ccpd] Phases 5–8 ViT-B Hessian / probes", flush=True)
    span = verify_quadratic_span(PRIMARY_D)
    write_json(out / "quadratic_span_check.json", span, force=True)
    vitb = run_vitb_pass(shared, cfg, out, sids)
    if "vit_base" in tables:
        vitb = vitb.merge(
            tables["vit_base"][["sample_id", "K_H_cross", "K_TF_cross", "K_dir_cross", *CONTROLS]],
            on="sample_id",
            how="left",
        )
        write_df(out / "tables" / "vitb_component_pass.parquet", vitb, force=True)
        if "delta_UQ2" in vitb.columns:
            vitb_probe_inf = {}
            for ycol in ("delta_IQ", "delta_TQ", "delta_UQ2", "A_H", "A_TF", "frac_Gamma_H", "frac_Gamma_TF"):
                if ycol in vitb.columns:
                    vitb_probe_inf[ycol] = model_component_inference(
                        vitb, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed + 29, ycol=ycol
                    )
            write_json(out / "inference" / "vitb_probe_component_associations.json", vitb_probe_inf, force=True)

    hess = _hessian_flags(vitb)
    align = _align_flags(vitb)
    probes = _probe_flags(vitb, qlca if "sample_id" in qlca.columns else None)
    write_json(out / "tables" / "label_hessian_decomposition.json", hess, force=True)
    write_json(out / "tables" / "component_alignment_summary.json", align, force=True)
    write_json(out / "tables" / "probe_gain_summary.json", probes, force=True)

    # matched-anchor geometry null for median A_H / A_TF
    if len(vitb) >= 8 and "A_H" in vitb.columns:
        rng = np.random.default_rng(cfg.seed + 41)
        obs_h, obs_tf = float(vitb.A_H.median()), float(vitb.A_TF.median())
        nulls_h, nulls_tf = [], []
        # use stored null medians as random-γ; geometry shuffle via permuting A_*_A against gamma identity
        for _ in range(min(cfg.n_boot_eff(), 400)):
            perm = rng.permutation(len(vitb))
            nulls_h.append(float(np.nanmedian(vitb.A_H.to_numpy()[perm])))
            nulls_tf.append(float(np.nanmedian(vitb.A_TF.to_numpy()[perm])))
        write_json(
            out / "tables" / "alignment_null_bootstrap.json",
            {
                "A_H_median": obs_h,
                "A_TF_median": obs_tf,
                "bootstrap_A_H_ci": list(np.nanpercentile(
                    [float(vitb.A_H.iloc[rng.choice(len(vitb), len(vitb), replace=True)].median()) for _ in range(cfg.n_boot_eff())],
                    [2.5, 97.5],
                )),
                "bootstrap_A_TF_ci": list(np.nanpercentile(
                    [float(vitb.A_TF.iloc[rng.choice(len(vitb), len(vitb), replace=True)].median()) for _ in range(cfg.n_boot_eff())],
                    [2.5, 97.5],
                )),
            },
            force=True,
        )

    kdir_mseg_sig = any(per_model_mseg[m]["Kdir"].get("ci_excludes_zero") for m in per_model_mseg)
    org_flag = {"kdir_mseg_sig": kdir_mseg_sig}

    decision = decide(
        parity_ok=bool(parity.get("ok")),
        tests_ok=bool(tests.get("ok")),
        joint_mseg=joint_mseg,
        per_model_mseg=per_model_mseg,
        vitb_dq=vitb_outcomes.get("delta_Q", {}),
        hessian=hess,
        alignment=align,
        probes=probes,
        org=org_flag,
    )
    write_json(out / "decision.json", decision, force=True)

    write_figures(out, per_model_mseg, joint_mseg, vitb, qlca)
    span = verify_quadratic_span(PRIMARY_D)
    write_methods(out, asdict(cfg), span)
    vitb_sum = summarize_vitb(vitb, qlca, hess, align, probes)
    runtime = time.time() - t0
    write_report(
        out,
        decision=decision,
        parity=parity,
        tests=tests,
        org_rows=org_rows,
        joint=joint_mseg,
        per_model=per_model_mseg,
        vitb_sum=vitb_sum,
        runtime_s=runtime,
    )
    write_manuscript_recommendation(out, decision, joint_mseg, vitb_sum)

    summary = {
        "ok": True,
        "decision": decision.get("label"),
        "reason": decision.get("reason"),
        "runtime_s": runtime,
        "n_anchors": len(sids),
        "models": models,
        "tests": {"n": tests["n"], "n_pass": tests["n_pass"], "ok": tests["ok"]},
        "parity_ok": parity.get("ok"),
        "unique_KH_mse_G_bar": joint_mseg.get("unique_KH_bar"),
        "unique_KTF_mse_G_bar": joint_mseg.get("unique_KTF_bar"),
        "vitb_delta_Q": vitb_outcomes.get("delta_Q"),
        "hessian": hess,
        "alignment": align,
        "probes": probes,
        "peak_rss_mb": peak_rss_mb(),
    }
    write_json(out / "summary.json", summary, force=True)
    if (not cfg.smoke) and decision.get("label") != "curvature_component_audit_blocked":
        write_json(
            out / "COMPLETE.json",
            {
                "ok": True,
                "decision": decision.get("label"),
                "runtime_s": runtime,
                "output_dir": str(out),
            },
            force=True,
        )
    print(f"[ccpd] done label={decision.get('label')} runtime={runtime:.1f}s", flush=True)
    return summary
