"""Bounded ViT-B task-aligned curvature. No manuscript edits, no new autoencoders."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_EXP = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[3]
_NB = _REPO / "notebooks"
for p in (_EXP, _NB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from geometry.physics_cross_model_pointwise_residual_curvature.decoder import compatible_vitb_legacy, load_model
from geometry.physics_curvature_probe_rank_sweep.inference import associate

from .algebra import holm, radial_b, trace_g
from .audit import decoder_ckpts, historical_local_r2, load_historical_weights, load_shared, reuse_manifest, run_parity
from .config import (
    CONTROLS,
    COVERAGE_FAIL_FRAC,
    DECODER_SEEDS,
    D_LAT,
    K,
    MIN_EVAL,
    MODEL,
    MULTISCALE_K,
    PROJECTED_CAP_S,
    PROTOCOL,
    TARGETS,
    ExpConfig,
)
from .decision import decide
from .fixtures import run_fixtures
from .figures import write_figures
from .geometry_d import eval_decoder_field, task_aligned_at_z
from .geometry_q import eval_q_all_weights
from .inference import controlled, dep_bootstrap_mse, per_target_assoc, seed_reliability_E, synchronized_aggregate
from .io_util import assert_not_preserved, peak_rss_mb, platonic_root, resolve_path, write_df, write_json
from .probes import confirmatory_for_target, historical_for_target
from .reports import write_markdowns
from .tests_unit import run_unit_tests


def _remaining(t0: float, wall: float) -> float:
    return wall - 180.0 - (time.time() - t0)


def run(cfg: ExpConfig) -> dict[str, Any]:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)

    shared = load_shared(cfg)
    ckpts = decoder_ckpts(shared)
    man = reuse_manifest(shared, ckpts)
    write_json(out / "reuse_manifest.json", man, force=True)
    write_json(
        out / "target_definitions.json",
        {
            "targets": list(TARGETS),
            "excluded": man["targets_excluded"],
            "align_by": "sample_id",
            "ridge_alpha": 100.0,
            "train_frac": 0.6,
            "min_eval": MIN_EVAL,
        },
        force=True,
    )
    if not man["four_target_claim_ok"]:
        return _block(out, t0, "target_join_blocker", man)

    parity = run_parity(shared, cfg, out)
    print("[tac] parity", parity.get("ok"), flush=True)
    if not ckpts["three_seed_ok"]:
        write_json(out / "BLOCKER.json", {"reason": "three_seed_checkpoints_missing", "ckpts": ckpts}, force=True)
        return _block(out, t0, "three_seed_checkpoints_missing", {"parity": parity, "ckpts": ckpts})

    tests0 = run_unit_tests(parity=parity)
    write_json(out / "unit_test_results.json", tests0, force=True)
    print("[tac] tests", tests0["n_passed"], "/", tests0["n_tests"], flush=True)

    fix = run_fixtures()
    write_df(out / "fixture_results.csv", pd.DataFrame([{k: json.dumps(v) if isinstance(v, dict) else v for k, v in fix.items()}]), force=True)
    fixtures_ok = bool(fix["ambient_linear"].get("pass") and fix["shuffled"].get("pass"))

    hist_w = load_historical_weights(shared)
    hist_r2 = historical_local_r2(shared)
    conf = {t: confirmatory_for_target(shared, t) for t in TARGETS}
    hist = {t: historical_for_target(shared, t, hist_w, hist_r2) for t in TARGETS}
    cov = {t: conf[t]["coverage"] for t in TARGETS}
    leak = {
        "ok": all(c["coverage"]["ok"] for c in conf.values()) and all(not (c["split"]["train"] & c["split"]["eval"]).any() for c in conf.values()),
        "per_target": cov,
        "train_labels_excluded_from_eval": True,
        "eval_labels_excluded_from_w": True,
        "min_eval": MIN_EVAL,
        "coverage_fail_frac": COVERAGE_FAIL_FRAC,
    }
    write_json(out / "leakage_audit.json", leak, force=True)
    if not leak["ok"]:
        return _block(out, t0, "coverage_or_leakage", {"leakage": leak, "parity": parity})

    tests = run_unit_tests(parity=parity, leakage=leak)
    write_json(out / "unit_test_results.json", tests, force=True)

    # load three-seed decoders
    models = {}
    for seed in DECODER_SEEDS:
        p = Path(ckpts["three_seed"][int(seed)]["path"])
        models[int(seed)] = load_model(p, in_dim=shared["D"], seed=int(seed))

    sids = shared["sids"]
    rows_X = np.array([shared["sid_to_row"][int(s)] for s in sids])
    X_anc = shared["X"][rows_X]

    # 32-anchor smoke on seed 0, mag_r confirmatory w
    n_smoke = min(32, len(sids))
    print(f"[tac] smoke n={n_smoke}", flush=True)
    t_s0 = time.time()
    _ = eval_decoder_field(models[0], X_anc[:n_smoke], conf["mag_r_desi"]["w"], normalized=True, sphere_residual=True)
    t_smoke = time.time() - t_s0
    n_jobs = len(DECODER_SEEDS) * len(TARGETS) * len(sids)
    projected = t_smoke * (n_jobs / max(n_smoke, 1))
    write_json(out / "smoke_runtime.json", {"smoke_s": t_smoke, "projected_D_s": projected}, force=True)
    print(f"[tac] smoke {t_smoke:.1f}s projected_D={projected:.0f}s", flush=True)
    resource_capped = False
    if (time.time() - t0) + projected > PROJECTED_CAP_S:
        return _block(out, t0, "projected_over_cap", {"projected": projected})

    # D-residual confirmatory per seed/target
    per_seed_E = {t: {} for t in TARGETS}
    per_seed_b = {t: {} for t in TARGETS}
    radial_rows = []
    seed_rows = []
    for seed, model in models.items():
        if _remaining(t0, cfg.wall_s) < 90:
            resource_capped = True
            break
        for t in TARGETS:
            print(f"[tac] D-residual seed={seed} target={t}", flush=True)
            fld = eval_decoder_field(model, X_anc, conf[t]["w"], normalized=True, sphere_residual=True)
            per_seed_E[t][seed] = fld["E"]
            per_seed_b[t][seed] = fld["b"]
            # radial diagnostic from first seed only
            if seed == 0:
                for i, sid in enumerate(sids):
                    radial_rows.append(
                        {
                            "sample_id": int(sid),
                            "target": t,
                            "T_radial": float(-D_LAT * np.dot(conf[t]["w"], X_anc[i] / max(np.linalg.norm(X_anc[i]), 1e-15))),
                            "pred": float(conf[t]["yhat"][rows_X[i]]),
                            "E_D_seed0": float(fld["E"][i]),
                            "radial_identity_ok": bool(fld["radial_identity_ok"][i]),
                            "kind": "sphere_radial_diagnostic",
                        }
                    )
            for i, sid in enumerate(sids):
                seed_rows.append(
                    {
                        "sample_id": int(sid),
                        "target": t,
                        "seed": int(seed),
                        "E_D_S": float(fld["E"][i]),
                        "T_D_S": float(fld["T"][i]),
                        "abs_T_D_S": float(fld["abs_T"][i]),
                        "finite": bool(fld["finite"][i]),
                        "kind": "D_residual_sphere_normal",
                        "analysis": "C_confirmatory",
                    }
                )
    write_df(out / "per_seed_task_aligned.parquet", pd.DataFrame(seed_rows), force=True)
    write_df(out / "radial_diagnostics.csv", pd.DataFrame(radial_rows), force=True)

    # D-raw sensitivity, seed 0 only
    raw_rows = []
    if not resource_capped and _remaining(t0, cfg.wall_s) > 120:
        for t in TARGETS:
            fld = eval_decoder_field(models[0], X_anc, conf[t]["w"], normalized=False, sphere_residual=False)
            for i, sid in enumerate(sids):
                raw_rows.append({"sample_id": int(sid), "target": t, "E_D_E": float(fld["E"][i]), "kind": "D_raw_euclidean_normal"})
    write_df(out / "d_raw_sensitivity.csv", pd.DataFrame(raw_rows) if raw_rows else pd.DataFrame([{"kind": "skipped"}]), force=True)

    rel_rows = []
    frames_D = {}
    frames_Q_ready = {}
    anchor_rows = []
    for t in TARGETS:
        rel = seed_reliability_E(per_seed_E[t], per_seed_b[t])
        rel_rows.append({"target": t, "passed": rel["passed"], "median_rho_E": rel["median_rho_E"], "best_seed_selected": False})
        Econs = rel["consensus_rank"] if rel["consensus_rank"] is not None else np.median(np.column_stack(list(per_seed_E[t].values())), axis=1)
        df = pd.DataFrame(
            {
                "sample_id": sids,
                "E_D_S": Econs,
                "mse_G": conf[t]["mse_G"],
                "r2_G": conf[t]["r2_G"],
                "log_knn_radius": _log_radius(shared, t),
                "local_eval_label_variance": conf[t]["local_eval_label_variance"],
                "local_evaluation_count": conf[t]["n_eval"],
                "target": t,
            }
        )
        frames_D[t] = df
        for i, sid in enumerate(sids):
            anchor_rows.append({**{k: df[k].iloc[i] for k in df.columns}, "analysis": "C_confirmatory", "instrument": "D_residual"})

    write_df(out / "seed_reliability.csv", pd.DataFrame(rel_rows), force=True)

    # Q reconstruct once for all confirmatory weights
    print("[tac] Q reconstruction", flush=True)
    q_w = {t: conf[t]["w"] for t in TARGETS}
    n_splits = 1 if cfg.smoke else cfg.n_q_splits
    if _remaining(t0, cfg.wall_s) < 180:
        resource_capped = True
        qfld = {"per_target": {t: {"E_Q_cross": np.full(len(sids), np.nan)} for t in TARGETS}, "K_H_cross_recon": np.full(len(sids), np.nan)}
    else:
        qfld = eval_q_all_weights(shared, q_w, n_splits=n_splits)
    # Q parity vs frozen KHcross
    fcr = shared["fcr"].copy()
    fcr["sample_id"] = fcr.sample_id.astype(int)
    kh_join = pd.DataFrame({"sample_id": sids, "K_H_recon": qfld["K_H_cross_recon"]}).merge(fcr[["sample_id", "K_H_cross"]], on="sample_id", how="left")
    kh_parity = float(np.nanmedian(np.abs(kh_join.K_H_recon - kh_join.K_H_cross))) if kh_join.K_H_cross.notna().any() else float("nan")

    frames_Q = {}
    agree_rows = []
    per_rows = []
    hist_vs = []
    for t in TARGETS:
        df = frames_D[t].copy()
        df["E_Q_cross"] = qfld["per_target"][t]["E_Q_cross"]
        frames_Q[t] = df
        aD = per_target_assoc(df, "E_D_S", "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
        aQ = per_target_assoc(df, "E_Q_cross", "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
        aR = per_target_assoc(df, "E_D_S", "r2_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
        agr = associate(df.E_D_S.to_numpy(float), df.E_Q_cross.to_numpy(float), None)
        agr_c = controlled(df, "E_D_S", "E_Q_cross")
        agree_rows.append({"target": t, "raw": agr["raw"], "rho_ctl": agr_c["controlled"]})
        per_rows.append(
            {
                "target": t,
                "rho_D": aD["observed"],
                "rho_Q": aQ["observed"],
                "rho_D_R2": aR["observed"],
                "p_D": aD["p_mc"],
                "p_Q": aQ["p_mc"],
                "sign_D": float(np.sign(aD["observed"])),
                "sign_Q": float(np.sign(aQ["observed"])),
                "analysis": "C_confirmatory",
                "primary_outcome": "mse_G",
            }
        )
        # historical descriptive
        hdf = df.copy()
        hdf["mse_H"] = hist[t]["mse_G"]
        hdf["E_H"] = df["E_D_S"]
        aH = controlled(hdf.dropna(subset=["mse_H", "E_H"]), "E_H", "mse_H") if hdf["mse_H"].notna().any() else {"controlled": float("nan")}
        hist_vs.append({"target": t, "rho_H": aH.get("controlled", float("nan")), "rho_C": aD["observed"], "rho_radial": float("nan")})
        write_json(out / f"assoc_{t}.json", {"D": aD, "Q": aQ, "R2": aR, "agree": agr_c}, force=True)

    write_df(out / "per_target_results.csv", pd.DataFrame(per_rows), force=True)
    write_df(out / "cross_instrument_agreement.csv", pd.DataFrame(agree_rows), force=True)

    print("[tac] P1/P2", flush=True)
    p1 = synchronized_aggregate(frames_D, "E_D_S", "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
    p2 = synchronized_aggregate(frames_Q, "E_Q_cross", "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
    hs = holm(np.array([p1["p_mc"], p2["p_mc"]]))
    p1["p_holm"] = float(hs[0])
    p2["p_holm"] = float(hs[1])
    p1["pass_holm"] = bool(p1["observed"] > 0 and hs[0] <= 0.05)
    p2["pass_holm"] = bool(p2["observed"] > 0 and hs[1] <= 0.05)
    p1["name"] = "P1_bar_rho_D"
    p2["name"] = "P2_bar_rho_Q"
    write_df(
        out / "primary_results.csv",
        pd.DataFrame(
            [
                {k: v for k, v in p1.items() if k != "per_target"} | {"per_target": json.dumps(p1["per_target"])},
                {k: v for k, v in p2.items() if k != "per_target"} | {"per_target": json.dumps(p2["per_target"])},
            ]
        ),
        force=True,
    )
    write_json(out / "permutation_bootstrap_summary.json", {"P1": p1, "P2": p2, "K_H_recon_median_abs_diff": kh_parity}, force=True)

    # dependence-aware sensitivity (may skip if tight)
    dep_rows = []
    if (not cfg.skip_patch) and _remaining(t0, cfg.wall_s) > 90:
        for t in TARGETS:
            ctrl = np.column_stack(
                [
                    frames_D[t].log_knn_radius.to_numpy(float),
                    frames_D[t].local_eval_label_variance.to_numpy(float),
                    frames_D[t].local_evaluation_count.to_numpy(float),
                ]
            )
            boots = dep_bootstrap_mse(
                shared["Y"][t],
                conf[t]["yhat"],
                shared["neigh"],
                conf[t]["split"]["eval"],
                frames_D[t].E_D_S.to_numpy(float),
                ctrl,
                n_boot=min(cfg.n_boot_eff(), 400),
                min_eval=MIN_EVAL,
            )
            dep_rows.append({"target": t, "dep_boot_median": float(np.nanmedian(boots)), "n_boot": int(np.isfinite(boots).sum())})
    write_df(out / "dependence_aware_bootstrap.csv", pd.DataFrame(dep_rows) if dep_rows else pd.DataFrame([{"status": "skipped"}]), force=True)

    write_df(out / "per_anchor_task_aligned.parquet", pd.DataFrame(anchor_rows), force=True)
    write_df(out / "historical_vs_confirmatory.csv", pd.DataFrame(hist_vs), force=True)

    # radial coupling
    rad_df = pd.DataFrame(radial_rows)
    radial_dom = False
    if len(rad_df):
        rabs = []
        for t in TARGETS:
            sub = rad_df[rad_df.target == t]
            m = frames_D[t].merge(sub, on="sample_id")
            if len(m) > 12:
                rabs.append(abs(associate(np.abs(m.T_radial.to_numpy(float)), m.mse_G.to_numpy(float), None)["raw"]))
        if rabs and p1["pass_holm"]:
            mean_e = float(np.mean([abs(p1["per_target"][t]) for t in TARGETS]))
            radial_dom = float(np.mean(rabs)) > mean_e + 0.05

    hist_strong = bool(np.nanmean([h["rho_H"] for h in hist_vs]) > 0.15)
    conf_weak = bool(p1["observed"] <= 0 and p2["observed"] <= 0)

    decision = decide(
        tests_ok=tests["all_passed"],
        parity_ok=bool(parity.get("q_ok")),
        coverage_ok=bool(leak["ok"]),
        fixtures_ok=fixtures_ok,
        p1=p1,
        p2=p2,
        signs_D=[float(r["rho_D"]) for r in per_rows],
        signs_Q=[float(r["rho_Q"]) for r in per_rows],
        agree=[float(r["rho_ctl"]) for r in agree_rows],
        historical_strong=hist_strong,
        confirmatory_weak=conf_weak,
        radial_dominates=radial_dom,
        resource_capped=resource_capped,
    )
    write_json(out / "decision.json", decision, force=True)

    try:
        write_figures(
            out,
            per_target=pd.DataFrame(per_rows),
            agree=pd.DataFrame(agree_rows),
            hist_vs_c=pd.DataFrame(hist_vs),
        )
    except Exception as exc:
        write_json(out / "figures_error.json", {"error": str(exc)}, force=True)

    runtime = {
        "wall_s": time.time() - t0,
        "rss_mb": peak_rss_mb(),
        "smoke_s": t_smoke,
        "projected_D_s": projected,
        "resource_capped": resource_capped,
        "n_q_splits": n_splits,
        "K_H_recon_median_abs_diff": kh_parity,
        "env": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
    }
    write_json(out / "runtime.json", runtime, force=True)
    write_json(
        out / "summary.json",
        {
            "label": decision["label"],
            "P1": p1,
            "P2": p2,
            "per_target": per_rows,
            "agree": agree_rows,
            "parity_ok": parity.get("ok"),
            "runtime_s": runtime["wall_s"],
            "output_dir": str(out),
        },
        force=True,
    )
    extra = (
        f"P1={p1['observed']:.3f} pHolm={p1['p_holm']:.4g}; "
        f"P2={p2['observed']:.3f} pHolm={p2['p_holm']:.4g}; "
        f"KH recon median |diff|={kh_parity:.4g}."
    )
    write_markdowns(out, decision=decision, runtime=runtime, tests=tests, parity=parity, extra=extra)
    write_json(
        out / "COMPLETE.json",
        {
            "status": "complete" if not resource_capped else "partial",
            "label": decision["label"],
            "n_anchors": len(sids),
            "targets": list(TARGETS),
            "wall_s": runtime["wall_s"],
            "experiments_run": True,
            "manuscript_modified": False,
        },
        force=True,
    )
    disk = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    write_json(out / "disk_usage.json", {"bytes": disk, "total": int(sum(disk.values()))}, force=True)
    return {"out": str(out), "decision": decision, "runtime": runtime}


def _log_radius(shared: dict, target: str) -> np.ndarray:
    # frozen k=2048 radius from FCR/CMCLA by sample_id
    src = shared["cmcla"].copy()
    src["sample_id"] = src.sample_id.astype(int)
    mp = {int(s): float(r) for s, r in zip(src.sample_id, src.log_knn_radius)}
    return np.array([mp.get(int(s), np.nan) for s in shared["sids"]], dtype=float)


def _block(out, t0, reason, extra):
    from .decision import decide

    d = decide(
        tests_ok=False,
        parity_ok=False,
        coverage_ok=False,
        fixtures_ok=False,
        p1=None,
        p2=None,
        signs_D=[],
        signs_Q=[],
        agree=[],
        historical_strong=False,
        confirmatory_weak=False,
        radial_dominates=False,
        resource_capped=True,
    )
    write_json(out / "decision.json", d, force=True)
    write_json(out / "COMPLETE.json", {"status": "blocked", "reason": reason, "label": d["label"]}, force=True)
    write_markdowns(out, decision=d, runtime={"wall_s": time.time() - t0}, tests={"n_passed": 0, "n_tests": 0}, parity={}, extra=reason)
    return {"blocked": True, "reason": reason, "extra": extra}
