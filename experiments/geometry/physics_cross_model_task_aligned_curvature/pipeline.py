"""Bounded cross-model task-aligned curvature. Q first; no new autoencoders."""

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

from geometry.physics_cross_model_pointwise_residual_curvature.decoder import load_model
from geometry.physics_curvature_probe_rank_sweep.inference import associate
from geometry.physics_task_aligned_curvature.algebra import holm
from geometry.physics_task_aligned_curvature.geometry_d import eval_decoder_field
from geometry.physics_task_aligned_curvature.inference import per_target_assoc, seed_reliability_E, synchronized_aggregate
from geometry.physics_task_aligned_curvature.probes import confirmatory_for_target

from .audit import (
    as_shared,
    decoder_ckpts,
    load_model_bundle,
    load_shared,
    log_radius,
    reuse_manifest,
    reuse_vitb_D,
    run_parity,
)
from .config import (
    DECODER_SEEDS,
    MIN_EVAL,
    PROJECTED_CAP_S,
    PROTOCOL,
    REFERENCE,
    REPLICATION,
    RESERVE_WRITE_S,
    SEED_RHO_GATE,
    TARGETS,
    VITB_P2,
    ExpConfig,
)
from .decision import decide
from .figures import write_figures
from .inference import synchronized_cross_model
from .io_util import assert_not_preserved, peak_rss_mb, platonic_root, resolve_path, write_df, write_json
from .qeval import eval_q_model
from .reports import write_markdowns
from .tests_unit import run_unit_tests


def _remaining(t0: float, wall: float) -> float:
    return wall - RESERVE_WRITE_S - (time.time() - t0)


def _env() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "rss_mb": peak_rss_mb(),
    }


def run(cfg: ExpConfig) -> dict[str, Any]:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)

    shared = load_shared(cfg)
    models = cfg.models()
    bundles = {m: load_model_bundle(shared, m) for m in models}
    ckpts = {m: decoder_ckpts(shared, m) for m in models}
    man = reuse_manifest(shared, bundles, ckpts)
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
            "split_salt": "task_aligned_curvature_v1",
            "models": list(models),
        },
        force=True,
    )
    if man["targets_excluded"]:
        return _block(out, t0, "target_join", {"manifest": man})

    missing_ck = [m for m, c in ckpts.items() if not c["three_seed_ok"]]
    write_json(out / "checkpoint_audit.json", {"ok": not missing_ck, "missing": missing_ck, "ckpts": ckpts}, force=True)

    parity = run_parity(shared, bundles, cfg, out)
    print("[cmtac] parity", parity.get("ok"), flush=True)

    leak = {"ok": True, "per_model": {}, "train_labels_excluded_from_eval": True, "eval_labels_excluded_from_w": True}
    frames_Q: dict[str, dict[str, pd.DataFrame]] = {}
    frames_D: dict[str, dict[str, pd.DataFrame]] = {}
    conf_store: dict[str, dict] = {}
    kh_parity = {}
    seed_rows = []
    per_rows = []
    agree_rows = []
    resource_capped = False
    q_ran = False

    for m in models:
        sh = as_shared(shared, bundles[m])
        conf = {t: confirmatory_for_target(sh, t) for t in TARGETS}
        conf_store[m] = conf
        cov_ok = all(conf[t]["coverage"]["ok"] for t in TARGETS)
        leak["per_model"][m] = {t: conf[t]["coverage"] for t in TARGETS}
        leak["ok"] = bool(leak["ok"] and cov_ok)
        if not cov_ok:
            print(f"[cmtac] {m} coverage fail", flush=True)
    write_json(out / "leakage_audit.json", leak, force=True)

    tests = run_unit_tests(parity=parity, leakage=leak, sids=shared["sids"])
    write_json(out / "unit_test_results.json", tests, force=True)
    print("[cmtac] tests", tests["n_passed"], "/", tests["n_tests"], flush=True)
    if not tests["all_passed"] or not parity.get("ok") or not leak["ok"]:
        return _finish(
            out,
            t0,
            cfg,
            decide(
                tests_ok=tests["all_passed"],
                parity_ok=bool(parity.get("ok")),
                coverage_ok=bool(leak["ok"]),
                r1=None,
                rd=None,
                vitb_q_bar=None,
                n_rep_q=0,
                resource_capped=False,
                q_ran=False,
            ),
            extra={"parity": parity, "tests": tests},
        )

    # Smoke Q on 8 anchors of the first replication model
    n_smoke = min(8, len(shared["sids"]))
    smoke_m = REPLICATION[0] if REPLICATION[0] in bundles else models[0]
    t_s0 = time.time()
    _ = eval_q_model(
        {**shared, "sids": shared["sids"][:n_smoke]},
        {**bundles[smoke_m], "neigh": bundles[smoke_m]["neigh"][:n_smoke]},
        {t: conf_store[smoke_m][t]["w"] for t in TARGETS},
        n_splits=1 if cfg.smoke else 1,
    )
    t_smoke = time.time() - t_s0
    n_q = len(models) * len(shared["sids"])
    # smoke used 1 split; production uses n_q_splits
    projected_q = t_smoke * (n_q / n_smoke) * (1.0 if cfg.smoke else float(cfg.n_q_splits))
    write_json(out / "smoke_runtime.json", {"smoke_s": t_smoke, "projected_Q_s": projected_q, "smoke_model": smoke_m}, force=True)
    print(f"[cmtac] smoke {t_smoke:.1f}s projected_Q={projected_q:.0f}s", flush=True)
    if (time.time() - t0) + projected_q > PROJECTED_CAP_S and not cfg.smoke:
        resource_capped = True
        print("[cmtac] Q projection exceeds cap", flush=True)

    n_splits = 1 if cfg.smoke else cfg.n_q_splits
    reused_D = reuse_vitb_D(shared) if REFERENCE in models and not cfg.smoke else None

    for m in models:
        if resource_capped:
            break
        if _remaining(t0, cfg.wall_s) < 90:
            resource_capped = True
            break
        print(f"[cmtac] Q {m}", flush=True)
        qfld = eval_q_model(shared, bundles[m], {t: conf_store[m][t]["w"] for t in TARGETS}, n_splits=n_splits)
        q_ran = True
        fcr = bundles[m]["fcr"].copy()
        fcr["sample_id"] = fcr.sample_id.astype(int)
        kh_join = pd.DataFrame({"sample_id": shared["sids"], "K_H_recon": qfld["K_H_cross_recon"]}).merge(
            fcr[["sample_id", "K_H_cross"]], on="sample_id", how="left"
        )
        kh_parity[m] = float(np.nanmedian(np.abs(kh_join.K_H_recon - kh_join.K_H_cross)))
        rad = log_radius(bundles[m], shared["sids"])
        frames_Q[m] = {}
        for t in TARGETS:
            frames_Q[m][t] = pd.DataFrame(
                {
                    "sample_id": shared["sids"],
                    "model": m,
                    "target": t,
                    "E_Q_cross": qfld["per_target"][t]["E_Q_cross"],
                    "mse_G": conf_store[m][t]["mse_G"],
                    "r2_G": conf_store[m][t]["r2_G"],
                    "log_knn_radius": rad,
                    "local_eval_label_variance": conf_store[m][t]["local_eval_label_variance"],
                    "local_evaluation_count": conf_store[m][t]["n_eval"],
                }
            )
        print(f"[cmtac] {m} KH_parity={kh_parity.get(m)} remain={_remaining(t0, cfg.wall_s):.0f}s", flush=True)

    # Decoder energy after all Q fields exist. ViT-B ranks reused; others only if wall remains.
    if reused_D is not None and REFERENCE in frames_Q:
        frames_D[REFERENCE] = {}
        for t in TARGETS:
            df = frames_Q[REFERENCE][t].copy()
            df["E_D_S"] = reused_D[t]
            frames_D[REFERENCE][t] = df
        seed_rows.append({"model": REFERENCE, "passed": True, "median_rho_E": float("nan"), "reused_vitb": True})
    if not cfg.skip_decoder:
        for m in models:
            if m == REFERENCE and REFERENCE in frames_D:
                continue
            if m not in frames_Q or not ckpts[m]["three_seed_ok"]:
                continue
            if _remaining(t0, cfg.wall_s) < 240:
                break
            print(f"[cmtac] D {m}", flush=True)
            decoders = {}
            for seed in DECODER_SEEDS:
                decoders[int(seed)] = load_model(Path(ckpts[m]["three_seed"][int(seed)]["path"]), in_dim=bundles[m]["D"], seed=int(seed))
            rows_X = np.array([shared["sid_to_row"][int(s)] for s in shared["sids"]])
            X_anc = bundles[m]["X"][rows_X]
            per_seed_E = {t: {} for t in TARGETS}
            for seed, model in decoders.items():
                if _remaining(t0, cfg.wall_s) < 60:
                    resource_capped = True
                    break
                for t in TARGETS:
                    fld = eval_decoder_field(model, X_anc, conf_store[m][t]["w"], normalized=True, sphere_residual=True)
                    per_seed_E[t][int(seed)] = fld["E"]
            if all(len(per_seed_E[t]) == len(DECODER_SEEDS) for t in TARGETS):
                frames_D[m] = {}
                for t in TARGETS:
                    rel = seed_reliability_E(per_seed_E[t])
                    seed_rows.append({"model": m, "target": t, "passed": rel["passed"], "median_rho_E": rel["median_rho_E"], "reused_vitb": False})
                    Econs = (
                        rel["consensus_rank"]
                        if rel["consensus_rank"] is not None
                        else np.median(np.column_stack(list(per_seed_E[t].values())), axis=1)
                    )
                    df = frames_Q[m][t].copy()
                    df["E_D_S"] = Econs
                    frames_D[m][t] = df
            del decoders

    write_json(out / "kh_recon_parity.json", kh_parity, force=True)
    write_df(out / "seed_reliability.csv", pd.DataFrame(seed_rows) if seed_rows else pd.DataFrame([{"status": "skipped"}]), force=True)

    if not frames_Q:
        return _finish(
            out,
            t0,
            cfg,
            decide(
                tests_ok=True,
                parity_ok=True,
                coverage_ok=True,
                r1=None,
                rd=None,
                vitb_q_bar=None,
                n_rep_q=0,
                resource_capped=True,
                q_ran=False,
            ),
            extra={"resource_capped": True},
        )

    anchor_rows = []
    for m, tmap in frames_Q.items():
        for t, df in tmap.items():
            aQ = per_target_assoc(df, "E_Q_cross", "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
            row = {
                "model": m,
                "target": t,
                "rho_Q": aQ["observed"],
                "p_Q": aQ["p_mc"],
                "sign_Q": float(np.sign(aQ["observed"])),
                "rho_D": float("nan"),
                "p_D": float("nan"),
                "sign_D": float("nan"),
                "agree_ctl": float("nan"),
                "analysis": "C_confirmatory",
                "primary_outcome": "mse_G",
            }
            if m in frames_D and t in frames_D[m] and "E_D_S" in frames_D[m][t].columns:
                aD = per_target_assoc(frames_D[m][t], "E_D_S", "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
                agr = associate(frames_D[m][t].E_D_S.to_numpy(float), df.E_Q_cross.to_numpy(float), None)
                from geometry.physics_task_aligned_curvature.inference import controlled

                agr_c = controlled(frames_D[m][t].assign(E_Q_cross=df.E_Q_cross.to_numpy(float)), "E_D_S", "E_Q_cross")
                row.update({"rho_D": aD["observed"], "p_D": aD["p_mc"], "sign_D": float(np.sign(aD["observed"])), "agree_ctl": agr_c["controlled"]})
                agree_rows.append({"model": m, "target": t, "raw": agr["raw"], "rho_ctl": agr_c["controlled"]})
            per_rows.append(row)
            for i in range(len(df)):
                rec = {k: df.iloc[i][k] for k in df.columns}
                if m in frames_D and t in frames_D[m]:
                    rec["E_D_S"] = frames_D[m][t].E_D_S.iloc[i]
                rec["kind_Q"] = "Q_residual_sphere_normal_cross"
                rec["kind_D"] = "D_residual_sphere_normal_or_reused_rank"
                rec["analysis"] = "C_confirmatory"
                anchor_rows.append(rec)
            write_json(out / f"assoc_{m}_{t}.json", {"Q": aQ}, force=True)

    write_df(out / "per_target_results.csv", pd.DataFrame(per_rows), force=True)
    write_df(out / "cross_instrument_agreement.csv", pd.DataFrame(agree_rows) if agree_rows else pd.DataFrame([{"status": "D_not_run"}]), force=True)
    write_df(out / "per_anchor_task_aligned.parquet", pd.DataFrame(anchor_rows), force=True)

    print("[cmtac] R1", flush=True)
    rep_models = tuple(m for m in REPLICATION if m in frames_Q)
    r1 = synchronized_cross_model(frames_Q, "E_Q_cross", "mse_G", rep_models, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
    model_p2 = {}
    for m, tmap in frames_Q.items():
        model_p2[m] = synchronized_aggregate(tmap, "E_Q_cross", "mse_G", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
        model_p2[m]["name"] = f"P2_{m}"

    rd = None
    d_models = tuple(m for m in REPLICATION if m in frames_D)
    if d_models:
        rd = synchronized_cross_model(frames_D, "E_D_S", "mse_G", d_models, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
        rd["name"] = "RD_bar_rho_D"

    ps = [r1["p_mc"]]
    names = ["R1"]
    if rd is not None:
        ps.append(rd["p_mc"])
        names.append("RD")
    hs = holm(np.asarray(ps))
    r1["p_holm"] = float(hs[0])
    r1["pass_holm"] = bool(r1["observed"] > 0 and hs[0] <= 0.05)
    r1["name"] = "R1_bar_rho_Q_replication"
    if rd is not None:
        rd["p_holm"] = float(hs[1])
        rd["pass_holm"] = bool(rd["observed"] > 0 and hs[1] <= 0.05)

    vitb_bar = float(model_p2[REFERENCE]["observed"]) if REFERENCE in model_p2 else None
    write_json(
        out / "permutation_bootstrap_summary.json",
        {"R1": r1, "RD": rd, "per_model_P2": model_p2, "kh_parity": kh_parity, "vitb_p2_expected": VITB_P2},
        force=True,
    )
    prim_rows = [{k: v for k, v in r1.items() if k not in ("per_model_targets",)} | {"family": "R1"}]
    if rd is not None:
        prim_rows.append({k: v for k, v in rd.items() if k not in ("per_model_targets",)} | {"family": "RD"})
    for m, rec in model_p2.items():
        prim_rows.append({k: v for k, v in rec.items() if k != "per_target"} | {"family": f"P2_{m}", "per_target": json.dumps(rec.get("per_target", {}))})
    write_df(out / "primary_results.csv", pd.DataFrame(prim_rows), force=True)

    mag_r_signs = {m: float(np.sign(frames_Q[m]["mag_r_desi"].pipe(lambda d: associate(d.E_Q_cross.to_numpy(float), d.mse_G.to_numpy(float), None)["raw"]))) for m in frames_Q}
    write_json(out / "mag_r_sign_table.json", {"per_model_raw_sign": mag_r_signs, "note": "descriptive; not a primary test"}, force=True)

    decision = decide(
        tests_ok=tests["all_passed"],
        parity_ok=bool(parity.get("ok")),
        coverage_ok=bool(leak["ok"]),
        r1=r1,
        rd=rd,
        vitb_q_bar=vitb_bar,
        n_rep_q=len(rep_models),
        resource_capped=resource_capped,
        q_ran=q_ran,
    )
    per_target_df = pd.DataFrame(per_rows)
    agree_df = pd.DataFrame(agree_rows) if agree_rows else None
    write_figures(out, per_target=per_target_df, agree=agree_df, r1=r1, model_p2=model_p2)
    write_markdowns(out, decision=decision, r1=r1, rd=rd, model_p2=model_p2, per_target=per_target_df, kh_parity=kh_parity, runtime=time.time() - t0, tests=tests)
    return _finish(out, t0, cfg, decision, extra={"kh_parity": kh_parity, "resource_capped": resource_capped, "n_models_Q": len(frames_Q)})


def _block(out, t0, reason, extra):
    d = decide(
        tests_ok=False,
        parity_ok=False,
        coverage_ok=False,
        r1=None,
        rd=None,
        vitb_q_bar=None,
        n_rep_q=0,
        resource_capped=False,
        q_ran=False,
    )
    d["reason"] = reason
    return _finish(out, t0, ExpConfig(), d, extra=extra)


def _finish(out, t0, cfg, decision, extra=None):
    elapsed = time.time() - t0
    write_json(out / "decision.json", decision, force=True)
    summary = {
        "label": decision["label"],
        "R1": decision.get("R1"),
        "RD": decision.get("RD"),
        "runtime_s": elapsed,
        "output_dir": str(out),
        **(extra or {}),
    }
    write_json(out / "summary.json", summary, force=True)
    write_json(out / "runtime.json", {"wall_s": elapsed, "env": _env(), "protocol": PROTOCOL}, force=True)
    if decision.get("R1") is not None:
        write_json(
            out / "COMPLETE.json",
            {
                "status": "complete",
                "label": decision["label"],
                "n_anchors": 512 if not getattr(cfg, "smoke", False) else 8,
                "wall_s": elapsed,
                "manuscript_modified": False,
            },
            force=True,
        )
    print("[cmtac] done", decision["label"], f"{elapsed:.1f}s", flush=True)
    return {"decision": decision, "elapsed_s": elapsed}
