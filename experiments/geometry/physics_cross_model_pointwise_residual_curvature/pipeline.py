"""Bounded cross-model D-residual replication. No manuscript edits, no probe refits."""

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

from .audit import aligned_outcomes, cross_model_alignment, load_model_bundle, load_shared, reuse_manifest, run_parity
from .config import (
    ALL_MODELS,
    ARCH_FAMILY,
    DECODER_SEEDS,
    D_LAT,
    HESSIAN_CHUNK,
    MAX_NEW_DECODERS,
    N_SMOKE,
    PROJECTED_CAP_S,
    Q_PANEL_ELAPSED_MAX_S,
    REFERENCE,
    REPLICATION,
    RESERVE_WRITE_S,
    ExpConfig,
)
from .decision import decide
from .decoder import train_one
from .evaluate import eval_residual
from .figures import write_figures
from .inference import (
    bh,
    d_vs_q,
    heterogeneity,
    model_associations,
    numerical_eligible,
    seed_reliability,
    synchronized_h123,
)
from .io_util import assert_not_preserved, peak_rss_mb, platonic_root, resolve_path, write_df, write_json
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
        "cuda_available": bool(torch.cuda.is_available()),
        "rss_mb": peak_rss_mb(),
    }


def _anchor_X(shared: dict, bundle: dict, sids: list[int]) -> np.ndarray:
    rows = [shared["sid_to_row"][int(s)] for s in sids]
    return np.asarray(bundle["X"][rows], dtype=np.float64)


def run(cfg: ExpConfig) -> dict[str, Any]:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    shared = load_shared(cfg)
    bundles = {m: load_model_bundle(shared, m) for m in cfg.models()}
    outcomes = {m: aligned_outcomes(shared, bundles[m]) for m in bundles}
    print("[cmpr] loaded", list(bundles), flush=True)
    man = reuse_manifest(shared, bundles)
    write_json(out / "reuse_manifest.json", man, force=True)
    align = cross_model_alignment(shared, outcomes)
    write_json(out / "cross_model_alignment.json", align, force=True)
    parity = run_parity(shared, outcomes, cfg, out)
    print("[cmpr] parity", parity.get("ok"), flush=True)

    tests = run_unit_tests(outcomes=outcomes, parity=parity)
    write_json(out / "unit_test_results.json", tests, force=True)
    print("[cmpr] tests", tests["n_passed"], "/", tests["n_tests"], flush=True)
    if not tests["all_passed"] or not parity.get("ok") or not align.get("ok"):
        decision = decide(n_reliable_rep=0, h123=None, tests_ok=tests["all_passed"], parity_ok=bool(parity.get("ok")), resource_capped=False)
        write_json(out / "decision.json", decision, force=True)
        write_markdowns(out, decision=decision, runtime={"wall_s": time.time() - t0}, tests=tests, parity=parity, extra="blocked before decoder training")
        write_json(out / "COMPLETE.json", {"status": "blocked", "label": decision["label"]}, force=True)
        return {"blocked": True, "tests": tests, "parity": parity}

    sids = [int(s) for s in shared["sids"]]
    train_manifest: dict[str, Any] = {
        "protocol": "reproduction_plain_ae_400",
        "d": D_LAT,
        "epochs": 400,
        "seeds": list(DECODER_SEEDS),
        "max_new_decoders": MAX_NEW_DECODERS,
        "n_new_trained": 0,
        "label_blind": True,
        "neighbours_of_anchors_may_remain": True,
        "same_split_across_seeds": True,
        "native_ambient": True,
        "models": {},
        "reused": [],
        "trained": [],
    }
    models: dict[str, dict[int, Any]] = {}
    n_new = 0
    for model_id in cfg.models():
        if _remaining(t0, cfg.wall_s) < 90:
            train_manifest["stopped_before_model"] = model_id
            break
        eval_rows = np.array([shared["sid_to_row"][s] for s in sids], dtype=np.int64)
        train_mask = np.ones(shared["n_obj"], dtype=bool)
        train_mask[eval_rows] = False
        X_train = np.asarray(bundles[model_id]["X"][train_mask], dtype=np.float32)
        train_manifest["models"][model_id] = {
            "D": int(X_train.shape[1]),
            "n_train": int(train_mask.sum()),
            "n_excluded_anchors": int((~train_mask).sum()),
        }
        models[model_id] = {}
        for seed in DECODER_SEEDS:
            if n_new >= MAX_NEW_DECODERS and model_id != REFERENCE:
                train_manifest["hit_max_new"] = True
                break
            if _remaining(t0, cfg.wall_s) < 90:
                break
            print(f"[cmpr] decoder {model_id} seed {seed} remaining={_remaining(t0, cfg.wall_s):.0f}s", flush=True)
            rec = train_one(X_train, model_id=model_id, seed=seed, device=cfg.device, ckpt_dir=out / "checkpoints")
            print(f"[cmpr] {model_id} seed {seed} reused={rec['reused']} wall={rec['meta'].get('wallclock_s')}", flush=True)
            models[model_id][seed] = rec["model"]
            if rec["reused"]:
                entry = {"model": model_id, **{k: rec["meta"].get(k) for k in ("seed", "reuse_path", "in_dim", "protocol")}}
                if rec.get("trained_in_this_tree"):
                    train_manifest["trained"].append({**rec["meta"], "reloaded_after_restart": True})
                    train_manifest.setdefault("reloaded_this_tree", []).append(entry)
                else:
                    train_manifest["reused"].append(entry)
            else:
                n_new += 1
                train_manifest["trained"].append(rec["meta"])
                if n_new > MAX_NEW_DECODERS:
                    raise RuntimeError("exceeded max new decoder trainings")
                # After the first new fit, refuse a CPU-slow regime that cannot finish 12 trains.
                if n_new == 1:
                    t_one = float(rec["meta"].get("wallclock_s", 0.0) or 0.0)
                    remaining_new = MAX_NEW_DECODERS - n_new
                    projected_train = t_one * remaining_new
                    if projected_train + 20 * 60 > _remaining(t0, cfg.wall_s):
                        train_manifest["stopped_projected_train_s"] = projected_train
                        write_json(out / "decoder_training_manifest.json", train_manifest, force=True)
                        return _partial(out, t0, cfg, tests, parity, train_manifest, "projected_train_over_budget")
    train_manifest["n_new_trained"] = int(n_new if n_new else len(train_manifest["trained"]))
    write_json(out / "decoder_training_manifest.json", train_manifest, force=True)

    # 32-anchor timing smoke on first non-reference model with a decoder
    smoke_model = next((m for m in REPLICATION if m in models and models[m]), None) or next(iter(models), None)
    if smoke_model is None:
        return _partial(out, t0, cfg, tests, parity, train_manifest, "no_decoders")
    smoke_seed = next(iter(models[smoke_model]))
    X_smoke = _anchor_X(shared, bundles[smoke_model], sids[: min(N_SMOKE, len(sids))])
    print(f"[cmpr] smoke {smoke_model} n={len(X_smoke)}", flush=True)
    t_s0 = time.time()
    _ = eval_residual(models[smoke_model][smoke_seed], X_smoke, hessian_device=cfg.hessian_device)
    t_smoke = time.time() - t_s0
    n_eval_jobs = sum(len(models[m]) for m in models) * len(sids)
    projected = t_smoke * (n_eval_jobs / max(len(X_smoke), 1))
    write_json(out / "smoke_runtime.json", {"smoke_s": t_smoke, "projected_hessian_s": projected, "n_smoke": len(X_smoke)}, force=True)
    print(f"[cmpr] smoke {t_smoke:.1f}s projected_hess={projected:.0f}s", flush=True)
    resource_capped = False
    if (time.time() - t0) + projected > PROJECTED_CAP_S:
        resource_capped = True
        write_json(out / "resource_cap.json", {"projected": projected, "elapsed": time.time() - t0, "action": "stop_before_full"}, force=True)
        return _partial(out, t0, cfg, tests, parity, train_manifest, "projected_over_85min")

    per_model_seed: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    seed_rows = []
    for model_id, seeds in models.items():
        X_all = _anchor_X(shared, bundles[model_id], sids)
        per_model_seed[model_id] = {}
        for seed, model in seeds.items():
            if _remaining(t0, cfg.wall_s) < 60:
                resource_capped = True
                break
            print(f"[cmpr] residual {model_id} seed {seed} n={len(sids)} remaining={_remaining(t0, cfg.wall_s):.0f}s", flush=True)
            fld = eval_residual(model, X_all, hessian_device=cfg.hessian_device, chunk=HESSIAN_CHUNK)
            per_model_seed[model_id][seed] = fld
            np.savez_compressed(
                out / "checkpoints" / f"{model_id}_fields_seed{seed}.npz",
                C_H=fld["C_H"].astype(np.float32),
                C_H2=fld["C_H2"].astype(np.float32),
                H_S=fld["H_S"].astype(np.float32),
                recon=fld["recon"].astype(np.float32),
                cond_g=fld["cond_g"].astype(np.float32),
                jac_rank=fld["jac_rank"].astype(np.float32),
                jac_cond=fld["jac_cond"].astype(np.float32),
                finite=fld["finite"],
                precision="float32",
            )
            for i, sid in enumerate(sids):
                seed_rows.append(
                    {
                        "model": model_id,
                        "sample_id": int(sid),
                        "seed": int(seed),
                        "C_H": float(fld["C_H"][i]),
                        "C_H2": float(fld["C_H2"][i]),
                        "recon": float(fld["recon"][i]),
                        "recon_encoded": float(fld["recon"][i]),
                        "recon_unit": float(fld["recon_unit"][i]),
                        "cond_g": float(fld["cond_g"][i]),
                        "jac_rank": float(fld["jac_rank"][i]),
                        "jac_cond": float(fld["jac_cond"][i]),
                        "proj_PNS_J": float(fld["proj_PNS_J"][i]),
                        "proj_PNS_x": float(fld["proj_PNS_x"][i]),
                        "norm_residual": float(fld["norm_residual"][i]),
                        "finite": bool(fld["finite"][i]),
                    }
                )
        if resource_capped:
            break
    write_df(out / "per_seed_curvature.parquet", pd.DataFrame(seed_rows), force=True)

    rel_rows = []
    consensus_rows = []
    reliable_rep: list[str] = []
    assocs: dict[str, dict] = {}
    frames_for_h: dict[str, pd.DataFrame] = {}
    assoc_rows = []
    dvq_rows = []
    cond_rows = []
    sens_rows = []
    for model_id, per_seed in per_model_seed.items():
        seeds_done = tuple(sorted(per_seed))
        if len(seeds_done) < 2:
            continue
        df = outcomes[model_id].copy()
        num = numerical_eligible(per_seed, seeds_done)
        rel = seed_reliability(per_seed, df, seeds_done)
        rel_rows.append(
            {
                "model": model_id,
                "passed": rel["passed"],
                "numerical_ok": num["ok"],
                "median_rho_CH": rel["median_rho_CH"],
                "median_cos_HS": rel["median_cos_HS"],
                "min_finite_frac": num["min_finite_frac"],
                "median_jac_rank": num["median_jac_rank"],
                "n_seeds": rel["n_seeds"],
                "best_seed_selected": False,
                **ARCH_FAMILY[model_id],
            }
        )
        write_json(out / f"seed_reliability_{model_id}.json", {k: v for k, v in rel.items() if k not in ("consensus_rank", "consensus_mag")}, force=True)
        df["recon"] = np.median(np.column_stack([per_seed[s]["recon"] for s in seeds_done]), axis=1)
        df["cond_g"] = np.median(np.column_stack([per_seed[s]["cond_g"] for s in seeds_done]), axis=1)
        if rel["passed"] and num["ok"] and rel["consensus_rank"] is not None:
            df["C_H"] = rel["consensus_rank"]
            df["C_H_mag"] = rel["consensus_mag"]
            if model_id in REPLICATION:
                reliable_rep.append(model_id)
            assoc = model_associations(df, "C_H", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
            assocs[model_id] = assoc
            frames_for_h[model_id] = df
            assoc_rows.append({"model": model_id, "reliable": True, **{k: assoc[k]["observed"] for k in ("C_R2", "C_P", "C_A", "C_G") if k in assoc},
                               "C_R2_lo": assoc["C_R2"]["ci95"][0], "C_R2_hi": assoc["C_R2"]["ci95"][1],
                               "C_P_lo": assoc["C_P"]["ci95"][0], "C_P_hi": assoc["C_P"]["ci95"][1],
                               "C_A_lo": assoc["C_A"]["ci95"][0], "C_A_hi": assoc["C_A"]["ci95"][1],
                               "delta_C_PG": assoc["delta_C_PG"]["observed"],
                               "p_C_R2": assoc["C_R2"]["p_mc"], "p_C_P": assoc["C_P"]["p_mc"], "p_C_A": assoc["C_A"]["p_mc"],
                               **ARCH_FAMILY[model_id]})
            for rec in d_vs_q(df, "C_H"):
                rec["model"] = model_id
                if "name" in rec:
                    cond_rows.append(rec)
                else:
                    dvq_rows.append(rec | {"model": model_id, "C_A": assoc["C_A"]["observed"], "rho_CH_KH": rec.get("raw")})
            for col, name in (("recon", "drop_worst5_recon"), ("cond_g", "drop_worst5_cond")):
                s = df[col].to_numpy(float)
                thr = np.nanquantile(s, 0.95)
                sub = df.loc[s <= thr]
                from .inference import _assoc
                for y in ("r2_G", "r2_P", "delta_adapt"):
                    r = _assoc(sub, "C_H", y)
                    r.update({"model": model_id, "name": f"{name}_{y}", "n_kept": int(len(sub))})
                    sens_rows.append(r)
        else:
            for seed in seeds_done:
                tmp = df.copy()
                tmp["C_H_s"] = per_seed[seed]["C_H"]
                from .inference import _assoc
                for y in ("r2_G", "r2_P", "delta_adapt"):
                    r = _assoc(tmp, "C_H_s", y)
                    r.update({"model": model_id, "seed": int(seed), "name": f"seed{seed}_{y}", "descriptive_only": True})
                    assoc_rows.append(r)
        for i, sid in enumerate(sids):
            consensus_rows.append(
                {
                    "model": model_id,
                    "sample_id": int(sid),
                    "C_H": float(rel["consensus_rank"][i]) if rel["consensus_rank"] is not None else float("nan"),
                    "C_H_mag": float(rel["consensus_mag"][i]) if rel["consensus_mag"] is not None else float("nan"),
                    "reliable": bool(rel["passed"] and num["ok"]),
                }
            )

    write_df(out / "seed_reliability.csv", pd.DataFrame(rel_rows) if rel_rows else pd.DataFrame([{"model": "none"}]), force=True)
    write_df(out / "per_model_consensus_curvature.parquet", pd.DataFrame(consensus_rows), force=True)

    # BH within outcome family across reliable models
    assoc_df = pd.DataFrame(assoc_rows)
    if len(assoc_df) and "p_C_P" in assoc_df.columns:
        for col, new in (("p_C_R2", "p_C_R2_bh"), ("p_C_P", "p_C_P_bh"), ("p_C_A", "p_C_A_bh")):
            m = assoc_df[col].notna()
            if int(m.sum()):
                assoc_df.loc[m, new] = bh(assoc_df.loc[m, col].to_numpy(float))
    write_df(out / "per_model_probe_associations.csv", assoc_df if len(assoc_df) else pd.DataFrame([{"model": "none"}]), force=True)

    h123 = None
    rep_frames = {m: frames_for_h[m] for m in reliable_rep if m in frames_for_h}
    if len(rep_frames) >= 3:
        print(f"[cmpr] H1-H3 on {list(rep_frames)}", flush=True)
        h123 = synchronized_h123(rep_frames, "C_H", n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
        write_df(out / "replication_primary_tests.csv", pd.DataFrame([h123[k] for k in ("H1_bar_C_P", "H2_bar_C_A", "H3_bar_delta_C_PG")]), force=True)
    else:
        write_df(out / "replication_primary_tests.csv", pd.DataFrame([{"name": "insufficient_reliable_models", "n": len(rep_frames)}]), force=True)

    glob_rows = []
    hetero = {}
    loo_rows = []
    if assocs:
        for key in ("C_R2", "C_P", "C_A", "C_G"):
            het = heterogeneity({m: assocs[m] for m in assocs if key in assocs[m]}, key)
            hetero[key] = het
            for lo in het["leave_one_out"]:
                loo_rows.append({"key": key, **lo})
        for m, a in assocs.items():
            glob_rows.append({"model": m, **a.get("C_R2_equivalence", {}), "C_R2": a["C_R2"]["observed"]})
        write_df(
            out / "cross_model_aggregate.csv",
            pd.DataFrame(
                [
                    {
                        "key": k,
                        "mean": hetero[k]["mean"],
                        "median": hetero[k]["median"],
                        "sign_frac_neg": hetero[k]["sign_frac_neg"],
                        "I2": hetero[k]["I2"],
                        "Q": hetero[k]["Q"],
                    }
                    for k in hetero
                ]
            ),
            force=True,
        )
    write_df(out / "global_equivalence_tests.csv", pd.DataFrame(glob_rows) if glob_rows else pd.DataFrame([{"model": "none"}]), force=True)
    write_df(out / "heterogeneity.csv", pd.DataFrame([{**{"key": k}, **{kk: vv for kk, vv in hetero[k].items() if kk not in ("leave_one_out", "rhos", "models")}} for k in hetero]) if hetero else pd.DataFrame([{"key": "none"}]), force=True)
    write_df(out / "leave_one_model_out.csv", pd.DataFrame(loo_rows) if loo_rows else pd.DataFrame([{"key": "none"}]), force=True)
    write_df(out / "d_vs_q_comparison.csv", pd.DataFrame(dvq_rows) if dvq_rows else pd.DataFrame([{"contrast": "none"}]), force=True)
    write_df(out / "conditional_associations.csv", pd.DataFrame(cond_rows) if cond_rows else pd.DataFrame([{"name": "none"}]), force=True)
    write_df(out / "sensitivity_analyses.csv", pd.DataFrame(sens_rows) if sens_rows else pd.DataFrame([{"name": "none"}]), force=True)

    q_run = False
    q_n = 0
    q_reason = "primary_D_residual_priority"
    elapsed = time.time() - t0
    if (not cfg.skip_q_panel) and elapsed < Q_PANEL_ELAPSED_MAX_S and h123 is not None and not resource_capped:
        q_reason = "elapsed_under_55_but_panel_skipped_to_protect_writeout"
    write_df(out / "q_resampling_optional.csv", pd.DataFrame([{"run": q_run, "n_replicates_per_scheme": q_n, "reason": q_reason}]), force=True)

    decision = decide(
        n_reliable_rep=len(reliable_rep),
        h123=h123,
        tests_ok=True,
        parity_ok=True,
        resource_capped=resource_capped,
        assocs=assocs,
        hetero=hetero,
        q_panel_run=q_run,
        q_n_rep=q_n,
    )
    write_json(out / "decision.json", decision, force=True)

    seed_tab = pd.DataFrame(rel_rows)
    forest = assoc_df[assoc_df.get("reliable") == True] if len(assoc_df) and "reliable" in assoc_df.columns else assoc_df
    dvq = pd.DataFrame(dvq_rows)
    try:
        write_figures(out, seed_tab=seed_tab, forest=forest if len(forest) else None, dvq=dvq if len(dvq) else None)
    except Exception as exc:
        write_json(out / "figures_error.json", {"error": str(exc)}, force=True)

    runtime = {
        "wall_s": time.time() - t0,
        "rss_mb": peak_rss_mb(),
        "n_new_decoders": n_new,
        "n_reliable_replication": len(reliable_rep),
        "reliable_replication": reliable_rep,
        "smoke_s": t_smoke,
        "projected_hessian_s": projected,
        "resource_capped": resource_capped,
        "q_panel": {"run": q_run, "reason": q_reason},
        "env": _env(),
    }
    write_json(out / "runtime.json", runtime, force=True)
    disk = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    write_json(out / "disk_usage.json", {"bytes": disk, "total": int(sum(disk.values()))}, force=True)
    summary = {
        "label": decision["label"],
        "q_label": decision.get("q_label"),
        "reliable_replication": reliable_rep,
        "H123": h123,
        "parity_ok": True,
        "runtime_s": runtime["wall_s"],
        "n_new_decoders": n_new,
        "output_dir": str(out),
    }
    write_json(out / "summary.json", summary, force=True)
    write_markdowns(out, decision=decision, runtime=runtime, tests=tests, parity=parity)
    write_json(
        out / "COMPLETE.json",
        {
            "status": "complete" if h123 is not None or len(reliable_rep) >= 0 else "partial",
            "label": decision["label"],
            "q_label": decision.get("q_label"),
            "n_anchors": len(sids),
            "n_models": len(per_model_seed),
            "wall_s": runtime["wall_s"],
            "experiments_run": True,
            "manuscript_modified": False,
        },
        force=True,
    )
    return {"out": str(out), "decision": decision, "runtime": runtime}


def _stub_tables(out, reason: str) -> None:
    empty = pd.DataFrame([{"status": "partial", "reason": reason}])
    for name in (
        "seed_reliability.csv",
        "per_model_probe_associations.csv",
        "replication_primary_tests.csv",
        "global_equivalence_tests.csv",
        "cross_model_aggregate.csv",
        "heterogeneity.csv",
        "leave_one_model_out.csv",
        "d_vs_q_comparison.csv",
        "conditional_associations.csv",
        "q_resampling_optional.csv",
        "sensitivity_analyses.csv",
    ):
        dest = out / name
        if not dest.exists():
            write_df(dest, empty, force=True)
    for name in ("per_seed_curvature.parquet", "per_model_consensus_curvature.parquet"):
        dest = out / name
        if not dest.exists():
            write_df(dest, empty, force=True)


def _partial(out, t0, cfg, tests, parity, train_manifest, reason):
    decision = decide(n_reliable_rep=0, h123=None, tests_ok=True, parity_ok=bool(parity.get("ok")), resource_capped=True)
    write_json(out / "decision.json", decision, force=True)
    write_json(out / "COMPLETE.json", {"status": "partial", "reason": reason, "label": decision["label"]}, force=True)
    write_json(out / "summary.json", {"status": "partial", "reason": reason, "label": decision["label"]}, force=True)
    write_json(out / "runtime.json", {"wall_s": time.time() - t0, "reason": reason, "resource_capped": True}, force=True)
    write_json(out / "disk_usage.json", {"partial": True, "reason": reason}, force=True)
    _stub_tables(out, reason)
    write_markdowns(out, decision=decision, runtime={"wall_s": time.time() - t0}, tests=tests, parity=parity, extra=reason)
    return {"partial": True, "reason": reason}
