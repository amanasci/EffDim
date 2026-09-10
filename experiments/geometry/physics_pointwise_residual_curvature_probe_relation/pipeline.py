"""Bounded ViT-B D-residual vs frozen G/P probes. No manuscript edits."""

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

from .audit import aligned_outcomes, load_bundle, reuse_manifest, run_parity
from .config import (
    CURVATURE_EXPERIMENTS_SHA,
    DECODER_SEEDS,
    D_LAT,
    HASH_SALT,
    HESSIAN_CHUNK,
    MAX_NEW_DECODERS,
    N_SMOKE,
    N_TENSOR_SUBSET,
    OTHER_ENCODERS,
    OUT_REL,
    RESERVE_WRITE_S,
    WALL_S,
    ExpConfig,
)
from .decision import decide
from .decoder import encode_decode, recon_row, train_one
from .figures import write_figures
from .inference import (
    decile_curves,
    drop_worst,
    null_results,
    primary_family,
    q_comparison,
    secondary_table,
    seed_label_permutation,
    seed_reliability,
    sensitivity_table,
    spearman_safe,
    _assoc,
)
from .io_util import (
    assert_not_preserved,
    peak_rss_mb,
    platonic_root,
    resolve_path,
    write_df,
    write_json,
)
from .reports import write_markdowns
from .residual import (
    full_field,
    hash_order,
    hash_subset,
    radial_diagnostics,
    residual_field,
    tensor_at_z,
)
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
        "git_sha_expected": CURVATURE_EXPERIMENTS_SHA,
        "rss_mb": peak_rss_mb(),
    }


def _anchor_X(bundle: dict, sids: list[int]) -> np.ndarray:
    rows = [bundle["sid_to_row"][int(s)] for s in sids]
    return np.asarray(bundle["X"][rows], dtype=np.float64)


def _eval_fields(model, X_anc: np.ndarray, *, hessian_device: str, averaged: bool = True) -> dict[str, np.ndarray]:
    z, y = encode_decode(model, X_anc)
    recon = recon_row(X_anc, y)
    zt = torch.as_tensor(z, dtype=torch.float64)
    if hessian_device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(hessian_device)
        model = model.to(dev).double()
        zt = zt.to(dev)
    else:
        model = model.cpu().double()
        zt = zt.cpu()
    res = residual_field(model, zt, averaged=averaged, chunk=HESSIAN_CHUNK)
    full = full_field(model.cpu().double(), zt.cpu(), averaged=averaged)
    rad = radial_diagnostics(full["H"], res["H"], res["xhat"])
    model.cpu()
    return {
        "z": z,
        "y": y,
        "recon": recon,
        "H_S": res["H"],
        "C_H": res["C_H"],
        "C_H2": res["C_H2"],
        "cond_g": res["cond_g"],
        "xhat": res["xhat"],
        "H_E": full["H"],
        "C_H_full": full["C_H"],
        "cond_g_full": full["cond_g"],
        "cos_HE_HR": rad["cos_HE_HR"],
        "cos_HS_HR": rad["cos_HS_HR"],
        "f_res_vec": rad["f_res_vec"],
    }


def run(cfg: ExpConfig) -> dict[str, Any]:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    bundle = load_bundle(cfg)
    outcomes = aligned_outcomes(bundle)
    print("[prcr] loaded bundle", flush=True)
    man = reuse_manifest(bundle)
    write_json(out / "reuse_manifest.json", man, force=True)
    parity = run_parity(bundle, outcomes, cfg, out)
    print("[prcr] parity", parity.get("ok"), flush=True)

    tests = run_unit_tests(bundle, outcomes)
    print("[prcr] unit tests", tests["n_passed"], "/", tests["n_tests"], flush=True)
    write_json(out / "unit_test_results.json", tests, force=True)
    if not tests["all_passed"]:
        decision = decide(
            seed_passed=False, primary=None, secondary=None, parity_ok=bool(parity.get("ok")), tests_ok=False, resource_capped=False
        )
        write_json(out / "decision.json", decision, force=True)
        write_json(out / "summary.json", {"blocked": "unit_tests", "decision": decision}, force=True)
        write_markdowns(out, decision=decision, seed_rel={}, primary=None, parity=parity, runtime={"wall_s": time.time() - t0}, tests=tests, train_manifest={})
        write_json(out / "COMPLETE.json", {"status": "blocked_tests", "decision": decision["label"]}, force=True)
        return {"blocked": True, "tests": tests}

    sids = [int(s) for s in outcomes.sample_id.tolist()]
    n_obj = bundle["n_obj"]
    eval_rows = np.array([bundle["sid_to_row"][s] for s in sids], dtype=np.int64)
    train_mask = np.ones(n_obj, dtype=bool)
    train_mask[eval_rows] = False
    X_train = np.asarray(bundle["X"][train_mask], dtype=np.float32)
    n_neighbor_kept = 0
    # disclosure: neighbours of anchors remain if they are not themselves anchors
    n_neighbor_kept = int(train_mask.sum())  # all non-anchors
    train_manifest = {
        "n_objects": n_obj,
        "n_eval_anchors_excluded": int((~train_mask).sum()),
        "n_train": int(train_mask.sum()),
        "neighbours_of_anchors_may_remain": True,
        "n_non_anchor_train_points": int(train_mask.sum()),
        "same_split_across_seeds": True,
        "split_changes": "initialization and documented minibatch permutation only",
        "label_blind": True,
        "architecture": "PlainAutoEncoder hidden (250,250,250) silu",
        "epochs": 400,
        "seeds": list(DECODER_SEEDS),
        "max_new_decoders": MAX_NEW_DECODERS,
        "n_new_trained": 0,
        "reused": [],
        "trained": [],
        "protocol": "reproduction_plain_ae_400",
    }

    models = {}
    n_new = 0
    for seed in DECODER_SEEDS:
        if _remaining(t0, cfg.wall_s) < 90:
            train_manifest["stopped_before_seed"] = int(seed)
            break
        print(f"[prcr] train seed {seed} remaining={_remaining(t0, cfg.wall_s):.0f}s", flush=True)
        rec = train_one(X_train, seed=seed, device=cfg.device, ckpt_dir=out / "checkpoints")
        print(f"[prcr] seed {seed} reused={rec['reused']} wall={rec['meta'].get('wallclock_s')}", flush=True)
        models[seed] = rec["model"]
        if rec["reused"]:
            train_manifest["reused"].append(rec["meta"])
        else:
            n_new += 1
            train_manifest["trained"].append(rec["meta"])
        if n_new > MAX_NEW_DECODERS:
            raise RuntimeError("exceeded max new decoder trainings")
    train_manifest["n_new_trained"] = n_new
    write_json(out / "decoder_training_manifest.json", train_manifest, force=True)

    if len(models) < 2:
        decision = decide(
            seed_passed=False, primary=None, secondary=None, parity_ok=bool(parity.get("ok")), tests_ok=True, resource_capped=True
        )
        write_json(out / "decision.json", decision, force=True)
        write_json(out / "COMPLETE.json", {"status": "partial", "reason": "insufficient_decoders"}, force=True)
        return {"partial": True}

    X_all = _anchor_X(bundle, sids)
    order = hash_order(np.asarray(sids), salt=HASH_SALT)
    smoke_sids = [int(s) for s in order[:N_SMOKE]]
    smoke_idx = [sids.index(s) for s in smoke_sids]
    X_smoke = X_all[smoke_idx]
    smoke_seed = next(iter(models))

    print(f"[prcr] smoke {len(smoke_idx)} anchors seed={smoke_seed}", flush=True)
    t_smoke0 = time.time()
    _ = _eval_fields(models[smoke_seed], X_smoke, hessian_device=cfg.hessian_device)
    t_smoke = time.time() - t_smoke0
    print(f"[prcr] smoke done in {t_smoke:.1f}s", flush=True)
    n_full = len(sids)
    projected = t_smoke * (n_full / max(len(smoke_idx), 1)) * len(models)
    # tensor subset extra ~ 128/512 * 1.5
    projected += t_smoke * (N_TENSOR_SUBSET / N_SMOKE) * 1.5
    write_json(
        out / "smoke_runtime.json",
        {"smoke_s": t_smoke, "n_smoke": len(smoke_idx), "projected_hessian_s": projected, "remaining_s": _remaining(t0, cfg.wall_s)},
        force=True,
    )
    resource_capped = False
    if projected > _remaining(t0, cfg.wall_s):
        resource_capped = True
        write_json(
            out / "resource_cap.json",
            {"projected_hessian_s": projected, "remaining_s": _remaining(t0, cfg.wall_s), "action": "stop_before_full_512"},
            force=True,
        )
        decision = decide(
            seed_passed=False, primary=None, secondary=None, parity_ok=bool(parity.get("ok")), tests_ok=True, resource_capped=True
        )
        write_json(out / "decision.json", decision, force=True)
        write_markdowns(
            out, decision=decision, seed_rel={}, primary=None, parity=parity,
            runtime={"wall_s": time.time() - t0}, tests=tests, train_manifest=train_manifest,
            extra={"notes": "Projected full Hessian exceeded remaining wall. Partial outputs only."},
        )
        write_json(out / "COMPLETE.json", {"status": "partial", "reason": "resource_cap"}, force=True)
        write_json(out / "runtime.json", {"wall_s": time.time() - t0, "rss_mb": peak_rss_mb(), "env": _env()}, force=True)
        return {"partial": True, "resource_capped": True}

    per_seed: dict[int, dict[str, np.ndarray]] = {}
    seed_rows = []
    for seed, model in models.items():
        if _remaining(t0, cfg.wall_s) < 60:
            resource_capped = True
            break
        print(f"[prcr] hessian seed {seed} n={len(sids)} remaining={_remaining(t0, cfg.wall_s):.0f}s", flush=True)
        fld = _eval_fields(model, X_all, hessian_device=cfg.hessian_device)
        print(f"[prcr] hessian seed {seed} done remaining={_remaining(t0, cfg.wall_s):.0f}s", flush=True)
        per_seed[seed] = fld
        keep = ("z", "recon", "C_H", "C_H2", "C_H_full", "cond_g", "cos_HE_HR", "cos_HS_HR", "f_res_vec")
        np.savez(out / "checkpoints" / f"fields_seed{seed}.npz", **{k: fld[k] for k in keep})
        for i, sid in enumerate(sids):
            seed_rows.append(
                {
                    "sample_id": int(sid),
                    "seed": int(seed),
                    "C_H": float(fld["C_H"][i]),
                    "C_H2": float(fld["C_H2"][i]),
                    "C_H_full": float(fld["C_H_full"][i]),
                    "cond_g": float(fld["cond_g"][i]),
                    "recon": float(fld["recon"][i]),
                    "cos_HE_HR": float(fld["cos_HE_HR"][i]),
                    "cos_HS_HR": float(fld["cos_HS_HR"][i]),
                    "f_res_vec": float(fld["f_res_vec"][i]),
                }
            )
    write_df(out / "per_seed_curvature.parquet", pd.DataFrame(seed_rows), force=True)

    seeds_done = tuple(sorted(per_seed))
    rel = seed_reliability(per_seed, outcomes, seeds_done)
    rel_json = {k: v for k, v in rel.items() if k not in ("consensus_rank", "consensus_mag")}
    rel_json["pairs"] = rel["pairs"]
    write_json(out / "seed_reliability.json", rel_json, force=True)

    df = outcomes.copy()
    # attach per-seed columns
    for seed in seeds_done:
        df[f"C_H_seed{seed}"] = per_seed[seed]["C_H"]
        df[f"C_H_full_seed{seed}"] = per_seed[seed]["C_H_full"]
        df[f"recon_seed{seed}"] = per_seed[seed]["recon"]
        df[f"cond_seed{seed}"] = per_seed[seed]["cond_g"]
        df[f"cos_HE_HR_seed{seed}"] = per_seed[seed]["cos_HE_HR"]
        df[f"f_res_vec_seed{seed}"] = per_seed[seed]["f_res_vec"]
    df["recon"] = np.median(np.column_stack([per_seed[s]["recon"] for s in seeds_done]), axis=1)
    df["cond_g"] = np.median(np.column_stack([per_seed[s]["cond_g"] for s in seeds_done]), axis=1)
    df["C_H_full"] = np.median(np.column_stack([per_seed[s]["C_H_full"] for s in seeds_done]), axis=1)
    df["cos_HE_HR"] = np.median(np.column_stack([per_seed[s]["cos_HE_HR"] for s in seeds_done]), axis=1)
    df["f_res_vec"] = np.median(np.column_stack([per_seed[s]["f_res_vec"] for s in seeds_done]), axis=1)

    xcol = None
    primary = None
    if rel["passed"] and rel["consensus_rank"] is not None:
        df["C_H"] = rel["consensus_rank"]
        df["C_H_mag"] = rel["consensus_mag"]
        df["C_H2"] = np.asarray(rel["consensus_mag"]) ** 2
        xcol = "C_H"
        primary = primary_family(df, xcol, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
    else:
        df["C_H"] = np.nan
        df["C_H_mag"] = np.median(np.column_stack([per_seed[s]["C_H"] for s in seeds_done]), axis=1)
        df["C_H2"] = df["C_H_mag"] ** 2

    # tensor subset (secondary)
    tensor_rows = []
    subset = [int(s) for s in hash_subset(np.asarray(sids), n=N_TENSOR_SUBSET) if int(s) in set(sids)]
    if seeds_done and _remaining(t0, cfg.wall_s) > 45:
        seed0 = seeds_done[0]
        model = models[seed0]
        z_all = per_seed[seed0]["z"]
        for sid in subset:
            if _remaining(t0, cfg.wall_s) < 30:
                break
            i = sids.index(int(sid))
            zt = torch.as_tensor(z_all[i], dtype=torch.float64)
            tn = tensor_at_z(model, zt, normalized=True)
            tf = tensor_at_z(model, zt, normalized=False)
            tensor_rows.append(
                {
                    "sample_id": int(sid),
                    "seed": int(seed0),
                    "C_B2": tn["C_B2"],
                    "K_dir_D": tn["K_dir_D"],
                    "delta_Scal_D": tn["delta_Scal_D"],
                    "Scal_D": tn["Scal_D"],
                    "f_res_tensor": tn["C_B2"] / max(tf["energy_II_E"], 1e-30),
                    "subset_diagnostic": True,
                }
            )
    tens = pd.DataFrame(tensor_rows)
    if len(tens):
        df = df.merge(tens.drop(columns=["seed"], errors="ignore"), on="sample_id", how="left")

    write_df(out / "per_anchor_curvature.parquet", df.drop(columns=["mag_r_desi_catalog_value"], errors="ignore"), force=True)
    probe_cols = [
        c
        for c in (
            "sample_id",
            "r2_G",
            "r2_P",
            "mse_G",
            "mse_P",
            "mae_G",
            "mae_P",
            "delta_adapt",
            "n_eval",
            "K_H_cross",
            "K_dir_cross",
            "log_knn_radius",
            "local_label_variance",
            "local_evaluation_count",
        )
        if c in df.columns
    ]
    write_df(out / "per_anchor_probe_outcomes.parquet", df[probe_cols], force=True)

    sec = pd.DataFrame()
    qtab = pd.DataFrame()
    sens = pd.DataFrame()
    nulls = {}
    contrasts = pd.DataFrame()
    full_vs = pd.DataFrame()
    deciles = None
    if xcol is not None:
        sec = secondary_table(df, xcol)
        qtab = q_comparison(df, xcol)
        sens = sensitivity_table(df, xcol)
        # worst-5% exclusions
        for col, name in (("recon", "drop_worst5_recon"), ("cond_g", "drop_worst5_cond")):
            sub = drop_worst(df, col, 0.05)
            rec = _assoc(sub, xcol, "r2_G")
            rec["name"] = name + "_R2G"
            rec["n_kept"] = int(len(sub))
            sens = pd.concat([sens, pd.DataFrame([rec])], ignore_index=True)
            rec2 = _assoc(sub, xcol, "r2_P")
            rec2["name"] = name + "_R2P"
            rec2["n_kept"] = int(len(sub))
            sens = pd.concat([sens, pd.DataFrame([rec2])], ignore_index=True)
        nulls = null_results(df, xcol, n_perm=cfg.n_perm_eff())
        nulls["seed_label_perm"] = seed_label_permutation(per_seed, seeds_done)
        if "C_B2" in df.columns and df.C_B2.notna().sum() >= 16:
            sub = df[df.C_B2.notna()]
            for ycol, lab in (("r2_G", "R2G"), ("r2_P", "R2P"), ("delta_adapt", "dAdapt")):
                for xc in ("C_B2", "K_dir_D", "delta_Scal_D"):
                    if xc not in sub.columns:
                        continue
                    rec = _assoc(sub, xc, ycol)
                    rec["name"] = f"subset_{xc}_{lab}"
                    rec["n"] = int(len(sub))
                    sec = pd.concat([sec, pd.DataFrame([rec])], ignore_index=True)
        full_vs = pd.DataFrame(
            [
                {"name": "rho_ctl_CHfull_R2G", **_assoc(df, "C_H_full", "r2_G")},
                {"name": "rho_ctl_CHfull_R2P", **_assoc(df, "C_H_full", "r2_P")},
                {"name": "rho_ctl_CHres_R2G", **_assoc(df, xcol, "r2_G")},
                {"name": "rho_ctl_CHres_R2P", **_assoc(df, xcol, "r2_P")},
                {"name": "rho_ctl_CHfull_DeltaAdapt", **_assoc(df, "C_H_full", "delta_adapt")},
                {"name": "rho_ctl_CHres_DeltaAdapt", **_assoc(df, xcol, "delta_adapt")},
                {
                    "name": "median_cos_HE_HR",
                    "observed": float(np.median(df.cos_HE_HR)),
                    "median_f_res_vec": float(np.median(df.f_res_vec)),
                },
            ]
        )
        contrasts = pd.DataFrame(
            [
                {"name": "P3_delta_rho", "observed": primary["P3"]["observed"], "p_holm": primary["P3"]["p_holm"], "ci95": primary["P3"]["ci95"]},
                {"name": "mean_delta_adapt", "observed": float(df.delta_adapt.mean())},
                {"name": "median_delta_adapt", "observed": float(df.delta_adapt.median())},
                {"name": "frac_patch_R2_gt_global", "observed": float((df.r2_P > df.r2_G).mean())},
            ]
        )
        deciles = decile_curves(df, xcol)
        write_df(out / "primary_tests.csv", pd.DataFrame([primary[k] for k in ("P1", "P2", "P3")]), force=True)
    else:
        # descriptive per-seed associations only
        rows = []
        for seed in seeds_done:
            tmp = df.copy()
            tmp["C_H_s"] = per_seed[seed]["C_H"]
            for ycol in ("r2_G", "r2_P", "mse_G", "mse_P", "delta_adapt"):
                rec = _assoc(tmp, "C_H_s", ycol)
                rec["name"] = f"seed{seed}_{ycol}"
                rec["seed"] = int(seed)
                rows.append(rec)
        sec = pd.DataFrame(rows)
        write_df(out / "primary_tests.csv", pd.DataFrame([{"name": "seed_gate_failed", "observed": float("nan")}]), force=True)

    # per-seed correlations always
    seed_corr_rows = []
    for seed in seeds_done:
        tmp = df.copy()
        tmp["C_H_s"] = per_seed[seed]["C_H"]
        for ycol in ("r2_G", "r2_P", "delta_adapt"):
            rec = _assoc(tmp, "C_H_s", ycol)
            rec["name"] = f"seed{seed}_{ycol}"
            rec["seed"] = int(seed)
            seed_corr_rows.append(rec)
    if seed_corr_rows:
        sec = pd.concat([sec, pd.DataFrame(seed_corr_rows)], ignore_index=True) if len(sec) else pd.DataFrame(seed_corr_rows)

    write_df(out / "secondary_tests.csv", sec if len(sec) else pd.DataFrame([{"name": "empty"}]), force=True)
    write_df(out / "q_comparison.csv", qtab if len(qtab) else pd.DataFrame([{"name": "seed_gate_failed"}]), force=True)
    write_df(out / "sensitivity_analyses.csv", sens if len(sens) else pd.DataFrame([{"name": "seed_gate_failed"}]), force=True)
    write_json(out / "null_results.json", nulls or {"seed_gate_failed": True}, force=True)
    # required filename
    null_rows = []
    for k, v in (nulls or {}).items():
        if isinstance(v, dict) and "observed" in v:
            null_rows.append({"name": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}})
        elif isinstance(v, dict):
            null_rows.append({"name": k, **{kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, bool, str))}})
        else:
            null_rows.append({"name": k, "value": v})
    write_df(out / "null_results.csv", pd.DataFrame(null_rows) if null_rows else pd.DataFrame([{"name": "none"}]), force=True)
    write_df(out / "correlation_contrasts.csv", contrasts if len(contrasts) else pd.DataFrame([{"name": "seed_gate_failed"}]), force=True)
    write_df(out / "full_vs_residual.csv", full_vs if len(full_vs) else pd.DataFrame([{"name": "seed_gate_failed"}]), force=True)

    # optional cross-model: no new training
    cross = {"evaluated": [], "skipped": list(OTHER_ENCODERS), "reason": "no_compatible_decoder_checkpoints_for_seeds_0_1_2"}
    write_json(out / "cross_model_optional.json", cross, force=True)

    decision = decide(
        seed_passed=bool(rel["passed"]),
        primary=primary,
        secondary=sec,
        parity_ok=bool(parity.get("ok")),
        tests_ok=True,
        resource_capped=resource_capped,
    )
    write_json(out / "decision.json", decision, force=True)

    try:
        write_figures(out, per_seed=per_seed, seeds=seeds_done, df=df, xcol=xcol or "C_H_mag", seed_rel=rel, deciles=deciles)
    except Exception as exc:
        write_json(out / "figures_error.json", {"error": str(exc)}, force=True)

    runtime = {
        "wall_s": time.time() - t0,
        "rss_mb": peak_rss_mb(),
        "n_new_decoders": n_new,
        "n_seeds_eval": len(seeds_done),
        "n_anchors": len(sids),
        "smoke_s": t_smoke,
        "projected_hessian_s": projected,
        "resource_capped": resource_capped,
        "env": _env(),
        "n_neighbor_kept_disclosure": n_neighbor_kept,
    }
    write_json(out / "runtime.json", runtime, force=True)

    summary = {
        "label": decision["label"],
        "seed_reliability": rel_json,
        "primary": primary,
        "parity": {k: parity[k] for k in parity if k != "historical_decoder_field"},
        "historical_decoder_field": parity.get("historical_decoder_field"),
        "mean_delta_adapt": float(df.delta_adapt.mean()),
        "median_delta_adapt": float(df.delta_adapt.median()),
        "signs_differ": decision.get("signs_differ"),
        "median_cos_HE_HR": float(np.median(df.cos_HE_HR)) if "cos_HE_HR" in df.columns else None,
        "median_f_res_vec": float(np.median(df.f_res_vec)) if "f_res_vec" in df.columns else None,
        "runtime_s": runtime["wall_s"],
        "output_dir": str(out),
    }
    write_json(out / "summary.json", summary, force=True)
    write_markdowns(
        out,
        decision=decision,
        seed_rel=rel,
        primary=primary,
        parity=parity,
        runtime=runtime,
        tests=tests,
        train_manifest=train_manifest,
    )
    write_json(
        out / "COMPLETE.json",
        {
            "status": "complete" if (rel["passed"] or not resource_capped) else "partial",
            "label": decision["label"],
            "n_anchors": len(sids),
            "n_seeds": len(seeds_done),
            "wall_s": runtime["wall_s"],
        },
        force=True,
    )
    return {"out": str(out), "decision": decision, "runtime": runtime}
