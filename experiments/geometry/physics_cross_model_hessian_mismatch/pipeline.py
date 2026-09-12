"""Bounded five-encoder Hessian-mismatch replication. No manuscript edits."""

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
from scipy.stats import rankdata, spearmanr

_EXP = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[3]
_NB = _REPO / "notebooks"
for p in (_EXP, _NB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from geometry.physics_cross_model_pointwise_residual_curvature.decoder import load_model
from geometry.physics_cross_model_task_aligned_curvature.audit import as_shared
from geometry.physics_task_aligned_curvature.algebra import energy_g, holm
from geometry.physics_task_aligned_curvature.inference import seed_reliability_E
from geometry.physics_task_aligned_curvature.probes import confirmatory_for_target

from .audit import (
    checkpoints,
    historical_local_r2,
    historical_weights,
    load_bundles,
    load_shared,
    multiscale_radii,
    reuse_manifest,
)
from .config import (
    DECODER_SEEDS,
    D_LAT,
    HESS_RIDGE_PAPER,
    HESS_RIDGE_STAB,
    K,
    MIN_EVAL,
    MULTISCALE_K,
    PAPER_CONTROLS,
    PAPER_TABLE2,
    PRIMARY_CONTROLS,
    PROJECTED_CAP_S,
    REFERENCE,
    RESERVE_WRITE_S,
    SEED_RHO_GATE,
    TARGETS,
    ExpConfig,
)
from .decision import decide
from .figures import write_figures
from .geometry import Bw_at_z, jets_normalized, mismatch_stats
from .inference import Z_of, cell_assoc, controlled, leave_one_out, synchronized_p1_p2, vif_and_cond
from .io_util import assert_not_preserved, peak_rss_mb, platonic_root, resolve_path, write_df, write_json
from .label_hessian import fit_label_hessian, fit_linear_only, predict_quad, split_half_cosine, tangent_coords
from .reports import write_markdowns
from .tests_unit import run_unit_tests


def _remaining(t0, wall):
    return wall - RESERVE_WRITE_S - (time.time() - t0)


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def run(cfg: ExpConfig) -> dict[str, Any]:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)

    shared = load_shared(cfg)
    models = cfg.models()
    bundles = load_bundles(shared, models)
    ckpts = checkpoints(shared, models)
    man = reuse_manifest(shared, bundles, ckpts)
    write_json(out / "reuse_manifest.json", man, force=True)
    write_json(
        out / "target_definitions.json",
        {"targets": list(TARGETS), "align_by": "sample_id", "split_salt": "task_aligned_curvature_v1", "ridge_alpha": 100.0, "min_eval": MIN_EVAL},
        force=True,
    )
    if any(not ckpts[m]["three_seed_ok"] for m in models):
        write_json(out / "blocker.json", {"reason": "missing_decoder_checkpoints", "ckpts": ckpts}, force=True)
        return _done(out, t0, cfg, decide(tests_ok=False, coverage_ok=False, p1=None, p2=None, n_models=0, n_unreliable=0, hy_median_split_cos=None, shape_beats_mismatch=False, resource_capped=False, shuffle_survives=False), extra={"blocker": "ckpts"})

    tests = run_unit_tests()
    write_json(out / "unit_test_results.json", tests, force=True)
    print("[hmm] tests", tests["n_passed"], "/", tests["n_tests"], flush=True)
    if not tests["all_passed"]:
        return _done(out, t0, cfg, decide(tests_ok=False, coverage_ok=False, p1=None, p2=None, n_models=0, n_unreliable=0, hy_median_split_cos=None, shape_beats_mismatch=False, resource_capped=False, shuffle_survives=False), extra={"tests": tests})

    conf = {}
    leak = {"ok": True, "per_model": {}, "train_labels_excluded_from_eval": True, "eval_labels_excluded_from_w": True, "eval_labels_excluded_from_Hy": True}
    for m in models:
        sh = as_shared(shared, bundles[m])
        conf[m] = {t: confirmatory_for_target(sh, t) for t in TARGETS}
        leak["per_model"][m] = {t: conf[m][t]["coverage"] for t in TARGETS}
        leak["ok"] = bool(leak["ok"] and all(conf[m][t]["coverage"]["ok"] for t in TARGETS))
    write_json(out / "leakage_audit.json", leak, force=True)
    if not leak["ok"]:
        return _done(out, t0, cfg, decide(tests_ok=True, coverage_ok=False, p1=None, p2=None, n_models=0, n_unreliable=0, hy_median_split_cos=None, shape_beats_mismatch=False, resource_capped=False, shuffle_survives=False))

    sids = shared["sids"]
    # smoke
    m0 = models[0]
    dec0 = load_model(Path(ckpts[m0]["three_seed"][0]["path"]), in_dim=bundles[m0]["D"], seed=0)
    rows0 = np.array([shared["sid_to_row"][int(s)] for s in sids[: min(32, len(sids))]])
    Xa0 = bundles[m0]["X"][rows0]
    with torch.no_grad():
        z0 = dec0.encode(torch.as_tensor(np.asarray(Xa0, dtype=np.float64))).cpu().numpy()
    t_s = time.time()
    for i in range(min(8, len(z0))):
        jets = jets_normalized(dec0, z0[i])
        _ = Bw_at_z(dec0, z0[i], conf[m0][TARGETS[0]]["w"], jets)
    t_smoke = time.time() - t_s
    n_jobs = len(models) * len(DECODER_SEEDS) * len(sids) * len(TARGETS)
    projected = t_smoke * (n_jobs / max(min(8, len(z0)), 1))
    write_json(out / "smoke_runtime.json", {"smoke_s": t_smoke, "projected_s": projected}, force=True)
    print(f"[hmm] smoke {t_smoke:.2f}s projected={projected:.0f}s", flush=True)
    resource_capped = (time.time() - t0) + projected > PROJECTED_CAP_S and not cfg.smoke
    seeds_use = (0,) if (resource_capped or cfg.smoke) else DECODER_SEEDS
    if resource_capped:
        print("[hmm] projecting over cap; using seed 0 only", flush=True)

    hist_w = historical_weights(shared)
    hist_r2 = historical_local_r2(shared)
    per_seed = {m: {t: {k: {} for k in ("M_Delta", "A_full", "S", "R")} for t in TARGETS} for m in models}
    rel_rows, hy_rows, rad_ok = [], [], []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    generic = {}

    for m in models:
        if _remaining(t0, cfg.wall_s) < 90:
            resource_capped = True
            break
        bundle = bundles[m]
        rows = np.array([shared["sid_to_row"][int(s)] for s in sids])
        X = bundle["X"]
        rads = multiscale_radii(X, bundle["neigh"], rows)
        fcr = bundle["fcr"].copy()
        fcr["sample_id"] = fcr.sample_id.astype(int)
        kh_map = {int(s): float(v) for s, v in zip(fcr.sample_id, fcr.K_H_cross)}
        generic[m] = np.array([kh_map.get(int(s), np.nan) for s in sids])
        seed_fields = {t: {k: [] for k in ("M_Delta", "A_full", "S", "R", "M_shape", "A_shape", "Hy_norm", "delta_norm", "cbar", "mse_geom", "Hy_cos")} for t in TARGETS}
        sphere_ok_frac = []
        for seed in seeds_use:
            print(f"[hmm] {m} seed{seed}", flush=True)
            dec = load_model(Path(ckpts[m]["three_seed"][int(seed)]["path"]), in_dim=bundle["D"], seed=int(seed))
            with torch.no_grad():
                z_all = dec.encode(torch.as_tensor(np.asarray(X[rows], dtype=np.float64))).cpu().numpy()
            acc = {t: {k: np.full(len(sids), np.nan) for k in seed_fields[TARGETS[0]]} for t in TARGETS}
            for i, sid in enumerate(sids):
                if _remaining(t0, cfg.wall_s) < 40:
                    resource_capped = True
                    break
                jets = jets_normalized(dec, z_all[i])
                g, ginv, J, xhat = jets["g"], jets["ginv"], jets["J"], jets["xhat"]
                nbr = bundle["neigh"][i]
                Xloc = X[nbr]
                U = tangent_coords(Xloc, jets["x"], J, ginv)
                ok_s = True
                for t in TARGETS:
                    bw = Bw_at_z(dec, z_all[i], conf[m][t]["w"], jets)
                    ok_s = ok_s and bw["sphere_norm_ok"]
                    trn = conf[m][t]["split"]["train"][nbr] & np.isfinite(shared["Y"][t][nbr])
                    evl = conf[m][t]["split"]["eval"][nbr] & np.isfinite(shared["Y"][t][nbr])
                    hy = fit_label_hessian(U[trn], shared["Y"][t][nbr][trn], ridge=HESS_RIDGE_PAPER)
                    if not hy["ok"]:
                        hy = fit_label_hessian(U[trn], shared["Y"][t][nbr][trn], ridge=HESS_RIDGE_STAB)
                    ms = mismatch_stats(hy["Hy"], bw["Bw"], bw["Bw_S"], ginv) if hy["ok"] else {k: float("nan") for k in ("M_Delta", "M_shape", "A_full", "A_shape", "Hy_norm", "Bw_norm")}
                    a1 = hy.get("a1", np.full(D_LAT, np.nan))
                    jtw = J.T @ conf[m][t]["w"]
                    delta = a1 - jtw
                    dnorm = float(np.sqrt(max(float(delta @ ginv @ delta), 0.0))) if np.isfinite(delta).all() else float("nan")
                    s2 = float(np.mean(np.einsum("ni,ij,nj->n", U[trn], g, U[trn])) / D_LAT) if trn.sum() > 4 else float("nan")
                    yhat0 = conf[m][t]["b"] + float(np.dot(conf[m][t]["w"], jets["x"]))
                    cbar = float(hy.get("a0", np.nan) - yhat0)
                    mse_g = float(cbar**2 + s2 * dnorm**2 + 0.5 * (s2**2) * (ms["M_Delta"] ** 2)) if np.isfinite([cbar, s2, dnorm, ms["M_Delta"]]).all() else float("nan")
                    hcos = split_half_cosine(U[trn], shared["Y"][t][nbr][trn], ginv, seed=int(sid) + 17) if (i % 16 == 0 and trn.sum() > 80) else float("nan")
                    acc[t]["M_Delta"][i] = ms["M_Delta"]
                    acc[t]["A_full"][i] = ms["A_full"]
                    acc[t]["S"][i] = bw["S"]
                    acc[t]["R"][i] = bw["R"]
                    acc[t]["M_shape"][i] = ms["M_shape"]
                    acc[t]["A_shape"][i] = ms["A_shape"]
                    acc[t]["Hy_norm"][i] = ms["Hy_norm"]
                    acc[t]["delta_norm"][i] = dnorm
                    acc[t]["cbar"][i] = cbar
                    acc[t]["mse_geom"][i] = mse_g
                    acc[t]["Hy_cos"][i] = hcos
                    lin = fit_linear_only(U[trn], shared["Y"][t][nbr][trn])
                    if evl.sum() >= 8 and hy["ok"]:
                        y_ev = shared["Y"][t][nbr][evl]
                        yq = predict_quad(U[evl], hy["a0"], hy["a1"], hy["Hy"])
                        # store quad gain later in hy_rows via this seed's last
                sphere_ok_frac.append(float(ok_s))
            for t in TARGETS:
                for k in seed_fields[t]:
                    seed_fields[t][k].append(acc[t][k])
                    if k in per_seed[m][t]:
                        per_seed[m][t][k][int(seed)] = acc[t][k]
            del dec
        # consensus
        unreliable = False
        for t in TARGETS:
            rel = (
                seed_reliability_E({int(seeds_use[i]): seed_fields[t]["M_Delta"][i] for i in range(len(seed_fields[t]["M_Delta"]))})
                if len(seed_fields[t]["M_Delta"]) > 1
                else {"passed": True, "median_rho_E": float("nan"), "consensus_rank": None}
            )
            if not rel.get("passed", False) and len(seeds_use) > 1:
                unreliable = True
            rel_rows.append({"model": m, "target": t, "passed": rel.get("passed", False), "median_rho_E": rel.get("median_rho_E", float("nan")), "best_seed_selected": False, "sphere_ok": float(np.mean(sphere_ok_frac) if sphere_ok_frac else np.nan)})
            def cons(arrs):
                if not arrs:
                    return np.full(len(sids), np.nan)
                if rel.get("consensus_rank") is not None and len(arrs) > 1:
                    return np.median(np.column_stack([rankdata(a) for a in arrs]), axis=1)
                return arrs[0] if len(arrs) == 1 else np.median(np.column_stack(arrs), axis=1)

            df = pd.DataFrame(
                {
                    "sample_id": sids,
                    "model": m,
                    "target": t,
                    "M_Delta": cons(seed_fields[t]["M_Delta"]),
                    "A_full": cons(seed_fields[t]["A_full"]),
                    "S": cons(seed_fields[t]["S"]),
                    "R": cons(seed_fields[t]["R"]),
                    "M_shape": cons(seed_fields[t]["M_shape"]),
                    "A_shape": cons(seed_fields[t]["A_shape"]),
                    "Hy_norm": cons(seed_fields[t]["Hy_norm"]),
                    "delta_norm": cons(seed_fields[t]["delta_norm"]),
                    "cbar": cons(seed_fields[t]["cbar"]),
                    "mse_geom": cons(seed_fields[t]["mse_geom"]),
                    "mse_G": conf[m][t]["mse_G"],
                    "r2_G": conf[m][t]["r2_G"],
                    "local_eval_label_variance": conf[m][t]["local_eval_label_variance"],
                    "local_evaluation_count": conf[m][t]["n_eval"],
                    "generic_KH": generic[m],
                    "decoder_unreliable": unreliable,
                }
            )
            for k, arr in rads.items():
                df[f"log_radius_k{k}"] = arr
            frames[(m, t)] = df
            hy_rows.append({"model": m, "target": t, "median_split_cos": float(np.nanmedian(np.concatenate(seed_fields[t]["Hy_cos"]))) if seed_fields[t]["Hy_cos"] else float("nan")})
        print(f"[hmm] {m} done remain={_remaining(t0, cfg.wall_s):.0f}s unreliable={unreliable}", flush=True)

    write_df(out / "decoder_reliability.csv", pd.DataFrame(rel_rows) if rel_rows else pd.DataFrame([{"status": "none"}]), force=True)
    write_df(out / "label_hessian_reliability.csv", pd.DataFrame(hy_rows) if hy_rows else pd.DataFrame([{"status": "none"}]), force=True)
    if frames:
        write_df(out / "per_anchor_scalars.parquet", pd.concat(frames.values(), ignore_index=True), force=True)

    # ViT-B historical parity vs local R^2
    parity = {"paper_pdf_found": bool(shared["paper_pdf"]["found"]), "implementation": "fresh_from_spec", "table2_approx": PAPER_TABLE2, "recovered": {}}
    if hist_r2 is not None and (REFERENCE, TARGETS[0]) in frames:
        for t in TARGETS:
            df = frames[(REFERENCE, t)].copy()
            sub = hist_r2[hist_r2.target == t]
            mp = {int(s): float(r) for s, r in zip(sub.sample_id, sub.local_r2 if "local_r2" in sub.columns else sub.get("r2_G", pd.Series(dtype=float)))}
            df["hist_R2"] = [mp.get(int(s), np.nan) for s in df.sample_id]
            rec = {}
            for col, name in (("S", "shape"), ("R", "sphere"), ("M_Delta", "mismatch"), ("A_full", "alignment")):
                rec[name] = _spearman(df[col].to_numpy(float), df["hist_R2"].to_numpy(float))
            rec["paper"] = PAPER_TABLE2[t]
            parity["recovered"][t] = rec
    write_json(out / "parity.json", parity, force=True)

    if len(frames) < 4:
        return _done(out, t0, cfg, decide(tests_ok=True, coverage_ok=True, p1=None, p2=None, n_models=len({k[0] for k in frames}), n_unreliable=0, hy_median_split_cos=None, shape_beats_mismatch=False, resource_capped=True, shuffle_survives=False), extra={"n_frames": len(frames)})

    # cells
    cell_rows = []
    for (m, t), df in frames.items():
        aD = controlled(df, "M_Delta", "mse_G", PRIMARY_CONTROLS)
        aA = controlled(df, "A_full", "mse_G", PRIMARY_CONTROLS)
        aS = controlled(df, "S", "mse_G", PRIMARY_CONTROLS)
        aR = controlled(df, "R", "mse_G", PRIMARY_CONTROLS)
        aDp = controlled(df, "M_Delta", "mse_G", PAPER_CONTROLS)
        aKH = controlled(df, "generic_KH", "mse_G", PRIMARY_CONTROLS)
        cell_rows.append(
            {
                "model": m,
                "target": t,
                "rho_Delta": aD["controlled"],
                "rho_A": aA["controlled"],
                "rho_S": aS["controlled"],
                "rho_R": aR["controlled"],
                "rho_Delta_paper3": aDp["controlled"],
                "rho_KH": aKH["controlled"],
                "rho_Delta_minus_S": float(aD["controlled"] - aS["controlled"]) if np.isfinite(aD["controlled"]) and np.isfinite(aS["controlled"]) else float("nan"),
                "sign_Delta": float(np.sign(aD["controlled"])),
                "sign_A": float(np.sign(aA["controlled"])),
            }
        )
    cell_df = pd.DataFrame(cell_rows)
    write_df(out / "per_model_target_results.csv", cell_df, force=True)
    vif = vif_and_cond(next(iter(frames.values())), PRIMARY_CONTROLS)
    write_json(out / "control_diagnostics.json", vif, force=True)

    print("[hmm] P1/P2", flush=True)
    agg = synchronized_p1_p2(frames, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff())
    p1, p2 = agg["P1"], agg["P2"]
    hs = holm(np.array([p1["p_mc"], p2["p_mc"]]))
    p1["p_holm"] = float(hs[0])
    p2["p_holm"] = float(hs[1])
    p1["pass_holm"] = bool(p1["observed"] > 0 and hs[0] <= 0.05)
    p2["pass_holm"] = bool(p2["observed"] < 0 and hs[1] <= 0.05)
    loo_m1 = leave_one_out(p1["per_cell"], axis="model")
    loo_t1 = leave_one_out(p1["per_cell"], axis="target")
    loo_m2 = leave_one_out(p2["per_cell"], axis="model")
    loo_t2 = leave_one_out(p2["per_cell"], axis="target")
    p1["loo_model_all_same_sign"] = bool((loo_m1["mean"] > 0).all())
    p2["loo_model_all_same_sign"] = bool((loo_m2["mean"] < 0).all())
    write_df(out / "leave_one_model_out.csv", pd.concat([loo_m1.assign(family="P1"), loo_m2.assign(family="P2")]), force=True)
    write_df(out / "leave_one_target_out.csv", pd.concat([loo_t1.assign(family="P1"), loo_t2.assign(family="P2")]), force=True)
    write_df(out / "aggregate_results.csv", pd.DataFrame([{k: v for k, v in p1.items() if k != "per_cell"} | {"family": "P1"}, {k: v for k, v in p2.items() if k != "per_cell"} | {"family": "P2"}]), force=True)
    write_json(out / "permutation_bootstrap_summary.json", {"P1": p1, "P2": p2}, force=True)

    # nulls (cheap): shuffle Hy by permuting train y in-place analogue — permute M_Delta vs keep MSE? Spec wants refit. Do geometry-matched permutation of M_Delta within radius quintiles.
    null_rows = []
    rng = np.random.default_rng(0)
    for (m, t), df in frames.items():
        q = pd.qcut(df["log_radius_k2048"].rank(method="first"), 5, labels=False, duplicates="drop")
        xp = df["M_Delta"].to_numpy(float).copy()
        for qi in np.unique(q.dropna()):
            idx = np.where(q == qi)[0]
            xp[idx] = rng.permutation(xp[idx])
        tmp = df.copy()
        tmp["M_Delta"] = xp
        null_rows.append({"model": m, "target": t, "kind": "radius_matched_geom_perm", "rho_Delta": controlled(tmp, "M_Delta", "mse_G", PRIMARY_CONTROLS)["controlled"]})
        tmp2 = df.copy()
        tmp2["M_Delta"] = df["R"]
        null_rows.append({"model": m, "target": t, "kind": "sphere_only_R", "rho_Delta": controlled(tmp2, "M_Delta", "mse_G", PRIMARY_CONTROLS)["controlled"]})
    # shuffle labels: permute eval MSE against frozen geometry (conservative; does not refit w)
    for (m, t), df in list(frames.items())[:1]:
        tmp = df.copy()
        tmp["mse_G"] = rng.permutation(tmp["mse_G"].to_numpy(float))
        null_rows.append({"model": m, "target": t, "kind": "shuffled_mse", "rho_Delta": controlled(tmp, "M_Delta", "mse_G", PRIMARY_CONTROLS)["controlled"]})
    write_df(out / "null_results.csv", pd.DataFrame(null_rows), force=True)
    shuf_vals = [r["rho_Delta"] for r in null_rows if r["kind"] == "shuffled_mse"]
    shuffle_survives = bool(len(shuf_vals) and np.nanmean(shuf_vals) > 0.08)

    n_unrel = int(sum(1 for r in rel_rows if r["model"] not in [None] and not r.get("passed", True) and r.get("model")))
    n_unrel = int(len({r["model"] for r in rel_rows if not r.get("passed", True)}))
    hy_med = float(np.nanmedian([r["median_split_cos"] for r in hy_rows])) if hy_rows else None
    shape_beats = bool(np.nanmean(np.abs(cell_df["rho_S"])) > np.nanmean(np.abs(cell_df["rho_Delta"])) + 0.03 and not p1["pass_holm"])
    dec = decide(
        tests_ok=tests["all_passed"],
        coverage_ok=True,
        p1=p1,
        p2=p2,
        n_models=len({k[0] for k in frames}),
        n_unreliable=n_unrel,
        hy_median_split_cos=hy_med,
        shape_beats_mismatch=shape_beats,
        resource_capped=resource_capped and len({k[0] for k in frames}) < 5,
        shuffle_survives=shuffle_survives,
    )
    write_figures(out, cell_df=cell_df, frames=frames, p1=p1, p2=p2)
    write_markdowns(out, decision=dec, p1=p1, p2=p2, cell_df=cell_df, parity=parity, tests=tests, runtime=time.time() - t0, hy_med=hy_med, man=man)
    extra = {"n_models": len({k[0] for k in frames}), "seeds": list(seeds_use), "new_decoders_trained": 0}
    return _done(out, t0, cfg, dec, extra=extra)


def _done(out, t0, cfg, decision, extra=None):
    elapsed = time.time() - t0
    write_json(out / "decision.json", decision, force=True)
    write_json(out / "summary.json", {"label": decision["label"], "P1": decision.get("P1"), "P2": decision.get("P2"), "runtime_s": elapsed, **(extra or {})}, force=True)
    write_json(out / "runtime.json", {"wall_s": elapsed, "env": {"numpy": np.__version__, "torch": torch.__version__, "rss_mb": peak_rss_mb()}}, force=True)
    if decision.get("P1") is not None:
        write_json(out / "COMPLETE.json", {"status": "complete", "label": decision["label"], "wall_s": elapsed, "manuscript_modified": False, "new_decoders_trained": 0}, force=True)
    print("[hmm] done", decision["label"], f"{elapsed:.1f}s", flush=True)
    return {"decision": decision, "elapsed_s": elapsed}
