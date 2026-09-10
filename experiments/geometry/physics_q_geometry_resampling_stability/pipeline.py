"""Bounded ViT-B Q geometry-resampling audit. No decoder/probe/label refits."""

from __future__ import annotations

import json
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[3]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from .associations import (
    adapt_pass,
    anchor_only_bootstrap,
    assoc_field,
    combined_interval,
    field_reliability,
    global_pass,
    spearman_safe,
    summarize_replicates,
)
from .audit import association_parity, load_frozen, q_refit_parity, reuse_manifest
from .config import (
    DISK_CAP_BYTES,
    ExpConfig,
    K,
    MAX_REPS,
    MIN_VALID_REPS,
    OUT_REL,
    PILOT_REPS,
    PROJECTION_BUDGET_S,
    RESERVE_WRITE_S,
    TARGET_REPS,
)
from .decision import decide
from .figures import write_figures
from .reports import write_reports
from .schemes import run_conditional_replicate, run_object_support_replicate, start_fit_pool, stop_fit_pool
from .tests_unit import run_unit_tests


def _remaining(t0, wall):
    return wall - RESERVE_WRITE_S - (time.time() - t0)


def _dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _disk(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    tree = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                tree += f.stat().st_size
    return {
        "filesystem_free_bytes": int(usage.free),
        "filesystem_total_bytes": int(usage.total),
        "output_tree_bytes": int(tree),
        "output_tree_mb": float(tree) / (1024**2),
        "cap_bytes": DISK_CAP_BYTES,
        "under_cap": tree < DISK_CAP_BYTES,
    }


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    out = Path(cfg.output_dir)
    if not out.is_absolute():
        out = (_REPO / cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = ["disk"]
    skipped = []
    disk0 = _disk(out)
    _dump(out / "disk_usage.json", disk0)
    if disk0["filesystem_free_bytes"] < 200 * 1024**2:
        _dump(out / "COMPLETE.json", {"status": "blocked", "reason": "low disk", "disk": disk0})
        return {"blocked": True, "reason": "disk"}

    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    print("[qrs] loading frozen artifacts", flush=True)
    ctx = load_frozen()
    man = reuse_manifest(ctx)
    _dump(out / "reuse_manifest.json", man)
    stages.append("reuse")

    par = association_parity(ctx["df"])
    print("[qrs] association parity", par.get("ok"), flush=True)
    if _remaining(t0, cfg.wall_s) > 60:
        try:
            par["q_refit"] = q_refit_parity(ctx, n_check=cfg.n_parity_refit)
        except Exception as exc:  # noqa: BLE001
            par["q_refit"] = {"ok": False, "error": str(exc)}
            print(traceback.format_exc(), flush=True)
    _dump(out / "parity.json", par)
    stages.append("parity")

    tests = run_unit_tests(parity=par, ctx=ctx)
    _dump(out / "unit_test_results.json", tests)
    if not par["ok"]:
        runtime = {"runtime_s": time.time() - t0, "blocked": "parity", "stages": stages}
        _dump(out / "runtime.json", runtime)
        _dump(out / "decision.json", {"summary_label": "q_geometry_resampling_unresolved", "reason": "parity"})
        _dump(out / "summary.json", {"blocked": True, "reason": "parity"})
        _dump(out / "COMPLETE.json", {"status": "blocked", "reason": "parity", "parity": par})
        write_reports(out, decision={"summary_label": "q_geometry_resampling_unresolved"}, parity=par, runtime=runtime, summary={})
        return {"blocked": True}

    if not tests["all_passed"]:
        print("[qrs] unit tests failed; continuing only if parity held", flush=True)

    import atexit

    try:
        start_fit_pool(cfg.n_workers, ctx["X"], ctx["neigh"], ctx["frames"], ctx["sids"])
        atexit.register(stop_fit_pool)
        print(f"[qrs] fit pool workers={cfg.n_workers}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qrs] fit pool failed ({exc}); falling back to threads", flush=True)
    df = ctx["df"]
    kh0 = df.K_H_cross.to_numpy(float)
    orig = assoc_field(kh0, df)
    signs = {"r2_G": -1, "mse_G": 1, "r2_P": 1, "mse_P": -1, "delta_adapt": 1}

    cond_rows: list[dict] = []
    obj_rows: list[dict] = []
    times_a, times_b = [], []
    manifest = []

    def persist():
        if cond_rows:
            pd.DataFrame(cond_rows).to_parquet(out / "conditional_support_fields.parquet", index=False)
        if obj_rows:
            pd.DataFrame(obj_rows).to_parquet(out / "object_support_fields.parquet", index=False)
        _dump(out / "replicate_manifest.json", {"rows": manifest})
        _dump(out / "disk_usage.json", _disk(out))
        _dump(out / "runtime.json", {"runtime_s": time.time() - t0, "stages": stages, "skipped": skipped, "rss_mb": _rss_mb()})

    print("[qrs] pilot 4+4", flush=True)
    for b in range(PILOT_REPS):
        if _remaining(t0, cfg.wall_s) < 90:
            skipped.append("pilot:wall")
            break
        ta0 = time.time()
        cond_rows.extend(run_conditional_replicate(X=ctx["X"], neigh=ctx["neigh"], frames=ctx["frames"], sids=ctx["sids"], replicate=b, device=cfg.device, n_workers=cfg.n_workers))
        times_a.append(time.time() - ta0)
        manifest.append({"scheme": "A", "replicate": b, "t_s": times_a[-1], "phase": "pilot"})
        print(f"[qrs] A pilot {b} t={times_a[-1]:.1f}s rss={_rss_mb():.0f}MB", flush=True)
        persist()
    for b in range(PILOT_REPS):
        if _remaining(t0, cfg.wall_s) < 90:
            skipped.append("pilotB:wall")
            break
        tb0 = time.time()
        rows, meta = run_object_support_replicate(
            X=ctx["X"], neigh=ctx["neigh"], frames=ctx["frames"], sids=ctx["sids"], query_rows=ctx["query_rows"], replicate=b, device=cfg.device, n_workers=cfg.n_workers
        )
        obj_rows.extend(rows)
        times_b.append(time.time() - tb0)
        manifest.append({"scheme": "B", "replicate": b, "t_s": times_b[-1], "phase": "pilot", **meta})
        print(f"[qrs] B pilot {b} t={times_b[-1]:.1f}s", flush=True)
        persist()
    stages.append("pilot")

    ta = float(np.median(times_a)) if times_a else 999.0
    tb = float(np.median(times_b)) if times_b else 999.0
    elapsed = time.time() - t0

    def proj(n):
        extra = max(0, n - PILOT_REPS)
        return elapsed + extra * (ta + tb)

    if cfg.force_reps:
        n_rep = int(cfg.force_reps)
    elif proj(MAX_REPS) < PROJECTION_BUDGET_S:
        n_rep = MAX_REPS
    elif proj(TARGET_REPS) < PROJECTION_BUDGET_S:
        n_rep = TARGET_REPS
    else:
        extra_budget = max(0.0, PROJECTION_BUDGET_S - elapsed)
        n_rep = int(PILOT_REPS + extra_budget // max(ta + tb, 1e-6))
        n_rep = max(MIN_VALID_REPS if extra_budget > (MIN_VALID_REPS - PILOT_REPS) * (ta + tb) else len(times_a), min(n_rep, TARGET_REPS))
    n_rep = max(len(times_a), n_rep)
    print(f"[qrs] selected n_rep={n_rep} ta={ta:.1f}s tb={tb:.1f}s proj64={proj(64):.0f}s", flush=True)

    for b in range(PILOT_REPS, n_rep):
        if _remaining(t0, cfg.wall_s) < 90:
            skipped.append(f"A{b}:wall")
            break
        ta0 = time.time()
        cond_rows.extend(run_conditional_replicate(X=ctx["X"], neigh=ctx["neigh"], frames=ctx["frames"], sids=ctx["sids"], replicate=b, device=cfg.device, n_workers=cfg.n_workers))
        times_a.append(time.time() - ta0)
        manifest.append({"scheme": "A", "replicate": b, "t_s": times_a[-1], "phase": "full"})
        print(f"[qrs] A {b} t={times_a[-1]:.1f}s", flush=True)
        persist()
    stages.append("scheme_A")
    for b in range(PILOT_REPS, n_rep):
        if _remaining(t0, cfg.wall_s) < 90:
            skipped.append(f"B{b}:wall")
            break
        tb0 = time.time()
        rows, meta = run_object_support_replicate(
            X=ctx["X"], neigh=ctx["neigh"], frames=ctx["frames"], sids=ctx["sids"], query_rows=ctx["query_rows"], replicate=b, device=cfg.device, n_workers=cfg.n_workers
        )
        obj_rows.extend(rows)
        times_b.append(time.time() - tb0)
        manifest.append({"scheme": "B", "replicate": b, "t_s": times_b[-1], "phase": "full", **meta})
        print(f"[qrs] B {b} t={times_b[-1]:.1f}s", flush=True)
        persist()
    stages.append("scheme_B")

    cond_df = pd.DataFrame(cond_rows)
    obj_df = pd.DataFrame(obj_rows)
    cond_df.to_parquet(out / "conditional_support_fields.parquet", index=False)
    obj_df.to_parquet(out / "object_support_fields.parquet", index=False)

    def wide(rep_df):
        w = rep_df.pivot_table(index="sample_id", columns="replicate", values="K_H_cross", aggfunc="first")
        w = w.reindex(df.sample_id.astype(int))
        w.columns = [f"r{int(c)}" for c in w.columns]
        return w

    rel_rows = []
    field_tbl = []
    overlap_rows = []

    def _rel_block(name, rdf):
        if rdf.empty:
            return {}, 0
        w = wide(rdf)
        extra = {"log_knn_radius": df.log_knn_radius.to_numpy(float)}
        if name == "object_support" and "radius" in rdf.columns:
            # mean replicate radius per anchor as a control diagnostic
            rad = rdf.groupby("sample_id")["radius"].mean().reindex(df.sample_id.astype(int)).to_numpy(float)
            extra["replicate_radius_mean"] = rad
        rel = field_reliability(w, kh0, extra)
        rel["scheme"] = name
        field_tbl.append(rel)
        for j, c in enumerate(w.columns):
            sub = rdf[rdf.replicate == int(c[1:])]
            rel_rows.append(
                {
                    "scheme": name,
                    "replicate": int(c[1:]),
                    "r_b0": rel["r_b0"][j],
                    "support_overlap": float(sub["support_overlap"].median()) if "support_overlap" in sub.columns else 1.0,
                    "jaccard": float(sub["jaccard"].median()) if "jaccard" in sub.columns else 1.0,
                    "n_ok": int(sub["ok"].sum()) if "ok" in sub.columns else len(sub),
                }
            )
        if "jaccard" in rdf.columns:
            g = rdf.groupby("replicate").agg(overlap=("support_overlap", "median"), jaccard=("jaccard", "median"), radius=("radius", "median"), retained_n=("retained_n", "first"), k_prime=("k_prime", "first"))
            for b, r in g.iterrows():
                overlap_rows.append({"scheme": name, "replicate": int(b), **r.to_dict()})
        return rel, int(w.shape[1])

    rel_a, n_a = _rel_block("conditional_support", cond_df)
    rel_b, n_b = _rel_block("object_support", obj_df)

    assoc_rows = []
    sens_rows = []
    kdir_rows = []

    def _assoc_scheme(name, rdf):
        stab = {}
        if rdf.empty:
            return stab
        for b, g in rdf.groupby("replicate"):
            g2 = g.set_index("sample_id").loc[df.sample_id.astype(int)]
            rec = assoc_field(g2.K_H_cross.to_numpy(float), df)
            rec_s = assoc_field(g2.K_H_cross.to_numpy(float), df, extra_z=g2.radius.to_numpy(float) if "radius" in g2.columns else None)
            rec_d = assoc_field(g2.K_dir_cross.to_numpy(float), df) if "K_dir_cross" in g2.columns else {}
            for oc, v in rec.items():
                assoc_rows.append({"scheme": name, "replicate": int(b), "outcome": oc, "controlled": v["controlled"], "raw": v["raw"], "n": v["n"], "estimator": "K_H"})
            for oc, v in rec_s.items():
                sens_rows.append({"scheme": name, "replicate": int(b), "outcome": oc, "controlled": v["controlled"], "analysis": "replicate_radius_sensitivity"})
            for oc, v in rec_d.items():
                kdir_rows.append({"scheme": name, "replicate": int(b), "outcome": oc, "controlled": v["controlled"], "estimator": "K_dir"})
        adf = pd.DataFrame(assoc_rows)
        adf = adf[adf.scheme == name]
        for oc, sgn in signs.items():
            vals = adf[adf.outcome == oc]["controlled"].to_numpy(float)
            stab[oc] = summarize_replicates(vals, orig[oc]["controlled"], sgn)
            stab[oc]["n_valid_anchors"] = int(len(df))
        return stab

    stab_a = _assoc_scheme("conditional_support", cond_df)
    stab_b = _assoc_scheme("object_support", obj_df)

    unexpected = (not global_pass(stab_a) and not adapt_pass(stab_a)) and (global_pass(stab_b) and adapt_pass(stab_b)) if stab_a and stab_b else False
    decision = decide(stab_a, stab_b, n_a, n_b, parity_ok=True, unexpected_ab=unexpected)

    boot0 = anchor_only_bootstrap(kh0, df)
    comb_a = combined_interval(cond_df, df) if not cond_df.empty else {}
    comb_b = combined_interval(obj_df, df) if not obj_df.empty else {}

    comb_rows = []
    for oc in signs:
        for label, blk in (
            ("anchor_only", boot0),
            ("conditional_geometry", {k: {"q025": stab_a.get(k, {}).get("q025"), "q975": stab_a.get(k, {}).get("q975")} for k in signs}),
            ("object_support_geometry", {k: {"q025": stab_b.get(k, {}).get("q025"), "q975": stab_b.get(k, {}).get("q975")} for k in signs}),
            ("combined_A", comb_a),
            ("combined_B", comb_b),
        ):
            rec = blk.get(oc, {})
            comb_rows.append(
                {
                    "interval": label,
                    "outcome": oc,
                    "original": orig[oc]["controlled"],
                    "q025": rec.get("q025", rec.get("ci95", [None, None])[0] if isinstance(rec.get("ci95"), list) else None),
                    "q975": rec.get("q975", rec.get("ci95", [None, None])[1] if isinstance(rec.get("ci95"), list) else None),
                    "label": rec.get("label", label),
                }
            )

    pd.DataFrame([{k: v for k, v in r.items() if k != "r_b0"} for r in field_tbl]).to_csv(out / "field_reliability.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(out / "support_overlap.csv", index=False)
    pd.DataFrame(assoc_rows).to_csv(out / "association_replicates.csv", index=False)
    stab_flat = []
    for scheme, st in (("conditional_support", stab_a), ("object_support", stab_b)):
        for oc, rec in st.items():
            stab_flat.append({"scheme": scheme, "outcome": oc, **rec})
    pd.DataFrame(stab_flat).to_csv(out / "association_stability.csv", index=False)
    pd.DataFrame(comb_rows).to_csv(out / "combined_uncertainty.csv", index=False)
    pd.DataFrame(kdir_rows).to_csv(out / "kdir_secondary.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(out / "sensitivity_controls.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "reliability_pairs.csv", index=False)

    figs = write_figures(out, pd.DataFrame(assoc_rows), pd.DataFrame(rel_rows), pd.DataFrame(comb_rows))

    runtime = {
        "runtime_s": time.time() - t0,
        "wall_s": cfg.wall_s,
        "pilot_n": PILOT_REPS,
        "selected_n_per_scheme": n_rep,
        "completed_A": n_a,
        "completed_B": n_b,
        "median_t_A_s": ta,
        "median_t_B_s": tb,
        "proj_32_s": proj(32),
        "proj_64_s": proj(64),
        "projection_rule": "64 if total<40min else 32 else max fitting under 40min",
        "rss_mb": _rss_mb(),
        "stages": stages,
        "skipped": skipped,
        "n_ae": 0,
        "figures": figs,
    }
    _dump(out / "runtime.json", runtime)
    _dump(out / "disk_usage.json", _disk(out))
    _dump(out / "decision.json", decision)
    headline = {
        "parity_ok": True,
        "decision": decision["summary_label"],
        "n_A": n_a,
        "n_B": n_b,
        "rel_A": {k: rel_a.get(k) for k in ("r_b0_median", "pairwise_median", "rank_icc") if rel_a},
        "rel_B": {k: rel_b.get(k) for k in ("r_b0_median", "pairwise_median", "rank_icc") if rel_b},
        "stab_A": stab_a,
        "stab_B": stab_b,
        "orig": {k: orig[k]["controlled"] for k in orig},
        "headline_md": f"label={decision['summary_label']} nA={n_a} nB={n_b} t={runtime['runtime_s']:.1f}s",
    }
    _dump(out / "summary.json", headline)
    write_reports(out, decision=decision, parity=par, runtime=runtime, summary=headline)
    status = "complete" if n_a >= MIN_VALID_REPS and n_b >= MIN_VALID_REPS else "complete_with_resource_cap"
    if n_a < MIN_VALID_REPS or n_b < MIN_VALID_REPS:
        status = "complete_with_resource_cap"
    _dump(
        out / "COMPLETE.json",
        {
            "status": status,
            "runtime_s": runtime["runtime_s"],
            "decision": decision["summary_label"],
            "n_A": n_a,
            "n_B": n_b,
            "tests": tests["all_passed"],
            "parity_ok": True,
            "skipped": skipped,
        },
    )
    print(f"[qrs] done {decision['summary_label']} t={runtime['runtime_s']:.1f}s nA={n_a} nB={n_b}", flush=True)
    stop_fit_pool()
    return decision
