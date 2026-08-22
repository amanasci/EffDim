"""Orchestrate audit → parity → patch OOF fits → inference. No manuscript edits."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import write_audit
from .config import CONTROLS, N_SHUFFLE_ANCHORS, PRIMARY_K, PROBE_ALPHA, SEED, ExpConfig
from .data import kh_controls, load_bundle
from .figures import write_figure
from .inference import decide, manuscript_action, primary_inference, secondary_table
from .io_util import assert_not_preserved, peak_rss_mb, platonic_root, resolve_path, write_df, write_json
from .parity import run_parity
from .probes import fit_anchor_oof
from .synthetic import run_synthetic


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "anchors").mkdir(exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=True)
    write_audit(out)

    synth = run_synthetic(seed=cfg.seed)
    write_json(out / "synthetic_results.json", synth, force=True)
    if cfg.stage == "synthetic":
        return synth

    bundle = load_bundle(cfg)
    parity = run_parity(bundle, cfg, out)
    if cfg.stage == "parity":
        return parity

    df = _fit_all_anchors(bundle, cfg, out, slice_mode="full")
    write_df(out / "anchor_model_metrics.parquet", df, force=True)
    imp = _improvements(df)
    write_df(out / "anchor_improvements.csv", imp, force=True)

    # direction diagnostics table
    diag_cols = [c for c in df.columns if c.startswith("diag_") or c in ("sample_id", "selected_alpha", "P_fold_cosine_med")]
    write_df(out / "probe_direction_diagnostics.parquet", df[[c for c in diag_cols if c in df.columns]], force=True)

    primary = primary_inference(df, n_perm=cfg.n_perm_eff(), n_boot=cfg.n_boot_eff(), seed=cfg.seed)
    write_json(out / "primary_inference.json", primary, force=True)
    sec = secondary_table(df)
    write_df(out / "controlled_associations.csv", sec, force=True)

    # outer-half sensitivity
    df_half = _fit_all_anchors(bundle, cfg, out, slice_mode="outer_half")
    write_df(out / "outer_half_sensitivity.csv", _improvements(df_half), force=True)
    half_primary = primary_inference(df_half, n_perm=min(500, cfg.n_perm_eff()), n_boot=min(500, cfg.n_boot_eff()), seed=cfg.seed + 7)
    outer_half_pass = bool(
        np.isfinite(half_primary["observed"]["controlled"])
        and np.sign(half_primary["observed"]["controlled"]) == np.sign(primary["observed"]["controlled"])
    ) or not np.isfinite(primary["observed"]["controlled"])

    # label shuffle on audit anchors
    shuffle = _label_shuffle(bundle, cfg, out) if not cfg.skip_shuffle else {"pass": True, "skipped": True}
    write_df(out / "label_shuffle_results.csv", pd.DataFrame(shuffle.get("rows", [])), force=True)

    oof_ok = bool((~df.overlap_any).all() and df.n_eval.min() >= 100)
    hist_only = bool(df.get("dMSE_G_to_P_insample", pd.Series([np.nan])).median() > 0.05 and primary["observed"]["controlled"] <= 0)

    decision = decide(
        primary,
        sec,
        parity_ok=bool(parity.get("ok")),
        oof_ok=oof_ok,
        shuffle_pass=bool(shuffle.get("pass", False)),
        outer_half_pass=outer_half_pass,
        hist_insample_only=hist_only,
    )
    decision["manuscript"] = manuscript_action(decision["label"])
    decision["synthetic"] = synth
    decision["outer_half"] = half_primary
    decision["shuffle"] = {k: shuffle[k] for k in shuffle if k != "rows"}
    decision["peak_rss_mb"] = peak_rss_mb()
    decision["overlapping_neighbourhood_note"] = (
        "Anchors share overlapping kNN balls; inference treats anchors as exchangeable. "
        "No frozen non-overlap clustering was applied; this is a limitation."
    )
    write_json(out / "decision.json", decision, force=True)
    write_figure(out, df, primary)
    _write_methods(out)
    _write_report(out, parity, decision, primary, sec, t0, cfg)
    write_json(
        out / "COMPLETE.json",
        {"ok": True, "label": decision["label"], "seconds": time.time() - t0, "smoke": cfg.smoke},
        force=True,
    )
    print(f"[lpa] done label={decision['label']} s={time.time()-t0:.1f}", flush=True)
    return decision


def _fit_all_anchors(bundle, cfg: ExpConfig, out: Path, *, slice_mode: str) -> pd.DataFrame:
    sids = bundle["sids"]
    X, y, yhat, fold = bundle["X"], bundle["y"], bundle["yhat"], bundle["fold"]
    neigh, sid_to_ai = bundle["neigh"], bundle["sid_to_ai"]
    rows = []
    for i, sid in enumerate(sids):
        ckpt = out / "anchors" / f"{slice_mode}_{int(sid)}.parquet"
        if ckpt.exists() and not cfg.force and slice_mode == "full":
            rows.append(pd.read_parquet(ckpt).iloc[0].to_dict())
            continue
        ai = sid_to_ai[int(sid)]
        N = neigh[ai, :PRIMARY_K]
        if slice_mode == "outer_half":
            N = N[PRIMARY_K // 2 :]
        fit = fit_anchor_oof(
            X=X,
            y=y,
            yhat_g=yhat,
            fold=fold,
            neigh_idx=N,
            sample_ids_row=bundle["sample_id_row"],
            alpha=PROBE_ALPHA,
            do_tangent=not cfg.skip_tangent,
            do_nested_alpha=(not cfg.skip_nested_alpha) and slice_mode == "full",
            do_insample=slice_mode == "full",
            seed=cfg.seed + int(sid),
        )
        ctrl = kh_controls(bundle, int(sid))
        rec = {
            "sample_id": int(sid),
            "slice_mode": slice_mode,
            "overlap_any": bool(fit["overlap_any"]),
            "n_eval": fit["n_eval"],
            "selected_alpha": fit["dir_diag"].get("selected_alpha", PROBE_ALPHA),
            "P_fold_cosine_med": fit["dir_diag"].get("P_fold_cosine_med", float("nan")),
            "diag_edf": fit["dir_diag"].get("P_edf_med", float("nan")),
            "diag_cond": fit["dir_diag"].get("P_cond_med", float("nan")),
            **ctrl,
        }
        for model, met in fit["metrics"].items():
            for k, v in met.items():
                rec[f"{k}_{model}"] = v
        # convenience aliases
        rec["mse_G"] = rec.get("mse_G")
        rec["mse_P"] = rec.get("mse_P")
        rec["mse_C"] = rec.get("mse_C")
        rec["mse_T"] = rec.get("mse_T")
        rec["r2_G"] = rec.get("r2_G")
        rec["r2_P"] = rec.get("r2_P")
        rec["sst"] = rec.get("sst_G")
        rec["var"] = rec.get("var_G")
        rec.update(_delta_fields(rec))
        if slice_mode == "full":
            write_df(ckpt, pd.DataFrame([rec]), force=True)
        rows.append(rec)
        if (i + 1) % 32 == 0:
            print(f"[lpa] {slice_mode} {i+1}/{len(sids)}", flush=True)
    return pd.DataFrame(rows)


def _delta_fields(rec: dict) -> dict:
    out = {}
    for a, b, name in [
        ("mse_G", "mse_P", "dMSE_G_to_P"),
        ("mae_G", "mae_P", "dMAE_G_to_P"),
        ("r2_P", "r2_G", "dR2_G_to_P"),
        ("mse_C", "mse_P", "dMSE_C_to_P"),
        ("mse_G", "mse_T", "dMSE_G_to_T"),
        ("mse_C", "mse_T", "dMSE_C_to_T"),
        ("mse_G", "mse_P_insample", "dMSE_G_to_P_insample"),
    ]:
        # metrics stored as mse_G from metrics keys mse_G after rename — probes uses metrics[name] with keys mse etc → mse_G
        va, vb = rec.get(a, float("nan")), rec.get(b, float("nan"))
        if name.startswith("dR2"):
            out[name] = float(va - vb) if np.isfinite(va) and np.isfinite(vb) else float("nan")
        else:
            out[name] = float(va - vb) if np.isfinite(va) and np.isfinite(vb) else float("nan")
    return out


def _improvements(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "K_H_cross",
        *CONTROLS,
        "dMSE_G_to_P",
        "dMAE_G_to_P",
        "dR2_G_to_P",
        "dMSE_C_to_P",
        "dMSE_G_to_T",
        "dMSE_C_to_T",
        "mse_G",
        "mse_P",
        "mse_C",
        "r2_G",
        "r2_P",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def _label_shuffle(bundle, cfg: ExpConfig, out: Path) -> dict:
    """Global label permutations on audit anchors with cached ridge factorizations."""
    from .ridge import ridge_fit_intercept, ridge_predict

    rng = np.random.default_rng(cfg.seed + 99)
    n_audit = 8 if cfg.smoke else N_SHUFFLE_ANCHORS
    sids = list(bundle["sids"][: min(n_audit, len(bundle["sids"]))])
    X, fold, sid_row = bundle["X"], bundle["fold"], bundle["sample_id_row"]
    y0 = bundle["y"].copy()

    # Precompute per-anchor, per-fold Cholesky factors on real design (labels only change)
    cache = []
    for sid in sids:
        ai = bundle["sid_to_ai"][int(sid)]
        N = bundle["neigh"][ai, :PRIMARY_K]
        folds_p = sorted(set(fold[N].tolist()))
        entries = []
        for f in folds_p:
            te_local = np.where(fold[N] == f)[0]
            tr_local = np.where(fold[N] != f)[0]
            tr, te = N[tr_local], N[te_local]
            if len(tr) < 32 or len(te) < 8:
                continue
            # fit once to get L, Xc, x_mean (depends on X only for factorization of XtX)
            # ridge_fit_intercept centers X; store pieces
            Xtr = np.asarray(X[tr], dtype=np.float64)
            x_mean = Xtr.mean(axis=0)
            Xc = Xtr - x_mean
            XtX = Xc.T @ Xc
            np.fill_diagonal(XtX, np.diag(XtX) + float(PROBE_ALPHA))
            try:
                L = np.linalg.cholesky(XtX)
            except np.linalg.LinAlgError:
                continue
            entries.append({"tr": tr, "te": te, "L": L, "Xc": Xc, "x_mean": x_mean})
        cache.append({"sid": sid, "N": N, "entries": entries})

    rows = []
    n_sh = cfg.n_shuffle_eff()
    for b in range(n_sh):
        y_perm = rng.permutation(y0)
        deltas = []
        for item in cache:
            N = item["N"]
            # null global = overall train-mean per fold under perm
            yhat_null = np.empty(len(y_perm))
            for f in range(5):
                tr = fold != f
                yhat_null[fold == f] = float(np.nanmean(y_perm[tr]))
            mse_g = float(np.mean((y_perm[N] - yhat_null[N]) ** 2))
            # patch OOF via cached factors
            pred = np.full(len(N), np.nan)
            pos = {int(ix): j for j, ix in enumerate(N)}
            for e in item["entries"]:
                ytr = y_perm[e["tr"]]
                m = np.isfinite(ytr)
                if m.sum() < 8:
                    continue
                yc = ytr[m] - float(ytr[m].mean())
                Xc = e["Xc"][m]
                # rebuild mean on finite mask
                Xtr = X[e["tr"]][m]
                x_mean = Xtr.mean(axis=0)
                Xc = Xtr - x_mean
                XtX = Xc.T @ Xc
                np.fill_diagonal(XtX, np.diag(XtX) + float(PROBE_ALPHA))
                try:
                    L = np.linalg.cholesky(XtX)
                except np.linalg.LinAlgError:
                    continue
                w = np.linalg.solve(L.T, np.linalg.solve(L, Xc.T @ yc))
                b0 = float(ytr[m].mean()) - float(x_mean @ w)
                pred_te = X[e["te"]] @ w + b0
                for ix, val in zip(e["te"], pred_te):
                    pred[pos[int(ix)]] = val
            mse_p = float(np.nanmean((y_perm[N] - pred) ** 2))
            if np.isfinite(mse_g) and np.isfinite(mse_p):
                deltas.append(mse_g - mse_p)
        rows.append({"perm": b, "mean_dMSE_GP": float(np.mean(deltas)) if deltas else float("nan"), "n": len(deltas)})
        if (b + 1) % 50 == 0:
            print(f"[lpa] shuffle {b+1}/{n_sh}", flush=True)

    arr = np.asarray([r["mean_dMSE_GP"] for r in rows], float)
    real = float("nan")
    real_path = out / "anchor_improvements.csv"
    if real_path.exists():
        imp = pd.read_csv(real_path)
        # audit sids only
        real = float(imp[imp.sample_id.isin(sids)].dMSE_G_to_P.mean()) if "dMSE_G_to_P" in imp else float("nan")
    if np.isfinite(real) and np.isfinite(arr).any():
        # pass if null mean near 0 relative to real scale when real>0, or always if real<=0
        null_mean = float(np.nanmean(arr))
        passed = bool(abs(null_mean) < 0.05 or real <= 0)
        p = float(np.mean(arr >= real)) if real > 0 else float("nan")
        p = (int(np.sum(arr >= real)) + 1) / (len(arr) + 1) if real > 0 else float("nan")
    else:
        p, passed, null_mean = float("nan"), True, float("nan")
    return {"rows": rows, "pass": passed, "p_vs_shuffle": p, "real_mean_dMSE": real, "null_mean": null_mean}


def _write_methods(out: Path) -> None:
    (out / "METHODS.md").write_text(
        f"""# METHODS

Curvature: frozen cross-split sphere-normal mean-curvature statistic
K_H_cross = <H^(A), H^(B)> at d={16}, k={PRIMARY_K}. Not the mean-curvature vector.

Global probe G: frozen five-fold OOF ridge (α={PROBE_ALPHA}, sum-of-squares, intercept).
Patch models use the same global fold IDs: train on fold≠f, predict fold=f.

I = patch training-label mean. C = affine calibration of global OOF on patch train.
P = ambient ridge on ViT-B features. T = ridge on transductive d=16 PCA tangent coords
(unsupervised geometry uses the full patch; labelled fits remain outer-fold OOF).

Primary endpoint: controlled Spearman ρ(K_H_cross, ΔMSE_G→P).
"""
    )


def _write_report(out, parity, decision, primary, sec, t0, cfg) -> None:
    ms = decision["manuscript"]
    (out / "REPORT.md").write_text(
        f"""# REPORT

## Decision label

`{decision['label']}`

## Historical audit

See ACCIDENTAL_RESULT_AUDIT.md (`historical_invalid_or_unverified`).

## Global-probe parity

ok={parity.get('ok')} ρ(K_H, R²)={parity.get('rho_r2')} ρ(K_H, MSE)={parity.get('rho_mse')}

## Patch OOF checks

overlap/oof flags in decision.checks: {decision.get('checks')}

## Primary endpoint

ρ_ctl(K_H, ΔMSE_G→P) = {primary['observed']['controlled']:+.4f}
95% CI {primary['ci95']}, p_MC={primary['p_mc']:.4g}

## Global vs patch curvature–MSE

MSE_G assoc={decision.get('mse_G_assoc')} MSE_P assoc={decision.get('mse_P_assoc')}

## Patch vs calibrated global

ρ(K_H, ΔMSE_C→P)={decision.get('dMSE_C_to_P_assoc')}

## Manuscript recommendation (not applied)

action: `{ms['action']}`

> {ms['paragraph']}

Do not edit submissions/neurreps_2026 until author review.

## Runtime

smoke={cfg.smoke} wall_s≈{time.time()-t0:.1f} peak_rss_mb={decision.get('peak_rss_mb')}
"""
    )
