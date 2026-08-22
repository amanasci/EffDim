"""Orchestrate final LPA audit."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y

from .alignment import build_alignment_table
from .config import MODEL, PROBE_ALPHA, SOURCE_ALIGN, SOURCE_LPA, TARGET, AuditConfig
from .controls import alignment_models, interaction_sensitivity
from .decision import decide, manuscript_action
from .figures import write_figure
from .io_util import assert_not_preserved, platonic_root, resolve_path, write_df, write_json
from .paired import run_all_paired
from .parity import load_lpa_tables, run_parity
from .pathway import pathway_table
from .shuffle import audit_subset, run_shuffle


def run(cfg: AuditConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=cfg.force)

    parity = run_parity(root, cfg, out)
    imp, met = load_lpa_tables(root)
    df = imp.merge(
        met[
            [
                "sample_id",
                "mae_G",
                "mae_P",
                "mse_T",
                "mse_C",
                "r2_G",
                "r2_P",
                "mse_P_insample",
                "dMSE_G_to_P_insample",
            ]
        ],
        on="sample_id",
        how="left",
        suffixes=("", "_m"),
    )

    # primary from LPA
    lpa_out = resolve_path(root, SOURCE_LPA)
    primary = json.loads((lpa_out / "primary_inference.json").read_text())

    paired = run_all_paired(df, n_boot=cfg.n_boot_eff(), seed=cfg.seed)
    write_df(out / "paired_correlation_contrasts.csv", paired, force=cfg.force)
    delta_mse_row = paired[paired.name == "delta_rho_MSE_GP"].iloc[0].to_dict() if len(paired) else {}

    # alignment
    mm = resolve_path(root, "outputs/geometry/physics_multimodel_graph_prior_quadratic")
    X = load_model_X(mm, MODEL)
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax2048.npz"))
    neigh = np.asarray(pack["neigh"], dtype=int)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_to_ai = {int(s): i for i, s in enumerate(anchors["anchors_sample_id"])}
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    y = folds["y_mag_r_desi"].to_numpy(float)
    fold = folds["fold"].to_numpy(int)
    z = np.load(resolve_path(root, SOURCE_ALIGN) / "global_probe_weights.npz")
    w_pool = np.asarray(z[f"w_{TARGET}"], dtype=float)
    gw = pd.read_parquet(mm / "global_probe_weights.parquet")
    align = build_alignment_table(
        root,
        sids=sorted(imp.sample_id.astype(int).tolist()),
        sid_to_ai=sid_to_ai,
        X=X,
        y=y,
        fold=fold,
        neigh=neigh,
        gw=gw,
        w_pooled=w_pool,
        alpha=PROBE_ALPHA,
    )
    write_df(out / "current_anchor_alignment.parquet", align, force=cfg.force)
    _write_alignment_methods(out)

    merged = df.merge(align, on="sample_id", how="inner")
    ctrl_tab = alignment_models(merged)
    write_df(out / "alignment_control_table.csv", ctrl_tab, force=cfg.force)
    interact = interaction_sensitivity(merged)

    pathway = pathway_table(df, align)
    write_df(out / "pathway_diagnostics.csv", pathway, force=cfg.force)

    pt = {r["test"]: float(r["rho"]) for _, r in pathway.iterrows()}
    pathway_reliable = bool(align.direction_reliable.fillna(False).sum() >= 128)
    pathway_supports = bool(
        pt.get("KH_predicts_DPG_given_align", float("nan")) > 0.05
        and pt.get("DPG_predicts_dMSE_given_KH_align", float("nan")) > 0.05
    )
    pathway_ok = bool(pt.get("DPG_predicts_dMSE_given_KH_align", 0.0) > 0.05)

    # shuffle
    shuffle = {"pass": False, "skipped": True}
    if not cfg.skip_shuffle:
        imp_idx = imp.set_index("sample_id")
        audit_sids = audit_subset(sorted(imp.sample_id.astype(int).tolist()), cfg.n_shuffle_anc(), cfg.seed)
        shuffle = run_shuffle(
            X=X,
            y=y,
            fold=fold,
            neigh=neigh,
            sid_to_ai=sid_to_ai,
            audit_sids=audit_sids,
            kh=imp_idx.K_H_cross,
            log_radius=imp_idx.log_knn_radius,
            eval_count=imp_idx.local_evaluation_count,
            n_perm=cfg.n_shuffle_eff(),
            seed=cfg.seed + 99,
        )
        write_df(out / "label_shuffle_results.csv", pd.DataFrame(shuffle["rows"]), force=cfg.force)
        write_json(out / "label_shuffle_summary.json", {k: shuffle[k] for k in shuffle if k != "rows"}, force=cfg.force)

    beats_c = False
    if "dMSE_C_to_P" in df.columns:
        beats_c = float(associate(df.K_H_cross, df.dMSE_C_to_P, control_matrix(df))["controlled"]) > 0
    tangent_ok = "dMSE_G_to_T" in df.columns and float(associate(df.K_H_cross, df.dMSE_G_to_T, control_matrix(df))["controlled"]) > 0
    insample = bool(df.dMSE_G_to_P_insample.median() > df.dMSE_G_to_P.median() + 0.02) if "dMSE_G_to_P_insample" in df.columns else False

    decision = decide(
        parity_ok=bool(parity.get("ok")),
        primary_rho=float(primary["observed"]["controlled"]),
        primary_p=float(primary["p_mc"]),
        primary_ci_excludes=bool(primary.get("ci_excludes_zero")),
        mean_dm=float(imp.dMSE_G_to_P.mean()),
        delta_rho_mse=delta_mse_row,
        ctrl_models=ctrl_tab,
        beats_c=beats_c,
        tangent_ok=tangent_ok,
        pathway_supports=pathway_supports,
        pathway_reliable=pathway_reliable,
        shuffle=shuffle,
        insample_artifact=insample,
    )
    decision["interaction"] = interact
    decision["shuffle"] = {k: shuffle[k] for k in shuffle if k != "rows"}
    decision["pathway"] = pt
    write_json(out / "decision.json", _json_safe(decision), force=cfg.force)
    write_df(out / "interaction_sensitivity.csv", pd.DataFrame([interact]), force=cfg.force)

    write_figure(out, df, paired, primary, shuffle)
    _write_audit_report(out, parity, primary, paired, ctrl_tab, decision, shuffle, t0, cfg)
    _write_manuscript_rec(out, decision)

    if decision["manuscript"]["action"] == "include_as_main_result":
        _maybe_copy_manuscript(root, cfg, decision)

    complete_ok = bool(not cfg.smoke and not cfg.skip_shuffle and parity.get("ok"))
    if complete_ok:
        import resource

        write_json(
            out / "COMPLETE.json",
            {
                "ok": True,
                "label": decision["label"],
                "seconds": time.time() - t0,
                "smoke": cfg.smoke,
                "n_boot": cfg.n_boot_eff(),
                "n_shuffle": cfg.n_shuffle_eff(),
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "manuscript_revision_created": (root / "submissions" / "neurreps_2026_lpa_revision").exists(),
            },
            force=cfg.force,
        )
    print(f"[lpa-audit] done label={decision['label']} s={time.time()-t0:.1f}", flush=True)
    return decision


def _write_alignment_methods(out: Path) -> None:
    (out / "ALIGNMENT_METHODS.md").write_text(
        """# ALIGNMENT_METHODS

## Global (A_H^G, A_B^G)

- Geometry packs: `physics_curvature_probe_multitarget/geometry_cache/k2048_aiXXXX.npz`
  (`T`, `x0u`, `UB`, `UNPCA`).
- Mean-curvature vector H: `physics_nested_dimension_curvature/H_vectors/{sample_id}.npz` field `H16`.
- Global probe weight: pooled `w_mag_r_desi` from `physics_global_probe_curvature_alignment/global_probe_weights.npz`.
- A_B^G: `projection_energies(w,T,x0u,UB,UN)["A_B_normal"]` (`global_probe_curvature_alignment.py`).
- A_H^G: `a_h_from_w_H(w,T,x0u,H)` (`global_probe_curvature_magnitude.py`).

## Patch (A_H^P, A_B^P)

- Patch weights reconstructed by strict global-fold OOF ambient ridge (α=100, no scaler), one w per outer fold.
- Fold-weighted mean of A_B^P and A_H^P across patch folds.
- Direction reliability: median pairwise cosine of fold weights ≥ 0.85.

## Pathway

- D_PG = arccos cosine between pooled global w and mean patch w (descriptive only).
- Not used as ordinary confounders in the primary adjustment set.
"""
    )


def _write_audit_report(out, parity, primary, paired, ctrl, decision, shuffle, t0, cfg) -> None:
    dr = paired[paired.name == "delta_rho_MSE_GP"].iloc[0] if len(paired) else None
    (out / "AUDIT_REPORT.md").write_text(
        f"""# AUDIT REPORT

## Label

`{decision['label']}`

## Parity

ok={parity.get('ok')}; mean ΔMSE_G→P={decision.get('mean_dMSE_GP'):.4f}

## Primary ρ(K_H, ΔMSE_G→P)

{primary['observed']['controlled']:+.4f} CI {primary['ci95']} p_MC={primary['p_mc']:.4g}

## Paired Δρ(MSE_G - MSE_P)

{dr.to_dict() if dr is not None else 'n/a'}

## Alignment controls

{ctrl.to_string(index=False) if len(ctrl) else 'n/a'}

## Shuffle

{decision.get('shuffle')}

## Manuscript

{decision['manuscript']['action']}

Runtime {time.time()-t0:.1f}s smoke={cfg.smoke}
"""
    )


def _write_manuscript_rec(out, decision) -> None:
    ms = decision["manuscript"]
    (out / "MANUSCRIPT_RECOMMENDATION.md").write_text(
        f"""# MANUSCRIPT RECOMMENDATION

action: `{ms['action']}`

> {ms['paragraph']}

Post-hoc adaptation analysis. Negative mean patch advantage must remain explicit.
"""
    )


def _json_safe(obj):
    import json

    return json.loads(json.dumps(obj, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))


def _maybe_copy_manuscript(root: Path, cfg: AuditConfig, decision: dict) -> None:
    src = root / "submissions" / "neurreps_2026"
    dst = root / "submissions" / "neurreps_2026_lpa_revision"
    if not dst.exists():
        shutil.copytree(src, dst)
    para = decision["manuscript"]["paragraph"]
    main = dst / "main.tex"
    text = main.read_text()
    if "local probe adaptation" not in text.lower():
        insert = (
            "\n\\paragraph{Post-hoc local probe adaptation.}\n"
            + para
            + "\n"
        )
        text = text.replace("\\paragraph{Scale.}", insert + "\\paragraph{Scale.}")
    if "draftwatermark" in text:
        import re

        text = re.sub(r"% Watermark:.*?\n", "", text)
        text = re.sub(r"\\usepackage\[.*?\]\{draftwatermark\}\n", "", text)
        text = re.sub(r"\\SetWatermarkText\{.*?\}\n", "", text)
        text = re.sub(r"\\SetWatermarkScale\{.*?\}\n", "", text)
        text = re.sub(r"\\SetWatermarkAngle\{.*?\}\n", "", text)
    main.write_text(text)
    app = dst / "appendix.tex"
    if app.exists() and "Local probe adaptation audit" not in app.read_text():
        app.write_text(
            app.read_text()
            + "\n\\section{Local probe adaptation audit (post hoc)}\n"
            + "See \\texttt{outputs/geometry/physics\\_local\\_probe\\_adaptation\\_audit/} for parity, "
            + "alignment controls ($A_H^G$, $A_B^G$), paired $\\Delta\\rho$ contrasts, pathway diagnostics, "
            + "and end-to-end label shuffle on 128 hash anchors ($B=200$).\n"
        )
    (dst / "LPA_REVISION_NOTE.md").write_text(
        "Automated minimal revision from physics_local_probe_adaptation_audit. "
        "Recompile with pdflatex; verify four-page limit.\n"
    )
