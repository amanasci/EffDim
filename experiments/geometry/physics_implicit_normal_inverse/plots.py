"""Figures for the implicit normal-space inverse."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(out: Path, name: str):
    p = out / name
    if not p.exists():
        return None
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    if name.endswith(".json"):
        import json

        return json.loads(p.read_text())
    return pd.read_csv(p)


def write_figures(out: Path, cfg) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    lin = _load(out, "linear_constraint_spectrum.parquet")
    quad = _load(out, "quadratic_constraint_spectrum.parquet")
    met = _load(out, "implicit_constraint_metrics.parquet")
    proj = _load(out, "normal_projectors.parquet")
    clas = _load(out, "normal_classification.parquet")
    scal = _load(out, "constraint_scaling.parquet")
    bounds = _load(out, "dimension_bounds.csv")
    curv = _load(out, "implicit_curvature_rank.csv")
    tail = _load(out, "tail_classification.parquet")
    gauss = _load(out, "gauss_validation.csv")
    seval = _load(out, "synthetic_evaluation.csv")
    assoc = _load(out, "probe_associations.csv")
    k_ref = getattr(cfg, "primary_k", 2048)

    def kfilter(df):
        if df is None or not len(df) or "k" not in df.columns:
            return df
        hit = df[df.k == k_ref]
        return hit if len(hit) else df

    # 1. linear constraint spectrum with null bands
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = kfilter(lin)
    if g is not None and len(g) and "eval_lin" in g.columns:
        for sid, sg in list(g.groupby("sample_id"))[:24]:
            ax.plot(sg.j, sg.eval_lin, color="0.75", lw=0.6, alpha=0.5)
        med = g.groupby("j")["eval_lin"].median()
        ax.plot(med.index, med.values, "k-o", ms=4, label="median linear eval")
        if "null_mse" in g.columns:
            ax.axhline(float(g.null_mse.median()), ls="--", color="C1", label="tangent-null band")
        ax.set_xlabel("bottom direction j")
        ax.set_ylabel("linear variance")
        ax.legend(fontsize=8)
    ax.set_title("Linear constraint spectrum")
    fig.tight_layout()
    fig.savefig(figdir / "01_linear_constraint_spectrum.png", dpi=140)
    plt.close(fig)

    # 2. quadratic-profiled spectrum
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = kfilter(quad)
    if g is not None and len(g) and "eval_K" in g.columns:
        for sid, sg in list(g.groupby("sample_id"))[:24]:
            ax.plot(sg.j, sg.eval_K, color="0.75", lw=0.6, alpha=0.5)
        med = g.groupby("j")["eval_K"].median()
        ax.plot(med.index, med.values, "k-o", ms=4, label="median K eval")
        if "null_mse" in g.columns:
            ax.axhline(float(g.null_mse.median()), ls="--", color="C1", label="null band")
        ax.set_xlabel("bottom direction j")
        ax.set_ylabel("profiled quadratic residual")
        ax.legend(fontsize=8)
    ax.set_title("Quadratic-profiled constraint spectrum")
    fig.tight_layout()
    fig.savefig(figdir / "02_quadratic_profiled_spectrum.png", dpi=140)
    plt.close(fig)

    # 3. held-out residual by candidate q
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = kfilter(met)
    if g is not None and len(g) and "q" in g.columns:
        gg = g[g.q > 0]
        if len(gg):
            med = gg.groupby("q")["corr_mse"].median()
            ax.plot(med.index, med.values, "o-", label="quadratic-corrected MSE")
            if "sampson" in gg.columns:
                meds = gg.groupby("q")["sampson"].median()
                ax.plot(meds.index, meds.values, "s--", label="Sampson d_F^2")
            ax.axvline(4, color="C1", ls=":", label="q=4 (d1=16)")
            ax.axvline(8, color="C2", ls=":", label="q=8 (d1=12)")
            ax.set_xlabel("candidate codimension q")
            ax.set_ylabel("held-out residual")
            ax.legend(fontsize=8)
            ax.set_title("Held-out constraint residual by q (not a selection rule)")
    fig.tight_layout()
    fig.savefig(figdir / "03_heldout_residual_by_q.png", dpi=140)
    plt.close(fig)

    # 4. projector split reliability
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = kfilter(quad)
    if g is not None and len(g) and "overlap" in g.columns:
        med = g.groupby("j")["overlap"].median()
        q25 = g.groupby("j")["overlap"].quantile(0.25)
        q75 = g.groupby("j")["overlap"].quantile(0.75)
        ax.fill_between(med.index, q25, q75, color="0.8")
        ax.plot(med.index, med.values, "k-o", ms=4)
        ax.set_xlabel("direction j")
        ax.set_ylabel("split projector overlap")
    ax.set_title("Normal-projector split reliability")
    fig.tight_layout()
    fig.savefig(figdir / "04_projector_split_reliability.png", dpi=140)
    plt.close(fig)

    # 5. raw vs corrected residual scaling
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = kfilter(scal)
    if g is not None and len(g):
        ax.hist(g.amp_exp_raw.dropna(), bins=20, alpha=0.6, label="raw amp exponent")
        ax.hist(g.amp_exp_corr.dropna(), bins=20, alpha=0.6, label="corrected amp exponent")
        ax.axvline(1.0, color="k", ls="--", label="O(r)")
        ax.axvline(2.0, color="C1", ls="--", label="O(r^2)")
        ax.axvline(3.0, color="C2", ls="--", label="O(r^3)")
        ax.set_xlabel("log-log amplitude exponent")
        ax.legend(fontsize=8)
    ax.set_title("Raw versus quadratic-corrected residual scaling")
    fig.tight_layout()
    fig.savefig(figdir / "05_raw_vs_corrected_scaling.png", dpi=140)
    plt.close(fig)

    # 6. certified survival
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = kfilter(clas)
    if g is not None and len(g) and "label" in g.columns:
        labs = [
            "curvature_active_normal",
            "approximately_flat_normal",
            "structured_thickness_normal_candidate",
            "first_order_tangent",
            "mixed_order",
            "unresolved",
        ]
        for lab in labs:
            sub = g[g.label == lab]
            if not len(sub):
                continue
            fr = sub.groupby("j").size() / g.groupby("j").size()
            ax.plot(fr.index, fr.values, "-o", ms=3, label=lab.replace("_", " "))
        ax.set_xlabel("ordered constraint j")
        ax.set_ylabel("fraction of anchors")
        ax.legend(fontsize=7)
    ax.set_title("Certified normal and tangent survival")
    fig.tight_layout()
    fig.savefig(figdir / "06_normal_tangent_survival.png", dpi=140)
    plt.close(fig)

    # 7. dimension bounds across scales
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if bounds is not None and len(bounds):
        ax.plot(bounds.k, bounds.median_d1_minus, "o-", label="d1-")
        ax.plot(bounds.k, bounds.median_d1_plus, "s-", label="d1+")
        ax.axhline(12, color="C0", ls=":", label="12")
        ax.axhline(16, color="C1", ls=":", label="16")
        ax.set_xlabel("k")
        ax.set_ylabel("tangent-dimension bounds")
        ax.legend(fontsize=8)
    ax.set_title("Dimension bounds across scales")
    fig.tight_layout()
    fig.savefig(figdir / "07_dimension_bounds.png", dpi=140)
    plt.close(fig)

    # 8. cN vs q2
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if proj is not None and len(proj) and curv is not None and len(curv):
        p = kfilter(proj)
        c8 = curv[curv.q == 8] if 8 in set(curv.q.tolist()) else curv
        m = p.merge(c8[["sample_id", "q2"]], on="sample_id", how="inner")
        if len(m):
            ax.scatter(m.cN_minus, m.q2, alpha=0.35, s=12)
            ax.set_xlabel("certified c_N-")
            ax.set_ylabel("q2 at q=8 candidate")
    ax.set_title("c_N versus q_2")
    fig.tight_layout()
    fig.savefig(figdir / "08_cN_vs_q2.png", dpi=140)
    plt.close(fig)

    # 9. E4 overlap
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if tail is not None and len(tail) and "e4_normal_frac" in tail.columns:
        ax.hist(tail.e4_normal_frac.dropna(), bins=20, color="C0", alpha=0.8)
        ax.axvline(float(tail.e4_normal_frac.median()), color="k", ls="--")
        ax.set_xlabel("E4 energy in learned normal space")
    ax.set_title("Normal overlap with E4")
    fig.tight_layout()
    fig.savefig(figdir / "09_E4_normal_overlap.png", dpi=140)
    plt.close(fig)

    # 10. unresolved tail classification
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if tail is not None and len(tail):
        cols = [c for c in ["e4_normal_frac", "e4_nonnormal_frac", "overlap_tail"] if c in tail.columns]
        if cols:
            ax.boxplot([tail[c].dropna().values for c in cols])
            ax.set_xticklabels(cols)
            ax.set_ylabel("fraction / overlap")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title("Classification of the unresolved tail")
    fig.tight_layout()
    fig.savefig(figdir / "10_unresolved_tail.png", dpi=140)
    plt.close(fig)

    # 11. synthetic true vs estimated cN
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if seval is not None and len(seval):
        kinds = list(seval.kind.unique())
        mat = []
        for kind in kinds:
            sub = seval[seval.kind == kind]
            mat.append([float(np.nanmean(sub.cN_hat)), float(np.nanmean(sub.true_cN))])
        ax.scatter(seval.true_cN, seval.cN_hat, alpha=0.4, s=14)
        ax.plot([0, 10], [0, 10], "k--", lw=0.8)
        ax.set_xlabel("true c_N")
        ax.set_ylabel("estimated c_N")
    ax.set_title("Synthetic true versus estimated codimension")
    fig.tight_layout()
    fig.savefig(figdir / "11_synth_codimension_matrix.png", dpi=140)
    plt.close(fig)

    # 12. synthetic tangent coverage
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if seval is not None and len(seval):
        ax.scatter(seval.true_d1, seval.d1_hat, alpha=0.4, s=14)
        ax.plot([0, 20], [0, 20], "k--", lw=0.8)
        ax.axhline(12, color="C0", ls=":", label="12")
        ax.axhline(16, color="C1", ls=":", label="16")
        ax.set_xlabel("true d1")
        ax.set_ylabel("estimated d1+")
        ax.legend(fontsize=8)
    ax.set_title("Synthetic tangent-dimension coverage")
    fig.tight_layout()
    fig.savefig(figdir / "12_synth_tangent_coverage.png", dpi=140)
    plt.close(fig)

    # 13. Gauss-map
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if gauss is not None and len(gauss) and "median_overlap" in gauss.columns:
        ax.hist(gauss.median_overlap.dropna(), bins=16, color="C2", alpha=0.85)
        ax.set_xlabel("transported normal-projector overlap")
    ax.set_title("Gauss-map consistency")
    fig.tight_layout()
    fig.savefig(figdir / "13_gauss_map.png", dpi=140)
    plt.close(fig)

    # 14. probe associations
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if assoc is not None and len(assoc) and "rho_mag_r" in assoc.columns:
        ax.barh(assoc.metric, assoc.rho_mag_r)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Spearman vs mag_r R2")
    ax.set_title("Secondary probe associations (frozen geometry)")
    fig.tight_layout()
    fig.savefig(figdir / "14_probe_associations.png", dpi=140)
    plt.close(fig)
