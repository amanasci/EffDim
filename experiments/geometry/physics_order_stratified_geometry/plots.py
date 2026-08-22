"""Figures for the order-stratified geometry audit."""

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
    return pd.read_csv(p)


def write_figures(out: Path, cfg) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    spec = _load(out, "quadratic_spectrum.parquet")
    qsum = _load(out, "quadratic_rank_summary.csv")
    ov = _load(out, "tail_quadratic_overlap.parquet")
    pred = _load(out, "conditional_tail_prediction.parquet")
    mix = _load(out, "mixed_scale_components.parquet")
    mc = _load(out, "model_comparison.csv")
    oe = _load(out, "odd_even_diagnostics.parquet")
    nb = _load(out, "normal_complement_bounds.csv")
    seval = _load(out, "synthetic_evaluation.csv")
    assoc = _load(out, "probe_associations.csv")
    repl = _load(out, "cross_model_order_dimensions.csv")

    # 1. quadratic singular spectrum
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if spec is not None and len(spec):
        g = spec[spec.k == spec.k.max()] if "k" in spec.columns else spec
        for sid, sg in list(g.groupby("sample_id"))[:24]:
            ax.plot(sg.q, sg.sA, color="0.7", lw=0.6, alpha=0.5)
        med = g.groupby("q")["sA"].median()
        ax.plot(med.index, med.values, "k-o", ms=4, label="median")
        if "energy_null" in g.columns:
            ax.axhline(np.sqrt(max(float(g.energy_null.median()), 0.0)), ls="--", color="C1", label="null band")
        ax.set_xlabel("quadratic-normal mode q")
        ax.set_ylabel("singular value")
        ax.legend(fontsize=8)
    ax.set_title("Quadratic-normal spectrum")
    fig.tight_layout()
    fig.savefig(figdir / "01_quadratic_spectrum.png", dpi=140)
    plt.close(fig)

    # 2. q2 survival
    fig, ax = plt.subplots(figsize=(6, 4))
    if qsum is not None and len(qsum):
        row = qsum.iloc[-1]
        qs = [c for c in qsum.columns if c.startswith("p_ge_")]
        xs = [int(c.split("_")[-1]) for c in qs]
        ax.plot(xs, [row[c] for c in qs], "o-")
        ax.set_xlabel("q")
        ax.set_ylabel("Pr[q2 >= q]")
    ax.set_title("Reliable q2 survival")
    fig.tight_layout()
    fig.savefig(figdir / "02_q2_survival.png", dpi=140)
    plt.close(fig)

    # 3. principal angles / overlap
    fig, ax = plt.subplots(figsize=(6, 4))
    if ov is not None and len(ov) and "O_E4_B" in ov.columns:
        ax.hist(ov.O_E4_B.dropna(), bins=20, color="C0", alpha=0.8)
        ax.axvline(float(ov.O_E4_B.median()), color="k", ls="--")
        ax.set_xlabel("overlap O(E4, im B12)")
    ax.set_title("E4 vs quadratic-normal image")
    fig.tight_layout()
    fig.savefig(figdir / "03_E4_principal_overlap.png", dpi=140)
    plt.close(fig)

    # 4. held-out quadratic prediction of E4
    fig, ax = plt.subplots(figsize=(6, 4))
    if pred is not None and len(pred) and "r2_E4" in pred.columns:
        ax.hist(pred.r2_E4.dropna(), bins=20)
        ax.set_xlabel("held-out R2 of E4 from phi(u12)")
    ax.set_title("Quadratic prediction of E4")
    fig.tight_layout()
    fig.savefig(figdir / "04_E4_quad_prediction.png", dpi=140)
    plt.close(fig)

    # 5. raw/pred/resid energy placeholder from mix r_ref
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if mix is not None and len(mix):
        for lab, gm in mix.groupby("series"):
            ax.scatter(np.zeros(len(gm)) + hash(lab) % 3, gm.r_ref if "r_ref" in gm.columns else gm.a, s=8, label=lab)
        ax.legend(fontsize=8)
    ax.set_title("Tail energy series (per-anchor r_ref)")
    fig.tight_layout()
    fig.savefig(figdir / "05_tail_energy_by_scale.png", dpi=140)
    plt.close(fig)

    # 6. mixture components
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if mix is not None and len(mix) and "pi_lin" in mix.columns:
        raw = mix[mix.series == "raw_E4"] if "series" in mix.columns else mix
        try:
            ax.boxplot(
                [raw.pi_lin.dropna(), raw.pi_quad.dropna(), raw.pi_thick.dropna()],
                tick_labels=["r^2", "r^4", "c"],
            )
        except TypeError:
            ax.boxplot(
                [raw.pi_lin.dropna(), raw.pi_quad.dropna(), raw.pi_thick.dropna()],
                labels=["r^2", "r^4", "c"],
            )
    ax.set_title("Mixed scale shares at reference radius")
    fig.tight_layout()
    fig.savefig(figdir / "06_mixed_scale_shares.png", dpi=140)
    plt.close(fig)

    # 7. M12q vs M16
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if mc is not None and len(mc) and "M16_linear" in mc.columns:
        qcols = [c for c in mc.columns if c.startswith("M12_quad_q") and c[-1].isdigit()]
        if qcols:
            xs = [int(c.split("q")[-1]) for c in qcols]
            ax.plot(xs, [mc[c].median() for c in qcols], "o-", label="M12,q")
        ax.axhline(float(mc.M16_linear.median()), color="C1", ls="--", label="M16 linear")
        ax.axhline(float(mc.M12_linear.median()), color="C2", ls=":", label="M12 linear")
        ax.legend(fontsize=8)
        ax.set_xlabel("q")
        ax.set_ylabel("held-out ambient MSE")
    ax.set_title("M12,q vs M16 linear")
    fig.tight_layout()
    fig.savefig(figdir / "07_model_comparison.png", dpi=140)
    plt.close(fig)

    # 8. odd/even
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if oe is not None and len(oe):
        cols = [c for c in ["O_odd_T12", "O_even_E4", "O_even_B", "O_odd_E4"] if c in oe.columns]
        if cols:
            try:
                ax.boxplot([oe[c].dropna() for c in cols], tick_labels=cols)
            except TypeError:
                ax.boxplot([oe[c].dropna() for c in cols], labels=cols)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title("Odd/even subspace overlaps")
    fig.tight_layout()
    fig.savefig(figdir / "08_odd_even.png", dpi=140)
    plt.close(fig)

    # 9. bounds
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if nb is not None and len(nb):
        ax.errorbar(np.arange(min(40, len(nb))), nb.d1_minus.iloc[:40], fmt="o", label="d1-")
        ax.scatter(np.arange(min(40, len(nb))), nb.d1_plus.iloc[:40], marker="s", label="d1+")
        if "q2" in nb.columns:
            ax.scatter(np.arange(min(40, len(nb))), nb.q2.iloc[:40], marker="^", label="q2")
        ax.legend(fontsize=8)
    ax.set_title("Tangent bounds and q2")
    fig.tight_layout()
    fig.savefig(figdir / "09_dimension_bounds.png", dpi=140)
    plt.close(fig)

    # 10. synthetic confusion
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if seval is not None and len(seval) and "kind" in seval.columns:
        kinds = list(seval.kind.unique())
        med = seval.groupby("kind")["median_q2"].median().reindex(kinds)
        ax.bar(np.arange(len(kinds)), med.values)
        ax.set_xticks(np.arange(len(kinds)))
        ax.set_xticklabels(kinds, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("median q2")
    ax.set_title("Synthetic recovered q2")
    fig.tight_layout()
    fig.savefig(figdir / "10_synthetic_confusion.png", dpi=140)
    plt.close(fig)

    # 11. rank-12 quadratic vs mag_r
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if assoc is not None and len(assoc):
        ax.barh(assoc.metric, assoc.rho_mag_r)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Spearman vs mag_r R2")
    ax.set_title("Quadratic modes vs rank-16 geography")
    fig.tight_layout()
    fig.savefig(figdir / "11_quad_vs_magr.png", dpi=140)
    plt.close(fig)

    # 12. cross-model
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if repl is not None and len(repl) and "q2" in repl.columns:
        ax.scatter(np.arange(len(repl)), repl.q2, label="q2")
        if "d1_minus" in repl.columns:
            ax.scatter(np.arange(len(repl)), repl.d1_minus, marker="s", label="d1-")
        ax.set_xticks(np.arange(len(repl)))
        ax.set_xticklabels(repl.model, rotation=30, ha="right")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "ViT-B primary only\n(replication pending)", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Cross-model (d1, q2)")
    fig.tight_layout()
    fig.savefig(figdir / "12_cross_model.png", dpi=140)
    plt.close(fig)
    print(f"[osg] figures -> {figdir}", flush=True)
