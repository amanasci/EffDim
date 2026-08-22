"""Figures for the stable-tangent-dimension audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def _save(fig, path: Path) -> None:
    try:
        fig.tight_layout()
    except Exception:  # noqa: BLE001
        pass
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _load(out: Path, name: str):
    p = out / name
    if not p.exists():
        return None
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def write_figures(out: Path, cfg: Any) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    loc = _load(out, "local_tangent_dimensions.parquet")
    summary = _load(out, "tangent_dimension_summary.csv")
    pca = _load(out, "nested_pca_spectra.parquet")
    nulls = _load(out, "null_distributions.parquet")
    trk = _load(out, "scale_tracking.parquet")
    e4 = _load(out, "e4_block_evidence.parquet")
    atlas = _load(out, "curvature_metric_atlas.parquet")
    rel = _load(out, "metric_reliability.csv")
    agr = _load(out, "metric_agreement.csv")
    sens = _load(out, "dimension_sensitivity.csv")
    syn_e = _load(out, "synthetic_evaluation.csv")
    assoc = _load(out, "probe_associations.csv")
    repl = _load(out, "cross_model_replication.csv")

    # 1. survival curves
    if loc is not None and len(loc):
        fig, ax = plt.subplots(figsize=(7, 4))
        d_max = int(loc.d_T.max()) if loc.d_T.max() == loc.d_T.max() else 16
        d_max = max(d_max, 16)
        for k, gk in loc.groupby("k"):
            p = np.array([np.mean(gk.d_T >= d) for d in range(1, d_max + 1)])
            ax.plot(range(1, d_max + 1), p, marker="o", ms=3, label=f"k={int(k)}")
        ax.set_xlabel("d")
        ax.set_ylabel("p_d(k) = Pr[d_T >= d]")
        ax.legend(fontsize=8)
        ax.set_title("Stable tangent-dimension survival")
        fig.tight_layout()
        fig.savefig(figdir / "01_survival_curves.png", dpi=140)
        plt.close(fig)

    # 2. d_T(k) with bootstrap bands
    if summary is not None and len(summary):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.fill_between(summary.k, summary.median_lo, summary.median_hi, alpha=0.25)
        ax.plot(summary.k, summary.median_dT, marker="o")
        ax.axhline(12, ls="--", color="gray", label=r"$d_G=12$")
        ax.set_xlabel("k")
        ax.set_ylabel("median d_T(k)")
        ax.legend()
        ax.set_title("Model-level stable tangent dimension")
        fig.tight_layout()
        fig.savefig(figdir / "02_dT_vs_k.png", dpi=140)
        plt.close(fig)

    # 3. eigenspectra by scale
    if pca is not None and len(pca):
        full = pca[pca.split == -1]
        if len(full):
            fig, ax = plt.subplots(figsize=(7, 4))
            for k, gk in full.groupby("k"):
                med = gk.groupby("d")["ev_full"].median()
                ax.semilogy(med.index, med.values, marker="o", ms=3, label=f"k={int(k)}")
            ax.set_xlabel("rank")
            ax.set_ylabel("eigenvalue")
            ax.legend(fontsize=8)
            ax.set_title("Nested PCA spectra by scale")
            fig.tight_layout()
            fig.savefig(figdir / "03_eigenspectra.png", dpi=140)
            plt.close(fig)

    # 4. observed vs null stability
    if pca is not None and nulls is not None and len(pca) and len(nulls):
        obs = pca[(pca.split >= 0) & (pca.d > 0)]
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(obs):
            medA = obs.groupby("d")["A"].median()
            ax.plot(medA.index, medA.values, marker="o", label="observed A(d)")
        q = nulls.groupby("d")["null_agree_iso_q99"].median()
        ax.plot(q.index, q.values, ls="--", label="isotropic null q99")
        ax.set_xlabel("d")
        ax.set_ylabel("prefix agreement")
        ax.legend()
        ax.set_title("Observed vs null block stability")
        fig.tight_layout()
        fig.savefig(figdir / "04_stability_vs_null.png", dpi=140)
        plt.close(fig)

    # 5. incremental gain
    if pca is not None and len(pca):
        obs = pca[(pca.split >= 0) & (pca.d > 0)]
        if len(obs):
            fig, ax = plt.subplots(figsize=(7, 4))
            k_ref = int(obs.k.mode().iloc[0])
            g = obs[obs.k == k_ref].groupby("d")["G"].median()
            ax.plot(g.index, g.values, marker="o")
            ax.axvline(12, ls="--", color="gray")
            ax.axvline(16, ls=":", color="gray")
            ax.set_xlabel("d")
            ax.set_ylabel("G(d)")
            ax.set_title("Incremental held-out linear gain")
            fig.tight_layout()
            fig.savefig(figdir / "05_incremental_gain.png", dpi=140)
            plt.close(fig)

    # 6. cross-scale persistence
    if trk is not None and len(trk):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(trk.mean_overlap.dropna(), bins=20)
        ax.set_xlabel("mean cross-scale overlap")
        ax.set_title("Cross-scale subspace persistence")
        fig.tight_layout()
        fig.savefig(figdir / "06_cross_scale_persistence.png", dpi=140)
        plt.close(fig)

    # 7. scaling exponents
    if trk is not None and len(trk):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(
            trk.rank0 + 0.5 * trk.width,
            trk.alpha,
            yerr=[trk.alpha - trk.alpha_lo, trk.alpha_hi - trk.alpha],
            fmt="o",
            alpha=0.35,
            ms=3,
        )
        ax.axhline(2, color="C0", ls="--", label="tangent a=2")
        ax.axhline(4, color="C1", ls="--", label="curvature a=4")
        ax.axhline(0, color="C2", ls="--", label="thickness a=0")
        ax.set_xlabel("block centre rank")
        ax.set_ylabel("alpha")
        ax.legend(fontsize=8)
        ax.set_title("Scaling exponents")
        fig.tight_layout()
        fig.savefig(figdir / "07_scaling_exponents.png", dpi=140)
        plt.close(fig)

    # 8. E4 panel
    if e4 is not None and len(e4):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].boxplot([e4[c].dropna() for c in ["A_13", "A_14", "A_15", "A_16"] if c in e4])
        axes[0].set_xticklabels(["13", "14", "15", "16"])
        axes[0].set_title("Split agreement")
        axes[1].boxplot([e4[c].dropna() for c in ["G_13", "G_14", "G_15", "G_16"] if c in e4])
        axes[1].set_xticklabels(["13", "14", "15", "16"])
        axes[1].set_title("Held-out gain")
        fig.suptitle("ViT-B directions 13–16")
        fig.tight_layout()
        fig.savefig(figdir / "08_e4_panel.png", dpi=140)
        plt.close(fig)

    # 9. D_lin and Q_S vs d
    if atlas is not None and len(atlas):
        g = atlas.groupby("d")[["D_lin", "Q_S"]].median()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(g.index, g.D_lin, marker="o", label="D_lin")
        ax.plot(g.index, g.Q_S, marker="s", label="Q_S")
        ax.set_xlabel("d")
        ax.legend()
        ax.set_title("Linear distortion and quadratic predictability")
        fig.tight_layout()
        fig.savefig(figdir / "09_Dlin_QS.png", dpi=140)
        plt.close(fig)

    # 10. reliability
    if rel is not None and len(rel) and "split_rho" in rel.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        d0 = rel[rel.d == rel.d.mode().iloc[0]] if len(rel) else rel
        ax.bar(np.arange(len(d0)), d0.split_rho.fillna(0))
        ax.set_xticks(np.arange(len(d0)))
        ax.set_xticklabels(d0.metric, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("split-half Spearman")
        ax.set_title("Metric reliability")
        fig.tight_layout()
        fig.savefig(figdir / "10_metric_reliability.png", dpi=140)
        plt.close(fig)

    # 11. agreement heatmap
    if agr is not None and len(agr):
        d0 = agr[agr.d == agr.d.mode().iloc[0]]
        names = sorted(set(d0.a) | set(d0.b))
        M = np.eye(len(names))
        idx = {n: i for i, n in enumerate(names)}
        for r in d0.itertuples():
            M[idx[r.a], idx[r.b]] = r.rho
            M[idx[r.b], idx[r.a]] = r.rho
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        fig.colorbar(im, ax=ax)
        ax.set_title("Metric agreement")
        fig.tight_layout()
        fig.savefig(figdir / "11_metric_agreement.png", dpi=140)
        plt.close(fig)

    # 12. dimension sensitivity
    if sens is not None and len(sens):
        fig, ax = plt.subplots(figsize=(7, 4))
        cols = [c for c in sens.columns if c.startswith("med_d")]
        for _, r in sens.iterrows():
            xs = [int(c.replace("med_d", "")) for c in cols]
            ys = [r[c] for c in cols]
            ax.plot(xs, ys, marker="o", label=r.metric)
        ax.set_xlabel("d")
        ax.legend(fontsize=8)
        ax.set_title("Dimension-sensitivity envelopes")
        fig.tight_layout()
        fig.savefig(figdir / "12_dimension_sensitivity.png", dpi=140)
        plt.close(fig)

    # 13. synthetic recovery
    if syn_e is not None and len(syn_e):
        fig, ax = plt.subplots(figsize=(8, 4))
        g = syn_e.groupby("kind")[["median_dT", "true_d"]].median()
        x = np.arange(len(g))
        ax.bar(x - 0.15, g.true_d, 0.3, label="true d")
        ax.bar(x + 0.15, g.median_dT, 0.3, label="recovered d_T")
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=40, ha="right", fontsize=8)
        ax.legend()
        ax.set_title("Synthetic recovery")
        fig.tight_layout()
        fig.savefig(figdir / "13_synthetic_recovery.png", dpi=140)
        plt.close(fig)

    # 14. probe paths
    if assoc is not None and len(assoc):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, gm in assoc.groupby("metric"):
            ax.plot(gm.d, gm.raw, marker="o", ms=3, label=m)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7)
        ax.set_xlabel("d")
        ax.set_ylabel("Spearman vs mag_r R2")
        ax.set_title("Secondary probe-association paths")
        fig.tight_layout()
        fig.savefig(figdir / "14_probe_paths.png", dpi=140)
        plt.close(fig)

    # 15. cross-model
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if repl is not None and len(repl) and "median_dT" in repl.columns:
        ok = repl[repl.get("ok", True) == True] if "ok" in repl.columns else repl
        ax.errorbar(np.arange(len(ok)), ok.median_dT, fmt="o", label="d_T")
        if "d_G" in ok.columns:
            ax.scatter(np.arange(len(ok)), ok.d_G, marker="s", label="d_G")
        ax.set_xticks(np.arange(len(ok)))
        ax.set_xticklabels(ok.model, rotation=30, ha="right")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "ViT-B primary only\n(replication pending)", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Cross-model stable dimension")
    fig.tight_layout()
    fig.savefig(figdir / "15_cross_model.png", dpi=140)
    plt.close(fig)
    print(f"[std] figures → {figdir}", flush=True)
