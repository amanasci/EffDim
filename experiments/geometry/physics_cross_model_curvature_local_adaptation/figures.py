"""Forest / diagnostic figures from saved tables only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figures(out: Path, primaries: dict, agg: dict, tables: dict[str, pd.DataFrame]) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    models = list(primaries)
    if not models:
        return
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    stats = ("C_G", "C_A", "A")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4 + 0.18 * len(models)), sharey=True)
    y = np.arange(len(models))
    for ax, name in zip(axes, stats):
        obs, lo, hi = [], [], []
        for m in models:
            rec = primaries[m][name]
            obs.append(float(rec["observed"]))
            ci = rec.get("ci95", [np.nan, np.nan])
            lo.append(float(ci[0]))
            hi.append(float(ci[1]))
        ax.errorbar(obs, y, xerr=[np.asarray(obs) - np.asarray(lo), np.asarray(hi) - np.asarray(obs)], fmt="o", ms=4, color="0.15")
        if name + "_bar" in agg:
            ax.axvline(float(agg[name + "_bar"]["observed"]), color="0.45", ls="--", lw=0.8)
        ax.axvline(0.0, color="0.7", lw=0.6)
        ax.set_xlabel(name)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_cross_model_forest.pdf")
    fig.savefig(figdir / "fig1_cross_model_forest.png", dpi=140)
    plt.close(fig)

    # ViT-B or first model: G vs P error vs curvature percentiles
    m0 = "vit_base" if "vit_base" in tables else models[0]
    df = tables[m0].copy()
    if "K_H_cross" not in df or df.K_H_cross.notna().sum() < 8:
        return
    q = pd.qcut(df.K_H_cross.rank(method="first"), 10, labels=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    xs = np.arange(10)
    g = df.groupby(q)
    axes[0].plot(xs, g.mse_G.mean(), "o-", label="G", ms=3)
    axes[0].plot(xs, g.mse_P.mean(), "s--", label="P", ms=3)
    axes[0].set_xlabel("KH decile")
    axes[0].set_ylabel("mean MSE")
    axes[0].legend(frameon=False)
    axes[1].plot(xs, g.delta_adapt.mean(), "o-", ms=3, color="0.2")
    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[1].set_xlabel("KH decile")
    axes[1].set_ylabel(r"mean $\Delta_{adapt}$")
    if "delta_affine" in df.columns:
        axes[2].plot(xs, g.delta_intercept.mean(), "o-", ms=3, label="intercept")
        axes[2].plot(xs, g.delta_affine.mean(), "s--", ms=3, label="affine")
        axes[2].plot(xs, g.delta_direction.mean(), "^-", ms=3, label="direction")
        axes[2].axhline(0.0, color="0.6", lw=0.6)
        axes[2].legend(frameon=False, fontsize=7)
        axes[2].set_xlabel("KH decile")
        axes[2].set_ylabel("mean gain vs G")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_adaptation_anatomy.pdf")
    fig.savefig(figdir / "fig2_adaptation_anatomy.png", dpi=140)
    plt.close(fig)
