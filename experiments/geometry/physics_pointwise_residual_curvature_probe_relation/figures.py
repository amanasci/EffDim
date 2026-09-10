"""At most three figures. Every panel names D-residual explicitly."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figures(
    out: Path,
    *,
    per_seed: dict,
    seeds: tuple[int, ...],
    df: pd.DataFrame,
    xcol: str,
    seed_rel: dict,
    deciles: pd.DataFrame | None,
) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    # 1. seed reliability / pointwise C_H agreement
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.45))
    pairs = [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    for ax, (s, t) in zip(axes, pairs[:3]):
        a = per_seed[s]["C_H"]
        b = per_seed[t]["C_H"]
        ax.scatter(a, b, s=8, c="0.2", alpha=0.7, linewidths=0)
        lim = [min(a.min(), b.min()), max(a.max(), b.max())]
        ax.plot(lim, lim, color="0.7", lw=0.7)
        ax.set_xlabel(f"D-residual $C_H$ seed {s}")
        ax.set_ylabel(f"D-residual $C_H$ seed {t}")
        ax.set_title("pointwise sphere-residual $C_H=\\|H^S\\|$")
    fig.suptitle("D-residual seed agreement (not historical Q)", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_seed_reliability_CH.png", dpi=140, bbox_inches="tight")
    fig.savefig(figdir / "fig1_seed_reliability_CH.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. controlled C_H vs global and patch R^2
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
    x = df[xcol].to_numpy(float)
    for ax, ycol, lab in ((axes[0], "r2_G", r"global $R_G^2$"), (axes[1], "r2_P", r"patch $R_P^2$")):
        ax.scatter(x, df[ycol].to_numpy(float), s=8, c="0.15", alpha=0.7, linewidths=0)
        ax.set_xlabel(r"D-residual consensus $C_H=\|H^S\|$")
        ax.set_ylabel(lab)
        ax.set_title("frozen probe outcome vs D-residual")
    fig.suptitle("D-residual $C_H$ vs frozen global/patch $R^2$ (Q not shown as validated curvature)", y=1.02, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_CH_vs_global_patch_R2.png", dpi=140, bbox_inches="tight")
    fig.savefig(figdir / "fig2_CH_vs_global_patch_R2.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. decile curves
    if deciles is None or len(deciles) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.45))
    xs = deciles.decile.to_numpy()
    axes[0].plot(xs, deciles.r2_G_mean, "o-", ms=3, label=r"$R_G^2$")
    axes[0].plot(xs, deciles.r2_P_mean, "s--", ms=3, label=r"$R_P^2$")
    axes[0].set_xlabel("D-residual $C_H$ decile")
    axes[0].set_ylabel("mean $R^2$")
    axes[0].legend(frameon=False)
    axes[1].plot(xs, deciles.mse_G_mean, "o-", ms=3, label="MSE$_G$")
    axes[1].plot(xs, deciles.mse_P_mean, "s--", ms=3, label="MSE$_P$")
    axes[1].set_xlabel("D-residual $C_H$ decile")
    axes[1].set_ylabel("mean MSE")
    axes[1].legend(frameon=False)
    axes[2].plot(xs, deciles.delta_adapt_mean, "o-", ms=3, color="0.2")
    axes[2].axhline(0.0, color="0.6", lw=0.6)
    axes[2].set_xlabel("D-residual $C_H$ decile")
    axes[2].set_ylabel(r"mean $\Delta_{\mathrm{adapt}}$")
    fig.suptitle("Deciles of D-residual $C_H$ (historical Q statistics are not this axis)", y=1.02, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_decile_GP_adaptation.png", dpi=140, bbox_inches="tight")
    fig.savefig(figdir / "fig3_decile_GP_adaptation.pdf", bbox_inches="tight")
    plt.close(fig)
