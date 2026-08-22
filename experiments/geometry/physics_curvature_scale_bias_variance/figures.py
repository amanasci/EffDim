"""At most two main figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import PRIMARY_D

W = {"blue": "#0072B2", "vermillion": "#D55E00", "grey": "#666666", "sky": "#56B4E9"}


def write_figures(out: Path, assoc: pd.DataFrame, rel: pd.DataFrame, drift: pd.DataFrame) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    if assoc is None or len(assoc) == 0 or "d" not in assoc.columns:
        # Smoke / tiny-n runs may yield an empty association table.
        return
    a = assoc[assoc.d == PRIMARY_D].copy()
    if len(a) == 0:
        return
    med = a.groupby(["R", "m"], as_index=False)[["r2fix_controlled", "msefix_controlled", "R_H_med"]].median()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    sub = med[med.R == 2048].sort_values("m")
    ax.plot(sub.m, sub.r2fix_controlled, "o-", color=W["blue"], label=r"controlled $\rho$ vs local $R^2$")
    ax.plot(sub.m, sub.msefix_controlled, "s--", color=W["vermillion"], label=r"controlled $\rho$ vs OOF MSE")
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.set_xlabel("curvature-fit sample count $m$ ($R=2048$)")
    ax.set_ylabel("controlled Spearman")
    ax.set_title("Sample count at fixed support")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    sub = med[med.m == 1024].sort_values("R")
    ax.plot(sub.R, sub.r2fix_controlled, "o-", color=W["blue"], label=r"$\rho$ vs $R^2$")
    ax.plot(sub.R, sub.msefix_controlled, "s--", color=W["vermillion"], label=r"$\rho$ vs MSE")
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.set_xlabel("geometric support $R$ ($m=1024$)")
    ax.set_ylabel("controlled Spearman")
    ax.set_title("Radius at fixed sample count")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "figure1_count_vs_radius.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    r = rel[rel.d == PRIMARY_D]
    r1 = r[r.R == 2048].sort_values("m")
    ax.plot(r1.m, r1.repeat_spearman_med, "o-", color=W["blue"], label="repeat Spearman of $K_H^{cross}$")
    ax.plot(r1.m, r1.R_H_med, "s--", color=W["grey"], label="median split $R_H$ (not classical reliability)")
    ax.set_xlabel("$m$ at $R=2048$")
    ax.set_ylabel("agreement")
    ax.set_title("Repeat stability vs sample count")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    if len(drift):
        ax.hist(drift.cosine.dropna(), bins=20, color=W["sky"], edgecolor="white")
        ax.set_xlabel("cosine of mean-curvature vectors $R=1024$ vs $R=2048$ ($m=1024$)")
        ax.set_title("Cross-radius vector drift")
    else:
        ax.text(0.5, 0.5, "no drift table", ha="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figdir / "figure2_reliability_drift.pdf")
    plt.close(fig)


write_figures = write_figures
