"""At most three figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_figures(out: Path, *, seed_tab, forest, dvq) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    if seed_tab is not None and len(seed_tab):
        models = list(seed_tab["model"])
        ax.bar(np.arange(len(models)) - 0.15, seed_tab["median_rho_CH"], 0.3, label=r"median $\rho$")
        ax.bar(np.arange(len(models)) + 0.15, seed_tab["median_cos_HS"], 0.3, label=r"median cos")
        ax.axhline(0.70, color="k", ls="--", lw=0.8)
        ax.axhline(0.80, color="0.4", ls=":", lw=0.8)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("seed reliability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig1_seed_reliability.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    if forest is not None and len(forest):
        y = np.arange(len(forest))
        ax.errorbar(forest["C_R2"], y - 0.2, xerr=_half(forest, "C_R2"), fmt="o", label=r"$C_H$ vs $R_G^2$")
        ax.errorbar(forest["C_P"], y, xerr=_half(forest, "C_P"), fmt="s", label=r"$C_H$ vs $R_P^2$")
        ax.errorbar(forest["C_A"], y + 0.2, xerr=_half(forest, "C_A"), fmt="^", label=r"$C_H$ vs $\Delta_{adapt}$")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(list(forest["model"]))
    ax.set_xlabel("controlled Spearman")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_forest.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    if dvq is not None and len(dvq):
        ax.scatter(dvq.get("rho_CH_KH", []), dvq.get("C_A", []), c="#0072B2")
        for _, r in dvq.iterrows():
            ax.annotate(str(r.get("model", "")), (r.get("rho_CH_KH", 0), r.get("C_A", 0)), fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"raw $\rho(C_H,K_H)$")
    ax.set_ylabel(r"$\rho_{\mathrm{ctl}}(C_H,\Delta_{\mathrm{adapt}})$")
    fig.tight_layout()
    fig.savefig(out / "fig3_d_vs_q.png", dpi=140)
    plt.close(fig)


def _half(df, col):
    lo = df.get(f"{col}_lo")
    hi = df.get(f"{col}_hi")
    if lo is None or hi is None:
        return None
    mid = df[col]
    return np.vstack([mid - lo, hi - mid])
