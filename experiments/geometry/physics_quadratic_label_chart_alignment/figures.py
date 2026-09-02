"""Figures for quadratic label chart alignment (≤3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figures(out: Path, anchor: pd.DataFrame, primary: dict) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # 1) paired gains
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cols = ["delta_Q", "delta_BS", "delta_FQ"]
    labels = [r"$\Delta_Q$ (UQ)", r"$\Delta_{BS}$", r"$\Delta_{FQ}$"]
    data = [anchor[c].dropna().to_numpy(float) for c in cols if c in anchor.columns]
    labs = [lab for c, lab in zip(cols, labels) if c in anchor.columns]
    if data:
        ax.boxplot(data, labels=labs, showfliers=False)
        ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Held-out MSE gain vs L")
    ax.set_title("Primary/secondary held-out quadratic gains")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_paired_gains.pdf")
    plt.close(fig)

    # 2) KH vs delta_Q
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    m = np.isfinite(anchor.K_H_cross) & np.isfinite(anchor.delta_Q)
    if m.sum():
        ax.scatter(anchor.K_H_cross[m], anchor.delta_Q[m], s=8, alpha=0.45, c="0.2")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$K_H^{\mathrm{cross}}$")
    ax.set_ylabel(r"$\Delta_Q=\mathrm{MSE}_L-\mathrm{MSE}_{UQ}$")
    rho = primary.get("rho_KH_delta_Q", float("nan"))
    ax.set_title(f"Primary 2: controlled ρ≈{rho:.3f}" if np.isfinite(rho) else "Primary 2")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_KH_vs_deltaQ.pdf")
    plt.close(fig)

    # 3) free vs constrained + alignment
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    if "delta_Q" in anchor and "delta_BS" in anchor:
        m = np.isfinite(anchor.delta_Q) & np.isfinite(anchor.delta_BS)
        axes[0].scatter(anchor.delta_Q[m], anchor.delta_BS[m], s=8, alpha=0.45)
        lim = np.nanpercentile(np.concatenate([anchor.delta_Q[m], anchor.delta_BS[m]]), [2, 98])
        axes[0].plot(lim, lim, "k--", lw=0.8)
        axes[0].set_xlabel(r"$\Delta_Q$")
        axes[0].set_ylabel(r"$\Delta_{BS}$")
        axes[0].set_title("UQ vs BS gain")
    if "A_B" in anchor.columns:
        axes[1].hist(anchor.A_B.dropna(), bins=30, color="0.35")
        axes[1].axvline(1.0, color="C1", ls="--", label="isotropic≈1")
        axes[1].set_xlabel(r"$A_B$")
        axes[1].set_title("Hessian–curvature alignment")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_BS_vs_UQ_alignment.pdf")
    plt.close(fig)
