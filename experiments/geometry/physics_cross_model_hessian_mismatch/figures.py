"""At most three figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figures(out: Path, *, cell_df, frames, p1, p2) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    if cell_df is not None and len(cell_df):
        piv_d = cell_df.pivot(index="model", columns="target", values="rho_Delta")
        piv_a = cell_df.pivot(index="model", columns="target", values="rho_A")
        for ax, piv, title in ((axes[0], piv_d, r"$\rho(M_\Delta,\mathrm{MSE})$"), (axes[1], piv_a, r"$\rho(A_{\mathrm{full}},\mathrm{MSE})$")):
            im = ax.imshow(piv.to_numpy(float), cmap="coolwarm", vmin=-0.3, vmax=0.3, aspect="auto")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(list(piv.columns), rotation=20, ha="right", fontsize=8)
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(list(piv.index), fontsize=8)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "fig1_mismatch_alignment.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    if cell_df is not None and len(cell_df):
        ax.scatter(cell_df["rho_S"], cell_df["rho_Delta"], c=["#4c72b0"] * len(cell_df))
        for _, r in cell_df.iterrows():
            ax.annotate(f"{r.model[:4]}/{r.target[:4]}", (r.rho_S, r.rho_Delta), fontsize=6)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$\rho(S,\mathrm{MSE})$")
        ax.set_ylabel(r"$\rho(M_\Delta,\mathrm{MSE})$")
    fig.tight_layout()
    fig.savefig(out / "fig2_shape_vs_mismatch.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if frames:
        key = next(iter(frames))
        df = frames[key]
        ax.scatter(df["mse_geom"], df["mse_G"], s=8, alpha=0.5)
        ax.set_xlabel(r"geometric MSE expansion")
        ax.set_ylabel("held-out MSE")
        ax.set_title(f"{key[0]} / {key[1]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_geom_mse_expansion.png", dpi=140)
    plt.close(fig)
    _ = p1, p2
