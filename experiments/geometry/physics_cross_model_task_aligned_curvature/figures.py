"""At most three figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figures(out: Path, *, per_target, agree, r1, model_p2) -> None:
    models = list(model_p2)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(models))
    vals = [model_p2[m]["observed"] for m in models]
    ax.barh(y, vals, color=["#4c72b0" if m != "vit_base" else "#dd8452" for m in models])
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel(r"$\bar\rho_Q$ (equal-target-weight, confirmatory MSE)")
    fig.tight_layout()
    fig.savefig(out / "fig1_per_model_Q.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if per_target is not None and len(per_target):
        piv = per_target.pivot(index="model", columns="target", values="rho_Q")
        im = ax.imshow(piv.to_numpy(float), cmap="coolwarm", vmin=-0.25, vmax=0.25, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(list(piv.columns), rotation=20, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(list(piv.index))
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\rho_{\mathrm{ctl}}(E_Q,MSE)$")
    fig.tight_layout()
    fig.savefig(out / "fig2_model_by_target_Q.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    if per_target is not None and len(per_target):
        mag = per_target[per_target.target == "mag_r_desi"]
        oth = per_target[per_target.target != "mag_r_desi"].groupby("model")["rho_Q"].mean()
        xs = np.arange(len(mag))
        ax.scatter(xs, mag["rho_Q"], marker="D", label="mag_r_desi")
        ax.scatter(xs, [oth.get(m, np.nan) for m in mag["model"]], marker="o", label="mean of other three")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(list(mag["model"]), rotation=15)
    ax.set_ylabel(r"$\rho_Q$ vs held-out MSE")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_mag_r_vs_others.png", dpi=140)
    plt.close(fig)
    _ = agree, r1
