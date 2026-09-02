#!/usr/bin/env python3
"""Physics cross-model level figure (not a size ladder)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
DENSE_C = "#222222"
SAE_C = "#1f77b4"
BSF_C = "#ff7f0e"

NICE = {
    "clip_base": "CLIP-B",
    "convnext_base": "ConvNeXt-B",
    "dinov3_vitb16": "DINOv3-B",
    "vit_base": "ViT-B",
    "vit_large": "ViT-L",
}


def style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_physics():
    df = pd.read_csv(ROOT / "paper_working" / "cross_model_results.csv")
    p = df[df.suite == "physics_holdout20"].copy()
    p["label"] = p.model_a.map(NICE) + r" $\leftrightarrow$ " + p.model_b.map(NICE)
    p.loc[p.within_family_size_pair, "label"] = p.loc[
        p.within_family_size_pair, "label"
    ] + r"$^\dagger$"
    return p.sort_values("dense_mknn")


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    p = load_physics()
    y = np.arange(len(p))
    h = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.15), sharey=True)

    ax = axes[0]
    ax.barh(y - h, p.dense_mknn, height=h, color=DENSE_C, label="Dense")
    ax.barh(y, p.sae_mknn, height=h, color=SAE_C, label="Shared SAE")
    ax.barh(y + h, p.bsf_mknn, height=h, color=BSF_C, label="Shared BSF")
    ax.set_xlabel("Held-out mKNN@10")
    ax.set_yticks(y)
    ax.set_yticklabels(p.label.tolist(), fontsize=8)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="x", alpha=0.3)
    ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.axvline(0, color="0.4", lw=1)
    ax.scatter(p.L_sae, y, marker="s", c=SAE_C, s=42, zorder=3, label="SAE")
    ax.scatter(p.L_bsf, y, marker="D", c=BSF_C, s=42, zorder=3, label="BSF")
    ax.set_xlabel(r"$L_R = M_R - M_{\mathrm{dense}}$")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.text(-0.08, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout()
    out = FIG / "cross_model_physics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
