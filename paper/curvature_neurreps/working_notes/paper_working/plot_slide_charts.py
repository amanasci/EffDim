#!/usr/bin/env python3
"""Three drop-in slide charts (widescreen, large type, takeaway titles)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
OUT = ROOT / "paper" / "slides"

FAMILIES = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
DENSE_C, SAE_C, BSF_C = "#222222", "#1f77b4", "#ff7f0e"


def style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_panel():
    sae = pd.read_csv(DATA / "sae_k10.csv")
    bsf = pd.read_csv(DATA / "bsf_k10.csv")
    d = sae[sae.method == "dense_cosine_heldout"][
        ["pair", "family", "size_name", "approx_params_m", "mknn"]
    ].copy()
    d["probe"] = "dense"
    s = sae[sae.method == "shared_side1_basis_idf"][
        ["pair", "family", "size_name", "approx_params_m", "mknn"]
    ].copy()
    s["probe"] = "sae"
    b = bsf[bsf.method == "shared_side1_basis_cosine"][
        ["pair", "family", "size_name", "approx_params_m", "mknn"]
    ].copy()
    b["probe"] = "bsf"
    out = pd.concat([d, s, b], ignore_index=True)
    out["logp"] = np.log10(out.approx_params_m.to_numpy() * 1e6)
    return out


def slide_level(df):
    fig, axes = plt.subplots(1, 5, figsize=(13.33, 4.6), sharey=True)
    probes = [
        ("dense", "Dense", DENSE_C, "o"),
        ("sae", "SAE", SAE_C, "s"),
        ("bsf", "BSF", BSF_C, "D"),
    ]
    for ax, fam in zip(axes, FAMILIES):
        sub = df[df.family == fam]
        for probe, lab, c, m in probes:
            p = sub[sub.probe == probe].sort_values("approx_params_m")
            ax.plot(p.logp, p.mknn, color=c, marker=m, lw=2.0, ms=7, label=lab if fam == "astropt" else None)
        ax.set_title(FAMILY_LABEL[fam], pad=6)
        ax.set_xlabel(r"$\log_{10} P$")
        ax.set_ylim(0.003, 0.036)
        ax.grid(True, alpha=0.28, lw=0.7)
    axes[0].set_ylabel("Held-out mKNN@10")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Probe choice changes the level", y=1.12, fontsize=18, fontweight="semibold")
    fig.tight_layout()
    path = OUT / "01_level.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def slide_interaction():
    inter = pd.read_csv(Path(__file__).resolve().parent / "probe_scale_interaction.csv")
    y = np.arange(len(inter))[::-1]
    fig, ax = plt.subplots(figsize=(13.33, 6.2))
    ax.axvline(0.0, color="0.25", lw=1.6, zorder=0)
    ax.scatter(inter.D_sae, y + 0.16, color=SAE_C, marker="s", s=70, zorder=2, label="SAE")
    ax.scatter(inter.D_bsf, y - 0.16, color=BSF_C, marker="D", s=70, zorder=2, label="BSF")
    for i, r in enumerate(inter.itertuples()):
        ax.plot([r.D_sae, r.D_bsf], [y[i] + 0.16, y[i] - 0.16], color="0.78", lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{FAMILY_LABEL[r.family]}  {r.size_from} → {r.size_to}" for r in inter.itertuples()],
        fontsize=12,
    )
    ax.set_xlabel(r"Probe × scale interaction  $D_R = \Delta M_R - \Delta M_{\mathrm{dense}}$")
    ax.set_xlim(-0.0078, 0.0064)
    ax.grid(True, axis="x", alpha=0.28, lw=0.7)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Extra alignment does not grow with model size", pad=12, fontsize=18, fontweight="semibold")
    fig.tight_layout()
    path = OUT / "02_interaction.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def slide_unpaired():
    u = pd.read_csv(DATA / "unpaired" / "family_scaling_unpaired.csv")
    u = u[(u.representation == "unpaired_dense") & (u.metric == "mknn") & (u.k_or_keff == 10)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 4.8), sharey=True)
    for ax, fam, title in (
        (axes[0], "convnext", "ConvNeXt"),
        (axes[1], "dinov2", "DINOv2"),
    ):
        p = u[u.family == fam].sort_values("approx_params_m")
        ax.errorbar(
            p.log10_parameter_count,
            p.score,
            yerr=p.seed_std,
            color="#2ca02c",
            marker="o",
            lw=2.2,
            ms=8,
            capsize=3,
        )
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$\log_{10} P$")
        ax.grid(True, alpha=0.28, lw=0.7)
    axes[0].set_ylabel("Unpaired relational mKNN@10")
    fig.suptitle("A different probe: still no clean size law", y=1.04, fontsize=18, fontweight="semibold")
    fig.tight_layout()
    path = OUT / "03_unpaired.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    slide_level(load_panel())
    slide_interaction()
    slide_unpaired()
