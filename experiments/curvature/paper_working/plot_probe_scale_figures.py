#!/usr/bin/env python3
"""Figure 1 (level) and Figure 2 (lift + probe × scale interaction)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
FIG = ROOT / "paper" / "figures"

FAMILIES = ["astropt", "convnext", "dinov2", "vit", "ijepa"]
FAMILY_LABEL = {
    "astropt": "AstroPT",
    "convnext": "ConvNeXt",
    "dinov2": "DINOv2",
    "vit": "ViT",
    "ijepa": "I-JEPA",
}
FAMILY_MARKER = {
    "astropt": "o",
    "convnext": "s",
    "dinov2": "^",
    "vit": "D",
    "ijepa": "P",
}
DENSE_C = "#222222"
SAE_C = "#1f77b4"
BSF_C = "#ff7f0e"


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


def fig1(df):
    # Taller aspect so NeurIPS 5.5in column keeps tick labels readable.
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.85), sharey=True)
    probes = [
        ("dense", "Dense", DENSE_C, "o"),
        ("sae", "SAE", SAE_C, "s"),
        ("bsf", "BSF", BSF_C, "D"),
    ]
    for ax, fam in zip(axes, FAMILIES):
        sub = df[df.family == fam]
        for probe, lab, c, m in probes:
            p = sub[sub.probe == probe].sort_values("approx_params_m")
            ax.plot(p.logp, p.mknn, color=c, marker=m, lw=1.3, ms=5.0, label=lab if fam == "astropt" else None)
        ax.set_title(FAMILY_LABEL[fam], fontsize=9)
        ax.set_xlabel(r"$\log_{10} P$")
        ax.set_ylim(0.004, 0.035)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.25, lw=0.6)
    axes[0].set_ylabel("mKNN@10")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.tight_layout()
    path = FIG / "legacy_small_multiples_k10.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig2():
    sae = pd.read_csv(DATA / "sae_side1_matched_lifts.csv")
    bsf = pd.read_csv(DATA / "bsf_side1_matched_lifts.csv")
    inter = pd.read_csv(Path(__file__).resolve().parent / "probe_scale_interaction.csv")
    inter["label"] = [
        f"{FAMILY_LABEL[r.family]} {r.size_from}$\\rightarrow${r.size_to}"
        for r in inter.itertuples()
    ]
    # keep family order already in CSV
    y = np.arange(len(inter))[::-1]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.02, 1.18]}
    )

    ax = axes[0]
    ax.axhline(0.0, color="0.55", lw=1.0, zorder=0)
    for fam in FAMILIES:
        for df, c in ((sae, SAE_C), (bsf, BSF_C)):
            p = df[df.family == fam].sort_values("approx_params_m")
            ax.plot(p.logp, p.lift, color=c, lw=0.9, alpha=0.35, zorder=1)
            ax.scatter(
                p.logp,
                p.lift,
                marker=FAMILY_MARKER[fam],
                c=c,
                s=40,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )
    fam_handles = [
        Line2D([0], [0], marker=FAMILY_MARKER[f], color="0.25", ls="None", ms=6, label=FAMILY_LABEL[f])
        for f in FAMILIES
    ]
    meth_handles = [
        Line2D([0], [0], marker="o", color=SAE_C, ls="None", ms=6, label="SAE"),
        Line2D([0], [0], marker="o", color=BSF_C, ls="None", ms=6, label="BSF"),
    ]
    leg1 = ax.legend(handles=fam_handles, loc="upper left", frameon=False, title="Family", fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=meth_handles, loc="upper right", frameon=False, fontsize=7.5)
    ax.set_xlabel(r"$\log_{10} P$")
    ax.set_ylabel(r"$L_R=M_R-M_{\mathrm{dense}}$")
    ax.set_ylim(-0.001, 0.019)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=7.5)
    ax.text(-0.12, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1]
    ax.axvline(0.0, color="0.35", lw=1.2, zorder=0)
    ax.scatter(inter.D_sae, y + 0.14, color=SAE_C, marker="s", s=36, zorder=2, label="SAE")
    ax.scatter(inter.D_bsf, y - 0.14, color=BSF_C, marker="D", s=36, zorder=2, label="BSF")
    for i, r in enumerate(inter.itertuples()):
        yy = y[i]
        ax.plot([r.D_sae, r.D_bsf], [yy + 0.14, yy - 0.14], color="0.75", lw=0.7, zorder=1)
    ax.set_yticks(y)
    def _step_label(r):
        a = str(r.size_from).removesuffix("m")
        b = str(r.size_to).removesuffix("m")
        return f"{FAMILY_LABEL[r.family]} {a}→{b}"

    ax.set_yticklabels([_step_label(r) for r in inter.itertuples()], fontsize=7)
    ax.set_xlabel(r"$D_R=\Delta M_R-\Delta M_{\mathrm{dense}}$")
    ax.set_xlim(-0.0075, 0.0062)
    ax.grid(True, axis="x", alpha=0.25, lw=0.6)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.text(-0.08, 1.02, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=10)

    fig.tight_layout()
    path = FIG / "probe_scale_interaction.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    # keep old filename as a copy so leftover tex refs do not break
    import shutil

    shutil.copyfile(path, FIG / "lift_vs_logp_matched.png")


if __name__ == "__main__":
    style()
    fig1(load_panel())
    fig2()
