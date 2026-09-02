#!/usr/bin/env python3
"""Rebuild workshop Figures 1–2 from local CSVs (matched held-out, SAE side1)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"

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


def fig1(df):
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 2.55), sharey=True)
    probes = [
        ("dense", "Held-out dense", DENSE_C, "o", "-"),
        ("sae", "Shared SAE", SAE_C, "s", "-"),
        ("bsf", "Shared BSF", BSF_C, "D", "-"),
    ]
    for ax, fam in zip(axes, FAMILIES):
        sub = df[df.family == fam]
        for probe, lab, c, m, ls in probes:
            p = sub[sub.probe == probe].sort_values("approx_params_m")
            ax.plot(
                p.logp,
                p.mknn,
                color=c,
                marker=m,
                ls=ls,
                lw=1.4,
                ms=5.5,
                label=lab if fam == "astropt" else None,
            )
        ax.set_title(FAMILY_LABEL[fam])
        ax.set_xlabel(r"$\log_{10} P$")
        ax.set_ylim(0.004, 0.035)
        ax.grid(True, alpha=0.25, lw=0.6)
    axes[0].set_ylabel("mKNN@10")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    path = FIG / "legacy_small_multiples_k10.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig2():
    sae = pd.read_csv(DATA / "sae_side1_matched_lifts.csv")
    bsf = pd.read_csv(DATA / "bsf_side1_matched_lifts.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.55))
    ax.axhline(0.0, color="0.55", lw=1.0, zorder=0)
    for fam in FAMILIES:
        for df, c, lab_prefix in ((sae, SAE_C, "SAE"), (bsf, BSF_C, "BSF")):
            p = df[df.family == fam].sort_values("approx_params_m")
            ax.plot(
                p.logp,
                p.lift,
                color=c,
                lw=0.9,
                alpha=0.35,
                zorder=1,
            )
            ax.scatter(
                p.logp,
                p.lift,
                marker=FAMILY_MARKER[fam],
                c=c,
                s=42,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
                label=f"{lab_prefix} {FAMILY_LABEL[fam]}" if False else None,
            )
    # proxy legend: families (markers) and methods (colors)
    from matplotlib.lines import Line2D

    fam_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKER[f],
            color="0.25",
            ls="None",
            ms=6,
            label=FAMILY_LABEL[f],
        )
        for f in FAMILIES
    ]
    meth_handles = [
        Line2D([0], [0], marker="o", color=SAE_C, ls="None", ms=6, label="SAE lift"),
        Line2D([0], [0], marker="o", color=BSF_C, ls="None", ms=6, label="BSF lift"),
    ]
    leg1 = ax.legend(
        handles=fam_handles, loc="upper left", frameon=False, title="Family", fontsize=7.5
    )
    ax.add_artist(leg1)
    ax.legend(handles=meth_handles, loc="upper right", frameon=False, fontsize=7.5)
    ax.text(
        0.02,
        0.02,
        r"Spearman $L$ vs $\log_{10}P$: SAE $\rho{=}{+}0.11$ ($p{\approx}0.68$)"
        "\n"
        r"BSF $\rho{=}{+}0.13$ ($p{\approx}0.64$)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )
    ax.set_xlabel(r"$\log_{10} P$")
    ax.set_ylabel(r"$L(P)=M_{\mathrm{shared}}(P)-M_{\mathrm{dense,heldout}}(P)$")
    ax.set_ylim(-0.001, 0.019)
    ax.grid(True, alpha=0.25, lw=0.6)
    fig.tight_layout()
    path = FIG / "lift_vs_logp_matched.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def write_tables(df):
    rows = []
    for fam in FAMILIES:
        sub = df[df.family == fam].sort_values("approx_params_m")
        sizes = list(sub[sub.probe == "dense"].size_name)
        def delta(probe):
            p = sub[sub.probe == probe].sort_values("approx_params_m")
            return float(p.mknn.iloc[-1] - p.mknn.iloc[0])
        rows.append(
            {
                "family": FAMILY_LABEL[fam],
                "sizes": rf"{sizes[0]}$\to${sizes[-1]}",
                "dense": delta("dense"),
                "sae": delta("sae"),
                "bsf": delta("bsf"),
            }
        )
    lines = [
        r"% Matched held-out first→last mKNN@10. Source: sae_k10 / bsf_k10, side1.",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Family & Sizes & Dense \(\Delta\) & SAE \(\Delta\) & BSF \(\Delta\) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['family']} & {r['sizes']} & "
            f"\\({r['dense']:+.4f}\\) & \\({r['sae']:+.4f}\\) & \\({r['bsf']:+.4f}\\) \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = TAB / "family_first_last.tex"
    path.write_text("\n".join(lines))
    print("wrote", path)
    panel = df.copy()
    panel.to_csv(DATA / "matched_heldout_k10_panel.csv", index=False)


if __name__ == "__main__":
    style()
    df = load_panel()
    fig1(df)
    fig2()
    write_tables(df)
