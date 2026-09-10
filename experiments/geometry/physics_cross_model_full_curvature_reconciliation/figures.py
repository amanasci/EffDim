"""At most three audit figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figures(
    out: Path,
    full_primaries: dict,
    kh_primaries: dict,
    full_agg: dict,
    tables: dict[str, pd.DataFrame],
) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    models = list(full_primaries)
    if not models:
        return
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = {
        "vit_base": "ViT-B",
        "dinov3": "DINOv3",
        "clip_base": "CLIP",
        "convnext_base": "ConvNeXt-B",
        "vit_large": "ViT-L",
    }

    # 1. Forest: global R^2 association, full vs trace
    fig, ax = plt.subplots(figsize=(6.4, 2.6 + 0.22 * len(models)))
    y = np.arange(len(models))
    for off, prim, name, col in (
        (-0.15, full_primaries, r"$K_{\mathrm{dir}}$ (full)", "C0"),
        (0.15, kh_primaries, r"$K_H$ (trace)", "C1"),
    ):
        obs, lo, hi = [], [], []
        for m in models:
            rec = prim[m]["C_R2"]
            obs.append(float(rec["observed"]))
            ci = rec.get("ci95", [np.nan, np.nan])
            lo.append(float(ci[0]))
            hi.append(float(ci[1]))
        ax.errorbar(
            obs,
            y + off,
            xerr=[np.asarray(obs) - np.asarray(lo), np.asarray(hi) - np.asarray(obs)],
            fmt="o",
            ms=4,
            color=col,
            label=name,
        )
    ax.axvline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$\rho_{\mathrm{ctl}}(K, R_G^2)$")
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(m, m) for m in models])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_full_vs_trace_global_forest.pdf")
    fig.savefig(figdir / "fig1_full_vs_trace_global_forest.png", dpi=140)
    plt.close(fig)

    # 2. Paired global vs absolute patch vs relative adaptation under full curvature
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5 + 0.18 * len(models)), sharey=True)
    y = np.arange(len(models))
    for ax, name, xlab in (
        (axes[0], "C_R2", r"$\rho(K_{\mathrm{dir}}, R_G^2)$" + "\nglobal performance"),
        (axes[1], "C_R2P", r"$\rho(K_{\mathrm{dir}}, R_P^2)$" + "\nabsolute patch"),
        (axes[2], "C_A", r"$\rho(K_{\mathrm{dir}}, \Delta_{\mathrm{adapt}})$" + "\nrelative adaptation"),
    ):
        obs, lo, hi = [], [], []
        for m in models:
            rec = full_primaries[m][name]
            obs.append(float(rec["observed"]))
            ci = rec.get("ci95", [np.nan, np.nan])
            lo.append(float(ci[0]))
            hi.append(float(ci[1]))
        ax.errorbar(
            obs,
            y,
            xerr=[np.asarray(obs) - np.asarray(lo), np.asarray(hi) - np.asarray(obs)],
            fmt="o",
            ms=4,
            color="0.15",
        )
        if name + "_bar" in full_agg:
            ax.axvline(float(full_agg[name + "_bar"]["observed"]), color="0.45", ls="--", lw=0.8)
        ax.axvline(0.0, color="0.7", lw=0.6)
        ax.set_xlabel(xlab)
        ax.set_yticks(y)
        ax.set_yticklabels([labels.get(m, m) for m in models])
    fig.tight_layout()
    fig.savefig(figdir / "fig2_global_vs_patch_vs_adaptation.pdf")
    fig.savefig(figdir / "fig2_global_vs_patch_vs_adaptation.png", dpi=140)
    plt.close(fig)

    # 3. Trace vs traceless energy / association
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    fracs, kh_r2, kdir_r2, kan_r2 = [], [], [], []
    for m in models:
        df = tables[m]
        if "traceless_fraction" in df.columns:
            fracs.append(float(np.nanmedian(df.traceless_fraction)))
        else:
            fracs.append(float("nan"))
        kh_r2.append(float(kh_primaries[m]["C_R2"]["observed"]))
        kdir_r2.append(float(full_primaries[m]["C_R2"]["observed"]))
        if "K_aniso" in str(full_primaries[m].get("xcol", "")):
            kan_r2.append(float("nan"))
        else:
            kan_r2.append(float("nan"))
    axes[0].bar(np.arange(len(models)) - 0.0, fracs, color="0.35")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("median traceless energy fraction")
    axes[0].set_xticks(np.arange(len(models)))
    axes[0].set_xticklabels([labels.get(m, m) for m in models], rotation=20, ha="right")
    axes[1].scatter(kh_r2, kdir_r2, c="0.15")
    for m, x, yv in zip(models, kh_r2, kdir_r2):
        axes[1].annotate(labels.get(m, m), (x, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = 0.45
    axes[1].plot([-lim, lim], [-lim, lim], color="0.7", lw=0.6)
    axes[1].axhline(0, color="0.8", lw=0.5)
    axes[1].axvline(0, color="0.8", lw=0.5)
    axes[1].set_xlabel(r"$\rho_{\mathrm{ctl}}(K_H, R_G^2)$")
    axes[1].set_ylabel(r"$\rho_{\mathrm{ctl}}(K_{\mathrm{dir}}, R_G^2)$")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_trace_vs_traceless.pdf")
    fig.savefig(figdir / "fig3_trace_vs_traceless.png", dpi=140)
    plt.close(fig)
