"""At most three figures: unique global-error forest, ViT-B gains, Hessian alignment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "vit_base": "ViT-B",
    "dinov3": "DINOv3",
    "clip_base": "CLIP",
    "convnext_base": "ConvNeXt-B",
    "vit_large": "ViT-L",
}


def write_figures(
    out: Path,
    per_model_mseg: dict[str, dict],
    joint_mseg: dict,
    vitb: pd.DataFrame | None,
    qlca: pd.DataFrame | None,
) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    models = [m for m in LABELS if m in per_model_mseg]
    if models:
        fig, ax = plt.subplots(figsize=(6.6, 2.7 + 0.24 * len(models)))
        y = np.arange(len(models))
        for off, key, name, col in (
            (-0.15, "unique_KH", r"unique $K_H\mid K_{\mathrm{TF}}$", "C1"),
            (0.15, "unique_KTF", r"unique $K_{\mathrm{TF}}\mid K_H$", "C0"),
        ):
            obs, lo, hi = [], [], []
            for m in models:
                rec = per_model_mseg[m][key]
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
        if "unique_KH_bar" in joint_mseg:
            ax.axvline(float(joint_mseg["unique_KH_bar"]["observed"]), color="C1", ls="--", lw=0.7, alpha=0.7)
            ax.axvline(float(joint_mseg["unique_KTF_bar"]["observed"]), color="C0", ls="--", lw=0.7, alpha=0.7)
        ax.axvline(0.0, color="0.7", lw=0.6)
        ax.set_xlabel(r"controlled Spearman with global OOF MSE (absolute probe error)")
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[m] for m in models])
        ax.legend(frameon=False, loc="best")
        ax.set_title("Unique curvature-component associations with global error")
        fig.tight_layout()
        fig.savefig(figdir / "fig1_unique_global_error_forest.pdf")
        fig.savefig(figdir / "fig1_unique_global_error_forest.png", dpi=140)
        plt.close(fig)

    if vitb is not None and "delta_IQ" in vitb.columns:
        cols = [
            ("delta_IQ", "IQ\nisotropic quad"),
            ("delta_TQ", "TQ\ntraceless quad"),
            ("delta_UQ2", "UQ-v2\nfull quad"),
            ("delta_BSH", "BSH\nmean chart"),
            ("delta_BSTF", "BSTF\ntraceless chart"),
            ("delta_BS_v2", "BS\nfull chart"),
        ]
        if qlca is not None and "delta_Q" in qlca.columns:
            merged = vitb.merge(qlca[["sample_id", "delta_Q", "delta_BS"]], on="sample_id", how="left")
        else:
            merged = vitb
            cols = cols
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        data, names = [], []
        for c, lab in cols:
            if c in merged.columns:
                data.append(merged[c].to_numpy(float))
                names.append(lab)
        if "delta_Q" in merged.columns:
            data.append(merged["delta_Q"].to_numpy(float))
            names.append("UQ frozen\n(parity)")
        if "delta_BS" in merged.columns:
            data.append(merged["delta_BS"].to_numpy(float))
            names.append("BS frozen\n(parity)")
        ax.axhline(0.0, color="0.7", lw=0.6)
        bp = ax.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True)
        colors = ["#d4a017", "#3d6b99", "#222222", "#d4a017", "#3d6b99", "#222222", "0.55", "0.55"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        ax.set_ylabel(r"held-out gain vs tangent-linear $L$")
        ax.set_title("ViT-B quadratic label gain (not local-adaptation $\\Delta_{adapt}$)")
        fig.tight_layout()
        fig.savefig(figdir / "fig2_vitb_heldout_gains.pdf")
        fig.savefig(figdir / "fig2_vitb_heldout_gains.png", dpi=140)
        plt.close(fig)

    if vitb is not None and "A_B" in vitb.columns:
        fig, ax = plt.subplots(figsize=(6.4, 3.1))
        items = [
            ("A_B", "A_B_null_median", r"$A_B$ full $B^S$", "0.15"),
            ("A_H", "A_H_null_median", r"$A_H$ mean mode", "C1"),
            ("A_TF", "A_TF_null_median", r"$A_{\mathrm{TF}}$ traceless", "C0"),
        ]
        pos = np.arange(len(items))
        obs = [float(np.nanmedian(vitb[c])) for c, _, _, _ in items]
        null = [float(np.nanmedian(vitb[n])) for _, n, _, _ in items]
        ax.bar(pos - 0.18, obs, width=0.36, color=[c for *_, c in items], label="observed median")
        ax.bar(pos + 0.18, null, width=0.36, color="0.75", label="Haar / random-$\\gamma$ median")
        ax.axhline(1.0, color="0.5", ls=":", lw=0.8, label="isotropic chance $\\approx 1$")
        ax.set_xticks(pos)
        ax.set_xticklabels([lab for _, _, lab, _ in items])
        ax.set_ylabel("normalized Hessian–geometry alignment")
        ax.set_title("Label Hessian vs curvature-component modes (not probe risk)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / "fig3_hessian_alignment_vs_haar.pdf")
        fig.savefig(figdir / "fig3_hessian_alignment_vs_haar.png", dpi=140)
        plt.close(fig)
