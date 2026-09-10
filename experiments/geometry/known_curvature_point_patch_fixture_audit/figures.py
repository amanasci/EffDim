"""At most four main figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_figures(out: Path, tables: dict) -> list[str]:
    paths = []
    anc = tables.get("anchors")
    scale = tables.get("scale")
    robust = tables.get("robust")
    stab = tables.get("stability")

    # Figure 1: mean and full-curvature recovery across analytic fixtures
    if anc is not None and len(anc):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
        df = anc[anc.suite == "A"] if "suite" in anc.columns else anc
        for ax, col_est, col_tr, title in (
            (axes[0], "H_norm_D", "H_norm_T1", "mean curvature ||H^S||"),
            (axes[1], "K_dir_D", "K_dir_T1", r"full residual $K_{\mathrm{dir}}$"),
        ):
            if col_est not in df.columns:
                continue
            for fx, sub in df.groupby("fixture"):
                ax.scatter(sub[col_tr], sub[col_est], s=10, alpha=0.7, label=fx)
            lo = np.nanmin([df[col_tr].min(), df[col_est].min()]) if len(df) else 0
            hi = np.nanmax([df[col_tr].max(), df[col_est].max()]) if len(df) else 1
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel("T1 pointwise truth")
            ax.set_ylabel("decoder estimate")
            ax.set_title(title)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("Figure 1. Decoder vs analytic pointwise truth (Suite A, S0/N0)", fontsize=10)
        p = out / "figures" / "fig1_analytic_recovery.png"
        _save(fig, p)
        paths.append(str(p))

    # Figure 2: pointwise vs matched finite-patch over scale
    if scale is not None and len(scale):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for col, lab, ls in (
            ("rho_Q_T1", "Q vs T1 (pointwise)", "-"),
            ("rho_Q_T2", "Q vs T2 (uniform patch)", "--"),
            ("rho_Q_T3", "Q vs T3 (sampling patch)", ":"),
            ("rho_D_T1", "D vs T1 (pointwise)", "-."),
        ):
            if col in scale.columns:
                ax.plot(scale["radius_median"], scale[col], ls, marker="o", label=lab)
        ax.set_xlabel("median physical neighbourhood radius")
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_title("Figure 2. Pointwise vs matched finite-patch recovery")
        ax.legend(fontsize=8, frameon=False)
        p = out / "figures" / "fig2_point_vs_patch_scale.png"
        _save(fig, p)
        paths.append(str(p))

    # Figure 3: sampling and noise robustness
    if robust is not None and len(robust):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
        if "sampling" in robust.columns:
            s = robust[robust.kind == "sampling"] if "kind" in robust.columns else robust
            axes[0].bar(np.arange(len(s)), s["rho_Kdir"], tick_label=s["condition"])
            axes[0].set_title("sampling regimes")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].set_ylabel(r"$\rho(\widehat K, K_{\mathrm{truth}})$")
        if "kind" in robust.columns:
            n = robust[robust.kind == "noise"]
            if len(n):
                axes[1].bar(np.arange(len(n)), n["rho_Kdir"], tick_label=n["condition"])
                axes[1].set_title("noise regimes")
                axes[1].tick_params(axis="x", rotation=45)
        fig.suptitle("Figure 3. Robustness under non-uniform sampling and normal noise", fontsize=10)
        p = out / "figures" / "fig3_sampling_noise_robustness.png"
        _save(fig, p)
        paths.append(str(p))

    # Figure 4: decoder seed stability vs quadratic split reliability
    if stab is not None and len(stab):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.scatter(stab.get("q_split_R", stab.iloc[:, 0]), stab.get("d_seed_rho", stab.iloc[:, 1]), s=28)
        ax.set_xlabel("quadratic split-half reliability")
        ax.set_ylabel("decoder cross-seed Spearman")
        ax.set_title("Figure 4. Decoder seed stability vs quadratic split reliability")
        ax.axhline(0.5, color="k", ls="--", lw=0.7)
        ax.axvline(0.5, color="k", ls="--", lw=0.7)
        p = out / "figures" / "fig4_seed_vs_split.png"
        _save(fig, p)
        paths.append(str(p))

    return paths
