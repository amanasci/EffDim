"""Three diagnostic figures. Not manuscript-ready."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_figures(out: Path, curves: pd.DataFrame, ceiling: pd.DataFrame, density: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    f4 = curves[curves["fixture"] == "F4"].copy() if "fixture" in curves.columns else curves
    if not f4.empty and "condition" in f4.columns:
        for est, col in (
            ("D-residual", "spearman"),
            ("Q K_H vs T2", "spearman"),
            ("D-full residualized", "spearman"),
        ):
            sub = f4[f4["estimator_target"] == est] if "estimator_target" in f4.columns else pd.DataFrame()
            if sub.empty:
                continue
            ax[0].plot(sub["condition"], sub["spearman"], marker="o", label=est)
            if "r_retained" in sub.columns:
                ax[1].plot(sub["condition"], sub["r_retained"], marker="o", label=est)
        ax[0].axhline(0.75, ls="--", c="0.6", lw=0.8)
        ax[0].axhline(0.40, ls="--", c="0.8", lw=0.8)
        ax[0].set_ylabel("Spearman ρ")
        ax[0].set_title("accuracy")
        ax[1].set_ylabel("ρ_stress / ρ_clean")
        ax[1].set_title("retained")
        ax[1].axhline(1.0, ls=":", c="0.5")
        for a in ax:
            a.tick_params(axis="x", rotation=35)
            a.legend(fontsize=8)
            a.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out / "fig01_operating_curves.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    paths.append(str(p1))

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    if ceiling is not None and not ceiling.empty and "r_rel" in ceiling.columns:
        x = ceiling["r_rel"].to_numpy()
        y = ceiling["rho_truth"].to_numpy() if "rho_truth" in ceiling.columns else np.full_like(x, np.nan)
        ax.scatter(x, y, c="#1f77b4")
        grid = np.linspace(0, 1, 50)
        ax.plot(grid, np.sqrt(np.clip(grid, 0, 1)), "k--", lw=1, label="√r_rel ceiling")
        for _, row in ceiling.iterrows():
            ax.annotate(str(row.get("estimator", "")), (row["r_rel"], row.get("rho_truth", np.nan)), fontsize=7)
        ax.set_xlabel("repeat reliability r_rel")
        ax.set_ylabel("ρ(estimate, truth)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("reliability vs truth correlation")
    fig.tight_layout()
    p2 = out / "fig02_reliability_ceiling.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    paths.append(str(p2))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if density is not None and not density.empty and "quintile" in density.columns:
        for est in density["estimator"].unique():
            sub = density[density["estimator"] == est]
            ax.plot(sub["quintile"], sub["spearman"], marker="o", label=est)
        ax.axhline(0.4, ls="--", c="0.7")
        ax.set_xlabel("true sampling-density quintile (1=sparsest)")
        ax.set_ylabel("Spearman ρ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("density-stratified rank recovery")
    fig.tight_layout()
    p3 = out / "fig03_density_stratified.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    paths.append(str(p3))
    return paths
