"""Audit figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figure(out: Path, df: pd.DataFrame, paired: pd.DataFrame, primary: dict, shuffle: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    # controlled rhos for MSE_G, MSE_P, dMSE
    from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

    labels, vals = [], []
    for ycol, lab in [("mse_G", "MSE_G"), ("mse_P", "MSE_P"), ("dMSE_G_to_P", "ΔMSE")]:
        r = associate(df.K_H_cross, df[ycol], control_matrix(df))["controlled"]
        labels.append(lab)
        vals.append(r)
    ax.bar(labels, vals, color=["#D55E00", "#009E73", "#0072B2"])
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.set_ylabel("controlled Spearman ρ")
    ax.set_title("Curvature associations")
    dr = paired[paired.name == "delta_rho_MSE_GP"]
    if len(dr):
        ax.text(0.02, 0.95, f"Δρ={dr.iloc[0].delta_rho:+.3f}", transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[1]
    if shuffle.get("rows"):
        arr = [r["rho_ctl"] for r in shuffle["rows"]]
        ax.hist(arr, bins=30, color="#56B4E9", edgecolor="white")
        ax.axvline(shuffle.get("obs_rho_ctl", 0), color="#D55E00", lw=2)
        ax.set_title("E2E shuffle null (128 anchors)")
        ax.set_xlabel("ρ_ctl(K_H, ΔMSE)")
    else:
        ax.text(0.5, 0.5, "shuffle skipped", ha="center")
    fig.tight_layout()
    fig.savefig(out / "figures" / "local_adaptation_audit.pdf")
    plt.close(fig)
