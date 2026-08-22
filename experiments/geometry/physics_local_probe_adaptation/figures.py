"""Single candidate paper figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figure(out: Path, df: pd.DataFrame, primary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    # associations for MSE_G, MSE_C, MSE_P vs KH — show scatter
    for col, color, lab in [
        ("mse_G", "#D55E00", "global MSE"),
        ("mse_C", "#0072B2", "calibrated MSE"),
        ("mse_P", "#009E73", "patch MSE"),
    ]:
        if col not in df.columns:
            continue
        ax.scatter(df.K_H_cross, df[col], s=8, alpha=0.35, c=color, label=lab)
    ax.set_xlabel(r"$K_H^{\mathrm{cross}}$")
    ax.set_ylabel("local OOF MSE")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Curvature vs probe error")

    ax = axes[1]
    if "dMSE_G_to_P" in df.columns:
        x = df.K_H_cross.to_numpy(float)
        y = df.dMSE_G_to_P.to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        # percentile bins
        qs = np.quantile(x[m], np.linspace(0, 1, 6))
        xs, ys, yerr = [], [], []
        for i in range(5):
            sel = m & (x >= qs[i]) & (x <= qs[i + 1] if i < 4 else x <= qs[i + 1] + 1e-12)
            if sel.sum() < 5:
                continue
            xs.append(0.5 * (qs[i] + qs[i + 1]))
            ys.append(float(np.mean(y[sel])))
            yerr.append(float(np.std(y[sel]) / np.sqrt(sel.sum())))
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color="#0072B2")
        rho = primary["observed"]["controlled"]
        ci = primary["ci95"]
        ax.set_title(rf"$\rho_{{\mathrm{{ctl}}}}={rho:+.3f}$ CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.set_xlabel(r"$K_H^{\mathrm{cross}}$ (bin centres)")
    ax.set_ylabel(r"$\Delta\mathrm{MSE}_{G\rightarrow P}$")
    fig.tight_layout()
    fig.savefig(out / "figures" / "local_probe_adaptation.pdf")
    plt.close(fig)
