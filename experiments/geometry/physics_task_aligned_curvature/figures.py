"""At most three figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_figures(out: Path, *, per_target, agree, hist_vs_c) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    if per_target is not None and len(per_target):
        y = np.arange(len(per_target))
        ax.errorbar(per_target["rho_D"], y - 0.12, fmt="o", label=r"$E_D^S$ vs MSE")
        ax.errorbar(per_target["rho_Q"], y + 0.12, fmt="s", label=r"$E_Q^{S,cross}$ vs MSE")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(list(per_target["target"]))
    ax.set_xlabel("controlled Spearman (confirmatory)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_per_target_associations.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    if agree is not None and len(agree):
        ax.bar(np.arange(len(agree)), agree["rho_ctl"])
        ax.set_xticks(np.arange(len(agree)))
        ax.set_xticklabels(list(agree["target"]), rotation=15)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(r"$\rho_{\mathrm{ctl}}(E_D^S,E_Q^{S,cross})$")
    fig.tight_layout()
    fig.savefig(out / "fig2_d_vs_q_agreement.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    if hist_vs_c is not None and len(hist_vs_c):
        y = np.arange(len(hist_vs_c))
        ax.plot(hist_vs_c["rho_H"], y, "o", label="historical / full-data")
        ax.plot(hist_vs_c["rho_C"], y, "s", label="leakage-safe")
        if "rho_radial" in hist_vs_c.columns:
            ax.plot(hist_vs_c["rho_radial"], y, "^", label="radial diagnostic")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(list(hist_vs_c["target"]))
    ax.set_xlabel("controlled Spearman vs held-out/local MSE")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_historical_vs_confirmatory.png", dpi=140)
    plt.close(fig)
