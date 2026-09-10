"""Three diagnostic figures. Not manuscript-ready."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_figures(out: Path, assoc: pd.DataFrame, rel_rows: pd.DataFrame, combined: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for i, (outc, title) in enumerate((("r2_G", "global R_G^2"), ("delta_adapt", "adaptation"))):
        for scheme, lab in (("conditional_support", "A conditional"), ("object_support", "B object")):
            sub = assoc[(assoc.scheme == scheme) & (assoc.outcome == outc)]
            if sub.empty:
                continue
            ax[i].hist(sub["controlled"], bins=12, alpha=0.5, label=lab)
        ax[i].axvline(0, c="k", lw=0.8)
        ax[i].set_title(title)
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    p1 = out / "fig01_association_distributions.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    paths.append(str(p1))

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    if not rel_rows.empty and "support_overlap" in rel_rows.columns:
        ax.scatter(rel_rows["support_overlap"], rel_rows["r_b0"], s=12)
        ax.set_xlabel("support overlap with frozen k=2048")
        ax.set_ylabel(r"$\rho(K_H^{(b)}, K_H^{(0)})$")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out / "fig02_reliability_vs_overlap.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    paths.append(str(p2))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if not combined.empty:
        y = np.arange(len(combined))
        ax.hlines(y, combined["q025"], combined["q975"])
        if "original" in combined.columns:
            ax.scatter(combined["original"], y, c="k", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(combined["interval"].astype(str) + " / " + combined["outcome"].astype(str), fontsize=7)
        ax.axvline(0, c="0.5", lw=0.8)
        ax.set_xlabel("controlled correlation")
    fig.tight_layout()
    p3 = out / "fig03_uncertainty_intervals.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    paths.append(str(p3))
    return paths
