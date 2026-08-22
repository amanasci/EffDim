"""Publication figures. Accessible colours; no manuscript-identifying paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import PREDECLARED_D
from .pipeline import ValConfig, resolve_path


WONG = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9", "grey": "#666666"}


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure1(root, cfg: ValConfig, dest: Path) -> Path:
    _style()
    qpd = pd.read_csv(resolve_path(root, cfg.qpd_dir) / "aggregate_risk_curves.csv")
    qpd = qpd[qpd.k == 2048].sort_values("d")
    ve = pd.read_csv(resolve_path(root, cfg.cprs_dir) / "variance_explained.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax = axes[0]
    ax.axvspan(11.5, 20.5, color=WONG["sky"], alpha=0.18, lw=0)
    ax.plot(qpd.d, qpd.nmse_lin_pooled, color=WONG["grey"], marker="o", ms=3.5, label="Linear NMSE")
    ax.plot(qpd.d, qpd.nmse_quad_pooled, color=WONG["vermillion"], marker="s", ms=3.5, label="Quadratic NMSE")
    ax.set_xlabel("Chart dimension $d$")
    ax.set_ylabel("Held-out reconstruction NMSE")
    ax.set_title("Held-out reconstruction error")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(3.5, 21.5)

    ax = axes[1]
    ax.axvspan(11.5, 20.5, color=WONG["sky"], alpha=0.18, lw=0, label="Plausible finite-scale range")
    ax.plot(ve.d, ve.r2_L_pooled, color=WONG["blue"], marker="o", ms=3.5, label=r"$R^{2}_L$")
    dlt = np.diff(ve.r2_L_pooled.to_numpy(float))
    ax.plot(ve.d.to_numpy(int)[1:], dlt, color=WONG["orange"], marker="^", ms=3.5, label=r"Marginal $\Delta R^{2}_L$")
    ax.set_xlabel("Chart dimension $d$")
    ax.set_ylabel("Held-out variance / increment")
    ax.set_title("Adequacy, not an eigengap")
    ax.legend(frameon=False, loc="center right")
    ax.set_xlim(7.5, 20.5)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)
    return dest


def figure2(root, cfg: ValConfig, dest: Path) -> Path:
    _style()
    out = cfg.resolved(root)
    assoc = pd.read_csv(out / "metric_associations.csv")
    boot = pd.read_csv(out / "metric_bootstrap.csv")
    metrics = pd.read_csv(out / "probe_metrics_full.csv")
    panel = pd.read_parquet(resolve_path(root, cfg.cprs_dir) / "per_anchor_rank_curve.parquet")
    ve = pd.read_csv(resolve_path(root, cfg.cprs_dir) / "variance_explained.csv")
    primary = assoc[(assoc.target_id == "mag_r_desi_local_oof_r2") & (assoc.slice_mode == "full")].sort_values("d")
    b = boot[(boot.target_id == "mag_r_desi_local_oof_r2") & (boot.slice_mode == "full")].set_index("d")
    error = assoc[(assoc.target_id == "mag_r_desi_oof_mse") & (assoc.slice_mode == "full")].set_index("d")
    use_error = bool(len(error) and np.isfinite(error.loc[16, "controlled"]) and abs(float(error.loc[16, "controlled"])) >= 0.08)
    ycol = "oof_mse" if use_error else "r2_local"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax = axes[0]
    d = primary.d.to_numpy(int)
    ctl = primary.controlled.to_numpy(float)
    ax.axvspan(15.5, 20.5, color=WONG["sky"], alpha=0.18, lw=0)
    ax.axhline(0.0, color="#bbbbbb", lw=0.8)
    ax.fill_between(d, b.loc[d, "ctl_sim_lo"].to_numpy(float), b.loc[d, "ctl_sim_hi"].to_numpy(float), color=WONG["blue"], alpha=0.15, lw=0)
    ax.plot(d, ctl, color=WONG["blue"], marker="o", ms=4, label=r"Controlled $\rho(K_H, R^{2}_{\mathrm{local}})$")
    ax.set_xlabel("Chart dimension $d$")
    ax.set_ylabel(r"Controlled Spearman $\rho$")
    ax.set_title("Association over chart position")
    ax.set_xticks(list(range(8, 21, 2)))
    ax2 = ax.twiny()
    r2map = ve.set_index("d")["r2_L_pooled"]
    ax2.set_xlim(ax.get_xlim())
    ticks = [8, 12, 16, 20]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{r2map.loc[t]:.2f}" if t in r2map.index else "" for t in ticks])
    ax2.set_xlabel(r"Held-out $R^{2}_L$")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    d_rep = PREDECLARED_D["middle"]
    kh = panel[panel.d == d_rep].groupby("sample_id", as_index=False)["K_H_cross"].mean()
    m = metrics.merge(kh, on="sample_id", how="inner")
    x = rankdata(m.K_H_cross.to_numpy(float))
    y = rankdata(m[ycol].to_numpy(float))
    x = 100 * x / len(x)
    y = 100 * y / len(y)
    ax.scatter(x, y, s=10, c=WONG["blue"], alpha=0.25, linewidths=0)
    edges = np.linspace(0, 100, 11)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mmask = (x >= lo) & (x < hi if hi < 100 else x <= hi)
        if mmask.any():
            centres.append(0.5 * (lo + hi))
            means.append(float(np.mean(y[mmask])))
    ax.plot(centres, means, color=WONG["vermillion"], lw=2.0, marker="s", ms=4, label="Decile mean")
    ylab = "OOF MSE percentile" if use_error else r"Local OOF $R^{2}$ percentile"
    ax.set_xlabel(rf"$K_H$ percentile ($d={d_rep}$)")
    ax.set_ylabel(ylab)
    ax.set_title("Anchor-level relationship")
    ax.legend(frameon=False, loc="best")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)
    return dest
