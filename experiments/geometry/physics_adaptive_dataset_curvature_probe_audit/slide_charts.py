"""Slide charts for the frozen discovery curve (local OOF R², never catalog mag)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .pipeline import platonic_root, resolve_path, write_df


Y_NAME = "mag_r_desi_local_oof_r2"
TITLE_1 = "Curvature predicts local probe difficulty under the extended chart"
TITLE_DIM = "Dimensionality is selected by held-out adequacy, not a single eigengap"
TITLE_CURV = "Curvature predicts where local linear probes struggle"
SUB_DIM = (
    r"Held-out linear and quadratic NMSE versus cumulative $R^{2}_L$."
    "\n"
    r"Thresholds $\tau=0.80$, $0.85$, $0.90$; quadratic plateau IQR $d=18$–$19$."
)
SUB_CURV = (
    r"Controlled Spearman $\rho(K_H,\,\mathrm{local\ OOF\ }R^{2})$ versus held-out $R^{2}_L$."
    "\n"
    "Negative values: higher curvature, worse local linear probes."
)
FOOTER_CURV = (
    "Curvature is finite-scale and rank-conditioned. "
    "Outcome: local OOF probe $R^{2}$, not raw catalogue magnitude."
)
SLIDE = (13.333, 7.5)  # 16:9 inches
DPI = 300
BLUE = "#1f4e8c"
GREY = "#7a7a7a"
TERRACOTTA = "#c45c26"


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.22,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 17,
            "axes.titlesize": 22,
            "axes.titleweight": "medium",
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "lines.linewidth": 2.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_frozen_tables(root: Path) -> dict[str, pd.DataFrame]:
    cprs = resolve_path(root, "outputs/geometry/physics_curvature_probe_rank_sweep")
    assoc = pd.read_csv(cprs / "dimension_associations.csv")
    assoc = assoc[(assoc["mask"] == "complete_case") & (assoc.metric == "K_H")].sort_values("d")
    boot = pd.read_csv(cprs / "bootstrap_results.csv")
    perm = pd.read_csv(cprs / "permutation_results.csv")
    panel = pd.read_parquet(cprs / "per_anchor_rank_curve.parquet")
    d16 = panel[panel.d == 16][["sample_id", "K_H_cross", "local_r2"]].drop_duplicates("sample_id")
    ve = pd.read_csv(cprs / "variance_explained.csv")
    rel = pd.read_csv(cprs / "curvature_reliability.csv")
    qpd = pd.read_csv(resolve_path(root, "outputs/geometry/physics_quadratic_predictive_dimension") / "aggregate_risk_curves.csv")
    qpd = qpd[qpd.k == 2048].sort_values("d")
    plat = pd.read_csv(resolve_path(root, "outputs/geometry/physics_quadratic_predictive_dimension") / "plateau_bootstrap.csv")
    lin = pd.read_csv(
        resolve_path(root, "outputs/geometry/physics_adaptive_dataset_curvature_probe")
        / "datasets/physics_vit_base/linear_risk_pooled.csv"
    )
    return {
        "assoc": assoc,
        "boot": boot,
        "perm": perm,
        "d16": d16,
        "ve": ve,
        "rel": rel,
        "qpd": qpd,
        "plat": plat,
        "lin": lin,
    }


def _save(fig: plt.Figure, dest: Path, stem: str) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"{stem}.png"
    svg = dest / f"{stem}.svg"
    jpg = dest / f"{stem}.jpg"
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg)
    fig.savefig(jpg, dpi=DPI, format="jpeg", pil_kwargs={"quality": 95})
    plt.close(fig)
    return [png, svg, jpg]


def chart1_main_result(tables: dict[str, pd.DataFrame], dest: Path) -> list[Path]:
    assoc, boot, perm = tables["assoc"], tables["boot"], tables["perm"]
    d = assoc.d.to_numpy(int)
    raw = assoc.raw.to_numpy(float)
    ctl = assoc.controlled.to_numpy(float)
    b = boot.set_index("d").reindex(d)
    pctl = perm[perm.kind == "controlled"].set_index("d")
    fwer = {int(i): float(pctl.loc[i, "p_fwer"]) for i in pctl.index if int(i) in set(d)}

    fig, ax = plt.subplots(figsize=SLIDE)
    ax.axvspan(15.5, 20.5, color="#1f4e8c", alpha=0.08, zorder=0, lw=0)
    ax.axhline(0.0, color="#b0b0b0", lw=1.0, zorder=1)
    ax.axvline(12, color="#333333", ls=":", lw=1.4, zorder=1)

    ax.fill_between(d, b.raw_lo.to_numpy(float), b.raw_hi.to_numpy(float), color="#9a9a9a", alpha=0.18, zorder=2, lw=0)
    ax.fill_between(d, b.ctl_lo.to_numpy(float), b.ctl_hi.to_numpy(float), color="#1f4e8c", alpha=0.16, zorder=2, lw=0)
    ax.plot(d, raw, ls="--", color="#7a7a7a", marker="o", ms=7, mfc="white", mew=1.6, zorder=3, label="Raw")
    ax.plot(d, ctl, ls="-", color="#1f4e8c", marker="o", ms=8, zorder=4, label="Controlled")

    fwer_d, fwer_y = [], []
    for di, yi in zip(d, ctl):
        p = fwer.get(int(di), 1.0)
        if np.isfinite(p) and p <= 0.05:
            fwer_d.append(int(di))
            fwer_y.append(float(yi))
    ax.scatter(fwer_d, fwer_y, s=220, marker="*", color="#1f4e8c", edgecolors="white", linewidths=0.6, zorder=5, label=r"Controlled FWER $p\leq 0.05$")

    notes = {12: (0.143, (12.15, 0.27)), 16: (-0.240, (13.35, -0.355)), 20: (-0.233, (18.05, -0.355))}
    for di, (val, xy) in notes.items():
        yi = float(assoc.loc[assoc.d == di, "controlled"].iloc[0])
        sign = "+" if val > 0 else ""
        ax.annotate(
            f"$d={di}$: {sign}{val:.3f}",
            xy=(di, yi),
            xytext=xy,
            textcoords="data",
            fontsize=15,
            color="#1f4e8c",
            arrowprops={"arrowstyle": "-", "color": "#1f4e8c", "lw": 0.9},
            ha="left",
            va="center",
        )

    ax.text(12.12, 0.40, r"$d=12$", fontsize=14, color="#333333", ha="left", va="bottom")
    ax.text(17.9, 0.40, r"stable negative  $d=16$–$20$", fontsize=14, color="#1f4e8c", ha="center", va="bottom")

    ax.set_xlim(7.6, 20.6)
    ax.set_xticks(list(range(8, 21)))
    ax.set_ylim(-0.55, 0.45)
    ax.set_xlabel("Chart rank  $d$")
    ax.set_ylabel(r"Spearman $\rho\!\left(K_H,\;\mathrm{local\ OOF\ }R^{2}\right)$")
    ax.set_title(TITLE_1, pad=12)
    ax.legend(frameon=False, loc="lower left", ncol=1)
    ax.text(
        0.0,
        -0.14,
        r"Probe: $\mathtt{mag\_r\_desi\_local\_oof\_r2}$.  Bands: paired anchor bootstrap 95% CI.  "
        r"Stars: max-statistic FWER on $d=12,\ldots,20$.  Not catalog magnitude.",
        transform=ax.transAxes,
        fontsize=12,
        color="#555555",
        ha="left",
        va="top",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18)
    return _save(fig, dest, "chart1_main_result")


def chart2_anchor_d16(tables: dict[str, pd.DataFrame], dest: Path) -> list[Path]:
    df = tables["d16"].copy()
    x = rankdata(df.K_H_cross.to_numpy(float), method="average")
    y = rankdata(df.local_r2.to_numpy(float), method="average")
    x = 100.0 * x / len(x)
    y = 100.0 * y / len(y)

    edges = np.linspace(0, 100, 11)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi if hi < 100 else x <= hi)
        if m.any():
            centres.append(0.5 * (lo + hi))
            means.append(float(np.mean(y[m])))

    fig, ax = plt.subplots(figsize=SLIDE)
    ax.plot([0, 100], [100, 0], color="#d0d0d0", ls=":", lw=1.2, zorder=0)
    ax.scatter(x, y, s=22, c="#1f4e8c", alpha=0.22, linewidths=0, zorder=1)
    ax.plot(centres, means, color="#1f4e8c", lw=3.0, marker="s", ms=8, zorder=2, label="10-bin mean")

    ax.text(
        0.04,
        0.08,
        r"raw $\rho=-0.412$" + "\n" + r"controlled $\rho=-0.240$",
        transform=ax.transAxes,
        fontsize=18,
        color="#1f4e8c",
        va="bottom",
        ha="left",
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d8d8d8", "linewidth": 0.8},
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Percentile rank of $K_H$  ($d=16$)")
    ax.set_ylabel(r"Percentile rank of local OOF probe $R^{2}$")
    ax.set_title(r"$d=16$: $K_H$ versus $\mathtt{mag\_r\_desi\_local\_oof\_r2}$", pad=12)
    ax.legend(frameon=False, loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.0,
        -0.14,
        r"$n=512$ anchors.  Higher local OOF $R^{2}$ is easier local probe reconstruction, not catalog magnitude.",
        transform=ax.transAxes,
        fontsize=12,
        color="#555555",
        ha="left",
        va="top",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.16)
    return _save(fig, dest, "chart2_anchor_d16")


def _head(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=20, fontweight="medium", color="#111111")
    fig.text(
        0.5,
        0.865,
        subtitle,
        ha="center",
        va="top",
        fontsize=13,
        color="#444444",
        linespacing=1.45,
    )


def _footer(ax, text: str) -> None:
    ax.text(
        0.0,
        -0.16,
        text,
        transform=ax.transAxes,
        fontsize=12,
        color="#444444",
        ha="left",
        va="top",
        wrap=True,
    )


def chart1_evaluating_dimensionality(tables: dict[str, pd.DataFrame], dest: Path) -> list[Path]:
    qpd = tables["qpd"].sort_values("d")
    lin = tables["lin"].sort_values("d")
    plat = tables["plat"]
    dQ_lo, dQ_hi = float(plat.dQ.quantile(0.25)), float(plat.dQ.quantile(0.75))

    d_q = qpd.d.to_numpy(int)
    nmse_lin = qpd.nmse_lin_pooled.to_numpy(float)
    nmse_quad = qpd.nmse_quad_pooled.to_numpy(float)

    lin_ext = lin[(lin.d >= 4) & (lin.d <= 44)].copy()
    d_var = lin_ext.d.to_numpy(int)
    r2_L = lin_ext.r2_L_pooled.to_numpy(float)
    nmse_lin_ext = 1.0 - r2_L

    fig, ax = plt.subplots(figsize=SLIDE)
    ax_r = ax.twinx()

    ax.axvspan(dQ_lo - 0.5, 20.5, color=TERRACOTTA, alpha=0.10, zorder=0, lw=0)
    ax.axvline(12, color="#333333", ls=":", lw=1.3, zorder=1)
    ax.axvline(20, color="#333333", ls=":", lw=1.3, zorder=1)
    ax.axvline(41, color="#333333", ls=":", lw=1.3, zorder=1)

    ax_r.axhline(0.80, color=BLUE, ls="--", lw=1.15, alpha=0.55, zorder=1)
    ax_r.axhline(0.85, color=BLUE, ls="--", lw=1.15, alpha=0.55, zorder=1)
    ax_r.axhline(0.90, color=BLUE, ls="--", lw=1.15, alpha=0.55, zorder=1)

    ax.plot(d_q, nmse_lin, color=GREY, marker="o", ms=6.5, zorder=3, label="Held-out linear NMSE")
    cont = d_var >= int(d_q.max())
    ax.plot(
        d_var[cont],
        nmse_lin_ext[cont],
        color=GREY,
        ls="--",
        lw=2.2,
        zorder=3,
        label=r"Linear NMSE ($d>20$)",
    )
    ax.plot(d_q, nmse_quad, color=TERRACOTTA, marker="s", ms=6.5, zorder=4, label="Held-out quadratic NMSE")
    ax_r.plot(d_var, r2_L, color=BLUE, lw=2.6, zorder=4, label=r"Held-out $R^{2}_L$")
    mark = np.isin(d_var, [4, 8, 12, 16, 20, 28, 36, 41, 44])
    ax_r.plot(d_var[mark], r2_L[mark], color=BLUE, ls="none", marker="o", ms=6.0, zorder=5)

    ax.text(12.25, 0.348, r"$\tau=0.80$" + "\n" + r"$d=12$", fontsize=13, color="#333333", ha="left", va="top", linespacing=1.25)
    ax.text(20.3, 0.348, r"$\tau=0.85$" + "\n" + r"$d=20$", fontsize=13, color="#333333", ha="left", va="top", linespacing=1.25)
    ax.text(41.0, 0.348, r"$\tau=0.90$" + "\n" + r"$d=41$", fontsize=13, color="#333333", ha="center", va="top", linespacing=1.25)
    ax.text(
        0.5 * (dQ_lo + 20.0),
        0.105,
        r"Quadratic plateau IQR" + "\n" + r"$d=18$–$19$",
        fontsize=13,
        color=TERRACOTTA,
        ha="center",
        va="bottom",
        linespacing=1.25,
    )

    ax.set_xlim(3.5, 44.8)
    ax.set_xticks([4, 8, 12, 16, 20, 28, 36, 41, 44])
    ax.set_ylim(0.08, 0.36)
    ax_r.set_ylim(0.64, 0.93)
    ax_r.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_xlabel(r"Candidate chart dimension  $d$")
    ax.set_ylabel("Held-out reconstruction NMSE")
    ax_r.set_ylabel(r"Held-out $R^{2}_L$  (variance explained)")
    _head(fig, ax, TITLE_DIM, SUB_DIM)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        frameon=True,
        fancybox=False,
        edgecolor="#e2e2e2",
        framealpha=0.94,
        loc="center left",
        bbox_to_anchor=(0.52, 0.58),
        fontsize=13,
    )
    _footer(
        ax,
        r"Pooled held-out NMSE and $R^{2}_L$ at $k=2048$. Quadratic series ends at the QPD family max $d=20$. "
        r"Linear $R^{2}_L$ is still rising at $d=44$ ($d_L^{\mathrm{plat}}=115$).",
    )
    fig.subplots_adjust(left=0.08, right=0.90, top=0.70, bottom=0.18)
    return _save(fig, dest, "chart1_evaluating_dimensionality")


def chart2_curvature_probe_performance(tables: dict[str, pd.DataFrame], dest: Path) -> list[Path]:
    ve = tables["ve"].copy()
    boot = tables["boot"].copy()
    rel = tables["rel"]
    m = boot.merge(ve, on="d").sort_values("d")
    reliable = set(rel.loc[~rel.fail_reliability.astype(bool), "d"].astype(int))
    m = m[m.d.isin(reliable)]
    x = m.r2_L_pooled.to_numpy(float)
    y = m.controlled.to_numpy(float)
    lo = m.ctl_lo.to_numpy(float)
    hi = m.ctl_hi.to_numpy(float)
    ds = m.d.to_numpy(int)

    x85 = float(m.loc[m.d == 20, "r2_L_pooled"].iloc[0])
    # FWER-supported negative span in this family is d=16–20
    x_neg = float(m.loc[m.d == 16, "r2_L_pooled"].iloc[0])

    fig, ax = plt.subplots(figsize=SLIDE)
    ax.axvspan(float(x.min()), float(x.max()), color=BLUE, alpha=0.06, zorder=0, lw=0)
    ax.axvspan(x_neg, x85, color=BLUE, alpha=0.12, zorder=0, lw=0)
    ax.axhline(0.0, color="#b0b0b0", lw=1.0, zorder=1)
    ax.axvline(0.80, color="#333333", ls=":", lw=1.3, zorder=1)
    ax.axvline(0.85, color="#333333", ls=":", lw=1.3, zorder=1)

    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.16, zorder=2, lw=0, label="Bootstrap 95% CI")
    ax.plot(x, y, color=BLUE, marker="o", ms=8, zorder=4, label=r"Controlled $\rho$")

    ax.text(0.8015, 0.325, r"$\tau=0.80$", fontsize=14, color="#333333", ha="left", va="bottom")
    ax.text(0.8495, 0.325, r"$\tau=0.85$", fontsize=14, color="#333333", ha="right", va="bottom")
    ax.text(
        0.5 * (float(x.min()) + float(x.max())),
        0.352,
        r"Reliable range  $d=8$–$20$",
        fontsize=14,
        color=BLUE,
        ha="center",
        va="bottom",
    )
    ax.text(
        0.5 * (x_neg + x85),
        -0.405,
        r"FWER-negative  $d=16$–$20$",
        fontsize=14,
        color=BLUE,
        ha="center",
        va="top",
    )

    for di, xi, yi in zip(ds, x, y):
        if int(di) in {8, 12, 16, 20}:
            dy = 0.055 if yi >= 0 else -0.065
            ax.annotate(
                rf"$d={int(di)}$",
                xy=(xi, yi),
                xytext=(xi, yi + dy),
                textcoords="data",
                fontsize=13,
                color=BLUE,
                ha="center",
                va="bottom" if yi >= 0 else "top",
            )

    ax.set_xlim(0.752, 0.858)
    ax.set_ylim(-0.45, 0.40)
    ax.set_xticks([0.76, 0.78, 0.80, 0.82, 0.84, 0.85])
    ax.set_xlabel(r"Held-out variance explained  $R^{2}_L$")
    ax.set_ylabel(r"Controlled $\rho\!\left(K_H,\;\mathrm{local\ OOF\ }R^{2}\right)$")
    _head(fig, ax, TITLE_CURV, SUB_CURV)
    ax.legend(frameon=True, fancybox=False, edgecolor="#e2e2e2", framealpha=0.94, loc="lower left")
    _footer(ax, FOOTER_CURV)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.70, bottom=0.18)
    return _save(fig, dest, "chart2_curvature_probe_performance")


def write_slide_charts(root: Path | None = None, dest: Path | None = None) -> list[Path]:
    _style()
    root = root or platonic_root()
    tables = load_frozen_tables(root)
    dest = dest or resolve_path(root, "outputs/geometry/physics_adaptive_dataset_curvature_probe_audit/figures/slides")
    cache = dest.parent.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    write_df(cache / "slide_chart_curve.csv", tables["assoc"], force=True)
    write_df(cache / "slide_chart_anchors_d16.csv", tables["d16"], force=True)
    write_df(cache / "slide_chart_variance.csv", tables["ve"], force=True)
    write_df(cache / "slide_chart_qpd_risk.csv", tables["qpd"], force=True)
    paths = []
    paths.extend(chart1_main_result(tables, dest))
    paths.extend(chart2_anchor_d16(tables, dest))
    paths.extend(chart1_evaluating_dimensionality(tables, dest))
    paths.extend(chart2_curvature_probe_performance(tables, dest))
    return paths


if __name__ == "__main__":
    for p in write_slide_charts():
        print(p)
