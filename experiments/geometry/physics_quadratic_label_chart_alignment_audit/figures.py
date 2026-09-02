"""Publication figures from frozen QLCA tables plus audit rank/null overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _residualize(x: np.ndarray, Z: np.ndarray) -> np.ndarray:
    m = np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if int(m.sum()) < 12:
        return out
    xr = rankdata(x[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    b, *_ = np.linalg.lstsq(A, xr, rcond=None)
    out[m] = xr - A @ b
    return out


def _save(fig, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".pdf"))
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def write_figures(
    out: Path,
    anchor: pd.DataFrame,
    *,
    primary: dict,
    rank_summary: dict,
    haar_summary: dict | None = None,
    trunc: pd.DataFrame | None = None,
) -> list[str]:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Fig 1: paired held-out risks and gains
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    risk_cols = [("mse_L", "L"), ("mse_UQ", "UQ"), ("mse_BS", "BS"), ("mse_FQ", "FQ")]
    risk_data, risk_labs = [], []
    for c, lab in risk_cols:
        if c in anchor.columns:
            risk_data.append(anchor[c].dropna().to_numpy(float))
            risk_labs.append(lab)
    axes[0].boxplot(risk_data, tick_labels=risk_labs, showfliers=False)
    axes[0].set_ylabel("Held-out MSE")
    axes[0].set_title("Paired held-out risk")
    gain_cols = [("delta_Q", r"$\Delta_Q$"), ("delta_BS", r"$\Delta_{BS}$"), ("delta_FQ", r"$\Delta_{FQ}$")]
    gdata, glabs = [], []
    for c, lab in gain_cols:
        if c in anchor.columns:
            gdata.append(anchor[c].dropna().to_numpy(float))
            glabs.append(lab)
    axes[1].boxplot(gdata, tick_labels=glabs, showfliers=False)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel("Gain vs L (MSE)")
    axes[1].set_title("Confirmatory / secondary gains")
    fig.suptitle("Figure 1 — primary and secondary held-out risk (frozen QLCA)", fontsize=10)
    fig.tight_layout()
    _save(fig, figdir / "fig1_paired_gains")
    written.append(str(figdir / "fig1_paired_gains.pdf"))

    # Fig 2: controlled KH vs delta_Q
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    Z = np.column_stack(
        [
            anchor["log_knn_radius"].to_numpy(float),
            anchor["local_label_variance"].to_numpy(float),
            anchor["local_evaluation_count"].to_numpy(float),
        ]
    )
    x = _residualize(anchor.K_H_cross.to_numpy(float), Z)
    y = _residualize(anchor.delta_Q.to_numpy(float), Z)
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=10, alpha=0.5, c="0.15", linewidths=0)
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.axvline(0.0, color="0.6", lw=0.7)
    rho = primary.get("rho_KH_delta_Q", float("nan"))
    lo = primary.get("delta_Q_ci_lo", float("nan"))
    hi = primary.get("delta_Q_ci_hi", float("nan"))
    ax.set_xlabel(r"controlled rank residual of $K_H^{\mathrm{cross}}$")
    ax.set_ylabel(r"controlled rank residual of $\Delta_Q$")
    ax.set_title(rf"Figure 2 — primary 2: $\rho_{{\mathrm{{ctl}}}}\approx{rho:.3f}$  (median $\Delta_Q$ CI [{lo:.3f},{hi:.3f}])")
    fig.tight_layout()
    _save(fig, figdir / "fig2_KH_vs_deltaQ")
    written.append(str(figdir / "fig2_KH_vs_deltaQ.pdf"))

    # Fig 3: UQ vs BS + alignment nulls + rank
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    m = np.isfinite(anchor.delta_Q) & np.isfinite(anchor.delta_BS)
    axes[0].scatter(anchor.delta_Q[m], anchor.delta_BS[m], s=10, alpha=0.45, c="0.15", linewidths=0)
    lim = np.nanpercentile(np.concatenate([anchor.delta_Q[m], anchor.delta_BS[m]]), [2, 98])
    axes[0].plot(lim, lim, "k--", lw=0.8, label="identity")
    axes[0].set_xlabel(r"$\Delta_Q$ (UQ)")
    axes[0].set_ylabel(r"$\Delta_{BS}$")
    r_med = rank_summary.get("median_r_original", float("nan"))
    r95 = rank_summary.get("median_r95", float("nan"))
    frac = rank_summary.get("median_rank_fraction_original", float("nan"))
    axes[0].set_title(rf"UQ vs BS  (impl. $r_{{\mathrm{{med}}}}={r_med:.0f}$, $r_{{95}}={r95:.0f}$, $r/136={frac:.2f}$)")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].hist(anchor.A_B.dropna(), bins=28, color="0.35", density=True, label=r"observed $A_B$")
    axes[1].axvline(1.0, color="C1", ls="--", label="isotropic ≈ 1")
    if haar_summary and np.isfinite(haar_summary.get("null_median", np.nan)):
        axes[1].axvline(haar_summary["null_median"], color="C0", ls=":", label="Haar-spectrum null")
    axes[1].axvline(float(np.nanmedian(anchor.A_B)), color="k", lw=1.0, label="observed median")
    axes[1].set_xlabel(r"$A_B$")
    axes[1].set_title("Hessian–curvature alignment")
    axes[1].legend(fontsize=7)
    fig.suptitle("Figure 3 — chart constraint vs alignment (rank annotated)", fontsize=10)
    fig.tight_layout()
    _save(fig, figdir / "fig3_BS_vs_UQ_alignment")
    written.append(str(figdir / "fig3_BS_vs_UQ_alignment.pdf"))

    # Secondary audit-only singular spectrum
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    if trunc is not None and "energy_cdf" not in trunc.columns:
        pass
    # plot median svals if present in rank_summary
    svals = rank_summary.get("median_svals_mean_split")
    if svals:
        s = np.asarray(svals, dtype=float)
        energy = np.cumsum(s * s) / max(float(np.sum(s * s)), 1e-18)
        ax.plot(np.arange(1, len(s) + 1), energy, color="0.15")
        ax.axhline(0.90, color="C1", ls="--", lw=0.8, label="90%")
        ax.axhline(0.95, color="C2", ls="--", lw=0.8, label="95%")
        ax.axhline(0.99, color="C3", ls="--", lw=0.8, label="99%")
        ax.axvline(r_med, color="0.4", ls=":", label=rf"original $r$={r_med:.0f}")
        ax.set_xlabel("singular index")
        ax.set_ylabel("cumulative squared energy")
        ax.set_title("Secondary: median $B^S$ singular-spectrum energy")
        ax.legend(fontsize=8)
        ax.set_xlim(1, min(136, len(s)))
    fig.tight_layout()
    _save(fig, figdir / "figS_singular_spectrum")
    written.append(str(figdir / "figS_singular_spectrum.pdf"))
    return written
