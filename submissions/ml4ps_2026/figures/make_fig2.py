"""Compose Figure 2 from frozen QLCA / audit tables. Does not recompute science."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


ROOT = Path(__file__).resolve().parents[3]
QLCA = ROOT / "paper/curvature_neurreps/audit_outputs/quadratic_label_chart_alignment"
AUDIT = ROOT / "paper/curvature_neurreps/audit_outputs/quadratic_label_chart_alignment_audit"
OUT = Path(__file__).resolve().parent


def _resid(x, Z):
    m = np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    out = np.full(x.shape, np.nan)
    xr = rankdata(x[m]).astype(float)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    b, *_ = np.linalg.lstsq(A, xr, rcond=None)
    out[m] = xr - A @ b
    return out


def main() -> None:
    anc = pd.read_csv(QLCA / "anchor_risks.csv")
    primary = json.loads((QLCA / "primary_inference.json").read_text())
    align = json.loads((AUDIT / "alignment_tests.json").read_text())
    trunc = json.loads((AUDIT / "truncated_bs_summary.json").read_text())
    haar = float(align["haar_all"]["null_median"])
    iso = float(align["isotropic_all"]["null_median"])
    matched = float(align["matched_anchor_spectrum"]["null_median"])
    ab_med = float(align["haar_all"]["observed_median"])
    rho = float(primary["rho_KH_delta_Q"])
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.45))

    ax = axes[0, 0]
    data = [anc[c].dropna().to_numpy(float) for c in ("delta_Q", "delta_BS")]
    ax.boxplot(data, tick_labels=[r"UQ $\Delta_Q$", r"cap-48 $B^S$"], showfliers=False, widths=0.55)
    ax.axhline(0.0, color="0.55", lw=0.7)
    ax.set_ylabel("held-out MSE gain vs $L$")
    ax.set_title("(a) Confirmatory / secondary gains")

    ax = axes[0, 1]
    Z = np.column_stack(
        [
            anc.log_knn_radius.to_numpy(float),
            anc.local_label_variance.to_numpy(float),
            anc.local_evaluation_count.to_numpy(float),
        ]
    )
    x = _resid(anc.K_H_cross.to_numpy(float), Z)
    y = _resid(anc.delta_Q.to_numpy(float), Z)
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=6, alpha=0.4, c="0.15", linewidths=0)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axvline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(r"controlled residual of $K_H^{\mathrm{cross}}$")
    ax.set_ylabel(r"controlled residual of $\Delta_Q$")
    ax.set_title(rf"(b) Primary: $\rho_{{\mathrm{{ctl}}}}\approx{rho:+.3f}$")

    ax = axes[1, 0]
    ax.hist(anc.A_B.dropna(), bins=24, color="0.35", density=True, label=r"observed $A_B$")
    ax.axvline(ab_med, color="k", lw=1.1, label=rf"median ${ab_med:.2f}$")
    ax.axvline(iso, color="C1", ls="--", lw=1.0, label=rf"isotropic ${iso:.2f}$")
    ax.axvline(haar, color="C0", ls=":", lw=1.2, label=rf"Haar / matched ${haar:.2f}$")
    _ = matched  # documented in CLAIM_SOURCE_MAP; Haar and matched agree to 0.99
    ax.set_xlabel(r"$A_B$")
    ax.set_ylabel("density")
    ax.set_title("(c) Audit: orientation alignment")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 1]
    rules = [
        ("original_rule", "cap 48"),
        ("e90", r"$r_{90}$"),
        ("e95", r"$r_{95}$"),
        ("e99", r"$r_{99}$"),
    ]
    ranks = [float(trunc[key]["median_r"]) for key, _ in rules]
    fracs = [100.0 * float(trunc[key]["median_frac_UQ"]) for key, _ in rules]
    labels = [lab for _, lab in rules]
    ax.plot(ranks, fracs, "o-", color="0.15", ms=5)
    for r, f, lab in zip(ranks, fracs, labels):
        ax.annotate(lab, (r, f), textcoords="offset points", xytext=(4, -9), fontsize=7)
    ax.set_xlabel("geometry-only retained rank")
    ax.set_ylabel("% of UQ gain (median ratio)")
    ax.set_ylim(90, 102)
    ax.set_title("(d) Audit: energy truncation")
    ax.axhline(100, color="0.7", lw=0.6)

    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "fig2_quadratic.pdf")
    fig.savefig(OUT / "fig2_quadratic.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "fig2_quadratic.pdf")


if __name__ == "__main__":
    main()
