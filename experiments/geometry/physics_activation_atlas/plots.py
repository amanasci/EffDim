"""Lightweight diagnostic plots for the atlas smoke report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def write_smoke_plots(out_root: Path) -> list[str]:
    import matplotlib.pyplot as plt

    plot_dir = out_root / "analyze" / "plots"
    written = []
    charts = json.loads((out_root / "charts" / "charts_meta.json").read_text())
    train = json.loads((out_root / "train" / "train_summary.json").read_text())
    priors = json.loads((out_root / "priors" / "prior_selection.json").read_text())
    overlaps = json.loads((out_root / "overlaps" / "overlaps.json").read_text())
    nerve = json.loads((out_root / "nerve" / "nerve.json").read_text())
    patch = json.loads((out_root / "patch_topology" / "patch_topology.json").read_text())
    boundary = json.loads((out_root / "boundary" / "boundary.json").read_text())
    curv = json.loads((out_root / "curvature" / "curvature.json").read_text())
    coords = json.loads((out_root / "coordinates" / "coordinates_meta.json").read_text())

    # chart supports
    fig, ax = plt.subplots()
    ax.bar(range(len(charts["support_sizes"])), charts["support_sizes"])
    ax.set_title("Chart support sizes")
    ax.set_xlabel("chart")
    ax.set_ylabel("n members (w>1e-6)")
    _savefig(fig, plot_dir / "chart_supports.png")
    written.append("chart_supports.png")

    # decoder vs pca
    fig, ax = plt.subplots()
    dec = [c["val_mse_decoder"] for c in train["charts"]]
    pca = [c["val_mse_pca"] for c in train["charts"]]
    x = np.arange(len(dec))
    ax.bar(x - 0.2, pca, 0.4, label="local PCA")
    ax.bar(x + 0.2, dec, 0.4, label="decoder")
    ax.legend()
    ax.set_title("Val reconstruction MSE")
    _savefig(fig, plot_dir / "decoder_vs_pca.png")
    written.append("decoder_vs_pca.png")

    # prior CF
    fig, ax = plt.subplots()
    ax.bar(
        ["std-G", "MLE-GMM", "CF-GMM"],
        [
            float("nan"),
            priors.get("mean_val_cf_mle", float("nan")),
            priors.get("mean_val_cf_cfmatched", float("nan")),
        ],
    )
    ax.set_title("Mean held-out CF distance")
    _savefig(fig, plot_dir / "prior_cf.png")
    written.append("prior_cf.png")

    # overlap masses
    fig, ax = plt.subplots()
    masses = [p["overlap_mass"] for p in overlaps.get("pairs", [])]
    ax.hist(masses, bins=20)
    ax.set_title("Overlap mass distribution")
    _savefig(fig, plot_dir / "overlap_mass.png")
    written.append("overlap_mass.png")

    # nerve filtration
    fig, ax = plt.subplots()
    fil = nerve.get("filtration", [])
    ax.plot([f["threshold"] for f in fil], [f["n_edges"] for f in fil], label="edges")
    ax.plot([f["threshold"] for f in fil], [f["n_triangles"] for f in fil], label="triangles")
    ax.invert_xaxis()
    ax.legend()
    ax.set_title("Nerve filtration (exploratory)")
    _savefig(fig, plot_dir / "nerve_filtration.png")
    written.append("nerve_filtration.png")

    # patch H1
    fig, ax = plt.subplots()
    excess = [p["excess_over_null_p90"] for p in patch.get("patches", [])]
    ax.hist(excess, bins=15)
    ax.axvline(0, color="k", ls="--")
    ax.set_title("Patch H1 excess over local null p90")
    _savefig(fig, plot_dir / "patch_h1.png")
    written.append("patch_h1.png")

    # boundary vs dtm corr
    fig, ax = plt.subplots()
    ax.scatter(
        [c["corr_with_dtm"] for c in boundary.get("charts", [])],
        [c["excess_vs_control"] for c in boundary.get("charts", [])],
    )
    ax.set_xlabel("corr(boundary, DTM)")
    ax.set_ylabel("excess vs control")
    ax.set_title("Boundary vs density confounding")
    _savefig(fig, plot_dir / "boundary_density.png")
    written.append("boundary_density.png")

    # jacobian cond vs recon
    fig, ax = plt.subplots()
    ax.scatter(
        [c["val_mse_decoder"] for c in train["charts"]],
        [c["median_condition"] for c in train["charts"]],
    )
    ax.set_xlabel("decoder val MSE")
    ax.set_ylabel("median Jacobian cond")
    ax.set_title("Jacobian condition vs reconstruction")
    _savefig(fig, plot_dir / "jac_vs_recon.png")
    written.append("jac_vs_recon.png")

    # local spectra rank95
    fig, ax = plt.subplots()
    ax.bar(range(len(coords["per_chart"])), [c["rank95"] for c in coords["per_chart"]])
    ax.set_title("Local PCA rank-95 by chart")
    _savefig(fig, plot_dir / "local_rank95.png")
    written.append("local_rank95.png")

    # curvature agreement
    fig, ax = plt.subplots()
    cos = [a.get("cosine_mean", float("nan")) for a in curv.get("overlap_agreements", []) if a.get("n_common", 0) > 0]
    if cos:
        ax.hist(cos, bins=10)
    ax.set_title("Overlap curvature cosine agreement")
    _savefig(fig, plot_dir / "curvature_agreement.png")
    written.append("curvature_agreement.png")

    # tangent disagreement
    fig, ax = plt.subplots()
    ax.hist([p["tangent_disagreement"] for p in overlaps.get("pairs", [])], bins=20)
    ax.set_title("Overlap tangent disagreement")
    _savefig(fig, plot_dir / "tangent_disagreement.png")
    written.append("tangent_disagreement.png")

    return written
