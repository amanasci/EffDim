"""Audit figures. Unreliable ranks stay visible and are marked."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, figdir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figdir / name, dpi=160)
    plt.close(fig)


def write_figures(out: Path) -> list[str]:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    names = []

    side = pd.read_csv(out / "discovery_curves_side_by_side.csv") if (out / "discovery_curves_side_by_side.csv").exists() else pd.DataFrame()
    if len(side):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(side.d, side.raw_discovery_local_r2, "o-", label="raw local_r2")
        ax.plot(side.d, side.frozen_discovery_control_local_r2, "s-", label="frozen-control local_r2")
        ax.plot(side.d, side.raw_catalog_mag, "o--", label="raw catalog mag")
        ax.plot(side.d, side.harmonized_control_catalog_mag, "s--", label="harmonized-control catalog mag")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("d")
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_title("Discovery curves: local_r2 vs catalog mag")
        ax.legend(fontsize=8)
        _save(fig, figdir, "01_discovery_curves_side_by_side.png")
        names.append("01_discovery_curves_side_by_side.png")

    fact = pd.read_csv(out / "factorial_discovery_correlations.csv") if (out / "factorial_discovery_correlations.csv").exists() else pd.DataFrame()
    if len(fact):
        sub = fact[fact.scope == "intersection"]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(sub.d, sub.rho_oldK_oldy, "o-", label=r"$\rho(K^{old}, y^{old})$")
        ax.plot(sub.d, sub.rho_newK_oldy, "s--", label=r"$\rho(K^{new}, y^{old})$")
        ax.plot(sub.d, sub.rho_oldK_newy, "o--", label=r"$\rho(K^{old}, y^{new})$")
        ax.plot(sub.d, sub.rho_newK_newy, "s-", label=r"$\rho(K^{new}, y^{new})$")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("d")
        ax.set_ylabel(r"raw Spearman $\rho$")
        ax.set_title("Factorial discovery correlations")
        ax.legend(fontsize=8)
        _save(fig, figdir, "02_factorial_correlations.png")
        names.append("02_factorial_correlations.png")

    sizes = pd.read_csv(out / "association_sample_sizes.csv") if (out / "association_sample_sizes.csv").exists() else pd.DataFrame()
    if len(sizes):
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        labels = [f"{r.dataset_id}:{r.label}" for _, r in sizes.iterrows()]
        ax.barh(labels, sizes.valid_labelled_anchors, color=["#c44" if u else "#47a" for u in sizes.underpowered])
        ax.axvline(64, color="k", ls="--", lw=0.8, label="min valid anchors")
        ax.set_xlabel("valid labelled curvature anchors")
        ax.set_title("Anchor-level sample sizes")
        _save(fig, figdir, "03_anchor_sample_sizes.png")
        names.append("03_anchor_sample_sizes.png")

    rel = pd.read_csv(out / "high_rank_reliability_sensitivity.csv") if (out / "high_rank_reliability_sensitivity.csv").exists() else pd.DataFrame()
    if len(rel):
        ov = rel[rel.cutoff == "overlay"]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for (ds, lab), g in ov.groupby(["dataset_id", "label"]):
            if lab not in ("smooth_fraction", "mag_r", "mag_r_desi", "photo_z"):
                continue
            ax.plot(g.peak_d, g.R_H, "o-", label=f"{ds}:{lab}", alpha=0.8)
        ax.axhline(0.2, color="0.3", ls="--", lw=0.8, label="frozen R_H=0.2")
        for c, ls in ((0.4, ":"), (0.5, ":"), (0.6, ":")):
            ax.axhline(c, color="0.6", ls=ls, lw=0.6)
        ax.set_xlabel("d")
        ax.set_ylabel(r"median $R_H$")
        ax.set_title("Reliability overlay (ranks remain visible)")
        ax.legend(fontsize=7)
        _save(fig, figdir, "04_reliability_overlay.png")
        names.append("04_reliability_overlay.png")

    return names
