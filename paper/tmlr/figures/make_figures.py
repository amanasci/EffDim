"""Figures for the TMLR draft, from existing result files and records only (no refits, no geometry recompute).

    .venv/bin/python paper/tmlr/figures/make_figures.py

fig_ii_spectrum   <- curvature-experiment/results/ii-rank/ii_rank_<enc>_d16.json (median_normalised_spectrum);
                     the Gaussian reference curves repeat 12_ii_rank_run.random_reference (seed 0, D x 136 normal matrix).
fig_normal_readout <- notebooks/.cache/09_physics_normal_scaling_<run>.npz (S / S_model qq and dR2, alpha = 100).
fig_diag_scatter  <- notebooks/.cache/scaling/records/scaling__<enc>__main_xfit.jsonl (multiscale partials).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "notebooks" / ".cache"
RES = REPO / "curvature-experiment" / "results"
OUT = Path(__file__).resolve().parent

LABELS = ["mag_r", "photo_z", "smooth_fraction", "stellar_mass"]
OTHER = [("vit_base", "ViT-B", "#eb6834"), ("vit_large", "ViT-L", "#1baf7a"), ("clip_base", "CLIP-B", "#eda100"),
         ("convnext_base", "ConvNeXt-B", "#e87ba4")]
DINO = [("dinov3_vits16", "S"), ("dinov3_vits16plus", "S+"), ("dinov3_vitb16", "B"), ("dinov3_vitl16", "L"),
        ("dinov3_vith16plus", "H+"), ("dinov3_vit7b16", "7B")]
BLUES = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABCOL = {"mag_r": "#2a78d6", "photo_z": "#eb6834", "smooth_fraction": "#1baf7a", "stellar_mass": "#4a3aa7"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / f"{name}.pdf")


def gaussian_ref(D, m=136, seed=0):
    s = np.linalg.svd(np.random.default_rng(seed).standard_normal((D, m)), compute_uv=False)
    return np.sort(s)[::-1] / s.max()


def fig_ii_spectrum():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    j = np.arange(1, 137)
    for enc, name, col in OTHER:
        sp = json.load(open(RES / "ii-rank" / f"ii_rank_{enc}_d16.json"))["median_normalised_spectrum"]
        ax.plot(j, sp, color=col, lw=1.6, label=name)
    for (enc, size), col in zip(DINO, BLUES):
        sp = json.load(open(RES / "ii-rank" / f"ii_rank_{enc}_d16.json"))["median_normalised_spectrum"]
        ax.plot(j, sp, color=col, lw=1.6, label=f"DINOv3-{size}")
    for D, ls in ((384, ":"), (4096, "--")):
        ax.plot(j, gaussian_ref(D), color=MUTED, lw=1.2, ls=ls, label=f"Gaussian $D{{=}}{D}$")
    ax.axvline(34, color=MUTED, lw=0.6)
    ax.text(35, 0.93, "$m/4$", color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("singular value index $j$ (of $m=136$)")
    ax.set_ylabel("median $s_j/s_1$ over anchors")
    ax.legend(ncol=2, fontsize=6.5, loc="lower left")
    ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    save(fig, "fig_ii_spectrum")


def fig_normal_readout():
    runs = [("vit_base", "d16", "ViT-B"), ("vit_base", "d20", "ViT-B $d{=}20$"), ("dinov3_vitb16", "d16", "DINOv3-B"),
            ("clip_base", "d16", "CLIP-B"), ("convnext_base", "d16", "ConvNeXt-B"), ("vit_large", "d16", "ViT-L")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    width = 0.18
    for i, (enc, dd, name) in enumerate(runs):
        z = np.load(CACHE / f"09_physics_normal_scaling_{enc}_{dd}.npz")
        for k, lab in enumerate(LABELS):
            r = np.sqrt(z[f"{lab}:S_model:qq"] / z[f"{lab}:S:qq"])
            pos = i + (k - 1.5) * width
            q1, med, q3 = np.percentile(r, [25, 50, 75])
            a.plot([pos, pos], [q1, q3], color=LABCOL[lab], lw=2.2, solid_capstyle="round")
            a.plot(pos, med, "o", ms=3.5, color=LABCOL[lab], mec="white", mew=0.6,
                   label=lab.replace("_", " ") if i == 0 else None)
            b.plot(np.median(z[f"{lab}:S:dR2"]), np.median(z[f"{lab}:S_model:dR2"]), "o", ms=4, color=LABCOL[lab],
                   mec="white", mew=0.6)
    a.axhline(1.0, color=MUTED, lw=0.6, ls="--")
    a.set_xticks(range(len(runs)), [r[2] for r in runs], rotation=25, ha="right")
    a.set_ylabel(r"$\|q_c\|/\|p_c\|$ per anchor (IQR, median)")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=6.5, loc="lower center", ncol=2)
    a.set_title("(a) amplitude of the decoder quadratic", fontsize=8, loc="left")
    lim = 0.14
    b.plot([0, lim], [0, lim], color=MUTED, lw=0.6, ls="--")
    b.set_xlim(0, lim); b.set_ylim(0, lim)
    b.set_xlabel(r"median $\Delta R^2(t{=}1)$, data-side $S$")
    b.set_ylabel(r"median $\Delta R^2(t{=}1)$, $S_{\mathrm{model}}$")
    b.set_title("(b) local gain at $t=1$", fontsize=8, loc="left")
    save(fig, "fig_normal_readout")


def fig_diag_scatter():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    encs = [e for e, _, _ in OTHER] + [e for e, _ in DINO]
    for enc in encs:
        recs = [json.loads(l) for l in open(CACHE / "scaling" / "records" / f"scaling__{enc}__main_xfit.jsonl")]
        for r in recs:
            if r["row"] != "result":
                continue
            c = r["columns"]
            hy = c["hess_label"]["multiscale"]["partial"]
            ax.plot(hy, c["hess_mismatch_emp"]["multiscale"]["partial"], "o", ms=4, color=LABCOL[r["label"]], mec="white", mew=0.6)
            ax.plot(hy, c["hess_mismatch_dec"]["multiscale"]["partial"], "x", ms=3.5, color=LABCOL[r["label"]], mew=0.9)
    ax.plot([-0.7, 0.1], [-0.7, 0.1], color=MUTED, lw=0.6, ls="--")
    ax.set_xlim(-0.7, 0.1); ax.set_ylim(-0.7, 0.1)
    ax.set_xlabel(r"partial of $\|\mathrm{Hess}_M y\|_g$ alone")
    ax.set_ylabel("partial of the mismatch")
    handles = [Line2D([], [], marker="o", ls="", color=LABCOL[l], label=l.replace("_", " ")) for l in LABELS]
    handles += [Line2D([], [], marker="o", ls="", color=MUTED, label="emp (residual quadratic)"),
                Line2D([], [], marker="x", ls="", color=MUTED, label=r"dec ($\langle w_N,\mathrm{II}\rangle$)")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left")
    ax.grid(color="#e6e5e1", lw=0.5)
    save(fig, "fig_diag_scatter")


if __name__ == "__main__":
    fig_ii_spectrum()
    fig_normal_readout()
    fig_diag_scatter()
