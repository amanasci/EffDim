"""Figure 1 for the ML4PS manuscript: the counterfactual scaling of the probe-facing shape term (Appendix E records).
Median over anchors of the change in local R^2 relative to shape-flat (t=0) as the in-sphere shape quadratic is scaled
by t; t=1 is the manifold's own bending seen from the readout's normal direction, t=-1 its reversal. One thin line per
run (five encoders at d=16 plus ViT-B at d=20); grey dashed = random in-sphere direction with matched ||q_c||.
No values typed by hand."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # paper/
from records import PAPER, RECORDS  # noqa: E402
C = str(RECORDS) + "/"
RUNS = [("ViT-B $d{=}16$", "vit_base_d16"), ("ViT-B $d{=}20$", "vit_base_d20"), ("DINOv3", "dinov3_vitb16_d16"),
        ("CLIP-B", "clip_base_d16"), ("ConvNeXt-B", "convnext_base_d16"), ("ViT-L", "vit_large_d16")]
LABELS = [("mag_r", "magnitude"), ("photo_z", "photo-$z$"), ("smooth_fraction", "smooth fraction"), ("stellar_mass", "stellar mass")]
DEC, RND = "S_model", "random_qmatched"   # Appendix E columns: decoder in-sphere shape term; random orientation, matched ||q_c||
BLUE, GREY = "#0072B2", "#8c8c8c"
env = next(json.loads(l) for l in open(C + f"09_physics_normal_scaling_{RUNS[0][1]}.jsonl") if '"environment"' in l)
t = np.array(env["t_grid"]); i0 = int(np.where(t == 0.0)[0][0])
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "lines.linewidth": 0.9, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.2), layout="constrained", sharey=True)
summary = {}
for ax, (lab, title) in zip(axes, LABELS):
    ax.axhline(0, color="#444444", lw=0.6); ax.axvline(0, color="#dddddd", lw=0.6)
    for name, run in RUNS:
        z = np.load(C + f"09_physics_normal_scaling_{run}.npz")
        for var, color, ls in ((DEC, BLUE, "-"), (RND, GREY, "--")):
            curve = z[f"{lab}:{var}:r2_curve"]
            d = np.median(curve - curve[:, [i0]], axis=0)
            ax.plot(t, d, color=color, ls=ls, alpha=0.9 if var == DEC else 0.8)
            summary[(lab, run, var)] = d
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_xticks([-1, 0, 1, 2]); ax.set_xticklabels(["$-1$", "$0$", "$+1$", "$2$"])
    ax.set_xlim(-1.05, 2.05)
axes[0].set_ylabel("median change in local $R^2$")
fig.supxlabel("$t$: scale of the probe-facing shape term ($-1$ reversed, $0$ removed, $+1$ the manifold's own bending)", fontsize=8)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], color=GREY, ls="--", lw=1.4)],
           ["probe-facing shape term $t\\,\\langle w_S,\\mathrm{II}^S\\rangle(u,u)/2$, six runs",
            "random in-sphere direction, matched $\\|q_c\\|$"],
           loc="outside upper center", ncol=2, frameon=False, handlelength=2.0, columnspacing=1.5)
for ext in ("pdf", "png"):
    fig.savefig(str(Path(__file__).resolve().parent / f"fig1_intervention.{ext}"), dpi=300)
# checks against Appendix E: median dR2 at t=+1 and t=-1 per run
for (lab, run, var), d in sorted(summary.items()):
    if var == DEC: print(f"{run:20s} {lab:16s} dR2(+1)={d[t==1][0]:+.3f} dR2(-1)={d[t==-1][0]:+.3f}")
