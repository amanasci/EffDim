"""Figure 1 for the ML4PS manuscript, from the JSONL records. No values typed by hand."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = "notebooks/.cache/"
def rows(f): return [json.loads(l) for l in open(C + f)]

# --- panel A: known surface ---------------------------------------------------------------
pf = [r for r in rows("09_fixture_probe_facing.jsonl") if r["row"] == "result" and r["label"] == "intrinsic_linear"]
dec = {}
for f in ("09_fixture_probe_decodability.jsonl", "09_fixture_probe_decodability_gamma2.jsonl"):
    for r in rows(f):
        if r["row"] == "partial" and r["label"] == "intrinsic_linear":
            dec.setdefault(r["gamma"], {})[r["column"]] = r["partial"]
pf.sort(key=lambda r: r["coupling_rho_H_tan_log_r"])
x = np.array([r["coupling_rho_H_tan_log_r"] for r in pf])
gam = [r["gamma"] for r in pf]
ser_A = [
    ("exact $\\|H_{\\mathrm{tan}}\\|$", [r["partials"]["exact_H_tan"]["partial"] for r in pf], "#000000", "o", "-"),
    ("decoder $\\|H_{\\mathrm{tan}}\\|$", [dec[g]["decoder_H_tan"] for g in gam], "#0072B2", "s", "--"),
    ("exact $\\|\\langle w_N,\\mathrm{II}\\rangle\\|$ (here the sphere term $\\sqrt{d}\\,|\\hat y-b_0|$)", [r["partials"]["pf_curv"]["partial"] for r in pf], "#009E73", "D", "-"),
    ("exact $\\|\\Delta\\|$ (Hessian mismatch)", [r["partials"]["hess_mismatch"]["partial"] for r in pf], "#999999", "v", ":"),
]
# --- panel B: physics ---------------------------------------------------------------------
ph = [r for r in rows("09_physics_probe_facing_split.jsonl") if r["row"] == "result"]
labels = ["mag_r", "photo_z", "smooth_fraction", "stellar_mass"]
ser_B = [  # (column, color, marker, filled, legend)
    ("H_tan_norm", "#0072B2", "s", True, "decoder $\\|H_{\\mathrm{tan}}\\|$"),
    ("pf_tan", "#009E73", "o", False, "decoder $\\|\\langle w_N,\\mathrm{II}^S\\rangle\\|$ (shape)"),
    ("pf_rad", "#D55E00", "^", False, "sphere term $\\sqrt{d}\\,|\\hat y-b_0|$"),
]
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4, "pdf.fonttype": 42})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.75), layout="constrained", gridspec_kw={"width_ratios": [1, 1.05]})
ax.axhline(0, color="#444444", lw=0.6)
for name, y, c, m, ls in ser_A:
    ax.plot(x, y, marker=m, color=c, ls=ls, label=name, markeredgecolor=c, markerfacecolor=c if m != "D" else c)
ax.set_xlabel("sample coupling $\\rho(\\|H_{\\mathrm{tan}}\\|, \\log r_k)$")
ax.set_ylabel("controlled partial vs local $R^2$")
ax.set_ylim(-0.9, 0.45); ax.invert_xaxis()
ax.set_title("(a) known surface, intrinsic-linear label", loc="left", fontsize=8)
# panel B
ypos = {}; k = 0
for lab in labels:
    for d in (16, 20):
        ypos[(lab, d)] = k; k += 1
bx.axvline(0, color="#444444", lw=0.6)
for i in range(0, len(labels) * 2, 2):
    bx.axhspan(i - 0.5, i + 1.5, color="#f2f2f2" if (i // 2) % 2 == 0 else "white", lw=0, zorder=0)
for col, c, m, filled, name in ser_B:
    xs = []; ys = []
    for r in ph:
        v = r["columns"][col]["multiscale"]["partial"]
        xs.append(v); ys.append(ypos[(r["label"], r["d"])])
    bx.scatter(xs, ys, marker=m, s=22, facecolors=c if filled else "white", edgecolors=c, linewidths=1.0, label=name, zorder=3)
bx.set_yticks(list(ypos.values()))
bx.set_yticklabels([f"{lab.replace('_', ' ')}, $d$={d}" for (lab, d) in ypos])
bx.invert_yaxis()
bx.set_xlabel("partial vs local $R^2$, multi-scale")
bx.set_xlim(-0.45, 0.5)
hA, lA = ax.get_legend_handles_labels(); hB, lB = bx.get_legend_handles_labels()
fig.legend(hA + hB, lA + lB, loc="outside lower center", ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2)
bx.set_title("(b) galaxies, decoder", loc="left", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(f"docs/latex/ml4ps/figures/fig1_probe_facing.{ext}", dpi=300)
print("saved; x (coupling) =", np.round(x, 3), "gammas", gam)
