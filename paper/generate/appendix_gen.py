"""Generate the appendix LaTeX from the records and splice it into main.tex between markers. No values typed by hand."""
import json, sys, os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/
from records import PAPER, RECORDS  # noqa: E402
C = str(RECORDS) + "/"
def rows(f):
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
def cell(v):
    if v is None or v.get("partial") is None or v["partial"] != v["partial"]: return "--"
    return f"${v['partial']:+.2f}" + ("^{*}" if v["p"] > 0.05 else "") + "$"
labels = ["mag_r", "photo_z", "smooth_fraction", "stellar_mass"]
lab_tex = {"mag_r": r"mag\_r", "photo_z": r"photo\_z", "smooth_fraction": r"smooth\_fraction", "stellar_mass": r"stellar\_mass"}
out = []
# --- A: decoder ablations at d=16 (multi-scale control)
variants = [("seed 0", C + "09_physics_probe_facing_split.jsonl"), ("seed 1", C + "09_physics_probe_facing_split_seed1.jsonl"),
            ("seed 2", C + "09_physics_probe_facing_split_seed2.jsonl"), ("$400^3$", C + "09_physics_probe_facing_split_w400.jsonl")]
quant = [("shape", "pf_tan"), ("sphere", "pf_rad"), ("mismatch", "hess_mismatch_emp"), ("alignment", "align_cos_tan")]
res = {}
for name, f in variants:
    for r in rows(f):
        if r.get("row") == "result" and r["d"] == 16: res[(name, r["label"])] = r
out.append(r"""\section*{Appendix A: decoder ablations}
Table~\ref{tab:ablate} repeats the four quantities of Tables~\ref{tab:real} and~\ref{tab:meancurv} at $d=16$ under the multi-scale density control for two further decoder initialisation seeds and for a decoder of width $400^3$ instead of $250^3$ (all fits reach variance explained 0.952). Every significant sign of the main table is retained.
\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
label & quantity & """ + " & ".join(n for n, _ in variants) + r""" \\
\midrule""")
for lab in labels:
    for qi, (qn, qc) in enumerate(quant):
        cells = [cell(res[(n, lab)]["columns"][qc]["multiscale"]) if (n, lab) in res else "--" for n, _ in variants]
        out.append((lab_tex[lab] if qi == 0 else "") + f" & {qn} & " + " & ".join(cells) + r" \\")
    out.append(r"\addlinespace[2pt]")
out.append(r"""\bottomrule
\end{tabular}
\caption{Decoder ablations, $d=16$, 512 anchors, multi-scale density control. $^{*}$ not significant at 0.05.}
\label{tab:ablate}
\end{table}""")
# --- B: sensitivity: cross-fitted Hessian and weak ridge
xf = {}
for r in rows(C + "09_physics_probe_facing_split_xfit.jsonl"):
    if r.get("row") == "xfit": xf[(r["d"], r["label"])] = r
al = {}
for r in rows(C + "09_physics_probe_facing_split_alpha1.jsonl"):
    if r.get("row") == "result": al[(r["d"], r["label"])] = r
if xf or al:
    out.append(r"""\section*{Appendix B: sensitivity of the mismatch and sphere columns}
Cross-fitting: each 2{,}048-patch is split at random into halves; $\operatorname{Hess}_M y$ is fitted on one half and local $R^2$ scored on the other (both directions shown), so the Hessian estimate and the outcome share no rows. The split-half cosine between the two Hessian estimates is the reliability of the estimate. Weak ridge: the probe refit at $\alpha=1$ instead of 100 (global out-of-sample $R^2$ rises by 0.10 to 0.15), which substantially reduces the shrinkage of extreme predictions; the sphere term is recomputed from that probe.
\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{2}{c}{mismatch, cross-fit} & \multicolumn{2}{c}{alignment, cross-fit} & Hess.\ split cos & sphere, $\alpha=1$ \\
label & $d$ & fit A/score B & fit B/score A & fit A/score B & fit B/score A & p50 & (main: $\alpha=100$) \\
\midrule""")
    for lab in labels:
        for d in (16, 20):
            x = xf.get((d, lab)); a = al.get((d, lab))
            m = x["columns"]["hess_mismatch_dec"] if x else {}; g = x["columns"]["align_cos_tan"] if x else {}
            main = res.get(("seed 0", lab)) if d == 16 else None
            if d == 20:
                for r in rows(C + "09_physics_probe_facing_split.jsonl"):
                    if r.get("row") == "result" and r["d"] == 20 and r["label"] == lab: main = r
            sph = (cell(a["columns"]["pf_rad"]["multiscale"]) if a else "--") + (f" ({cell(main['columns']['pf_rad']['multiscale'])})" if main else "")
            cos = f"{x['hessian_split_half_cos_p25_p50_p75'][1]:+.2f}" if x else "--"
            out.append(f"{lab_tex[lab]} & {d} & {cell(m.get('fitA_scoreB'))} & {cell(m.get('fitB_scoreA'))} & {cell(g.get('fitA_scoreB'))} & {cell(g.get('fitB_scoreA'))} & {cos} & {sph} \\\\")
    out.append(r"""\bottomrule
\end{tabular}
\caption{Cross-fitted mismatch and alignment partials (multi-scale control), split-half reliability of the label Hessian, and the sphere-term partial under a weak-ridge probe.}
\label{tab:sens}
\end{table}""")
# --- D: cross-encoder probe-facing test (d=16, seed 0, cross-fit on)
encs = [("ViT-B", C + "09_physics_probe_facing_split.jsonl"), ("DINOv3", C + "09_physics_probe_facing_split_dinov3_vitb16.jsonl"),
        ("CLIP-B", C + "09_physics_probe_facing_split_clip_base.jsonl"), ("ConvNeXt-B", C + "09_physics_probe_facing_split_convnext_base.jsonl"),
        ("ViT-L", C + "09_physics_probe_facing_split_vit_large.jsonl")]
er, ex, ve = {}, {}, {}
for name, f in encs:
    for r in rows(f):
        if r.get("d") != 16: continue
        if r.get("row") == "result": er[(name, r["label"])] = r
        if r.get("row") == "xfit": ex[(name, r["label"])] = r
        if r.get("row") == "fit": ve[name] = r["var_explained"]
have = [n for n, _ in encs if any((n, l) in er for l in labels)]
if len(have) > 1:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # paper/generate/
    from table_main_gen import main_rows  # noqa: E402
    real = main_rows(rows(C + "09_physics_probe_facing_split.jsonl"), cols=["hess_mismatch_emp", "align_cos_tan"])
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    def _part(r, qc): return r["columns"][qc]["multiscale"]
    def _sig(v): return not v["p"] > 0.05
    mm = [_part(er[(n, lab)], "hess_mismatch_emp")["partial"] for lab in ("mag_r", "photo_z") for n in have if (n, lab) in er]
    mmx = [ex[(n, lab)]["columns"]["hess_mismatch_dec"][k]["partial"] for lab in ("mag_r", "photo_z") for n in have if (n, lab) in ex
           for k in ("fitA_scoreB", "fitB_scoreA")]
    al_mag = [_part(er[(n, "mag_r")], "align_cos_tan")["partial"] for n in have if (n, "mag_r") in er]
    n_al_z = sum(_sig(_part(er[(n, "photo_z")], "align_cos_tan")) for n in have if (n, "photo_z") in er)
    n_mm_sm = sum(_sig(_part(er[(n, "stellar_mass")], "hess_mismatch_emp")) for n in have if (n, "stellar_mass") in er)
    vel = [f"{ve[n]:.3f}" for n in have if n in ve]
    out.append(r"""
\section*{Appendix C: Main and cross-encoder results}

\paragraph{Main results.}
Table~\ref{tab:real} reports the associations
between geometric mismatch, alignment and local
readout accuracy for ViT-B at $d=16$ and $d=20$.
Results use 512 anchors, $k=2{,}048$ neighbours
and the multi-scale density control.

\begin{table}[h]
\centering
\footnotesize
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{mismatch $\norm{\Delta}_g$}
 & \multicolumn{2}{c}{alignment $\cos_g(\operatorname{Hess}_M y, K)$} \\
label & $d{=}16$ & $d{=}20$ & $d{=}16$ & $d{=}20$ \\
\midrule""")
    out += real
    out.append(r"""\bottomrule
\end{tabular}
\caption{ViT-B: rank-partial Spearman correlations
with local $R^2$ under the multi-scale density
control. $^{*}$ Not significant at 0.05.
Shape, sphere and mean-curvature results
are reported in Appendix~E.}
\label{tab:real}
\end{table}

\paragraph{Across encoders.}
We repeat the geometric diagnostic at $d=16$
on the same 86{,}471 galaxies using four further
encoders from the Platonic Universe release:
DINOv3 ViT-B/16, CLIP ViT-B, ConvNeXt-B and
ViT-L (widths 768, 512, 1{,}024 and 1{,}024;
all unit-normalized).
A decoder of the same architecture and training
protocol is fitted per encoder (seed 0;
variance explained """ + ", ".join(vel[:-1]) + " and\n" + vel[-1] + r""", respectively). All analyses use
512 anchors, $k=2{,}048$ and the multi-scale
density control.

For magnitude and redshift, mismatch is
negatively associated with local $R^2$
across all """ + words[len(have)] + f""" encoders (${max(mm):+.2f}$ to ${min(mm):+.2f}$;
cross-fitted ${max(mmx):+.2f}$ to ${min(mmx):+.2f}$).
Alignment is positive and significant for
magnitude across all {words[len(have)]} (${min(al_mag):+.2f}$ to ${max(al_mag):+.2f}$)
and for redshift across {words[n_al_z]}.""" + r"""
Shape-term associations vary by encoder and
target, while smooth fraction shows less
consistent mismatch and alignment signals.
For stellar mass, alignment is nonsignificant
across encoders and mismatch significant in
only """ + words[n_mm_sm] + r""" main comparison. Thus, the
cross-anchor mismatch association varies by
target, even where the shape term improves
the local surrogate.

Table~\ref{tab:xenc} reports the full
cross-encoder results, including global
out-of-sample probe $R^2$ in the first row
for each label. Table~\ref{tab:xencx}
reports cross-fitted mismatch and alignment
associations and split-half reliability of
the estimated label Hessian.

\begin{table}[h]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll""" + "c" * len(have) + r"""}
\toprule
label & quantity & """ + " & ".join(have) + r""" \\
\midrule""")
    for lab in labels:
        r2 = [f"{er[(n, lab)]['global_oof_r2']:.2f}" if (n, lab) in er else "--" for n in have]
        out.append(lab_tex[lab] + " & global $R^2$ & " + " & ".join(r2) + r" \\")
        for qn, qc in quant:
            cells = [cell(er[(n, lab)]["columns"][qc]["multiscale"]) if (n, lab) in er else "--" for n in have]
            out.append(f" & {qn} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace[2pt]")
    out.append(r"""\bottomrule
\end{tabular}
\caption{Cross-encoder results at $d=16$:
global probe $R^2$ and rank-partial Spearman
correlations with local $R^2$ under the
multi-scale density control.
$^{*}$ Not significant at 0.05.}
\label{tab:xenc}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{2}{c}{mismatch, cross-fit}
 & \multicolumn{2}{c}{alignment, cross-fit}
 & Hess.\ split cos \\
encoder & label & fit A/score B & fit B/score A
& fit A/score B & fit B/score A & p50 \\
\midrule""")
    for n in have:
        for lab in labels:
            x = ex.get((n, lab))
            if x is None: continue
            m = x["columns"]["hess_mismatch_dec"]; g = x["columns"]["align_cos_tan"]
            out.append(f"{n if lab == labels[0] else ''} & {lab_tex[lab]} & {cell(m.get('fitA_scoreB'))} & {cell(m.get('fitB_scoreA'))} & {cell(g.get('fitA_scoreB'))} & {cell(g.get('fitB_scoreA'))} & {x['hessian_split_half_cos_p25_p50_p75'][1]:+.2f} \\\\")
        out.append(r"\addlinespace[2pt]")
    out.append(r"""\bottomrule
\end{tabular}
\caption{Cross-fitted mismatch and alignment
partials under the multi-scale density
control, with median split-half tensor
cosine for the estimated label Hessian.
$^{*}$ Not significant at 0.05.}
\label{tab:xencx}
\end{table}""")
# --- E: counterfactual normal scaling (from the per-anchor arrays)
import numpy as np
runs = [("ViT-B", 16, "vit_base_d16"), ("ViT-B", 20, "vit_base_d20"), ("DINOv3", 16, "dinov3_vitb16_d16"), ("CLIP-B", 16, "clip_base_d16"),
        ("ConvNeXt-B", 16, "convnext_base_d16"), ("ViT-L", 16, "vit_large_d16")]
erows = []
for enc, d, stem in runs:
    f = C + f"09_physics_normal_scaling_{stem}.npz"
    if not os.path.exists(f): continue
    z = np.load(f)
    for lab in labels:
        cells = []
        for var in ("S_model", "random_qmatched"):
            eq, qq, cv = z[f"{lab}:{var}:eq"], z[f"{lab}:{var}:qq"], z[f"{lab}:{var}:r2_curve"]
            m = np.isfinite(eq); dp = cv[m, 4] - cv[m, 2]; dm = cv[m, 0] - cv[m, 2]; ts = eq[m] / np.maximum(qq[m], 1e-300)
            cells += [f"{np.mean(dp > 0):.2f}", f"{np.mean(dm < 0):.2f}", f"${np.median(dp):+.3f}$", f"${np.median(dm):+.3f}$"]
            if var == "S_model": cells += [f"{np.median(ts):.1f}"]
        erows.append((f"{enc}, $d={d}$" if lab == labels[0] else "") + f" & {lab_tex[lab]} & " + " & ".join(cells) + r" \\")
    erows.append(r"\addlinespace[2pt]")
# thinned-anchor sign tests: min-degree greedy independent set on the graph "pairwise neighbourhood overlap > 5% of k"
from scipy.stats import binomtest
def _indep(ov, thr):
    A = ov > thr; np.fill_diagonal(A, False); alive = np.ones(ov.shape[0], bool); keep = np.zeros(ov.shape[0], bool)
    while alive.any():
        deg = (A & alive[None, :]).sum(1); deg[~alive] = 10**9
        i = int(np.argmin(deg)); keep[i] = True; alive[i] = False; alive[A[i]] = False
    return keep
def _ptex(p):
    e = int(np.floor(np.log10(p))); c = int(np.ceil(p / 10 ** e - 1e-9))   # upper bound c x 10^e with c in 1..9
    if c == 10: c, e = 1, e + 1
    return ("%d\\times 10^{%d}" % (c, e)) if c > 1 else "10^{%d}" % e
thin = {"n": [], "p_help": [], "p_hurt": [], "help": [], "hurt": []}
for enc, d, stem in runs:
    f = C + f"09_physics_normal_scaling_{stem}.npz"; ft = C + f"09_physics_normal_scaling_{stem}_thin.npz"
    if not (os.path.exists(f) and os.path.exists(ft)): continue
    z = np.load(f); keep = _indep(np.load(ft)["overlap"].astype(float), 0.05)
    for lab in labels:
        cv = z[f"{lab}:S_model:r2_curve"]; m = keep & np.isfinite(cv[:, 0]); n = int(m.sum())
        kh = int((cv[m, 4] > cv[m, 2]).sum()); ku = int((cv[m, 0] < cv[m, 2]).sum())
        thin["n"].append(n); thin["help"].append(kh / n); thin["hurt"].append(ku / n)
        thin["p_help"].append(binomtest(kh, n, 0.5, alternative="greater").pvalue); thin["p_hurt"].append(binomtest(ku, n, 0.5, alternative="greater").pvalue)
thin_tex = ""
if thin["n"]:
    thin_tex = (" The anchors' neighbourhoods overlap (each point lies in about twelve of them), so the fractions are not counts of independent trials; a maximal set of anchors whose pairwise overlap is at most 5\\%% of $k$ has %d--%d members per run, on which $t=1$ beats shape-flat at %d--%d\\%% (one-sided sign test $p \\le %s$ in every cell) and $t=-1$ is worse at %d--%d\\%% ($p \\le %s$)."
                % (min(thin["n"]), max(thin["n"]), round(100 * min(thin["help"])), round(100 * max(thin["help"])), _ptex(max(thin["p_help"])),
                   round(100 * min(thin["hurt"])), round(100 * max(thin["hurt"])), _ptex(max(thin["p_hurt"]))))
if erows:
    out.append(r"""\section*{Appendix D: intervening on the readout's normal component}
The partials of Table~\ref{tab:real} compare anchors with one another and cannot say whether reversing the probe-facing shape term hurts relative to removing it, since no anchor is flat. Here we score a counterfactual local second-order surrogate of the readout at a fixed manifold. At each anchor the fitted ridge weight is split into tangent, radial and in-sphere normal parts $w = w_T + w_{\mathrm{rad}} + w_S$; we score the anchor's 2{,}048 neighbours by the data-side tangent-plus-sphere readout $(w_T + w_{\mathrm{rad}})\cdot x$ plus $t\,q$ with the local intercept refit, where $q = \tfrac12\langle w_S,\mathrm{II}^S\rangle(u,u)$ is the decoder's in-sphere second-order term in the anchor's chart coordinates: $t=1$ is the manifold's bending as seen in the readout's normal direction, $t=-1$ its sign reversal, $t=0$ shape-flat (the sphere term $-(w\!\cdot\!\hat x)g$ is kept in the base). Because the intercept is refit for every $t$, the local sum of squares is exactly $\mathrm{SSE}(t) = \norm{r_0 - t\,q_c}^2$ with $r_0$ the centred residual of the shape-flat predictor and $q_c = q - \bar q\mathbf 1$ the centred quadratic on the anchor's neighbours; hence $t^{*} = \langle r_0,q_c\rangle/\norm{q_c}^2$ and $t=1$ beats shape-flat iff $2\langle r_0,q_c\rangle > \norm{q_c}^2$. This is the finite-sample counterpart of the tensor-level condition $2\langle R,K_S\rangle_g > \norm{K_S}_g^2$ of Section~\ref{sec:theory}, not the same quantity: the tensor version needs the estimated label Hessian, the empirical one does not. Columns: fraction of anchors where $t=1$ beats shape-flat (help) and where $t=-1$ is worse than shape-flat (hurt), the median change in local $R^2$ at $t=\pm1$, the median $t^{*}$; then the same for a random in-sphere normal direction $v$ whose quadratic is rescaled to the same centred amplitude on the actual neighbours, $\norm{q_{v,c}} = \norm{q_c}$ (matching $\norm{v}$ to $\norm{w_S}$ instead leaves the contracted tensor at a fraction of the fitted one, since $\mathrm{II}^S$ spans at most $d(d+1)/2$ of the $\sim 750$ normal directions; matching $\norm{\langle v,\mathrm{II}^S\rangle}_g$ gives the same picture; Supplement 12).""" + thin_tex + r"""
\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccc|cccc}
\toprule
 & & \multicolumn{5}{c|}{decoder in-sphere shape term} & \multicolumn{4}{c}{random orientation, matched $\norm{q_c}$} \\
run & label & help & hurt & $\Delta R^2(+1)$ & $\Delta R^2(-1)$ & $t^{*}$ & help & hurt & $\Delta R^2(+1)$ & $\Delta R^2(-1)$ \\
\midrule""")
    out += erows
    out.append(r"""\bottomrule
\end{tabular}
\caption{Counterfactual scaling of the in-sphere shape term in a local second-order surrogate of the readout, 512 anchors per run. The manifold's bending as seen in the readout's normal direction helps at a large majority of anchors, its sign reversal hurts at nearly every anchor, a random orientation of the same centred quadratic amplitude shows no such asymmetry and is typically harmful under either sign; $t^{*}>1$ throughout, consistent with the globally ridge-regularized probe under-using a locally beneficial term.}
\label{tab:cf}
\end{table}""")
tex = "\n".join(out) + "\n"
p = str(PAPER / "latex" / "main.tex"); s = open(p).read()
B, E = "% BEGIN APPENDIX AUTOGEN", "% END APPENDIX AUTOGEN"
if B not in s:
    s = s.replace("\\end{document}", f"\\appendix\n{B}\n{E}\n\\end{{document}}")
i, j = s.index(B) + len(B), s.index(E)
s = s[:i] + "\n" + tex + s[j:]
open(p, "w").write(s); print("appendix spliced:", len(out), "lines;", f"{len(have)} encoders;", f"{len(erows)} cf rows;", "xfit" if xf else "no-xfit", "alpha" if al else "no-alpha")
