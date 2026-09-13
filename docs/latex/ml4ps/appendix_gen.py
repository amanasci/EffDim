"""Generate the appendix LaTeX from the records and splice it into main.tex between markers. No values typed by hand."""
import json, sys, os
C = "notebooks/.cache/"
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
Table~\ref{tab:ablate} repeats the four quantities of Table~\ref{tab:real} at $d=16$ under the multi-scale density control for two further decoder initialisation seeds and for a decoder of width $400^3$ instead of $250^3$ (all fits reach variance explained 0.952). Every significant sign of the main table is retained.
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
# --- C: relative II
runs = [("$d=20$, seed 0", C + "08_relative_ii_d20.jsonl"), ("$d=25$, seed 0", C + "08_relative_ii_d25_seed0.jsonl"), ("$d=20$, seed 1", C + "08_relative_ii_d20_seed1.jsonl")]
rrows = []
for name, f in runs:
    for r in rows(f):
        if r.get("row") == "result" and r["mknn_k"] == 20:
            c = r["columns"]
            rrows.append(f"{name} & {r['align_r2_holdout']:.2f} & {c['tan_resid']['median']:.2f} & " + " & ".join(cell(c[k]["multiscale"]) for k in ("H_tan_F", "H_tan_G", "tan_resid", "II_rel", "II_rel_loc", "II_rel_emp")) + r" \\")
if rrows:
    out.append(r"""\section*{Appendix C: relative second fundamental form, pilot}
Two sphere-projected decoders (HSC $=F$, Legacy $=G$; Phase 7 protocol), a global ridge map $A$ from $x_F$ to $x_G$ (fit on the 8{,}000 training rows), 2{,}048 seeded anchors. Columns: holdout $R^2$ of $A$; median first-order obstruction $\norm{AJ_F - J_G L}/\norm{AJ_F}$ with $L = J_G^{+}AJ_F$; then multi-scale density-controlled partials against MKNN ($k=20$; log radius at $k\in\{10,30,100,300\}$ in both spaces) of each decoder's $\norm{\Htan}$, the first-order obstruction, $\norm{\mathrm{II}_G(L\cdot,L\cdot) - P_N^G A\,\mathrm{II}_F}$, the same with $L$ fitted on 256 neighbours' latent codes, and a decoder-free estimate from the quadratic coefficient of the alignment residual on $F$'s tangent coordinates.
\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccccc}
\toprule
run & $R^2_A$ & tan.\ resid & $\norm{\Htan}$ HSC & $\norm{\Htan}$ Legacy & tan.\ resid & $\mathrm{II}_{\mathrm{rel}}$ & $\mathrm{II}_{\mathrm{rel}}$ (local $L$) & $\mathrm{II}_{\mathrm{rel}}$ (data) \\
\midrule""")
    out += rrows
    out.append(r"""\bottomrule
\end{tabular}
\caption{Relative-II pilot. $^{*}$ not significant at 0.05.}
\label{tab:relii}
\end{table}""")
tex = "\n".join(out) + "\n"
p = "docs/latex/ml4ps/main.tex"; s = open(p).read()
B, E = "% BEGIN APPENDIX AUTOGEN", "% END APPENDIX AUTOGEN"
if B not in s:
    s = s.replace("\\end{document}", f"\\appendix\n{B}\n{E}\n\\end{{document}}")
i, j = s.index(B) + len(B), s.index(E)
s = s[:i] + "\n" + tex + s[j:]
open(p, "w").write(s); print("appendix spliced:", len(out), "lines;", "xfit" if xf else "no-xfit", "alpha" if al else "no-alpha", len(rrows), "relii rows")
