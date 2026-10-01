"""Aggregate the scaling-sweep records into LaTeX tables, figures and a computed report.

Every number in the outputs is read from the records/arrays through `sweep.extract`; missing
encoders/jobs print as `--` and are left out of the report's counts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sweep.extract import (LABELS, _ptex, cf_summary, devices, read_rows, sign_test, split_cells,
                           var_explained, xfit_cells, xfit_hess_cos_p50)
from sweep.jobs import JOB_SUFFIXES
from sweep.manifest import DEFAULT_PATH, Encoder, Manifest, load_manifest

HERE = Path(__file__).resolve().parents[1]
SPLIT_JOBS = ("main_xfit", "seed1", "seed2")
ROBUST_VARIANTS = ("main_xfit", "seed1", "seed2")
MISMATCH, ALIGN = "hess_mismatch_emp", "align_cos_tan"
LAB_TEX = {lab: lab.replace("_", r"\_") for lab in LABELS}
ALPHA = 0.05
# The paper's alignment claim per label (main.tex, cross-encoder paragraph); None = no stated claim.
ALIGN_EXPECT = {"mag_r": "pos_sig", "photo_z": "pos_sig", "stellar_mass": "nonsig", "smooth_fraction": None}
EXPECT_TEXT = {"pos_sig": "positive and significant", "nonsig": "non-significant"}


def cell(v) -> str:
    """As appendix_gen.py's `cell`."""
    if v is None or v.get("partial") is None or v["partial"] != v["partial"]: return "--"
    return f"${v['partial']:+.2f}" + ("^{*}" if v["p"] > 0.05 else "") + "$"


def tex_name(name: str) -> str:
    return name.replace("_", r"\_")


def ordered(m: Manifest) -> List[Encoder]:
    fams: List[str] = []
    for e in m.encoders:
        if e.family not in fams: fams.append(e.family)
    return [e for f in fams for e in sorted((x for x in m.encoders if x.family == f), key=lambda x: x.params)]


def _ok(v) -> bool:
    return v is not None and v.get("partial") is not None and v["partial"] == v["partial"]


def _staleness(mx_rows, cf_rows, rb_rows) -> Optional[str]:
    """None when cf and robust read the geometry main_xfit saved; 'not checked' when a sha is missing."""
    fit = next((r for r in mx_rows if r.get("row") == "fit"), {})
    ref = fit.get("geometry_npz_sha256")
    cf_env = next((r for r in cf_rows if r.get("row") == "environment"), {})
    rb_env = next((r for r in rb_rows if r.get("row") == "environment"), {})
    if not ref or not cf_env.get("geometry_sha256") or not rb_env.get("geometry_sha256"):
        return "not checked" if (mx_rows or cf_rows or rb_rows) else None
    bad = [n for n, e in (("cf", cf_env), ("robust", rb_env)) if e.get("geometry_sha256") != ref]
    return (" and ".join(bad) + " geometry sha differs from main_xfit's") if bad else None


class EncData:
    def __init__(self, enc: Encoder, records_dir: Path, arrays_dir: Path, labels=LABELS, extra_split=()):
        self.enc = enc
        self.rows = {s: read_rows(records_dir / f"scaling__{enc.name}__{s}.jsonl") for s in SPLIT_JOBS + tuple(extra_split)}
        self.robust_rows = read_rows(records_dir / f"scaling__{enc.name}__robust.jsonl")
        cf_rows = read_rows(records_dir / f"scaling__{enc.name}__cf.jsonl")
        self.cf_rows = cf_rows
        self.split = {s: split_cells(r) for s, r in self.rows.items()}
        self.xfit = xfit_cells(self.rows["main_xfit"])
        self.hcos = xfit_hess_cos_p50(self.rows["main_xfit"])
        cf_p, thin_p = arrays_dir / f"scaling__{enc.name}__cf.npz", arrays_dir / f"scaling__{enc.name}__thin.npz"
        self.cf = cf_summary(cf_p, labels=labels) if cf_p.exists() else {}
        self.sign = sign_test(cf_p, thin_p, labels=labels) if cf_p.exists() and thin_p.exists() else {}
        self.has_any = any(self.rows.values()) or cf_p.exists() or thin_p.exists() \
            or (records_dir / f"scaling__{enc.name}__cf.jsonl").exists()
        self.complete = all(any(r.get("row") == "result" for r in self.rows[s]) for s in SPLIT_JOBS) \
            and any(r.get("row") == "result" for r in self.robust_rows) and cf_p.exists() and thin_p.exists()
        self.stale = _staleness(self.rows["main_xfit"], cf_rows, self.robust_rows)

    def part(self, variant: str, lab: str, col: str) -> Optional[dict]:
        return self.split[variant].get((lab, col))

    def robust(self, lab: str, col: str):
        vals = [self.part(v, lab, col) for v in ROBUST_VARIANTS]
        xs = [v["partial"] for v in vals if _ok(v)]
        if not xs or not _ok(vals[0]):
            return None
        s0 = np.sign(vals[0]["partial"])
        flips = sum(1 for v in vals[1:] if _ok(v) and np.sign(v["partial"]) != s0)
        return min(xs), max(xs), flips, len(xs)


# ------------------------------------------------------------------ tables
def _tab_main(data: List[EncData]) -> str:
    devs = sorted({d for x in data for d in devices(x.rows["main_xfit"])})
    dev = ", ".join(devs) if devs else "not recorded"
    head = " & ".join(r"\multicolumn{2}{c}{" + LAB_TEX[l] + "}" for l in LABELS)
    sub = " & ".join("mism. & align." for _ in LABELS)
    out = [r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{l" + "cc" * len(LABELS) + "c}", r"\toprule",
           f"encoder & {head} & var.\\ expl. \\\\", f" & {sub} & \\\\", r"\midrule"]
    for x in data:
        cells = [cell(x.part("main_xfit", l, c)) for l in LABELS for c in (MISMATCH, ALIGN)]
        ve = var_explained(x.rows["main_xfit"])
        out.append(tex_name(x.enc.name) + " & " + " & ".join(cells) + " & " + (f"{ve:.3f}" if ve is not None else "--") + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Encoder scaling: rank-partial Spearman correlations of the mismatch and alignment with local $R^2$ "
            r"under the multi-scale density control; $d=16$, decoder seed 0, device: " + dev + r". "
            r"$^{*}$ Not significant at 0.05; -- no record.}",
            r"\label{tab:scaling_main}", r"\end{table*}", ""]
    return "\n".join(out)


def _tab_xfit(data: List[EncData]) -> str:
    out = [r"\begin{table}[h]", r"\centering\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llccccc}",
           r"\toprule", r" & & \multicolumn{2}{c}{mismatch, cross-fit} & \multicolumn{2}{c}{alignment, cross-fit} & Hess.\ split cos \\",
           r"encoder & label & fit A/score B & fit B/score A & fit A/score B & fit B/score A & p50 \\", r"\midrule"]
    for x in data:
        for lab in LABELS:
            m = x.xfit.get((lab, "hess_mismatch_dec"), {}); g = x.xfit.get((lab, ALIGN), {})
            hc = x.hcos.get(lab)
            out.append(f"{tex_name(x.enc.name) if lab == LABELS[0] else ''} & {LAB_TEX[lab]} & {cell(m.get('fitA_scoreB'))} & "
                       f"{cell(m.get('fitB_scoreA'))} & {cell(g.get('fitA_scoreB'))} & {cell(g.get('fitB_scoreA'))} & "
                       + (f"{hc:+.2f}" if hc is not None else "--") + r" \\")
        out.append(r"\addlinespace[2pt]")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Encoder scaling: cross-fitted mismatch and alignment partials under the multi-scale density control, "
            r"with median split-half tensor cosine for the estimated label Hessian. $^{*}$ Not significant at 0.05.}",
            r"\label{tab:scaling_xfit}", r"\end{table}", ""]
    return "\n".join(out)


def _tab_cf(data: List[EncData]) -> str:
    out = [r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccccccccc}",
           r"\toprule", r" & & \multicolumn{5}{c}{model normal $S$} & \multicolumn{2}{c}{random, $q$-matched} & sign test \\",
           r"encoder & label & help & hurt & $\Delta R^2(+1)$ & $\Delta R^2(-1)$ & $t^{*}$ & help & hurt & $p_{\mathrm{help}}\le$ \\",
           r"\midrule"]
    for x in data:
        for lab in LABELS:
            c = x.cf.get(lab)
            if c is None:
                cells = ["--"] * 7
            else:
                s, r = c["S_model"], c["random_qmatched"]
                cells = [f"{s['help']:.2f}", f"{s['hurt']:.2f}", f"${s['d_r2_plus']:+.3f}$", f"${s['d_r2_minus']:+.3f}$",
                         f"{s['t_star']:.1f}", f"{r['help']:.2f}", f"{r['hurt']:.2f}"]
            st = x.sign.get(lab)
            cells.append(f"${_ptex(st['p_help'])}$" if st and st["n"] and st["p_help"] > 0 else "--")
            out.append(f"{tex_name(x.enc.name) if lab == LABELS[0] else ''} & {LAB_TEX[lab]} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace[2pt]")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Encoder scaling: counterfactual normal scaling at $d=16$. Fraction of anchors where $t=1$ beats "
            r"shape-flat (help) and where $t=-1$ is worse (hurt), median $\Delta R^2$ at $t=\pm1$, median $t^{*}$; the same "
            r"for a random in-sphere normal direction rescaled to the same centred amplitude; one-sided sign-test upper bound "
            r"on $p$ for help on a maximal anchor set with pairwise overlap at most 5\% of $k$.}",
            r"\label{tab:scaling_cf}", r"\end{table*}", ""]
    return "\n".join(out)


def _rob_cells(r) -> List[str]:
    if r is None: return ["--", "--"]
    lo, hi, flips, n = r
    cov = f" ($n={n}$)" if n < len(ROBUST_VARIANTS) else ""
    return [f"$[{lo:+.2f}, {hi:+.2f}]$" + cov, str(flips)]


def _tab_robust(data: List[EncData]) -> str:
    out = [r"\begin{table}[h]", r"\centering\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llcccc}",
           r"\toprule", r" & & \multicolumn{2}{c}{mismatch} & \multicolumn{2}{c}{alignment} \\",
           r"encoder & label & min--max & sign flips & min--max & sign flips \\", r"\midrule"]
    for x in data:
        for lab in LABELS:
            cells = _rob_cells(x.robust(lab, MISMATCH)) + _rob_cells(x.robust(lab, ALIGN))
            out.append(f"{tex_name(x.enc.name) if lab == LABELS[0] else ''} & {LAB_TEX[lab]} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace[2pt]")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Encoder scaling: range of the partial over the three decoder fits (seeds 0, 1, 2) and the number of "
            r"fits whose sign differs from seed 0; $(n=\cdot)$ marks a range over fewer than three fits.}",
            r"\label{tab:scaling_robust}", r"\end{table}", ""]
    return "\n".join(out)


# ------------------------------------------------------------------ figures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.hashsalt"] = "effdim-scaling"
    return plt


def _save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out_dir / f"{stem}.png", dpi=150, metadata={"Software": None})


def _family_colors(data: List[EncData]) -> Dict[str, tuple]:
    plt = _mpl()
    fams: List[str] = []
    for x in data:
        if x.enc.family not in fams: fams.append(x.enc.family)
    cmap = plt.get_cmap("tab20")
    return {f: cmap(i % 20) for i, f in enumerate(fams)}


def _fig_partials(data: List[EncData], out_dir: Path) -> None:
    plt = _mpl(); col = _family_colors(data)
    fig, axes = plt.subplots(2, len(LABELS), figsize=(14, 6), sharex=True)
    for ri, (qn, qc) in enumerate((("mismatch", MISMATCH), ("alignment", ALIGN))):
        for li, lab in enumerate(LABELS):
            ax = axes[ri, li]
            for x in data:
                v = x.part("main_xfit", lab, qc)
                if not _ok(v): continue
                lx = math.log10(x.enc.params)
                ax.scatter([lx], [v["partial"]], color=col[x.enc.family], s=22,
                           marker="o" if v["p"] <= ALPHA else "x", zorder=3)
                if x.enc.in_paper:
                    ax.scatter([lx], [v["partial"]], facecolors="none", edgecolors="k", s=90, linewidths=1.0, zorder=4)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_title(f"{qn}: {lab}", fontsize=9)
            if ri == 1: ax.set_xlabel("log10(params)")
            if li == 0: ax.set_ylabel("partial")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in col.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, out_dir, "fig_scaling_partials"); plt.close(fig)


def _fig_cf(data: List[EncData], out_dir: Path) -> None:
    plt = _mpl(); col = _family_colors(data)
    fig, axes = plt.subplots(1, len(LABELS), figsize=(14, 3.6), sharey=True)
    for li, lab in enumerate(LABELS):
        ax = axes[li]
        for x in data:
            c = x.cf.get(lab)
            if c is None: continue
            lx = math.log10(x.enc.params)
            ax.scatter([lx], [c["S_model"]["help"]], color=col[x.enc.family], s=22, zorder=3)
            ax.scatter([lx], [c["random_qmatched"]["help"]], facecolors="none", edgecolors=col[x.enc.family], s=22, zorder=3)
        ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
        ax.set_title(lab, fontsize=9); ax.set_xlabel("log10(params)")
        if li == 0: ax.set_ylabel("help fraction (filled: model S; hollow: random)")
    fig.tight_layout()
    _save(fig, out_dir, "fig_scaling_cf"); plt.close(fig)


def _fig_robust(data: List[EncData], out_dir: Path) -> None:
    plt = _mpl()
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(data)), 4))
    for i, x in enumerate(data):
        for li, lab in enumerate(LABELS):
            r = x.robust(lab, MISMATCH)
            if r is None: continue
            xp = i + (li - 1.5) * 0.18
            ax.vlines(xp, r[0], r[1], color=cmap(li), lw=2.5)
    for li, lab in enumerate(LABELS):
        ax.plot([], [], color=cmap(li), lw=2.5, label=lab)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(data))); ax.set_xticklabels([x.enc.name for x in data], rotation=90, fontsize=6)
    ax.set_ylabel("mismatch partial, min-max over variants"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig_scaling_robust"); plt.close(fig)


# ------------------------------------------------------------------ report
def _guard_lines(data: List[EncData]) -> List[str]:
    L = ["", "## Reproduction guard (robust job)", ""]
    for x in data:
        if not x.robust_rows: continue
        g = next((r for r in x.robust_rows if r.get("row") == "guard" and r.get("passed") is True), None)
        L.append(f"- {x.enc.name}: {g['mode']} PASS, {g['n_split']} split cells and {g['n_cf']} counterfactual values, "
                 f"max |diff| {g['max_abs_diff_split']:.2g} (split), {g['max_abs_diff_cf']:.2g} (counterfactual)"
                 if g else f"- {x.enc.name}: no guard row")
    no_robust = [x.enc.name for x in data if not x.robust_rows]
    L.append(f"- not run: {', '.join(no_robust) if no_robust else 'none'}")
    return L


def _stale_lines(data: List[EncData]) -> List[str]:
    L = ["", "## Stale encoders", ""]
    stale = [x for x in data if x.stale not in (None, "not checked")]
    unchecked = [x.enc.name for x in data if x.stale == "not checked"]
    L += [f"- {x.enc.name}: {x.stale}" for x in stale]
    if unchecked: L.append(f"- not checked: {', '.join(unchecked)}")
    if not stale and not unchecked: L.append("- none")
    return L


def _report(data: List[EncData], n_total: int, published_dir: Optional[Path] = None) -> str:
    have = [x for x in data if x.has_any]
    done = [x for x in data if x.complete]
    L = [f"{len(have)} of {n_total} encoders have records", f"{len(done)} of {n_total} encoders complete all {len(JOB_SUFFIXES)} jobs", ""]
    breaks: Dict[str, List[str]] = {}

    L += ["## (a) Mismatch partial negative and significant (main_xfit record)", ""]
    for lab in LABELS:
        xs = [(x, x.part("main_xfit", lab, MISMATCH)) for x in data]; xs = [(x, v) for x, v in xs if _ok(v)]
        good = [x for x, v in xs if v["partial"] < 0 and v["p"] <= ALPHA]
        L.append(f"- {lab}: {len(good)} of {len(xs)}")
        breaks[f"(a) mismatch negative and significant, {lab}"] = [x.enc.name for x, _ in xs if x not in good]
    L += ["", "## (b) Alignment partial sign and significance (main_xfit record)", "",
          "Paper's claim per label (main.tex): mag_r and photo_z positive and significant; stellar_mass non-significant; "
          "smooth_fraction no stated claim (\"less consistent\"). Exceptions are encoders that do not match their label's claim.", ""]
    for lab in LABELS:
        xs = [(x, x.part("main_xfit", lab, ALIGN)) for x in data]; xs = [(x, v) for x, v in xs if _ok(v)]
        neg = [x for x, v in xs if v["partial"] < 0 and v["p"] <= ALPHA]
        pos = [x for x, v in xs if v["partial"] > 0 and v["p"] <= ALPHA]
        ns = [x for x, _ in xs if x not in neg and x not in pos]
        exp = ALIGN_EXPECT[lab]
        claim = {"pos_sig": "paper claim: positive-significant", "nonsig": "paper claim: non-significant",
                 None: "no paper claim"}[exp]
        L.append(f"- {lab}: {len(neg)} negative-significant / {len(pos)} positive-significant / "
                 f"{len(ns)} non-significant (of {len(xs)}); {claim}")
        if exp is not None:
            match = pos if exp == "pos_sig" else ns
            breaks[f"(b) alignment {EXPECT_TEXT[exp]}, {lab}"] = [x.enc.name for x, _ in xs if x not in match]
    L += ["", "## (c) Counterfactual: model-normal help exceeds random help, and help > 0.5", ""]
    for lab in LABELS:
        xs = [x for x in data if lab in x.cf]
        g1 = [x for x in xs if x.cf[lab]["S_model"]["help"] > x.cf[lab]["random_qmatched"]["help"]]
        g2 = [x for x in xs if x.cf[lab]["S_model"]["help"] > 0.5]
        L.append(f"- {lab}: help > random help in {len(g1)} of {len(xs)}; help > 0.5 in {len(g2)} of {len(xs)}")
        breaks[f"(c) help > random help, {lab}"] = [x.enc.name for x in xs if x not in g1]
        breaks[f"(c) help > 0.5, {lab}"] = [x.enc.name for x in xs if x not in g2]
    L += ["", "## (c') Counterfactual: sign reversal hurts (model-normal hurt > 0.5 and > random hurt)", ""]
    for lab in LABELS:
        xs = [x for x in data if lab in x.cf]
        g = [x for x in xs if x.cf[lab]["S_model"]["hurt"] > 0.5
             and x.cf[lab]["S_model"]["hurt"] > x.cf[lab]["random_qmatched"]["hurt"]]
        L.append(f"- {lab}: hurt > 0.5 and hurt > random hurt in {len(g)} of {len(xs)}")
        breaks[f"(c') hurt > 0.5 and hurt > random hurt, {lab}"] = [x.enc.name for x in xs if x not in g]
    L += ["", f"## (d) Thinned-anchor sign test p_help < {ALPHA}", ""]
    for lab in LABELS:
        xs = [x for x in data if lab in x.sign and x.sign[lab]["n"]]
        good = [x for x in xs if x.sign[lab]["p_help"] < ALPHA]
        L.append(f"- {lab}: {len(good)} of {len(xs)}")
        breaks[f"(d) sign test p_help < {ALPHA}, {lab}"] = [x.enc.name for x in xs if x not in good]
    L += ["", "## Encoders that break a claim", ""]
    for k, v in breaks.items():
        L.append(f"- {k}: " + (", ".join(v) if v else "none"))
    L += _guard_lines(data)
    L += _stale_lines(data)
    if published_dir is not None:
        L += _section_published(data, published_dir)
    L += _section_ladder(data)
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ published comparison and ladder
PUBLISHED_SPLIT = {"vit_base": "09_physics_probe_facing_split.jsonl"}
VIT_SEEDS = ("09_physics_probe_facing_split.jsonl", "09_physics_probe_facing_split_seed1.jsonl",
             "09_physics_probe_facing_split_seed2.jsonl")
BORDER = (0.01, 0.1)


def _d16_cells(path: Path) -> Dict:
    rows = [r for r in read_rows(path) if r.get("row") != "result" or r.get("d") == 16]
    return split_cells(rows)


def seed_tolerance(published_dir: Path) -> Dict:
    seeds = [_d16_cells(Path(published_dir) / f) for f in VIT_SEEDS]
    tol = {}
    for key in seeds[0]:
        xs = [s[key]["partial"] for s in seeds if _ok(s.get(key))]
        tol[key] = max(abs(a - b) for a in xs for b in xs) if len(xs) > 1 else float("nan")
    return tol


def compare_cell(pub, gpu, tol: float) -> str:
    if not _ok(pub) or not _ok(gpu):
        return "missing"
    if any(BORDER[0] < v["p"] < BORDER[1] for v in (pub, gpu)):
        return "borderline"
    same = np.sign(pub["partial"]) == np.sign(gpu["partial"]) and (pub["p"] <= ALPHA) == (gpu["p"] <= ALPHA)
    return "agree" if same and abs(pub["partial"] - gpu["partial"]) <= tol else "disagree"


def oof_identity(x: "EncData", published_dir: Path) -> Optional[float]:
    p = Path(published_dir) / PUBLISHED_SPLIT.get(x.enc.name, f"09_physics_probe_facing_split_{x.enc.name}.jsonl")
    pub = {r["label"]: r["global_oof_r2"] for r in read_rows(p) if r.get("row") == "result" and r.get("d") == 16}
    ours = {r["label"]: r["global_oof_r2"] for r in x.rows["main_xfit"] if r.get("row") == "result"}
    diffs = [abs(ours[l] - pub[l]) for l in pub if l in ours]
    return max(diffs) if diffs else None


def _section_published(data: List["EncData"], published_dir: Path) -> List[str]:
    pub_dir = Path(published_dir); tol = seed_tolerance(pub_dir)
    L = ["", "## Published five: sweep GPU versus published CPU", "",
         "Per label and column, the sweep's main_xfit multiscale partial against the published record. A cell agrees when "
         f"the sign and significance at {ALPHA} match and |GPU - CPU| is at most the tolerance; tolerance = ViT-B's "
         "published seed spread (seeds 0, 1, 2; the only CPU seed spread that exists), used for all five encoders. "
         f"Borderline: p in ({BORDER[0]}, {BORDER[1]}) on either side, counted separately. -- = no published value.", "",
         "| encoder | label | column | published | sweep | result |", "|---|---|---|---|---|---|"]
    counts: Dict[str, int] = {}
    oof: List[str] = []
    for x in data:
        if not x.enc.in_paper:
            continue
        pub_cells = _d16_cells(pub_dir / PUBLISHED_SPLIT.get(x.enc.name, f"09_physics_probe_facing_split_{x.enc.name}.jsonl"))
        for lab in LABELS:
            for c in (MISMATCH, ALIGN):
                pv, gv = pub_cells.get((lab, c)), x.part("main_xfit", lab, c)
                res = compare_cell(pv, gv, tol.get((lab, c), float("nan")))
                counts[res] = counts.get(res, 0) + 1
                fmt = lambda v: f"{v['partial']:+.2f}" if _ok(v) else "--"
                L.append(f"| {x.enc.name} | {lab} | {c} | {fmt(pv)} | {fmt(gv)} | {res} |")
        dv = oof_identity(x, pub_dir)
        oof.append(f"OOF R2 identity: {x.enc.name} max |diff| " + ("--" if dv is None else f"{dv:.2g}"))
    L += ["", "Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), ""] + oof
    L += ["", "Counterfactual (yes/no agreement; there is no CPU seed spread for it):", "",
          "| encoder | label | help>0.5 pub/sweep | hurt>0.5 pub/sweep | sign test p<0.05 pub/sweep | agree |", "|---|---|---|---|---|---|"]
    for x in data:
        if not x.enc.in_paper:
            continue
        cf_p = pub_dir / f"09_physics_normal_scaling_{x.enc.name}_d16.npz"
        th_p = pub_dir / f"09_physics_normal_scaling_{x.enc.name}_d16_thin.npz"
        pcf = cf_summary(cf_p) if cf_p.exists() else {}
        psg = sign_test(cf_p, th_p) if cf_p.exists() and th_p.exists() else {}
        for lab in LABELS:
            def flags(cf, sg):
                if lab not in cf or lab not in sg:
                    return None
                return (cf[lab]["S_model"]["help"] > 0.5, cf[lab]["S_model"]["hurt"] > 0.5, sg[lab]["p_help"] < ALPHA)
            a, b = flags(pcf, psg), flags(x.cf, x.sign)
            yn = lambda f, i: "--" if f is None else ("y" if f[i] else "n")
            res = "missing" if a is None or b is None else ("agree" if a == b else "disagree")
            L.append(f"| {x.enc.name} | {lab} | {yn(a, 0)}/{yn(b, 0)} | {yn(a, 1)}/{yn(b, 1)} | {yn(a, 2)}/{yn(b, 2)} | {res} |")
    return L


def _section_ladder(data: List["EncData"]) -> List[str]:
    lad = sorted((x for x in data if x.enc.family == "DINOv3"), key=lambda x: x.enc.params)
    if not lad:
        return []
    L = ["", "## DINOv3 size ladder", "",
         "One family, one training recipe. Supports: sign and significance of the partials and the counterfactual pattern "
         "across sizes. Does not support 'the effect scales with size' (D changes with size at fixed d = 16; n = 6). "
         "Spearman with log params is descriptive.", "",
         "| encoder | params | label | mismatch (seeds min..max) | alignment (seeds min..max) | help/hurt | mismatch @alpha* |",
         "|---|---|---|---|---|---|---|"]
    for x in lad:
        for lab in LABELS:
            def span(c):
                r = x.robust(lab, c)
                v = x.part("main_xfit", lab, c)
                return ("--" if not _ok(v) else f"{v['partial']:+.2f}") + ("" if r is None else f" ({r[0]:+.2f}..{r[1]:+.2f})")
            cfv = x.cf.get(lab)
            hh = "--" if cfv is None else f"{cfv['S_model']['help']:.2f}/{cfv['S_model']['hurt']:.2f}"
            tuned = next((r for r in x.robust_rows if r.get("row") == "result" and r.get("label") == lab and r.get("alpha_mode") == "tuned"), None)
            tv = "--" if tuned is None else f"{tuned['partials']['published_controls'][MISMATCH]['partial']:+.2f}"
            L.append(f"| {x.enc.name} | {x.enc.params:,} | {lab} | {span(MISMATCH)} | {span(ALIGN)} | {hh} | {tv} |")
    from scipy.stats import spearmanr
    L += ["", "Spearman with log params (descriptive):"]
    for lab in LABELS:
        for c in (MISMATCH, ALIGN):
            pts = [(math.log(x.enc.params), x.part("main_xfit", lab, c)["partial"]) for x in lad if _ok(x.part("main_xfit", lab, c))]
            rho = float(spearmanr(*zip(*pts)).statistic) if len(pts) >= 3 else float("nan")
            L.append(f"- {lab} {c}: " + ("--" if rho != rho else f"{rho:+.2f}") + f" (n={len(pts)})")
    return L


# ------------------------------------------------------------------ molecules (QM9)
QM9_REPORT = "QM9_REPORT.md"
MOL_EXTRA_SPLIT = ("main_d16",)
ESTIMATORS = ("mle", "two_nn", "tle", "mind_mlk")
MOL_LIMITS = (
    "- Only ChemFM 1B -> 3B is a model-size pair; the five ChemBERTa-2 models share one architecture "
    "(5M/10M/77M are pretraining-set sizes).",
    "- The MTR models were pretrained on RDKit descriptors including molar refractivity (close to alpha), so alpha is "
    "expected near-linear for them.",
    "- Neighbourhoods (k = 2,048 of 130,744 molecules) cover about 1/64 of the data (galaxies: 1/42).",
    "- Special tokens are inside the mean pool.",
    "- d = 20 is the cap and sits at an open question from the d = 20 spike findings; the d = 16 baseline covers it.",
)


def read_timing(path) -> Optional[dict]:
    """A queue timing file {"exit", "max_rss_kb", "wall_s"}; None when it is empty or truncated."""
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return None


def d16_variant(d_run: int) -> str:
    """Claim (a) at d = 16 reads main_d16, or main_xfit where d_run is 16."""
    return "main_xfit" if d_run == 16 else "main_d16"


def _md(v) -> str:
    if not _ok(v): return "--"
    return f"{v['partial']:+.2f}" + ("" if v["p"] <= ALPHA else " (ns)")


def _env_rows(x: EncData) -> List[dict]:
    rows = [r for rs in x.rows.values() for r in rs] + list(x.cf_rows) + list(x.robust_rows)
    return [r for r in rows if r.get("row") == "environment"]


def _report_molecules(data: List[EncData], labels, d_info: Dict, d_sha: str, synth: Optional[dict], timing: Dict[str, dict]) -> str:
    n = len(data)
    drun = {x.enc.name: (d_info.get(x.enc.name) or {}).get("d_run") for x in data}
    have = [x for x in data if x.has_any]
    done = [x for x in data if drun[x.enc.name] is not None and x.complete
            and (drun[x.enc.name] == 16 or any(r.get("row") == "result" for r in x.rows["main_d16"]))]
    L = [f"{len(have)} of {n} encoders have records",
         f"{len(done)} of {n} encoders complete their battery (main_xfit, seed1, seed2, cf, thin, robust; main_d16 where d_run != 16)", ""]
    breaks: Dict[str, List[str]] = {}

    L += ["## d per encoder", "",
          "d_ID = median of mle, two_nn, tle, mind_mlk, rounded half to even (Python round), on a 10,000-row subsample of "
          "the row-normalised embeddings (one shared k-NN, k = 10); d_run = min(d_ID, 20).", "",
          "| encoder | D | params | d_ID | d_run | " + " | ".join(ESTIMATORS) + " |", "|---|---|---|---|---|" + "---|" * len(ESTIMATORS)]
    for x in data:
        e = d_info.get(x.enc.name)
        est = " | ".join("--" for _ in ESTIMATORS) if e is None else " | ".join(f"{e['estimates'][k]:.2f}" for k in ESTIMATORS)
        dd = "-- | --" if e is None else f"{e['d_ID']} | {e['d_run']}"
        L.append(f"| {x.enc.name} | {x.enc.dim} | {x.enc.params:,} | {dd} | {est} |")

    L += ["", "## Synthetic intrinsic-dimension check (unit spheres)", ""]
    if synth is None:
        L.append("- not run")
    else:
        L += [f"Estimate minus true dimension; n = {synth['n']:,} points on a unit sphere of true dimension d, rotated into R^D.", "",
              "| true d | D | " + " | ".join(ESTIMATORS) + " |", "|---|---|" + "---|" * len(ESTIMATORS)]
        for r in synth["rows"]:
            L.append(f"| {r['true_d']} | {r['D']} | " + " | ".join(f"{r['estimates'][k] - r['true_d']:+.2f}" for k in ESTIMATORS) + " |")
        top = max(r["true_d"] for r in synth["rows"])
        low = [k for k in ESTIMATORS if all(r["estimates"][k] < r["true_d"] for r in synth["rows"] if r["true_d"] == top)]
        L += ["", f"- read low at true d = {top} in every D: {', '.join(low) if low else 'none'} (of {len(ESTIMATORS)})"]

    L += ["", "## (a) Mismatch partial negative and significant"]
    for title, key, variant in (("### at d_run (main_xfit)", "(a) mismatch negative and significant at d_run", lambda x: "main_xfit"),
                                ("### at d = 16 (main_d16, or main_xfit where d_run = 16)", "(a) mismatch negative and significant at d = 16",
                                 lambda x: None if drun[x.enc.name] is None else d16_variant(drun[x.enc.name]))):
        L += ["", title, ""]
        for lab in labels:
            xs = [(x, x.part(variant(x), lab, MISMATCH)) for x in data if variant(x) is not None]
            xs = [(x, v) for x, v in xs if _ok(v)]
            good = [x for x, v in xs if v["partial"] < 0 and v["p"] <= ALPHA]
            L.append(f"- {lab}: {len(good)} of {len(xs)}")
            breaks[f"{key}, {lab}"] = [x.enc.name for x, _ in xs if x not in good]

    L += ["", "## (c) Counterfactual: model-normal help exceeds random help, and help > 0.5 (at d_run)", ""]
    for lab in labels:
        xs = [x for x in data if lab in x.cf]
        g1 = [x for x in xs if x.cf[lab]["S_model"]["help"] > x.cf[lab]["random_qmatched"]["help"]]
        g2 = [x for x in xs if x.cf[lab]["S_model"]["help"] > 0.5]
        L.append(f"- {lab}: help > random help in {len(g1)} of {len(xs)}; help > 0.5 in {len(g2)} of {len(xs)}")
        breaks[f"(c) help > random help, {lab}"] = [x.enc.name for x in xs if x not in g1]
        breaks[f"(c) help > 0.5, {lab}"] = [x.enc.name for x in xs if x not in g2]
    L += ["", "## (c') Counterfactual: sign reversal hurts (model-normal hurt > 0.5 and > random hurt, at d_run)", ""]
    for lab in labels:
        xs = [x for x in data if lab in x.cf]
        g = [x for x in xs if x.cf[lab]["S_model"]["hurt"] > 0.5 and x.cf[lab]["S_model"]["hurt"] > x.cf[lab]["random_qmatched"]["hurt"]]
        L.append(f"- {lab}: hurt > 0.5 and hurt > random hurt in {len(g)} of {len(xs)}")
        breaks[f"(c') hurt > 0.5 and hurt > random hurt, {lab}"] = [x.enc.name for x in xs if x not in g]
    L += ["", f"## (d) Thinned-anchor sign test p_help < {ALPHA} (at d_run)", ""]
    for lab in labels:
        xs = [x for x in data if lab in x.sign and x.sign[lab]["n"]]
        good = [x for x in xs if x.sign[lab]["p_help"] < ALPHA]
        L.append(f"- {lab}: {len(good)} of {len(xs)}")
        breaks[f"(d) sign test p_help < {ALPHA}, {lab}"] = [x.enc.name for x in xs if x not in good]

    L += ["", "## Alignment partial (descriptive; no molecular prior for its sign), at d_run", ""]
    for lab in labels:
        xs = [(x, x.part("main_xfit", lab, ALIGN)) for x in data]; xs = [(x, v) for x, v in xs if _ok(v)]
        neg = [x for x, v in xs if v["partial"] < 0 and v["p"] <= ALPHA]
        pos = [x for x, v in xs if v["partial"] > 0 and v["p"] <= ALPHA]
        L.append(f"- {lab}: {len(neg)} negative-significant / {len(pos)} positive-significant / "
                 f"{len(xs) - len(neg) - len(pos)} non-significant (of {len(xs)})")

    L += ["", "## Per encoder and label", "",
          "| encoder | d_run | label | mismatch @d_run | alignment @d_run | mismatch @16 | help | random help | hurt | p_help (thinned) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in data:
        dr = drun[x.enc.name]
        for lab in labels:
            c, st = x.cf.get(lab), x.sign.get(lab)
            cells = [_md(x.part("main_xfit", lab, MISMATCH)), _md(x.part("main_xfit", lab, ALIGN)),
                     "--" if dr is None else _md(x.part(d16_variant(dr), lab, MISMATCH))]
            cells += ["--"] * 3 if c is None else [f"{c['S_model']['help']:.2f}", f"{c['random_qmatched']['help']:.2f}", f"{c['S_model']['hurt']:.2f}"]
            cells.append(f"{st['p_help']:.2g}" if st and st["n"] else "--")
            L.append(f"| {x.enc.name} | {'--' if dr is None else dr} | {lab} | " + " | ".join(cells) + " |")

    L += ["", "## Encoders that break a claim", ""]
    for k, v in breaks.items():
        L.append(f"- {k}: " + (", ".join(v) if v else "none"))
    L += _guard_lines(data)
    L += _stale_lines(data)
    L += ["", "## d file in the environment rows", "",
          f"molecules_d.json sha256 {d_sha}; every split, cf and robust record should carry it.", ""]
    for x in data:
        envs = _env_rows(x)
        if envs:
            k = sum(1 for r in envs if r.get("d_file_sha256") == d_sha)
            L.append(f"- {x.enc.name}: {k} of {len(envs)} records carry it")
    L += ["", "## Wall time and peak RSS per job", "", "Exit 124: killed by the --timeout-h limit.", "",
          "| job | exit | wall (h) | peak RSS (GB) |", "|---|---|---|---|"]
    for jid in sorted(timing):
        t = timing[jid]
        L.append(f"| {jid} | {t['exit']} | {t['wall_s'] / 3600:.2f} | {t['max_rss_kb'] / 1024 ** 2:.1f} |")
    if not timing:
        L.append("| -- | -- | -- | -- |")
    L += ["", "## Stated limits", ""] + list(MOL_LIMITS)
    return "\n".join(L) + "\n"


def aggregate_molecules(manifest: Manifest, records_dir: Path, arrays_dir: Path, out_dir: Path, d_file: Path,
                        timing_dir: Optional[Path] = None, id_synthetic: Optional[Path] = None) -> None:
    records_dir, arrays_dir, out_dir = Path(records_dir), Path(arrays_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = tuple(manifest.labels)
    raw = Path(d_file).read_bytes()
    d_info, d_sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
    data = [EncData(e, records_dir, arrays_dir, labels=labels, extra_split=MOL_EXTRA_SPLIT) for e in ordered(manifest)]
    timing: Dict[str, dict] = {}
    if timing_dir is not None and Path(timing_dir).exists():
        for p in sorted(Path(timing_dir).glob("*.json")):
            t = read_timing(p)
            if t is not None:
                timing[p.stem] = t
    synth = json.loads(Path(id_synthetic).read_text()) if id_synthetic is not None and Path(id_synthetic).exists() else None
    (out_dir / QM9_REPORT).write_text(_report_molecules(data, labels, d_info, d_sha, synth, timing))


def aggregate(manifest: Manifest, records_dir: Path, arrays_dir: Path, out_dir: Path,
              encoders: Optional[List[str]] = None, published_dir: Optional[Path] = None) -> None:
    records_dir, arrays_dir, out_dir = Path(records_dir), Path(arrays_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encs = [e for e in ordered(manifest) if encoders is None or e.name in set(encoders)]
    data = [EncData(e, records_dir, arrays_dir) for e in encs]
    (out_dir / "tab_scaling_main.tex").write_text(_tab_main(data))
    (out_dir / "tab_scaling_xfit.tex").write_text(_tab_xfit(data))
    (out_dir / "tab_scaling_cf.tex").write_text(_tab_cf(data))
    (out_dir / "tab_scaling_robust.tex").write_text(_tab_robust(data))
    _fig_partials(data, out_dir); _fig_cf(data, out_dir); _fig_robust(data, out_dir)
    (out_dir / "SCALING_REPORT.md").write_text(_report(data, len(encs), published_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--arrays", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--encoders", type=lambda s: [t for t in s.split(",") if t], default=None)
    ap.add_argument("--published-dir", type=Path, default=None)
    ap.add_argument("--timing-dir", type=Path, default=None)
    ap.add_argument("--d-file", type=Path, default=HERE / "data" / "qm9" / "molecules_d.json")
    ap.add_argument("--id-synthetic", type=Path, default=HERE / "data" / "qm9" / "id_synthetic.json")
    a = ap.parse_args()
    m = load_manifest(a.manifest)
    sub = "scaling" if m.labels is None else "qm9"
    records = a.records or HERE / ".cache" / sub / "records"
    arrays = a.arrays or HERE / ".cache" / sub / "arrays"
    out = a.out or HERE / "results" / sub
    if m.labels is None:
        aggregate(m, records, arrays, out, encoders=a.encoders, published_dir=a.published_dir)
    else:
        aggregate_molecules(m, records, arrays, out, a.d_file, timing_dir=a.timing_dir or HERE / ".cache" / "qm9" / "timing",
                            id_synthetic=a.id_synthetic)


if __name__ == "__main__":
    main()
