"""Aggregate the scaling-sweep records into LaTeX tables, figures and a computed report.

Every number in the outputs is read from the records/arrays through `sweep.extract`; missing
encoders/jobs print as `--` and are left out of the report's counts."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sweep.extract import (LABELS, _ptex, cf_summary, devices, read_rows, sign_test, split_cells,
                           var_explained, xfit_cells, xfit_hess_cos_p50)
from sweep.jobs import JOB_SUFFIXES
from sweep.manifest import Encoder, Manifest, load_manifest

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
    def __init__(self, enc: Encoder, records_dir: Path, arrays_dir: Path):
        self.enc = enc
        self.rows = {s: read_rows(records_dir / f"scaling__{enc.name}__{s}.jsonl") for s in SPLIT_JOBS}
        self.robust_rows = read_rows(records_dir / f"scaling__{enc.name}__robust.jsonl")
        cf_rows = read_rows(records_dir / f"scaling__{enc.name}__cf.jsonl")
        self.split = {s: split_cells(r) for s, r in self.rows.items()}
        self.xfit = xfit_cells(self.rows["main_xfit"])
        self.hcos = xfit_hess_cos_p50(self.rows["main_xfit"])
        cf_p, thin_p = arrays_dir / f"scaling__{enc.name}__cf.npz", arrays_dir / f"scaling__{enc.name}__thin.npz"
        self.cf = cf_summary(cf_p) if cf_p.exists() else {}
        self.sign = sign_test(cf_p, thin_p) if cf_p.exists() and thin_p.exists() else {}
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
def _report(data: List[EncData], n_total: int) -> str:
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
    L += ["", "## Stale encoders", ""]
    stale = [x for x in data if x.stale not in (None, "not checked")]
    unchecked = [x.enc.name for x in data if x.stale == "not checked"]
    L += [f"- {x.enc.name}: {x.stale}" for x in stale]
    if unchecked: L.append(f"- not checked: {', '.join(unchecked)}")
    if not stale and not unchecked: L.append("- none")
    return "\n".join(L) + "\n"


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
    (out_dir / "SCALING_REPORT.md").write_text(_report(data, len(encs)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=HERE / ".cache" / "scaling" / "records")
    ap.add_argument("--arrays", type=Path, default=HERE / ".cache" / "scaling" / "arrays")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "scaling")
    ap.add_argument("--encoders", type=lambda s: [t for t in s.split(",") if t], default=None)
    ap.add_argument("--published-dir", type=Path, default=None)
    a = ap.parse_args()
    aggregate(load_manifest(), a.records, a.arrays, a.out, encoders=a.encoders, published_dir=a.published_dir)


if __name__ == "__main__":
    main()
