"""Tensor fidelity report: records -> REPORT.md + fig_tensor_fidelity.png.

Usage:
    python curvature-experiment/runners/10_tensor_fidelity_report.py \\
        --record-path curvature-experiment/results/tensor-fidelity/records/*.jsonl
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = DIAGNOSTICS_ROOT.parent / "results" / "tensor-fidelity"
_spec = importlib.util.spec_from_file_location("tensor_fidelity_run", DIAGNOSTICS_ROOT / "10_tensor_fidelity_run.py")
tf = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
try:
    _spec.loader.exec_module(tf)
finally:
    sys.argv = _argv

KEY = ("mode", "n", "noise_frac", "seed", "label")
LABELS = ["lin", "nonlin"] + [tf.label_name(lam) for lam in tf.LAMBDAS]


def load_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("row") == "result":
                by_key[tuple(r[k] for k in KEY)] = r
    return [by_key[k] for k in sorted(by_key, key=lambda k: tuple(str(v) for v in k))]


def _med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.median(v)) if v else float("nan")


def pass_lines(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean = [r for r in rows if r["mode"] == "small" and r["noise_frac"] == 0.0]
    n = max(r["n"] for r in clean)
    per: Dict[str, Any] = {}
    for lab in LABELS:
        rs = [r for r in clean if r["n"] == n and r["label"] == lab]
        if not rs:
            continue
        cos = _med(rs, "cos_pf_full_p50")
        rho = None if lab == tf.label_name(0.0) else _med(rs, "rho_mismatch")
        ok = cos >= tf.TENSOR_COS_PASS and (rho is None or rho >= tf.MISMATCH_RHO_PASS)
        per[lab] = {"cos": cos, "rho": rho, "pass": bool(ok)}
    return {"n": n, "per_label": per, "pass": bool(per) and all(v["pass"] for v in per.values())}


def _f(v) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:+.3f}"


def _table(rows: List[Dict[str, Any]], mode: str) -> List[str]:
    L = ["| n | noise | label | cos pf_full | cos pf_tan | cos hess_y | cos mismatch | rho mismatch | rho align | var. expl. |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rs = [r for r in rows if r["mode"] == mode]
    for n in sorted({r["n"] for r in rs}):
        for nz in sorted({r["noise_frac"] for r in rs}):
            for lab in LABELS:
                g = [r for r in rs if r["n"] == n and r["noise_frac"] == nz and r["label"] == lab]
                if g:
                    L.append(f"| {n} | {nz:g} | {lab} | " + " | ".join(_f(_med(g, f"cos_{t}_p50")) for t in tf.TENSORS)
                             + f" | {_f(_med(g, 'rho_mismatch'))} | {_f(_med(g, 'rho_align'))} | {_med(g, 'var_explained'):.3f} |")
    return L


def write_report(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = pass_lines(rows)
    L = ["# Tensor fidelity", "",
         f"Pass lines (fixed in source before any run): noise-free small fixture at n={pl['n']}, per label the median over "
         f"seeds of the median pf_full cosine >= {tf.TENSOR_COS_PASS} and mismatch-norm Spearman >= {tf.MISMATCH_RHO_PASS} "
         f"(Spearman not tested for {tf.label_name(0.0)}, whose true mismatch is ~0 by construction).", "",
         f"**Overall: {'PASS' if pl['pass'] else 'FAIL'}**", "", "| label | cos pf_full | rho mismatch | result |", "|---|---|---|---|"]
    for lab, v in pl["per_label"].items():
        L.append(f"| {lab} | {_f(v['cos'])} | {_f(v['rho'])} | {'PASS' if v['pass'] else 'FAIL'} |")
    L += ["", "## Small fixture (d=4, D=64), medians over seeds", ""] + _table(rows, "small")
    L += ["", "## Paper scale (d=16, D=768, n=86,471)", ""] + _table(rows, "full") + [""]
    (out_dir / "REPORT.md").write_text("\n".join(L))

    small = [r for r in rows if r["mode"] == "small"]
    ns = sorted({r["n"] for r in small})
    fig, axes = plt.subplots(1, max(len(ns), 1), figsize=(3.2 * max(len(ns), 1), 2.8), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], ns):
        nzs = sorted({r["noise_frac"] for r in small if r["n"] == n})
        for t in tf.TENSORS:
            ax.plot(nzs, [_med([r for r in small if r["n"] == n and r["noise_frac"] == nz], f"cos_{t}_p50") for nz in nzs],
                    marker="o", label=t)
        ax.axhline(tf.TENSOR_COS_PASS, color="0.6", lw=0.8, ls="--")
        ax.set_title(f"n={n}", fontsize=9); ax.set_xlabel("noise / patch radius")
    axes[0][0].set_ylabel("median tensor cosine"); axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tensor_fidelity.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record-path", nargs="+", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    rows = load_rows([Path(x) for x in args.record_path])
    write_report(rows, Path(args.out_dir))
    print(f"{len(rows)} result rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
