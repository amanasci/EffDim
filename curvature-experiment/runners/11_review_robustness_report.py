"""Review robustness report: records -> REPORT.md, one rebuttal-ready table per concern."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "review-robustness"
MISMATCH, ALIGN = "hess_mismatch_emp", "align_cos_tan"
ENC_ORDER = ["vit_base", "dinov3_vitb16", "clip_base", "convnext_base", "vit_large"]
LABELS = ["mag_r", "photo_z", "smooth_fraction", "stellar_mass"]


def load(paths: List[Path]) -> Dict[str, Any]:
    env, guard, rows = [], [], {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r["row"] == "environment":
                env.append(r)
            elif r["row"] == "guard":
                guard.append(r)
            elif r["row"] == "result":
                rows[(r["encoder"], r["label"], r["alpha_mode"])] = r
    return {"env": env, "guard": guard, "rows": list(rows.values())}


def _f(v, fmt="+.3f"):
    return "--" if v is None else format(v, fmt)


def _p(v):
    return "--" if v is None else (f"{v:.3f}" if v >= 0.001 else "<0.001")


def _get(rows, enc, lab, mode):
    return next((r for r in rows if r["encoder"] == enc and r["label"] == lab and r["alpha_mode"] == mode), None)


def _encs(rows):
    have = {r["encoder"] for r in rows}
    return [e for e in ENC_ORDER if e in have] + sorted(have - set(ENC_ORDER))


def write_report(data: Dict[str, Any], out_dir: Path) -> None:
    rows = data["rows"]; out_dir.mkdir(parents=True, exist_ok=True)
    L = ["# Review robustness (concerns 2-5)", "",
         "All numbers recomputed from the published decoder geometry (sha256-verified). Mismatch = `hess_mismatch_emp`, "
         "alignment = `align_cos_tan`, partial Spearman with the published multi-scale controls unless stated.", ""]
    for g in data["guard"]:
        L.append(f"- {g['encoder']}: guard: {g['mode']} {'PASS' if g['passed'] else 'FAIL'}")
    L += ["", "## Concern 2: validation-tuned probe", "",
          "| encoder | label | alpha* | OOF R2 @100 | OOF R2 @alpha* | mismatch @100 (p) | mismatch @alpha* (p) | align @100 (p) | align @alpha* (p) | help/hurt @100 | help/hurt @alpha* |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in _encs(rows):
        for lab in LABELS:
            a, b = _get(rows, e, lab, "published"), _get(rows, e, lab, "tuned")
            if not (a and b):
                continue
            pa, pb = a["partials"]["published_controls"], b["partials"]["published_controls"]
            L.append(f"| {e} | {lab} | {b['alpha']:.3g} | {a['global_oof_r2']:.3f} | {b['global_oof_r2']:.3f} | "
                     f"{_f(pa[MISMATCH]['partial'])} ({_p(pa[MISMATCH]['p'])}) | {_f(pb[MISMATCH]['partial'])} ({_p(pb[MISMATCH]['p'])}) | "
                     f"{_f(pa[ALIGN]['partial'])} ({_p(pa[ALIGN]['p'])}) | {_f(pb[ALIGN]['partial'])} ({_p(pb[ALIGN]['p'])}) | "
                     f"{a['cf']['S_model']['help']:.2f}/{a['cf']['S_model']['hurt']:.2f} | {b['cf']['S_model']['help']:.2f}/{b['cf']['S_model']['hurt']:.2f} |")
    for mode, title in (("published", "alpha = 100"), ("tuned", "alpha*")):
        L += ["", f"## Concern 3: added value beyond target difficulty ({title})", "",
              "Extended controls = published controls + label-Hessian norm + local label roughness (1 - local linear R2). "
              "Held-out: out-of-sample Delta R2 of local R2 from adding mismatch and alignment to the extended controls, "
              "fit on half the overlap blocks, scored on the rest (20 splits; ranks).", "",
              "| encoder | label | mismatch published ctl | mismatch extended ctl (p) | align extended ctl (p) | held-out Delta R2 median [p05, p95] | splits > 0 |",
              "|---|---|---|---|---|---|---|"]
        for e in _encs(rows):
            for lab in LABELS:
                r = _get(rows, e, lab, mode)
                if not r:
                    continue
                pe = r["partials"]["extended_controls"]; h = r["heldout"]
                L.append(f"| {e} | {lab} | {_f(r['partials']['published_controls'][MISMATCH]['partial'])} | "
                         f"{_f(pe[MISMATCH]['partial'])} ({_p(pe[MISMATCH]['p'])}) | {_f(pe[ALIGN]['partial'])} ({_p(pe[ALIGN]['p'])}) | "
                         f"{_f(h['median'])} [{_f(h['p05'])}, {_f(h['p95'])}] | {h['frac_pos']:.2f} |")
    L += ["", "## Concern 4: dependence across anchors (alpha = 100)", "",
          "Cluster bootstrap: 32 overlap blocks (average linkage on 1 - neighbourhood overlap), 2000 resamples of whole blocks, "
          "95% percentile interval; excludes-0 at 16 and 64 blocks as sensitivity. Adjacent blocks still share boundary points, "
          "so these intervals are more honest than anchor-level permutation, not exact. Thinned: anchors with pairwise overlap "
          "<= 0.10 (low power).", "",
          "| encoder | label | column | partial | 95% CI (32 blocks) | excl. 0 at 16/32/64 | thinned partial (p, n) |",
          "|---|---|---|---|---|---|---|"]
    for e in _encs(rows):
        for lab in LABELS:
            r = _get(rows, e, lab, "published")
            if not r:
                continue
            for c in (MISMATCH, ALIGN):
                b = {G: r["bootstrap"][G][c] for G in ("16", "32", "64")}; t = r["thinned"][c]
                L.append(f"| {e} | {lab} | {c} | {_f(r['partials']['published_controls'][c]['partial'])} | "
                         f"[{_f(b['32']['lo'])}, {_f(b['32']['hi'])}] | {'/'.join('y' if b[G]['excludes_zero'] else 'n' for G in ('16', '32', '64'))} | "
                         f"{_f(t['partial'])} ({_p(t['p'])}, {t['n_kept']}) |")
    L += ["", "## Concern 5: surrogate fidelity (alpha = 100)", "",
          "The Section 4 figure plots the counterfactual variant `S_model`: the data readout's tangent-plus-radial part is "
          "kept exactly on the neighbours, and the scaled term is the decoder's second-order in-sphere term "
          "(1/2)<w_S, II^S>(u,u). Variant `S` scales the data-side in-sphere readout w_S.x instead (exact, ambient). "
          "Below: agreement of their t = 1 change in local R2 across anchors.", "",
          "| encoder | label | Spearman(S_model, S) | median abs diff | n |", "|---|---|---|---|---|"]
    for e in _encs(rows):
        for lab in LABELS:
            r = _get(rows, e, lab, "published")
            if r:
                s = r["surrogate"]
                L.append(f"| {e} | {lab} | {_f(s['spearman'])} | {s['median_abs_diff']:.4f} | {s['n']} |")
    L += ["", "## Limitation", "",
          "On a known manifold (results/tensor-fidelity) the label-Hessian estimate has the right direction but its magnitude "
          "is about 2x too large at paper scale (d=16, n=86,471). The partials above are rank-based and unaffected; any claim "
          "about mismatch magnitudes would be.", ""]
    (out_dir / "REPORT.md").write_text("\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record-path", nargs="+", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = p.parse_args()
    data = load([Path(x) for x in a.record_path])
    write_report(data, Path(a.out_dir))
    print(f"{len(data['rows'])} result rows -> {a.out_dir}")


if __name__ == "__main__":
    main()
