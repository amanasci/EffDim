"""Lightweight CLI. --help must not import torch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import decision_label, file_meta, is_complete, load_json, source_of_truth_rank
from .definitions import FIELDS

KNOWN_TREES = (
    "physics_curvature_probe_submission_validation",
    "physics_local_probe_adaptation",
    "physics_quadratic_label_chart_alignment",
    "physics_quadratic_label_chart_alignment_audit",
    "physics_cross_model_curvature_local_adaptation",
    "physics_cross_model_full_curvature_reconciliation",
    "pointwise_decoder_curvature_reproduction",
    "known_curvature_point_patch_fixture_audit",
    "known_curvature_dual_estimator_robustness",
    "known_curvature_estimator_operating_characteristics",
    "physics_pointwise_residual_curvature_probe_relation",
    "physics_q_geometry_resampling_stability",
    "physics_task_aligned_curvature",
    "physics_cross_model_task_aligned_curvature",
    "physics_cross_model_pointwise_residual_curvature",
    "physics_cross_model_hessian_mismatch",
    "curvature_program_synthesis",
    "physics_curvature_component_predictive_decomposition",
)


def _geom_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = __import__("os").environ.get("PLATONIC_OUTPUTS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "outputs" / "geometry").is_dir():
            return cand / "outputs" / "geometry"
    return Path("outputs/geometry")


def cmd_verify(args: argparse.Namespace) -> int:
    root = _geom_root(args.outputs)
    rows = []
    for name in KNOWN_TREES:
        p = root / name
        rec = {
            "name": name,
            "exists": p.exists(),
            "complete": is_complete(p) if p.exists() else False,
            "label": decision_label(p) if p.exists() else None,
            "complete_marker": file_meta(p / "COMPLETE.json") if p.exists() else {"exists": False},
        }
        rows.append(rec)
        print(f"{name:56} exists={rec['exists']} complete={rec['complete']} label={rec['label']}")
    print("source_of_truth:", " > ".join(source_of_truth_rank()))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def cmd_headlines(args: argparse.Namespace) -> int:
    root = _geom_root(args.outputs)
    headlines = []
    mapping = {
        "physics_task_aligned_curvature": ("P2.observed",),
        "physics_cross_model_task_aligned_curvature": ("R1.observed",),
        "physics_cross_model_hessian_mismatch": ("P1.observed", "P2.observed"),
        "physics_pointwise_residual_curvature_probe_relation": ("P1.observed",),
        "physics_q_geometry_resampling_stability": ("summary_label",),
        "physics_curvature_probe_submission_validation": ("label",),
        "physics_local_probe_adaptation": ("label", "primary_rho"),
        "physics_quadratic_label_chart_alignment": ("label", "checks.median_delta_Q"),
        "physics_cross_model_curvature_local_adaptation": ("label", "models_with_both"),
        "physics_cross_model_full_curvature_reconciliation": ("label",),
        "known_curvature_instrument_failure_localization": ("label",),
    }
    for name, keys in mapping.items():
        dpath = root / name / "decision.json"
        if not dpath.exists():
            print(f"{name}: MISSING decision.json")
            continue
        d = load_json(dpath)
        print(f"{name}: label={d.get('label') or d.get('summary_label')}")
        for k in keys:
            cur = d
            ok = True
            for part in k.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok:
                print(f"  {k} = {cur}")
                headlines.append({"tree": name, "field": k, "value": cur})
    if args.json:
        Path(args.json).write_text(json.dumps(headlines, indent=2) + "\n")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    key = args.field
    if key not in FIELDS:
        print("unknown field; choose from:", ", ".join(FIELDS))
        return 2
    rec = FIELDS[key]
    print(json.dumps(rec, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from .metric import energy_g, metric_from_J, projectors
    from .probe_aligned import complete_normal_w, probe_aligned_second_fundamental_form, sphere_component, sphere_term_norm_g
    from .quadratic import phi2_frob

    rng = __import__("numpy").random.default_rng(0)
    J = rng.normal(size=(12, 4))
    x = rng.normal(size=12)
    x = x / __import__("numpy").linalg.norm(x)
    proj = projectors(x, J)
    assert __import__("numpy").allclose(proj["P_T"], proj["P_T"].T)
    assert __import__("numpy").allclose(proj["P_T"] @ proj["P_T"], proj["P_T"], atol=1e-8)
    g, ginv = proj["g"], proj["ginv"]
    w = rng.normal(size=12)
    wN = complete_normal_w(w, proj)
    Hess = rng.normal(size=(12, 4, 4))
    Hess = 0.5 * (Hess + __import__("numpy").transpose(Hess, (0, 2, 1)))
    II = __import__("numpy").einsum("ij,jab->iab", proj["P_N"], Hess)
    b = probe_aligned_second_fundamental_form(wN, II)
    assert b.shape == (4, 4)
    bR = sphere_component(w, proj["xhat"], g)
    chk = sphere_term_norm_g(bR, ginv, w, proj["xhat"], 4)
    assert chk["ok"], chk
    U = rng.normal(size=(20, 4))
    assert phi2_frob(U).shape[1] == 10
    _ = energy_g(b, ginv)
    print("curvature-handoff-smoke: ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="curvature-handoff", description="Curvature programme verification (no fitting).")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify-curvature-artifacts", help="Validate completion markers; no fitting.")
    v.add_argument("--outputs", default=None, help="outputs/geometry root or $PLATONIC_OUTPUTS")
    v.add_argument("--json", default=None)
    v.set_defaults(func=cmd_verify)
    h = sub.add_parser("reproduce-curvature-headlines", help="Re-read frozen decision tables; no refits.")
    h.add_argument("--outputs", default=None)
    h.add_argument("--json", default=None)
    h.set_defaults(func=cmd_headlines)
    i = sub.add_parser("inspect-curvature-definition", help="Print formula and source trees for a named field.")
    i.add_argument("field", help="KHcross|Kdircross|D_full|D_residual|E_Q_task_aligned|M_delta")
    i.set_defaults(func=cmd_inspect)
    s = sub.add_parser("curvature-handoff-smoke", help="Analytic identities only; <5 minutes.")
    s.set_defaults(func=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
